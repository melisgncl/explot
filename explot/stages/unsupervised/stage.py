from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from explot.stages.base import BaseStage, StageMeta, StageResult


class UnsupervisedStage(BaseStage):
    meta = StageMeta(
        name="unsupervised",
        depends_on=("dimensionality",),
        optional_deps=("exploration",),
    )

    def run(self, state, config, hooks) -> StageResult:
        dim_result = state.results["dimensionality"]
        autoencoder = state.results.get("autoencoder")
        transformed_df = dim_result.outputs.get("transformed_df")
        representation_df = transformed_df
        representation_name = "dimensionality"
        if autoencoder and autoencoder.success and not autoencoder.outputs.get("latent_df", pd.DataFrame()).empty:
            representation_df = autoencoder.outputs.get("latent_df")
            representation_name = "dvae"

        if representation_df is None or representation_df.empty or len(representation_df) < 10:
            return self._empty_result("Too few rows for unsupervised analysis.")

        X = representation_df.values
        n_rows = X.shape[0]
        is_fast = getattr(config, "budget", None) and getattr(config.budget, "mode", "") == "fast"
        analysis_idx = self._analysis_indices(n_rows, is_fast)
        X_analysis = X[analysis_idx]
        if len(X_analysis) < n_rows:
            if hasattr(hooks, "log"):
                hooks.log(
                    self.meta.name,
                    f"Sampling {len(X_analysis)} of {n_rows} rows for KMeans and DBSCAN analysis.",
                )

        hooks.progress(self.meta.name, 10, "Running KMeans sweep.")

        # --- KMeans ---
        max_k = 5 if is_fast else min(10, max(2, len(X_analysis) // 10))
        max_k = max(max_k, 2)
        kmeans_results = self._kmeans_sweep(X_analysis, max_k)

        hooks.progress(self.meta.name, 40, "Running DBSCAN.")

        # --- DBSCAN ---
        dbscan_results = self._dbscan_auto(X_analysis)

        hooks.progress(self.meta.name, 65, "Running Isolation Forest.")

        # --- Isolation Forest ---
        iso_scores, anomaly_rows = self._isolation_forest(X)
        dvae_anomaly_rows = []
        autoencoder = state.results.get("autoencoder")
        if autoencoder and autoencoder.success:
            dvae_anomaly_rows = self._dvae_anomaly_rows(autoencoder.outputs.get("reconstruction_errors", []))

        hooks.progress(self.meta.name, 80, "Computing overlap and interpretations.")

        # --- Overlap with Stage 2 outliers ---
        exploration = state.results.get("exploration")
        overlap = self._compute_overlap(anomaly_rows, exploration)
        anomaly_signal_comparison = self._compare_anomaly_signals(anomaly_rows, dvae_anomaly_rows, overlap)

        hopkins = None
        if exploration and exploration.success:
            hopkins = exploration.outputs.get("hopkins_statistic")

        outputs = {
            "representation_source": representation_name,
            "analysis_rows": int(len(X_analysis)),
            "sampled_for_analysis": bool(len(X_analysis) < n_rows),
            "kmeans_results": kmeans_results,
            "dbscan_results": dbscan_results,
            "isolation_forest_scores": iso_scores,
            "anomaly_rows": anomaly_rows,
            "dvae_anomaly_rows": dvae_anomaly_rows,
            "cluster_outlier_overlap": overlap,
            "anomaly_signal_comparison": anomaly_signal_comparison,
        }
        interpretations = {
            "kmeans_silhouette": self._kmeans_interpretation(kmeans_results, hopkins),
            "dbscan_results": self._dbscan_interpretation(dbscan_results),
            "isolation_forest": self._iforest_interpretation(anomaly_rows, n_rows, overlap),
            "anomaly_signal_comparison": self._anomaly_signal_interpretation(anomaly_signal_comparison),
        }
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs=outputs,
            interpretations=interpretations,
        )

    # ------------------------------------------------------------------
    def _analysis_indices(self, n_rows: int, is_fast: bool) -> np.ndarray:
        max_rows = 3000 if is_fast else 8000
        if n_rows <= max_rows:
            return np.arange(n_rows, dtype=int)
        return np.sort(np.random.default_rng(42).choice(n_rows, size=max_rows, replace=False))

    def _kmeans_sweep(self, X: np.ndarray, max_k: int) -> dict:
        silhouette_scores = {}
        best_k = 2
        best_sil = -1.0
        best_labels = None

        for k in range(2, max_k + 1):
            if k >= len(X):
                break
            km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            sil = float(silhouette_score(X, labels))
            silhouette_scores[k] = round(sil, 4)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels.tolist()

        if best_labels is None:
            best_labels = [0] * len(X)

        cluster_sizes = {}
        for label in best_labels:
            cluster_sizes[str(label)] = cluster_sizes.get(str(label), 0) + 1

        return {
            "optimal_k": best_k,
            "best_silhouette": round(best_sil, 4),
            "silhouette_scores": silhouette_scores,
            "cluster_labels": best_labels,
            "cluster_sizes": cluster_sizes,
        }

    def _dbscan_auto(self, X: np.ndarray) -> dict:
        n_rows = len(X)
        k = min(5, n_rows - 1)
        min_samples = max(5, n_rows // 100)

        try:
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            sorted_dists = np.sort(distances[:, -1])

            # Elbow detection: max second derivative
            if len(sorted_dists) > 4:
                diffs = np.diff(sorted_dists)
                diffs2 = np.diff(diffs)
                elbow_idx = int(np.argmax(diffs2)) + 2
                eps = float(sorted_dists[min(elbow_idx, len(sorted_dists) - 1)])
            else:
                eps = float(np.median(sorted_dists))

            eps = max(eps, 1e-6)
        except Exception:
            eps = 0.5

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())

        return {
            "n_clusters": n_clusters,
            "noise_fraction": round(noise_count / len(X), 4) if len(X) else 0.0,
            "eps_used": round(eps, 4),
            "cluster_labels": labels.tolist(),
        }

    def _isolation_forest(self, X: np.ndarray) -> tuple[np.ndarray, list[int]]:
        fit_X = X
        if len(X) > 10000:
            fit_idx = np.random.default_rng(42).choice(len(X), size=10000, replace=False)
            fit_X = X[np.sort(fit_idx)]
        iso = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
        iso.fit(fit_X)
        scores = -iso.decision_function(X)  # higher = more anomalous
        threshold = np.percentile(scores, 99)
        anomaly_rows = [int(i) for i in np.where(scores >= threshold)[0]]
        return scores, anomaly_rows

    def _compute_overlap(self, anomaly_rows: list[int], exploration) -> dict:
        if not exploration or not exploration.success:
            return {"stage2_outlier_rows": [], "overlap_count": 0, "overlap_rows": []}
        stage2_outliers = set(exploration.outputs.get("outlier_rows", []))
        iso_set = set(anomaly_rows)
        overlap = sorted(iso_set & stage2_outliers)
        return {
            "stage2_outlier_rows": sorted(stage2_outliers),
            "overlap_count": len(overlap),
            "overlap_rows": overlap,
        }

    def _dvae_anomaly_rows(self, reconstruction_errors: list[float]) -> list[int]:
        if not reconstruction_errors:
            return []
        scores = np.asarray(reconstruction_errors, dtype=float)
        threshold = float(np.percentile(scores, 99))
        return [int(i) for i in np.where(scores >= threshold)[0]]

    def _compare_anomaly_signals(
        self,
        iso_rows: list[int],
        dvae_rows: list[int],
        overlap: dict,
    ) -> dict[str, object]:
        stage2_rows = overlap.get("stage2_outlier_rows", [])
        iso_set = set(iso_rows)
        dvae_set = set(dvae_rows)
        stage2_set = set(stage2_rows)
        iso_dvae_overlap = sorted(iso_set & dvae_set)
        triple_overlap = sorted(iso_set & dvae_set & stage2_set)
        return {
            "isolation_forest_rows": sorted(iso_set),
            "dvae_rows": sorted(dvae_set),
            "stage2_rows": sorted(stage2_set),
            "iso_dvae_overlap_count": len(iso_dvae_overlap),
            "iso_dvae_overlap_rows": iso_dvae_overlap,
            "triple_overlap_count": len(triple_overlap),
            "triple_overlap_rows": triple_overlap,
        }

    def _empty_result(self, reason: str) -> StageResult:
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={
                "kmeans_results": {"optimal_k": 0, "best_silhouette": 0.0,
                                   "silhouette_scores": {}, "cluster_labels": [],
                                   "cluster_sizes": {}},
                "dbscan_results": {"n_clusters": 0, "noise_fraction": 0.0,
                                   "eps_used": 0.0, "cluster_labels": []},
                "isolation_forest_scores": np.array([]),
                "anomaly_rows": [],
                "dvae_anomaly_rows": [],
                "cluster_outlier_overlap": {},
                "anomaly_signal_comparison": {},
            },
            interpretations={
                "kmeans_silhouette": reason,
                "dbscan_results": reason,
                "isolation_forest": reason,
                "anomaly_signal_comparison": reason,
            },
        )

    # ------------------------------------------------------------------
    def _kmeans_interpretation(self, results: dict, hopkins: float | None) -> str:
        k = results["optimal_k"]
        sil = results["best_silhouette"]
        sizes = results["cluster_sizes"]

        if sil < 0.1:
            strength = "very weak"
            advice = "No meaningful cluster structure detected."
        elif sil < 0.3:
            strength = "weak"
            advice = "Clusters are present but poorly separated — interpret with caution."
        elif sil < 0.6:
            strength = "moderate"
            advice = "Clusters show reasonable separation."
        else:
            strength = "strong"
            advice = "Well-separated clusters detected."

        parts = [f"Silhouette analysis shows optimum at k={k} (score: {sil:.2f})."]
        parts.append(f"Cluster quality: {strength}. {advice}")

        if sizes:
            size_list = [f"cluster {label}: {count}" for label, count in sorted(sizes.items())]
            parts.append(f"Cluster sizes at k={k}: {', '.join(size_list)}.")

            counts = list(sizes.values())
            if counts:
                ratio = max(counts) / max(min(counts), 1)
                if ratio > 5:
                    parts.append("Warning: clusters are highly imbalanced.")
                elif ratio < 2:
                    parts.append("Clusters are well-balanced.")

        if hopkins is not None:
            if hopkins > 0.5 and sil > 0.3:
                parts.append(
                    f"This is consistent with the Hopkins statistic ({hopkins:.2f}) "
                    "which indicated cluster tendency."
                )
            elif hopkins <= 0.5 and sil < 0.3:
                parts.append(
                    f"Consistent with Hopkins ({hopkins:.2f}) — the data lacks strong "
                    "cluster structure."
                )

        return " ".join(parts)

    def _dbscan_interpretation(self, results: dict) -> str:
        n = results["n_clusters"]
        noise = results["noise_fraction"]
        eps = results["eps_used"]

        parts = [f"DBSCAN (eps={eps:.3f}) found {n} cluster(s)."]
        parts.append(f"Noise fraction: {noise:.1%} of rows classified as noise.")

        if n == 0:
            parts.append(
                "No density-based clusters found — the data may lack dense regions "
                "or the eps parameter needs manual tuning."
            )
        elif noise > 0.3:
            parts.append(
                "High noise fraction suggests sparse or broadly distributed data."
            )
        return " ".join(parts)

    def _iforest_interpretation(
        self, anomaly_rows: list[int], n_rows: int, overlap: dict
    ) -> str:
        n_anom = len(anomaly_rows)
        pct = n_anom / n_rows * 100 if n_rows else 0

        parts = [
            f"Isolation Forest flagged {n_anom} rows ({pct:.1f}%) as anomalous."
        ]
        overlap_count = overlap.get("overlap_count", 0)
        stage2_count = len(overlap.get("stage2_outlier_rows", []))
        if stage2_count > 0:
            parts.append(
                f"Of these, {overlap_count} overlap with Stage 2 row-level outliers "
                f"(out of {stage2_count} Stage 2 outliers)."
            )
        else:
            parts.append(
                "Stage 2 did not flag any row-level outliers, so overlap analysis "
                "is empty for this dataset."
            )
        if n_anom > 0:
            parts.append(
                "These rows may represent rare phenotypes, data entry errors, or "
                "legitimate edge cases worth manual inspection."
            )
        return " ".join(parts)

    def _anomaly_signal_interpretation(self, comparison: dict[str, object]) -> str:
        if not comparison:
            return "DVAE anomaly comparison is unavailable."
        dvae_count = len(comparison.get("dvae_rows", []))
        iso_count = len(comparison.get("isolation_forest_rows", []))
        overlap_count = int(comparison.get("iso_dvae_overlap_count", 0))
        triple_count = int(comparison.get("triple_overlap_count", 0))
        if dvae_count == 0:
            return "DVAE reconstruction errors did not produce a usable anomaly tail for comparison."
        return (
            f"Isolation Forest flagged {iso_count} rows and the DVAE reconstruction tail flagged {dvae_count} rows. "
            f"They overlap on {overlap_count} row(s), and {triple_count} row(s) are supported by Stage 2, DVAE, and Isolation Forest together. "
            "The shared rows are the most credible anomaly candidates because multiple signals agree on them."
        )
