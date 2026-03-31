from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from explot.stages.base import BaseStage, StageMeta, StageResult


class DimensionalityStage(BaseStage):
    meta = StageMeta(name="dimensionality", depends_on=("profiling", "exploration"))

    def run(self, state, config, hooks) -> StageResult:
        profiling = state.results["profiling"]
        exploration = state.results["exploration"]
        df = state.raw_df

        hooks.progress(self.meta.name, 10, "Selecting numeric columns and planning transforms.")

        numeric_cols = list(profiling.outputs.get("numeric_column_names", []))
        norm_guess = profiling.outputs.get("normalization_guess", "unknown")
        suspicious = profiling.outputs.get("suspicious_columns", [])
        redundant_pairs = exploration.outputs.get("redundant_pairs", [])

        transform_log: list[str] = []
        dropped_columns: list[dict[str, str]] = []

        # --- Column filtering ---
        drop_suspicious_reasons = {"id_like", "near_constant"}
        suspicious_names = {
            item["name"]
            for item in suspicious
            if item.get("reason") in drop_suspicious_reasons
        }
        for name in sorted(suspicious_names):
            if name in numeric_cols:
                numeric_cols.remove(name)
                reason = next(
                    (item["reason"] for item in suspicious if item["name"] == name), "suspicious"
                )
                dropped_columns.append({"name": name, "reason": reason})
                transform_log.append(f"Dropped '{name}' (flagged as {reason} by profiling).")

        redundant_to_drop: set[str] = set()
        for pair in redundant_pairs:
            cols = pair.get("columns", [])
            if len(cols) == 2 and cols[1] in numeric_cols and cols[1] not in redundant_to_drop:
                redundant_to_drop.add(cols[1])
        for name in sorted(redundant_to_drop):
            if name in numeric_cols:
                numeric_cols.remove(name)
                dropped_columns.append({"name": name, "reason": "redundant"})
                transform_log.append(
                    f"Dropped '{name}' (redundant pair detected by exploration)."
                )

        hooks.progress(self.meta.name, 30, "Building cleaned numeric matrix.")

        if not numeric_cols:
            return self._empty_result(transform_log, dropped_columns)

        cleaned = df[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()

        # --- Imputation ---
        null_count = int(cleaned.isna().sum().sum())
        if null_count > 0:
            medians = cleaned.median()
            cleaned = cleaned.fillna(medians)
            transform_log.append(
                f"Imputed {null_count} NaN values with column medians across "
                f"{int((medians.index.size - (cleaned.isna().sum() == 0).sum()))} columns."
            )
            # Recount — fillna covers everything, but if a column was all-NaN
            # the median is NaN, so fill remaining with 0.
            if cleaned.isna().any().any():
                cleaned = cleaned.fillna(0.0)
                transform_log.append("Filled remaining all-NaN columns with 0.")

        cleaned_df = cleaned.copy()

        hooks.progress(self.meta.name, 50, "Applying transforms.")

        # --- Optional log1p ---
        if norm_guess == "raw counts":
            cleaned = cleaned.clip(lower=0)
            cleaned = np.log1p(cleaned)
            transform_log.append(
                "Applied log1p (profiling detected raw counts)."
            )

        # --- StandardScaler ---
        scaler = StandardScaler()
        try:
            scaled = scaler.fit_transform(cleaned)
        except Exception:
            scaled = cleaned.values
            transform_log.append("StandardScaler failed — using unscaled values.")
        else:
            transform_log.append("Applied StandardScaler to all numeric features.")

        transformed_df = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)

        hooks.progress(self.meta.name, 70, "Running PCA.")

        # --- PCA ---
        n_components = min(scaled.shape[0], scaled.shape[1])
        pca_2d = np.zeros((len(df), 2))
        explained = []
        components = np.array([])
        intrinsic_dim = 0

        if n_components >= 2:
            try:
                pca = PCA(n_components=n_components)
                proj = pca.fit_transform(scaled)
                pca_2d = proj[:, :2]
                explained = pca.explained_variance_ratio_.tolist()
                components = pca.components_
                intrinsic_dim = self._participation_ratio(pca.explained_variance_)
            except Exception as exc:
                transform_log.append(f"PCA failed: {exc}. Using zero projections.")
        elif n_components == 1:
            try:
                pca = PCA(n_components=1)
                proj = pca.fit_transform(scaled)
                pca_2d = np.column_stack([proj[:, 0], np.zeros(len(df))])
                explained = pca.explained_variance_ratio_.tolist()
                components = pca.components_
                intrinsic_dim = 1
            except Exception as exc:
                transform_log.append(f"PCA failed: {exc}.")
        else:
            transform_log.append("No components available for PCA.")

        cum = np.cumsum(explained) if explained else np.array([])
        n50 = int(np.searchsorted(cum, 0.50) + 1) if len(cum) else 0
        n80 = int(np.searchsorted(cum, 0.80) + 1) if len(cum) else 0
        n95 = int(np.searchsorted(cum, 0.95) + 1) if len(cum) else 0

        transform_log.append(
            f"Final matrix: {scaled.shape[0]} rows x {scaled.shape[1]} features."
        )

        hooks.progress(self.meta.name, 90, "Building interpretations.")

        outputs = {
            "cleaned_df": cleaned_df,
            "transformed_df": transformed_df,
            "pca_components": components,
            "pca_2d": pca_2d,
            "pca_explained_variance": explained,
            "intrinsic_dim": intrinsic_dim,
            "n_components_50": n50,
            "n_components_80": n80,
            "n_components_95": n95,
            "transform_log": transform_log,
            "dropped_columns": dropped_columns,
        }
        figures = {
            "scree_plot": self._scree_svg(explained),
            "projection_plot": self._projection_svg(pca_2d, explained),
        }
        interpretations = {
            "pca_variance": self._pca_interpretation(explained, intrinsic_dim, n50, n80, n95),
            "transform_log": self._transform_interpretation(
                transform_log, dropped_columns, scaled.shape
            ),
            "svd_explainer": self._svd_explainer(),
        }
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs=outputs,
            figures=figures,
            interpretations=interpretations,
        )

    # ------------------------------------------------------------------
    def _participation_ratio(self, eigenvalues: np.ndarray) -> int:
        ev = eigenvalues[eigenvalues > 0]
        if len(ev) == 0:
            return 0
        pr = float((ev.sum() ** 2) / (ev**2).sum())
        return max(1, min(int(round(pr)), len(ev)))

    def _empty_result(self, transform_log, dropped_columns) -> StageResult:
        transform_log.append("No numeric columns remain after filtering. Skipping PCA.")
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={
                "cleaned_df": pd.DataFrame(),
                "transformed_df": pd.DataFrame(),
                "pca_components": np.array([]),
                "pca_2d": np.array([]),
                "pca_explained_variance": [],
                "intrinsic_dim": 0,
                "n_components_50": 0,
                "n_components_80": 0,
                "n_components_95": 0,
                "transform_log": transform_log,
                "dropped_columns": dropped_columns,
            },
            interpretations={
                "pca_variance": "No numeric features available for PCA after column filtering.",
                "transform_log": "All numeric columns were dropped as suspicious or redundant.",
                "svd_explainer": self._svd_explainer(),
            },
        )

    # ------------------------------------------------------------------
    def _pca_interpretation(
        self,
        explained: list[float],
        intrinsic_dim: int,
        n50: int,
        n80: int,
        n95: int,
    ) -> str:
        if not explained:
            return "No PCA results available."
        total = len(explained)
        top1 = explained[0] * 100
        cum5 = sum(explained[:min(5, total)]) * 100

        # Detect elbow: largest drop between consecutive components
        elbow_k = 1
        if len(explained) > 2:
            drops = [explained[i] - explained[i + 1] for i in range(len(explained) - 1)]
            elbow_k = int(np.argmax(drops)) + 1

        parts = [
            f"First {min(5, total)} components explain {cum5:.1f}% of total variance "
            f"(PC1 alone: {top1:.1f}%).",
        ]
        if len(explained) > 2:
            parts.append(
                f"Elbow visible at PC{elbow_k} (largest variance drop between consecutive "
                f"components)."
            )
        parts.append(f"Intrinsic dimensionality estimate: {intrinsic_dim} (participation ratio).")
        parts.append(
            f"Components needed: {n50} for 50% variance, {n80} for 80%, {n95} for 95%."
        )
        if intrinsic_dim <= 5:
            parts.append(
                "Low intrinsic dimensionality — most information concentrates in a compact "
                "subspace, favorable for clustering and supervised learning."
            )
        elif intrinsic_dim <= 20:
            parts.append(
                "Moderate intrinsic dimensionality — dimensionality reduction is beneficial "
                "but the data has meaningful spread across multiple axes."
            )
        else:
            parts.append(
                "High intrinsic dimensionality — the data is genuinely complex. "
                "Downstream methods should expect noise in lower-dimensional projections."
            )
        return " ".join(parts)

    def _transform_interpretation(
        self,
        transform_log: list[str],
        dropped_columns: list[dict[str, str]],
        shape: tuple[int, ...],
    ) -> str:
        n_dropped = len(dropped_columns)
        parts = []
        if n_dropped:
            reasons = {}
            for col in dropped_columns:
                reasons.setdefault(col["reason"], []).append(col["name"])
            reason_strs = [
                f"{len(names)} {reason} ({', '.join(names[:3])}{'...' if len(names) > 3 else ''})"
                for reason, names in reasons.items()
            ]
            parts.append(f"Dropped {n_dropped} columns: {'; '.join(reason_strs)}.")
        parts.append(f"Final transformed matrix: {shape[0]} rows x {shape[1]} features.")
        return " ".join(parts)

    def _svd_explainer(self) -> str:
        return (
            "This stage uses PCA, which is computed from an SVD-style decomposition of the standardized "
            "numeric matrix. In plain language, it rotates the feature space to find new axes that capture "
            "the largest shared variation, so we can see whether many columns are really driven by a smaller "
            "set of latent patterns."
        )

    def _scree_svg(self, explained: list[float]) -> str:
        if not explained:
            return ""
        values = [float(v) * 100.0 for v in explained[:10]]
        cumulative = np.cumsum(values)
        width = 420
        height = 240
        margin = 36
        chart_width = width - margin * 2
        chart_height = height - margin * 2
        slot_width = chart_width / max(len(values), 1)
        bar_width = max(16.0, slot_width - 8.0)

        bars = []
        line_points = []
        labels = []
        for idx, value in enumerate(values):
            x = margin + idx * slot_width + 4
            bar_height = chart_height * min(value, 100.0) / 100.0
            y = height - margin - bar_height
            bars.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' "
                "fill='#0f6a8b' rx='4' />"
            )
            cx = x + bar_width / 2
            cy = height - margin - (chart_height * min(cumulative[idx], 100.0) / 100.0)
            line_points.append(f"{cx:.1f},{cy:.1f}")
            labels.append(
                f"<text x='{cx:.1f}' y='{height - 14}' font-size='10' text-anchor='middle' fill='#5f7584'>PC{idx + 1}</text>"
            )

        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:520px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            f"<rect width='{width}' height='{height}' fill='#f8fbfd' rx='12' />"
            f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            + "".join(bars)
            + f"<polyline points='{' '.join(line_points)}' fill='none' stroke='#ef7d57' stroke-width='3' />"
            + "".join(labels)
            + "<text x='36' y='20' font-size='14' fill='#193042'>Scree Plot</text>"
            + "<text x='292' y='24' font-size='11' fill='#5f7584'>bars = variance, line = cumulative</text>"
            + "</svg>"
        )

    def _projection_svg(self, pca_2d: np.ndarray, explained: list[float]) -> str:
        if pca_2d.size == 0:
            return ""

        points = pca_2d
        if len(points) > 400:
            rng = np.random.default_rng(42)
            points = points[np.sort(rng.choice(len(points), size=400, replace=False))]

        x = points[:, 0]
        y = points[:, 1]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        width = 420
        height = 260
        margin = 34
        chart_width = width - margin * 2
        chart_height = height - margin * 2

        circles = []
        for px, py in zip(x, y):
            sx = margin + ((float(px) - x_min) / (x_max - x_min)) * chart_width
            sy = height - margin - ((float(py) - y_min) / (y_max - y_min)) * chart_height
            circles.append(
                f"<circle cx='{sx:.1f}' cy='{sy:.1f}' r='2.2' fill='rgba(15,106,139,0.55)' />"
            )

        pc1 = explained[0] * 100 if explained else 0.0
        pc2 = explained[1] * 100 if len(explained) > 1 else 0.0

        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:520px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            f"<rect width='{width}' height='{height}' fill='#f8fbfd' rx='12' />"
            f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            + "".join(circles)
            + f"<text x='{width / 2:.1f}' y='20' font-size='14' text-anchor='middle' fill='#193042'>PC1 vs PC2 projection</text>"
            + f"<text x='{width / 2:.1f}' y='{height - 10}' font-size='11' text-anchor='middle' fill='#5f7584'>PC1 ({pc1:.1f}% variance)</text>"
            + f"<text x='16' y='{height / 2:.1f}' font-size='11' text-anchor='middle' fill='#5f7584' transform='rotate(-90 16 {height / 2:.1f})'>PC2 ({pc2:.1f}% variance)</text>"
            + "</svg>"
        )
