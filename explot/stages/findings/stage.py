from __future__ import annotations

from explot.stages.base import BaseStage, StageMeta, StageResult


class FindingsStage(BaseStage):
    meta = StageMeta(
        name="findings",
        depends_on=("profiling",),
        optional_deps=("exploration", "dimensionality", "autoencoder", "unsupervised", "supervised"),
    )

    def run(self, state, config, hooks) -> StageResult:
        findings: list[dict] = []
        hooks.progress(self.meta.name, 10, "Collecting findings from all stages.")

        findings.extend(self._profiling_findings(state))
        findings.extend(self._exploration_findings(state))
        findings.extend(self._dimensionality_findings(state))
        findings.extend(self._autoencoder_findings(state))
        findings.extend(self._unsupervised_findings(state))
        findings.extend(self._supervised_findings(state))

        # Sort: HIGH first, then MEDIUM, then LOW
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda f: order.get(f["confidence"], 3))

        summary_card = [f["text"] for f in findings[:3]]
        next_steps = self._suggest_next_steps(state, findings)

        hooks.progress(self.meta.name, 90, "Done.")

        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={
                "findings_list": findings,
                "summary_card": summary_card,
                "suggested_next_steps": next_steps,
            },
            interpretations={
                "summary": self._summary_interp(findings, summary_card),
            },
        )

    # ------------------------------------------------------------------
    def _get(self, state, stage, key, default=None):
        r = state.results.get(stage)
        if r and r.success:
            return r.outputs.get(key, default)
        return default

    def _profiling_findings(self, state) -> list[dict]:
        findings = []
        quality = self._get(state, "profiling", "quality_score")
        if quality is not None:
            if quality >= 80:
                findings.append(self._f(f"Dataset quality score is {quality}/100 (good).",
                                        "HIGH", "profiling", "quality_score"))
            elif quality >= 50:
                findings.append(self._f(f"Dataset quality score is {quality}/100 (moderate).",
                                        "MEDIUM", "profiling", "quality_score"))
            else:
                findings.append(self._f(f"Dataset quality score is {quality}/100 (low).",
                                        "HIGH", "profiling", "quality_score"))

        norm = self._get(state, "profiling", "normalization_guess")
        if norm and norm != "unknown":
            findings.append(self._f(f"Data appears to be {norm}.",
                                    "MEDIUM", "profiling", "normalization_guess"))

        suspicious = self._get(state, "profiling", "suspicious_columns", [])
        if suspicious:
            names = [s["name"] for s in suspicious[:5]]
            findings.append(self._f(
                f"{len(suspicious)} suspicious column(s) detected: {', '.join(names)}.",
                "MEDIUM", "profiling", "suspicious_columns"))
        return findings

    def _exploration_findings(self, state) -> list[dict]:
        findings = []
        redundant = self._get(state, "exploration", "redundant_pairs", [])
        if redundant:
            n = len(redundant)
            conf = "HIGH" if n >= 3 else "MEDIUM"
            findings.append(self._f(
                f"{n} redundant column pair(s) found (|r| > 0.95).",
                conf, "exploration", "redundant_pairs"))

        hopkins = self._get(state, "exploration", "hopkins_statistic")
        if hopkins is not None:
            if hopkins > 0.7:
                findings.append(self._f(
                    f"Strong cluster tendency detected (Hopkins={hopkins:.2f}).",
                    "HIGH", "exploration", "hopkins_statistic"))
            elif hopkins > 0.5:
                findings.append(self._f(
                    f"Moderate cluster tendency (Hopkins={hopkins:.2f}).",
                    "MEDIUM", "exploration", "hopkins_statistic"))
            else:
                findings.append(self._f(
                    f"Weak cluster tendency (Hopkins={hopkins:.2f}) — data may lack natural groupings.",
                    "LOW", "exploration", "hopkins_statistic"))

        miss_type = self._get(state, "exploration", "missingness_type")
        if miss_type and miss_type != "minimal":
            findings.append(self._f(
                f"Missingness pattern: {miss_type}.",
                "MEDIUM", "exploration", "missingness_type"))
        return findings

    def _dimensionality_findings(self, state) -> list[dict]:
        findings = []
        intrinsic = self._get(state, "dimensionality", "intrinsic_dim")
        if intrinsic and intrinsic > 0:
            findings.append(self._f(
                f"Intrinsic dimensionality estimate: {intrinsic}.",
                "MEDIUM", "dimensionality", "intrinsic_dim"))
        return findings

    def _unsupervised_findings(self, state) -> list[dict]:
        findings = []
        km = self._get(state, "unsupervised", "kmeans_results", {})
        sil = km.get("best_silhouette", 0)
        k = km.get("optimal_k", 0)

        if sil > 0.6:
            findings.append(self._f(
                f"Well-separated clusters found: k={k}, silhouette={sil:.2f}.",
                "HIGH", "unsupervised", "kmeans_silhouette"))
        elif sil > 0.3:
            findings.append(self._f(
                f"Moderate cluster structure: k={k}, silhouette={sil:.2f}.",
                "MEDIUM", "unsupervised", "kmeans_silhouette"))
        elif k > 0:
            findings.append(self._f(
                f"Weak cluster structure: k={k}, silhouette={sil:.2f}.",
                "LOW", "unsupervised", "kmeans_silhouette"))

        anomaly_rows = self._get(state, "unsupervised", "anomaly_rows", [])
        if anomaly_rows:
            findings.append(self._f(
                f"{len(anomaly_rows)} anomalous row(s) detected by Isolation Forest.",
                "MEDIUM", "unsupervised", "isolation_forest"))
        return findings

    def _autoencoder_findings(self, state) -> list[dict]:
        findings = []
        mse = self._get(state, "autoencoder", "reconstruction_mse")
        bottleneck = self._get(state, "autoencoder", "bottleneck_dim")
        if mse is None or not bottleneck:
            return findings
        if mse < 0.1:
            findings.append(self._f(
                f"DVAE reconstruction is strong at latent dimension {bottleneck} (MSE={mse:.3f}).",
                "MEDIUM", "autoencoder", "reconstruction_mse"))
        elif mse < 0.5:
            findings.append(self._f(
                f"DVAE reconstruction is moderate at latent dimension {bottleneck} (MSE={mse:.3f}).",
                "LOW", "autoencoder", "reconstruction_mse"))
        return findings

    def _supervised_findings(self, state) -> list[dict]:
        findings = []
        best_models = self._get(state, "supervised", "best_models", {})
        for target, info in best_models.items():
            score = info.get("mean", 0)
            model = info.get("model", "?")
            metric = info.get("metric", "?")
            flags = info.get("trust_flags", [])
            if score > 0.8:
                findings.append(self._f(
                    f"Target '{target}' is highly predictable: {model} {metric}={score:.2f}.",
                    "HIGH", "supervised", "model_recommendation"))
            elif score > 0.5:
                findings.append(self._f(
                    f"Target '{target}' is moderately predictable: {model} {metric}={score:.2f}.",
                    "MEDIUM", "supervised", "model_recommendation"))
            else:
                findings.append(self._f(
                    f"Target '{target}' shows weak predictability: {model} {metric}={score:.2f}.",
                    "LOW", "supervised", "model_recommendation"))
            if any(flag in flags for flag in ("proxy_like_feature", "near_perfect_score", "possible_leakage", "exact_copy_feature", "high_correlation_proxy", "single_feature_leakage")):
                findings.append(self._f(
                    f"Target '{target}' may be proxy-like or leakage-prone under current heuristics.",
                    "MEDIUM", "supervised", "trust_flags"))
        return findings

    def _suggest_next_steps(self, state, findings) -> list[str]:
        steps = []
        high = [f for f in findings if f["confidence"] == "HIGH"]

        if any(f["rule"] == "kmeans_silhouette" for f in high):
            steps.append("Investigate cluster composition — which features drive separation?")

        best_models = self._get(state, "supervised", "best_models", {})
        for target, info in best_models.items():
            if info.get("mean", 0) > 0.7:
                steps.append(
                    f"Target '{target}' is predictable — consider hyperparameter tuning "
                    f"on {info.get('model', 'best model')} for production use.")

        redundant = self._get(state, "exploration", "redundant_pairs", [])
        if redundant:
            steps.append("Consider removing redundant features before downstream modeling.")

        miss_type = self._get(state, "exploration", "missingness_type")
        if miss_type == "structured":
            steps.append("Investigate structured missingness — MNAR-aware imputation may help.")

        if not steps:
            steps.append("Review the full report for detailed per-stage analysis.")
        return steps

    def _summary_interp(self, findings, summary_card) -> str:
        n = len(findings)
        high = sum(1 for f in findings if f["confidence"] == "HIGH")
        return (
            f"Generated {n} findings: {high} high-confidence, "
            f"{sum(1 for f in findings if f['confidence'] == 'MEDIUM')} medium, "
            f"{sum(1 for f in findings if f['confidence'] == 'LOW')} low."
        )

    @staticmethod
    def _f(text, confidence, source_stage, rule) -> dict:
        return {"text": text, "confidence": confidence,
                "source_stage": source_stage, "rule": rule}
