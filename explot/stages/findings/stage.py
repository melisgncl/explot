from __future__ import annotations

from explot.stages.base import BaseStage, StageMeta, StageResult


class FindingsStage(BaseStage):
    meta = StageMeta(
        name="findings",
        depends_on=("profiling",),
        optional_deps=("exploration", "preprocessing", "dimensionality", "autoencoder", "unsupervised", "supervised"),
    )
    _HARD_STOP_FLAGS = frozenset({
        "group_leakage", "temporal_leakage", "exact_copy_feature",
        "proxy_like_feature", "single_feature_leakage",
    })
    _SOFT_FLAGS = frozenset({
        "high_correlation_proxy", "near_perfect_score",
        "suspicious_feature_name", "severe_class_imbalance",
        "possible_leakage",
    })

    def run(self, state, config, hooks) -> StageResult:
        findings: list[dict] = []
        hooks.progress(self.meta.name, 10, "Collecting findings from all stages.")

        findings.extend(self._profiling_findings(state))
        findings.extend(self._preprocessing_findings(state))
        findings.extend(self._exploration_findings(state))
        findings.extend(self._dimensionality_findings(state))
        findings.extend(self._autoencoder_findings(state))
        findings.extend(self._unsupervised_findings(state))
        findings.extend(self._supervised_findings(state))
        findings.extend(self._survival_findings(state))

        # Sort: HIGH first, then MEDIUM, then LOW
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        findings.sort(key=lambda f: order.get(f["confidence"], 3))

        summary_card = [f["text"] for f in findings[:3]]
        next_steps = self._suggest_next_steps(state, findings)
        verdict = self._verdict(state, config)

        hooks.progress(self.meta.name, 90, "Done.")

        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={
                "findings_list": findings,
                "summary_card": summary_card,
                "suggested_next_steps": next_steps,
                "verdict": verdict,
            },
            interpretations={
                "summary": self._summary_interp(findings, summary_card),
                "verdict": verdict["headline"],
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

        dup_count = self._get(state, "profiling", "duplicate_row_count", 0)
        dup_pct = self._get(state, "profiling", "duplicate_row_percent", 0.0)
        if dup_count > 0:
            conf = "HIGH" if dup_pct > 10 else "MEDIUM" if dup_pct > 1 else "LOW"
            findings.append(self._f(
                f"{dup_count} exact duplicate rows ({dup_pct:.1f}%) — consider deduplication before modeling.",
                conf, "profiling", "duplicate_rows"))

        for s in suspicious:
            if s.get("reason") == "high_dimensional":
                findings.append(self._f(
                    s["details"],
                    "HIGH", "profiling", "high_dimensional"))
                break
        return findings

    def _preprocessing_findings(self, state) -> list[dict]:
        findings = []
        preprocessing = state.results.get("preprocessing")
        if not (preprocessing and preprocessing.success):
            return findings

        imputation_stats = preprocessing.outputs.get("imputation_stats", {})
        heavy_cols = preprocessing.outputs.get("columns_with_heavy_imputation", [])
        dropped_cols = preprocessing.outputs.get("dropped_columns", [])

        for col in heavy_cols:
            stats = imputation_stats.get(col, {})
            pct = stats.get("pct_filled", "?")
            method = stats.get("method", "median")
            findings.append(self._f(
                f"Column '{col}' had {pct}% missing values — imputed with {method}. "
                "Consider whether missingness is informative (a missing value may itself be a signal).",
                "HIGH", "preprocessing", "heavy_imputation",
            ))

        hc_drops = [d for d in dropped_cols if d["reason"] == "high_cardinality"]
        for drop in hc_drops:
            findings.append(self._f(
                f"Column '{drop['name']}' was dropped: {drop['n_unique']} unique string values "
                "exceed the encoding threshold. Consider target encoding or text embeddings.",
                "HIGH", "preprocessing", "high_cardinality_dropped",
            ))

        freq_encoded = preprocessing.outputs.get("frequency_encoded", [])
        if freq_encoded:
            names = ", ".join(f"'{c}'" for c in freq_encoded[:5])
            more = f" (+{len(freq_encoded) - 5} more)" if len(freq_encoded) > 5 else ""
            findings.append(self._f(
                f"{len(freq_encoded)} medium-cardinality column(s) frequency-encoded: {names}{more}. "
                "Frequency encoding preserves prevalence but loses category identity — "
                "consider ordinal or target encoding if category identity matters.",
                "MEDIUM", "preprocessing", "frequency_encoded",
            ))

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

        comparison = self._get(state, "unsupervised", "anomaly_signal_comparison", {})
        triple_count = int(comparison.get("triple_overlap_count", 0))
        if triple_count > 0:
            findings.append(self._f(
                f"{triple_count} row(s) flagged as anomalous by all three signals "
                "(Isolation Forest, DVAE reconstruction, and row-level outlier scoring).",
                "HIGH", "unsupervised", "anomaly_consensus"))
        return findings

    def _autoencoder_findings(self, state) -> list[dict]:
        findings = []
        mse = self._get(state, "autoencoder", "reconstruction_mse")
        bottleneck = self._get(state, "autoencoder", "bottleneck_dim")
        if mse is None or not bottleneck:
            return findings
        if mse < 0.1:
            findings.append(self._f(
                f"DVAE nonlinear compression is strong at latent dimension {bottleneck} (MSE={mse:.3f}), "
                "suggesting the data has compact nonlinear structure beyond what PCA captures.",
                "MEDIUM", "autoencoder", "reconstruction_mse"))
        elif mse < 0.5:
            findings.append(self._f(
                f"DVAE nonlinear compression is moderate at latent dimension {bottleneck} (MSE={mse:.3f}). "
                "The data may be approximately linear, so PCA features are likely sufficient.",
                "LOW", "autoencoder", "reconstruction_mse"))

        # Cross-reference: did DVAE latent features help supervised performance?
        track_comp = self._get(state, "supervised", "track_comparison", {})
        for target, comp in track_comp.items():
            if comp.get("available") and comp.get("winner") == "track_b":
                delta = comp.get("delta", 0)
                findings.append(self._f(
                    f"DVAE latent features outperformed PCA for target '{target}' by {delta:+.4f}, "
                    "confirming meaningful nonlinear structure in this dataset.",
                    "HIGH", "autoencoder", "dvae_supervised_lift"))
                break
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
            elif score > 0.0:
                findings.append(self._f(
                    f"Target '{target}' shows weak predictability: {model} {metric}={score:.2f}.",
                    "LOW", "supervised", "model_recommendation"))
            else:
                findings.append(self._f(
                    f"Target '{target}' is not predictable with current features: {model} {metric}={score:.2f}.",
                    "LOW", "supervised", "model_recommendation"))
            if any(flag in flags for flag in ("proxy_like_feature", "near_perfect_score", "possible_leakage", "exact_copy_feature", "high_correlation_proxy", "single_feature_leakage")):
                findings.append(self._f(
                    f"Target '{target}' may be proxy-like or leakage-prone under current heuristics.",
                    "MEDIUM", "supervised", "trust_flags"))
            if "severe_class_imbalance" in flags:
                findings.append(self._f(
                    f"Target '{target}' has severe class imbalance (>10:1 ratio) — metrics may be misleading.",
                    "HIGH", "supervised", "class_imbalance"))
            elif "class_imbalance" in flags:
                findings.append(self._f(
                    f"Target '{target}' has moderate class imbalance (>5:1 ratio).",
                    "MEDIUM", "supervised", "class_imbalance"))
            if "temporal_feature_present" in flags:
                findings.append(self._f(
                    f"Target '{target}' has temporal features in the dataset — check for temporal leakage.",
                    "MEDIUM", "supervised", "temporal_leakage"))
            diag = info.get("diagnostics", {}) or {}
            if diag.get("used_balanced_weights"):
                ratio = diag.get("imbalance_ratio")
                ratio_str = f" ({ratio:.1f}:1 ratio)" if ratio is not None else ""
                findings.append(self._f(
                    f"Target '{target}': class imbalance detected{ratio_str} — models retrained with balanced class weights.",
                    "MEDIUM", "supervised", "balanced_class_weights"))
            group_audit = diag.get("group_audit", {}) or {}
            if group_audit.get("leakage_detected"):
                findings.append(self._f(
                    f"Target '{target}': group leakage detected on id-like column "
                    f"'{group_audit.get('id_column')}'. "
                    f"KFold score {group_audit.get('kfold_score'):.2f} drops to "
                    f"{group_audit.get('group_score'):.2f} under GroupKFold "
                    f"(delta {group_audit.get('delta'):+.2f}). "
                    "Random folds are leaking the same entities into train and test.",
                    "HIGH", "supervised", "group_leakage"))
            temporal_audit = diag.get("temporal_audit", {}) or {}
            if temporal_audit.get("leakage_detected"):
                findings.append(self._f(
                    f"Target '{target}': temporal leakage detected on time column "
                    f"'{temporal_audit.get('time_column')}'. "
                    f"KFold score {temporal_audit.get('kfold_score'):.2f} drops to "
                    f"{temporal_audit.get('time_score'):.2f} under TimeSeriesSplit "
                    f"(delta {temporal_audit.get('delta'):+.2f}). "
                    "Random folds are using the future to predict the past.",
                    "HIGH", "supervised", "temporal_leakage_empirical"))
            lift = info.get("lift_over_baseline")
            baseline_score = info.get("baseline_score")
            if lift is not None and baseline_score is not None:
                if lift < 0.05:
                    findings.append(self._f(
                        f"Target '{target}': best model barely beats the baseline (lift {lift:+.4f}). "
                        "The target may not be learnable from these features.",
                        "HIGH", "supervised", "low_baseline_lift"))
                elif lift > 0.3:
                    findings.append(self._f(
                        f"Target '{target}': strong lift over baseline ({lift:+.4f}), confirming genuine predictive signal.",
                        "MEDIUM", "supervised", "baseline_lift"))
        return findings

    def _survival_findings(self, state) -> list[dict]:
        findings = []
        survival = state.results.get("survival")
        if not (survival and survival.success):
            return findings
        if not survival.outputs.get("detected"):
            return findings

        n_total = survival.outputs.get("n_total", 0)
        n_events = survival.outputs.get("n_events", 0)
        event_rate = survival.outputs.get("event_rate", 0)
        median = survival.outputs.get("median_survival")
        concordance = survival.outputs.get("cox_concordance")
        time_col = survival.outputs.get("time_column", "?")
        event_col = survival.outputs.get("event_column", "?")
        strat_col = survival.outputs.get("stratify_column")

        findings.append(self._f(
            f"Survival analysis detected: {n_events} events in {n_total:,} observations "
            f"({100 * event_rate:.1f}% event rate) on '{time_col}' / '{event_col}'.",
            "HIGH", "survival", "survival_detected",
        ))

        if median is not None:
            findings.append(self._f(
                f"Median survival time: {median:.1f} (unit: '{time_col}').",
                "MEDIUM", "survival", "median_survival",
            ))

        if concordance is not None:
            if concordance >= 0.70:
                conf, label = "HIGH", "good"
            elif concordance >= 0.60:
                conf, label = "MEDIUM", "moderate"
            else:
                conf, label = "LOW", "weak"
            findings.append(self._f(
                f"Cox PH model C-index: {concordance:.3f} ({label} discrimination). "
                "C-index > 0.70 suggests the covariates meaningfully predict survival.",
                conf, "survival", "cox_concordance",
            ))

        if event_rate < 0.10:
            findings.append(self._f(
                f"Low event rate ({100 * event_rate:.1f}%) — Cox model may be underpowered. "
                "Rule of thumb: ≥10 events per covariate for reliable estimates.",
                "HIGH", "survival", "low_event_rate",
            ))

        if strat_col:
            findings.append(self._f(
                f"Significant survival difference by '{strat_col}' (log-rank p < 0.20) — "
                "see KM stratified curve.",
                "MEDIUM", "survival", "stratified_km",
            ))

        # Top significant Cox covariates
        cox = survival.outputs.get("cox_summary", []) or []
        sig = [c for c in cox if c.get("significant")]
        if sig:
            top = sig[0]
            direction = "increases" if top["coef"] > 0 else "decreases"
            findings.append(self._f(
                f"Top Cox covariate: '{top['covariate']}' (HR={top['exp_coef']:.2f}, p={top['p']:.3f}) — "
                f"higher values {direction} hazard.",
                "HIGH", "survival", "cox_top_covariate",
            ))

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
            flags = info.get("trust_flags", [])
            if "temporal_feature_present" in flags:
                steps.append(
                    f"Review temporal features for target '{target}' — ensure no future-leaking time columns are used as predictors."
                )

        redundant = self._get(state, "exploration", "redundant_pairs", [])
        if redundant:
            steps.append("Consider removing redundant features before downstream modeling.")

        miss_type = self._get(state, "exploration", "missingness_type")
        if miss_type == "structured":
            steps.append("Investigate structured missingness — MNAR-aware imputation may help.")

        if not steps:
            steps.append("Review the full report for detailed per-stage analysis.")
        return steps

    def _verdict(self, state, config=None) -> dict:
        """Synthesize trust flags into a SHIP / INVESTIGATE / DO_NOT_SHIP decision."""
        best_models = self._get(state, "supervised", "best_models", {}) or {}
        verdict_lift_floor = float(getattr(getattr(config, "budget", None), "verdict_lift_floor", 0.05))
        reasons: list[str] = []

        if not best_models:
            return {
                "decision": "NO_MODEL",
                "headline": "No supervised target evaluated — verdict unavailable.",
                "reasons": [],
            }

        worst_decision = "SHIP"
        for target, info in best_models.items():
            flags = set(info.get("trust_flags", []) or [])
            score = float(info.get("mean", 0.0))
            lift = info.get("lift_over_baseline")
            baseline_score = info.get("baseline_score")
            hard_hits = flags & self._HARD_STOP_FLAGS
            soft_hits = flags & self._SOFT_FLAGS

            if hard_hits:
                worst_decision = "DO_NOT_SHIP"
                reasons.append(
                    f"Target '{target}': hard-stop trust flag(s) — {', '.join(sorted(hard_hits))}."
                )
            elif soft_hits:
                if worst_decision != "DO_NOT_SHIP":
                    worst_decision = "INVESTIGATE"
                reasons.append(
                    f"Target '{target}': review flag(s) — {', '.join(sorted(soft_hits))}."
                )

            # Low absolute lift: model barely beats a coin flip
            if lift is not None and float(lift) < verdict_lift_floor and score > 0:
                if worst_decision == "SHIP":
                    worst_decision = "INVESTIGATE"
                reasons.append(
                    f"Target '{target}': lift over dummy baseline is only {lift:+.3f} — model may not be learning."
                )

            # Performance floor: model score within 0.03 of dummy baseline
            if baseline_score is not None and score <= float(baseline_score) + 0.03:
                if worst_decision == "SHIP":
                    worst_decision = "INVESTIGATE"
                reasons.append(
                    f"Target '{target}': best model ({info.get('model', '?')}) scores {score:.3f} — "
                    f"within 0.03 of dummy baseline {float(baseline_score):.3f}. "
                    "Model may not be learning meaningful patterns."
                )

            # Severe class imbalance: F1-macro averages across classes equally, so
            # it structurally hides minority-class failures at extreme ratios.
            # A model with 0.99 majority F1 and 0.20 minority F1 reports 0.60 macro —
            # which looks passable but misses almost every minority case.
            # Force INVESTIGATE so the user reviews per-class metrics before shipping.
            diag = info.get("diagnostics", {}) or {}
            ratio = diag.get("imbalance_ratio")
            if ratio is not None and float(ratio) >= 20:
                if worst_decision == "SHIP":
                    worst_decision = "INVESTIGATE"
                reasons.append(
                    f"Target '{target}': severe class imbalance ({ratio:.0f}:1) — "
                    "F1-macro may mask poor minority-class recall. "
                    "Review per-class precision/recall before shipping."
                )

        # Row-cap disclosure: if data was downsampled, surface it in the verdict
        sampling_notes = self._get(state, "supervised", "sampling_notes", []) or []
        if sampling_notes:
            if worst_decision == "SHIP":
                worst_decision = "INVESTIGATE"
            for note in sampling_notes[:3]:
                reasons.append(f"[Sampling] {note}")

        headlines = {
            "SHIP": "Ship with caution — no blocking trust flags detected.",
            "INVESTIGATE": "Investigate before shipping — trust issues need review.",
            "DO_NOT_SHIP": "Do not ship — one or more leakage or proxy flags fired.",
        }
        return {
            "decision": worst_decision,
            "headline": headlines[worst_decision],
            "reasons": reasons[:10],
        }

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
