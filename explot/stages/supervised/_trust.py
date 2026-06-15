from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.preprocessing import LabelEncoder


class _TrustAuditor:
    _TRACK_LABELS = {"track_a": "PCA features", "track_b": "DVAE latent features"}

    def _target_diagnostics(
        self,
        raw_df: pd.DataFrame,
        target_name: str,
        valid_idx: pd.Index,
        track_outputs: dict[str, dict[str, object]],
        is_classification: bool,
        profiling=None,
    ) -> dict[str, object]:
        best_score = max(track["best"]["mean"] for track in track_outputs.values())
        sample_idx = valid_idx
        if len(sample_idx) > 5000:
            sample_idx = sample_idx.to_series().sample(n=5000, random_state=42).index

        target_sample = raw_df.loc[sample_idx, target_name]
        exact_copy_columns: list[str] = []
        deterministic_proxy_columns: list[str] = []
        high_corr_proxy_columns: list[str] = []
        suspicious_name_columns: list[str] = []
        target_name_tokens = {token for token in str(target_name).lower().replace("_", " ").split() if token}
        for col in raw_df.columns:
            if col == target_name:
                continue
            feature = raw_df.loc[sample_idx, col]
            aligned = pd.DataFrame({"target": target_sample, "feature": feature}).dropna()
            if len(aligned) < 20:
                continue
            feature_as_str = aligned["feature"].astype(str)
            target_as_str = aligned["target"].astype(str)
            if feature_as_str.equals(target_as_str):
                exact_copy_columns.append(str(col))
                continue
            feature_card = int(aligned["feature"].nunique())
            # Only check feature->target determinism when the feature has
            # genuinely low cardinality relative to row count. For continuous
            # features, each row has a unique value and would trivially map
            # to one target, producing a false proxy flag.
            if feature_card <= 2000 and feature_card <= max(5, len(aligned) // 2):
                mapping = aligned.groupby("feature")["target"].nunique()
                if not mapping.empty and float((mapping <= 1).mean()) >= 0.995:
                    deterministic_proxy_columns.append(str(col))
                    continue

            numeric_feature = pd.to_numeric(aligned["feature"], errors="coerce")
            numeric_target = pd.to_numeric(aligned["target"], errors="coerce")
            numeric_pair = pd.DataFrame({"feature": numeric_feature, "target": numeric_target}).dropna()
            if len(numeric_pair) >= 20 and numeric_pair["feature"].nunique() > 1 and numeric_pair["target"].nunique() > 1:
                corr = float(np.corrcoef(numeric_pair["feature"], numeric_pair["target"])[0, 1])
                if np.isfinite(corr) and abs(corr) >= 0.995:
                    high_corr_proxy_columns.append(str(col))

            column_tokens = {token for token in str(col).lower().replace("_", " ").split() if token}
            shared_tokens = target_name_tokens & column_tokens
            if shared_tokens and len(shared_tokens) >= max(1, min(2, len(target_name_tokens))):
                suspicious_name_columns.append(str(col))

        # Check for temporal features used as predictors
        datetime_cols = set()
        if profiling and profiling.success:
            datetime_cols = set(profiling.outputs.get("datetime_column_names", []))
            role_by_col = self._role_by_column(profiling)
            datetime_cols |= {col for col, role in role_by_col.items() if role == "time_like"}
        temporal_feature_columns = [
            str(col) for col in raw_df.columns
            if str(col) in datetime_cols and str(col) != target_name
        ]

        single_feature_leakage_columns: list[str] = []
        if is_classification and best_score > 0.0:
            single_feature_leakage_columns = self._single_feature_leakage_check(
                raw_df, target_name, sample_idx, is_classification, best_score,
            )

        trust_flags: list[str] = []
        if single_feature_leakage_columns:
            trust_flags.append("single_feature_leakage")
            trust_flags.append("possible_leakage")
        if exact_copy_columns:
            trust_flags.append("exact_copy_feature")
            trust_flags.append("possible_leakage")
        if deterministic_proxy_columns:
            trust_flags.append("proxy_like_feature")
            trust_flags.append("possible_leakage")
        if high_corr_proxy_columns:
            trust_flags.append("high_correlation_proxy")
        if suspicious_name_columns:
            trust_flags.append("suspicious_feature_name")
        near_perfect_threshold = 0.95 if is_classification else 0.99
        if best_score >= near_perfect_threshold:
            trust_flags.append("near_perfect_score")
        if best_score >= 0.9 and (exact_copy_columns or deterministic_proxy_columns or high_corr_proxy_columns):
            trust_flags.append("possible_leakage")
        if temporal_feature_columns:
            trust_flags.append("temporal_feature_present")
        if "track_b" in track_outputs and track_outputs["track_b"]["best"]["mean"] > track_outputs.get("track_a", {"best": {"mean": -np.inf}})["best"]["mean"]:
            trust_flags.append("latent_representation_helped")

        # Class imbalance detection
        imbalance_ratio = None
        if is_classification:
            counts = target_sample.value_counts()
            if len(counts) >= 2:
                imbalance_ratio = round(float(counts.iloc[0] / counts.iloc[-1]), 2)
                if imbalance_ratio >= 10:
                    trust_flags.append("severe_class_imbalance")
                elif imbalance_ratio >= 5:
                    trust_flags.append("class_imbalance")

        return {
            "trust_flags": sorted(set(trust_flags)),
            "single_feature_leakage_columns": sorted(set(single_feature_leakage_columns))[:3],
            "proxy_columns": sorted(set(exact_copy_columns + deterministic_proxy_columns + high_corr_proxy_columns))[:5],
            "exact_copy_columns": sorted(set(exact_copy_columns))[:5],
            "deterministic_proxy_columns": sorted(set(deterministic_proxy_columns))[:5],
            "high_corr_proxy_columns": sorted(set(high_corr_proxy_columns))[:5],
            "suspicious_name_columns": sorted(set(suspicious_name_columns))[:5],
            "temporal_feature_columns": temporal_feature_columns[:5],
            "imbalance_ratio": imbalance_ratio,
        }

    def _single_feature_leakage_check(
        self,
        raw_df: pd.DataFrame,
        target_name: str,
        sample_idx: pd.Index,
        is_classification: bool,
        best_score: float,
    ) -> list[str]:
        threshold = 0.8 * best_score
        if threshold < 0.6:
            return []
        idx = sample_idx
        if len(idx) > 2000:
            idx = idx.to_series().sample(n=2000, random_state=42).index
        y = raw_df.loc[idx, target_name]
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        if len(np.unique(y_enc)) < 2:
            return []
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        leaky: list[str] = []
        for col in raw_df.columns:
            if col == target_name:
                continue
            x = pd.to_numeric(raw_df.loc[idx, col], errors="coerce")
            if x.isna().sum() > len(x) * 0.5:
                continue
            x_filled = x.fillna(x.median()).to_numpy().reshape(-1, 1)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_validate(
                        LogisticRegression(max_iter=200, random_state=42),
                        x_filled, y_enc, cv=cv, scoring="f1_macro",
                    )
                mean_score = float(np.mean(scores["test_score"]))
                if mean_score >= threshold:
                    leaky.append(str(col))
                    if len(leaky) >= 3:
                        break
            except Exception:
                continue
        return leaky

    def _role_by_column(self, profiling) -> dict[str, str]:
        if not (profiling and profiling.success):
            return {}
        return {
            str(name): str(profile.get("role_guess", "unknown"))
            for name, profile in profiling.outputs.get("column_profiles", {}).items()
        }

    def _id_like_columns(self, profiling) -> list[str]:
        return [col for col, role in self._role_by_column(profiling).items() if role == "id_like"]

    def _rescore_with_splitter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        estimator,
        is_clf: bool,
        splitter,
        groups: np.ndarray | None,
    ) -> float | None:
        """Score estimator under a different CV splitter."""
        if estimator is None:
            return None
        scoring = "f1_macro" if is_clf else "r2"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_validate(
                    clone(estimator),
                    X,
                    y,
                    cv=splitter,
                    scoring=scoring,
                    groups=groups,
                )
            return float(np.mean(scores["test_score"]))
        except Exception:
            return None

    def _group_leakage_audit(
        self,
        raw_df: pd.DataFrame,
        target_name: str,
        valid_idx: pd.Index,
        best_track: dict,
        estimator,
        is_clf: bool,
        profiling,
        leakage_delta_threshold: float = 0.10,
        leakage_score_floor: float = 0.60,
    ) -> dict:
        """If an id-like column exists, re-score the best model with GroupKFold
        and compare to KFold. A large positive delta means random folds leaked
        the id into both train and test.
        """
        id_cols = [c for c in self._id_like_columns(profiling) if c != target_name and c in raw_df.columns]
        result = {
            "checked": False,
            "leakage_detected": False,
            "id_column": None,
            "kfold_score": None,
            "group_score": None,
            "delta": None,
            "note": "No id-like column detected — group leakage check skipped.",
        }
        if not id_cols:
            return result

        # Pick the id column with the highest cardinality (most likely a true id)
        id_col = max(id_cols, key=lambda c: raw_df[c].nunique())
        groups = raw_df.loc[valid_idx, id_col].astype(str).to_numpy()
        n_unique_groups = len(np.unique(groups))
        if n_unique_groups < 3 or n_unique_groups >= len(groups):
            # Either too few groups for GroupKFold, or every row is its own
            # group (which makes GroupKFold identical to KFold).
            result["note"] = (
                f"Group audit skipped: '{id_col}' has {n_unique_groups} unique groups "
                f"over {len(groups)} rows."
            )
            return result

        X_df = best_track.get("X_df")
        y_encoded = best_track.get("y_encoded")
        if X_df is None or y_encoded is None:
            return result
        X = X_df.to_numpy(dtype=float)
        kfold_score = best_track["best"]["mean"]
        n_splits = min(5, n_unique_groups)
        group_score = self._rescore_with_splitter(
            X, y_encoded, estimator, is_clf, GroupKFold(n_splits=n_splits), groups,
        )
        if group_score is None:
            return result

        delta = round(float(kfold_score - group_score), 4)
        detected = delta >= leakage_delta_threshold and kfold_score >= leakage_score_floor
        return {
            "checked": True,
            "leakage_detected": detected,
            "id_column": id_col,
            "kfold_score": round(float(kfold_score), 4),
            "group_score": round(float(group_score), 4),
            "delta": delta,
            "note": (
                f"GroupKFold on '{id_col}' scored {group_score:.3f} vs KFold {kfold_score:.3f} "
                f"(delta {delta:+.3f}). " + ("Likely fold leakage." if detected else "No leakage signal.")
            ),
        }

    def _temporal_leakage_audit(
        self,
        raw_df: pd.DataFrame,
        target_name: str,
        valid_idx: pd.Index,
        best_track: dict,
        estimator,
        is_clf: bool,
        profiling,
        leakage_delta_threshold: float = 0.10,
        leakage_score_floor: float = 0.60,
    ) -> dict:
        """If a datetime column exists, re-score the best model with
        TimeSeriesSplit on time-sorted data. A large positive delta means
        random folds were using the future to predict the past.
        """
        result = {
            "checked": False,
            "leakage_detected": False,
            "time_column": None,
            "kfold_score": None,
            "time_score": None,
            "delta": None,
            "note": "No datetime column detected — temporal leakage check skipped.",
        }
        datetime_cols: list[str] = []
        if profiling and profiling.success:
            datetime_cols = list(profiling.outputs.get("datetime_column_names", []))
        datetime_cols = [c for c in datetime_cols if c != target_name and c in raw_df.columns]
        if not datetime_cols:
            return result

        time_col = datetime_cols[0]
        time_series = pd.to_datetime(raw_df.loc[valid_idx, time_col], errors="coerce")
        ordered_idx = time_series.dropna().sort_values().index
        if len(ordered_idx) < 40:
            result["note"] = f"Temporal audit skipped: only {len(ordered_idx)} parseable timestamps."
            return result

        X_df = best_track.get("X_df")
        y_encoded = best_track.get("y_encoded")
        if X_df is None or y_encoded is None:
            return result

        aligned_X = X_df.loc[X_df.index.intersection(ordered_idx)].reindex(ordered_idx)
        aligned_y = pd.Series(y_encoded, index=X_df.index).reindex(ordered_idx)
        aligned = pd.concat([aligned_X, aligned_y.rename("_y")], axis=1).dropna()
        if len(aligned) < 40:
            result["note"] = f"Temporal audit skipped: only {len(aligned)} rows after alignment."
            return result

        X = aligned.drop(columns=["_y"]).to_numpy(dtype=float)
        y_arr = aligned["_y"].to_numpy()
        if is_clf and len(np.unique(y_arr)) < 2:
            result["note"] = "Temporal audit skipped: only one class present after time ordering."
            return result

        n_splits = min(5, max(2, len(aligned) // 20))
        time_score = self._rescore_with_splitter(
            X, y_arr, estimator, is_clf, TimeSeriesSplit(n_splits=n_splits), None,
        )
        if time_score is None:
            return result
        kfold_score = best_track["best"]["mean"]
        delta = round(float(kfold_score - time_score), 4)
        detected = delta >= leakage_delta_threshold and kfold_score >= leakage_score_floor
        return {
            "checked": True,
            "leakage_detected": detected,
            "time_column": time_col,
            "kfold_score": round(float(kfold_score), 4),
            "time_score": round(float(time_score), 4),
            "delta": delta,
            "note": (
                f"TimeSeriesSplit on '{time_col}' scored {time_score:.3f} vs KFold {kfold_score:.3f} "
                f"(delta {delta:+.3f}). " + ("Likely temporal leakage." if detected else "No leakage signal.")
            ),
        }

    def _comparison_interp(self, best_models) -> str:
        if not best_models:
            return "No models were successfully trained."
        parts = []
        for target, info in best_models.items():
            track = info.get("track", "track_a")
            track_label = self._TRACK_LABELS.get(track, track)
            parts.append(
                f"Target '{target}': best overall result came from {track_label} using "
                f"{info.get('model', '?')} ({info.get('metric', '?')}={info.get('mean', 0):.2f} +/- {info.get('std', 0):.2f})."
            )
        return " ".join(parts)

    def _fi_interp(self, feature_importances) -> str:
        if not feature_importances:
            return "No feature importance data available."
        parts = []
        for target, track_map in feature_importances.items():
            for track_name, feats in track_map.items():
                top3 = feats[:3]
                if top3:
                    names = ", ".join(f["feature"] for f in top3)
                    parts.append(f"Target '{target}' {track_name} top features are {names}.")
        return " ".join(parts)

    def _track_comparison_interp(self, track_comparison) -> str:
        if not track_comparison:
            return (
                "No PCA vs DVAE comparison is available. "
                "The DVAE autoencoder provides a nonlinear alternative to PCA for feature compression; "
                "when both are available, Explot compares model performance on each representation."
            )
        parts = []
        for target, item in track_comparison.items():
            if not item.get("available"):
                parts.append(str(item.get("summary", "")))
                continue
            winner = item.get("winner", "track_a")
            winner_label = self._TRACK_LABELS.get(winner, winner)
            delta = item.get("delta", 0)
            parts.append(
                f"Target '{target}': {winner_label} won by {abs(delta):+.4f}. "
                f"{'The DVAE latent space captured nonlinear structure that PCA missed.' if winner == 'track_b' else 'PCA features were sufficient — the data structure is mostly linear.'}"
            )
        return " ".join(parts)

    def _trust_notes(self, best_models) -> str:
        if not best_models:
            return "No model trust notes are available because no targets were successfully probed."
        parts = []
        for target, info in best_models.items():
            diagnostics = info.get("diagnostics", {})
            flags = diagnostics.get("trust_flags", [])
            proxies = diagnostics.get("proxy_columns", [])
            exact = diagnostics.get("exact_copy_columns", [])
            deterministic = diagnostics.get("deterministic_proxy_columns", [])
            corr_like = diagnostics.get("high_corr_proxy_columns", [])
            suspicious_names = diagnostics.get("suspicious_name_columns", [])
            single_feat = diagnostics.get("single_feature_leakage_columns", [])
            if single_feat:
                parts.append(f"Target '{target}' can be predicted by a single feature alone ({', '.join(single_feat[:3])}), which is a strong leakage signal.")
            if exact:
                parts.append(f"Target '{target}' has exact-copy feature(s) ({', '.join(exact[:3])}), which is a strong leakage warning.")
            elif deterministic:
                parts.append(f"Target '{target}' has near-deterministic proxy feature(s) ({', '.join(deterministic[:3])}) where each feature value maps to a single target value, so high scores should be treated cautiously.")
            elif corr_like:
                parts.append(f"Target '{target}' has feature(s) with near-perfect numeric correlation ({', '.join(corr_like[:3])}), which can indicate proxy leakage.")
            elif "near_perfect_score" in flags:
                parts.append(f"Target '{target}' reached a near-perfect score, which can indicate a very easy task or potential proxy leakage.")
            elif suspicious_names:
                parts.append(f"Target '{target}' has feature names that look closely related ({', '.join(suspicious_names[:3])}), so review for duplicated semantics.")
            elif "proxy_like_feature" in flags and proxies:
                parts.append(f"Target '{target}' has proxy-like columns ({', '.join(proxies[:3])}), so high scores should be interpreted cautiously.")
            else:
                parts.append(f"Target '{target}' does not show obvious proxy flags under the current heuristics.")
            imbalance = diagnostics.get("imbalance_ratio")
            if imbalance and imbalance >= 5:
                severity = "severely" if imbalance >= 10 else "moderately"
                parts.append(
                    f"Target '{target}' is {severity} imbalanced (majority:minority ratio {imbalance:.0f}:1). "
                    "Consider stratified sampling or resampling techniques before production use."
                )
        return " ".join(parts)

    def _recommendation_interp(self, best_models, intrinsic_dim, silhouette) -> str:
        if not best_models:
            return "No model recommendations - no targets were successfully probed."
        parts = []
        for target, info in best_models.items():
            model = info["model"]
            score = info["mean"]
            metric = info["metric"]
            flags = info.get("trust_flags", [])
            track = info.get("track", "track_a")
            track_label = self._TRACK_LABELS.get(track, track)
            parts.append(f"For target '{target}', {model} on {track_label} achieved the best {metric} of {score:.2f}.")
            lift = info.get("lift_over_baseline")
            baseline_score = info.get("baseline_score")
            if lift is not None and baseline_score is not None:
                parts.append(f"Baseline score: {baseline_score:.2f}, lift: {lift:+.2f}.")
            if "Forest" in model or "XGBoost" in model or "LightGBM" in model:
                parts.append("Tree-based models excel here, suggesting nonlinear feature interactions or heterogeneous subgroups in the data.")
            elif "Logistic" in model or "Ridge" in model:
                parts.append("A linear model won, suggesting the target is largely explained by linear combinations of features.")
            elif "SVM" in model:
                parts.append("SVM with RBF kernel won, suggesting nonlinear but smooth decision boundaries.")

            if intrinsic_dim and intrinsic_dim <= 10:
                parts.append(f"The data has low intrinsic dimensionality ({intrinsic_dim}), which generally favors simpler models.")
            if silhouette is not None and silhouette > 0.5:
                parts.append(f"Strong cluster structure (silhouette={silhouette:.2f}) suggests the target may align with natural groupings.")
            if metric == "R2" and score <= 0.0:
                parts.append("The model does not explain any variance — the target may be noise or require different features entirely.")
            elif score > 0.8:
                parts.append("This is a strong result — the target is well-predictable.")
            elif score > 0.5:
                parts.append("Moderate predictability — further feature engineering may help.")
            else:
                parts.append("Weak predictability — the target may not be well-explained by these features, or the sample size may be too small.")
            if any(flag in flags for flag in ("proxy_like_feature", "near_perfect_score", "possible_leakage", "exact_copy_feature", "high_correlation_proxy")):
                parts.append("Trust note: this target shows proxy/leakage-like patterns under the current heuristics.")
        return " ".join(parts)
