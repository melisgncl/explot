from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.base import clone

from explot.stages.base import BaseStage, StageMeta, StageResult
from explot.stages.supervised._importance import _FeatureImportance
from explot.stages.supervised._trust import _TrustAuditor

_TARGET_KEYWORDS = {"target", "label", "class", "outcome", "group", "type", "diagnosis", "status"}
_REGRESSION_KEYWORDS = {
    "score", "value", "price", "cost", "amount", "survival", "duration",
    "response", "expression", "concentration", "ic50", "ec50", "dose",
    "yield", "output", "result", "measure", "index", "ratio",
}


def _try_import_xgboost():
    try:
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier, XGBRegressor
    except ImportError:
        return None, None


def _try_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        return LGBMClassifier, LGBMRegressor
    except ImportError:
        return None, None


class SupervisedStage(BaseStage):
    meta = StageMeta(
        name="supervised",
        depends_on=("dimensionality",),
        optional_deps=("profiling", "preprocessing", "unsupervised", "autoencoder"),
    )
    _trust = _TrustAuditor()
    _importance = _FeatureImportance()

    def run(self, state, config, hooks) -> StageResult:
        dim = state.results["dimensionality"]
        autoencoder = state.results.get("autoencoder")
        profiling = state.results.get("profiling")
        unsupervised = state.results.get("unsupervised")

        track_a_df = dim.outputs.get("transformed_df")
        track_b_df = None
        if autoencoder and autoencoder.success:
            latent_df = autoencoder.outputs.get("latent_df")
            if isinstance(latent_df, pd.DataFrame) and not latent_df.empty:
                track_b_df = latent_df

        # Track C: raw preprocessed features (no PCA). Tree models often outperform
        # PCA tracks when the encoded feature structure is preserved intact.
        track_c_df = None
        preprocessing = state.results.get("preprocessing")
        if preprocessing and preprocessing.success:
            prep_df = preprocessing.outputs.get("preprocessed_df")
            if isinstance(prep_df, pd.DataFrame) and not prep_df.empty:
                track_c_df = prep_df

        if track_a_df is None or track_a_df.empty:
            return self._empty_result("No features available for supervised probes.")

        hooks.progress(self.meta.name, 10, "Detecting candidate target columns.")

        # Respect explicit --target flag from user
        if state.target_column:
            if state.target_column not in state.raw_df.columns:
                return self._empty_result(
                    f"Specified target column '{state.target_column}' not found in data. "
                    f"Available columns: {', '.join(str(c) for c in state.raw_df.columns[:20])}"
                )
            series = state.raw_df[state.target_column].dropna()
            nunique = series.nunique()
            if state.task_type == "classification":
                task = "classification"
            elif state.task_type == "regression":
                task = "regression"
            else:
                task = "classification" if nunique <= 20 else "regression"
            candidates = [{"name": state.target_column, "task_type": task, "n_classes": int(nunique)}]
        else:
            candidates = self._detect_targets(state.raw_df, profiling)

        if not candidates:
            return self._empty_result(
                "No candidate target columns detected. Explot looks for categorical "
                "columns with 2-20 unique values or columns named like 'target', 'label', "
                "'class', 'outcome', etc. Use --target COLUMN to specify one explicitly."
            )

        is_fast = self._is_fast(config)
        budget = getattr(config, "budget", None)
        n_folds = 3 if is_fast else 5
        max_fit_rows = int(getattr(budget, "max_fit_rows_fast" if is_fast else "max_fit_rows", 3000 if is_fast else 8000))
        leakage_delta_threshold = float(getattr(budget, "leakage_delta_threshold", 0.10))
        leakage_score_floor = float(getattr(budget, "leakage_score_floor", 0.60))

        model_results_track_a: dict[str, list[dict]] = {}
        model_results_track_b: dict[str, list[dict]] = {}
        model_results_track_c: dict[str, list[dict]] = {}
        feature_importances: dict[str, dict[str, list[dict]]] = {}
        best_models: dict[str, dict] = {}
        track_comparison: dict[str, dict[str, object]] = {}
        evaluation_details: dict[str, dict[str, object]] = {}
        sampling_notes: list[str] = []
        sampling_info: dict[str, dict[str, int]] = {}

        for i, cand in enumerate(candidates):
            pct = 20 + int(60 * i / len(candidates))
            target_name = cand["name"]
            hooks.progress(self.meta.name, pct, f"Probing target '{target_name}'.")

            y_raw = state.raw_df[target_name].dropna()
            valid_idx = y_raw.index.intersection(track_a_df.index)
            y_series = y_raw.loc[valid_idx]
            if len(y_series) < 20:
                continue

            is_clf = cand["task_type"] == "classification"
            total_rows = len(valid_idx)
            sampled_idx = self._sample_indices(y_series, is_clf, max_fit_rows)
            if sampled_idx is not None:
                note = (
                    f"Target '{target_name}' was scored on a deterministic sample of "
                    f"{len(sampled_idx)} rows from {total_rows} available rows."
                )
                sampling_notes.append(note)
                sampling_info[target_name] = {"sampled": len(sampled_idx), "total": total_rows}
                if hasattr(hooks, "log"):
                    hooks.log(self.meta.name, note)
                valid_idx = sampled_idx
                y_series = y_series.loc[valid_idx]

            y_encoded, target_labels = self._encode_target(y_series, is_clf)
            cv = self._build_cv(y_encoded, is_clf, n_folds)

            # Detect class imbalance before building models so we can apply
            # balanced weights to LR, RF, and SVM when ratio >= 5.
            imbalance_ratio = None
            use_balanced = False
            if is_clf:
                counts = pd.Series(y_encoded).value_counts()
                if len(counts) >= 2 and counts.iloc[-1] > 0:
                    imbalance_ratio = round(float(counts.iloc[0] / counts.iloc[-1]), 2)
                    if imbalance_ratio >= 5:
                        use_balanced = True

            track_outputs: dict[str, dict[str, object]] = {}
            for track_name, feature_df in (("track_a", track_a_df), ("track_b", track_b_df), ("track_c", track_c_df)):
                if feature_df is None or feature_df.empty:
                    continue
                prepared = self._prepare_track_features(feature_df, target_name, valid_idx, hooks)
                if prepared is None:
                    continue
                X_df, removed_target = prepared
                track_result = self._evaluate_track(
                    X_df,
                    y_series,
                    y_encoded,
                    target_labels,
                    is_clf,
                    is_fast,
                    cv,
                    use_balanced=use_balanced,
                )
                if track_result is None:
                    continue
                track_result["removed_target"] = removed_target
                track_outputs[track_name] = track_result

            if not track_outputs:
                continue

            if "track_a" in track_outputs:
                model_results_track_a[target_name] = track_outputs["track_a"]["results"]
            if "track_b" in track_outputs:
                model_results_track_b[target_name] = track_outputs["track_b"]["results"]
            if "track_c" in track_outputs:
                model_results_track_c[target_name] = track_outputs["track_c"]["results"]

            diagnostics = self._trust._target_diagnostics(state.raw_df, target_name, valid_idx, track_outputs, is_clf, profiling)
            diagnostics["used_balanced_weights"] = use_balanced
            if imbalance_ratio is not None:
                diagnostics["imbalance_ratio"] = imbalance_ratio
            best_track_name, best_track = max(
                track_outputs.items(),
                key=lambda item: item[1]["best"]["mean"],
            )

            # Audit for group- and time-based leakage by re-scoring the best
            # model with alternative splitters and comparing to the KFold score.
            audit_models = dict(self._build_models(is_clf, is_fast=is_fast, n_rows=len(valid_idx)))
            audit_estimator = audit_models.get(best_track["best"]["model"])
            group_audit = self._trust._group_leakage_audit(
                state.raw_df, target_name, valid_idx, best_track, audit_estimator, is_clf, profiling,
                leakage_delta_threshold, leakage_score_floor,
            )
            temporal_audit = self._trust._temporal_leakage_audit(
                state.raw_df, target_name, valid_idx, best_track, audit_estimator, is_clf, profiling,
                leakage_delta_threshold, leakage_score_floor,
            )
            best_track.pop("X_df", None)
            best_track.pop("y_encoded", None)
            diagnostics["group_audit"] = group_audit
            diagnostics["temporal_audit"] = temporal_audit
            if group_audit.get("leakage_detected"):
                diagnostics["trust_flags"] = sorted(set(diagnostics["trust_flags"] + ["group_leakage", "possible_leakage"]))
            if temporal_audit.get("leakage_detected"):
                diagnostics["trust_flags"] = sorted(set(diagnostics["trust_flags"] + ["temporal_leakage", "possible_leakage"]))

            best_models[target_name] = {
                **best_track["best"],
                "track": best_track_name,
                "trust_flags": diagnostics["trust_flags"],
                "diagnostics": diagnostics,
            }
            feature_importances[target_name] = {
                track_name: data["feature_importance"] for track_name, data in track_outputs.items()
            }
            # Store permutation and SHAP importance from the best track
            best_perm = track_outputs.get(best_track_name, {}).get("permutation_importance", [])
            if best_perm:
                best_models[target_name]["permutation_importance"] = best_perm
            best_shap = track_outputs.get(best_track_name, {}).get("shap_importance", [])
            if best_shap:
                best_models[target_name]["shap_importance"] = best_shap
            best_waterfall = track_outputs.get(best_track_name, {}).get("shap_waterfall", {})
            if best_waterfall:
                best_models[target_name]["shap_waterfall"] = best_waterfall
            # Store export estimator (refitted on full data) — excluded from JSON by _SKIP_KEYS
            export_est = track_outputs.get(best_track_name, {}).get("export_estimator")
            if export_est is not None:
                best_models[target_name]["_export_estimator"] = export_est
                best_models[target_name]["_export_feature_names"] = (
                    track_outputs.get(best_track_name, {}).get("export_feature_names", [])
                )
            evaluation_details[target_name] = {
                track_name: {
                    "metrics": data["best_metrics"],
                    "confusion_matrix": data["confusion_matrix"],
                    "labels": data["labels"],
                    "best_model": data["best"],
                }
                for track_name, data in track_outputs.items()
            }
            track_comparison[target_name] = self._compare_tracks(track_outputs)

        hooks.progress(self.meta.name, 90, "Building interpretations.")

        intrinsic_dim = dim.outputs.get("intrinsic_dim", 0)
        sil = None
        if unsupervised and unsupervised.success:
            sil = unsupervised.outputs.get("kmeans_results", {}).get("best_silhouette")

        outputs = {
            "candidate_targets": candidates,
            "model_results_track_a": model_results_track_a,
            "model_results_track_b": model_results_track_b,
            "model_results_track_c": model_results_track_c,
            "feature_importances": feature_importances,
            "best_models": best_models,
            "track_comparison": track_comparison,
            "evaluation_details": evaluation_details,
            "sampling_notes": sampling_notes,
            "sampling_info": sampling_info,
        }
        interpretations = {
            "model_comparison": self._trust._comparison_interp(best_models),
            "feature_importances": self._trust._fi_interp(feature_importances),
            "model_recommendation": self._trust._recommendation_interp(best_models, intrinsic_dim, sil),
            "track_comparison": self._trust._track_comparison_interp(track_comparison),
            "trust_notes": self._trust._trust_notes(best_models),
        }
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs=outputs,
            interpretations=interpretations,
        )

    def _prepare_track_features(
        self,
        feature_df: pd.DataFrame,
        target_name: str,
        valid_idx: pd.Index,
        hooks,
    ) -> tuple[pd.DataFrame, bool] | None:
        reduced = feature_df.drop(columns=[target_name], errors="ignore")
        removed_target = reduced.shape[1] != feature_df.shape[1]
        if removed_target and hasattr(hooks, "log"):
            hooks.log(
                self.meta.name,
                f"Removed target column '{target_name}' from the feature matrix to avoid self-leakage.",
            )
        if reduced.empty:
            return None
        return reduced.loc[valid_idx], removed_target

    def _evaluate_track(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series,
        y_encoded: np.ndarray,
        target_labels: list[str],
        is_clf: bool,
        is_fast: bool,
        cv,
        use_balanced: bool = False,
    ) -> dict[str, object] | None:
        models = self._build_models(is_clf, is_fast, len(X_df), use_balanced=use_balanced)
        results: list[dict] = []
        fitted_models: list[tuple[str, object]] = []

        for name, model in models:
            scoring = self._scoring(is_clf, len(target_labels))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = cross_validate(model, X_df.to_numpy(dtype=float), y_encoded, cv=cv, scoring=scoring)
                row = {
                    "model": name,
                    "metric": "F1 (macro)" if is_clf else "R2",
                    "mean": round(float(np.mean(scores["test_primary"])), 4),
                    "std": round(float(np.std(scores["test_primary"])), 4),
                }
                if is_clf:
                    row["precision_macro"] = round(float(np.mean(scores["test_precision_macro"])), 4)
                    row["recall_macro"] = round(float(np.mean(scores["test_recall_macro"])), 4)
                    row["accuracy"] = round(float(np.mean(scores["test_accuracy"])), 4)
                    if "test_roc_auc" in scores:
                        row["roc_auc"] = round(float(np.mean(scores["test_roc_auc"])), 4)
                results.append(row)
                fitted_models.append((name, model))
            except Exception:
                continue

        if not results:
            return None

        best = max(results, key=lambda item: item["mean"])
        baseline_row = next((r for r in results if r["model"] == "Baseline"), None)
        best_non_baseline = max((r for r in results if r["model"] != "Baseline"), key=lambda item: item["mean"], default=best)
        if best["model"] == "Baseline" and best_non_baseline["model"] != "Baseline":
            best = best_non_baseline  # Don't pick Baseline as "best" if real models exist
        best["baseline_score"] = baseline_row["mean"] if baseline_row else None
        best["lift_over_baseline"] = round(best["mean"] - baseline_row["mean"], 4) if baseline_row else None
        best_estimator = next(model for name, model in fitted_models if name == best["model"])
        best_metrics, confusion, labels = self._importance._best_model_diagnostics(
            best_estimator,
            X_df.to_numpy(dtype=float),
            y_encoded,
            target_labels,
            is_clf,
            cv,
        )
        feature_importance = self._importance._rf_importance(
            X_df.to_numpy(dtype=float),
            y_encoded,
            is_clf,
            X_df.columns.tolist(),
        )
        perm_importance = self._importance._permutation_importance(
            best_estimator,
            X_df.to_numpy(dtype=float),
            y_encoded,
            is_clf,
            X_df.columns.tolist(),
        )
        shap_importance = self._importance._shap_importance(
            best_estimator,
            X_df.to_numpy(dtype=float),
            y_encoded,
            is_clf,
            X_df.columns.tolist(),
        )
        shap_waterfall = self._importance._shap_waterfall(
            best_estimator,
            X_df.to_numpy(dtype=float),
            y_encoded,
            is_clf,
            X_df.columns.tolist(),
        )
        # Refit best estimator on the full available data for model export
        X_arr = X_df.to_numpy(dtype=float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                export_estimator = clone(best_estimator).fit(X_arr, y_encoded)
        except Exception:
            export_estimator = None

        return {
            "results": results,
            "best": best,
            "best_metrics": best_metrics,
            "confusion_matrix": confusion,
            "labels": labels,
            "feature_importance": feature_importance,
            "permutation_importance": perm_importance,
            "shap_importance": shap_importance,
            "shap_waterfall": shap_waterfall,
            "export_estimator": export_estimator,
            "export_feature_names": X_df.columns.tolist(),
            "X_df": X_df,
            "y_encoded": y_encoded,
        }

    def _encode_target(self, y_series: pd.Series, is_clf: bool) -> tuple[np.ndarray, list[str]]:
        if is_clf:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(y_series.astype(str))
            return encoded, [str(v) for v in encoder.classes_]
        return y_series.to_numpy(dtype=float), []

    def _build_cv(self, y: np.ndarray, is_clf: bool, n_folds: int):
        if is_clf:
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        return KFold(n_splits=n_folds, shuffle=True, random_state=42)

    def _scoring(self, is_clf: bool, n_classes: int):
        if not is_clf:
            return {"primary": "r2"}
        scoring = {
            "primary": "f1_macro",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "accuracy": "accuracy",
        }
        if n_classes == 2:
            scoring["roc_auc"] = "roc_auc"
        return scoring

    def _sample_indices(
        self,
        y_series: pd.Series,
        is_classification: bool,
        max_rows: int,
    ) -> pd.Index | None:
        if len(y_series) <= max_rows:
            return None

        if not is_classification:
            return y_series.sample(n=max_rows, random_state=42).index

        grouped = y_series.groupby(y_series.astype(str), group_keys=False)
        total = len(y_series)
        sample_parts: list[pd.Index] = []
        for _, group in grouped:
            share = len(group) / total
            target_n = min(len(group), max(1, int(round(max_rows * share))))
            sample_parts.append(group.sample(n=target_n, random_state=42).index)

        sampled = pd.Index(np.concatenate([idx.to_numpy() for idx in sample_parts])) if sample_parts else pd.Index([])
        sampled = sampled.drop_duplicates()
        if len(sampled) > max_rows:
            sampled = sampled.to_series().sample(n=max_rows, random_state=42).index
        elif len(sampled) < max_rows:
            remaining = y_series.index.difference(sampled)
            if len(remaining):
                top_up = min(max_rows - len(sampled), len(remaining))
                extra = remaining.to_series().sample(n=top_up, random_state=42).index
                sampled = pd.Index(np.concatenate([sampled.to_numpy(), extra.to_numpy()]))
        return sampled

    def _detect_targets(self, raw_df: pd.DataFrame, profiling) -> list[dict]:
        candidates = []
        cat_cols = set()
        numeric_cols = set()
        role_by_column: dict[str, str] = {}
        if profiling and profiling.success:
            cat_cols = set(profiling.outputs.get("categorical_column_names", []))
            numeric_cols = set(profiling.outputs.get("numeric_column_names", []))
            role_by_column = self._trust._role_by_column(profiling)

        for col in raw_df.columns:
            series = raw_df[col].dropna()
            if series.empty:
                continue
            nunique = series.nunique()
            name_lower = str(col).lower().strip("_")
            name_tokens = set(name_lower.replace("-", "_").split("_"))
            role_guess = role_by_column.get(str(col), "unknown")
            if role_guess in {"id_like", "time_like"}:
                continue

            # Classification candidates
            is_keyword = any(kw in name_lower for kw in _TARGET_KEYWORDS)
            is_low_card_cat = col in cat_cols and 2 <= nunique <= 20
            is_binary_numeric = nunique == 2
            if is_keyword or is_low_card_cat or is_binary_numeric:
                task = "classification" if nunique <= 20 else "regression"
                candidates.append({"name": str(col), "task_type": task, "n_classes": int(nunique)})
                continue

            # Regression candidates: continuous numeric columns whose name
            # suggests an outcome variable
            if col in numeric_cols and nunique > 20:
                is_regression_keyword = bool(name_tokens & _REGRESSION_KEYWORDS)
                if is_regression_keyword:
                    candidates.append({"name": str(col), "task_type": "regression", "n_classes": int(nunique)})

        return candidates[:5]

    def _build_models(self, is_clf: bool, is_fast: bool, n_rows: int, use_balanced: bool = False):
        cw = "balanced" if (use_balanced and is_clf) else None
        models = []
        if is_clf:
            models.append(("Baseline", DummyClassifier(strategy="most_frequent", random_state=42)))
            models.append(("LogisticRegression", SklearnPipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=42, class_weight=cw)),
            ])))
            models.append(("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1, class_weight=cw)))
            if not (is_fast and n_rows > 5000):
                models.append(("SVM_RBF", SklearnPipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", random_state=42, probability=True, class_weight=cw)),
                ])))
        else:
            models.append(("Baseline", DummyRegressor(strategy="mean")))
            models.append(("Ridge", SklearnPipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ])))
            models.append(("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)))
            if not (is_fast and n_rows > 5000):
                models.append(("SVM_RBF", SklearnPipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVR(kernel="rbf")),
                ])))

        xgb_clf, xgb_reg = _try_import_xgboost()
        if xgb_clf:
            if is_clf:
                models.append(("XGBoost", xgb_clf(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False)))
            else:
                models.append(("XGBoost", xgb_reg(n_estimators=100, random_state=42, verbosity=0)))

        lgbm_clf, lgbm_reg = _try_import_lightgbm()
        if lgbm_clf:
            if is_clf:
                lgbm_kw = {"class_weight": "balanced"} if use_balanced else {}
                models.append(("LightGBM", lgbm_clf(n_estimators=100, random_state=42, verbose=-1, **lgbm_kw)))
            else:
                models.append(("LightGBM", lgbm_reg(n_estimators=100, random_state=42, verbose=-1)))
        return models

    def _compare_tracks(self, track_outputs: dict[str, dict[str, object]]) -> dict[str, object]:
        has_a = "track_a" in track_outputs
        has_b = "track_b" in track_outputs
        if not has_a:
            return {"available": False, "summary": "Track A results are unavailable."}
        if not has_b:
            return {"available": False, "summary": "DVAE latent track was unavailable; only Track A ran."}

        a_best = track_outputs["track_a"]["best"]
        b_best = track_outputs["track_b"]["best"]
        winner = "track_b" if b_best["mean"] > a_best["mean"] else "track_a"
        delta = round(float(b_best["mean"] - a_best["mean"]), 4)
        return {
            "available": True,
            "winner": winner,
            "delta": delta,
            "track_a_best": a_best,
            "track_b_best": b_best,
            "summary": (
                f"Track B {'outperformed' if delta > 0 else 'did not outperform'} Track A by {delta:+.4f} "
                f"on the best model score."
            ),
        }

    def _empty_result(self, reason: str) -> StageResult:
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={
                "candidate_targets": [],
                "model_results_track_a": {},
                "model_results_track_b": {},
                "feature_importances": {},
                "best_models": {},
                "track_comparison": {},
                "evaluation_details": {},
                "sampling_notes": [],
                "sampling_info": {},
            },
            interpretations={
                "model_comparison": reason,
                "feature_importances": reason,
                "model_recommendation": reason,
                "track_comparison": reason,
                "trust_notes": reason,
            },
        )
