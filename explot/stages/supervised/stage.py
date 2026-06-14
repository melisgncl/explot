from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from explot.stages.base import BaseStage, StageMeta, StageResult

_TARGET_KEYWORDS = {"target", "label", "class", "outcome", "group", "type", "diagnosis", "status"}
_REGRESSION_KEYWORDS = {
    "score", "value", "price", "cost", "amount", "survival", "duration",
    "response", "expression", "concentration", "ic50", "ec50", "dose",
    "yield", "output", "result", "measure", "index", "ratio",
}
_TREE_TYPES = frozenset((
    "RandomForestClassifier", "RandomForestRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
))
_LINEAR_TYPES = frozenset(("LogisticRegression", "Ridge"))


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

            diagnostics = self._target_diagnostics(state.raw_df, target_name, valid_idx, track_outputs, is_clf, profiling)
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
            group_audit = self._group_leakage_audit(
                state.raw_df, target_name, valid_idx, best_track, audit_estimator, is_clf, profiling,
                leakage_delta_threshold, leakage_score_floor,
            )
            temporal_audit = self._temporal_leakage_audit(
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
            "model_comparison": self._comparison_interp(best_models),
            "feature_importances": self._fi_interp(feature_importances),
            "model_recommendation": self._recommendation_interp(best_models, intrinsic_dim, sil),
            "track_comparison": self._track_comparison_interp(track_comparison),
            "trust_notes": self._trust_notes(best_models),
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
        best_metrics, confusion, labels = self._best_model_diagnostics(
            best_estimator,
            X_df.to_numpy(dtype=float),
            y_encoded,
            target_labels,
            is_clf,
            cv,
        )
        feature_importance = self._rf_importance(
            X_df.to_numpy(dtype=float),
            y_encoded,
            is_clf,
            X_df.columns.tolist(),
        )
        perm_importance = self._permutation_importance(
            best_estimator,
            X_df.to_numpy(dtype=float),
            y_encoded,
            is_clf,
            X_df.columns.tolist(),
        )
        shap_importance = self._shap_importance(
            best_estimator,
            X_df.to_numpy(dtype=float),
            y_encoded,
            is_clf,
            X_df.columns.tolist(),
        )
        shap_waterfall = self._shap_waterfall(
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

    def _best_model_diagnostics(self, estimator, X, y, target_labels, is_clf: bool, cv):
        if not is_clf:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = cross_val_predict(clone(estimator), X, y, cv=cv)
            residuals = y - preds
            metrics = {
                "r2": round(float(1.0 - np.sum(residuals ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)), 4),
                "mae": round(float(np.mean(np.abs(residuals))), 4),
                "rmse": round(float(np.sqrt(np.mean(residuals ** 2))), 4),
            }
            return metrics, [], []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict")

        metrics = {
            "precision_macro": round(float(precision_score(y, preds, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y, preds, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, preds)), 4),
        }

        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            try:
                if hasattr(estimator, "predict_proba"):
                    scores = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba")[:, 1]
                elif hasattr(estimator, "decision_function"):
                    scores = cross_val_predict(clone(estimator), X, y, cv=cv, method="decision_function")
                else:
                    scores = None
                if scores is not None:
                    metrics["roc_auc"] = round(float(roc_auc_score(y, scores)), 4)
            except Exception:
                pass

        matrix = confusion_matrix(y, preds, labels=unique_labels)
        matrix_list = matrix.astype(int).tolist()
        labels = [str(target_labels[idx]) for idx in unique_labels]
        return metrics, matrix_list, labels

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
            role_by_column = self._role_by_column(profiling)

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

    def _permutation_importance(self, estimator, X, y, is_clf: bool, col_names: list[str]) -> list[dict]:
        """Model-agnostic permutation importance on the best model."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = clone(estimator).fit(X, y)
                scoring = "f1_macro" if is_clf else "r2"
                result = permutation_importance(
                    fitted, X, y, n_repeats=5, random_state=42,
                    scoring=scoring, n_jobs=1,
                )
            importances = result.importances_mean
            indices = np.argsort(importances)[::-1][:10]
            return [
                {
                    "feature": col_names[i] if i < len(col_names) else f"feature_{i}",
                    "importance": round(float(importances[i]), 4),
                    "std": round(float(result.importances_std[i]), 4),
                }
                for i in indices if importances[i] > 0
            ]
        except Exception:
            return []

    @staticmethod
    def _unwrap_pipeline(fitted, X: np.ndarray) -> tuple:
        """Return (inner_estimator, X_transformed) by stripping any sklearn Pipeline wrapper."""
        if hasattr(fitted, "named_steps"):
            steps = list(fitted.named_steps.keys())
            return fitted.named_steps[steps[-1]], fitted[:-1].transform(X)
        return fitted, X

    def _shap_importance(self, estimator, X, y, is_clf: bool, col_names: list[str]) -> list[dict]:
        """SHAP-based feature importance. Returns top-10 by mean |SHAP|."""
        try:
            import shap  # optional dependency
        except ImportError:
            return []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = clone(estimator).fit(X, y)
            inner, X_transformed = self._unwrap_pipeline(fitted, X)
            type_name = type(inner).__name__

            sample_size = min(500, len(X_transformed))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_transformed), size=sample_size, replace=False)
            X_sample = X_transformed[idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if type_name in _TREE_TYPES:
                    explainer = shap.TreeExplainer(inner)
                    shap_values = explainer.shap_values(X_sample)
                elif type_name in _LINEAR_TYPES:
                    background = shap.maskers.Independent(X_transformed, max_samples=min(50, len(X_transformed)))
                    explainer = shap.LinearExplainer(inner, background)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    return []

            # Normalise shap_values to (n_samples, n_features):
            # - old shap: list of (n_samples, n_features) arrays (one per class)
            # - new shap: (n_samples, n_features, n_classes) ndarray for trees
            # - regression / binary new shap: (n_samples, n_features)
            sv = np.array(shap_values) if not isinstance(shap_values, np.ndarray) else shap_values
            if sv.ndim == 3:
                arr = np.abs(sv).mean(axis=2)   # (n_samples, n_features)
            elif sv.ndim == 1 and len(sv) > 0 and isinstance(sv[0], np.ndarray):
                arr = np.mean([np.abs(s) for s in sv], axis=0)
            else:
                arr = np.abs(sv)

            mean_abs = arr.mean(axis=0)
            indices = np.argsort(mean_abs)[::-1][:10]
            return [
                {
                    "feature": col_names[i] if i < len(col_names) else f"feature_{i}",
                    "importance": round(float(mean_abs[i]), 4),
                }
                for i in indices if mean_abs[i] > 0
            ]
        except Exception:
            return []

    def _shap_waterfall(self, estimator, X, y, is_clf: bool, col_names: list[str]) -> dict:
        """Return signed SHAP values for one representative sample (highest-confidence prediction).

        Returns dict with keys: base_value, prediction, features (list of {feature, value, shap}).
        Returns {} on any failure.
        """
        try:
            import shap  # optional dependency
        except ImportError:
            return {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = clone(estimator).fit(X, y)
            inner, X_transformed = self._unwrap_pipeline(fitted, X)
            type_name = type(inner).__name__

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if type_name in _TREE_TYPES:
                    explainer = shap.TreeExplainer(inner)
                elif type_name in _LINEAR_TYPES:
                    background = shap.maskers.Independent(X_transformed, max_samples=min(50, len(X_transformed)))
                    explainer = shap.LinearExplainer(inner, background)
                else:
                    return {}

            # Pick the sample with the most extreme prediction (highest confidence)
            sample_size = min(200, len(X_transformed))
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_transformed), size=sample_size, replace=False)
            X_sample = X_transformed[idx]

            # Get predictions to find highest-confidence sample
            try:
                if is_clf and hasattr(inner, "predict_proba"):
                    proba = inner.predict_proba(X_sample)
                    confidence = np.max(proba, axis=1)
                    best_idx = int(np.argmax(confidence))
                else:
                    best_idx = 0
            except Exception:
                best_idx = 0

            X_one = X_sample[best_idx : best_idx + 1]
            shap_values = explainer.shap_values(X_one)

            # Normalise to 1D signed array for the chosen sample
            sv = np.array(shap_values) if not isinstance(shap_values, np.ndarray) else shap_values
            if sv.ndim == 3:
                # (1, n_features, n_classes) — use class 1 for binary, mean for multi
                signed = sv[0, :, 1] if sv.shape[2] == 2 else sv[0].mean(axis=1)
            elif sv.ndim == 1 and len(sv) > 0 and isinstance(sv[0], np.ndarray):
                # old shap list format — use class 1
                signed = sv[1][0] if len(sv) > 1 else sv[0][0]
            elif sv.ndim == 2:
                signed = sv[0]
            else:
                return {}

            signed = np.array(signed, dtype=float)
            if signed.ndim != 1 or len(signed) != len(col_names):
                return {}

            # Base value
            try:
                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
                else:
                    base_val = float(base_val)
            except Exception:
                base_val = 0.0

            # Prediction for this sample
            try:
                if is_clf and hasattr(inner, "predict_proba"):
                    pred = float(inner.predict_proba(X_one)[0, -1])
                else:
                    pred = float(inner.predict(X_one)[0])
            except Exception:
                pred = float(base_val + signed.sum())

            # Top 10 by abs value
            top_idx = np.argsort(np.abs(signed))[::-1][:10]
            raw_x = X_one[0]
            features = [
                {
                    "feature": col_names[i] if i < len(col_names) else f"feature_{i}",
                    "value": round(float(raw_x[i]), 4) if i < len(raw_x) else 0.0,
                    "shap": round(float(signed[i]), 4),
                }
                for i in top_idx
            ]
            return {
                "base_value": round(base_val, 4),
                "prediction": round(pred, 4),
                "features": features,
            }
        except Exception:
            return {}

    def _rf_importance(self, X, y, is_clf: bool, col_names: list[str]) -> list[dict]:
        try:
            rf = (RandomForestClassifier if is_clf else RandomForestRegressor)(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            all_ranked = [
                {"feature": col_names[i] if i < len(col_names) else f"feature_{i}", "importance": round(float(importances[i]), 4)}
                for i in indices
            ]
            return [r for r in all_ranked if r["importance"] > 0] or all_ranked
        except Exception:
            return []

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

    _TRACK_LABELS = {"track_a": "PCA features", "track_b": "DVAE latent features"}

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
