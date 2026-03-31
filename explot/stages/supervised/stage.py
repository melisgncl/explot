from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR

from explot.stages.base import BaseStage, StageMeta, StageResult

_TARGET_KEYWORDS = {"target", "label", "class", "outcome", "group", "type", "diagnosis", "status"}


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
        optional_deps=("profiling", "unsupervised", "autoencoder"),
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

        if track_a_df is None or track_a_df.empty:
            return self._empty_result("No features available for supervised probes.")

        hooks.progress(self.meta.name, 10, "Detecting candidate target columns.")
        candidates = self._detect_targets(state.raw_df, profiling)
        if not candidates:
            return self._empty_result(
                "No candidate target columns detected. Explot looks for categorical "
                "columns with 2-20 unique values or columns named like 'target', 'label', "
                "'class', 'outcome', etc."
            )

        is_fast = getattr(config, "budget", None) and getattr(config.budget, "mode", "") == "fast"
        n_folds = 3 if is_fast else 5

        model_results_track_a: dict[str, list[dict]] = {}
        model_results_track_b: dict[str, list[dict]] = {}
        feature_importances: dict[str, dict[str, list[dict]]] = {}
        best_models: dict[str, dict] = {}
        track_comparison: dict[str, dict[str, object]] = {}
        evaluation_details: dict[str, dict[str, object]] = {}
        sampling_notes: list[str] = []

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
            sampled_idx = self._sample_indices(y_series, is_clf, is_fast)
            if sampled_idx is not None:
                note = (
                    f"Target '{target_name}' was scored on a deterministic sample of "
                    f"{len(sampled_idx)} rows from {len(valid_idx)} available rows."
                )
                sampling_notes.append(note)
                if hasattr(hooks, "log"):
                    hooks.log(self.meta.name, note)
                valid_idx = sampled_idx
                y_series = y_series.loc[valid_idx]

            y_encoded, target_labels = self._encode_target(y_series, is_clf)
            cv = self._build_cv(y_encoded, is_clf, n_folds)

            track_outputs: dict[str, dict[str, object]] = {}
            for track_name, feature_df in (("track_a", track_a_df), ("track_b", track_b_df)):
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

            diagnostics = self._target_diagnostics(state.raw_df, target_name, valid_idx, track_outputs, is_clf)
            best_track_name, best_track = max(
                track_outputs.items(),
                key=lambda item: item[1]["best"]["mean"],
            )

            best_models[target_name] = {
                **best_track["best"],
                "track": best_track_name,
                "trust_flags": diagnostics["trust_flags"],
                "diagnostics": diagnostics,
            }
            feature_importances[target_name] = {
                track_name: data["feature_importance"] for track_name, data in track_outputs.items()
            }
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
            "feature_importances": feature_importances,
            "best_models": best_models,
            "track_comparison": track_comparison,
            "evaluation_details": evaluation_details,
            "sampling_notes": sampling_notes,
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
    ) -> dict[str, object] | None:
        models = self._build_models(is_clf, is_fast, len(X_df))
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
        return {
            "results": results,
            "best": best,
            "best_metrics": best_metrics,
            "confusion_matrix": confusion,
            "labels": labels,
            "feature_importance": feature_importance,
        }

    def _best_model_diagnostics(self, estimator, X, y, target_labels, is_clf: bool, cv):
        if not is_clf:
            return {"r2": None}, [], []

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
        is_fast: bool,
    ) -> pd.Index | None:
        max_rows = 3000 if is_fast else 8000
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
                sampled = sampled.append(remaining.to_series().sample(n=top_up, random_state=42).index)
        return sampled

    def _detect_targets(self, raw_df: pd.DataFrame, profiling) -> list[dict]:
        candidates = []
        cat_cols = set()
        role_by_column: dict[str, str] = {}
        if profiling and profiling.success:
            cat_cols = set(profiling.outputs.get("categorical_column_names", []))
            role_by_column = {
                str(name): str(profile.get("role_guess", "unknown"))
                for name, profile in profiling.outputs.get("column_profiles", {}).items()
            }

        for col in raw_df.columns:
            series = raw_df[col].dropna()
            if series.empty:
                continue
            nunique = series.nunique()
            name_lower = str(col).lower().strip("_")
            role_guess = role_by_column.get(str(col), "unknown")
            if role_guess in {"id_like", "time_like"}:
                continue

            is_keyword = any(kw in name_lower for kw in _TARGET_KEYWORDS)
            is_low_card_cat = col in cat_cols and 2 <= nunique <= 20
            is_binary_numeric = nunique == 2
            if is_keyword or is_low_card_cat or is_binary_numeric:
                task = "classification" if nunique <= 20 else "regression"
                candidates.append({"name": str(col), "task_type": task, "n_classes": int(nunique)})
        return candidates[:5]

    def _build_models(self, is_clf: bool, is_fast: bool, n_rows: int):
        models = []
        if is_clf:
            models.append(("LogisticRegression", LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=42)))
            models.append(("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)))
            if not (is_fast and n_rows > 5000):
                models.append(("SVM_RBF", SVC(kernel="rbf", random_state=42, probability=True)))
        else:
            models.append(("Ridge", Ridge(alpha=1.0)))
            models.append(("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)))
            if not (is_fast and n_rows > 5000):
                models.append(("SVM_RBF", SVR(kernel="rbf")))

        xgb_clf, xgb_reg = _try_import_xgboost()
        if xgb_clf:
            if is_clf:
                models.append(("XGBoost", xgb_clf(n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False)))
            else:
                models.append(("XGBoost", xgb_reg(n_estimators=100, random_state=42, verbosity=0)))

        lgbm_clf, lgbm_reg = _try_import_lightgbm()
        if lgbm_clf:
            if is_clf:
                models.append(("LightGBM", lgbm_clf(n_estimators=100, random_state=42, verbose=-1)))
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
            if aligned["feature"].nunique() <= 2000:
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
        if is_classification and best_score >= 0.95:
            trust_flags.append("near_perfect_score")
        if best_score >= 0.9 and (exact_copy_columns or deterministic_proxy_columns or high_corr_proxy_columns):
            trust_flags.append("possible_leakage")
        if "track_b" in track_outputs and track_outputs["track_b"]["best"]["mean"] > track_outputs.get("track_a", {"best": {"mean": -np.inf}})["best"]["mean"]:
            trust_flags.append("latent_representation_helped")

        return {
            "trust_flags": sorted(set(trust_flags)),
            "single_feature_leakage_columns": sorted(set(single_feature_leakage_columns))[:3],
            "proxy_columns": sorted(set(exact_copy_columns + deterministic_proxy_columns + high_corr_proxy_columns))[:5],
            "exact_copy_columns": sorted(set(exact_copy_columns))[:5],
            "deterministic_proxy_columns": sorted(set(deterministic_proxy_columns))[:5],
            "high_corr_proxy_columns": sorted(set(high_corr_proxy_columns))[:5],
            "suspicious_name_columns": sorted(set(suspicious_name_columns))[:5],
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

    def _rf_importance(self, X, y, is_clf: bool, col_names: list[str]) -> list[dict]:
        try:
            rf = (RandomForestClassifier if is_clf else RandomForestRegressor)(n_estimators=50, random_state=42, n_jobs=1)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            ranked = [
                {"feature": col_names[i] if i < len(col_names) else f"feature_{i}", "importance": round(float(importances[i]), 4)}
                for i in indices if importances[i] > 0
            ]
            if ranked:
                return ranked
            return [
                {"feature": col_names[i] if i < len(col_names) else f"feature_{i}", "importance": round(float(importances[i]), 4)}
                for i in indices
            ]
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
            },
            interpretations={
                "model_comparison": reason,
                "feature_importances": reason,
                "model_recommendation": reason,
                "track_comparison": reason,
                "trust_notes": reason,
            },
        )

    def _comparison_interp(self, best_models) -> str:
        if not best_models:
            return "No models were successfully trained."
        parts = []
        for target, info in best_models.items():
            parts.append(
                f"Target '{target}': best overall result came from {info.get('track', 'track_a')} using "
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
            return "No Track A vs Track B comparison is available."
        return " ".join(str(item.get("summary", "")) for item in track_comparison.values())

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
                parts.append(f"Target '{target}' has near-deterministic proxy feature(s) ({', '.join(deterministic[:3])}), so high scores should be treated cautiously.")
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
            parts.append(f"For target '{target}', {model} on {info.get('track', 'track_a')} achieved the best {metric} of {score:.2f}.")
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
            if score > 0.8:
                parts.append("This is a strong result - the target is well-predictable.")
            elif score > 0.5:
                parts.append("Moderate predictability - further feature engineering may help.")
            else:
                parts.append("Weak predictability - the target may not be well-explained by these features, or the sample size may be too small.")
            if any(flag in flags for flag in ("proxy_like_feature", "near_perfect_score", "possible_leakage", "exact_copy_feature", "high_correlation_proxy")):
                parts.append("Trust note: this target shows proxy/leakage-like patterns under the current heuristics.")
        return " ".join(parts)
