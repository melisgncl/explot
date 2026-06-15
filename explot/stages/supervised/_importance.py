from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, cross_validate

_TREE_TYPES = frozenset((
    "RandomForestClassifier", "RandomForestRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
))
_LINEAR_TYPES = frozenset(("LogisticRegression", "Ridge"))


class _FeatureImportance:
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
