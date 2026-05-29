"""Tests for PreprocessingStage — Phase 1a/1b/1c."""
from __future__ import annotations

import numpy as np
import pandas as pd

from explot.stages.findings.stage import FindingsStage
from explot.stages.preprocessing.stage import PreprocessingStage
from explot.stages.profiling.stage import ProfilingStage
from explot.state import PipelineState

# ---------------------------------------------------------------------------
# Minimal pipeline helpers
# ---------------------------------------------------------------------------

class _Hooks:
    def progress(self, *a, **k): ...
    def stage_started(self, *a): ...
    def stage_finished(self, *a): return 0.0
    def stage_failed(self, *a): ...
    def log(self, *a, **k): ...


class _Config:
    class budget:
        mode = "fast"


def _profile(df: pd.DataFrame):
    state = PipelineState(raw_df=df)
    state.results["profiling"] = ProfilingStage().run(state, _Config(), _Hooks())
    return state


def _preprocess(df: pd.DataFrame):
    state = _profile(df)
    state.results["preprocessing"] = PreprocessingStage().run(state, _Config(), _Hooks())
    return state


# ---------------------------------------------------------------------------
# 1a: Categorical encoding
# ---------------------------------------------------------------------------

def test_low_cardinality_categorical_is_ordinal_encoded():
    df = pd.DataFrame({
        "sex": ["M", "F", "M", "F", "M"] * 20,
        "age": np.random.default_rng(0).integers(20, 80, 100).astype(float),
        "target": [0, 1] * 50,
    })
    state = _preprocess(df)
    result = state.results["preprocessing"]
    assert result.success
    prep = result.outputs["preprocessed_df"]
    assert "sex" in prep.columns
    assert "sex" in result.outputs["ordinal_encoded"]
    # OrdinalEncoder maps to floats 0..n-1
    assert set(prep["sex"].unique()).issubset({0.0, 1.0})


def test_high_cardinality_column_is_excluded():
    """A high-cardinality string column must not appear in the preprocessed matrix,
    whether excluded via id_like role detection or via the cardinality threshold."""
    rng = np.random.default_rng(1)
    # Use a name that won't be flagged id_like by profiling but has >100 unique values
    categories = [f"city_{i}" for i in range(150)]
    df = pd.DataFrame({
        "city": rng.choice(categories, 200),   # 150 unique → high_cardinality drop
        "f1": rng.normal(size=200),
        "target": rng.integers(0, 2, 200),
    })
    state = _preprocess(df)
    result = state.results["preprocessing"]
    assert result.success
    prep = result.outputs["preprocessed_df"]
    assert "city" not in prep.columns
    drops = result.outputs["dropped_columns"]
    assert any(d["name"] == "city" and d["reason"] == "high_cardinality" for d in drops)


def test_mid_cardinality_column_is_frequency_encoded():
    rng = np.random.default_rng(2)
    n = 300
    categories = [f"cat_{i}" for i in range(50)]   # 50 unique → frequency encoded
    df = pd.DataFrame({
        "region": rng.choice(categories, n),
        "f1": rng.normal(size=n),
        "target": rng.integers(0, 2, n),
    })
    state = _preprocess(df)
    result = state.results["preprocessing"]
    assert result.success
    prep = result.outputs["preprocessed_df"]
    assert "region" in prep.columns
    assert "region" in result.outputs["frequency_encoded"]
    # Frequency encoded values should be in (0, 1]
    assert prep["region"].between(0.0, 1.0).all()


def test_id_like_column_is_skipped():
    """id_like columns must not appear in the preprocessed matrix."""
    rng = np.random.default_rng(3)
    n = 100
    df = pd.DataFrame({
        "sample_id": [f"S{i:04d}" for i in range(n)],  # profiling marks as id_like
        "f1": rng.normal(size=n),
        "target": rng.integers(0, 2, n),
    })
    state = _preprocess(df)
    result = state.results["preprocessing"]
    prep = result.outputs["preprocessed_df"]
    assert "sample_id" not in prep.columns


# ---------------------------------------------------------------------------
# 1a: Imputation
# ---------------------------------------------------------------------------

def test_numeric_nan_imputed_with_median():
    rng = np.random.default_rng(4)
    n = 100
    ages = rng.integers(20, 80, n).astype(float)
    ages[::5] = np.nan  # 20% missing
    df = pd.DataFrame({"age": ages, "f1": rng.normal(size=n), "target": rng.integers(0, 2, n)})
    state = _preprocess(df)
    result = state.results["preprocessing"]
    prep = result.outputs["preprocessed_df"]
    assert prep["age"].isna().sum() == 0
    stats = result.outputs["imputation_stats"]
    assert "age" in stats
    assert stats["age"]["method"] == "median"


def test_heavy_imputation_flagged():
    rng = np.random.default_rng(5)
    n = 100
    col = rng.normal(size=n).astype(object)
    col[:25] = np.nan   # 25% missing — above the 20% threshold
    df = pd.DataFrame({"sparse_col": col, "f1": rng.normal(size=n), "target": rng.integers(0, 2, n)})
    state = _preprocess(df)
    result = state.results["preprocessing"]
    assert "sparse_col" in result.outputs["columns_with_heavy_imputation"]


def test_categorical_nan_imputed_with_mode():
    rng = np.random.default_rng(6)
    n = 100
    cats = np.array(["A", "B", "A", "A", "B"] * 20, dtype=object)
    cats[::10] = None
    df = pd.DataFrame({
        "status": cats,
        "f1": rng.normal(size=n),
        "target": rng.integers(0, 2, n),
    })
    state = _preprocess(df)
    result = state.results["preprocessing"]
    prep = result.outputs["preprocessed_df"]
    assert prep["status"].isna().sum() == 0
    stats = result.outputs["imputation_stats"]
    assert stats.get("status", {}).get("method") == "most_frequent"


# ---------------------------------------------------------------------------
# 1a: Scaling — LR and SVM wrapped in Pipeline
# ---------------------------------------------------------------------------

def test_linear_models_wrapped_in_pipeline():
    from sklearn.pipeline import Pipeline as SklearnPipeline

    from explot.stages.supervised.stage import SupervisedStage
    stage = SupervisedStage()
    models = dict(stage._build_models(is_clf=True, is_fast=True, n_rows=100))
    assert isinstance(models["LogisticRegression"], SklearnPipeline)
    assert isinstance(models["SVM_RBF"], SklearnPipeline)
    assert not isinstance(models["RandomForest"], SklearnPipeline)
    assert not isinstance(models["Baseline"], SklearnPipeline)


def test_ridge_regression_wrapped_in_pipeline():
    from sklearn.pipeline import Pipeline as SklearnPipeline

    from explot.stages.supervised.stage import SupervisedStage
    stage = SupervisedStage()
    models = dict(stage._build_models(is_clf=False, is_fast=True, n_rows=100))
    assert isinstance(models["Ridge"], SklearnPipeline)
    assert isinstance(models["SVM_RBF"], SklearnPipeline)
    assert not isinstance(models["RandomForest"], SklearnPipeline)


# ---------------------------------------------------------------------------
# 1b: Preprocessing findings surface in FindingsStage
# ---------------------------------------------------------------------------

def test_heavy_imputation_produces_high_finding():
    from explot.stages.dimensionality.stage import DimensionalityStage
    from explot.stages.exploration.stage import ExplorationStage
    rng = np.random.default_rng(7)
    n = 100
    sparse = rng.normal(size=n).astype(object)
    sparse[:30] = np.nan   # 30% missing
    df = pd.DataFrame({
        "sparse_col": sparse,
        "f1": rng.normal(size=n),
        "target": rng.integers(0, 2, n),
    })
    state = _preprocess(df)
    state.results["exploration"] = ExplorationStage().run(state, _Config(), _Hooks())
    state.results["dimensionality"] = DimensionalityStage().run(state, _Config(), _Hooks())
    findings_result = FindingsStage().run(state, _Config(), _Hooks())
    findings = findings_result.outputs["findings_list"]
    heavy_findings = [f for f in findings if f["rule"] == "heavy_imputation"]
    assert len(heavy_findings) >= 1
    assert heavy_findings[0]["confidence"] == "HIGH"


def test_high_cardinality_drop_produces_high_finding():
    from explot.stages.dimensionality.stage import DimensionalityStage
    from explot.stages.exploration.stage import ExplorationStage
    rng = np.random.default_rng(8)
    n = 200
    # 'city' is not an id-like name, so profiling won't skip it before the cardinality check
    categories = [f"city_{i}" for i in range(150)]
    df = pd.DataFrame({
        "city": rng.choice(categories, n),
        "f1": rng.normal(size=n),
        "target": rng.integers(0, 2, n),
    })
    state = _preprocess(df)
    state.results["exploration"] = ExplorationStage().run(state, _Config(), _Hooks())
    state.results["dimensionality"] = DimensionalityStage().run(state, _Config(), _Hooks())
    findings_result = FindingsStage().run(state, _Config(), _Hooks())
    findings = findings_result.outputs["findings_list"]
    hc_findings = [f for f in findings if f["rule"] == "high_cardinality_dropped"]
    assert len(hc_findings) >= 1


# ---------------------------------------------------------------------------
# 1c: Verdict performance floor
# ---------------------------------------------------------------------------

def test_verdict_investigates_near_baseline_model():
    """A model that barely beats baseline should trigger INVESTIGATE."""
    from explot.stages.base import StageMeta, StageResult
    # Construct a fake supervised result where score ≈ baseline
    fake_best_models = {
        "target": {
            "model": "RandomForest",
            "mean": 0.52,
            "baseline_score": 0.50,
            "lift_over_baseline": 0.02,
            "trust_flags": [],
            "diagnostics": {},
        }
    }
    state = PipelineState(raw_df=pd.DataFrame({"x": [1], "target": [0]}))
    state.results["profiling"] = StageResult(
        stage_name="profiling", meta=StageMeta(name="profiling"),
        outputs={"quality_score": 80},
    )
    state.results["supervised"] = StageResult(
        stage_name="supervised", meta=StageMeta(name="supervised"),
        outputs={"best_models": fake_best_models, "sampling_notes": []},
    )
    verdict = FindingsStage()._verdict(state)
    assert verdict["decision"] == "INVESTIGATE"
    assert any("within 0.03" in r for r in verdict["reasons"])


def test_verdict_investigates_when_data_was_sampled():
    """Sampling note should escalate a clean SHIP to INVESTIGATE."""
    from explot.stages.base import StageMeta, StageResult
    fake_best_models = {
        "target": {
            "model": "RandomForest",
            "mean": 0.85,
            "baseline_score": 0.50,
            "lift_over_baseline": 0.35,
            "trust_flags": [],
            "diagnostics": {},
        }
    }
    sampling_note = "Target 'target' was scored on a deterministic sample of 8,000 rows from 500,000 available rows."
    state = PipelineState(raw_df=pd.DataFrame({"x": [1], "target": [0]}))
    state.results["profiling"] = StageResult(
        stage_name="profiling", meta=StageMeta(name="profiling"),
        outputs={"quality_score": 90},
    )
    state.results["supervised"] = StageResult(
        stage_name="supervised", meta=StageMeta(name="supervised"),
        outputs={"best_models": fake_best_models, "sampling_notes": [sampling_note]},
    )
    verdict = FindingsStage()._verdict(state)
    assert verdict["decision"] == "INVESTIGATE"
    assert any("[Sampling]" in r for r in verdict["reasons"])


# ---------------------------------------------------------------------------
# Integration: Titanic-like dataset (categorical + NaN + binary target)
# ---------------------------------------------------------------------------

def test_titanic_like_categorical_features_reach_model():
    """Sex and Embarked (categorical) must appear as features and improve score
    over a numeric-only baseline."""
    from explot.stages.dimensionality.stage import DimensionalityStage
    from explot.stages.exploration.stage import ExplorationStage
    from explot.stages.supervised.stage import SupervisedStage

    rng = np.random.default_rng(42)
    n = 400
    sex = rng.choice(["M", "F"], n)
    # Sex perfectly correlates with survival (strong signal)
    survived = (sex == "F").astype(int)
    df = pd.DataFrame({
        "sex": sex,
        "age": rng.integers(1, 80, n).astype(float),
        "fare": rng.exponential(30, n),
        "embarked": rng.choice(["S", "C", "Q"], n),
        "survived": survived,
    })
    # Introduce some NaN in age (20%)
    df.loc[rng.choice(n, n // 5, replace=False), "age"] = np.nan

    state = PipelineState(raw_df=df)
    hooks, config = _Hooks(), _Config()
    state.results["profiling"] = ProfilingStage().run(state, config, hooks)
    state.results["preprocessing"] = PreprocessingStage().run(state, config, hooks)
    state.results["exploration"] = ExplorationStage().run(state, config, hooks)
    state.results["dimensionality"] = DimensionalityStage().run(state, config, hooks)
    sup = SupervisedStage().run(state, config, hooks)
    state.results["supervised"] = sup

    assert sup.success
    best = sup.outputs["best_models"].get("survived", {})
    # Should score well because sex is a perfect predictor
    assert best.get("mean", 0) >= 0.80, f"Expected score ≥ 0.80 with sex encoded, got {best.get('mean')}"
    # track_c (raw preprocessed) should have participated
    assert "model_results_track_c" in sup.outputs
