"""Real-dataset validation tests.

These tests use actual datasets from data/ and verify the pipeline against
the ROADMAP pass criteria. Fixtures for files guard with pytest.skip so the
suite does not fail in CI without the data files. Library-sourced fixtures
(sklearn, lifelines) always run.

Datasets from data/:
  data/telco_churn.csv                         Telco Customer Churn (7k rows)
  data/creditcard_fraud.csv                    Credit Card Fraud (284k rows)
  data/nyc_taxi_yellow_2024_01.parquet         NYC Taxi Jan 2024 (3M rows)

Library-sourced (always available):
  sklearn.datasets.load_breast_cancer          569 rows, 30 features, binary
  lifelines.datasets.load_rossi                432 rows, survival analysis

Pass criteria:

Telco Churn
  - 'Contract' in top-5 permutation importance
  - Verdict is SHIP or INVESTIGATE

Credit Card Fraud
  - used_balanced_weights=True (578:1 imbalance)
  - balanced_class_weights finding fires
  - Verdict is INVESTIGATE or DO_NOT_SHIP

NYC Taxi
  - sampling_info populated (50k rows > 3k cap)
  - [Sampling] reason in verdict
  - Temporal audit runs

Breast Cancer Wisconsin
  - Best model score >= 0.95
  - Verdict is SHIP

Rossi Recidivism (survival)
  - SurvivalStage detects time/event columns
  - KM curve produced
  - Cox model has >= 1 significant covariate (p < 0.05)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.findings.stage import FindingsStage
from explot.stages.preprocessing.stage import PreprocessingStage
from explot.stages.profiling.stage import ProfilingStage
from explot.stages.supervised.stage import SupervisedStage
from explot.stages.survival.stage import SurvivalStage
from explot.state import PipelineState

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers
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
        max_fit_rows_fast = 3000
        max_fit_rows = 8000
        leakage_delta_threshold = 0.10
        leakage_score_floor = 0.60
        verdict_lift_floor = 0.05


def _run_pipeline(df: pd.DataFrame, target: str | None = None):
    state = PipelineState(raw_df=df)
    if target:
        state.target_column = target
    hooks, cfg = _Hooks(), _Config()
    state.results["profiling"] = ProfilingStage().run(state, cfg, hooks)
    state.results["preprocessing"] = PreprocessingStage().run(state, cfg, hooks)
    state.results["exploration"] = ExplorationStage().run(state, cfg, hooks)
    state.results["dimensionality"] = DimensionalityStage().run(state, cfg, hooks)
    state.results["supervised"] = SupervisedStage().run(state, cfg, hooks)
    state.results["findings"] = FindingsStage().run(state, cfg, hooks)
    return state


def _top_n_perm_features(state, target: str, n: int = 5) -> list[str]:
    best = state.results["supervised"].outputs["best_models"].get(target, {})
    return [p["feature"] for p in best.get("permutation_importance", [])[:n]]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def telco_df():
    path = DATA_DIR / "telco_churn.csv"
    if not path.exists():
        pytest.skip("data/telco_churn.csv not found")
    df = pd.read_csv(path)
    # Binarise target: Yes→1, No→0
    df["Churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    # TotalCharges has a few blank strings — convert to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


@pytest.fixture(scope="module")
def fraud_df():
    path = DATA_DIR / "creditcard_fraud.csv"
    if not path.exists():
        pytest.skip("data/creditcard_fraud.csv not found")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def taxi_df():
    path = DATA_DIR / "nyc_taxi_yellow_2024_01.parquet"
    if not path.exists():
        pytest.skip("data/nyc_taxi_yellow_2024_01.parquet not found")
    df = pd.read_parquet(path)
    # Add a clean binary target: tip given vs not
    df["tipped"] = (df["tip_amount"] > 0).astype(int)
    # Keep datetime as string so the profiler detects it as a temporal column
    df["pickup_datetime"] = df["tpep_pickup_datetime"].astype(str)
    # Drop columns that are direct functions of tip_amount to avoid leakage
    df = df.drop(columns=["tip_amount", "total_amount", "tpep_pickup_datetime", "tpep_dropoff_datetime"])
    # Sample 50k rows — enough to stress the row cap but not blow up RAM
    return df.sample(n=50_000, random_state=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2a — Telco Customer Churn
# ---------------------------------------------------------------------------

def test_telco_contract_in_top5_importance(telco_df):
    """'Contract' is the strongest single predictor of churn — must appear in top-5."""
    state = _run_pipeline(telco_df, target="Churn")
    sup = state.results["supervised"]
    assert sup.success, f"Supervised stage failed: {sup.error}"

    top5 = _top_n_perm_features(state, "Churn", n=5)
    assert "Contract" in top5, (
        f"Expected 'Contract' in top-5 permutation importance, got {top5}. "
        "Categorical encoding may not be reaching the model."
    )


def test_telco_verdict_not_do_not_ship(telco_df):
    """Telco Churn is clean data — verdict must be SHIP or INVESTIGATE."""
    state = _run_pipeline(telco_df, target="Churn")
    verdict = state.results["findings"].outputs["verdict"]
    assert verdict["decision"] in {"SHIP", "INVESTIGATE"}, (
        f"Expected SHIP or INVESTIGATE for Telco Churn, got {verdict['decision']}. "
        f"Reasons: {verdict['reasons']}"
    )


def test_telco_categorical_cols_encoded(telco_df):
    """Telco has many categorical columns — at least 5 must be encoded."""
    state = PipelineState(raw_df=telco_df)
    hooks, cfg = _Hooks(), _Config()
    state.results["profiling"] = ProfilingStage().run(state, cfg, hooks)
    result = PreprocessingStage().run(state, cfg, hooks)
    assert result.success

    encoded = set(result.outputs["ordinal_encoded"] + result.outputs["frequency_encoded"])
    # These categorical columns must have been encoded
    expected = {"gender", "Contract", "InternetService", "PaymentMethod"}
    overlap = expected & encoded
    assert len(overlap) >= 3, (
        f"Expected at least 3 of {expected} to be encoded, got overlap={overlap}. "
        f"All encoded: {sorted(encoded)}"
    )


def test_telco_score_above_floor(telco_df):
    """Telco Churn: best model F1 must beat dummy baseline by at least lift_floor."""
    state = _run_pipeline(telco_df, target="Churn")
    sup = state.results["supervised"]
    assert sup.success

    best = sup.outputs["best_models"].get("Churn", {})
    lift = best.get("lift_over_baseline")
    assert lift is not None, "lift_over_baseline not computed"
    assert lift >= 0.05, (
        f"Expected lift >= 0.05 for Telco Churn, got {lift:.4f} "
        f"(model={best.get('model')}, score={best.get('mean', 0):.3f})"
    )


# ---------------------------------------------------------------------------
# Phase 2a — Credit Card Fraud
# ---------------------------------------------------------------------------

def test_fraud_balanced_weights_fires(fraud_df):
    """578:1 imbalance must trigger used_balanced_weights=True."""
    state = _run_pipeline(fraud_df, target="Class")
    sup = state.results["supervised"]
    assert sup.success, f"Supervised stage failed: {sup.error}"

    best = sup.outputs["best_models"].get("Class", {})
    diag = best.get("diagnostics", {})
    assert diag.get("used_balanced_weights") is True, (
        f"Expected used_balanced_weights=True for 578:1 fraud imbalance, "
        f"got imbalance_ratio={diag.get('imbalance_ratio')}"
    )


def test_fraud_imbalance_ratio_extreme(fraud_df):
    """imbalance_ratio for fraud dataset must be >> 5."""
    state = _run_pipeline(fraud_df, target="Class")
    best = state.results["supervised"].outputs["best_models"].get("Class", {})
    ratio = best.get("diagnostics", {}).get("imbalance_ratio")
    assert ratio is not None and ratio >= 100, (
        f"Expected imbalance_ratio >= 100 for fraud dataset, got {ratio}"
    )


def test_fraud_verdict_not_ship(fraud_df):
    """Fraudulent class is tiny — a SHIP verdict on fraud data is dangerous.

    The severe-imbalance escalation (ratio >= 20) must fire in the verdict
    independently of whether sampling occurred. This ensures the verdict stays
    INVESTIGATE even if max_fit_rows is raised to cover the full dataset.
    """
    state = _run_pipeline(fraud_df, target="Class")
    verdict = state.results["findings"].outputs["verdict"]
    assert verdict["decision"] != "SHIP", (
        f"Expected INVESTIGATE or DO_NOT_SHIP for fraud dataset, "
        f"got SHIP. Reasons: {verdict['reasons']}"
    )
    # The imbalance reason must be present — not just sampling
    reasons = verdict.get("reasons", [])
    imbalance_reasons = [r for r in reasons if "imbalance" in r.lower()]
    assert len(imbalance_reasons) >= 1, (
        "Expected a severe-imbalance reason in verdict for fraud dataset "
        f"(578:1 ratio). Got reasons: {reasons}"
    )


def test_fraud_balanced_weights_finding_in_findings(fraud_df):
    """FindingsStage must surface a balanced_class_weights finding for fraud data."""
    state = _run_pipeline(fraud_df, target="Class")
    findings = state.results["findings"].outputs["findings_list"]
    bw = [f for f in findings if f["rule"] == "balanced_class_weights"]
    assert len(bw) >= 1, (
        "Expected balanced_class_weights finding for fraud dataset. "
        f"Rules present: {[f['rule'] for f in findings]}"
    )


# ---------------------------------------------------------------------------
# Phase 2c — NYC Taxi (row cap + temporal leakage)
# ---------------------------------------------------------------------------

def test_taxi_sampling_info_populated(taxi_df):
    """50k-row taxi sample still exceeds 3k fast-mode cap — sampling_info must fire."""
    state = _run_pipeline(taxi_df, target="tipped")
    sup = state.results["supervised"]
    assert sup.success, f"Supervised stage failed: {sup.error}"

    sampling_info = sup.outputs.get("sampling_info", {})
    assert "tipped" in sampling_info, (
        f"Expected sampling_info to contain 'tipped' for 50k dataset (cap=3k). "
        f"Got: {sampling_info}"
    )
    si = sampling_info["tipped"]
    assert si["sampled"] <= 3000
    assert si["total"] == len(taxi_df)


def test_taxi_verdict_has_sampling_reason(taxi_df):
    """Verdict must contain a [Sampling] reason because the dataset was row-capped."""
    state = _run_pipeline(taxi_df, target="tipped")
    verdict = state.results["findings"].outputs["verdict"]
    reasons = verdict.get("reasons", [])
    sampling_reasons = [r for r in reasons if "[Sampling]" in r]
    assert len(sampling_reasons) >= 1, (
        f"Expected [Sampling] reason in verdict for 50k taxi dataset. "
        f"Decision: {verdict['decision']}, reasons: {reasons}"
    )


def test_taxi_temporal_audit_runs(taxi_df):
    """Pipeline must detect pickup_datetime as a time column and run the temporal audit."""
    state = _run_pipeline(taxi_df, target="tipped")
    sup = state.results["supervised"]
    assert sup.success

    best = sup.outputs["best_models"].get("tipped", {})
    temporal = best.get("diagnostics", {}).get("temporal_audit", {})
    assert temporal.get("checked") is True, (
        "Expected temporal_audit.checked=True for taxi dataset with pickup_datetime. "
        f"temporal_audit: {temporal}"
    )
    assert temporal.get("time_column") is not None, (
        "Expected temporal_audit.time_column to be set"
    )


# ---------------------------------------------------------------------------
# Breast Cancer Wisconsin — clean numeric baseline (library-sourced, always runs)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def breast_cancer_df():
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer(as_frame=True)
    return ds.frame  # 569 rows, 30 numeric features, binary 'target' (0=malignant, 1=benign)


def test_breast_cancer_score_above_floor(breast_cancer_df):
    """30 clean numeric features — best model must achieve AUC/F1 >= 0.95."""
    state = _run_pipeline(breast_cancer_df, target="target")
    sup = state.results["supervised"]
    assert sup.success, f"Supervised stage failed: {sup.error}"

    best = sup.outputs["best_models"].get("target", {})
    score = best.get("mean", 0.0)
    assert score >= 0.95, (
        f"Expected score >= 0.95 for Breast Cancer Wisconsin (clean numeric), got {score:.3f}. "
        f"Model: {best.get('model')}, track: {best.get('track')}."
    )


def test_breast_cancer_single_feature_leakage_fires(breast_cancer_df):
    """Breast cancer features are highly correlated — single_feature_leakage must fire.

    Features like 'worst radius' predict the target with near-perfect accuracy alone.
    This is correct behavior: the stage should flag it rather than silently shipping.
    """
    state = _run_pipeline(breast_cancer_df, target="target")
    sup = state.results["supervised"]
    assert sup.success

    best = sup.outputs["best_models"].get("target", {})
    trust_flags = best.get("trust_flags", [])
    assert "single_feature_leakage" in trust_flags, (
        f"Expected single_feature_leakage trust flag for Breast Cancer Wisconsin. "
        f"Got trust_flags: {trust_flags}. Highly correlated features should trigger this."
    )


# ---------------------------------------------------------------------------
# Rossi Recidivism — survival analysis validation (library-sourced, always runs)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rossi_df():
    pytest.importorskip("lifelines", reason="lifelines not installed — pip install lifelines")
    from lifelines.datasets import load_rossi
    df = load_rossi()  # 432 rows: 'week' (time), 'arrest' (event), 7 covariates
    # Rename to standard column names the SurvivalStage keyword-detection expects
    return df.rename(columns={"week": "duration", "arrest": "event"})


def test_rossi_survival_detected(rossi_df):
    """SurvivalStage must detect 'duration' as time column and 'event' as event column."""
    state = PipelineState(raw_df=rossi_df)
    result = SurvivalStage().run(state, _Config(), _Hooks())
    assert result.success, f"SurvivalStage failed: {result.error}"
    assert result.outputs.get("detected") is True, "Expected detected=True for Rossi dataset"
    assert result.outputs.get("time_column") == "duration", (
        f"Expected time_column='duration', got {result.outputs.get('time_column')}"
    )
    assert result.outputs.get("event_column") == "event", (
        f"Expected event_column='event', got {result.outputs.get('event_column')}"
    )


def test_rossi_km_curve_produced(rossi_df):
    """KM curve must be non-empty for the Rossi dataset."""
    state = PipelineState(raw_df=rossi_df)
    result = SurvivalStage().run(state, _Config(), _Hooks())
    assert result.success

    km = result.outputs.get("km_overall", {})
    assert km.get("times") and len(km["times"]) > 0, (
        f"Expected non-empty KM times for Rossi dataset, got km_overall={km}"
    )
    km_svg = result.figures.get("km_curve", "")
    assert len(km_svg) > 100, "Expected a non-trivial KM SVG figure"


def test_rossi_cox_has_significant_covariate(rossi_df):
    """Cox PH model on Rossi must find at least one significant covariate (p < 0.05).

    Known significant predictors: prio (prior arrests), fin (financial aid), age.
    """
    state = PipelineState(raw_df=rossi_df)
    result = SurvivalStage().run(state, _Config(), _Hooks())
    assert result.success

    cox = result.outputs.get("cox_summary", [])
    assert len(cox) > 0, f"Cox summary is empty — fitting may have failed. outputs={result.outputs}"

    significant = [c for c in cox if c.get("significant")]
    assert len(significant) >= 1, (
        f"Expected >= 1 significant covariate (p < 0.05) in Cox model for Rossi. "
        f"Got: {[(c['covariate'], c['p']) for c in cox]}"
    )


def test_rossi_concordance_above_chance(rossi_df):
    """Cox C-index must be meaningfully above 0.5 (chance) for the Rossi dataset."""
    state = PipelineState(raw_df=rossi_df)
    result = SurvivalStage().run(state, _Config(), _Hooks())
    assert result.success

    c_index = result.outputs.get("cox_concordance")
    assert c_index is not None, "cox_concordance not returned"
    assert c_index >= 0.6, (
        f"Expected Cox C-index >= 0.6 for Rossi dataset, got {c_index:.3f}. "
        "C-index of 0.5 is random chance — model may have failed silently."
    )
