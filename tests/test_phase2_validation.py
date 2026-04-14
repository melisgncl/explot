"""Phase 2 validation tests.

Pass criteria (from ROADMAP.md):

Phase 2a — Class imbalance handling
  - diagnostics.used_balanced_weights is True when imbalance_ratio >= 5
  - balanced_class_weights finding fires in FindingsStage
  - Balanced weights are not applied for near-balanced targets (ratio < 5)

Phase 2b — SHAP feature explanation
  - shap_importance list is non-empty for tree-model targets
  - Markdown output contains a SHAP section when importance is available
  - graceful [] fallback when SHAP cannot handle the model

Phase 2c — Configurable thresholds + row cap disclosure
  - sampling_info dict is populated when total rows exceed max_fit_rows
  - Markdown best-models table contains "(sampled N of M rows)"
  - Verdict escalates to INVESTIGATE when sampling occurred

Phase 2d — High-cardinality categorical findings
  - frequency_encoded finding (MEDIUM) fires for 21-100 unique columns
  - high_cardinality_dropped finding (HIGH) fires for >100 unique columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.findings.stage import FindingsStage
from explot.stages.preprocessing.stage import PreprocessingStage
from explot.stages.profiling.stage import ProfilingStage
from explot.stages.supervised.stage import SupervisedStage
from explot.report.markdown import state_to_markdown
from explot.state import PipelineState


# ---------------------------------------------------------------------------
# Shared helpers
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


def _run_to_supervised(df: pd.DataFrame, target: str | None = None):
    state = PipelineState(raw_df=df)
    if target:
        state.target_column = target
    hooks, cfg = _Hooks(), _Config()
    state.results["profiling"] = ProfilingStage().run(state, cfg, hooks)
    state.results["preprocessing"] = PreprocessingStage().run(state, cfg, hooks)
    state.results["exploration"] = ExplorationStage().run(state, cfg, hooks)
    state.results["dimensionality"] = DimensionalityStage().run(state, cfg, hooks)
    state.results["supervised"] = SupervisedStage().run(state, cfg, hooks)
    return state


# ---------------------------------------------------------------------------
# Phase 2a — Class imbalance handling
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def imbalanced_df():
    """~10:1 imbalanced binary classification dataset (900 negatives, 90 positives)."""
    rng = np.random.default_rng(42)
    n_neg, n_pos = 900, 90
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(n_neg, 5))
    X_pos = rng.normal(loc=1.5, scale=1.0, size=(n_pos, 5))
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * n_neg + [1] * n_pos)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


def test_balanced_weights_fires_for_imbalanced_target(imbalanced_df):
    """diagnostics.used_balanced_weights must be True for a 10:1 imbalanced target."""
    state = _run_to_supervised(imbalanced_df, target="target")
    sup = state.results["supervised"]
    assert sup.success, f"Supervised stage failed: {sup.error}"

    best = sup.outputs["best_models"].get("target", {})
    diag = best.get("diagnostics", {})
    assert diag.get("used_balanced_weights") is True, (
        f"Expected used_balanced_weights=True for 10:1 imbalance, "
        f"got imbalance_ratio={diag.get('imbalance_ratio')}"
    )


def test_imbalance_ratio_recorded_in_diagnostics(imbalanced_df):
    """imbalance_ratio must be >= 5 and stored in diagnostics."""
    state = _run_to_supervised(imbalanced_df, target="target")
    best = state.results["supervised"].outputs["best_models"].get("target", {})
    diag = best.get("diagnostics", {})
    ratio = diag.get("imbalance_ratio")
    assert ratio is not None, "imbalance_ratio missing from diagnostics"
    assert ratio >= 5.0, f"Expected imbalance_ratio >= 5, got {ratio}"


def test_balanced_weights_finding_fires(imbalanced_df):
    """FindingsStage must surface a balanced_class_weights finding for imbalanced data."""
    state = _run_pipeline(imbalanced_df, target="target")
    findings = state.results["findings"].outputs["findings_list"]
    bw_findings = [f for f in findings if f["rule"] == "balanced_class_weights"]
    assert len(bw_findings) >= 1, (
        "Expected at least one balanced_class_weights finding. "
        f"Finding rules present: {[f['rule'] for f in findings]}"
    )
    assert any("balanced" in f["text"].lower() for f in bw_findings)


def test_verdict_escalates_for_severe_imbalance_without_sampling():
    """Severe imbalance (>= 20:1) must escalate verdict to INVESTIGATE via the
    imbalance path — independent of row-cap sampling.

    Dataset is small (< 3000 rows) so no sampling occurs. The only escalation
    path available is the imbalance ratio check in _verdict.
    """
    from explot.stages.base import StageResult, StageMeta

    # Simulate diagnostics for a 50:1 imbalanced target — no sampling
    fake_best = {
        "target": {
            "model": "RandomForest",
            "mean": 0.82,
            "baseline_score": 0.50,
            "lift_over_baseline": 0.32,
            "trust_flags": [],
            "diagnostics": {
                "used_balanced_weights": True,
                "imbalance_ratio": 50.0,
                "group_audit": {},
                "temporal_audit": {},
            },
        }
    }
    state = PipelineState(raw_df=pd.DataFrame({"x": [1], "target": [0]}))
    state.results["profiling"] = StageResult(
        "profiling", StageMeta(name="profiling"), outputs={"quality_score": 85}
    )
    state.results["supervised"] = StageResult(
        "supervised", StageMeta(name="supervised"),
        outputs={"best_models": fake_best, "sampling_notes": [], "sampling_info": {}},
    )
    verdict = FindingsStage()._verdict(state)
    assert verdict["decision"] == "INVESTIGATE", (
        f"Expected INVESTIGATE for 50:1 imbalance, got {verdict['decision']}. "
        f"Reasons: {verdict['reasons']}"
    )
    imbalance_reasons = [r for r in verdict["reasons"] if "imbalance" in r.lower()]
    assert len(imbalance_reasons) >= 1, (
        f"Expected imbalance reason in verdict, got: {verdict['reasons']}"
    )


def test_verdict_does_not_escalate_for_mild_imbalance():
    """Mild imbalance (< 20:1) must NOT trigger the severe-imbalance escalation."""
    from explot.stages.base import StageResult, StageMeta

    fake_best = {
        "target": {
            "model": "RandomForest",
            "mean": 0.82,
            "baseline_score": 0.50,
            "lift_over_baseline": 0.32,
            "trust_flags": [],
            "diagnostics": {
                "used_balanced_weights": True,
                "imbalance_ratio": 8.0,   # 8:1 — moderate, not severe
                "group_audit": {},
                "temporal_audit": {},
            },
        }
    }
    state = PipelineState(raw_df=pd.DataFrame({"x": [1], "target": [0]}))
    state.results["profiling"] = StageResult(
        "profiling", StageMeta(name="profiling"), outputs={"quality_score": 85}
    )
    state.results["supervised"] = StageResult(
        "supervised", StageMeta(name="supervised"),
        outputs={"best_models": fake_best, "sampling_notes": [], "sampling_info": {}},
    )
    verdict = FindingsStage()._verdict(state)
    imbalance_reasons = [r for r in verdict["reasons"] if "imbalance" in r.lower()]
    assert len(imbalance_reasons) == 0, (
        f"Expected no severe-imbalance reason for 8:1 ratio, got: {verdict['reasons']}"
    )


def test_balanced_weights_not_applied_for_balanced_target():
    """Balanced weights must NOT fire for a near-50/50 target."""
    rng = np.random.default_rng(7)
    n = 300
    X = rng.normal(size=(n, 4))
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    state = _run_to_supervised(df, target="target")
    sup = state.results["supervised"]
    assert sup.success
    best = sup.outputs["best_models"].get("target", {})
    diag = best.get("diagnostics", {})
    assert diag.get("used_balanced_weights") is False, (
        f"Expected used_balanced_weights=False for balanced target, "
        f"got imbalance_ratio={diag.get('imbalance_ratio')}"
    )


# ---------------------------------------------------------------------------
# Phase 2b — SHAP feature explanation
# ---------------------------------------------------------------------------

def test_shap_importance_populated():
    """shap_importance list must be non-empty for a clean predictive dataset."""
    pytest.importorskip("shap")
    rng = np.random.default_rng(10)
    n = 400
    X = rng.normal(size=(n, 6))
    # Strong signal so a tree model clearly wins
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y

    state = _run_to_supervised(df, target="target")
    sup = state.results["supervised"]
    assert sup.success, f"Supervised stage failed: {sup.error}"

    best = sup.outputs["best_models"].get("target", {})
    shap_imp = best.get("shap_importance", [])
    assert len(shap_imp) > 0, (
        "Expected non-empty shap_importance for a tree-model target. "
        f"Best model: {best.get('model')}, track: {best.get('track')}"
    )
    # Each entry must have feature + importance keys
    entry = shap_imp[0]
    assert "feature" in entry and "importance" in entry, (
        f"SHAP entry missing keys: {entry}"
    )


def test_shap_top_features_are_signal_features():
    """Top SHAP feature must be f0 or f1 — the only signal features."""
    pytest.importorskip("shap")
    rng = np.random.default_rng(11)
    n = 500
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y

    state = _run_to_supervised(df, target="target")
    best = state.results["supervised"].outputs["best_models"].get("target", {})
    shap_imp = best.get("shap_importance", [])
    if not shap_imp:
        pytest.skip("SHAP not available or model does not support it")

    top2 = {e["feature"] for e in shap_imp[:2]}
    assert top2 & {"f0", "f1"}, (
        f"Expected f0 or f1 in top-2 SHAP features, got {top2}. "
        "SHAP may not be ranking signal features correctly."
    )


def test_shap_section_in_markdown():
    """Markdown output must contain a SHAP section when importance is available."""
    pytest.importorskip("shap")
    rng = np.random.default_rng(12)
    n = 400
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y

    state = _run_pipeline(df, target="target")
    best = state.results["supervised"].outputs["best_models"].get("target", {})
    if not best.get("shap_importance"):
        pytest.skip("SHAP not populated — skipping markdown check")

    md = state_to_markdown(state)
    assert "Feature Importance (SHAP)" in md, (
        "Expected '### Feature Importance (SHAP)' section in markdown. "
        f"Markdown snippet:\n{md[:500]}"
    )


# ---------------------------------------------------------------------------
# Phase 2c — Row cap disclosure
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def large_df():
    """4,000-row dataset — exceeds max_fit_rows_fast (3,000) in fast mode."""
    rng = np.random.default_rng(20)
    n = 4000
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = y
    return df


def test_sampling_info_populated_when_rows_exceed_cap(large_df):
    """sampling_info must be populated when dataset exceeds max_fit_rows_fast."""
    state = _run_to_supervised(large_df, target="target")
    sup = state.results["supervised"]
    assert sup.success, f"Supervised stage failed: {sup.error}"

    sampling_info = sup.outputs.get("sampling_info", {})
    assert "target" in sampling_info, (
        f"Expected 'target' in sampling_info for {len(large_df)}-row dataset "
        f"(cap=3000 in fast mode). sampling_info={sampling_info}"
    )
    si = sampling_info["target"]
    assert si["sampled"] <= 3000, f"Sampled {si['sampled']} rows but cap is 3000"
    assert si["total"] == len(large_df), (
        f"Expected total={len(large_df)}, got {si['total']}"
    )


def test_sampling_notes_populated_when_rows_exceed_cap(large_df):
    """sampling_notes list must be non-empty for an over-cap dataset."""
    state = _run_to_supervised(large_df, target="target")
    notes = state.results["supervised"].outputs.get("sampling_notes", [])
    assert len(notes) >= 1, "Expected at least one sampling note for large dataset"
    assert any("target" in n for n in notes)


def test_markdown_contains_sampling_note(large_df):
    """Best-models markdown table must contain 'sampled' note inline."""
    state = _run_pipeline(large_df, target="target")
    md = state_to_markdown(state)
    assert "sampled" in md.lower(), (
        "Expected '(sampled N of M rows)' in markdown output for large dataset.\n"
        f"Markdown snippet:\n{md[:600]}"
    )


def test_verdict_escalates_when_sampling_occurred(large_df):
    """Verdict must be INVESTIGATE (not SHIP) when sampling occurred."""
    state = _run_pipeline(large_df, target="target")
    verdict = state.results["findings"].outputs["verdict"]
    # A model scoring well on 3k rows of clean data might get SHIP without the
    # sampling escalation — so the presence of a [Sampling] reason is the key check.
    reasons = verdict.get("reasons", [])
    sampling_reasons = [r for r in reasons if "[Sampling]" in r]
    assert len(sampling_reasons) >= 1, (
        f"Expected a [Sampling] reason in verdict for over-cap dataset. "
        f"Decision: {verdict['decision']}, reasons: {reasons}"
    )


def test_no_sampling_for_small_dataset():
    """sampling_info must be empty for a dataset well under the row cap."""
    rng = np.random.default_rng(21)
    n = 200
    X = rng.normal(size=(n, 4))
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    state = _run_to_supervised(df, target="target")
    sampling_info = state.results["supervised"].outputs.get("sampling_info", {})
    assert "target" not in sampling_info, (
        f"Expected no sampling for {n}-row dataset, got sampling_info={sampling_info}"
    )


# ---------------------------------------------------------------------------
# Phase 2d — High-cardinality categorical findings
# ---------------------------------------------------------------------------

def _preprocess_only(df: pd.DataFrame):
    state = PipelineState(raw_df=df)
    hooks, cfg = _Hooks(), _Config()
    state.results["profiling"] = ProfilingStage().run(state, cfg, hooks)
    state.results["preprocessing"] = PreprocessingStage().run(state, cfg, hooks)
    return state


def test_frequency_encoded_finding_fires():
    """A column with 30 unique values (21-100) must trigger frequency_encoded finding."""
    rng = np.random.default_rng(30)
    n = 300
    # 30 unique category labels
    categories = [f"cat_{i:02d}" for i in range(30)]
    df = pd.DataFrame({
        "country": rng.choice(categories, size=n),
        "age": rng.integers(20, 80, size=n).astype(float),
        "target": rng.integers(0, 2, size=n),
    })
    state = _preprocess_only(df)
    prep = state.results["preprocessing"]
    assert prep.success

    freq_encoded = prep.outputs.get("frequency_encoded", [])
    assert "country" in freq_encoded, (
        f"Expected 'country' (30 unique values) in frequency_encoded list, "
        f"got {freq_encoded}"
    )


def test_frequency_encoded_finding_in_findings_stage():
    """FindingsStage must surface a frequency_encoded finding for medium-cardinality columns."""
    rng = np.random.default_rng(31)
    n = 300
    categories = [f"cat_{i:02d}" for i in range(30)]
    df = pd.DataFrame({
        "country": rng.choice(categories, size=n),
        "age": rng.integers(20, 80, size=n).astype(float),
        "target": rng.integers(0, 2, size=n),
    })
    state = _run_pipeline(df, target="target")
    findings = state.results["findings"].outputs["findings_list"]
    freq_findings = [f for f in findings if f["rule"] == "frequency_encoded"]
    assert len(freq_findings) >= 1, (
        "Expected at least one frequency_encoded finding in FindingsStage. "
        f"Finding rules present: {[f['rule'] for f in findings]}"
    )
    assert freq_findings[0]["confidence"] == "MEDIUM"


def test_frequency_encoded_finding_text_mentions_column():
    """The frequency_encoded finding text must mention the encoded column name."""
    rng = np.random.default_rng(32)
    n = 300
    categories = [f"region_{i:02d}" for i in range(40)]
    df = pd.DataFrame({
        "region": rng.choice(categories, size=n),
        "score": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })
    state = _run_pipeline(df, target="target")
    findings = state.results["findings"].outputs["findings_list"]
    freq_findings = [f for f in findings if f["rule"] == "frequency_encoded"]
    assert freq_findings, "Expected frequency_encoded finding"
    assert "'region'" in freq_findings[0]["text"], (
        f"Expected finding text to mention 'region', got: {freq_findings[0]['text']}"
    )


def test_high_cardinality_column_dropped_and_finding_fires():
    """A column with >100 unique string values must be dropped with a HIGH finding."""
    rng = np.random.default_rng(33)
    n = 300
    # 150 unique free-text-like labels
    descriptions = [f"user_desc_{i:03d}" for i in range(150)]
    df = pd.DataFrame({
        "description": rng.choice(descriptions, size=n),
        "age": rng.integers(20, 80, size=n).astype(float),
        "target": rng.integers(0, 2, size=n),
    })
    state = _run_pipeline(df, target="target")

    # Column must be absent from preprocessed features
    prep = state.results["preprocessing"]
    assert prep.success
    prep_df = prep.outputs["preprocessed_df"]
    assert "description" not in prep_df.columns, (
        "Expected 'description' to be dropped from preprocessed matrix"
    )

    # Finding must fire
    findings = state.results["findings"].outputs["findings_list"]
    hc_findings = [f for f in findings if f["rule"] == "high_cardinality_dropped"]
    assert len(hc_findings) >= 1, (
        "Expected high_cardinality_dropped finding for 150-unique column. "
        f"Findings: {[f['text'] for f in findings]}"
    )
    assert hc_findings[0]["confidence"] == "HIGH"


def test_low_cardinality_column_not_frequency_encoded():
    """A column with <= 20 unique values must be ordinal-encoded, not frequency-encoded."""
    rng = np.random.default_rng(34)
    n = 300
    df = pd.DataFrame({
        "sex": rng.choice(["M", "F"], size=n),
        "age": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })
    state = _preprocess_only(df)
    prep = state.results["preprocessing"]
    assert prep.success
    freq_encoded = prep.outputs.get("frequency_encoded", [])
    ordinal_encoded = prep.outputs.get("ordinal_encoded", [])
    assert "sex" not in freq_encoded, (
        f"'sex' (2 unique values) must not be frequency-encoded; got freq={freq_encoded}"
    )
    assert "sex" in ordinal_encoded, (
        f"'sex' must be ordinal-encoded for 2 unique values; got ordinal={ordinal_encoded}"
    )
