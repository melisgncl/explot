from __future__ import annotations

import numpy as np
import pandas as pd

from explot.stages.autoencoder.stage import AutoencoderStage
from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.profiling.stage import ProfilingStage
from explot.stages.supervised.stage import SupervisedStage
from explot.stages.unsupervised.stage import UnsupervisedStage
from explot.state import PipelineState


def _run_supervised(df: pd.DataFrame) -> tuple:
    state = PipelineState(raw_df=df)

    class _Hooks:
        def progress(self, *a, **k): ...
        def stage_started(self, *a): ...
        def stage_finished(self, *a): return 0.0
        def stage_failed(self, *a): ...

    class _Config:
        class budget:
            mode = "fast"  # fast mode for tests

    hooks = _Hooks()
    config = _Config()
    state.results["profiling"] = ProfilingStage().run(state, config, hooks)
    state.results["exploration"] = ExplorationStage().run(state, config, hooks)
    state.results["dimensionality"] = DimensionalityStage().run(state, config, hooks)
    state.results["autoencoder"] = AutoencoderStage().run(state, config, hooks)
    state.results["unsupervised"] = UnsupervisedStage().run(state, config, hooks)
    result = SupervisedStage().run(state, config, hooks)
    return result, state


def test_supervised_detects_cell_type(scrna_data):
    df, meta = scrna_data
    result, _ = _run_supervised(df)
    assert result.success
    target_names = [c["name"] for c in result.outputs["candidate_targets"]]
    assert "cell_type" in target_names


def test_supervised_scrna_f1_reasonable(scrna_data):
    df, meta = scrna_data
    result, _ = _run_supervised(df)
    best = result.outputs["best_models"].get("cell_type", {})
    assert best.get("mean", 0) > 0.5, f"F1 too low: {best}"


def test_supervised_detects_treatment(proteomics_data):
    df, meta = proteomics_data
    result, _ = _run_supervised(df)
    target_names = [c["name"] for c in result.outputs["candidate_targets"]]
    assert "treatment" in target_names


def test_supervised_has_feature_importance(scrna_data):
    df, meta = scrna_data
    result, _ = _run_supervised(df)
    fi = result.outputs["feature_importances"]
    assert len(fi) > 0
    for target, track_map in fi.items():
        assert len(track_map) > 0
        assert any(len(feats) > 0 for feats in track_map.values())


def test_supervised_recommendation_interpretation(scrna_data):
    df, meta = scrna_data
    result, _ = _run_supervised(df)
    rec = result.interpretations["model_recommendation"]
    assert "cell_type" in rec.lower() or "target" in rec.lower()
    assert any(w in rec.lower() for w in ["best", "achieved", "model"])


def test_supervised_avoids_self_leakage_for_binary_numeric_target():
    rng = np.random.default_rng(42)
    n = 120
    target = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "feature_signal": rng.normal(size=n),
            "target_flag": target,
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
        }
    )

    result, _ = _run_supervised(df)

    best = result.outputs["best_models"].get("target_flag", {})
    assert best
    assert best.get("mean", 0) < 0.95


def test_supervised_adds_trust_notes_for_proxy_like_target():
    df = pd.DataFrame(
        {
            "feature_signal": list(range(160)),
            "proxy_flag": [0, 0, 1, 1] * 40,
            "target_flag": [0, 0, 1, 1] * 40,
        }
    )

    result, _ = _run_supervised(df)

    notes = result.interpretations["trust_notes"].lower()
    assert "proxy-like" in notes or "proxy" in notes


def test_supervised_flags_exact_copy_feature_as_leakage():
    df = pd.DataFrame(
        {
            "feature_signal": np.arange(120),
            "target_flag": [0, 1] * 60,
            "target_flag_copy": [0, 1] * 60,
        }
    )

    result, _ = _run_supervised(df)

    best = result.outputs["best_models"]["target_flag_copy"]
    flags = set(best["trust_flags"])
    notes = result.interpretations["trust_notes"].lower()
    assert "exact_copy_feature" in flags
    assert "possible_leakage" in flags
    assert "exact-copy" in notes or "strong leakage" in notes


def test_supervised_flags_high_correlation_proxy():
    x = np.linspace(0.0, 1.0, 120)
    df = pd.DataFrame(
        {
            "almost_target": x,
            "target_flag": x,
            "aux_noise": np.linspace(1.0, 2.0, 120),
        }
    )

    result, _ = _run_supervised(df)

    best = result.outputs["best_models"]["target_flag"]
    flags = set(best["trust_flags"])
    assert "high_correlation_proxy" in flags or "exact_copy_feature" in flags


def test_supervised_flags_single_feature_leakage():
    rng = np.random.default_rng(42)
    n = 200
    target = rng.integers(0, 3, size=n)
    df = pd.DataFrame(
        {
            "leaky_feature": target + rng.normal(0, 0.05, size=n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "target_col": target,
        }
    )
    result, _ = _run_supervised(df)
    best = result.outputs["best_models"].get("target_col", {})
    assert "single_feature_leakage" in best.get("trust_flags", [])


def test_supervised_exports_track_comparison_and_confusion_metrics(scrna_data):
    df, meta = scrna_data
    result, _ = _run_supervised(df)
    assert "track_comparison" in result.outputs
    assert "cell_type" in result.outputs["track_comparison"]
    details = result.outputs["evaluation_details"]["cell_type"]
    assert "track_a" in details
    assert "metrics" in details["track_a"]
    assert "precision_macro" in details["track_a"]["metrics"]
    assert "recall_macro" in details["track_a"]["metrics"]
