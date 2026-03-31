from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explot.stages.autoencoder.stage import AutoencoderStage
from explot.stages.base import StageResult
from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.profiling.stage import ProfilingStage
from explot.stages.unsupervised.stage import UnsupervisedStage
from explot.state import PipelineState


def _run_unsup(df: pd.DataFrame) -> StageResult:
    state = PipelineState(raw_df=df)

    class _Hooks:
        def progress(self, *a, **k): ...
        def stage_started(self, *a): ...
        def stage_finished(self, *a): return 0.0
        def stage_failed(self, *a): ...

    class _Config:
        class budget:
            mode = "full"

    hooks = _Hooks()
    config = _Config()
    state.results["profiling"] = ProfilingStage().run(state, config, hooks)
    state.results["exploration"] = ExplorationStage().run(state, config, hooks)
    state.results["dimensionality"] = DimensionalityStage().run(state, config, hooks)
    state.results["autoencoder"] = AutoencoderStage().run(state, config, hooks)
    return UnsupervisedStage().run(state, config, hooks)


# ---- scrna: expect ~4 clusters ----

def test_unsup_scrna_finds_clusters(scrna_data):
    df, meta = scrna_data
    result = _run_unsup(df)
    assert result.success
    k = result.outputs["kmeans_results"]["optimal_k"]
    true_k = meta.get("unsupervised", {}).get("true_k", 4)
    assert abs(k - true_k) <= 2, f"Detected k={k}, expected ~{true_k}"


def test_unsup_scrna_silhouette_positive(scrna_data):
    df, meta = scrna_data
    result = _run_unsup(df)
    sil = result.outputs["kmeans_results"]["best_silhouette"]
    assert sil > 0.1, f"Silhouette {sil} too low for data with planted clusters"


def test_unsup_scrna_has_anomalies(scrna_data):
    df, meta = scrna_data
    result = _run_unsup(df)
    assert isinstance(result.outputs["anomaly_rows"], list)


# ---- proteomics: expect ~3 treatment groups ----

def test_unsup_proteomics_finds_clusters(proteomics_data):
    df, meta = proteomics_data
    result = _run_unsup(df)
    assert result.success
    k = result.outputs["kmeans_results"]["optimal_k"]
    # Proteomics has 3 treatment groups but they may not be perfectly separable
    assert 2 <= k <= 6


# ---- tabular: no strong clusters expected ----

def test_unsup_tabular_low_silhouette(tabular_data):
    df, meta = tabular_data
    result = _run_unsup(df)
    assert result.success
    # Tabular data has no planted clusters, silhouette should be modest
    sil = result.outputs["kmeans_results"]["best_silhouette"]
    assert sil < 0.8, f"Unexpectedly high silhouette {sil} for structureless data"


# ---- interpretations ----

def test_unsup_interpretations_present(scrna_data):
    df, meta = scrna_data
    result = _run_unsup(df)
    assert "kmeans_silhouette" in result.interpretations
    assert "dbscan_results" in result.interpretations
    assert "isolation_forest" in result.interpretations
    assert "silhouette" in result.interpretations["kmeans_silhouette"].lower()


def test_unsup_dbscan_runs(scrna_data):
    df, meta = scrna_data
    result = _run_unsup(df)
    db = result.outputs["dbscan_results"]
    assert "n_clusters" in db
    assert "noise_fraction" in db
    assert 0.0 <= db["noise_fraction"] <= 1.0


def test_unsup_overlap_uses_stage2_outlier_rows():
    rng = np.random.default_rng(42)
    base = rng.normal(loc=0.0, scale=1.0, size=(118, 4))
    outliers = np.array([[12.0, 12.0, 12.0, 12.0], [14.0, 14.0, 14.0, 14.0]])
    matrix = np.vstack([base, outliers])
    df = pd.DataFrame(matrix, columns=["f1", "f2", "f3", "f4"])
    df["label"] = ["baseline"] * 118 + ["rare", "rare"]

    result = _run_unsup(df)

    overlap = result.outputs["cluster_outlier_overlap"]
    stage2_rows = overlap["stage2_outlier_rows"]
    stage4_rows = result.outputs["anomaly_rows"]
    assert stage2_rows == sorted(stage2_rows)
    assert overlap["overlap_count"] == len(set(stage2_rows) & set(stage4_rows))
    assert overlap["overlap_rows"] == sorted(set(stage2_rows) & set(stage4_rows))
    assert overlap["overlap_count"] >= 1


def test_unsup_exports_dvae_anomaly_comparison(scrna_data):
    df, _ = scrna_data
    result = _run_unsup(df)

    comparison = result.outputs["anomaly_signal_comparison"]
    assert result.outputs["representation_source"] in {"dvae", "dimensionality"}
    assert isinstance(result.outputs["dvae_anomaly_rows"], list)
    assert comparison["iso_dvae_overlap_count"] == len(
        set(result.outputs["anomaly_rows"]) & set(result.outputs["dvae_anomaly_rows"])
    )
    assert comparison["triple_overlap_count"] == len(
        set(result.outputs["anomaly_rows"])
        & set(result.outputs["dvae_anomaly_rows"])
        & set(result.outputs["cluster_outlier_overlap"]["stage2_outlier_rows"])
    )
    assert "anomaly_signal_comparison" in result.interpretations
