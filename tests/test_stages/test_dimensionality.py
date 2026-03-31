from __future__ import annotations

import pandas as pd
import pytest

from explot.stages.base import StageResult
from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.profiling.stage import ProfilingStage
from explot.state import PipelineState


def _run_dim(df: pd.DataFrame) -> StageResult:
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
    return DimensionalityStage().run(state, config, hooks)


# ---- scrna ----

def test_dim_applies_log1p_for_raw_counts(scrna_data):
    df, meta = scrna_data
    result = _run_dim(df)
    assert result.success
    log_entries = " ".join(result.outputs["transform_log"])
    assert "log1p" in log_entries.lower()


def test_dim_excludes_id_column_for_scrna(scrna_data):
    df, meta = scrna_data
    result = _run_dim(df)
    transformed_cols = list(result.outputs["transformed_df"].columns)
    assert "cell_id" not in transformed_cols


def test_dim_intrinsic_dim_scrna(scrna_data):
    df, meta = scrna_data
    result = _run_dim(df)
    dim = result.outputs["intrinsic_dim"]
    assert 1 <= dim <= 30, f"Intrinsic dim {dim} seems unreasonable for scrna"


def test_dim_pca_2d_shape(scrna_data):
    df, meta = scrna_data
    result = _run_dim(df)
    pca_2d = result.outputs["pca_2d"]
    assert pca_2d.shape == (len(df), 2)


# ---- proteomics ----

def test_dim_no_log1p_for_log_normalized(proteomics_data):
    df, meta = proteomics_data
    result = _run_dim(df)
    log_entries = " ".join(result.outputs["transform_log"])
    assert "log1p" not in log_entries.lower() or "raw counts" not in log_entries.lower()


def test_dim_excludes_sample_id(proteomics_data):
    df, meta = proteomics_data
    result = _run_dim(df)
    transformed_cols = list(result.outputs["transformed_df"].columns)
    assert "sample_id" not in transformed_cols


# ---- tabular ----

def test_dim_excludes_id_for_tabular(tabular_data):
    df, meta = tabular_data
    result = _run_dim(df)
    transformed_cols = list(result.outputs["transformed_df"].columns)
    id_cols = [c["name"] for c in meta["profiling"]["suspicious_columns"] if c["reason"] == "id_like"]
    for col in id_cols:
        assert col not in transformed_cols, f"Expected '{col}' excluded from transformed_df"


# ---- interpretations ----

def test_dim_interpretations_exist(scrna_data):
    df, meta = scrna_data
    result = _run_dim(df)
    assert "pca_variance" in result.interpretations
    assert "transform_log" in result.interpretations
    assert "svd_explainer" in result.interpretations
    assert "intrinsic dimensionality" in result.interpretations["pca_variance"].lower()


def test_dim_variance_components_reasonable(scrna_data):
    df, meta = scrna_data
    result = _run_dim(df)
    n50 = result.outputs["n_components_50"]
    n80 = result.outputs["n_components_80"]
    n95 = result.outputs["n_components_95"]
    assert 1 <= n50 <= n80 <= n95
    assert n95 <= len(result.outputs["pca_explained_variance"])


def test_dimensionality_generates_pca_figures(scrna_data):
    df, meta = scrna_data
    result = _run_dim(df)
    assert "scree_plot" in result.figures
    assert "projection_plot" in result.figures
    assert "<svg" in result.figures["scree_plot"]
    assert "<svg" in result.figures["projection_plot"]
