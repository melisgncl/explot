from __future__ import annotations

import pandas as pd

from explot.stages.autoencoder.stage import AutoencoderStage
from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.profiling.stage import ProfilingStage
from explot.state import PipelineState


def _run_autoencoder(df: pd.DataFrame):
    state = PipelineState(raw_df=df)

    class _Hooks:
        def progress(self, *a, **k): ...
        def stage_started(self, *a): ...
        def stage_finished(self, *a): return 0.0
        def stage_failed(self, *a): ...

    class _Config:
        class budget:
            mode = "fast"

    hooks = _Hooks()
    config = _Config()
    state.results["profiling"] = ProfilingStage().run(state, config, hooks)
    state.results["exploration"] = ExplorationStage().run(state, config, hooks)
    state.results["dimensionality"] = DimensionalityStage().run(state, config, hooks)
    return AutoencoderStage().run(state, config, hooks)


def test_autoencoder_outputs_scrna(scrna_data):
    df, _ = scrna_data
    result = _run_autoencoder(df)
    assert result.success
    assert result.outputs["bottleneck_dim"] >= 2
    assert result.outputs["latent_df"].shape[0] == len(df)
    assert result.outputs["reconstruction_mse"] is not None
    assert result.outputs["reconstruction_errors"]
    assert result.outputs["training_loss_curve"]
    assert result.outputs["model_type"] == "dvae"


def test_autoencoder_generates_latent_projection(scrna_data):
    df, _ = scrna_data
    result = _run_autoencoder(df)
    assert "<svg" in result.figures["latent_projection"]
    assert "<svg" in result.figures["training_loss"]
    assert "<svg" in result.figures["reconstruction_error"]
    assert "DVAE" in result.interpretations["summary"]
