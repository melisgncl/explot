from __future__ import annotations

import json

from explot.stages.autoencoder.stage import AutoencoderStage
from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.profiling.stage import ProfilingStage
from explot.stages.supervised.stage import SupervisedStage
from explot.stages.unsupervised.stage import UnsupervisedStage
from explot.stages.findings.stage import FindingsStage
from explot.export import state_to_json, state_to_dict
from explot.state import PipelineState


def _run_pipeline(df):
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
    for stage_cls in (ProfilingStage, ExplorationStage, DimensionalityStage,
                      AutoencoderStage, UnsupervisedStage, SupervisedStage, FindingsStage):
        stage = stage_cls()
        try:
            state.results[stage.meta.name] = stage.run(state, config, hooks)
        except Exception:
            pass
    return state


def test_json_export_produces_valid_json(scrna_data):
    df, _ = scrna_data
    state = _run_pipeline(df)
    text = state_to_json(state)
    data = json.loads(text)
    assert "stages" in data
    assert "profiling" in data["stages"]
    assert "supervised" in data["stages"]
    assert data["stages"]["profiling"]["success"] is True


def test_json_export_contains_key_outputs(scrna_data):
    df, _ = scrna_data
    state = _run_pipeline(df)
    data = state_to_dict(state)
    sup = data["stages"]["supervised"]
    assert "best_models" in sup["outputs"]
    assert "interpretations" in sup
    findings = data["stages"]["findings"]
    assert "findings_list" in findings["outputs"]
