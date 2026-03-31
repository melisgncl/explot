from __future__ import annotations

import pandas as pd

from explot.stages.autoencoder.stage import AutoencoderStage
from explot.stages.dimensionality.stage import DimensionalityStage
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.findings.stage import FindingsStage
from explot.stages.profiling.stage import ProfilingStage
from explot.stages.supervised.stage import SupervisedStage
from explot.stages.unsupervised.stage import UnsupervisedStage
from explot.state import PipelineState


def _run_findings(df: pd.DataFrame):
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
    state.results["autoencoder"] = AutoencoderStage().run(state, config, hooks)
    state.results["unsupervised"] = UnsupervisedStage().run(state, config, hooks)
    state.results["supervised"] = SupervisedStage().run(state, config, hooks)
    return FindingsStage().run(state, config, hooks)


def test_findings_scrna_has_findings(scrna_data):
    df, meta = scrna_data
    result = _run_findings(df)
    assert result.success
    findings = result.outputs["findings_list"]
    assert len(findings) >= 3, f"Expected at least 3 findings, got {len(findings)}"


def test_findings_scrna_detects_clusters(scrna_data):
    df, meta = scrna_data
    result = _run_findings(df)
    texts = " ".join(f["text"].lower() for f in result.outputs["findings_list"])
    assert "cluster" in texts


def test_findings_scrna_detects_target(scrna_data):
    df, meta = scrna_data
    result = _run_findings(df)
    texts = " ".join(f["text"].lower() for f in result.outputs["findings_list"])
    assert "cell_type" in texts


def test_findings_has_summary_card(scrna_data):
    df, meta = scrna_data
    result = _run_findings(df)
    card = result.outputs["summary_card"]
    assert len(card) > 0 and len(card) <= 3


def test_findings_has_next_steps(scrna_data):
    df, meta = scrna_data
    result = _run_findings(df)
    steps = result.outputs["suggested_next_steps"]
    assert len(steps) > 0


def test_findings_confidence_levels(scrna_data):
    df, meta = scrna_data
    result = _run_findings(df)
    confidences = {f["confidence"] for f in result.outputs["findings_list"]}
    assert confidences.issubset({"HIGH", "MEDIUM", "LOW"})
