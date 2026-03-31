from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from explot.config import load_config
from explot.hooks import HookRegistry
from explot.state import PipelineState
from explot.stages.exploration.stage import ExplorationStage
from explot.stages.profiling.stage import ProfilingStage


def run_exploration_stage(df):
    config = load_config(Path("config/default.yaml"))
    hooks = HookRegistry(budget_mode=config.budget.mode)
    state = PipelineState(raw_df=df)
    state.results["profiling"] = ProfilingStage().run(state, config, hooks)
    return ExplorationStage().run(state, config, hooks)


def test_exploration_finds_scrna_redundant_pair(scrna_data) -> None:
    df, metadata = scrna_data
    result = run_exploration_stage(df)

    pairs = {tuple(item["columns"]) for item in result.outputs["redundant_pairs"]}
    expected = {tuple(item) for item in metadata["exploration"]["expected_redundant_pairs"]}
    assert expected & pairs


def test_exploration_marks_proteomics_missingness_structured(proteomics_data) -> None:
    df, metadata = proteomics_data
    result = run_exploration_stage(df)

    assert result.outputs["missingness_type"] == metadata["exploration"]["missingness_type"]


def test_exploration_marks_tabular_missingness_minimal(tabular_data) -> None:
    df, metadata = tabular_data
    result = run_exploration_stage(df)

    assert result.outputs["missingness_type"] == metadata["exploration"]["missingness_type"]


def test_exploration_generates_figures(scrna_data) -> None:
    df, _ = scrna_data
    result = run_exploration_stage(df)

    assert "correlation_heatmap" in result.figures
    assert "<svg" in result.figures["distribution_overview"]
    assert "hopkins_statistic" in result.interpretations
    assert "distribution_overview" in result.interpretations


def test_exploration_exports_sorted_integer_outlier_rows(scrna_data) -> None:
    df, _ = scrna_data
    result = run_exploration_stage(df)

    outlier_rows = result.outputs["outlier_rows"]
    assert isinstance(outlier_rows, list)
    assert outlier_rows == sorted(outlier_rows)
    assert all(isinstance(idx, int) for idx in outlier_rows)
    assert all(0 <= idx < len(df) for idx in outlier_rows)
    assert "row_outliers" in result.interpretations


def test_exploration_recovers_planted_extreme_row_outlier() -> None:
    rows = []
    for idx in range(99):
        rows.append({"f1": float(idx % 3), "f2": float((idx + 1) % 4), "group": "baseline"})
    rows.append({"f1": 250.0, "f2": 250.0, "group": "extreme"})
    df = pd.DataFrame(rows)

    result = run_exploration_stage(df)

    assert result.outputs["outlier_rows"] == [99]


def test_exploration_builds_grouped_distribution_plot() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "value_a": np.concatenate([rng.normal(0, 1, 40), rng.normal(4, 1, 40)]),
            "value_b": np.concatenate([rng.normal(1, 1, 40), rng.normal(5, 1, 40)]),
            "group": ["left"] * 40 + ["right"] * 40,
        }
    )

    result = run_exploration_stage(df)

    grouped = result.outputs["grouped_distributions"]
    assert grouped
    assert grouped[0]["group_column"] == "group"
    assert "<svg" in result.figures["grouped_distributions"]
