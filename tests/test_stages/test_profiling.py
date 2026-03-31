from __future__ import annotations

from pathlib import Path

from explot.config import load_config
from explot.hooks import HookRegistry
from explot.state import PipelineState
from explot.stages.profiling.stage import ProfilingStage


def run_profiling_stage(df):
    config = load_config(Path("config/default.yaml"))
    state = PipelineState(raw_df=df)
    hooks = HookRegistry(budget_mode=config.budget.mode)
    return ProfilingStage().run(state, config, hooks)


def test_profiling_detects_raw_counts(scrna_data) -> None:
    df, metadata = scrna_data
    result = run_profiling_stage(df)

    assert result.outputs["normalization_guess"] == metadata["profiling"]["expected_normalization"]


def test_profiling_detects_log_normalized(proteomics_data) -> None:
    df, metadata = proteomics_data
    result = run_profiling_stage(df)

    assert result.outputs["normalization_guess"] == metadata["profiling"]["expected_normalization"]


def test_profiling_flags_suspicious_columns_from_metadata(scrna_data) -> None:
    df, metadata = scrna_data
    result = run_profiling_stage(df)

    flagged = {(entry["name"], entry["reason"]) for entry in result.outputs["suspicious_columns"]}
    expected = {
        (entry["name"], entry["reason"]) for entry in metadata["profiling"]["suspicious_columns"]
    }
    assert expected.issubset(flagged)


def test_profiling_quality_score_in_expected_range(proteomics_data) -> None:
    df, metadata = proteomics_data
    result = run_profiling_stage(df)

    low, high = metadata["profiling"]["expected_quality_score_range"]
    assert low <= result.outputs["quality_score"] <= high


def test_profiling_interpretations_include_expected_keywords(scrna_data) -> None:
    df, metadata = scrna_data
    result = run_profiling_stage(df)

    for key, expected_words in metadata["interpretation_checks"].items():
        interpretation = result.interpretations.get(key, "")
        for word in expected_words:
            assert word.lower() in interpretation.lower()


def test_profiling_assigns_role_guesses_for_scrna(scrna_data) -> None:
    df, _ = scrna_data
    result = run_profiling_stage(df)

    assert result.outputs["column_profiles"]["cell_id"]["role_guess"] == "id_like"
    assert result.outputs["column_profiles"]["gene_1"]["role_guess"] == "count_like"


def test_profiling_tracks_zero_stats_for_numeric_columns(scrna_data) -> None:
    df, _ = scrna_data
    result = run_profiling_stage(df)

    profile = result.outputs["column_profiles"]["gene_1"]
    assert "zero_count" in profile
    assert "zero_percent" in profile
