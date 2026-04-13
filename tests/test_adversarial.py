"""Adversarial / edge-case datasets: the pipeline must degrade gracefully, not crash."""
from pathlib import Path

import numpy as np
import pandas as pd

from explot.config import load_config
from explot.orchestrator import Pipeline


def _run(data: pd.DataFrame, workspace_tmp_path: Path):
    data_path = workspace_tmp_path / "data.csv"
    report_path = workspace_tmp_path / "report.html"
    data.to_csv(data_path, index=False)
    config = load_config(Path("config/fast.yaml"))
    state = Pipeline(config=config).run(data_path, output_path=report_path)
    assert report_path.exists()
    return state


def test_pipeline_survives_all_null_dataframe(workspace_tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [np.nan] * 20, "b": [np.nan] * 20, "c": [np.nan] * 20})
    state = _run(df, workspace_tmp_path)
    assert "profiling" in state.results
    assert state.results["profiling"].success is True


def test_pipeline_survives_all_constant_columns(workspace_tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1] * 30, "y": ["same"] * 30, "z": [0.5] * 30})
    state = _run(df, workspace_tmp_path)
    assert state.results["profiling"].success is True


def test_pipeline_survives_tiny_dataset(workspace_tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "label": ["x", "y", "x"]})
    state = _run(df, workspace_tmp_path)
    assert state.results["profiling"].success is True


def test_pipeline_survives_single_row(workspace_tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1], "b": [2.0], "c": ["only"]})
    state = _run(df, workspace_tmp_path)
    assert state.results["profiling"].success is True


def test_pipeline_survives_single_column(workspace_tmp_path: Path) -> None:
    df = pd.DataFrame({"only": list(range(50))})
    state = _run(df, workspace_tmp_path)
    assert state.results["profiling"].success is True
