from pathlib import Path

import pandas as pd

from explot.config import load_config
from explot.orchestrator import Pipeline


def test_pipeline_runs_profiling_and_writes_report(workspace_tmp_path: Path) -> None:
    data_path = workspace_tmp_path / "data.csv"
    report_path = workspace_tmp_path / "report.html"
    pd.DataFrame({"value": [1, 2, 3], "group": ["a", "a", "b"]}).to_csv(data_path, index=False)

    config = load_config(Path("config/default.yaml"))
    state = Pipeline(config=config).run(data_path, output_path=report_path)

    assert "profiling" in state.results
    assert state.results["profiling"].success is True
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "Explot Report" in report_text
    assert "profiling, exploration, dimensionality reduction" in report_text
