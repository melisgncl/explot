from pathlib import Path

from explot.config import load_config
from explot.orchestrator import Pipeline


def test_report_renders_supervised_track_comparison_and_confusion(workspace_tmp_path):
    report_path = workspace_tmp_path / "report.html"
    state = Pipeline(load_config(Path("config/fast.yaml"))).run(
        Path("data/sleep_health_dataset.csv"),
        output_path=report_path,
    )

    assert state.results["supervised"].success is True
    html = report_path.read_text(encoding="utf-8")
    assert "DVAE" in html
    assert "Track A: Original Features" in html
    assert "Track B: Latent Features" in html
    assert "True \\ Pred" in html
    assert "Top Numeric Feature Distributions" in html
    assert "Isolation Forest vs DVAE vs Stage 2" in html
