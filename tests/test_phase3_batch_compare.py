"""Phase 3a/3b validation tests — batch mode and run comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from explot.batch import collect_paths, run_batch, write_index
from explot.cli import main
from explot.compare import diff_reports, diff_to_markdown
from explot.config import load_config

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _config():
    return load_config(_PACKAGE_ROOT / "config" / "fast.yaml")


def _make_csv(path: Path, seed: int = 0, n: int = 200) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "f0": rng.normal(size=n),
        "f1": rng.normal(size=n),
        "target": rng.integers(0, 2, size=n),
    })
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Phase 3a — collect_paths
# ---------------------------------------------------------------------------

def test_collect_paths_single_file(workspace_tmp_path):
    p = _make_csv(workspace_tmp_path / "data.csv")
    result = collect_paths(str(p))
    assert result == [p]


def test_collect_paths_directory(workspace_tmp_path):
    for i in range(3):
        _make_csv(workspace_tmp_path / f"ds{i}.csv", seed=i)
    result = collect_paths(str(workspace_tmp_path))
    assert len(result) == 3
    assert all(p.suffix == ".csv" for p in result)


def test_collect_paths_glob(workspace_tmp_path):
    for i in range(4):
        _make_csv(workspace_tmp_path / f"ds{i}.csv", seed=i)
    (workspace_tmp_path / "notes.txt").write_text("ignore me")
    result = collect_paths(str(workspace_tmp_path / "*.csv"))
    assert len(result) == 4


def test_collect_paths_recursive(workspace_tmp_path):
    sub = workspace_tmp_path / "subdir"
    sub.mkdir()
    _make_csv(workspace_tmp_path / "root.csv", seed=0)
    _make_csv(sub / "nested.csv", seed=1)
    result = collect_paths(str(workspace_tmp_path), recursive=True)
    names = {p.name for p in result}
    assert "root.csv" in names
    assert "nested.csv" in names


def test_collect_paths_empty_returns_empty(workspace_tmp_path):
    result = collect_paths(str(workspace_tmp_path / "*.csv"))
    assert result == []


def test_collect_paths_ignores_unsupported_extensions(workspace_tmp_path):
    (workspace_tmp_path / "readme.md").write_text("x")
    (workspace_tmp_path / "image.png").write_bytes(b"x")
    _make_csv(workspace_tmp_path / "data.csv")
    result = collect_paths(str(workspace_tmp_path))
    assert len(result) == 1
    assert result[0].name == "data.csv"


# ---------------------------------------------------------------------------
# Phase 3a — run_batch + write_index
# ---------------------------------------------------------------------------

def test_run_batch_produces_one_report_per_file(workspace_tmp_path):
    data_dir = workspace_tmp_path / "data"
    data_dir.mkdir()
    out_dir = workspace_tmp_path / "reports"

    for i in range(3):
        _make_csv(data_dir / f"ds{i}.csv", seed=i)

    paths = collect_paths(str(data_dir))
    results = run_batch(paths, out_dir, _config(), verbose=False)

    assert len(results) == 3
    assert out_dir.exists()
    html_files = list(out_dir.glob("*.html"))
    assert len(html_files) == 3
    assert all(r["error"] is None for r in results)


def test_run_batch_json_format(workspace_tmp_path):
    data_dir = workspace_tmp_path / "data"
    data_dir.mkdir()
    out_dir = workspace_tmp_path / "out"

    for i in range(2):
        _make_csv(data_dir / f"ds{i}.csv", seed=i)

    paths = collect_paths(str(data_dir))
    run_batch(paths, out_dir, _config(), verbose=False, output_format="json")
    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == 2
    # Verify parseable JSON
    for jf in json_files:
        parsed = json.loads(jf.read_text())
        assert "stages" in parsed


def test_write_index_creates_index_md(workspace_tmp_path):
    data_dir = workspace_tmp_path / "data"
    data_dir.mkdir()
    out_dir = workspace_tmp_path / "reports"

    for i in range(3):
        _make_csv(data_dir / f"ds{i}.csv", seed=i)

    paths = collect_paths(str(data_dir))
    results = run_batch(paths, out_dir, _config(), verbose=False)
    index_path = write_index(results, out_dir)

    assert index_path.exists()
    md = index_path.read_text(encoding="utf-8")
    assert "# Explot Batch Report" in md
    assert "## Summary" in md
    # All three file names should appear in the table
    for i in range(3):
        assert f"ds{i}.csv" in md


def test_write_index_contains_verdict_and_score(workspace_tmp_path):
    data_dir = workspace_tmp_path / "data"
    data_dir.mkdir()
    out_dir = workspace_tmp_path / "reports"

    _make_csv(data_dir / "clean.csv", seed=0)
    paths = collect_paths(str(data_dir))
    results = run_batch(paths, out_dir, _config(), verbose=False)
    md = write_index(results, out_dir).read_text(encoding="utf-8")

    # Table must contain verdict icon and a quality score
    assert any(icon in md for icon in ("✅", "⚠️", "🛑", "ℹ️"))
    assert "/100" in md   # quality score


def test_write_index_marks_failed_runs(workspace_tmp_path):
    out_dir = workspace_tmp_path / "out"
    out_dir.mkdir()
    # Inject a fake failed result
    results = [{"file": Path("broken.csv"), "output": out_dir / "broken.html",
                "state": None, "error": "load failed"}]
    md = write_index(results, out_dir).read_text(encoding="utf-8")
    assert "ERROR" in md
    assert "broken.csv" in md


# ---------------------------------------------------------------------------
# Phase 3a — CLI batch mode
# ---------------------------------------------------------------------------

def test_cli_batch_via_directory(workspace_tmp_path):
    data_dir = workspace_tmp_path / "data"
    data_dir.mkdir()
    out_dir = workspace_tmp_path / "reports"

    for i in range(2):
        _make_csv(data_dir / f"ds{i}.csv", seed=i)

    rc = main(["run", str(data_dir), "--output", str(out_dir), "--fast", "--quiet"])
    assert rc == 0
    assert (out_dir / "index.md").exists()


def test_cli_backward_compat_single_file(workspace_tmp_path):
    """explot data.csv still works without the 'run' subcommand."""
    csv_path = _make_csv(workspace_tmp_path / "data.csv")
    out_path = workspace_tmp_path / "report.html"
    rc = main([str(csv_path), "--output", str(out_path), "--fast", "--quiet"])
    assert rc == 0
    assert out_path.exists()


def test_cli_batch_markdown_format(workspace_tmp_path):
    data_dir = workspace_tmp_path / "data"
    data_dir.mkdir()
    out_dir = workspace_tmp_path / "out"

    for i in range(2):
        _make_csv(data_dir / f"ds{i}.csv", seed=i)

    rc = main(["run", str(data_dir), "--output", str(out_dir),
               "--fast", "--quiet", "--markdown"])
    assert rc == 0
    md_files = list(out_dir.glob("*.md"))
    # two per-file reports + index
    assert len(md_files) == 3


def test_cli_run_no_files_returns_error(workspace_tmp_path):
    empty_dir = workspace_tmp_path / "empty"
    empty_dir.mkdir()
    rc = main(["run", str(empty_dir), "--fast", "--quiet"])
    assert rc == 1


# ---------------------------------------------------------------------------
# Phase 3b — compare: diff_reports
# ---------------------------------------------------------------------------

def _fake_report(quality: int, target: str, model: str, score: float,
                 flags: list[str], feats: list[str]) -> dict:
    return {
        "stages": {
            "profiling": {"outputs": {"quality_score": quality}},
            "supervised": {
                "outputs": {
                    "best_models": {
                        target: {
                            "model": model,
                            "mean": score,
                            "trust_flags": flags,
                        }
                    },
                    "feature_importances": {
                        target: {
                            "track_c": [{"feature": f, "importance": 0.1} for f in feats]
                        }
                    },
                }
            },
            "findings": {
                "outputs": {
                    "verdict": {"decision": "SHIP" if not flags else "INVESTIGATE"}
                }
            },
        }
    }


def test_diff_quality_delta():
    a = _fake_report(72, "income", "RandomForest", 0.702, [], ["age", "hours"])
    b = _fake_report(85, "income", "LightGBM", 0.769, [], ["age", "hours", "sex"])
    diff = diff_reports(a, b)
    assert diff["quality"]["delta"] == 13


def test_diff_score_delta():
    a = _fake_report(72, "income", "RandomForest", 0.702, [], [])
    b = _fake_report(72, "income", "LightGBM", 0.769, [], [])
    diff = diff_reports(a, b)
    t = diff["targets"][0]
    assert t["score_delta"] == pytest.approx(0.067, abs=0.001)
    assert t["model_a"] == "RandomForest"
    assert t["model_b"] == "LightGBM"


def test_diff_flags_added():
    a = _fake_report(80, "y", "RF", 0.8, [], [])
    b = _fake_report(80, "y", "RF", 0.8, ["group_leakage"], [])
    diff = diff_reports(a, b)
    t = diff["targets"][0]
    assert "group_leakage" in t["flags_added"]
    assert t["flags_removed"] == []


def test_diff_flags_removed():
    a = _fake_report(80, "y", "RF", 0.8, ["heavy_imputation"], [])
    b = _fake_report(80, "y", "RF", 0.8, [], [])
    diff = diff_reports(a, b)
    t = diff["targets"][0]
    assert "heavy_imputation" in t["flags_removed"]
    assert t["flags_added"] == []


def test_diff_features_added():
    a = _fake_report(80, "y", "RF", 0.8, [], ["age"])
    b = _fake_report(80, "y", "RF", 0.8, [], ["age", "sex", "occupation"])
    diff = diff_reports(a, b)
    assert "sex" in diff["features"]["added"]
    assert "occupation" in diff["features"]["added"]
    assert diff["features"]["removed"] == []


def test_diff_verdict_changed():
    a = _fake_report(80, "y", "RF", 0.8, [], [])
    b = _fake_report(80, "y", "RF", 0.8, ["group_leakage"], [])
    diff = diff_reports(a, b)
    assert diff["verdict"]["changed"] is True
    assert diff["verdict"]["a"] == "SHIP"


def test_diff_verdict_unchanged():
    a = _fake_report(80, "y", "RF", 0.8, [], [])
    b = _fake_report(85, "y", "LightGBM", 0.85, [], [])
    diff = diff_reports(a, b)
    assert diff["verdict"]["changed"] is False


# ---------------------------------------------------------------------------
# Phase 3b — compare: diff_to_markdown
# ---------------------------------------------------------------------------

def test_diff_to_markdown_structure():
    a = _fake_report(72, "income", "RandomForest", 0.702, [], ["age"])
    b = _fake_report(85, "income", "LightGBM", 0.769, [], ["age", "sex"])
    diff = diff_reports(a, b)
    md = diff_to_markdown(diff, label_a="before", label_b="after")

    assert "# Explot Comparison: before → after" in md
    assert "## Quality Score" in md
    assert "## Best Models" in md
    assert "+13" in md        # quality delta
    assert "+0.067" in md     # score delta
    assert "sex" in md        # added feature


def test_diff_to_markdown_flag_changes():
    a = _fake_report(80, "y", "RF", 0.8, ["heavy_imputation"], [])
    b = _fake_report(80, "y", "RF", 0.85, [], [])
    diff = diff_reports(a, b)
    md = diff_to_markdown(diff)
    assert "## Trust Flag Changes" in md
    assert "heavy_imputation" in md
    assert "Removed" in md


def test_diff_to_markdown_no_changes():
    a = _fake_report(80, "y", "RF", 0.8, [], ["age"])
    b = _fake_report(80, "y", "RF", 0.8, [], ["age"])
    diff = diff_reports(a, b)
    md = diff_to_markdown(diff)
    assert "No trust flag changes." in md
    assert "No feature changes detected." in md


# ---------------------------------------------------------------------------
# Phase 3b — CLI compare command
# ---------------------------------------------------------------------------

def test_cli_compare_prints_markdown(workspace_tmp_path, capsys):
    rng = np.random.default_rng(0)
    n = 300

    def _run_and_export(seed, out_json):
        from explot.orchestrator import Pipeline
        df = pd.DataFrame({
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        })
        csv = workspace_tmp_path / f"data_{seed}.csv"
        df.to_csv(csv, index=False)
        config = _config()
        state = Pipeline(config=config).run(csv, output_path=None, verbose=False,
                                            target_column="target")
        from explot.export import state_to_json
        out_json.write_text(state_to_json(state), encoding="utf-8")

    json_a = workspace_tmp_path / "v1.json"
    json_b = workspace_tmp_path / "v2.json"
    _run_and_export(1, json_a)
    _run_and_export(2, json_b)

    rc = main(["compare", str(json_a), str(json_b)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "# Explot Comparison" in captured.out
    assert "## Best Models" in captured.out


def test_cli_compare_writes_output_file(workspace_tmp_path):
    rng = np.random.default_rng(5)
    n = 300

    def _export(seed):
        from explot.orchestrator import Pipeline
        df = pd.DataFrame({
            "f0": rng.normal(size=n), "f1": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        })
        csv = workspace_tmp_path / f"d{seed}.csv"
        df.to_csv(csv, index=False)
        out = workspace_tmp_path / f"r{seed}.json"
        state = Pipeline(config=_config()).run(csv, output_path=None, verbose=False,
                                               target_column="target")
        from explot.export import state_to_json
        out.write_text(state_to_json(state), encoding="utf-8")
        return out

    a = _export(1)
    b = _export(2)
    diff_out = workspace_tmp_path / "diff.md"

    rc = main(["compare", str(a), str(b), "--output", str(diff_out)])
    assert rc == 0
    assert diff_out.exists()
    assert "# Explot Comparison" in diff_out.read_text()
