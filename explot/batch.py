"""Batch pipeline runner and index report generator.

Usage (internal):
    results = run_batch(paths, output_dir, config, verbose=True)
    write_index(results, output_dir)
"""
from __future__ import annotations

import sys
from pathlib import Path

from explot.config import AppConfig
from explot.orchestrator import Pipeline
from explot.state import PipelineState

_SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xls", ".xlsx", ".parquet"}

_VERDICT_ICON = {
    "SHIP": "✅",
    "INVESTIGATE": "⚠️",
    "DO_NOT_SHIP": "🛑",
    "NO_MODEL": "ℹ️",
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def collect_paths(input_arg: str, recursive: bool = False) -> list[Path]:
    """Resolve a file path, directory, or glob pattern to a sorted list of data files."""
    p = Path(input_arg)

    if p.is_file():
        return [p]

    if p.is_dir():
        pattern = "**/*" if recursive else "*"
        return sorted(
            f for f in p.glob(pattern)
            if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
        )

    # Glob pattern (contains * ? [)
    parent = p.parent if p.parent != Path(".") else Path(".")
    glob_pattern = p.name
    if recursive:
        glob_pattern = "**/" + glob_pattern
    return sorted(
        f for f in parent.glob(glob_pattern)
        if f.is_file() and f.suffix.lower() in _SUPPORTED_SUFFIXES
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    paths: list[Path],
    output_dir: Path,
    config: AppConfig,
    verbose: bool = True,
    target_column: str | None = None,
    task_type: str | None = None,
    output_format: str = "html",
) -> list[dict]:
    """Run the pipeline on each path; write one report per file.

    Returns a list of result dicts, one per file, for use by write_index.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(config=config)
    results = []

    for i, path in enumerate(paths, 1):
        stem = path.stem
        suffix = ".json" if output_format == "json" else ".md" if output_format == "markdown" else ".html"
        out_path = output_dir / f"{stem}{suffix}"

        if verbose:
            print(f"  [{i}/{len(paths)}] {path.name} → {out_path.name}", file=sys.stderr)

        entry: dict = {"file": path, "output": out_path, "state": None, "error": None}
        try:
            state = pipeline.run(
                path,
                output_path=out_path if output_format == "html" else None,
                verbose=False,
                target_column=target_column,
                task_type=task_type,
            )
            if output_format == "json":
                from explot.export import state_to_json
                out_path.write_text(state_to_json(state), encoding="utf-8")
            elif output_format == "markdown":
                from explot.report.markdown import state_to_markdown
                out_path.write_text(state_to_markdown(state), encoding="utf-8")
            entry["state"] = state
        except Exception as exc:
            entry["error"] = str(exc)
            if verbose:
                print(f"    ERROR: {exc}", file=sys.stderr)

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# index.md generator
# ---------------------------------------------------------------------------

def write_index(results: list[dict], output_dir: Path) -> Path:
    """Write a summary index.md comparing all batch run results."""
    lines: list[str] = ["# Explot Batch Report", ""]
    lines.append(f"Analyzed **{len(results)}** file(s).")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| File | Rows × Cols | Quality | Verdict | Best Model | Score | Top Finding |")
    lines.append("|---|---|---|---|---|---|---|")

    detail_blocks: list[str] = []

    for entry in results:
        fname = entry["file"].name
        state: PipelineState | None = entry["state"]

        if entry["error"] or state is None:
            lines.append(f"| `{fname}` | — | — | ❌ ERROR | — | — | {entry.get('error', 'unknown')} |")
            continue

        # Shape
        n_rows, n_cols = state.raw_df.shape
        shape_str = f"{n_rows:,} × {n_cols}"

        # Quality
        profiling = state.results.get("profiling")
        quality = profiling.outputs.get("quality_score") if profiling and profiling.success else None
        quality_str = f"{quality}/100" if quality is not None else "—"

        # Verdict
        findings = state.results.get("findings")
        verdict = findings.outputs.get("verdict") if findings and findings.success else None
        if verdict:
            decision = verdict.get("decision", "?")
            icon = _VERDICT_ICON.get(decision, "•")
            verdict_str = f"{icon} {decision}"
        else:
            verdict_str = "—"

        # Best model
        supervised = state.results.get("supervised")
        best_models = supervised.outputs.get("best_models", {}) if supervised and supervised.success else {}
        if best_models:
            # Pick target with highest score
            best_target, best_info = max(best_models.items(), key=lambda kv: kv[1].get("mean", 0))
            model_str = best_info.get("model", "?")
            score_str = f"{best_info.get('mean', 0):.3f}"
        else:
            model_str = "—"
            score_str = "—"

        # Top finding
        finding_list = findings.outputs.get("findings_list", []) if findings and findings.success else []
        top_finding = finding_list[0]["text"][:60] + "…" if finding_list else "—"

        lines.append(
            f"| `{fname}` | {shape_str} | {quality_str} | {verdict_str} | "
            f"{model_str} | {score_str} | {top_finding} |"
        )

        # Per-file detail block
        block: list[str] = [f"### `{fname}`", ""]
        if verdict and verdict.get("reasons"):
            for r in verdict["reasons"][:3]:
                block.append(f"- {r}")
            block.append("")

        if finding_list:
            for f in finding_list[:3]:
                conf = f["confidence"]
                block.append(f"- **[{conf}]** {f['text']}")
            block.append("")

        detail_blocks.append("\n".join(block))

    if detail_blocks:
        lines.append("")
        lines.append("## Per-File Details")
        lines.append("")
        lines.extend(detail_blocks)

    index_path = output_dir / "index.md"
    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return index_path
