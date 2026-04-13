from __future__ import annotations

import argparse
import sys
from pathlib import Path

from explot.config import load_config
from explot.orchestrator import Pipeline

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _resolve_config(path: Path) -> Path:
    """Resolve config path: try as-is first, then relative to package root."""
    if path.exists():
        return path
    fallback = _PACKAGE_ROOT / path
    if fallback.exists():
        return fallback
    return path  # let load_config raise the error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="explot",
        description="Analyze a tabular dataset and generate an HTML summary report.",
    )
    parser.add_argument("input_path", type=Path, help="Path to CSV, TSV, Excel, or parquet data.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("report.html"),
        help="Output report path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Configuration file path.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast-mode overrides from config/fast.yaml.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of HTML.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Explicitly specify the target column for supervised modeling.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Task type for the target column (default: auto-detect).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = args.input_path
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1
    if input_path.suffix.lower() not in {".csv", ".tsv", ".txt", ".xls", ".xlsx", ".parquet"}:
        print(f"Error: unsupported file type: {input_path.suffix}", file=sys.stderr)
        return 1

    config_path = Path("config/fast.yaml") if args.fast else args.config
    config_path = _resolve_config(config_path)
    verbose = not args.quiet

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    pipeline = Pipeline(config=config)

    if verbose:
        mode = "fast" if args.fast else "full"
        print(f"explot: analyzing {input_path.name} ({mode} mode)", file=sys.stderr)

    target_col = args.target
    task_type = args.task

    try:
        if args.json:
            output_path = args.output.with_suffix(".json") if args.output.suffix != ".json" else args.output
            state = pipeline.run(args.input_path, output_path=None, verbose=verbose,
                                 target_column=target_col, task_type=task_type)
            from explot.export import state_to_json
            output_path.write_text(state_to_json(state), encoding="utf-8")
        else:
            output_path = args.output
            state = pipeline.run(args.input_path, output_path=args.output, verbose=verbose,
                                 target_column=target_col, task_type=task_type)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if verbose:
        n_stages = sum(1 for r in state.results.values() if r.success)
        total_time = sum(r.duration_seconds for r in state.results.values() if r.duration_seconds)
        print(f"  {n_stages} stages completed in {total_time:.1f}s", file=sys.stderr)
        print(f"  Report written to {output_path}", file=sys.stderr)
    return 0

