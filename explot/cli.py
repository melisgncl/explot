from __future__ import annotations

import argparse
import sys
from pathlib import Path

from explot.config import load_config

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xls", ".xlsx", ".parquet"}


def _resolve_config(path: Path) -> Path:
    if path.exists():
        return path
    fallback = _PACKAGE_ROOT / path
    if fallback.exists():
        return fallback
    return path


# ---------------------------------------------------------------------------
# Parser builders
# ---------------------------------------------------------------------------

def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    """Shared flags for both single-file and batch run modes."""
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--fast", action="store_true", help="Use fast-mode config.")
    p.add_argument("--json", action="store_true", help="Output JSON instead of HTML.")
    p.add_argument("--markdown", action="store_true", help="Output Markdown instead of HTML.")
    p.add_argument("--target", type=str, default=None, help="Target column for supervised modeling.")
    p.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression", "auto"],
        default="auto",
    )
    p.add_argument("-q", "--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="explot",
        description="Analyze tabular datasets and generate reports.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # run  (also the implicit default when no subcommand is given)
    # ------------------------------------------------------------------
    run_p = subparsers.add_parser(
        "run",
        help="Analyze one or more datasets.",
        description=(
            "Analyze a single file, a directory, or a glob pattern. "
            "When multiple files are resolved, reports go into --output dir "
            "and an index.md summary is generated."
        ),
    )
    run_p.add_argument(
        "input",
        type=str,
        help="File path, directory, or glob pattern (e.g. data/*.csv).",
    )
    run_p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help=(
            "Output path. For single file: path to report (default report.html). "
            "For batch: output directory (default ./reports/)."
        ),
    )
    run_p.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when input is a directory or glob.",
    )
    _add_common_run_args(run_p)

    # ------------------------------------------------------------------
    # compare
    # ------------------------------------------------------------------
    cmp_p = subparsers.add_parser(
        "compare",
        help="Diff two explot JSON reports.",
        description=(
            "Compare two pipeline runs exported as JSON. "
            "Shows score deltas, trust flag changes, and feature changes."
        ),
    )
    cmp_p.add_argument("report_a", type=Path, help="First JSON report.")
    cmp_p.add_argument("report_b", type=Path, help="Second JSON report.")
    cmp_p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Write markdown diff to this file (default: print to stdout).",
    )
    cmp_p.add_argument(
        "--label-a", type=str, default=None,
        help="Label for the first report (default: filename stem).",
    )
    cmp_p.add_argument(
        "--label-b", type=str, default=None,
        help="Label for the second report (default: filename stem).",
    )

    return parser


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

def _output_format(args) -> str:
    if args.json:
        return "json"
    if args.markdown:
        return "markdown"
    return "html"


def _main_run(args) -> int:
    from explot.batch import collect_paths, run_batch, write_index
    from explot.orchestrator import Pipeline

    config_path = Path("config/fast.yaml") if args.fast else args.config
    config_path = _resolve_config(config_path)
    verbose = not args.quiet
    fmt = _output_format(args)
    target_col = args.target
    task_type = args.task if args.task != "auto" else None

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        return 1

    paths = collect_paths(args.input, recursive=getattr(args, "recursive", False))
    if not paths:
        print(f"Error: no supported data files found for '{args.input}'", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Single-file mode
    # ------------------------------------------------------------------
    if len(paths) == 1:
        input_path = paths[0]
        if input_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            print(f"Error: unsupported file type: {input_path.suffix}", file=sys.stderr)
            return 1

        if args.output is None:
            suffix = ".json" if fmt == "json" else ".md" if fmt == "markdown" else ".html"
            output_path = Path("report" + suffix)
        else:
            output_path = args.output

        if verbose:
            mode = "fast" if args.fast else "full"
            print(f"explot: analyzing {input_path.name} ({mode} mode)", file=sys.stderr)

        try:
            pipeline = Pipeline(config=config)
            if fmt == "json":
                output_path = output_path.with_suffix(".json")
                state = pipeline.run(input_path, output_path=None, verbose=verbose,
                                     target_column=target_col, task_type=task_type)
                from explot.export import state_to_json
                output_path.write_text(state_to_json(state), encoding="utf-8")
            elif fmt == "markdown":
                output_path = output_path.with_suffix(".md")
                state = pipeline.run(input_path, output_path=None, verbose=verbose,
                                     target_column=target_col, task_type=task_type)
                from explot.report.markdown import state_to_markdown
                output_path.write_text(state_to_markdown(state), encoding="utf-8")
            else:
                state = pipeline.run(input_path, output_path=output_path, verbose=verbose,
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

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    output_dir = args.output if args.output is not None else Path("reports")
    if verbose:
        mode = "fast" if args.fast else "full"
        print(f"explot: batch mode — {len(paths)} file(s) → {output_dir}/ ({mode} mode)", file=sys.stderr)

    try:
        results = run_batch(
            paths, output_dir, config,
            verbose=verbose,
            target_column=target_col,
            task_type=task_type,
            output_format=fmt,
        )
        index_path = write_index(results, output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if verbose:
        n_ok = sum(1 for r in results if r["error"] is None)
        n_err = len(results) - n_ok
        status = f"{n_ok} succeeded" + (f", {n_err} failed" if n_err else "")
        print(f"  {status} — index written to {index_path}", file=sys.stderr)

    return 1 if any(r["error"] for r in results) else 0


# ---------------------------------------------------------------------------
# compare subcommand
# ---------------------------------------------------------------------------

def _main_compare(args) -> int:
    from explot.compare import load_report, diff_reports, diff_to_markdown

    for p in (args.report_a, args.report_b):
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            return 1
        if p.suffix.lower() != ".json":
            print(f"Error: expected a JSON report, got: {p}", file=sys.stderr)
            return 1

    try:
        report_a = load_report(args.report_a)
        report_b = load_report(args.report_b)
    except Exception as exc:
        print(f"Error reading reports: {exc}", file=sys.stderr)
        return 1

    label_a = args.label_a or args.report_a.stem
    label_b = args.label_b or args.report_b.stem

    try:
        diff = diff_reports(report_a, report_b)
        md = diff_to_markdown(diff, label_a=label_a, label_b=label_b)
    except Exception as exc:
        print(f"Error comparing reports: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(md, encoding="utf-8")
        print(f"Diff written to {args.output}", file=sys.stderr)
    else:
        print(md)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]

    # Backward compat: `explot data.csv` with no subcommand → inject "run"
    if raw and raw[0] not in ("run", "compare", "-h", "--help"):
        raw = ["run"] + raw

    parser = build_parser()
    args = parser.parse_args(raw)

    if args.command == "compare":
        return _main_compare(args)

    # "run" (explicit or implicit)
    if not hasattr(args, "input"):
        parser.print_help()
        return 1

    return _main_run(args)
