from __future__ import annotations

import argparse
from pathlib import Path

from explot.config import load_config
from explot.orchestrator import Pipeline


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path("config/fast.yaml") if args.fast else args.config
    config = load_config(config_path)
    pipeline = Pipeline(config=config)

    if args.json:
        output_path = args.output.with_suffix(".json") if args.output.suffix != ".json" else args.output
        state = pipeline.run(args.input_path, output_path=None)
        from explot.export import state_to_json
        output_path.write_text(state_to_json(state), encoding="utf-8")
    else:
        pipeline.run(args.input_path, output_path=args.output)
    return 0

