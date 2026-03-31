from __future__ import annotations

import argparse
from pathlib import Path

from simulator.proteomics import ProteomicsSimulator
from simulator.scrna import ScrnaSimulator
from simulator.tabular import TabularSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Explot simulator datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulator/output"),
        help="Directory where simulator outputs will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    simulators = [TabularSimulator(), ScrnaSimulator(), ProteomicsSimulator()]
    for simulator in simulators:
        simulator.save(args.output_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
