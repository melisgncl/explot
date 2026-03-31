from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    enabled_stages: tuple[str, ...]
    fail_fast: bool = False
    cache_state: bool = False

    def is_enabled(self, stage_name: str) -> bool:
        return not self.enabled_stages or stage_name in self.enabled_stages


@dataclass(frozen=True)
class ReportConfig:
    title: str = "Explot Report"
    include_debug: bool = False


@dataclass(frozen=True)
class BudgetConfig:
    mode: str = "full"


@dataclass(frozen=True)
class AppConfig:
    pipeline: PipelineConfig
    report: ReportConfig
    budget: BudgetConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def load_config(path: Path) -> AppConfig:
    raw = _read_yaml(path)
    pipeline_raw = raw.get("pipeline", {})
    report_raw = raw.get("report", {})
    budget_raw = raw.get("budget", {})

    return AppConfig(
        pipeline=PipelineConfig(
            enabled_stages=tuple(pipeline_raw.get("enabled_stages", [])),
            fail_fast=bool(pipeline_raw.get("fail_fast", False)),
            cache_state=bool(pipeline_raw.get("cache_state", False)),
        ),
        report=ReportConfig(
            title=str(report_raw.get("title", "Explot Report")),
            include_debug=bool(report_raw.get("include_debug", False)),
        ),
        budget=BudgetConfig(mode=str(budget_raw.get("mode", "full"))),
    )

