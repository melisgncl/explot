from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

from explot.cache import save_state
from explot.config import AppConfig
from explot.hooks import HookRegistry
from explot.loader import load_table
from explot.report.generator import ReportGenerator
from explot.state import PipelineState
from explot.stages.base import StageMeta, StageResult


class Pipeline:
    def __init__(self, config: AppConfig):
        self.config = config

    def load_manifest(self) -> list[dict[str, Any]]:
        manifest_path = Path(__file__).resolve().parent / "stages" / "manifest.yaml"
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return list(raw.get("stages", []))

    def load_stage(self, entry: dict[str, Any]):
        module = import_module(entry["module"])
        stage_class = getattr(module, entry["class_name"])
        return stage_class()

    def run(self, input_path: Path, output_path: Path | None = None) -> PipelineState:
        state = PipelineState(raw_df=load_table(input_path))
        hooks = HookRegistry(budget_mode=self.config.budget.mode)

        for entry in self.load_manifest():
            stage_name = entry["name"]
            if not self.config.pipeline.is_enabled(stage_name):
                continue

            stage = self.load_stage(entry)
            missing = stage.validate_inputs(state)
            optional_deps = set(entry.get("optional_deps", []))
            required_missing = [dep for dep in missing if dep not in optional_deps]
            if required_missing:
                state.results[stage_name] = StageResult(
                    stage_name=stage_name,
                    meta=StageMeta(name=stage_name),
                    success=False,
                    error=f"Missing dependencies: {', '.join(required_missing)}",
                )
                if self.config.pipeline.fail_fast:
                    break
                continue

            hooks.stage_started(stage_name)
            try:
                result = stage.run(state, self.config, hooks)
            except Exception as exc:
                hooks.stage_failed(stage_name, exc)
                result = StageResult(
                    stage_name=stage_name,
                    meta=StageMeta(name=stage_name),
                    success=False,
                    error=str(exc),
                )
                if self.config.pipeline.fail_fast:
                    state.results[stage_name] = result
                    break

            result.duration_seconds = hooks.stage_finished(stage_name)
            state.results[stage_name] = result

        if self.config.pipeline.cache_state:
            cache_path = Path(output_path).with_suffix(".state.pkl") if output_path else Path("state.pkl")
            save_state(state, cache_path)

        if output_path is not None:
            ReportGenerator().write(state, self.config, output_path)
        return state

