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
from explot.stages.base import StageMeta, StageResult
from explot.state import PipelineState


def _save_best_model(state, model_path: Path) -> None:
    """Refit and save the highest-scoring model across all targets as a joblib file."""
    supervised = state.results.get("supervised")
    if not (supervised and supervised.success):
        return
    best_models = supervised.outputs.get("best_models", {}) or {}
    if not best_models:
        return

    # Pick the target with the highest score
    best_target, best_info = max(best_models.items(), key=lambda kv: kv[1].get("mean", 0))
    estimator = best_info.get("_export_estimator")
    feature_names = best_info.get("_export_feature_names", [])
    if estimator is None:
        return

    try:
        import joblib
        payload = {
            "estimator": estimator,
            "target": best_target,
            "feature_names": feature_names,
            "model_name": best_info.get("model", "unknown"),
            "metric": best_info.get("metric", "unknown"),
            "score": best_info.get("mean", 0),
        }
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, model_path)
        # Store path in state for the report to reference
        supervised.outputs["exported_model_path"] = str(model_path)
        supervised.outputs["exported_model_target"] = best_target
        supervised.outputs["exported_model_features"] = feature_names
    except Exception:
        pass  # model export is best-effort


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

    def run(
        self,
        input_path: Path,
        output_path: Path | None = None,
        verbose: bool = False,
        target_column: str | None = None,
        task_type: str | None = None,
    ) -> PipelineState:
        state = PipelineState(raw_df=load_table(input_path))
        state.target_column = target_column
        state.task_type = task_type if task_type != "auto" else None
        hooks = HookRegistry(budget_mode=self.config.budget.mode, verbose=verbose)
        hooks.pipeline_started()

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

            hooks.stage_started(stage_name, phase=entry.get("phase", ""))
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
            model_path = Path(str(output_path).rsplit(".", 1)[0] + "_model.joblib")
            _save_best_model(state, model_path)
            ReportGenerator().write(state, self.config, output_path)
        return state

