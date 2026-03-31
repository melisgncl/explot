from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class HookRegistry:
    budget_mode: str
    logs: list[dict[str, str]] = field(default_factory=list)
    _timers: dict[str, float] = field(default_factory=dict)

    def progress(self, stage: str, percent: int, message: str = "") -> None:
        self.logs.append(
            {"level": "info", "stage": stage, "message": f"{percent}% {message}".strip()}
        )

    def log(self, stage: str, message: str, level: str = "info") -> None:
        self.logs.append({"level": level, "stage": stage, "message": message})

    def check_budget(self, stage: str) -> dict[str, str]:
        return {"mode": self.budget_mode, "stage": stage}

    def stage_started(self, stage: str) -> None:
        self._timers[stage] = perf_counter()
        self.log(stage, "Stage started.")

    def stage_finished(self, stage: str) -> float:
        started = self._timers.pop(stage, None)
        duration = 0.0 if started is None else perf_counter() - started
        self.log(stage, f"Stage finished in {duration:.3f}s.")
        return duration

    def stage_failed(self, stage: str, error: Exception) -> str:
        self.log(stage, f"Stage failed: {error}", level="error")
        return "skip"

