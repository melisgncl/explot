from __future__ import annotations

import sys
from dataclasses import dataclass, field
from time import perf_counter

_STAGE_LABELS = {
    "profiling": "Profiling",
    "exploration": "Exploration",
    "preprocessing": "Preprocessing",
    "dimensionality": "Dimensionality",
    "autoencoder": "DVAE Autoencoder",
    "unsupervised": "Unsupervised",
    "supervised": "Model Selection",
    "survival": "Survival Analysis",
    "findings": "Findings",
}

_PHASES = ("Profile", "Model", "Synthesize")


@dataclass
class HookRegistry:
    budget_mode: str
    verbose: bool = False
    logs: list[dict[str, str]] = field(default_factory=list)
    _timers: dict[str, float] = field(default_factory=dict)
    _pipeline_start: float = 0.0
    _current_phase: str = ""

    def progress(self, stage: str, percent: int, message: str = "") -> None:
        self.logs.append(
            {"level": "info", "stage": stage, "message": f"{percent}% {message}".strip()}
        )

    def log(self, stage: str, message: str, level: str = "info") -> None:
        self.logs.append({"level": level, "stage": stage, "message": message})

    def check_budget(self, stage: str) -> dict[str, str]:
        return {"mode": self.budget_mode, "stage": stage}

    def pipeline_started(self) -> None:
        self._pipeline_start = perf_counter()

    def stage_started(self, stage: str, phase: str = "") -> None:
        self._timers[stage] = perf_counter()
        self.log(stage, "Stage started.")
        if self.verbose:
            if phase and phase != self._current_phase:
                self._current_phase = phase
                idx = _PHASES.index(phase) + 1 if phase in _PHASES else "?"
                header = f"── Phase {idx}/{len(_PHASES)}: {phase} "
                print(f"\n{header}{'─' * max(1, 48 - len(header))}", file=sys.stderr)
            label = _STAGE_LABELS.get(stage, stage)
            print(f"  [{label}] running...", end="", flush=True, file=sys.stderr)

    def stage_finished(self, stage: str) -> float:
        started = self._timers.pop(stage, None)
        duration = 0.0 if started is None else perf_counter() - started
        self.log(stage, f"Stage finished in {duration:.3f}s.")
        if self.verbose:
            print(f" done ({duration:.1f}s)", file=sys.stderr)
        return duration

    def stage_failed(self, stage: str, error: Exception) -> str:
        self.log(stage, f"Stage failed: {error}", level="error")
        if self.verbose:
            print(f" FAILED: {error}", file=sys.stderr)
        return "skip"

    def pipeline_finished(self) -> float:
        return perf_counter() - self._pipeline_start if self._pipeline_start else 0.0

