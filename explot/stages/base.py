from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageMeta:
    name: str
    version: int = 1
    depends_on: tuple[str, ...] = ()
    optional_deps: tuple[str, ...] = ()


@dataclass
class StageResult:
    stage_name: str
    meta: StageMeta
    outputs: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, str] = field(default_factory=dict)
    interpretations: dict[str, str] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    duration_seconds: float = 0.0

    def summary(self) -> str:
        if self.success:
            return "success"
        return f"failed ({self.error})"


class BaseStage(ABC):
    meta: StageMeta

    def validate_inputs(self, state: Any) -> list[str]:
        missing: list[str] = []
        for dep in self.meta.depends_on:
            if dep not in state.results:
                missing.append(dep)
        return missing

    @abstractmethod
    def run(self, state: Any, config: Any, hooks: Any) -> StageResult:
        raise NotImplementedError

