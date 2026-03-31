from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from explot.stages.base import StageResult


@dataclass
class PipelineState:
    raw_df: pd.DataFrame
    results: dict[str, StageResult] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def stage_output(self, stage_name: str, key: str, default: Any = None) -> Any:
        result = self.results.get(stage_name)
        if result is None:
            return default
        return result.outputs.get(key, default)

