from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any

import pandas as pd


class BaseSimulator(ABC):
    name: str

    @abstractmethod
    def generate(self, seed: int = 42, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return a dataframe plus metadata describing planted structure."""

    def save(self, output_dir: Path, seed: int = 42, **kwargs: Any) -> Path:
        df, metadata = self.generate(seed=seed, **kwargs)
        target_dir = output_dir / self.name
        target_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_dir / "data.csv", index=False)
        (target_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return target_dir

