from __future__ import annotations

import pickle
from pathlib import Path

from explot.state import PipelineState


def save_state(state: PipelineState, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(state, handle)


def load_state(path: Path) -> PipelineState:
    with path.open("rb") as handle:
        return pickle.load(handle)

