from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from explot.state import PipelineState

_SKIP_KEYS = {"cleaned_df", "transformed_df", "latent_df", "pca_2d", "umap_2d"}


def _make_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if np.isfinite(obj):
            return round(obj, 6)
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        if obj.size > 200:
            return f"<array shape={obj.shape}>"
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return f"<DataFrame {obj.shape[0]}x{obj.shape[1]}>"
    if isinstance(obj, pd.Series):
        return f"<Series len={len(obj)}>"
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items() if k not in _SKIP_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return str(obj)


def state_to_dict(state: PipelineState) -> dict:
    result = {"stages": {}}
    for name, sr in state.results.items():
        entry: dict[str, Any] = {
            "success": sr.success,
            "error": sr.error,
            "outputs": _make_serializable(sr.outputs),
            "interpretations": _make_serializable(sr.interpretations),
        }
        if hasattr(sr, "duration_seconds") and sr.duration_seconds is not None:
            entry["duration_seconds"] = round(sr.duration_seconds, 2)
        result["stages"][name] = entry
    return result


def state_to_json(state: PipelineState) -> str:
    return json.dumps(state_to_dict(state), indent=2, ensure_ascii=False)
