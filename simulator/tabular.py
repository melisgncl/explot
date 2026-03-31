from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator.base import BaseSimulator


@dataclass
class TabularSimulator(BaseSimulator):
    name: str = "tabular"
    n_rows: int = 1000
    n_numeric: int = 10
    n_categories: int = 5

    def generate(self, seed: int = 42, **kwargs) -> tuple[pd.DataFrame, dict]:
        rng = np.random.default_rng(seed)
        n_rows = int(kwargs.get("n_rows", self.n_rows))
        n_numeric = int(kwargs.get("n_numeric", self.n_numeric))
        n_categories = int(kwargs.get("n_categories", self.n_categories))

        data: dict[str, object] = {
            "transaction_id": [f"txn_{index:06d}" for index in range(n_rows)],
            "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="h"),
            "amount": rng.lognormal(mean=4.2, sigma=0.8, size=n_rows).round(2),
        }

        for idx in range(n_numeric):
            data[f"metric_{idx + 1}"] = rng.normal(loc=0.0, scale=1.0 + idx * 0.05, size=n_rows)

        category_levels = [f"category_{idx + 1}" for idx in range(n_categories)]
        data["segment"] = rng.choice(category_levels, size=n_rows, replace=True)
        data["region"] = rng.choice(["north", "south", "east", "west"], size=n_rows, replace=True)
        data["is_priority"] = rng.choice([0, 1], size=n_rows, replace=True, p=[0.8, 0.2])

        df = pd.DataFrame(data)
        metadata = {
            "simulator": self.name,
            "seed": seed,
            "n_rows": n_rows,
            "n_cols": int(df.shape[1]),
            "profiling": {
                "expected_normalization": "log-normalized",
                "suspicious_columns": [{"name": "transaction_id", "reason": "id_like"}],
            },
            "exploration": {
                "expected_hopkins_clusterable": False,
                "grouping_column": "segment",
                "missingness_type": "minimal",
            },
            "findings": {
                "expected_findings": [
                    "should flag transaction_id as id-like",
                    "should detect right-skewed amount distribution",
                ]
            },
        }
        return df, metadata
