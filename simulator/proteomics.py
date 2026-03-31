from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator.base import BaseSimulator


@dataclass
class ProteomicsSimulator(BaseSimulator):
    name: str = "proteomics"
    n_samples: int = 150
    n_proteins: int = 120

    def generate(self, seed: int = 42, **kwargs) -> tuple[pd.DataFrame, dict]:
        rng = np.random.default_rng(seed)
        n_samples = int(kwargs.get("n_samples", self.n_samples))
        n_proteins = int(kwargs.get("n_proteins", self.n_proteins))

        intensities = rng.lognormal(mean=2.2, sigma=0.6, size=(n_samples, n_proteins))
        intensities[:, 1] = intensities[:, 0] * rng.normal(1.0, 0.02, size=n_samples)
        df = pd.DataFrame(np.log1p(intensities), columns=[f"protein_{idx + 1}" for idx in range(n_proteins)])
        df["sample_id"] = [f"sample_{idx:04d}" for idx in range(n_samples)]
        df["treatment"] = rng.choice(["control", "drug_a", "drug_b"], size=n_samples, replace=True)

        low_signal_columns = df.columns[:8]
        shared_mask = rng.random(n_samples) < 0.45
        for column in low_signal_columns:
            df.loc[shared_mask, column] = np.nan

        metadata = {
            "simulator": self.name,
            "seed": seed,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "profiling": {
                "expected_normalization": "log-normalized",
                "suspicious_columns": [{"name": "sample_id", "reason": "id_like"}],
                "expected_quality_score_range": [60, 100],
            },
            "exploration": {
                "expected_redundant_pairs": [["protein_1", "protein_2"]],
                "missingness_type": "structured",
            },
            "interpretation_checks": {
                "normalization_guess": ["log-normalized", "continuous"],
                "quality_score": ["Quality score", "sample_id"],
            },
        }
        return df, metadata
