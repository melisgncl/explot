from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator.base import BaseSimulator


@dataclass
class ScrnaSimulator(BaseSimulator):
    name: str = "scrna"
    n_cells: int = 500
    n_genes: int = 200
    n_types: int = 4

    def generate(self, seed: int = 42, **kwargs) -> tuple[pd.DataFrame, dict]:
        rng = np.random.default_rng(seed)
        n_cells = int(kwargs.get("n_cells", self.n_cells))
        n_genes = int(kwargs.get("n_genes", self.n_genes))
        n_types = int(kwargs.get("n_types", self.n_types))

        cell_types = np.repeat(np.arange(n_types), n_cells // n_types)
        if len(cell_types) < n_cells:
            cell_types = np.concatenate([cell_types, rng.integers(0, n_types, size=n_cells - len(cell_types))])
        rng.shuffle(cell_types)

        counts = np.zeros((n_cells, n_genes), dtype=float)
        base_means = rng.uniform(1.0, 4.0, size=(n_types, n_genes))
        marker_genes = np.array_split(np.arange(n_genes), n_types)
        for cluster_id, genes in enumerate(marker_genes):
            base_means[cluster_id, genes] *= 5.0

        for row_idx, cluster_id in enumerate(cell_types):
            counts[row_idx] = rng.poisson(lam=base_means[cluster_id])
        counts[:, 1] = counts[:, 0] + rng.integers(0, 2, size=n_cells)

        columns = [f"gene_{idx + 1}" for idx in range(n_genes)]
        df = pd.DataFrame(counts.astype(int), columns=columns)
        df["cell_id"] = [f"cell_{idx:05d}" for idx in range(n_cells)]
        df["cell_type"] = [f"type_{label}" for label in cell_types]

        metadata = {
            "simulator": self.name,
            "seed": seed,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "profiling": {
                "expected_normalization": "raw counts",
                "suspicious_columns": [{"name": "cell_id", "reason": "id_like"}],
                "expected_quality_score_range": [75, 100],
            },
            "exploration": {
                "expected_redundant_pairs": [["gene_1", "gene_2"]],
                "expected_hopkins_clusterable": True,
            },
            "interpretation_checks": {
                "normalization_guess": ["raw counts", "log"],
                "quality_score": ["Quality score", "cell_id"],
            },
        }
        return df, metadata
