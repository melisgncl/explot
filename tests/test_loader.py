from pathlib import Path

import pandas as pd

from explot.loader import load_table


def test_load_csv(workspace_tmp_path: Path) -> None:
    path = workspace_tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(path, index=False)

    df = load_table(path)

    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
