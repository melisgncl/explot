def test_tabular_simulator_shape(tabular_data) -> None:
    df, metadata = tabular_data

    assert df.shape[0] == metadata["n_rows"]
    assert df.shape[1] == metadata["n_cols"]
    assert "transaction_id" in df.columns
    assert "amount" in df.columns


def test_tabular_simulator_metadata_points_to_real_columns(tabular_data) -> None:
    df, metadata = tabular_data
    expected = metadata["profiling"]["suspicious_columns"]

    for entry in expected:
        assert entry["name"] in df.columns


def test_tabular_has_expected_id_like_column(tabular_data) -> None:
    df, _ = tabular_data

    assert df["transaction_id"].nunique() == len(df)
