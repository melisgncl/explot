from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from explot.stages.base import BaseStage, StageMeta, StageResult


class PreprocessingStage(BaseStage):
    meta = StageMeta(
        name="preprocessing",
        depends_on=("profiling",),
    )

    def run(self, state, config, hooks) -> StageResult:
        profiling = state.results["profiling"]
        df = state.raw_df

        hooks.progress(self.meta.name, 10, "Classifying columns for encoding and imputation.")

        col_profiles = profiling.outputs.get("column_profiles", {})
        numeric_names = set(profiling.outputs.get("numeric_column_names", []))
        role_by_col: dict[str, str] = {
            str(name): str(profile.get("role_guess", "unknown"))
            for name, profile in col_profiles.items()
        }

        encoding_log: list[str] = []
        imputation_stats: dict[str, dict] = {}
        dropped_columns: list[dict] = []
        ordinal_encoded: list[str] = []
        frequency_encoded: list[str] = []
        columns_with_heavy_imputation: list[str] = []
        result_cols: dict[str, pd.Series] = {}

        for col in df.columns:
            col_str = str(col)
            role = role_by_col.get(col_str, "unknown")

            # Skip columns that are identifiers or timestamps — not model features
            if role in {"id_like", "time_like"}:
                encoding_log.append(f"Skipped '{col_str}' (role: {role}).")
                continue

            series = df[col]
            n_total = len(series)
            n_null = int(series.isna().sum())

            is_numeric = col_str in numeric_names or pd.api.types.is_numeric_dtype(series)

            if is_numeric:
                median_val = series.median()
                if pd.isna(median_val):
                    median_val = 0.0
                if n_null > 0:
                    pct = round(100.0 * n_null / n_total, 1)
                    imputation_stats[col_str] = {
                        "method": "median",
                        "n_filled": n_null,
                        "pct_filled": pct,
                        "fill_value": float(median_val),
                    }
                    if pct > 20.0:
                        columns_with_heavy_imputation.append(col_str)
                        encoding_log.append(
                            f"Heavy imputation on '{col_str}': {pct}% NaN filled with median ({median_val:.3g})."
                        )
                    else:
                        encoding_log.append(
                            f"Imputed '{col_str}': {n_null} NaN → median ({median_val:.3g})."
                        )
                result_cols[col_str] = pd.to_numeric(series, errors="coerce").fillna(float(median_val))

            else:
                # Categorical / object column
                n_unique = int(series.dropna().nunique())

                if n_unique == 0:
                    encoding_log.append(f"Dropped '{col_str}': empty column.")
                    dropped_columns.append({"name": col_str, "reason": "empty", "n_unique": 0})
                    continue

                if n_unique > 100:
                    encoding_log.append(
                        f"Dropped '{col_str}': {n_unique} unique values exceed encoding threshold (>100)."
                    )
                    dropped_columns.append({"name": col_str, "reason": "high_cardinality", "n_unique": n_unique})
                    continue

                # Impute missing with mode before encoding
                if n_null > 0:
                    mode_val = series.mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else "unknown"
                    series = series.fillna(fill_val).infer_objects(copy=False)
                    pct = round(100.0 * n_null / n_total, 1)
                    imputation_stats[col_str] = {
                        "method": "most_frequent",
                        "n_filled": n_null,
                        "pct_filled": pct,
                        "fill_value": str(fill_val),
                    }
                    if pct > 20.0:
                        columns_with_heavy_imputation.append(col_str)

                if n_unique <= 20:
                    # OrdinalEncoder: maps each category string to 0..n_unique-1
                    enc = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    encoded = enc.fit_transform(
                        series.astype(str).to_numpy().reshape(-1, 1)
                    ).ravel()
                    result_cols[col_str] = pd.Series(encoded, index=df.index, name=col_str)
                    ordinal_encoded.append(col_str)
                    encoding_log.append(f"OrdinalEncoded '{col_str}' ({n_unique} categories).")
                else:
                    # Frequency encoding: replace each category with its row proportion
                    freq_map = series.astype(str).value_counts(normalize=True)
                    encoded = series.astype(str).map(freq_map).fillna(0.0)
                    result_cols[col_str] = pd.Series(
                        encoded.to_numpy(dtype=float), index=df.index, name=col_str
                    )
                    frequency_encoded.append(col_str)
                    encoding_log.append(
                        f"FrequencyEncoded '{col_str}' ({n_unique} unique values)."
                    )

        hooks.progress(self.meta.name, 90, "Building preprocessed matrix.")

        preprocessed_df = (
            pd.DataFrame(result_cols, index=df.index) if result_cols else pd.DataFrame(index=df.index)
        )

        n_encoded = len(ordinal_encoded) + len(frequency_encoded)
        n_imputed = len(imputation_stats)

        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={
                "preprocessed_df": preprocessed_df,
                "encoding_log": encoding_log,
                "imputation_stats": imputation_stats,
                "dropped_columns": dropped_columns,
                "ordinal_encoded": ordinal_encoded,
                "frequency_encoded": frequency_encoded,
                "columns_with_heavy_imputation": columns_with_heavy_imputation,
                "n_encoded": n_encoded,
                "n_imputed": n_imputed,
            },
            interpretations={
                "summary": self._interpret(
                    n_encoded, n_imputed,
                    ordinal_encoded, frequency_encoded,
                    dropped_columns, columns_with_heavy_imputation,
                ),
            },
        )

    # ------------------------------------------------------------------
    def _interpret(
        self,
        n_encoded: int,
        n_imputed: int,
        ordinal_encoded: list[str],
        frequency_encoded: list[str],
        dropped_columns: list[dict],
        heavy_cols: list[str],
    ) -> str:
        parts: list[str] = []

        if n_encoded:
            parts.append(
                f"Encoded {n_encoded} categorical column(s): "
                f"{len(ordinal_encoded)} ordinal (≤20 categories), "
                f"{len(frequency_encoded)} frequency-encoded (21–100 categories)."
            )
        if n_imputed:
            parts.append(f"Imputed missing values in {n_imputed} column(s).")
        if heavy_cols:
            sample = ", ".join(f"'{c}'" for c in heavy_cols[:5])
            parts.append(
                f"Heavy imputation (>20% NaN) on {len(heavy_cols)} column(s): {sample}. "
                "Missingness may itself carry signal — consider an indicator column."
            )

        hc_drops = [d for d in dropped_columns if d["reason"] == "high_cardinality"]
        if hc_drops:
            sample = ", ".join(f"'{d['name']}'" for d in hc_drops[:3])
            parts.append(
                f"Dropped {len(hc_drops)} high-cardinality column(s) ({sample}). "
                "Consider target encoding or embeddings for these features."
            )

        if not parts:
            parts.append("No encoding or imputation needed — dataset is already fully numeric.")

        return " ".join(parts)
