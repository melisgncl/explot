from __future__ import annotations

from math import isnan

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from scipy.stats import kurtosis, skew

from explot.stages.base import BaseStage, StageMeta, StageResult


class ProfilingStage(BaseStage):
    meta = StageMeta(name="profiling")

    def run(self, state, config, hooks) -> StageResult:
        df = state.raw_df
        hooks.progress(self.meta.name, 10, "Collecting dataset shape and column types.")

        numeric_columns = [column for column in df.columns if is_numeric_dtype(df[column])]
        categorical_columns = [column for column in df.columns if column not in numeric_columns]
        hooks.progress(self.meta.name, 45, "Summarizing columns and suspicious patterns.")

        column_profiles = {}
        suspicious_columns = []

        for column in df.columns:
            series = df[column]
            null_count = int(series.isna().sum())
            non_null = series.dropna()
            cardinality = int(non_null.nunique(dropna=True))
            top_frequency = 0.0
            top_value = None
            if not non_null.empty:
                value_counts = non_null.value_counts(dropna=True)
                top_value = value_counts.index[0]
                top_frequency = float(value_counts.iloc[0] / len(non_null))

            profile = {
                "dtype": str(series.dtype),
                "null_count": null_count,
                "null_percent": round((null_count / len(df) * 100.0), 2) if len(df) else 0.0,
                "cardinality": cardinality,
            }

            if is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                summary = self._numeric_summary(numeric)
                profile["summary"] = summary
                zero_count = int((numeric == 0).sum())
                profile["zero_count"] = zero_count
                profile["zero_percent"] = round((zero_count / len(numeric) * 100.0), 2) if len(numeric) else 0.0
            else:
                value_counts = non_null.astype(str).value_counts().head(20)
                profile["top_values"] = {key: int(value) for key, value in value_counts.items()}
                profile["other_count"] = max(0, cardinality - len(value_counts))

            role_guess = self._guess_column_role(str(column), series, len(df), cardinality, profile)
            profile["role_guess"] = role_guess

            column_profiles[str(column)] = profile

            if len(non_null) and top_frequency >= 0.95:
                suspicious_columns.append(
                    {
                        "name": str(column),
                        "reason": "near_constant",
                        "details": f"Top value {top_value!r} occupies {top_frequency:.1%} of non-null rows.",
                    }
                )
            if len(df) and null_count / len(df) > 0.80:
                suspicious_columns.append(
                    {
                        "name": str(column),
                        "reason": "mostly_null",
                        "details": f"Null rate is {null_count / len(df):.1%}.",
                    }
                )
            column_name = str(column).lower()
            if (
                role_guess == "id_like"
                and len(df)
                and len(non_null) >= max(10, int(0.9 * len(df)))
            ):
                suspicious_columns.append(
                    {
                        "name": str(column),
                        "reason": "id_like",
                        "details": "Column has near-row-level uniqueness and looks like an identifier.",
                    }
                )

        normalization_guess = self._guess_normalization(df[numeric_columns]) if numeric_columns else "unknown"
        quality_breakdown = self._quality_breakdown(df, suspicious_columns, column_profiles)
        quality_score = round(sum(quality_breakdown.values()))
        fingerprint = self._fingerprint(df, numeric_columns, suspicious_columns)
        memory_usage = {str(column): int(value) for column, value in df.memory_usage(deep=True).items()}

        outputs = {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "column_names": [str(column) for column in df.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
            "memory_usage_bytes": {
                "total": int(sum(memory_usage.values())),
                "by_column": memory_usage,
            },
            "column_profiles": column_profiles,
            "numeric_column_names": [str(column) for column in numeric_columns],
            "categorical_column_names": [str(column) for column in categorical_columns],
            "suspicious_columns": suspicious_columns,
            "normalization_guess": normalization_guess,
            "quality_score": quality_score,
            "quality_breakdown": quality_breakdown,
            "fingerprint": fingerprint,
        }
        hooks.progress(self.meta.name, 80, "Building profiling interpretations.")
        figures = {"fingerprint_radar": self._fingerprint_radar_svg(fingerprint)}
        interpretations = {
            "dataset_shape": (
                f"Dataset contains {outputs['n_rows']} rows and {outputs['n_columns']} columns. "
                f"Detected {len(numeric_columns)} numeric columns and "
                f"{len(categorical_columns)} non-numeric columns."
            ),
            "normalization_guess": self._normalization_interpretation(normalization_guess, numeric_columns),
            "quality_score": self._quality_interpretation(quality_score, quality_breakdown, suspicious_columns),
            "fingerprint_radar": self._fingerprint_interpretation(fingerprint),
            "column_roles": self._role_interpretation(column_profiles),
        }
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs=outputs,
            figures=figures,
            interpretations=interpretations,
        )

    def _numeric_summary(self, numeric: pd.Series) -> dict[str, float | None]:
        if numeric.empty:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "skew": None,
                "kurtosis": None,
            }

        values = numeric.to_numpy(dtype=float)
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "skew": self._safe_stat(skew(values, bias=False)),
            "kurtosis": self._safe_stat(kurtosis(values, fisher=True, bias=False)),
        }

    def _safe_stat(self, value: float | np.floating | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        if isnan(value) or np.isinf(value):
            return None
        return value

    def _looks_temporal(self, series: pd.Series, column_name: str) -> bool:
        if any(token in column_name for token in ("time", "date", "timestamp")):
            return True
        if is_numeric_dtype(series):
            return False
        sample = series.dropna().astype(str).head(25)
        if sample.empty:
            return False
        temporal_like = sample.str.contains(r"[-/:T ]", regex=True).mean()
        if temporal_like < 0.8:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return bool((~parsed.isna()).mean() >= 0.8)

    def _guess_normalization(self, numeric_df: pd.DataFrame) -> str:
        if numeric_df.empty:
            return "unknown"

        flattened = pd.Series(numeric_df.to_numpy(dtype=float).ravel()).dropna()
        if flattened.empty:
            return "unknown"

        values = flattened.to_numpy(dtype=float)
        non_negative = bool(np.all(values >= 0))
        integer_like = bool(np.allclose(values, np.round(values)))
        positive = bool(np.all(values > 0))
        sample_skew = self._safe_stat(skew(values, bias=False)) or 0.0
        dynamic_range = float(np.max(values) - np.min(values))
        row_sums = numeric_df.sum(axis=1, skipna=True).to_numpy(dtype=float)
        row_sums = row_sums[np.isfinite(row_sums)]
        row_sum_cv = None
        if row_sums.size and np.mean(row_sums) != 0:
            row_sum_cv = float(np.std(row_sums) / np.mean(row_sums))

        if non_negative and integer_like and sample_skew > 1.0:
            return "raw counts"
        if positive and not integer_like and row_sum_cv is not None and row_sum_cv <= 0.01:
            return "CPM/TPM-like"
        if positive and not integer_like and sample_skew > 0.2 and dynamic_range > 3.0:
            return "log-normalized"
        if abs(float(np.mean(values))) < 0.3 and 0.5 < float(np.std(values)) < 2.5:
            return "StandardScaler output"
        return "unknown"

    def _guess_column_role(
        self,
        column_name: str,
        series: pd.Series,
        n_rows: int,
        cardinality: int,
        profile: dict[str, object],
    ) -> str:
        lowered = column_name.lower()
        non_null = series.dropna()
        if non_null.empty:
            return "unknown"

        id_name_hint = lowered == "id" or lowered.endswith("_id") or lowered.endswith("id") or "person_id" in lowered
        uniqueness_ratio = cardinality / max(len(non_null), 1)
        if id_name_hint and (uniqueness_ratio >= 0.98 or cardinality >= max(20, int(n_rows * 0.01))):
            return "id_like"

        if is_datetime64_any_dtype(series) or self._looks_temporal(series, lowered):
            return "time_like"

        if is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                return "unknown"
            unique_values = numeric.nunique()
            zero_percent = float(profile.get("zero_percent", 0.0))
            is_integer_like = bool(np.allclose(numeric.to_numpy(dtype=float), np.round(numeric.to_numpy(dtype=float))))
            summary = profile.get("summary", {})
            sample_skew = summary.get("skew") if isinstance(summary, dict) else None
            sample_skew = 0.0 if sample_skew is None else float(sample_skew)

            if id_name_hint and (uniqueness_ratio >= 0.98 or cardinality >= max(20, int(n_rows * 0.01))):
                return "id_like"
            if uniqueness_ratio >= 0.995 and is_integer_like and not self._looks_measurement_like(lowered):
                return "id_like"
            if unique_values == 2:
                return "binary_flag"
            if is_integer_like and unique_values <= 12:
                if any(token in lowered for token in ("type", "class", "stage", "grade", "result")):
                    return "categorical_code"
                if any(token in lowered for token in ("time", "day", "week", "month", "year", "hour")):
                    return "time_like"
                return "ordinal_integer"
            if is_integer_like and np.all(numeric >= 0) and (zero_percent >= 30.0 or sample_skew > 1.0):
                return "count_like"
            return "continuous_measurement"

        if uniqueness_ratio >= 0.98 and id_name_hint:
            return "id_like"
        return "categorical_text"

    def _looks_measurement_like(self, column_name: str) -> bool:
        return any(
            token in column_name
            for token in (
                "age",
                "bp",
                "chol",
                "score",
                "rate",
                "duration",
                "steps",
                "temperature",
                "reads",
                "count",
            )
        )

    def _quality_breakdown(
        self,
        df: pd.DataFrame,
        suspicious_columns: list[dict[str, str]],
        column_profiles: dict[str, dict[str, object]],
    ) -> dict[str, float]:
        n_columns = max(len(df.columns), 1)
        completeness = 40.0 * float(1.0 - df.isna().sum().sum() / max(df.size, 1))
        redundant_penalty = sum(
            1 for entry in suspicious_columns if entry["reason"] in {"id_like", "near_constant"}
        )
        non_redundancy = 20.0 * max(0.0, 1.0 - redundant_penalty / n_columns)
        suspicious_rate = len(suspicious_columns) / n_columns
        non_suspicious = 20.0 * max(0.0, 1.0 - suspicious_rate)
        dtype_counts: dict[str, int] = {}
        for profile in column_profiles.values():
            dtype = str(profile["dtype"])
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        dominant_dtype_share = max(dtype_counts.values(), default=0) / n_columns
        dtype_consistency = 20.0 * dominant_dtype_share
        return {
            "completeness": round(completeness, 2),
            "non_redundancy": round(non_redundancy, 2),
            "non_suspicious": round(non_suspicious, 2),
            "dtype_consistency": round(dtype_consistency, 2),
        }

    def _normalization_interpretation(self, guess: str, numeric_columns: list[object]) -> str:
        if not numeric_columns:
            return "No numeric columns were detected, so a normalization guess is not available."
        if guess == "raw counts":
            return (
                "Numeric values look non-negative and integer-like with strong right skew. "
                "This is consistent with raw counts, so a log-style transform is likely useful later."
            )
        if guess == "CPM/TPM-like":
            return (
                "Numeric values are positive and row sums are nearly constant across samples. "
                "That is consistent with a library-size normalized matrix such as CPM or TPM."
            )
        if guess == "log-normalized":
            return (
                "Numeric values are continuous, positive, and still right-skewed without looking integer-like. "
                "This pattern is consistent with log-normalized intensities."
            )
        if guess == "StandardScaler output":
            return (
                "Numeric values are centered near zero with moderate spread. "
                "This pattern is consistent with previously standardized features."
            )
        return (
            "The numeric columns do not cleanly match the current heuristics for counts, "
            "log-normalized data, or standardized values."
        )

    def _quality_interpretation(
        self,
        quality_score: int,
        quality_breakdown: dict[str, float],
        suspicious_columns: list[dict[str, str]],
    ) -> str:
        if suspicious_columns:
            suspicious_text = ", ".join(
                f"{entry['name']} ({entry['reason']})" for entry in suspicious_columns[:5]
            )
        else:
            suspicious_text = "no suspicious columns flagged"
        return (
            f"Quality score: {quality_score}/100. "
            f"Completeness {quality_breakdown['completeness']:.1f}, "
            f"non-redundancy {quality_breakdown['non_redundancy']:.1f}, "
            f"non-suspicious {quality_breakdown['non_suspicious']:.1f}, "
            f"dtype consistency {quality_breakdown['dtype_consistency']:.1f}. "
            f"Current review flags: {suspicious_text}."
        )

    def _role_interpretation(self, column_profiles: dict[str, dict[str, object]]) -> str:
        role_counts: dict[str, int] = {}
        for profile in column_profiles.values():
            role = str(profile.get("role_guess", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
        ordered = sorted(role_counts.items(), key=lambda item: (-item[1], item[0]))
        summary = ", ".join(f"{role}={count}" for role, count in ordered)
        return f"Heuristic column roles: {summary}."

    def _fingerprint(
        self,
        df: pd.DataFrame,
        numeric_columns: list[object],
        suspicious_columns: list[dict[str, str]],
    ) -> dict[str, float]:
        n_rows = max(len(df), 1)
        n_cols = max(len(df.columns), 1)
        completeness = max(0.0, min(100.0, 100.0 * (1.0 - df.isna().sum().sum() / max(df.size, 1))))
        dimensionality_complexity = max(0.0, min(100.0, 100.0 * min(1.0, n_cols / max(n_rows, 1))))
        cluster_tendency = 50.0
        if numeric_columns:
            numeric_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
            variances = numeric_df.var(numeric_only=True, skipna=True)
            if not variances.empty:
                cluster_tendency = max(0.0, min(100.0, 20.0 + float((variances > 0).mean()) * 60.0))
            signal_strength = max(
                0.0,
                min(100.0, 100.0 * float((variances > 1e-8).mean()) if not variances.empty else 0.0),
            )
        else:
            signal_strength = 0.0
        anomaly_rate = max(0.0, min(100.0, 100.0 * len(suspicious_columns) / n_cols))
        return {
            "completeness": round(completeness, 2),
            "dimensionality_complexity": round(dimensionality_complexity, 2),
            "cluster_tendency": round(cluster_tendency, 2),
            "signal_strength": round(signal_strength, 2),
            "anomaly_rate": round(anomaly_rate, 2),
        }

    def _fingerprint_interpretation(self, fingerprint: dict[str, float]) -> str:
        return (
            "Dataset fingerprint: "
            f"completeness {fingerprint['completeness']:.1f}, "
            f"dimensionality complexity {fingerprint['dimensionality_complexity']:.1f}, "
            f"cluster tendency {fingerprint['cluster_tendency']:.1f}, "
            f"signal strength {fingerprint['signal_strength']:.1f}, "
            f"anomaly rate {fingerprint['anomaly_rate']:.1f}."
        )

    def _fingerprint_radar_svg(self, fingerprint: dict[str, float]) -> str:
        labels = list(fingerprint.keys())
        values = [max(0.0, min(100.0, float(value))) for value in fingerprint.values()]
        center_x = 180.0
        center_y = 180.0
        radius = 110.0

        def point(angle_deg: float, scale: float) -> tuple[float, float]:
            angle_rad = np.deg2rad(angle_deg - 90.0)
            return (
                center_x + np.cos(angle_rad) * radius * scale,
                center_y + np.sin(angle_rad) * radius * scale,
            )

        axis_angles = [idx * (360.0 / len(labels)) for idx in range(len(labels))]
        grid_levels = [0.25, 0.5, 0.75, 1.0]
        grid_polygons = []
        for level in grid_levels:
            pts = [point(angle, level) for angle in axis_angles]
            grid_polygons.append(" ".join(f"{x:.1f},{y:.1f}" for x, y in pts))

        data_points = [point(angle, value / 100.0) for angle, value in zip(axis_angles, values)]
        data_polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in data_points)

        axis_lines = []
        axis_labels = []
        for angle, label in zip(axis_angles, labels):
            x, y = point(angle, 1.0)
            lx, ly = point(angle, 1.22)
            axis_lines.append(
                f"<line x1='{center_x:.1f}' y1='{center_y:.1f}' x2='{x:.1f}' y2='{y:.1f}' "
                "stroke='#8ea3b5' stroke-width='1' />"
            )
            axis_labels.append(
                f"<text x='{lx:.1f}' y='{ly:.1f}' fill='#1e2a33' font-size='12' "
                "text-anchor='middle'>"
                f"{label.replace('_', ' ')}</text>"
            )

        return (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 360 360' "
            "style='max-width:420px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            "<rect x='0' y='0' width='360' height='360' fill='#f8fbfd' rx='12' />"
            + "".join(
                f"<polygon points='{polygon}' fill='none' stroke='#d7e0e8' stroke-width='1' />"
                for polygon in grid_polygons
            )
            + "".join(axis_lines)
            + f"<polygon points='{data_polygon}' fill='rgba(27,99,146,0.22)' "
            "stroke='#1b6392' stroke-width='2' />"
            + "".join(axis_labels)
            + "<text x='180' y='28' fill='#143047' font-size='16' text-anchor='middle'>"
            "Data Fingerprint</text></svg>"
        )
