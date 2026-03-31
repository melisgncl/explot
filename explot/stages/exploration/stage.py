from __future__ import annotations

import json

import numpy as np
import pandas as pd

from explot.stages.base import BaseStage, StageMeta, StageResult


class ExplorationStage(BaseStage):
    meta = StageMeta(name="exploration", depends_on=("profiling",))

    def run(self, state, config, hooks) -> StageResult:
        df = state.raw_df
        profiling = state.results["profiling"]
        hooks.progress(self.meta.name, 15, "Collecting numeric columns for relationship analysis.")

        numeric_columns = list(profiling.outputs.get("numeric_column_names", []))
        numeric_df = df[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame()

        correlation_matrix = self._correlation_matrix(numeric_df)
        redundant_pairs = self._redundant_pairs(correlation_matrix)
        top_variable_features = self._top_variable_features(numeric_df)
        missingness_correlations, missingness_type = self._missingness_structure(df)
        grouping_candidates = self._grouping_candidates(df, profiling.outputs.get("categorical_column_names", []))
        hopkins_statistic = self._hopkins_statistic(numeric_df)
        outlier_rows = self._row_outliers(numeric_df)
        top_feature_distributions = self._top_feature_distributions(numeric_df, top_variable_features)
        grouped_distributions = self._grouped_distributions(df, numeric_df, grouping_candidates, top_variable_features)

        outputs = {
            "correlation_matrix": correlation_matrix,
            "redundant_pairs": redundant_pairs,
            "top_variable_features": top_variable_features,
            "missingness_type": missingness_type,
            "missingness_correlations": missingness_correlations,
            "grouping_candidates": grouping_candidates,
            "hopkins_statistic": hopkins_statistic,
            "outlier_rows": outlier_rows,
            "top_feature_distributions": top_feature_distributions,
            "grouped_distributions": grouped_distributions,
        }
        figures = {
            "correlation_heatmap": self._heatmap_svg(correlation_matrix),
            "top_variable_features": json.dumps(top_variable_features),
            "distribution_overview": self._distribution_grid_svg(top_feature_distributions),
            "grouped_distributions": self._grouped_box_svg(grouped_distributions),
        }
        interpretations = {
            "correlation_heatmap": self._correlation_interpretation(redundant_pairs, correlation_matrix),
            "missingness_analysis": self._missingness_interpretation(missingness_type, missingness_correlations),
            "hopkins_statistic": self._hopkins_interpretation(hopkins_statistic),
            "row_outliers": self._outlier_interpretation(outlier_rows, numeric_df),
            "distribution_overview": self._distribution_interpretation(top_feature_distributions),
            "grouped_distributions": self._grouped_distribution_interpretation(grouped_distributions),
        }
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs=outputs,
            figures=figures,
            interpretations=interpretations,
        )

    def _correlation_matrix(self, numeric_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return {}
        corr = numeric_df.corr(method="pearson", min_periods=max(3, int(len(numeric_df) * 0.1)))
        corr = corr.fillna(0.0)
        return {
            str(col): {str(inner): float(value) for inner, value in row.items()}
            for col, row in corr.iterrows()
        }

    def _redundant_pairs(self, correlation_matrix: dict[str, dict[str, float | None]]) -> list[dict[str, object]]:
        if not correlation_matrix:
            return []
        columns = list(correlation_matrix.keys())
        pairs: list[dict[str, object]] = []
        for idx, left in enumerate(columns):
            for right in columns[idx + 1 :]:
                value = correlation_matrix[left][right]
                if abs(value) > 0.95:
                    pairs.append({"columns": [left, right], "correlation": round(float(value), 4)})
        pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        return pairs

    def _top_variable_features(self, numeric_df: pd.DataFrame) -> list[dict[str, float]]:
        if numeric_df.empty:
            return []
        variances = numeric_df.var(skipna=True, numeric_only=True).sort_values(ascending=False).head(10)
        return [{"name": str(name), "variance": round(float(value), 6)} for name, value in variances.items()]

    def _missingness_structure(self, df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], str]:
        missing = df.isna().astype(int)
        if missing.sum().sum() == 0:
            return {}, "minimal"
        active_missing = missing.loc[:, missing.sum(axis=0) > 0]
        missing_corr = active_missing.corr().fillna(0.0) if not active_missing.empty else missing.corr().fillna(0.0)
        matrix = {
            str(col): {str(inner): float(value) for inner, value in row.items()}
            for col, row in missing_corr.iterrows()
        }
        off_diagonal = []
        columns = list(missing_corr.columns)
        for idx, left in enumerate(columns):
            for right in columns[idx + 1 :]:
                off_diagonal.append(abs(float(missing_corr.loc[left, right])))
        mean_abs_corr = float(np.mean(off_diagonal)) if off_diagonal else 0.0
        max_abs_corr = float(np.max(off_diagonal)) if off_diagonal else 0.0
        if max_abs_corr > 0.5 or mean_abs_corr > 0.25:
            missingness_type = "structured"
        else:
            missingness_type = "random-looking"
        return matrix, missingness_type

    def _grouping_candidates(self, df: pd.DataFrame, categorical_columns: list[str]) -> list[dict[str, object]]:
        candidates = []
        for column in categorical_columns:
            unique = int(df[column].nunique(dropna=True))
            if 2 <= unique <= 10:
                candidates.append({"name": column, "n_groups": unique})
        return candidates

    def _row_outliers(self, numeric_df: pd.DataFrame) -> list[int]:
        if numeric_df.empty:
            return []

        usable = numeric_df.dropna(axis=1, how="all").copy()
        if usable.empty:
            return []

        variances = usable.var(skipna=True, numeric_only=True)
        usable = usable.loc[:, variances.fillna(0.0) > 0]
        if usable.shape[0] < 10 or usable.shape[1] < 2:
            return []

        medians = usable.median()
        usable = usable.fillna(medians)
        if usable.isna().any().any():
            usable = usable.fillna(0.0)

        centered = usable - usable.mean(axis=0)
        scaled = centered / usable.std(axis=0, ddof=0).replace(0, 1.0)
        distances = np.sqrt(np.sum(np.square(scaled.to_numpy(dtype=float)), axis=1))
        n_outliers = min(len(usable), max(1, int(np.ceil(len(usable) * 0.01))))
        ranked = np.argsort(distances)[-n_outliers:]
        row_indices = sorted(int(usable.index[idx]) for idx in ranked)
        return row_indices

    def _hopkins_statistic(self, numeric_df: pd.DataFrame) -> float | None:
        if numeric_df.empty or numeric_df.shape[0] < 25 or numeric_df.shape[1] < 2:
            return None
        clean = numeric_df.dropna(axis=0)
        if len(clean) < 25:
            return None
        sample_size = min(max(10, len(clean) // 10), 50)
        sample = clean.sample(n=sample_size, random_state=42)
        mins = clean.min(axis=0)
        maxs = clean.max(axis=0)
        if np.any((maxs - mins).to_numpy(dtype=float) == 0):
            clean = clean.loc[:, (maxs - mins) > 0]
            sample = sample[clean.columns]
            mins = clean.min(axis=0)
            maxs = clean.max(axis=0)
        if clean.shape[1] < 2:
            return None
        scaled = (clean - clean.mean(axis=0)) / clean.std(axis=0, ddof=0).replace(0, 1.0)
        scaled_array = scaled.to_numpy(dtype=float)
        scaled_mins = np.min(scaled_array, axis=0)
        scaled_maxs = np.max(scaled_array, axis=0)
        u = np.random.default_rng(42).uniform(scaled_mins, scaled_maxs, size=(sample_size, scaled_array.shape[1]))
        sample_scaled = scaled.loc[sample.index].to_numpy(dtype=float)
        u_distances = self._nearest_distances(u, scaled_array)
        w_distances = self._nearest_distances(sample_scaled, scaled_array, skip_self=True)
        denominator = float(np.sum(u_distances) + np.sum(w_distances))
        if denominator == 0:
            return None
        return round(float(np.sum(u_distances) / denominator), 4)

    def _top_feature_distributions(
        self,
        numeric_df: pd.DataFrame,
        top_variable_features: list[dict[str, float]],
    ) -> list[dict[str, object]]:
        if numeric_df.empty or not top_variable_features:
            return []
        distributions: list[dict[str, object]] = []
        for item in top_variable_features[:4]:
            column = item["name"]
            series = pd.to_numeric(numeric_df[column], errors="coerce").dropna()
            if len(series) < 5:
                continue
            counts, edges = np.histogram(series.to_numpy(dtype=float), bins=min(18, max(8, int(np.sqrt(len(series))))))
            distributions.append(
                {
                    "name": column,
                    "counts": counts.astype(int).tolist(),
                    "edges": [round(float(edge), 4) for edge in edges.tolist()],
                    "median": round(float(series.median()), 4),
                    "iqr": round(float(series.quantile(0.75) - series.quantile(0.25)), 4),
                }
            )
        return distributions

    def _grouped_distributions(
        self,
        df: pd.DataFrame,
        numeric_df: pd.DataFrame,
        grouping_candidates: list[dict[str, object]],
        top_variable_features: list[dict[str, float]],
    ) -> list[dict[str, object]]:
        if not grouping_candidates or not top_variable_features or numeric_df.empty:
            return []
        feature_names = [item["name"] for item in top_variable_features[:2]]
        grouped_cards: list[dict[str, object]] = []
        for group_info in grouping_candidates[:2]:
            group_col = group_info["name"]
            for feature_name in feature_names:
                aligned = pd.DataFrame(
                    {
                        "group": df[group_col],
                        "value": pd.to_numeric(df[feature_name], errors="coerce"),
                    }
                ).dropna()
                if aligned.empty or aligned["group"].nunique() < 2:
                    continue
                summaries = []
                for group_value, group_df in aligned.groupby("group"):
                    values = group_df["value"].to_numpy(dtype=float)
                    if len(values) < 3:
                        continue
                    summaries.append(
                        {
                            "group": str(group_value),
                            "median": round(float(np.median(values)), 4),
                            "q1": round(float(np.quantile(values, 0.25)), 4),
                            "q3": round(float(np.quantile(values, 0.75)), 4),
                            "min": round(float(np.min(values)), 4),
                            "max": round(float(np.max(values)), 4),
                            "n": int(len(values)),
                        }
                    )
                if len(summaries) >= 2:
                    grouped_cards.append(
                        {
                            "group_column": group_col,
                            "feature": feature_name,
                            "groups": summaries[:6],
                        }
                    )
            if grouped_cards:
                break
        return grouped_cards[:2]

    def _nearest_distances(
        self,
        query: np.ndarray,
        reference: np.ndarray,
        skip_self: bool = False,
    ) -> np.ndarray:
        distances = []
        for row in query:
            diff = reference - row
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            if skip_self:
                positive = dist[dist > 1e-12]
                distances.append(float(np.min(positive)) if len(positive) else 0.0)
            else:
                distances.append(float(np.min(dist)) if len(dist) else 0.0)
        return np.array(distances, dtype=float)

    def _correlation_interpretation(
        self,
        redundant_pairs: list[dict[str, object]],
        correlation_matrix: dict[str, dict[str, float | None]],
    ) -> str:
        if not correlation_matrix:
            return "Not enough numeric columns were available to compute a correlation matrix."
        if redundant_pairs:
            pairs = ", ".join(
                f"({item['columns'][0]}, {item['columns'][1]}) r={item['correlation']:.2f}"
                for item in redundant_pairs[:3]
            )
            return f"Correlation analysis found highly redundant feature pairs: {pairs}."
        return "Correlation analysis did not find any highly redundant numeric feature pairs above |r| > 0.95."

    def _missingness_interpretation(
        self,
        missingness_type: str,
        missingness_correlations: dict[str, dict[str, float]],
    ) -> str:
        if missingness_type == "minimal":
            return "Missingness is minimal, so structured missing-data patterns are not a major concern in this dataset."
        if not missingness_correlations:
            return "Missingness could not be summarized because no usable missingness matrix was available."
        return (
            f"Missingness pattern is {missingness_type}. "
            "This is a conservative label based on how often columns are missing together."
        )

    def _hopkins_interpretation(self, hopkins_statistic: float | None) -> str:
        if hopkins_statistic is None:
            return "Hopkins statistic was not computed because the numeric matrix was too small or sparse."
        if hopkins_statistic > 0.7:
            strength = "strong"
        elif hopkins_statistic > 0.5:
            strength = "moderate"
        else:
            strength = "weak"
        return (
            f"Hopkins statistic: {hopkins_statistic:.2f}. "
            f"This suggests {strength} cluster tendency in the current numeric feature space."
        )

    def _outlier_interpretation(self, outlier_rows: list[int], numeric_df: pd.DataFrame) -> str:
        usable_rows = numeric_df.shape[0]
        usable_cols = numeric_df.dropna(axis=1, how="all").shape[1] if not numeric_df.empty else 0
        if usable_rows < 10 or usable_cols < 2:
            return "Row-level outlier scoring was skipped because the numeric matrix was too small or sparse."
        if not outlier_rows:
            return "Row-level outlier scoring did not flag any rows."
        return (
            f"Row-level outlier scoring flagged {len(outlier_rows)} row(s) as the most distant "
            "1% of samples from the numeric feature-space center."
        )

    def _distribution_interpretation(self, top_feature_distributions: list[dict[str, object]]) -> str:
        if not top_feature_distributions:
            return "Distribution plots were skipped because there were not enough numeric features with usable values."
        parts = []
        for item in top_feature_distributions[:2]:
            parts.append(
                f"{item['name']} shows a median of {item['median']:.2f} and IQR {item['iqr']:.2f}, "
                "which helps show how wide and skewed the most variable features are."
            )
        return " ".join(parts)

    def _grouped_distribution_interpretation(self, grouped_distributions: list[dict[str, object]]) -> str:
        if not grouped_distributions:
            return "Grouped distribution plots were skipped because no low-cardinality grouping feature lined up with usable numeric signals."
        card = grouped_distributions[0]
        return (
            f"Grouped distributions compare '{card['feature']}' across '{card['group_column']}'. "
            "Differences in medians and spread help show whether simple category splits correspond to meaningful numeric shifts."
        )

    def _heatmap_svg(self, correlation_matrix: dict[str, dict[str, float | None]]) -> str:
        if not correlation_matrix:
            return ""
        columns = list(correlation_matrix.keys())[:12]
        size = 28
        margin = 100
        width = margin + size * len(columns) + 20
        height = margin + size * len(columns) + 20
        rects = []
        x_labels = []
        y_labels = []
        for row_idx, row_name in enumerate(columns):
            y = margin + row_idx * size
            y_labels.append(
                f"<text x='92' y='{y + 18}' font-size='10' text-anchor='end' fill='#1e2a33'>{row_name}</text>"
            )
            for col_idx, col_name in enumerate(columns):
                x = margin + col_idx * size
                if row_idx == 0:
                    x_labels.append(
                        f"<text x='{x + 12}' y='88' font-size='10' text-anchor='start' "
                        f"fill='#1e2a33' transform='rotate(-45 {x + 12} 88)'>{col_name}</text>"
                    )
                value = float(correlation_matrix[row_name][col_name])
                color = self._corr_color(value)
                rects.append(
                    f"<rect x='{x}' y='{y}' width='{size}' height='{size}' fill='{color}' stroke='#ffffff' />"
                )
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:520px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            "<rect width='100%' height='100%' fill='#f8fbfd' rx='12' />"
            + "".join(rects)
            + "".join(x_labels)
            + "".join(y_labels)
            + "</svg>"
        )

    def _corr_color(self, value: float) -> str:
        clipped = max(-1.0, min(1.0, value))
        if clipped >= 0:
            intensity = int(255 - clipped * 120)
            return f"rgb(255,{intensity},{intensity})"
        intensity = int(255 - abs(clipped) * 120)
        return f"rgb({intensity},{intensity},255)"

    def _distribution_grid_svg(self, distributions: list[dict[str, object]]) -> str:
        if not distributions:
            return ""
        card_width = 210
        card_height = 170
        gap = 18
        width = gap + len(distributions) * (card_width + gap)
        height = card_height + gap * 2
        blocks = []
        for idx, item in enumerate(distributions):
            x0 = gap + idx * (card_width + gap)
            y0 = gap
            counts = np.asarray(item["counts"], dtype=float)
            max_count = float(np.max(counts)) if len(counts) else 1.0
            n_bins = max(len(counts), 1)
            bars = []
            for b_idx, count in enumerate(counts):
                bar_w = (card_width - 28) / n_bins
                bar_h = ((count / max_count) * 88) if max_count else 0.0
                bx = x0 + 14 + b_idx * bar_w
                by = y0 + 122 - bar_h
                bars.append(
                    f"<rect x='{bx:.1f}' y='{by:.1f}' width='{max(bar_w - 2, 1):.1f}' height='{bar_h:.1f}' fill='rgba(15,106,139,0.55)' />"
                )
            blocks.append(
                f"<rect x='{x0}' y='{y0}' width='{card_width}' height='{card_height}' rx='14' fill='#ffffff' stroke='#d7e2ea' />"
                f"<text x='{x0 + 14}' y='{y0 + 22}' font-size='13' fill='#193042'>{item['name']}</text>"
                f"<text x='{x0 + 14}' y='{y0 + 38}' font-size='10' fill='#5f7584'>median {item['median']:.2f} | IQR {item['iqr']:.2f}</text>"
                + "".join(bars)
                + f"<line x1='{x0 + 12}' y1='{y0 + 122}' x2='{x0 + card_width - 12}' y2='{y0 + 122}' stroke='#b6c5cf' />"
                + f"<text x='{x0 + 14}' y='{y0 + 142}' font-size='10' fill='#5f7584'>{item['edges'][0]:.2f}</text>"
                + f"<text x='{x0 + card_width - 14}' y='{y0 + 142}' font-size='10' text-anchor='end' fill='#5f7584'>{item['edges'][-1]:.2f}</text>"
            )
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:100%;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            f"<rect width='{width}' height='{height}' fill='#f8fbfd' rx='12' />"
            + "".join(blocks)
            + "</svg>"
        )

    def _grouped_box_svg(self, grouped_distributions: list[dict[str, object]]) -> str:
        if not grouped_distributions:
            return ""
        card = grouped_distributions[0]
        groups = card["groups"]
        width = 440
        height = 260
        margin_left = 58
        margin_right = 18
        margin_top = 34
        margin_bottom = 42
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        all_values = [g[key] for g in groups for key in ("min", "q1", "median", "q3", "max")]
        y_min = float(min(all_values))
        y_max = float(max(all_values))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        def y_scale(value: float) -> float:
            return margin_top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

        boxes = []
        n = len(groups)
        slot = plot_w / max(n, 1)
        for idx, group in enumerate(groups):
            cx = margin_left + idx * slot + slot / 2
            box_w = min(42, slot * 0.55)
            y_q1 = y_scale(group["q1"])
            y_q3 = y_scale(group["q3"])
            y_med = y_scale(group["median"])
            y_minv = y_scale(group["min"])
            y_maxv = y_scale(group["max"])
            boxes.append(
                f"<line x1='{cx:.1f}' y1='{y_minv:.1f}' x2='{cx:.1f}' y2='{y_q1:.1f}' stroke='#0f6a8b' />"
                f"<line x1='{cx:.1f}' y1='{y_q3:.1f}' x2='{cx:.1f}' y2='{y_maxv:.1f}' stroke='#0f6a8b' />"
                f"<rect x='{cx - box_w/2:.1f}' y='{y_q3:.1f}' width='{box_w:.1f}' height='{max(y_q1 - y_q3, 1):.1f}' fill='rgba(239,125,87,0.4)' stroke='#0f6a8b' />"
                f"<line x1='{cx - box_w/2:.1f}' y1='{y_med:.1f}' x2='{cx + box_w/2:.1f}' y2='{y_med:.1f}' stroke='#193042' stroke-width='2' />"
                f"<line x1='{cx - box_w/4:.1f}' y1='{y_minv:.1f}' x2='{cx + box_w/4:.1f}' y2='{y_minv:.1f}' stroke='#0f6a8b' />"
                f"<line x1='{cx - box_w/4:.1f}' y1='{y_maxv:.1f}' x2='{cx + box_w/4:.1f}' y2='{y_maxv:.1f}' stroke='#0f6a8b' />"
                f"<text x='{cx:.1f}' y='{height - 18}' font-size='10' text-anchor='middle' fill='#5f7584'>{group['group']}</text>"
            )
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:520px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            f"<rect width='{width}' height='{height}' fill='#f8fbfd' rx='12' />"
            f"<line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' stroke='#b6c5cf' />"
            f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height - margin_bottom}' stroke='#b6c5cf' />"
            + "".join(boxes)
            + f"<text x='{width / 2:.1f}' y='20' font-size='14' text-anchor='middle' fill='#193042'>{card['feature']} by {card['group_column']}</text>"
            + f"<text x='16' y='{height / 2:.1f}' font-size='11' text-anchor='middle' fill='#5f7584' transform='rotate(-90 16 {height / 2:.1f})'>{card['feature']}</text>"
            + f"<text x='{margin_left - 8:.1f}' y='{margin_top + 6:.1f}' font-size='10' text-anchor='end' fill='#5f7584'>{y_max:.2f}</text>"
            + f"<text x='{margin_left - 8:.1f}' y='{height - margin_bottom + 4:.1f}' font-size='10' text-anchor='end' fill='#5f7584'>{y_min:.2f}</text>"
            + "</svg>"
        )
