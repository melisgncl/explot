"""Survival analysis stage — Kaplan-Meier + Cox Proportional Hazards.

Detects survival data patterns (time-to-event + censoring indicator),
fits KM curves and a CoxPH model, and returns structured results for
the report and FindingsStage.

Requires `lifelines` (soft dependency — stage skips gracefully if absent).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from explot.stages.base import BaseStage, StageMeta, StageResult

_TIME_KEYWORDS = {
    "time", "duration", "days", "months", "years", "survival",
    "os", "pfs", "ttd", "followup", "follow_up", "t_event",
    "event_time", "time_to", "tte",
}
_EVENT_KEYWORDS = {
    "event", "status", "censored", "vital", "dead", "died",
    "outcome", "censor", "death", "observed", "failure",
}

_COLORS = ["#0f6a8b", "#ef7d57", "#2ca02c", "#9467bd", "#8c564b"]


class SurvivalStage(BaseStage):
    meta = StageMeta(
        name="survival",
        depends_on=("profiling",),
        optional_deps=("preprocessing",),
    )

    def run(self, state, config, hooks) -> StageResult:
        hooks.progress(self.meta.name, 10, "Detecting survival columns.")

        try:
            import lifelines  # noqa: F401
        except ImportError:
            return self._skip("lifelines not installed — pip install lifelines")

        df = state.raw_df
        time_col, event_col = self._detect_columns(df)
        if time_col is None or event_col is None:
            return self._skip(
                "No survival pattern detected. "
                "Expected a numeric time column and a binary event column "
                "(names like: time/duration/days + event/status/vital_status)."
            )

        hooks.progress(self.meta.name, 30, f"Fitting KM on '{time_col}' / '{event_col}'.")

        # Clean: drop rows where time or event is NaN; require time > 0
        mask = df[time_col].notna() & df[event_col].notna() & (pd.to_numeric(df[time_col], errors="coerce") > 0)
        sub = df[mask].copy()
        T = pd.to_numeric(sub[time_col], errors="coerce")
        E = pd.to_numeric(sub[event_col], errors="coerce").astype(int)

        if len(T) < 20:
            return self._skip(f"Too few valid rows after cleaning ({len(T)}) for survival analysis.")

        n_events = int(E.sum())
        n_total = len(T)

        # Overall KM
        km_overall = self._fit_km(T, E)
        median_survival = km_overall.get("median")

        # Stratified KM — pick best categorical column
        strat_col, km_stratified = self._fit_stratified_km(sub, T, E, time_col, event_col)

        hooks.progress(self.meta.name, 60, "Fitting Cox PH model.")

        # Cox PH — use numeric + encoded covariates
        cox_summary, cox_concordance = self._fit_cox(sub, T, E, time_col, event_col, state)

        # KM figure (SVG)
        hooks.progress(self.meta.name, 85, "Rendering KM figure.")
        km_svg = self._km_svg(km_overall, km_stratified, strat_col, time_col)

        outputs: dict[str, Any] = {
            "detected": True,
            "time_column": time_col,
            "event_column": event_col,
            "n_total": n_total,
            "n_events": n_events,
            "event_rate": round(n_events / n_total, 4),
            "median_survival": median_survival,
            "km_overall": km_overall,
            "km_stratified": km_stratified,
            "stratify_column": strat_col,
            "cox_summary": cox_summary,
            "cox_concordance": cox_concordance,
        }
        interpretations = {
            "summary": self._interpret(n_total, n_events, median_survival, cox_concordance),
        }
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs=outputs,
            figures={"km_curve": km_svg},
            interpretations=interpretations,
        )

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    def _detect_columns(self, df: pd.DataFrame) -> tuple[str | None, str | None]:
        """Return (time_col, event_col) or (None, None) if not detected."""
        time_col = self._find_col(df, _TIME_KEYWORDS, numeric=True, nonneg=True)
        event_col = self._find_col(df, _EVENT_KEYWORDS, binary=True)
        return time_col, event_col

    def _find_col(
        self, df: pd.DataFrame, keywords: set[str],
        numeric: bool = False, nonneg: bool = False, binary: bool = False,
    ) -> str | None:
        candidates = []
        for col in df.columns:
            name_lower = col.lower().replace("-", "_")
            if not any(kw in name_lower for kw in keywords):
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 10:
                continue
            if numeric and not pd.api.types.is_numeric_dtype(df[col].dropna()):
                # Allow if coercible
                if series.isna().mean() > 0.5:
                    continue
            if nonneg and (series < 0).any():
                continue
            if binary and set(series.unique()) - {0, 1}:
                # Allow 0/1 encoded or True/False
                unique_vals = set(series.dropna().astype(int).unique())
                if unique_vals != {0, 1}:
                    continue
            candidates.append(col)
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # KM fitting
    # ------------------------------------------------------------------

    def _fit_km(self, T: pd.Series, E: pd.Series) -> dict:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmf.fit(T, event_observed=E)
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_survival_function_
        median = float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None
        return {
            "times": sf.index.tolist(),
            "survival": sf.iloc[:, 0].tolist(),
            "ci_lower": ci.iloc[:, 0].tolist(),
            "ci_upper": ci.iloc[:, 1].tolist(),
            "median": median,
        }

    def _fit_stratified_km(
        self, df: pd.DataFrame, T: pd.Series, E: pd.Series,
        time_col: str, event_col: str,
    ) -> tuple[str | None, dict]:
        """Pick the best categorical column for stratification, fit per-group KM."""
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test

        best_col = None
        best_p = 1.0
        cat_cols = [
            c for c in df.columns
            if c not in (time_col, event_col)
            and df[c].dtype == object
            and 2 <= df[c].nunique() <= 6
            and df[c].notna().sum() >= 20
        ]

        for col in cat_cols[:10]:
            groups = df[col].dropna().unique()
            if len(groups) < 2:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = multivariate_logrank_test(T, df[col], E)
                    p = float(result.p_value)
                if p < best_p:
                    best_p = p
                    best_col = col
            except Exception:
                continue

        if best_col is None or best_p > 0.2:
            return None, {}

        # Fit per group
        groups = df[best_col].dropna().unique()
        stratified: dict[str, dict] = {}
        for g in groups:
            mask = df[best_col] == g
            if mask.sum() < 5:
                continue
            kmf = KaplanMeierFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kmf.fit(T[mask], event_observed=E[mask], label=str(g))
            sf = kmf.survival_function_
            stratified[str(g)] = {
                "times": sf.index.tolist(),
                "survival": sf.iloc[:, 0].tolist(),
                "n": int(mask.sum()),
            }

        return best_col, stratified

    # ------------------------------------------------------------------
    # Cox PH
    # ------------------------------------------------------------------

    def _fit_cox(
        self, df: pd.DataFrame, T: pd.Series, E: pd.Series,
        time_col: str, event_col: str, state,
    ) -> tuple[list[dict], float | None]:
        from lifelines import CoxPHFitter

        # Build covariate matrix: numeric cols only, drop time/event
        exclude = {time_col, event_col}
        num_cols = [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() > 1
        ]

        # If preprocessing ran, also include ordinal-encoded columns
        preprocessing = state.results.get("preprocessing")
        if preprocessing and preprocessing.success:
            prep_df = preprocessing.outputs.get("preprocessed_df")
            if prep_df is not None:
                encoded_cols = [
                    c for c in prep_df.columns
                    if c not in exclude and prep_df[c].nunique() > 1
                ]
                # Merge encoded features with T/E
                cox_df = prep_df.loc[df.index[df[time_col].notna() & df[event_col].notna()]].copy()
                cox_df = cox_df[[c for c in encoded_cols if c in cox_df.columns]]
                cox_df[time_col] = T.values
                cox_df[event_col] = E.values
            else:
                cox_df = self._build_cox_df(df, num_cols, T, E, time_col, event_col)
        else:
            cox_df = self._build_cox_df(df, num_cols, T, E, time_col, event_col)

        # Limit to 20 covariates to keep Cox stable
        covariate_cols = [c for c in cox_df.columns if c not in (time_col, event_col)][:20]
        if not covariate_cols:
            return [], None

        cox_df = cox_df[covariate_cols + [time_col, event_col]].dropna()
        if len(cox_df) < 30 or E[cox_df.index].sum() < 10:
            return [], None

        try:
            cph = CoxPHFitter(penalizer=0.1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(cox_df, duration_col=time_col, event_col=event_col)
            summary = cph.summary.reset_index()
            results = []
            for _, row in summary.iterrows():
                results.append({
                    "covariate": str(row.get("covariate", row.iloc[0])),
                    "coef": round(float(row.get("coef", 0)), 4),
                    "exp_coef": round(float(row.get("exp(coef)", 1)), 4),
                    "p": round(float(row.get("p", 1)), 4),
                    "significant": float(row.get("p", 1)) < 0.05,
                })
            results.sort(key=lambda x: x["p"])
            concordance = round(float(cph.concordance_index_), 4)
            return results[:15], concordance
        except Exception:
            return [], None

    def _build_cox_df(self, df, num_cols, T, E, time_col, event_col) -> pd.DataFrame:
        cox_df = df[num_cols].copy()
        cox_df[time_col] = T.values
        cox_df[event_col] = E.values
        return cox_df

    # ------------------------------------------------------------------
    # KM SVG figure
    # ------------------------------------------------------------------

    def _km_svg(
        self,
        km_overall: dict,
        km_stratified: dict,
        strat_col: str | None,
        time_col: str,
    ) -> str:
        W, H = 560, 320
        ml, mr, mt, mb = 60, 20, 20, 50  # margins
        pw = W - ml - mr   # plot width
        ph = H - mt - mb   # plot height

        # Determine x/y ranges
        all_times: list[float] = km_overall.get("times", [])
        max_t = max(all_times) if all_times else 1.0

        def sx(t: float) -> float:
            return ml + (t / max_t) * pw

        def sy(s: float) -> float:
            return mt + ph - s * ph

        lines: list[str] = []

        # Grid lines
        for frac in [0.25, 0.5, 0.75, 1.0]:
            y = sy(frac)
            lines.append(
                f"<line x1='{ml}' y1='{y:.1f}' x2='{ml+pw}' y2='{y:.1f}' "
                "stroke='#d7e2ea' stroke-width='1'/>"
            )

        def _step_path(times: list, survival: list) -> str:
            if not times:
                return ""
            pts = [f"M {sx(times[0]):.1f} {sy(survival[0]):.1f}"]
            for i in range(1, len(times)):
                # Horizontal then vertical (step)
                pts.append(f"H {sx(times[i]):.1f}")
                pts.append(f"V {sy(survival[i]):.1f}")
            return " ".join(pts)

        # CI band for overall
        ci_l = km_overall.get("ci_lower", [])
        ci_u = km_overall.get("ci_upper", [])
        times = km_overall.get("times", [])
        if ci_l and ci_u and not km_stratified:
            # Build polygon for CI band
            fwd = [f"{sx(t):.1f},{sy(s):.1f}" for t, s in zip(times, ci_u)]
            bck = [f"{sx(t):.1f},{sy(s):.1f}" for t, s in reversed(list(zip(times, ci_l)))]
            pts = " ".join(fwd + bck)
            lines.append(f"<polygon points='{pts}' fill='#0f6a8b' fill-opacity='0.12' stroke='none'/>")

        # KM curves
        if km_stratified:
            for i, (grp, data) in enumerate(km_stratified.items()):
                color = _COLORS[i % len(_COLORS)]
                path = _step_path(data["times"], data["survival"])
                if path:
                    lines.append(
                        f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>"
                    )
                # Legend entry
                lx = ml + (i % 3) * 160
                ly = H - 14 + (i // 3) * 14
                lines.append(
                    f"<line x1='{lx}' y1='{ly - 4}' x2='{lx + 18}' y2='{ly - 4}' "
                    f"stroke='{color}' stroke-width='2'/>"
                )
                label = (str(grp)[:18] + "…") if len(str(grp)) > 18 else str(grp)
                lines.append(
                    f"<text x='{lx + 22}' y='{ly}' font-size='10' fill='#193042'>{label}</text>"
                )
        else:
            path = _step_path(km_overall.get("times", []), km_overall.get("survival", []))
            if path:
                lines.append(
                    f"<path d='{path}' fill='none' stroke='#0f6a8b' stroke-width='2.5'/>"
                )
            # Median line
            median = km_overall.get("median")
            if median is not None and median <= max_t:
                xm = sx(median)
                lines.append(
                    f"<line x1='{xm:.1f}' y1='{sy(0.5):.1f}' x2='{xm:.1f}' y2='{sy(0):.1f}' "
                    "stroke='#ef7d57' stroke-width='1.5' stroke-dasharray='4 3'/>"
                )
                lines.append(
                    f"<text x='{xm + 4:.1f}' y='{sy(0.08):.1f}' font-size='10' fill='#ef7d57'>"
                    f"median={median:.1f}</text>"
                )

        # Axes
        lines.append(
            f"<line x1='{ml}' y1='{mt}' x2='{ml}' y2='{mt+ph}' stroke='#193042' stroke-width='1.5'/>"
        )
        lines.append(
            f"<line x1='{ml}' y1='{mt+ph}' x2='{ml+pw}' y2='{mt+ph}' stroke='#193042' stroke-width='1.5'/>"
        )

        # Y axis ticks and labels
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            y = sy(frac)
            lines.append(
                f"<line x1='{ml-4}' y1='{y:.1f}' x2='{ml}' y2='{y:.1f}' stroke='#193042' stroke-width='1'/>"
            )
            lines.append(
                f"<text x='{ml-7}' y='{y+4:.1f}' font-size='10' text-anchor='end' fill='#5f7584'>{frac:.2f}</text>"
            )

        # X axis ticks
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            t = frac * max_t
            x = sx(t)
            lines.append(
                f"<line x1='{x:.1f}' y1='{mt+ph}' x2='{x:.1f}' y2='{mt+ph+4}' stroke='#193042' stroke-width='1'/>"
            )
            lines.append(
                f"<text x='{x:.1f}' y='{mt+ph+16}' font-size='10' text-anchor='middle' fill='#5f7584'>{t:.1f}</text>"
            )

        # Axis labels
        lines.append(
            f"<text x='{ml + pw / 2:.1f}' y='{H - 2}' font-size='11' text-anchor='middle' fill='#193042'>"
            f"{time_col}</text>"
        )
        lines.append(
            f"<text x='12' y='{mt + ph / 2:.1f}' font-size='11' text-anchor='middle' fill='#193042' "
            f"transform='rotate(-90 12 {mt + ph / 2:.1f})'>Survival probability</text>"
        )

        if strat_col:
            lines.append(
                f"<text x='{ml + pw / 2:.1f}' y='{mt + 14}' font-size='11' text-anchor='middle' "
                f"fill='#5f7584'>Stratified by: {strat_col}</text>"
            )

        inner = "\n  ".join(lines)
        return (
            f"<svg viewBox='0 0 {W} {H}' width='100%' xmlns='http://www.w3.org/2000/svg' "
            f"style='max-width:{W}px'>\n  {inner}\n</svg>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _skip(self, reason: str) -> StageResult:
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={"detected": False},
            success=True,
            error=reason,
        )

    def _interpret(self, n: int, n_events: int, median: float | None, concordance: float | None) -> str:
        parts = [f"Survival analysis on {n:,} observations ({n_events} events, {100*n_events/n:.1f}%)."]
        if median is not None:
            parts.append(f"Median survival time: {median:.1f}.")
        if concordance is not None:
            parts.append(f"Cox C-index: {concordance:.3f} ({'good' if concordance > 0.7 else 'moderate' if concordance > 0.6 else 'weak'} discrimination).")
        return " ".join(parts)
