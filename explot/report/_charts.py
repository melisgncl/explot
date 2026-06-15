from __future__ import annotations

import json
from html import escape


class _ChartMixin:
    """Plotly and SHAP chart builders extracted from ReportGenerator."""

    def _projection_plotly_chart(self, outputs: dict) -> str:
        """Render PCA or UMAP 2D scatter as interactive Plotly chart."""
        import numpy as np

        umap_2d = outputs.get("umap_2d")
        pca_2d = outputs.get("pca_2d")

        # Prefer UMAP if available and non-empty
        if umap_2d is not None and isinstance(umap_2d, np.ndarray) and umap_2d.size > 0:
            pts = umap_2d
            title = "UMAP 2D Projection"
            x_label, y_label = "UMAP 1", "UMAP 2"
        elif pca_2d is not None and isinstance(pca_2d, np.ndarray) and pca_2d.size > 0:
            pts = pca_2d
            ev = outputs.get("explained_variance_ratio_", [0, 0])
            x_pct = f" ({100*ev[0]:.1f}%)" if len(ev) > 0 else ""
            y_pct = f" ({100*ev[1]:.1f}%)" if len(ev) > 1 else ""
            title = "PCA 2D Projection"
            x_label = f"PC1{x_pct}"
            y_label = f"PC2{y_pct}"
        else:
            return "<h3>PC1 vs PC2</h3><p class='muted'>No projection available.</p>"

        # Subsample to 2000 for lightweight HTML
        n = len(pts)
        if n > 2000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=2000, replace=False)
            pts = pts[idx]

        traces = [{
            "type": "scatter",
            "mode": "markers",
            "x": [round(float(p[0]), 4) for p in pts],
            "y": [round(float(p[1]), 4) for p in pts],
            "marker": {
                "color": "#0f6a8b",
                "size": 4,
                "opacity": 0.5,
                "line": {"width": 0},
            },
            "hovertemplate": f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>",
            "showlegend": False,
        }]
        layout = {
            "height": 280,
            "margin": {"l": 50, "r": 10, "t": 30, "b": 50},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(255,255,255,0.5)",
            "font": {"family": "Segoe UI, Arial, sans-serif", "size": 11, "color": "#193042"},
            "title": {"text": title, "font": {"size": 12}, "x": 0.5},
            "xaxis": {"title": x_label, "gridcolor": "#d7e2ea", "zeroline": False},
            "yaxis": {"title": y_label, "gridcolor": "#d7e2ea", "zeroline": False},
        }
        config_obj = {"displayModeBar": False, "responsive": True}
        return (
            f"<h3>{title}</h3>"
            "<div id='proj_chart' style='width:100%;min-height:280px'></div>"
            f"<script>Plotly.newPlot('proj_chart',{json.dumps(traces)},"
            f"{json.dumps(layout)},{json.dumps(config_obj)});</script>"
        )

    def _shap_waterfall_chart(self, waterfall: dict, target: str) -> str:
        """Return an inline Plotly waterfall chart for a single-sample SHAP breakdown."""
        if not waterfall or not waterfall.get("features"):
            return ""
        features = waterfall["features"]  # [{feature, value, shap}, ...]
        base = waterfall.get("base_value", 0.0)
        pred = waterfall.get("prediction", None)

        # Show features ordered bottom-to-top for horizontal waterfall readability
        names = [f"{f['feature']} = {f['value']:.3g}" for f in features][::-1]
        values = [f["shap"] for f in features][::-1]
        colors = ["#0f6a8b" if v >= 0 else "#ef7d57" for v in values]

        safe_id = "shap_wf_" + "".join(c if c.isalnum() else "_" for c in target)
        height = max(240, 28 * len(names) + 80)

        traces = [{
            "type": "bar",
            "orientation": "h",
            "x": values,
            "y": names,
            "marker": {"color": colors, "opacity": 0.85},
            "hovertemplate": "%{y}<br>SHAP: %{x:.4f}<extra></extra>",
            "name": "SHAP contribution",
        }]
        pred_annotation = f" → prediction: {pred:.3f}" if pred is not None else ""
        layout = {
            "height": height,
            "margin": {"l": 10, "r": 10, "t": 30, "b": 30},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Segoe UI, Arial, sans-serif", "size": 11, "color": "#193042"},
            "title": {"text": f"Base: {base:.3f}{pred_annotation}", "font": {"size": 11, "color": "#5f7584"}, "x": 0},
            "xaxis": {"title": "SHAP value", "gridcolor": "#d7e2ea", "zeroline": True,
                      "zerolinecolor": "#193042", "zerolinewidth": 1},
            "yaxis": {"automargin": True, "tickfont": {"size": 10}},
            "showlegend": False,
        }
        config_obj = {"displayModeBar": False, "responsive": True}
        return (
            f"<div id='{safe_id}' style='width:100%;min-height:{height}px'></div>"
            f"<script>Plotly.newPlot('{safe_id}',{json.dumps(traces)},"
            f"{json.dumps(layout)},{json.dumps(config_obj)});</script>"
        )

    def _fi_with_dropdown(self, fi_by_track: dict, shap_importance: list, target: str) -> str:
        """Feature importance chart with a dropdown to toggle between RF impurity and SHAP."""
        safe = "".join(c if c.isalnum() else "_" for c in target)
        rf_id = f"fi_rf_{safe}"
        shap_id = f"fi_shap_{safe}"
        select_id = f"fi_sel_{safe}"

        rf_html = self._feature_importance_chart(fi_by_track, target, chart_id=rf_id)
        # SHAP importance: single-track, already a flat list [{feature, importance}]
        shap_fi_by_track = {"SHAP": shap_importance} if shap_importance else {}
        shap_html = self._feature_importance_chart(shap_fi_by_track, target, chart_id=shap_id)
        has_shap = bool(shap_importance)

        if not has_shap:
            return rf_html

        # Wrap with toggle controls
        toggle_js = (
            f"function _fiToggle_{safe}(v){{"
            f"document.getElementById('{rf_id}').style.display=v==='rf'?'block':'none';"
            f"document.getElementById('{shap_id}').style.display=v==='shap'?'block':'none';"
            f"}}"
        )
        select_html = (
            f"<select id='{select_id}' onchange=\"_fiToggle_{safe}(this.value)\" "
            f"style='margin-bottom:6px;font-size:0.82rem;padding:3px 8px;"
            f"border:1px solid #d7e2ea;border-radius:8px;background:#f8fbfd;color:#193042'>"
            f"<option value='rf'>RF Impurity</option>"
            f"<option value='shap'>SHAP</option>"
            f"</select>"
        )
        shap_div = f"<div id='{shap_id}' style='display:none'>{shap_html}</div>"
        rf_div = f"<div id='{rf_id}'>{rf_html}</div>"
        return f"<script>{toggle_js}</script>{select_html}{rf_div}{shap_div}"

    def _feature_importance_chart(self, fi_by_track: dict, target: str, chart_id: str = "") -> str:
        """Return an inline Plotly horizontal bar chart for feature importances."""
        if not fi_by_track:
            return "<p class='muted'>No feature importance data.</p>"

        _TRACK_COLORS = {"track_a": "#0f6a8b", "track_b": "#ef7d57"}
        _TRACK_LABELS = {"track_a": "PCA Features", "track_b": "DVAE Features"}

        traces = []
        for track_name, feats in fi_by_track.items():
            top = sorted(feats, key=lambda f: f["importance"], reverse=True)[:10]
            if not top:
                continue
            features = [f["feature"] for f in top][::-1]   # reversed → most important at top
            importances = [round(f["importance"], 6) for f in top][::-1]
            traces.append({
                "type": "bar",
                "orientation": "h",
                "name": _TRACK_LABELS.get(track_name, track_name),
                "x": importances,
                "y": features,
                "marker": {"color": _TRACK_COLORS.get(track_name, "#5f7584"), "opacity": 0.85},
                "hovertemplate": "%{y}: %{x:.4f}<extra></extra>",
            })

        if not traces:
            return "<p class='muted'>No feature importance data.</p>"

        safe_id = chart_id or ("fi_" + "".join(c if c.isalnum() else "_" for c in target))
        n_feats = max(len(t["y"]) for t in traces)
        height = max(220, 32 * n_feats + 60)

        layout = {
            "barmode": "group",
            "height": height,
            "margin": {"l": 10, "r": 10, "t": 10, "b": 30},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Segoe UI, Arial, sans-serif", "size": 12, "color": "#193042"},
            "xaxis": {"title": "Importance", "gridcolor": "#d7e2ea", "zeroline": False},
            "yaxis": {"automargin": True, "tickfont": {"size": 11}},
            "legend": {"orientation": "h", "y": -0.12},
            "showlegend": len(traces) > 1,
        }
        config_obj = {"displayModeBar": False, "responsive": True}

        traces_json = json.dumps(traces)
        layout_json = json.dumps(layout)
        config_json = json.dumps(config_obj)

        return (
            f"<div id='{safe_id}' style='width:100%;min-height:{height}px'></div>"
            f"<script>Plotly.newPlot('{safe_id}',{traces_json},{layout_json},{config_json});</script>"
        )

    def _km_plotly_chart(self, out: dict, time_col: str) -> str:
        """Render Kaplan-Meier curves as an interactive Plotly step chart."""
        km_overall = out.get("km_overall", {})
        km_stratified = out.get("km_stratified", {})
        if not km_overall.get("times"):
            return "<p class='muted'>KM figure unavailable.</p>"

        _COLORS = ["#0f6a8b", "#ef7d57", "#2ca02c", "#9467bd", "#8c564b"]
        traces = []

        if km_stratified:
            for i, (grp, data) in enumerate(km_stratified.items()):
                color = _COLORS[i % len(_COLORS)]
                traces.append({
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"shape": "hv", "color": color, "width": 2},
                    "x": data["times"],
                    "y": data["survival"],
                    "name": str(grp),
                    "hovertemplate": f"Group {escape(str(grp))}<br>Time: %{{x}}<br>Survival: %{{y:.3f}}<extra></extra>",
                })
        else:
            # CI band
            ci_lower = km_overall.get("ci_lower", [])
            ci_upper = km_overall.get("ci_upper", [])
            times = km_overall["times"]
            if ci_lower and ci_upper:
                traces.append({
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"shape": "hv", "color": "rgba(15,106,139,0.0)", "width": 0},
                    "x": times + times[::-1],
                    "y": ci_upper + ci_lower[::-1],
                    "fill": "toself",
                    "fillcolor": "rgba(15,106,139,0.12)",
                    "name": "95% CI",
                    "hoverinfo": "skip",
                    "showlegend": False,
                })
            traces.append({
                "type": "scatter",
                "mode": "lines",
                "line": {"shape": "hv", "color": "#0f6a8b", "width": 2.5},
                "x": times,
                "y": km_overall["survival"],
                "name": "Overall",
                "hovertemplate": "Time: %{x}<br>Survival: %{y:.3f}<extra></extra>",
            })
            # Median annotation
            median = out.get("median_survival")
            if median is not None:
                traces.append({
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#ef7d57", "width": 1.5, "dash": "dash"},
                    "x": [median, median],
                    "y": [0, 0.5],
                    "name": f"Median={median:.1f}",
                    "hovertemplate": f"Median={median:.1f}<extra></extra>",
                })

        layout = {
            "height": 300,
            "margin": {"l": 50, "r": 20, "t": 20, "b": 50},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(255,255,255,0.5)",
            "font": {"family": "Segoe UI, Arial, sans-serif", "size": 11, "color": "#193042"},
            "xaxis": {"title": time_col, "gridcolor": "#d7e2ea"},
            "yaxis": {"title": "Survival probability", "range": [0, 1.02], "gridcolor": "#d7e2ea"},
            "legend": {"orientation": "h", "y": -0.18},
            "showlegend": bool(km_stratified) or out.get("median_survival") is not None,
        }
        config_obj = {"displayModeBar": False, "responsive": True}
        return (
            "<div id='km_chart' style='width:100%;min-height:300px'></div>"
            f"<script>Plotly.newPlot('km_chart',{json.dumps(traces)},"
            f"{json.dumps(layout)},{json.dumps(config_obj)});</script>"
        )
