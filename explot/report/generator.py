from __future__ import annotations

import json
from html import escape
from pathlib import Path

from explot.config import AppConfig
from explot.state import PipelineState


class ReportGenerator:
    def render(self, state: PipelineState, config: AppConfig) -> str:
        profiling = state.results.get("profiling")
        exploration = state.results.get("exploration")
        dimensionality = state.results.get("dimensionality")
        autoencoder = state.results.get("autoencoder")
        unsupervised = state.results.get("unsupervised")
        supervised = state.results.get("supervised")
        findings = state.results.get("findings")

        title = escape(config.report.title)
        rows = len(state.raw_df)
        cols = len(state.raw_df.columns)

        overview_cards = [
            self._metric_card("Rows", str(rows), "Dataset height"),
            self._metric_card("Columns", str(cols), "Dataset width"),
        ]
        if profiling and profiling.success:
            overview_cards.extend([
                self._metric_card("Quality Score",
                    str(profiling.outputs.get("quality_score", "n/a")), "Stage 1 summary"),
                self._metric_card("Normalization Guess",
                    str(profiling.outputs.get("normalization_guess", "unknown")), "Heuristic only"),
            ])
        if exploration and exploration.success:
            hopkins = exploration.outputs.get("hopkins_statistic")
            overview_cards.append(self._metric_card("Hopkins",
                "n/a" if hopkins is None else f"{hopkins:.2f}", "Cluster tendency"))
        if unsupervised and unsupervised.success:
            km = unsupervised.outputs.get("kmeans_results", {})
            overview_cards.append(self._metric_card("Clusters",
                f"k={km.get('optimal_k', '?')}", f"Silhouette {km.get('best_silhouette', 0):.2f}"))
        if supervised and supervised.success:
            best = supervised.outputs.get("best_models", {})
            if best:
                first_target = next(iter(best))
                info = best[first_target]
                overview_cards.append(self._metric_card("Best Model",
                    info.get("model", "?"), f"{info.get('metric', '?')}={info.get('mean', 0):.2f}"))

        overview_html = self._tab_overview(state, "".join(overview_cards))
        profiling_html = self._tab_profiling(profiling)
        exploration_html = self._tab_exploration(exploration)
        dimensionality_html = self._tab_dimensionality(dimensionality)
        autoencoder_html = self._tab_autoencoder(autoencoder)
        unsupervised_html = self._tab_unsupervised(unsupervised)
        supervised_html = self._tab_supervised(supervised)
        findings_html = self._tab_findings(findings)

        # Build tab list dynamically based on what ran
        tabs = [("overview", "Overview", overview_html)]
        tabs.append(("profiling", "Profiling", profiling_html))
        tabs.append(("exploration", "Exploration", exploration_html))
        if dimensionality:
            tabs.append(("dimensionality", "Dimensionality", dimensionality_html))
        if autoencoder:
            tabs.append(("autoencoder", "DVAE", autoencoder_html))
        if unsupervised:
            tabs.append(("unsupervised", "Unsupervised", unsupervised_html))
        if supervised:
            tabs.append(("supervised", "Model Selection", supervised_html))
        if findings:
            tabs.append(("findings", "Findings", findings_html))

        tab_buttons = "".join(
            f"<button class='tablink{' active' if i == 0 else ''}' data-tab='{tid}'>{escape(tname)}</button>"
            for i, (tid, tname, _) in enumerate(tabs)
        )
        tab_sections = "".join(
            f"<section id='{tid}' class='tabcontent{' active' if i == 0 else ''}'>{thtml}</section>"
            for i, (tid, _, thtml) in enumerate(tabs)
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f2f7fb;
      --paper: #ffffff;
      --paper-alt: #f8fbfd;
      --ink: #193042;
      --muted: #5f7584;
      --line: #d7e2ea;
      --accent: #0f6a8b;
      --accent-soft: #dceef5;
      --warm: #ef7d57;
      --shadow: 0 16px 40px rgba(19, 48, 71, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,106,139,0.09), transparent 32%),
        radial-gradient(circle at top right, rgba(239,125,87,0.10), transparent 28%),
        linear-gradient(180deg, #eef5fa 0%, var(--bg) 100%);
    }}
    main {{ max-width: 1220px; margin: 0 auto; padding: 32px 20px 56px; }}
    .hero {{
      display: grid;
      gap: 18px;
      grid-template-columns: 1.2fr 0.8fr;
      align-items: stretch;
      margin-bottom: 26px;
    }}
    .hero-card, .summary-card, .panel, .metric, .callout, .table-wrap {{
      background: rgba(255,255,255,0.92);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(215,226,234,0.9);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }}
    .hero-card {{ padding: 28px; }}
    .summary-card {{ padding: 24px; display: grid; gap: 14px; }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.6rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .eyebrow {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 0.85rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }}
    .lede, .muted {{ color: var(--muted); }}
    .tabbar {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 18px 0 24px;
    }}
    .tablink {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.85);
      color: var(--ink);
      border-radius: 999px;
      padding: 11px 16px;
      font: inherit;
      cursor: pointer;
      transition: 0.2s ease;
    }}
    .tablink.active {{
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }}
    .tabcontent {{ display: none; }}
    .tabcontent.active {{ display: block; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .metric {{
      padding: 18px;
      min-height: 120px;
      display: grid;
      align-content: start;
      gap: 6px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }}
    .metric-value {{
      font-size: 1.8rem;
      font-weight: 800;
      letter-spacing: -0.04em;
    }}
    .metric-note {{ color: var(--muted); font-size: 0.95rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 18px;
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-5 {{ grid-column: span 5; }}
    .span-6 {{ grid-column: span 6; }}
    .span-7 {{ grid-column: span 7; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .panel {{ padding: 22px; }}
    .panel h2, .panel h3 {{
      margin-top: 0;
      margin-bottom: 12px;
      letter-spacing: -0.03em;
    }}
    .section-kicker {{
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 10px;
    }}
    .list {{
      margin: 0;
      padding-left: 18px;
      color: var(--ink);
    }}
    .list li {{ margin-bottom: 8px; }}
    .callout {{
      padding: 18px 20px;
      background: linear-gradient(135deg, rgba(15,106,139,0.08), rgba(255,255,255,0.95));
    }}
    .pill {{
      display: inline-block;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.86rem;
      font-weight: 700;
      margin: 4px 6px 0 0;
    }}
    .figure {{
      padding: 12px;
      border-radius: 18px;
      background: var(--paper-alt);
      border: 1px solid var(--line);
      overflow: auto;
    }}
    .table-wrap {{
      padding: 8px;
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    code, pre {{
      font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      background: #f4f8fb;
      padding: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      font-size: 0.88rem;
    }}
    @media (max-width: 920px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 {{ grid-column: span 12; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-card">
        <p class="eyebrow">Explot Report</p>
        <h1>{title}</h1>
        <p class="lede">A self-contained analysis snapshot spanning profiling, exploration, dimensionality reduction, unsupervised probes, supervised model selection, and synthesized findings.</p>
      </div>
      <div class="summary-card">
        <div>
          <div class="metric-label">Dataset Shape</div>
          <div class="metric-value">{rows} x {cols}</div>
          <div class="metric-note">Rows by columns in the input table.</div>
        </div>
        <div>
          <div class="metric-label">Implemented Stages</div>
          <div class="metric-value">{len(state.results)}</div>
          <div class="metric-note">{escape(", ".join(state.results.keys()) or "No stages")}</div>
        </div>
      </div>
    </section>

    <nav class="tabbar" aria-label="Report Tabs">
      {tab_buttons}
    </nav>

    {tab_sections}
  </main>

  <script>
    const buttons = Array.from(document.querySelectorAll('.tablink'));
    const tabs = Array.from(document.querySelectorAll('.tabcontent'));
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        const target = button.dataset.tab;
        buttons.forEach((item) => item.classList.toggle('active', item === button));
        tabs.forEach((tab) => tab.classList.toggle('active', tab.id === target));
      }});
    }});
  </script>
</body>
</html>"""

    def write(self, state: PipelineState, config: AppConfig, output_path: Path) -> None:
        output_path.write_text(self.render(state, config), encoding="utf-8")

    def _tab_overview(self, state: PipelineState, metric_cards: str) -> str:
        profiling = state.results.get("profiling")
        exploration = state.results.get("exploration")
        findings_result = state.results.get("findings")
        supervised = state.results.get("supervised")

        suspicious = profiling.outputs.get("suspicious_columns", []) if profiling and profiling.success else []
        redundant = exploration.outputs.get("redundant_pairs", []) if exploration and exploration.success else []

        suspicious_items = "".join(
            f"<li><strong>{escape(item['name'])}</strong> <span class='muted'>({escape(item['reason'])})</span></li>"
            for item in suspicious[:8]
        ) or "<li>No suspicious columns flagged.</li>"
        redundant_items = "".join(
            f"<li>{escape(item['columns'][0])} / {escape(item['columns'][1])} <span class='muted'>(r={item['correlation']:.2f})</span></li>"
            for item in redundant[:8]
        ) or "<li>No highly redundant numeric pairs detected.</li>"

        # Top findings
        findings_html = ""
        if findings_result and findings_result.success:
            summary = findings_result.outputs.get("summary_card", [])
            if summary:
                items = "".join(f"<li>{escape(f)}</li>" for f in summary)
                findings_html = (
                    "<div class='panel span-12'>"
                    "<div class='section-kicker'>Key Findings</div>"
                    "<h2>Top Findings</h2>"
                    f"<ul class='list'>{items}</ul>"
                    "</div>"
                )

        # Model recommendation
        model_html = ""
        if supervised and supervised.success:
            rec = supervised.interpretations.get("model_recommendation", "")
            if rec:
                model_html = (
                    "<div class='panel span-12'>"
                    "<div class='section-kicker'>Model Selection</div>"
                    "<h2>Recommendation</h2>"
                    f"<div class='callout'>{escape(rec)}</div>"
                    "</div>"
                )

        return (
            f"<div class='metrics'>{metric_cards}</div>"
            "<div class='grid'>"
            + findings_html
            + model_html
            + "<div class='panel span-6'>"
            "<div class='section-kicker'>Review Flags</div>"
            "<h3>Suspicious Columns</h3>"
            f"<ul class='list'>{suspicious_items}</ul>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Redundancy</div>"
            "<h3>Highly Correlated Pairs</h3>"
            f"<ul class='list'>{redundant_items}</ul>"
            "</div>"
            "</div>"
        )

    def _tab_profiling(self, profiling) -> str:
        if not profiling:
            return self._empty_panel("Profiling has not run.")

        outputs = profiling.outputs
        figures = profiling.figures
        fingerprint_svg = figures.get("fingerprint_radar", "")
        suspicious_rows = "".join(
            f"<tr><td>{escape(item['name'])}</td><td>{escape(item['reason'])}</td><td>{escape(item['details'])}</td></tr>"
            for item in outputs.get("suspicious_columns", [])
        ) or "<tr><td colspan='3'>No suspicious columns flagged.</td></tr>"

        top_columns = list(outputs.get("column_profiles", {}).items())[:8]
        profile_rows = []
        for name, profile in top_columns:
            summary = profile.get("summary")
            if summary:
                detail = (
                    f"mean {summary['mean']:.2f}, std {summary['std']:.2f}, "
                    f"zeros {profile.get('zero_percent', 0.0)}%"
                )
            else:
                detail = ", ".join(
                    f"{key}: {value}" for key, value in list(profile.get("top_values", {}).items())[:3]
                ) or "categorical values"
            profile_rows.append(
                f"<tr><td>{escape(name)}</td><td>{escape(str(profile['dtype']))}</td>"
                f"<td>{escape(str(profile.get('role_guess', 'unknown')))}</td>"
                f"<td>{profile['null_percent']}%</td><td>{profile['cardinality']}</td><td>{escape(detail)}</td></tr>"
            )
        profile_rows_html = "".join(profile_rows) or "<tr><td colspan='6'>No column profiles available.</td></tr>"

        quality = outputs.get("quality_breakdown", {})
        quality_cards = "".join(
            self._metric_card(key.replace("_", " ").title(), f"{value:.1f}", "Weighted component")
            for key, value in quality.items()
        )
        role_counts = self._role_counts(outputs.get("column_profiles", {}))
        role_pills = "".join(
            f"<span class='pill'>{escape(role)}: {count}</span>" for role, count in role_counts.items()
        ) or "<span class='muted'>No role guesses available.</span>"

        return (
            "<div class='grid'>"
            "<div class='panel span-8'>"
            "<div class='section-kicker'>Stage 1</div>"
            "<h2>Profiling Summary</h2>"
            f"<p class='muted'>{escape(profiling.interpretations.get('dataset_shape', ''))}</p>"
            f"<div class='callout'><strong>Normalization guess:</strong> {escape(profiling.interpretations.get('normalization_guess', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Quality:</strong> {escape(profiling.interpretations.get('quality_score', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Column roles:</strong> {escape(profiling.interpretations.get('column_roles', ''))}</div>"
            "</div>"
            "<div class='panel span-4'>"
            "<div class='section-kicker'>Fingerprint</div>"
            "<h3>Radar View</h3>"
            f"<div class='figure'>{fingerprint_svg or '<p class=\"muted\">No fingerprint figure available.</p>'}</div>"
            f"<p class='muted' style='margin-bottom:0'>{escape(profiling.interpretations.get('fingerprint_radar', ''))}</p>"
            "</div>"
            f"<div class='span-12'><div class='metrics'>{quality_cards}</div></div>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Role Summary</div>"
            "<h3>Heuristic Column Roles</h3>"
            f"<div>{role_pills}</div>"
            "</div>"
            "<div class='panel span-7'>"
            "<div class='section-kicker'>Columns</div>"
            "<h3>Column Snapshot</h3>"
            "<div class='table-wrap'><table><thead><tr><th>Column</th><th>Type</th><th>Role</th><th>Null %</th><th>Cardinality</th><th>Profile</th></tr></thead><tbody>"
            f"{profile_rows_html}</tbody></table></div>"
            "</div>"
            "<div class='panel span-5'>"
            "<div class='section-kicker'>Flags</div>"
            "<h3>Suspicious Columns</h3>"
            "<div class='table-wrap'><table><thead><tr><th>Name</th><th>Reason</th><th>Details</th></tr></thead><tbody>"
            f"{suspicious_rows}</tbody></table></div>"
            "</div>"
            "</div>"
        )

    def _tab_exploration(self, exploration) -> str:
        if not exploration:
            return self._empty_panel("Exploration has not run.")

        outputs = exploration.outputs
        figures = exploration.figures
        redundant_rows = "".join(
            f"<tr><td>{escape(item['columns'][0])}</td><td>{escape(item['columns'][1])}</td><td>{item['correlation']:.4f}</td></tr>"
            for item in outputs.get("redundant_pairs", [])
        ) or "<tr><td colspan='3'>No highly redundant pairs detected.</td></tr>"
        variable_rows = "".join(
            f"<tr><td>{escape(item['name'])}</td><td>{item['variance']:.6f}</td></tr>"
            for item in outputs.get("top_variable_features", [])
        ) or "<tr><td colspan='2'>No variable features available.</td></tr>"
        grouping_pills = "".join(
            f"<span class='pill'>{escape(item['name'])}: {item['n_groups']} groups</span>"
            for item in outputs.get("grouping_candidates", [])
        ) or "<span class='muted'>No grouping candidates found.</span>"

        hopkins = outputs.get("hopkins_statistic")
        hopkins_text = "n/a" if hopkins is None else f"{hopkins:.2f}"

        return (
            "<div class='grid'>"
            "<div class='panel span-8'>"
            "<div class='section-kicker'>Stage 2</div>"
            "<h2>Exploration Summary</h2>"
            f"<div class='callout'><strong>Correlation:</strong> {escape(exploration.interpretations.get('correlation_heatmap', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Missingness:</strong> {escape(exploration.interpretations.get('missingness_analysis', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Cluster tendency:</strong> {escape(exploration.interpretations.get('hopkins_statistic', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Row outliers:</strong> {escape(exploration.interpretations.get('row_outliers', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Distributions:</strong> {escape(exploration.interpretations.get('distribution_overview', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Grouped shifts:</strong> {escape(exploration.interpretations.get('grouped_distributions', ''))}</div>"
            "</div>"
            "<div class='panel span-4'>"
            "<div class='section-kicker'>Heuristics</div>"
            "<h3>Quick Signals</h3>"
            + self._metric_card("Hopkins", hopkins_text, "Higher suggests stronger structure")
            + self._metric_card("Missingness", str(outputs.get("missingness_type", "unknown")), "Summary label")
            + self._metric_card("Row Outliers", str(len(outputs.get("outlier_rows", []))), "Top 1% farthest rows")
            + "</div>"
            "<div class='panel span-7'>"
            "<div class='section-kicker'>Heatmap</div>"
            "<h3>Correlation Matrix</h3>"
            f"<div class='figure'>{figures.get('correlation_heatmap', '<p class=\"muted\">No heatmap.</p>')}</div>"
            "</div>"
            "<div class='panel span-5'>"
            "<div class='section-kicker'>Grouping</div>"
            "<h3>Low-Cardinality Candidates</h3>"
            f"<div>{grouping_pills}</div>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Redundant Pairs</div>"
            "<h3>Above |r| &gt; 0.95</h3>"
            "<div class='table-wrap'><table><thead><tr><th>Column A</th><th>Column B</th><th>Correlation</th></tr></thead><tbody>"
            f"{redundant_rows}</tbody></table></div>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Variability</div>"
            "<h3>Top Variable Features</h3>"
            "<div class='table-wrap'><table><thead><tr><th>Feature</th><th>Variance</th></tr></thead><tbody>"
            f"{variable_rows}</tbody></table></div>"
            "</div>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Distributions</div>"
            "<h3>Top Numeric Feature Distributions</h3>"
            f"<div class='figure'>{figures.get('distribution_overview', '<p class=\"muted\">No distribution plots.</p>')}</div>"
            "</div>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Grouped View</div>"
            "<h3>Numeric Shift Across A Low-Cardinality Group</h3>"
            f"<div class='figure'>{figures.get('grouped_distributions', '<p class=\"muted\">No grouped distribution plot.</p>')}</div>"
            "</div>"
            "</div>"
        )

    def _tab_dimensionality(self, dim) -> str:
        if not dim or not dim.success:
            return self._empty_panel("Dimensionality stage did not run." if not dim
                                     else f"Dimensionality failed: {dim.error}")
        outputs = dim.outputs
        figures = dim.figures
        transform_items = "".join(
            f"<li>{escape(step)}</li>" for step in outputs.get("transform_log", [])
        ) or "<li>No transforms applied.</li>"
        dropped_rows = "".join(
            f"<tr><td>{escape(d['name'])}</td><td>{escape(d['reason'])}</td></tr>"
            for d in outputs.get("dropped_columns", [])
        ) or "<tr><td colspan='2'>No columns dropped.</td></tr>"

        n50 = outputs.get("n_components_50", 0)
        n80 = outputs.get("n_components_80", 0)
        n95 = outputs.get("n_components_95", 0)
        intrinsic = outputs.get("intrinsic_dim", 0)

        return (
            "<div class='grid'>"
            "<div class='panel span-8'>"
            "<div class='section-kicker'>Stage 3</div>"
            "<h2>Dimensionality Reduction</h2>"
            f"<div class='callout'>{escape(dim.interpretations.get('pca_variance', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'>{escape(dim.interpretations.get('transform_log', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>What PCA/SVD is doing:</strong> {escape(dim.interpretations.get('svd_explainer', ''))}</div>"
            "</div>"
            "<div class='panel span-4'>"
            + self._metric_card("Intrinsic Dim", str(intrinsic), "Participation ratio")
            + self._metric_card("50% Var", f"{n50} PCs", "Components needed")
            + self._metric_card("80% Var", f"{n80} PCs", "Components needed")
            + self._metric_card("95% Var", f"{n95} PCs", "Components needed")
            + "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Variance</div>"
            "<h3>Scree Plot</h3>"
            f"<div class='figure'>{figures.get('scree_plot', '<p class=\"muted\">No scree plot available.</p>')}</div>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Projection</div>"
            "<h3>PC1 vs PC2</h3>"
            f"<div class='figure'>{figures.get('projection_plot', '<p class=\"muted\">No projection available.</p>')}</div>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Transform Log</div>"
            "<h3>What Was Applied</h3>"
            f"<ul class='list'>{transform_items}</ul>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Dropped Columns</div>"
            "<h3>Removed Before PCA</h3>"
            "<div class='table-wrap'><table><thead><tr><th>Column</th><th>Reason</th></tr></thead><tbody>"
            f"{dropped_rows}</tbody></table></div>"
            "</div>"
            "</div>"
        )

    def _tab_autoencoder(self, auto) -> str:
        if not auto or not auto.success:
            return self._empty_panel("DVAE stage did not run." if not auto
                                     else f"DVAE failed: {auto.error}")
        outputs = auto.outputs
        figures = auto.figures
        sample_note = (
            f"Training used {outputs.get('fit_rows', 0)} sampled rows."
            if outputs.get("sampled") else
            f"Training used all {outputs.get('fit_rows', 0)} rows."
        )
        return (
            "<div class='grid'>"
            "<div class='panel span-8'>"
            "<div class='section-kicker'>Stage 4</div>"
            "<h2>DVAE Compression</h2>"
            f"<div class='callout'>{escape(auto.interpretations.get('summary', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Sampling:</strong> {escape(sample_note)}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Training:</strong> {escape(auto.interpretations.get('training_loss', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Reconstruction errors:</strong> {escape(auto.interpretations.get('reconstruction_error', ''))}</div>"
            "</div>"
            "<div class='panel span-4'>"
            + self._metric_card("Latent Dim", str(outputs.get("bottleneck_dim", 0)), "Compression width")
            + self._metric_card("Recon MSE", str(outputs.get("reconstruction_mse", "n/a")), "Lower is better")
            + self._metric_card("Fit Rows", str(outputs.get("fit_rows", 0)), "Rows used during training")
            + "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Latent Space</div>"
            "<h3>First Two Latent Dimensions</h3>"
            f"<div class='figure'>{figures.get('latent_projection', '<p class=\"muted\">No latent projection available.</p>')}</div>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Optimization</div>"
            "<h3>Training Loss</h3>"
            f"<div class='figure'>{figures.get('training_loss', '<p class=\"muted\">No training-loss figure available.</p>')}</div>"
            "</div>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Anomaly Signal</div>"
            "<h3>Reconstruction Error Distribution</h3>"
            f"<div class='figure'>{figures.get('reconstruction_error', '<p class=\"muted\">No reconstruction-error figure available.</p>')}</div>"
            "</div>"
            "</div>"
        )

    def _tab_unsupervised(self, unsup) -> str:
        if not unsup or not unsup.success:
            return self._empty_panel("Unsupervised stage did not run." if not unsup
                                     else f"Unsupervised failed: {unsup.error}")
        km = unsup.outputs.get("kmeans_results", {})
        db = unsup.outputs.get("dbscan_results", {})
        anomaly_rows = unsup.outputs.get("anomaly_rows", [])
        sample_note = (
            f"Sampling note: KMeans/DBSCAN used {unsup.outputs.get('analysis_rows', 0)} rows for analysis."
            if unsup.outputs.get("sampled_for_analysis") else
            f"Sampling note: KMeans/DBSCAN used all {unsup.outputs.get('analysis_rows', 0)} rows."
        )

        sil_rows = "".join(
            f"<tr><td>{k}</td><td>{score:.4f}</td></tr>"
            for k, score in sorted(km.get("silhouette_scores", {}).items())
        ) or "<tr><td colspan='2'>No silhouette data.</td></tr>"

        size_items = "".join(
            f"<li>Cluster {label}: {count} rows</li>"
            for label, count in sorted(km.get("cluster_sizes", {}).items())
        ) or "<li>No cluster data.</li>"

        return (
            "<div class='grid'>"
            "<div class='panel span-8'>"
            "<div class='section-kicker'>Stage 5</div>"
            "<h2>Unsupervised Probes</h2>"
            f"<div class='callout'><strong>KMeans:</strong> {escape(unsup.interpretations.get('kmeans_silhouette', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>DBSCAN:</strong> {escape(unsup.interpretations.get('dbscan_results', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Anomalies:</strong> {escape(unsup.interpretations.get('isolation_forest', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Signal agreement:</strong> {escape(unsup.interpretations.get('anomaly_signal_comparison', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Representation:</strong> {escape(str(unsup.outputs.get('representation_source', 'dimensionality')))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Sampling:</strong> {escape(sample_note)}</div>"
            "</div>"
            "<div class='panel span-4'>"
            + self._metric_card("Optimal k", str(km.get("optimal_k", "?")), f"Silhouette {km.get('best_silhouette', 0):.2f}")
            + self._metric_card("DBSCAN Clusters", str(db.get("n_clusters", "?")), f"Noise {db.get('noise_fraction', 0):.1%}")
            + self._metric_card("Anomalies", str(len(anomaly_rows)), "Isolation Forest top 1%")
            + self._metric_card("DVAE Tail", str(len(unsup.outputs.get("dvae_anomaly_rows", []))), "Top 1% reconstruction errors")
            + "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Silhouette Sweep</div>"
            "<h3>Score by k</h3>"
            "<div class='table-wrap'><table><thead><tr><th>k</th><th>Silhouette</th></tr></thead><tbody>"
            f"{sil_rows}</tbody></table></div>"
            "</div>"
            "<div class='panel span-6'>"
            "<div class='section-kicker'>Cluster Sizes</div>"
            f"<h3>At k={km.get('optimal_k', '?')}</h3>"
            f"<ul class='list'>{size_items}</ul>"
            "</div>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Anomaly Agreement</div>"
            "<h3>Isolation Forest vs DVAE vs Stage 2</h3>"
            "<div class='table-wrap'><table><thead><tr><th>Signal</th><th>Count</th><th>Notes</th></tr></thead><tbody>"
            f"<tr><td>Isolation Forest</td><td>{len(unsup.outputs.get('anomaly_rows', []))}</td><td>Top 1% anomaly scores.</td></tr>"
            f"<tr><td>DVAE reconstruction tail</td><td>{len(unsup.outputs.get('dvae_anomaly_rows', []))}</td><td>Top 1% row-level reconstruction errors.</td></tr>"
            f"<tr><td>Stage 2 outliers</td><td>{len(unsup.outputs.get('cluster_outlier_overlap', {}).get('stage2_outlier_rows', []))}</td><td>Distance-based row outliers from exploration.</td></tr>"
            f"<tr><td>Isolation Forest ∩ DVAE</td><td>{unsup.outputs.get('anomaly_signal_comparison', {}).get('iso_dvae_overlap_count', 0)}</td><td>Rows supported by both anomaly models.</td></tr>"
            f"<tr><td>All three signals</td><td>{unsup.outputs.get('anomaly_signal_comparison', {}).get('triple_overlap_count', 0)}</td><td>Highest-confidence anomaly candidates.</td></tr>"
            "</tbody></table></div>"
            "</div>"
            "</div>"
        )

    def _tab_supervised(self, sup) -> str:
        if not sup or not sup.success:
            return self._empty_panel("Model Selection stage did not run." if not sup
                                     else f"Model Selection failed: {sup.error}")
        model_results_a = sup.outputs.get("model_results_track_a", {})
        model_results_b = sup.outputs.get("model_results_track_b", {})
        best_models = sup.outputs.get("best_models", {})
        feature_importances = sup.outputs.get("feature_importances", {})
        track_comparison = sup.outputs.get("track_comparison", {})
        evaluation_details = sup.outputs.get("evaluation_details", {})
        sampling_notes = sup.outputs.get("sampling_notes", [])

        targets_html = ""
        for target, best in best_models.items():
            results_a = model_results_a.get(target, [])
            results_b = model_results_b.get(target, [])
            best = best_models.get(target, {})
            rows_a = "".join(
                f"<tr><td>{escape(r['model'])}</td><td>{r['mean']:.4f} +/- {r['std']:.4f}</td>"
                f"<td>{r.get('precision_macro', 'n/a')}</td><td>{r.get('recall_macro', 'n/a')}</td>"
                f"<td>{r.get('accuracy', 'n/a')}</td><td>{r.get('roc_auc', 'n/a')}</td></tr>"
                for r in sorted(results_a, key=lambda x: -x["mean"])
            ) or "<tr><td colspan='6'>Track A unavailable.</td></tr>"
            rows_b = "".join(
                f"<tr><td>{escape(r['model'])}</td><td>{r['mean']:.4f} +/- {r['std']:.4f}</td>"
                f"<td>{r.get('precision_macro', 'n/a')}</td><td>{r.get('recall_macro', 'n/a')}</td>"
                f"<td>{r.get('accuracy', 'n/a')}</td><td>{r.get('roc_auc', 'n/a')}</td></tr>"
                for r in sorted(results_b, key=lambda x: -x["mean"])
            ) or "<tr><td colspan='6'>Track B unavailable.</td></tr>"

            fi_rows = ""
            for track_name, feats in feature_importances.get(target, {}).items():
                fi_rows += "".join(
                    f"<tr><td>{escape(track_name)}</td><td>{escape(f['feature'])}</td><td>{f['importance']:.4f}</td></tr>"
                    for f in feats[:5]
                )
            fi_rows = fi_rows or "<tr><td colspan='3'>No feature importance data.</td></tr>"

            cmp = track_comparison.get(target, {})
            cmp_text = cmp.get("summary", "No track comparison available.")
            best_track = best.get("track", "track_a")
            metrics = evaluation_details.get(target, {}).get(best_track, {}).get("metrics", {})
            labels = evaluation_details.get(target, {}).get(best_track, {}).get("labels", [])
            matrix = evaluation_details.get(target, {}).get(best_track, {}).get("confusion_matrix", [])
            cm_html = self._confusion_table(labels, matrix)

            targets_html += (
                f"<div class='panel span-6'>"
                f"<div class='section-kicker'>Target: {escape(target)}</div>"
                f"<h3>Track A: Original Features</h3>"
                f"<div class='table-wrap'><table><thead><tr><th>Model</th><th>F1/R2</th><th>Precision</th><th>Recall</th><th>Accuracy</th><th>ROC AUC</th></tr></thead><tbody>"
                f"{rows_a}</tbody></table></div>"
                f"<h3 style='margin-top:16px'>Track B: Latent Features</h3>"
                f"<div class='table-wrap'><table><thead><tr><th>Model</th><th>F1/R2</th><th>Precision</th><th>Recall</th><th>Accuracy</th><th>ROC AUC</th></tr></thead><tbody>"
                f"{rows_b}</tbody></table></div>"
                f"</div>"
                f"<div class='panel span-6'>"
                f"<div class='section-kicker'>Trust And Diagnostics</div>"
                f"<h3>Best Track: {escape(best_track)}</h3>"
                f"<div class='callout'><strong>Track comparison:</strong> {escape(cmp_text)}</div>"
                f"<div class='callout' style='margin-top:12px'><strong>Best-model metrics:</strong> "
                f"precision {metrics.get('precision_macro', 'n/a')}, recall {metrics.get('recall_macro', 'n/a')}, "
                f"F1 {metrics.get('f1_macro', best.get('mean', 'n/a'))}, accuracy {metrics.get('accuracy', 'n/a')}, "
                f"ROC AUC {metrics.get('roc_auc', 'n/a')}.</div>"
                f"<div style='margin-top:12px'>{cm_html}</div>"
                f"<h3 style='margin-top:16px'>Feature Importance</h3>"
                f"<div class='table-wrap'><table><thead><tr><th>Track</th><th>Feature</th><th>Importance</th></tr></thead><tbody>"
                f"{fi_rows}</tbody></table></div>"
                f"</div>"
            )

        return (
            "<div class='grid'>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Stage 6</div>"
            "<h2>Model Selection</h2>"
            f"<div class='callout'>{escape(sup.interpretations.get('model_recommendation', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Track comparison:</strong> {escape(sup.interpretations.get('track_comparison', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Trust notes:</strong> {escape(sup.interpretations.get('trust_notes', ''))}</div>"
            f"<div class='callout' style='margin-top:12px'><strong>Sampling:</strong> {escape(' '.join(sampling_notes) if sampling_notes else 'No target-level sampling was needed.')}</div>"
            "<div class='callout' style='margin-top:12px'><strong>What the models are doing:</strong> "
            "each candidate target is predicted from the transformed feature matrix using cross-validation, "
            "so the reported score estimates how well that target can be inferred from the rest of the dataset rather than from one lucky train/test split.</div>"
            "<div class='callout' style='margin-top:12px'><strong>How feature importance is calculated:</strong> "
            "Explot fits a Random Forest probe for each target and ranks features by how much they reduce impurity across the trees. "
            "Higher importance means that feature helped the forest split the data into purer target groups more often.</div>"
            "</div>"
            + targets_html
            + "</div>"
        )

    def _tab_findings(self, findings) -> str:
        if not findings or not findings.success:
            return self._empty_panel("Findings stage did not run." if not findings
                                     else f"Findings failed: {findings.error}")
        findings_list = findings.outputs.get("findings_list", [])
        next_steps = findings.outputs.get("suggested_next_steps", [])

        conf_colors = {"HIGH": "#0f6a8b", "MEDIUM": "#5f7584", "LOW": "#aab8c2"}
        finding_items = "".join(
            f"<tr><td><span class='pill' style='background:{conf_colors.get(f['confidence'], '#aab8c2')}20;"
            f"color:{conf_colors.get(f['confidence'], '#aab8c2')}'>"
            f"{escape(f['confidence'])}</span></td>"
            f"<td>{escape(f['text'])}</td>"
            f"<td class='muted'>{escape(f['source_stage'])}</td></tr>"
            for f in findings_list
        ) or "<tr><td colspan='3'>No findings generated.</td></tr>"

        steps_items = "".join(f"<li>{escape(s)}</li>" for s in next_steps) or "<li>No suggestions.</li>"

        return (
            "<div class='grid'>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Stage 7</div>"
            "<h2>Findings</h2>"
            f"<p class='muted'>{escape(findings.interpretations.get('summary', ''))}</p>"
            "<div class='table-wrap'><table><thead><tr><th>Confidence</th><th>Finding</th><th>Source</th></tr></thead><tbody>"
            f"{finding_items}</tbody></table></div>"
            "</div>"
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Next Steps</div>"
            "<h2>Suggested Actions</h2>"
            f"<ul class='list'>{steps_items}</ul>"
            "</div>"
            "</div>"
        )

    def _metric_card(self, label: str, value: str, note: str) -> str:
        return (
            "<article class='metric'>"
            f"<div class='metric-label'>{escape(label)}</div>"
            f"<div class='metric-value'>{escape(value)}</div>"
            f"<div class='metric-note'>{escape(note)}</div>"
            "</article>"
        )

    def _empty_panel(self, message: str) -> str:
        return (
            "<div class='panel span-12'>"
            "<div class='section-kicker'>Unavailable</div>"
            f"<h2>{escape(message)}</h2>"
            "</div>"
        )

    def _confusion_table(self, labels: list[str], matrix: list[list[int]]) -> str:
        if not labels or not matrix:
            return "<p class='muted'>No confusion matrix available for this target.</p>"
        header = "".join(f"<th>{escape(label)}</th>" for label in labels)
        rows = []
        for label, values in zip(labels, matrix):
            cells = "".join(f"<td>{int(value)}</td>" for value in values)
            rows.append(f"<tr><th>{escape(label)}</th>{cells}</tr>")
        return (
            "<div class='table-wrap'><table><thead><tr><th>True \\ Pred</th>"
            + header
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )

    def _role_counts(self, column_profiles: dict[str, dict[str, object]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for profile in column_profiles.values():
            role = str(profile.get("role_guess", "unknown"))
            counts[role] = counts.get(role, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
