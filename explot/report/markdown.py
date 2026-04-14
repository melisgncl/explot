"""Short Markdown summary of a pipeline run — designed to paste into issues or chat."""
from __future__ import annotations

from explot.state import PipelineState


def state_to_markdown(state: PipelineState) -> str:
    lines: list[str] = ["# Explot Report", ""]

    findings = state.results.get("findings")
    verdict = None
    if findings and findings.success:
        verdict = findings.outputs.get("verdict")

    if verdict:
        icon = {
            "SHIP": "✅",
            "INVESTIGATE": "⚠️",
            "DO_NOT_SHIP": "🛑",
            "NO_MODEL": "ℹ️",
        }.get(verdict.get("decision", ""), "•")
        lines.append(f"## Verdict: {icon} {verdict.get('decision', 'UNKNOWN')}")
        lines.append("")
        lines.append(f"> {verdict.get('headline', '')}")
        lines.append("")
        reasons = verdict.get("reasons") or []
        if reasons:
            lines.append("**Reasons:**")
            for r in reasons:
                lines.append(f"- {r}")
            lines.append("")

    profiling = state.results.get("profiling")
    if profiling and profiling.success:
        q = profiling.outputs.get("quality_score")
        n_rows, n_cols = state.raw_df.shape
        lines.append("## Dataset")
        lines.append(f"- Shape: {n_rows} rows × {n_cols} columns")
        if q is not None:
            lines.append(f"- Quality score: **{q}/100**")
        susp = profiling.outputs.get("suspicious_columns", [])
        if susp:
            names = ", ".join(f"`{s['name']}`" for s in susp[:5])
            lines.append(f"- Suspicious columns: {names}")
        lines.append("")

    supervised = state.results.get("supervised")
    if supervised and supervised.success:
        best_models = supervised.outputs.get("best_models", {}) or {}
        sampling_info = supervised.outputs.get("sampling_info", {}) or {}
        if best_models:
            lines.append("## Best Models")
            lines.append("")
            lines.append("| Target | Model | Metric | Score | Balanced | Lift | Trust flags |")
            lines.append("|---|---|---|---|---|---|---|")
            for target, info in best_models.items():
                model = info.get("model", "?")
                metric = info.get("metric", "?")
                score = info.get("mean", 0)
                lift = info.get("lift_over_baseline")
                lift_str = f"{lift:+.3f}" if lift is not None else "—"
                flags = ", ".join(info.get("trust_flags", [])) or "—"
                si = sampling_info.get(target)
                sample_str = f" *(sampled {si['sampled']:,} of {si['total']:,} rows)*" if si else ""
                used_balanced = info.get("diagnostics", {}).get("used_balanced_weights", False)
                balanced_str = "✓" if used_balanced else "—"
                lines.append(f"| `{target}`{sample_str} | {model} | {metric} | {score:.3f} | {balanced_str} | {lift_str} | {flags} |")
            lines.append("")

            # Surface audit details when leakage was detected
            for target, info in best_models.items():
                diag = info.get("diagnostics", {}) or {}
                ga = diag.get("group_audit", {}) or {}
                ta = diag.get("temporal_audit", {}) or {}
                if ga.get("leakage_detected"):
                    lines.append(
                        f"- **Group leakage on `{target}`** — via `{ga.get('id_column')}`: "
                        f"KFold {ga.get('kfold_score'):.2f} → GroupKFold {ga.get('group_score'):.2f} "
                        f"(delta {ga.get('delta'):+.2f})"
                    )
                if ta.get("leakage_detected"):
                    lines.append(
                        f"- **Temporal leakage on `{target}`** — via `{ta.get('time_column')}`: "
                        f"KFold {ta.get('kfold_score'):.2f} → TimeSeriesSplit {ta.get('time_score'):.2f} "
                        f"(delta {ta.get('delta'):+.2f})"
                    )
            # SHAP feature importance per target
            shap_found = False
            for target, info in best_models.items():
                shap_imp = info.get("shap_importance") or []
                if shap_imp:
                    if not shap_found:
                        lines.append("### Feature Importance (SHAP)")
                        lines.append("")
                        shap_found = True
                    lines.append(f"**`{target}`** — top features by mean |SHAP|:")
                    lines.append("")
                    for rank, entry in enumerate(shap_imp[:10], 1):
                        lines.append(f"{rank}. `{entry['feature']}` ({entry['importance']:.4f})")
                    lines.append("")

    if findings and findings.success:
        summary_card = findings.outputs.get("summary_card", [])
        if summary_card:
            lines.append("## Top Findings")
            for f in summary_card:
                lines.append(f"- {f}")
            lines.append("")

        steps = findings.outputs.get("suggested_next_steps", [])
        if steps:
            lines.append("## Suggested Next Steps")
            for s in steps:
                lines.append(f"- {s}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
