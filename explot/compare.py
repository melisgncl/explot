"""Compare two explot JSON report files and surface the diff as markdown.

Usage:
    explot compare report_v1.json report_v2.json
    explot compare report_v1.json report_v2.json --output diff.md

The diff covers:
  - Quality score delta
  - Best model + score delta per target
  - Trust flags added / removed per target
  - Features added / removed (track_c permutation importance)
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(report: dict, name: str) -> dict:
    return report.get("stages", {}).get(name, {})


def _outputs(report: dict, stage: str) -> dict:
    return _stage(report, stage).get("outputs") or {}


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def diff_reports(report_a: dict, report_b: dict) -> dict:
    diff: dict = {}

    # Quality score
    qa = _outputs(report_a, "profiling").get("quality_score")
    qb = _outputs(report_b, "profiling").get("quality_score")
    diff["quality"] = {"a": qa, "b": qb, "delta": _delta(qa, qb)}

    # Per-target model scores and trust flags
    bm_a = _outputs(report_a, "supervised").get("best_models") or {}
    bm_b = _outputs(report_b, "supervised").get("best_models") or {}
    all_targets = sorted(set(bm_a) | set(bm_b))

    targets: list[dict] = []
    for target in all_targets:
        info_a = bm_a.get(target, {})
        info_b = bm_b.get(target, {})
        score_a = info_a.get("mean")
        score_b = info_b.get("mean")
        flags_a = set(info_a.get("trust_flags") or [])
        flags_b = set(info_b.get("trust_flags") or [])
        targets.append({
            "target": target,
            "model_a": info_a.get("model", "—"),
            "model_b": info_b.get("model", "—"),
            "score_a": score_a,
            "score_b": score_b,
            "score_delta": _delta(score_a, score_b),
            "flags_added": sorted(flags_b - flags_a),
            "flags_removed": sorted(flags_a - flags_b),
        })
    diff["targets"] = targets

    # Verdict
    v_a = _outputs(report_a, "findings").get("verdict") or {}
    v_b = _outputs(report_b, "findings").get("verdict") or {}
    diff["verdict"] = {
        "a": v_a.get("decision", "—"),
        "b": v_b.get("decision", "—"),
        "changed": v_a.get("decision") != v_b.get("decision"),
    }

    # Feature sets (track_c permutation importance, first target)
    feats_a: set[str] = set()
    feats_b: set[str] = set()
    fi_a = _outputs(report_a, "supervised").get("feature_importances") or {}
    fi_b = _outputs(report_b, "supervised").get("feature_importances") or {}
    for target in all_targets:
        for entry in (fi_a.get(target) or {}).get("track_c") or []:
            feats_a.add(entry["feature"])
        for entry in (fi_b.get(target) or {}).get("track_c") or []:
            feats_b.add(entry["feature"])
    diff["features"] = {
        "added": sorted(feats_b - feats_a),
        "removed": sorted(feats_a - feats_b),
    }

    return diff


def _delta(a, b) -> float | None:
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 4)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def diff_to_markdown(diff: dict, label_a: str = "v1", label_b: str = "v2") -> str:
    lines: list[str] = [f"# Explot Comparison: {label_a} → {label_b}", ""]

    # Verdict change
    v = diff["verdict"]
    if v["changed"]:
        lines.append(f"> Verdict changed: **{v['a']}** → **{v['b']}**")
    else:
        lines.append(f"> Verdict unchanged: **{v['a']}**")
    lines.append("")

    # Quality score
    q = diff["quality"]
    lines.append("## Quality Score")
    lines.append("")
    lines.append(f"| | {label_a} | {label_b} | Delta |")
    lines.append("|---|---|---|---|")
    q_a = f"{q['a']}/100" if q["a"] is not None else "—"
    q_b = f"{q['b']}/100" if q["b"] is not None else "—"
    d = f"{q['delta']:+.0f}" if q["delta"] is not None else "—"
    lines.append(f"| Quality | {q_a} | {q_b} | {d} |")
    lines.append("")

    # Best models
    lines.append("## Best Models")
    lines.append("")
    lines.append(f"| Target | {label_a} Model | {label_a} Score | {label_b} Model | {label_b} Score | Delta |")
    lines.append("|---|---|---|---|---|---|")
    for t in diff["targets"]:
        s_a = f"{t['score_a']:.3f}" if t["score_a"] is not None else "—"
        s_b = f"{t['score_b']:.3f}" if t["score_b"] is not None else "—"
        d = f"{t['score_delta']:+.3f}" if t["score_delta"] is not None else "—"
        lines.append(f"| `{t['target']}` | {t['model_a']} | {s_a} | {t['model_b']} | {s_b} | {d} |")
    lines.append("")

    # Trust flag changes
    flag_changes = [t for t in diff["targets"] if t["flags_added"] or t["flags_removed"]]
    if flag_changes:
        lines.append("## Trust Flag Changes")
        lines.append("")
        lines.append("| Target | Added | Removed |")
        lines.append("|---|---|---|")
        for t in flag_changes:
            added = ", ".join(f"`{f}`" for f in t["flags_added"]) or "—"
            removed = ", ".join(f"`{f}`" for f in t["flags_removed"]) or "—"
            lines.append(f"| `{t['target']}` | {added} | {removed} |")
        lines.append("")
    else:
        lines.append("## Trust Flag Changes")
        lines.append("")
        lines.append("No trust flag changes.")
        lines.append("")

    # Feature changes
    feats = diff["features"]
    if feats["added"] or feats["removed"]:
        lines.append("## Feature Changes")
        lines.append("")
        if feats["added"]:
            sample = feats["added"][:8]
            more = f" (+{len(feats['added']) - 8} more)" if len(feats["added"]) > 8 else ""
            lines.append(f"**Added ({len(feats['added'])}):** {', '.join(f'`{f}`' for f in sample)}{more}")
        if feats["removed"]:
            sample = feats["removed"][:8]
            more = f" (+{len(feats['removed']) - 8} more)" if len(feats["removed"]) > 8 else ""
            lines.append(f"**Removed ({len(feats['removed'])}):** {', '.join(f'`{f}`' for f in sample)}{more}")
        lines.append("")
    else:
        lines.append("## Feature Changes")
        lines.append("")
        lines.append("No feature changes detected.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
