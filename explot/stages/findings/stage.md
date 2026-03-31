---
name: findings
version: 1
stage_order: 7
depends_on:
  - profiling
optional_deps:
  - exploration
  - dimensionality
  - unsupervised
  - supervised
---

# Findings Generator

## Purpose
Read ALL stage outputs and synthesize plain English findings with confidence
levels and provenance. Produces a summary card and suggested next steps.

## Outputs
| Key | Type | Description |
|-----|------|-------------|
| `findings_list` | list[dict] | Each: text, confidence (HIGH/MEDIUM/LOW), source_stage, rule |
| `summary_card` | list[str] | Top 3 most important findings |
| `suggested_next_steps` | list[str] | Actionable recommendations |

## Confidence Scoring
- HIGH: silhouette > 0.6, |r| > 0.95, F1 > 0.8
- MEDIUM: silhouette 0.3-0.6, |r| 0.8-0.95, F1 0.5-0.8
- LOW: weak or borderline signals

## Failure Behavior
- If a stage is missing or failed: skip findings from that stage, note
  which stages were unavailable.
- Never crash the pipeline.
