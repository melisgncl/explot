# Explot

Explot is a data-analysis pipeline for tabular datasets that aims to produce a
self-contained report explaining what is in the data, what structure is present,
and what downstream analysis paths look promising.

## Current Status

- Core package skeleton exists
- Stage 1 Profiling is partially implemented and runnable
- Simulators exist for `tabular`, `scrna`, and `proteomics`
- The HTML report is still an early scaffold, not the final tabbed report

## Build Philosophy

1. Simulator-first development
2. Stage specs before large implementations
3. Graceful failure over pipeline crashes
4. Reports should explain what metrics mean
5. Tests should verify planted truth, not just runtime success

## Stage Intent

1. Profiling: passive per-column description only
2. Exploration: passive between-column and between-row patterns
3. Dimensionality: first transformation stage
4. Unsupervised: clustering and anomaly probes
5. DVAE: optional nonlinear representation learning
6. Supervised: target-detection probe models
7. Findings: final synthesis with provenance

## Near-Term Execution Order

1. Finish Stage 1 polish
2. Build Stage 2 Exploration
3. Build Stage 3 Dimensionality
4. Build Stage 4 Unsupervised
5. Upgrade the report into a real multi-tab HTML output
6. Add DVAE and supervised probes after the core path is stable

## Ground Rules For Future Sessions

- Read `RISKS.md` before expanding scope.
- Do not treat the current HTML output as the final report architecture.
- If a heuristic is uncertain, label it as heuristic in the output.
- Avoid heavy model dependencies until the first four stages are stable.
- Keep suspicious-column detection conservative; false positives erode trust.

## Repo Workflow

- Stage contracts live in `explot/stages/<stage_name>/stage.md`
- Implementations live in `explot/stages/<stage_name>/stage.py`
- Simulators live in `simulator/`
- Stage tests live in `tests/test_stages/`
- Simulator tests live in `tests/test_simulators/`

## Claude And Codex Notes

This repo includes `.claude/` guidance because it is useful project memory,
but different assistants do not execute those files the same way.

- Claude Code may use `.claude/settings.local.json` hooks and skill docs directly.
- Codex can still read these files as repo guidance, but does not depend on
  Claude-specific automation behavior.
- Treat `.claude/` as durable project instructions first, automation second.

