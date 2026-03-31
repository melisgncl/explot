# Explot Risks And Scope Guardrails

This file keeps the early architectural risks visible while implementation
starts. It is intentionally short and practical.

## Current Guidance

1. Keep v1 focused on the core pipeline:
   profiling, exploration, dimensionality, unsupervised probes, and findings.
   Treat DVAE and the full supervised model suite as phase-2 work unless the
   core path is already stable.

2. Be careful with statistical language around missingness.
   Prefer "random-looking" vs "structured" missingness in user-facing output.
   Only label patterns as MAR or MNAR when the implementation is explicitly
   heuristic and the report says so.

3. Avoid a model zoo too early.
   Start with lightweight probes and optional extras. Heavy dependencies and
   long runtimes can sink the usefulness of a first release.

4. Never silently drop context-rich columns.
   Columns excluded from modeling should still remain visible in provenance,
   reports, and stage logs.

5. Keep interpretations structured, not only free text.
   Findings should be traceable to metrics, thresholds, and source stages.

6. Watch report size.
   A single self-contained HTML report is a strong goal, but large interactive
   figures need sampling, truncation, or summarization to stay usable.

7. Prefer robust tests over brittle prose matching.
   Test structured outputs first, then use keyword checks as a lightweight
   smoke test for interpretation text.

## Immediate Build Order

1. Core project skeleton
2. Simulator base and first datasets
3. Profiling stage
4. Exploration stage
5. Dimensionality stage
6. Unsupervised stage
7. Findings and report integration

