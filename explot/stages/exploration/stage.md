---
name: exploration
version: 1
stage_order: 2
depends_on:
  - profiling
---

# Exploration

## Purpose
Analyze relationships and structure in the raw dataset without mutating it.
This stage focuses on numeric correlations, missingness structure, variable
features, and a conservative cluster-tendency estimate.

## Outputs
- `correlation_matrix`
- `redundant_pairs`
- `top_variable_features`
- `missingness_type`
- `missingness_correlations`
- `hopkins_statistic`
- `grouping_candidates`

## Notes
This first implementation slice stays conservative. Missingness is reported as
`random-looking`, `structured`, or `minimal`, rather than stronger claims that
cannot be cleanly identified from one observed table.

