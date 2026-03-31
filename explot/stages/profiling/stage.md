---
name: profiling
version: 1
stage_order: 1
depends_on: []
---

# Profiling

## Purpose
Describe the raw dataset without mutating it. This stage is intentionally
passive: it computes per-column summaries, flags suspicious columns, estimates
basic dataset quality, and makes a heuristic normalization guess that later
stages may use.

## Outputs
- `n_rows`
- `n_columns`
- `column_names`
- `dtypes`
- `memory_usage_bytes`
- `column_profiles`
- `numeric_column_names`
- `categorical_column_names`
- `suspicious_columns`
- `normalization_guess`
- `quality_score`
- `quality_breakdown`

## Heuristics
- `near_constant`: flag columns with one value occupying at least 95% of
  non-null rows.
- `mostly_null`: flag columns with more than 80% null values.
- `id_like`: flag columns with cardinality equal to row count.
- `raw_counts_guess`: integer-valued numeric data, non-negative values, and
  strong right skew.
- `log_normalized_guess`: positive continuous numeric data with lower skew than
  raw counts and wide dynamic range.

## Notes
This stage stays conservative. User-facing wording should prefer plain-English
 descriptions such as "raw counts" or "log-normalized" over stronger claims
 that are not directly identifiable from a single table.
