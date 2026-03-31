---
name: dimensionality
version: 1
stage_order: 3
depends_on:
  - profiling
  - exploration
---

# Dimensionality

## Purpose
First stage that transforms data. Decisions are informed by Stage 1
(normalization guess, suspicious columns) and Stage 2 (redundant pairs).
Produces cleaned, scaled feature matrices and PCA projections used by all
downstream stages.

## Inputs
- `state.raw_df`
- Stage 1: `normalization_guess`, `suspicious_columns`, `numeric_column_names`,
  `categorical_column_names`
- Stage 2: `redundant_pairs`

## Outputs
| Key | Type | Description |
|-----|------|-------------|
| `cleaned_df` | DataFrame | Numeric matrix after dropping suspicious/redundant columns and imputing NaNs |
| `transformed_df` | DataFrame | cleaned_df after optional log1p + StandardScaler |
| `pca_components` | ndarray | Full PCA component matrix |
| `pca_2d` | ndarray | First 2 principal components (n_rows x 2) |
| `pca_explained_variance` | list[float] | Per-component explained variance ratio |
| `intrinsic_dim` | int | Estimated intrinsic dimensionality (participation ratio) |
| `n_components_50` | int | Components needed for 50% variance |
| `n_components_80` | int | Components needed for 80% variance |
| `n_components_95` | int | Components needed for 95% variance |
| `transform_log` | list[str] | Ordered list of transformations applied with reasons |
| `dropped_columns` | list[dict] | Columns dropped with name and reason |

## Figures + Interpretations
| Key | Figure Description | Interpretation Must Address |
|-----|--------------------|-----------------------------|
| `pca_variance` | Explained variance curve (cumulative + per-component) | How many components matter, where the elbow is, intrinsic dim estimate |
| `transform_log` | (no figure, text only) | What was done and why, final matrix shape |

## Heuristics
- `apply_log1p`: if normalization_guess is "raw counts", apply log1p to all
  numeric values before scaling. Log reason in transform_log.
- `drop_suspicious`: remove columns flagged as `id_like` or `near_constant`
  by Stage 1. Keep `mostly_null` columns (impute instead).
- `drop_redundant`: for each redundant pair (|r| > 0.95 from Stage 2), drop
  the second column in the pair.
- `impute_median`: fill remaining NaN values with column median.
- `participation_ratio`: intrinsic dimensionality = (sum of eigenvalues)^2 /
  sum of eigenvalues^2. Round to nearest integer, clamp to [1, n_features].

## Failure Behavior
- If no numeric columns remain after cleaning: return success=True with empty
  transformed_df and intrinsic_dim=0. Downstream stages handle empty input.
- If PCA fails (e.g. all-constant columns after scaling): catch exception,
  set pca_2d to zeros, log warning in transform_log.
- Never crash the pipeline.

## Budget Overrides (fast mode)
- No difference for this stage (PCA is already fast).
