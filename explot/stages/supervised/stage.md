---
name: supervised
version: 1
stage_order: 6
depends_on:
  - dimensionality
optional_deps:
  - profiling
  - unsupervised
---

# Model Selection (Supervised Probes)

## Purpose
Auto-detect candidate target columns, run multiple ML models with cross-
validation, and recommend which model fits the data best with an explanation
grounded in earlier stage outputs. This is the headline feature of Explot.

## Inputs
- Stage 3: `transformed_df` (features), `intrinsic_dim`
- Stage 1 (optional): `column_profiles`, `categorical_column_names`
- Stage 4 (optional): `kmeans_results` (for context in recommendation)

## Outputs
| Key | Type | Description |
|-----|------|-------------|
| `candidate_targets` | list[dict] | Each with name, task_type (classification/regression), n_classes |
| `model_results` | dict[str, list[dict]] | target_name -> list of {model, metric, mean, std} |
| `feature_importances` | dict[str, list[dict]] | target_name -> top features from RF |
| `best_models` | dict[str, dict] | target_name -> {model, metric, mean, std, recommendation} |

## Figures + Interpretations
| Key | Figure | Interpretation Must Address |
|-----|--------|-----------------------------|
| `model_comparison` | (table) | Best model, baseline gap, which models struggled |
| `feature_importances` | (table) | Top features and what they mean |
| `model_recommendation` | (none) | Plain English "use this because..." paragraph |

## Heuristics
- `target_detection`: categorical columns with 2-20 unique values, or columns
  named like target/label/class/outcome/group/type/diagnosis/status.
- `task_type`: classification if target has <= 20 unique values, regression
  if numeric with > 20 unique values.
- `cv_folds`: 5-fold (3-fold in fast mode).
- Models: LogisticRegression/Ridge, RandomForest, SVM(RBF). Optional:
  XGBoost, LightGBM (graceful skip if not installed).

## Failure Behavior
- No candidate targets found: return success=True with empty results and
  interpretation explaining why.
- A model fails to fit: skip it, log warning, continue with remaining models.
- Never crash the pipeline.

## Budget Overrides (fast mode)
- 3-fold CV instead of 5-fold.
- Skip SVM if n_rows > 5000.
