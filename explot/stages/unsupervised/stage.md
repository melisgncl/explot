---
name: unsupervised
version: 1
stage_order: 4
depends_on:
  - dimensionality
optional_deps:
  - exploration
---

# Unsupervised Probes

## Purpose
Discover natural groupings and anomalies in the transformed data using
clustering and outlier detection. All methods operate on the Stage 3
transformed feature matrix and are visualized in the PCA 2D projection.

## Inputs
- Stage 3: `transformed_df`, `pca_2d`
- Stage 2 (optional): `hopkins_statistic` (for context in interpretations)

## Outputs
| Key | Type | Description |
|-----|------|-------------|
| `kmeans_results` | dict | `optimal_k`, `best_silhouette`, `silhouette_scores` (k->score), `cluster_labels`, `cluster_sizes` |
| `dbscan_results` | dict | `n_clusters`, `noise_fraction`, `eps_used`, `cluster_labels` |
| `isolation_forest_scores` | ndarray | Anomaly score per row |
| `anomaly_rows` | list[int] | Row indices flagged as anomalous (top 1%) |
| `cluster_outlier_overlap` | dict | Overlap between IF anomalies and multi-dim outliers from Stage 2 |

## Figures + Interpretations
| Key | Figure Description | Interpretation Must Address |
|-----|--------------------|-----------------------------|
| `kmeans_silhouette` | Silhouette score vs k line plot | Optimal k, quality assessment, cluster balance |
| `cluster_scatter` | PCA 2D scatter colored by KMeans labels | Visual cluster separation |
| `isolation_forest` | (metric summary) | Anomaly count, severity distribution |

## Heuristics
- `kmeans_sweep`: k from 2 to min(10, n_rows/10). In fast mode: k from 2 to 5.
- `dbscan_eps`: auto-tune using k-distance plot (k=5 nearest neighbors,
  eps = elbow of sorted distances). min_samples = max(5, n_rows // 100).
- `isolation_forest_threshold`: flag top 1% by anomaly score.

## Failure Behavior
- If transformed_df has fewer than 10 rows: skip clustering, return empty results.
- If all silhouette scores are below 0.1: report "no meaningful clusters found."
- DBSCAN may find 0 clusters — report this honestly.
- Never crash the pipeline.

## Budget Overrides (fast mode)
- KMeans sweep: k = 2..5 instead of 2..10
