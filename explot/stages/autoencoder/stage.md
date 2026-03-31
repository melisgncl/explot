---
name: autoencoder
version: 2
stage_order: 4
depends_on:
  - dimensionality
---

# DVAE

## Purpose
Train a denoising variational autoencoder on the Stage 3 transformed matrix.
This stage provides a nonlinear latent representation, row-level reconstruction
errors, and training diagnostics for downstream unsupervised and supervised probes.

## Inputs
- Stage 3: `transformed_df`

## Outputs
| Key | Type | Description |
|-----|------|-------------|
| `latent_df` | DataFrame | Latent representation per row using the encoder mean |
| `latent_2d` | ndarray | First two latent dimensions for visualization |
| `bottleneck_dim` | int | Latent dimensionality |
| `reconstruction_mse` | float | Global reconstruction mean squared error |
| `reconstruction_errors` | list[float] | Row-level reconstruction errors |
| `training_loss_curve` | list[float] | Mean training loss per epoch |
| `fit_rows` | int | Number of rows used for training |
| `sampled` | bool | Whether training used a sampled subset |
| `model_type` | str | Compression model identifier |

## Figures + Interpretations
| Key | Figure Description | Interpretation Must Address |
|-----|--------------------|-----------------------------|
| `latent_projection` | 2D latent scatter | Whether the latent space shows visible structure |
| `training_loss` | Training loss by epoch | Whether optimization converged smoothly |
| `reconstruction_error` | Distribution of row-level reconstruction error | Whether a high-error tail exists |

## Notes
- Uses Gaussian noise injection during training.
- Uses KL regularization to keep the latent space smooth.
- Large datasets may be sampled deterministically for practical runtime.
