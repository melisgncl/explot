from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from explot.models.dvae import DVAETrainingConfig, DenoisingVAE, dvae_loss
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from explot.stages.base import BaseStage, StageMeta, StageResult


class AutoencoderStage(BaseStage):
    meta = StageMeta(name="autoencoder", depends_on=("dimensionality",))

    def run(self, state, config, hooks) -> StageResult:
        if not _HAS_TORCH:
            return self._empty_result(
                "PyTorch is not installed. Install with: pip install explot[autoencoder]"
            )
        dim = state.results["dimensionality"]
        transformed_df = dim.outputs.get("transformed_df")
        if transformed_df is None or transformed_df.empty or transformed_df.shape[1] < 2:
            return self._empty_result("Not enough numeric features for DVAE training.")

        transformed_df = transformed_df.astype(float)
        is_fast = getattr(config, "budget", None) and getattr(config.budget, "mode", "") == "fast"
        min_rows = 50 if is_fast else 100
        if len(transformed_df) < min_rows:
            return self._empty_result(
                f"DVAE skipped because only {len(transformed_df)} rows are available; at least {min_rows} rows are needed."
            )

        max_fit_rows = 3000 if is_fast else 8000
        fit_df = transformed_df
        sampled = False
        if len(transformed_df) > max_fit_rows:
            fit_df = transformed_df.sample(n=max_fit_rows, random_state=42)
            sampled = True
            if hasattr(hooks, "log"):
                hooks.log(
                    self.meta.name,
                    f"Sampling {len(fit_df)} of {len(transformed_df)} rows for DVAE training.",
                )

        input_dim = fit_df.shape[1]
        latent_dim = min(max(2, input_dim // 2), 16)
        hidden_dim = min(max(latent_dim * 2, 16), 128)
        train_cfg = DVAETrainingConfig(
            epochs=16 if is_fast else 28,
            batch_size=min(256, max(32, 2 ** int(math.log2(max(32, min(len(fit_df), 256)))))),
            learning_rate=1e-3,
            noise_std=0.05 if is_fast else 0.08,
            beta=0.05 if is_fast else 0.08,
        )

        hooks.progress(self.meta.name, 20, "Training denoising variational autoencoder.")

        try:
            model, loss_curve = self._train_dvae(fit_df.to_numpy(dtype=np.float32), latent_dim, hidden_dim, train_cfg)
            latent_df, latent_2d, reconstruction_errors, reconstruction_mse = self._infer_outputs(
                model,
                transformed_df,
            )
        except Exception as exc:
            return self._empty_result(f"DVAE training failed: {exc}")

        outputs = {
            "latent_df": latent_df,
            "latent_2d": latent_2d,
            "bottleneck_dim": latent_dim,
            "reconstruction_mse": round(reconstruction_mse, 6),
            "reconstruction_errors": reconstruction_errors,
            "training_loss_curve": loss_curve,
            "fit_rows": int(len(fit_df)),
            "sampled": sampled,
            "model_type": "dvae",
        }
        figures = {
            "latent_projection": self._projection_svg(latent_2d),
            "training_loss": self._line_svg(loss_curve, "DVAE training loss", "epoch", "loss"),
            "reconstruction_error": self._histogram_svg(reconstruction_errors, "Reconstruction error", "row MSE"),
        }
        interpretations = {
            "summary": self._summary_text(latent_dim, reconstruction_mse, len(fit_df), len(transformed_df), sampled),
            "training_loss": self._loss_text(loss_curve),
            "reconstruction_error": self._reconstruction_text(reconstruction_errors),
        }
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs=outputs,
            figures=figures,
            interpretations=interpretations,
        )

    def _train_dvae(
        self,
        matrix: np.ndarray,
        latent_dim: int,
        hidden_dim: int,
        train_cfg: DVAETrainingConfig,
    ) -> tuple[DenoisingVAE, list[float]]:
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = DenoisingVAE(matrix.shape[1], latent_dim, hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
        dataset = TensorDataset(torch.from_numpy(matrix))
        loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)
        losses: list[float] = []

        model.train()
        for _epoch in range(train_cfg.epochs):
            epoch_losses = []
            for (batch,) in loader:
                clean = batch.to(device)
                noisy = clean + torch.randn_like(clean) * train_cfg.noise_std
                optimizer.zero_grad(set_to_none=True)
                reconstructed, mu, logvar = model(noisy)
                loss, _recon, _kl = dvae_loss(reconstructed, clean, mu, logvar, train_cfg.beta)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            losses.append(round(float(np.mean(epoch_losses)), 6))

        return model, losses

    def _infer_outputs(
        self,
        model: DenoisingVAE,
        transformed_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray, list[float], float]:
        model.eval()
        with torch.no_grad():
            X = torch.from_numpy(transformed_df.to_numpy(dtype=np.float32))
            mu, logvar = model.encode(X)
            reconstructed = model.decode(mu)
            recon_errors = torch.mean((reconstructed - X) ** 2, dim=1).cpu().numpy()
            latent = mu.cpu().numpy()

        latent_df = pd.DataFrame(
            latent,
            columns=[f"latent_{idx + 1}" for idx in range(latent.shape[1])],
            index=transformed_df.index,
        )
        if latent.shape[1] >= 2:
            latent_2d = latent[:, :2]
        else:
            latent_2d = np.column_stack([latent[:, 0], np.zeros(len(latent))])
        return latent_df, latent_2d, [round(float(v), 6) for v in recon_errors], float(np.mean(recon_errors))

    def _summary_text(
        self,
        bottleneck_dim: int,
        reconstruction_mse: float,
        fit_rows: int,
        total_rows: int,
        sampled: bool,
    ) -> str:
        if reconstruction_mse < 0.1:
            quality = "strong"
        elif reconstruction_mse < 0.5:
            quality = "moderate"
        else:
            quality = "weak"
        sample_note = (
            f" Training used a deterministic sample of {fit_rows} rows from {total_rows} total rows."
            if sampled else
            f" Training used all {total_rows} rows."
        )
        return (
            f"The DVAE compressed the transformed matrix into {bottleneck_dim} latent dimensions using denoising noise injection and a KL regularizer. "
            f"Reconstruction MSE is {reconstruction_mse:.4f}, which suggests {quality} nonlinear compression quality."
            f"{sample_note}"
        )

    def _loss_text(self, loss_curve: list[float]) -> str:
        if not loss_curve:
            return "No training-loss curve is available."
        start = loss_curve[0]
        end = loss_curve[-1]
        if end < start * 0.7:
            trend = "dropped substantially"
        elif end < start:
            trend = "decreased modestly"
        else:
            trend = "stayed flat"
        return (
            f"DVAE training loss started at {start:.4f} and ended at {end:.4f}. "
            f"The loss {trend}, which suggests the model learned a usable latent representation without obvious instability."
        )

    def _reconstruction_text(self, reconstruction_errors: list[float]) -> str:
        if not reconstruction_errors:
            return "No reconstruction-error distribution is available."
        arr = np.asarray(reconstruction_errors, dtype=float)
        p95 = float(np.percentile(arr, 95))
        median = float(np.median(arr))
        return (
            f"Median row-level reconstruction error is {median:.4f}, and the 95th percentile is {p95:.4f}. "
            "Rows in the high-error tail are harder for the DVAE to reconstruct and may represent unusual structure or anomalies."
        )

    def _projection_svg(self, latent_2d: np.ndarray) -> str:
        if latent_2d.size == 0:
            return ""
        points = latent_2d
        if len(points) > 400:
            rng = np.random.default_rng(42)
            points = points[np.sort(rng.choice(len(points), size=400, replace=False))]

        return self._scatter_svg(points, "DVAE latent space", "latent 1", "latent 2")

    def _scatter_svg(self, points: np.ndarray, title: str, xlabel: str, ylabel: str) -> str:
        x = points[:, 0]
        y = points[:, 1]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0

        width = 420
        height = 260
        margin = 34
        chart_width = width - margin * 2
        chart_height = height - margin * 2
        circles = []
        for px, py in zip(x, y):
            sx = margin + ((float(px) - x_min) / (x_max - x_min)) * chart_width
            sy = height - margin - ((float(py) - y_min) / (y_max - y_min)) * chart_height
            circles.append(f"<circle cx='{sx:.1f}' cy='{sy:.1f}' r='2.2' fill='rgba(239,125,87,0.55)' />")

        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:520px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            f"<rect width='{width}' height='{height}' fill='#f8fbfd' rx='12' />"
            f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            + "".join(circles)
            + f"<text x='{width / 2:.1f}' y='20' font-size='14' text-anchor='middle' fill='#193042'>{title}</text>"
            + f"<text x='{width / 2:.1f}' y='{height - 10}' font-size='11' text-anchor='middle' fill='#5f7584'>{xlabel}</text>"
            + f"<text x='16' y='{height / 2:.1f}' font-size='11' text-anchor='middle' fill='#5f7584' transform='rotate(-90 16 {height / 2:.1f})'>{ylabel}</text>"
            + "</svg>"
        )

    def _line_svg(self, values: list[float], title: str, xlabel: str, ylabel: str) -> str:
        if not values:
            return ""
        x = np.arange(1, len(values) + 1, dtype=float)
        y = np.asarray(values, dtype=float)
        return self._path_svg(x, y, title, xlabel, ylabel)

    def _histogram_svg(self, values: list[float], title: str, xlabel: str) -> str:
        if not values:
            return ""
        arr = np.asarray(values, dtype=float)
        bins = min(24, max(8, int(np.sqrt(len(arr)))))
        counts, edges = np.histogram(arr, bins=bins)
        width = 420
        height = 260
        margin = 34
        chart_width = width - margin * 2
        chart_height = height - margin * 2
        y_max = max(int(np.max(counts)), 1)
        bars = []
        for idx, count in enumerate(counts):
            bar_w = chart_width / len(counts)
            x0 = margin + idx * bar_w
            bar_h = (count / y_max) * chart_height
            y0 = height - margin - bar_h
            bars.append(
                f"<rect x='{x0 + 1:.1f}' y='{y0:.1f}' width='{max(bar_w - 2, 1):.1f}' height='{bar_h:.1f}' fill='rgba(15,106,139,0.55)' />"
            )
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:520px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            f"<rect width='{width}' height='{height}' fill='#f8fbfd' rx='12' />"
            f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            + "".join(bars)
            + f"<text x='{width / 2:.1f}' y='20' font-size='14' text-anchor='middle' fill='#193042'>{title}</text>"
            + f"<text x='{width / 2:.1f}' y='{height - 10}' font-size='11' text-anchor='middle' fill='#5f7584'>{xlabel}</text>"
            + f"<text x='16' y='{height / 2:.1f}' font-size='11' text-anchor='middle' fill='#5f7584' transform='rotate(-90 16 {height / 2:.1f})'>count</text>"
            + f"<text x='{margin:.1f}' y='{height - 18}' font-size='10' text-anchor='start' fill='#5f7584'>{edges[0]:.3f}</text>"
            + f"<text x='{width - margin:.1f}' y='{height - 18}' font-size='10' text-anchor='end' fill='#5f7584'>{edges[-1]:.3f}</text>"
            + "</svg>"
        )

    def _path_svg(self, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> str:
        width = 420
        height = 260
        margin = 34
        chart_width = width - margin * 2
        chart_height = height - margin * 2
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        coords = []
        for px, py in zip(x, y):
            sx = margin + ((float(px) - x_min) / max(x_max - x_min, 1e-9)) * chart_width
            sy = height - margin - ((float(py) - y_min) / (y_max - y_min)) * chart_height
            coords.append(f"{sx:.1f},{sy:.1f}")
        path = " ".join(coords)
        return (
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
            "style='max-width:520px;background:#f8fbfd;border:1px solid #d8e3ea;border-radius:12px'>"
            f"<rect width='{width}' height='{height}' fill='#f8fbfd' rx='12' />"
            f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#b6c5cf' />"
            + f"<polyline fill='none' stroke='#0f6a8b' stroke-width='2.5' points='{path}' />"
            + f"<text x='{width / 2:.1f}' y='20' font-size='14' text-anchor='middle' fill='#193042'>{title}</text>"
            + f"<text x='{width / 2:.1f}' y='{height - 10}' font-size='11' text-anchor='middle' fill='#5f7584'>{xlabel}</text>"
            + f"<text x='16' y='{height / 2:.1f}' font-size='11' text-anchor='middle' fill='#5f7584' transform='rotate(-90 16 {height / 2:.1f})'>{ylabel}</text>"
            + "</svg>"
        )

    def _empty_result(self, reason: str) -> StageResult:
        return StageResult(
            stage_name=self.meta.name,
            meta=self.meta,
            outputs={
                "latent_df": pd.DataFrame(),
                "latent_2d": np.array([]),
                "bottleneck_dim": 0,
                "reconstruction_mse": None,
                "reconstruction_errors": [],
                "training_loss_curve": [],
                "fit_rows": 0,
                "sampled": False,
                "model_type": "dvae",
            },
            figures={
                "latent_projection": "",
                "training_loss": "",
                "reconstruction_error": "",
            },
            interpretations={
                "summary": reason,
                "training_loss": reason,
                "reconstruction_error": reason,
            },
        )
