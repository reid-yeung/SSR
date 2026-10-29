"""Image-domain GHPF reconstruction loss for Stage-1 training.

This module computes a loss between high-frequency maps of:
  1) target image (original input before encoder normalization)
  2) reconstructed image (decoder output)

Both maps are obtained by applying Gaussian High-Pass Filter (GHPF).
Supported losses:
  - mse
  - cosine

Optional map normalization is provided to reduce scale sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from stage1.ghpf_filter import apply_ghpf


@dataclass
class GHPFReconLossConfig:
    enabled: bool = False
    start_epoch: int = 0
    end_epoch: int | None = None
    weight: float = 1.0
    warmup_steps: int = 0
    warmup_schedule: str = "none"  # none | linear | cosine
    loss_type: str = "mse"  # mse | cosine
    cutoff_ratio: float = 0.05
    normalize_map: str = "none"  # none | l2 | channel_std
    eps: float = 1e-6


class GHPFReconLoss(torch.nn.Module):
    """Compute image-domain loss on GHPF(filtered target/reconstruction)."""

    def __init__(
        self,
        *,
        loss_type: str = "mse",
        cutoff_ratio: float = 0.05,
        normalize_map: str = "none",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        loss_type_l = str(loss_type).lower()
        if loss_type_l not in {"mse", "cosine"}:
            raise ValueError(
                f"Unsupported ghpf_recon.loss_type '{loss_type}'. "
                "Choose from ['mse', 'cosine']."
            )
        if not (0.0 < float(cutoff_ratio) <= 1.0):
            raise ValueError(
                f"ghpf_recon.cutoff_ratio must be in (0, 1], got {cutoff_ratio}"
            )

        normalize_map_l = str(normalize_map).lower()
        if normalize_map_l not in {"none", "l2", "channel_std"}:
            raise ValueError(
                f"Unsupported ghpf_recon.normalize_map '{normalize_map}'. "
                "Choose from ['none', 'l2', 'channel_std']."
            )
        if float(eps) <= 0.0:
            raise ValueError(f"ghpf_recon.eps must be positive, got {eps}")

        self.loss_type = loss_type_l
        self.cutoff_ratio = float(cutoff_ratio)
        self.normalize_map = normalize_map_l
        self.eps = float(eps)

    def _normalize_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_map == "none":
            return x

        if self.normalize_map == "l2":
            flat = x.flatten(start_dim=1)
            denom = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(self.eps)
            flat = flat / denom
            return flat.view_as(x)

        # channel_std: standardize each channel over spatial dims per sample.
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std

    def forward(self, recon_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        """Compute loss between GHPF(recon_image) and GHPF(target_image).

        Args:
            recon_image: (B, C, H, W), typically decoder output in [0, 1].
            target_image: (B, C, H, W), original training image in [0, 1].
        Returns:
            Scalar loss tensor.
        """
        if recon_image.ndim != 4 or target_image.ndim != 4:
            raise ValueError(
                "recon_image and target_image must be 4D tensors (B, C, H, W); "
                f"got {tuple(recon_image.shape)} and {tuple(target_image.shape)}"
            )
        if recon_image.shape != target_image.shape:
            raise ValueError(
                "Shape mismatch between recon_image and target_image: "
                f"{tuple(recon_image.shape)} vs {tuple(target_image.shape)}"
            )

        recon_hf = apply_ghpf(recon_image, self.cutoff_ratio)
        target_hf = apply_ghpf(target_image, self.cutoff_ratio)

        recon_hf = self._normalize_map(recon_hf)
        target_hf = self._normalize_map(target_hf)

        if self.loss_type == "mse":
            return F.mse_loss(recon_hf, target_hf, reduction="mean")

        recon_flat = recon_hf.flatten(start_dim=1)
        target_flat = target_hf.flatten(start_dim=1)
        cos = F.cosine_similarity(recon_flat, target_flat, dim=1, eps=self.eps)
        return (1.0 - cos).mean()
