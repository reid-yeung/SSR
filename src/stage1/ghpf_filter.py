"""Gaussian High-Pass Filter (GHPF) for PyTorch tensors.

Formula:
    H(u,v) = 1 - exp(-(D(u,v)^2) / (2 * D0^2))

Post-processing keeps only the high-frequency magnitude:
    1. abs(highpass)

The output is the raw high-pass magnitude map (non-negative, unnormalized),
which is then normalized by the encoder mean/std in the training pipeline.
"""

from __future__ import annotations

import torch


def build_ghpf_mask(height: int, width: int, d0: float, device: torch.device) -> torch.Tensor:
    """Build a Gaussian high-pass mask in the frequency domain.

    Args:
        height: image height
        width: image width
        d0: cutoff frequency in pixels
    Returns:
        (H, W) float32 mask
    """
    u = torch.arange(height, device=device, dtype=torch.float32) - (height // 2)
    v = torch.arange(width, device=device, dtype=torch.float32) - (width // 2)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    dist2 = uu ** 2 + vv ** 2
    hpf = 1.0 - torch.exp(-dist2 / (2.0 * d0 * d0))
    return hpf


def apply_ghpf(images: torch.Tensor, cutoff_ratio: float) -> torch.Tensor:
    """Apply Gaussian high-pass filter to a batch of images.

    Applies frequency-domain Gaussian high-pass filtering and returns the
    absolute magnitude response without percentile normalization.

    Args:
        images: (B, C, H, W) float tensor in [0, 1], NOT encoder-normalized.
                Typically ``extra["teacher_image"]`` from the training loop.
        cutoff_ratio: normalized cutoff, d0 = cutoff_ratio * min(H, W).
    Returns:
        highpass_abs: (B, C, H, W) float tensor, non-negative and unnormalized.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got {tuple(images.shape)}")
    if not (0.0 < cutoff_ratio <= 1.0):
        raise ValueError(f"cutoff_ratio must be in (0, 1], got {cutoff_ratio}")

    B, C, H, W = images.shape
    d0 = float(cutoff_ratio) * float(min(H, W))

    mask = build_ghpf_mask(H, W, d0, device=images.device)  # (H, W)
    mask = mask.view(1, 1, H, W)  # broadcast over B, C

    # Per-channel FFT
    freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask
    highpass = torch.fft.ifft2(
        torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)
    ).real  # (B, C, H, W), signed

    # Keep raw high-frequency energy magnitude (no percentile normalization).
    highpass_abs = highpass.abs()
    return highpass_abs
