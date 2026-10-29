from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FFTFilterConfig:
    enabled: bool = False
    cutoff_ratio: float = 0.0


def _validate_image_tensor(images: torch.Tensor) -> None:
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (B, C, H, W), got {tuple(images.shape)}")
    if images.shape[-2] <= 0 or images.shape[-1] <= 0:
        raise ValueError(f"Invalid spatial shape {tuple(images.shape[-2:])}")


def resolve_cutoff_radius(shape_hw: tuple[int, int], cutoff_ratio: float) -> int:
    height, width = shape_hw
    max_radius = min(height, width) // 2
    if max_radius <= 0:
        raise ValueError(f"Cannot build FFT mask for shape {shape_hw}")
    ratio = float(cutoff_ratio)
    if ratio <= 0.0:
        raise ValueError(f"cutoff_ratio must be positive, got {cutoff_ratio}")
    radius = int(round(ratio * max_radius))
    return max(1, min(radius, max_radius))


def make_circular_high_pass_mask(
    shape_hw: tuple[int, int],
    cutoff_ratio: float,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    height, width = shape_hw
    radius = resolve_cutoff_radius(shape_hw, cutoff_ratio)
    y = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    x = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    cy = height // 2
    cx = width // 2
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    mask = (dist2 > radius**2).to(dtype=torch.float32)
    return mask


def fft_magnitude_spectrum(images: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    _validate_image_tensor(images)
    gray = images.mean(dim=1, keepdim=True)
    freq = torch.fft.fftshift(torch.fft.fft2(gray, dim=(-2, -1)), dim=(-2, -1))
    mag = torch.log1p(freq.abs() + eps)
    return mag


def apply_abs_high_pass(images: torch.Tensor, cutoff_ratio: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_image_tensor(images)
    radius = resolve_cutoff_radius(images.shape[-2:], cutoff_ratio)
    mask = make_circular_high_pass_mask(images.shape[-2:], cutoff_ratio, device=images.device)
    mask = mask.view(1, 1, images.shape[-2], images.shape[-1])

    freq = torch.fft.fftshift(torch.fft.fft2(images, dim=(-2, -1)), dim=(-2, -1))
    filtered_freq = freq * mask
    recon = torch.fft.ifft2(torch.fft.ifftshift(filtered_freq, dim=(-2, -1)), dim=(-2, -1)).real
    abs_high_pass = recon.abs()

    debug = {
        "mask": mask,
        "radius": torch.tensor(radius, device=images.device),
        "input_spectrum": torch.log1p(freq.abs()),
        "filtered_spectrum": torch.log1p(filtered_freq.abs()),
        "reconstruction": recon,
        "abs_high_pass": abs_high_pass,
    }
    return abs_high_pass, debug


def normalize_for_visualization(images: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if images.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {tuple(images.shape)}")
    flat = images.flatten(start_dim=-2)
    min_v = flat.min(dim=-1, keepdim=True).values[..., None]
    max_v = flat.max(dim=-1, keepdim=True).values[..., None]
    denom = (max_v - min_v).clamp_min(eps)
    return (images - min_v) / denom
