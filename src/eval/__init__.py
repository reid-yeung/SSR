"""Offline metrics for saved reconstruction image arrays."""

import numpy as np
import torch

from .fid import calculate_rfid
from .ref_iqa import calculate_psnr, calculate_ssim


def compute_reconstruction_metrics(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Score paired, identically ordered NHWC uint8 images offline."""
    if reference.shape != reconstruction.shape:
        raise ValueError(f"Image arrays differ in shape: {reference.shape} != {reconstruction.shape}")
    if reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError("Expected arrays shaped [N, H, W, 3].")
    if reference.dtype != np.uint8 or reconstruction.dtype != np.uint8:
        raise ValueError("Expected uint8 arrays in [0, 255].")
    if len(reference) < 2:
        raise ValueError("At least two images are required to compute rFID.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    device_name = device.type
    return {
        "psnr": calculate_psnr(reference, reconstruction, batch_size, device_name),
        "ssim": calculate_ssim(reference, reconstruction, batch_size, device_name),
        "rfid": calculate_rfid(reference, reconstruction, batch_size, device_name),
    }
