"""Offline reconstruction FID using torch-fidelity."""

import numpy as np
from torch_fidelity import calculate_metrics

from .utils import ImgArrDataset


def calculate_rfid(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    bs: int = 64,
    device: str = "cuda",
) -> float:
    metrics = calculate_metrics(
        input1=ImgArrDataset(reconstruction),
        input2=ImgArrDataset(reference),
        batch_size=bs,
        fid=True,
        cuda=device == "cuda",
    )
    return float(metrics["frechet_inception_distance"])
