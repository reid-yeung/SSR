"""Array adapters used by the offline reconstruction metrics."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ImgArrDataset(Dataset):
    """Provide NHWC uint8 images as CHW uint8 tensors for torch-fidelity."""

    def __init__(self, images: np.ndarray) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.images[index]).permute(2, 0, 1)


def to_torch_tensor(images: np.ndarray) -> torch.Tensor:
    """Convert NHWC uint8 images to NCHW float tensors in [0, 1]."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
