from __future__ import annotations

import os

from torch import nn
from transformers import AutoConfig

from stage1.encoders.dinov3 import Dinov3withNorm


def build_teacher_encoder(
    teacher_path: str,
    *,
    normalize: bool = True,
) -> nn.Module:
    """Build the frozen DINOv3 teacher used by the selected SSR experiment."""
    hf_offline = os.environ.get("RAE_HF_OFFLINE", "0") == "1"
    try:
        config = AutoConfig.from_pretrained(teacher_path, local_files_only=True)
    except (OSError, ValueError, AttributeError):
        if hf_offline:
            raise
        config = AutoConfig.from_pretrained(teacher_path, local_files_only=False)

    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type.startswith("dinov3"):
        return Dinov3withNorm(dinov3_path=teacher_path, normalize=normalize)
    raise ValueError(
        f"Unsupported teacher model_type={model_type!r} for teacher_path={teacher_path!r}."
    )
