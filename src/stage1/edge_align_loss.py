"""Edge-alignment loss for Stage-1 RAE training.

Applies a Sobel gradient filter (spatial domain, no ringing) to both the
student and teacher input images, then aligns the resulting patch tokens
using cosine-similarity or MSE loss.

Compared to struct_align (which constrains token similarity matrices),
edge_align constrains the token values themselves on edge-filtered inputs,
providing a direct structural signal without frequency-domain artifacts.

Supports:
  * Single-layer (final-layer) and multi-layer alignment.
  * Cosine-similarity and MSE loss types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from stage1.encoders.dinov3 import Dinov3withNorm


# ---------------------------------------------------------------------------
# Sobel filter
# ---------------------------------------------------------------------------

def apply_sobel_edge(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude per channel.

    Args:
        x: ``(B, C, H, W)`` float tensor in [0, 1].

    Returns:
        Gradient magnitude map of the same shape, values in [0, ~∞).
        No ringing (pure spatial-domain 3×3 convolution).
    """
    if x.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got {tuple(x.shape)}")
    C = x.shape[1]
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32, device=x.device,
    ).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32, device=x.device,
    ).view(1, 1, 3, 3).expand(C, 1, 3, 3).contiguous()
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class EdgeAlignmentConfig:
    teacher_path: str = ""
    enabled: bool = False
    start_epoch: int = 0
    end_epoch: Optional[int] = None
    weight: float = 1.0
    warmup_steps: int = 0
    warmup_schedule: str = "none"   # "none" | "linear" | "cosine"
    normalize: bool = True
    loss_type: str = "cosine"       # "cosine" | "mse"
    layers: Optional[list] = None
    layer_weights: Optional[list] = None


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class EdgeAlignmentLoss(torch.nn.Module):
    """Align student patch tokens (on Sobel-filtered input) to a frozen teacher.

    Both student and teacher images are passed through the Sobel filter before
    being forwarded through their respective encoders.  The resulting patch
    tokens are aligned directly (not via a similarity matrix), capturing
    edge/texture structure without frequency-domain ringing.

    Multi-layer support (``layers`` / ``layer_weights``):
      When ``layers`` is provided, the loss is computed at each specified
      transformer block and summed with ``layer_weights``.  If ``layers`` is
      ``None`` the loss is computed on the final-layer tokens only.
    """

    def __init__(
        self,
        teacher_path: str,
        *,
        normalize: bool = True,
        loss_type: str = "cosine",
        layers: Optional[list[int]] = None,
        layer_weights: Optional[list[float]] = None,
    ) -> None:
        super().__init__()

        loss_type_l = str(loss_type).lower()
        if loss_type_l not in {"cosine", "mse"}:
            raise ValueError(
                f"Unsupported edge_align loss_type '{loss_type}'. "
                "Choose from ['cosine', 'mse']."
            )
        self.loss_type = loss_type_l

        self.teacher = Dinov3withNorm(dinov3_path=teacher_path, normalize=normalize)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        proc = AutoImageProcessor.from_pretrained(teacher_path)
        self.register_buffer(
            "image_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False
        )

        # Multi-layer config (mirrors SpatialGradientAlignmentLoss)
        if layers is None or len(layers) == 0:
            self.layers: Optional[list[int]] = None
            self.layer_weights: Optional[list[float]] = None
        else:
            self.layers = [int(x) for x in list(layers)]
            if layer_weights is None or len(layer_weights) == 0:
                self.layer_weights = [1.0 for _ in self.layers]
            else:
                self.layer_weights = [float(x) for x in list(layer_weights)]
                if len(self.layer_weights) != len(self.layers):
                    raise ValueError(
                        "edge_align.layer_weights must have the same length as edge_align.layers; "
                        f"got {len(self.layer_weights)} vs {len(self.layers)}"
                    )
            if any(w < 0 for w in self.layer_weights):
                raise ValueError("edge_align.layer_weights must be non-negative.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_teacher_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.image_mean.to(img.device)) / self.image_std.to(img.device)

    def _token_loss(
        self, student_tokens: torch.Tensor, teacher_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Cosine or MSE loss between two (B, N, C) token maps."""
        if self.loss_type == "mse":
            return F.mse_loss(student_tokens, teacher_tokens, reduction="mean")
        # cosine: 1 - cos_sim per token, averaged
        return (
            1.0 - F.cosine_similarity(student_tokens, teacher_tokens, dim=-1, eps=1e-6)
        ).mean()

    # ------------------------------------------------------------------
    # Teacher feature extraction (frozen, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _teacher_patch_tokens(self, sobel_image: torch.Tensor) -> torch.Tensor:
        """Return teacher patch tokens for the final layer on a Sobel-filtered image."""
        self.teacher.eval()
        inp = self._normalize_teacher_image(sobel_image)
        return self.teacher(inp).float()

    @torch.no_grad()
    def _teacher_patch_tokens_by_layer(
        self, sobel_image: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Return teacher patch tokens for each requested layer on a Sobel-filtered image."""
        if self.layers is None:
            raise RuntimeError(
                "_teacher_patch_tokens_by_layer() called but layers are not configured."
            )
        self.teacher.eval()
        inp = self._normalize_teacher_image(sobel_image)
        tokens = self.teacher.forward_tokens(inp, layers=self.layers)
        patch_by_layer = tokens.get("patch_tokens_by_layer", None)
        if patch_by_layer is None:
            raise ValueError(
                "Teacher encoder did not return patch_tokens_by_layer for requested layers."
            )
        return {int(k): v.float() for k, v in patch_by_layer.items()}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        student_patch_tokens: torch.Tensor,
        teacher_image: torch.Tensor,
        student_feat_by_layer: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute edge-alignment loss.

        Args:
            student_patch_tokens: ``(B, N, C)`` — student final-layer patch tokens
                extracted from the Sobel-filtered image.
            teacher_image: ``(B, 3, H, W)`` — teacher input image (already resized
                to encoder resolution, **not** normalised).
            student_feat_by_layer: ``{layer_idx: (B, N, C)}`` — student intermediate
                patch tokens from the Sobel-filtered image.  Required when
                ``layers`` is set.
        """
        student_patch_tokens = student_patch_tokens.float()

        # Apply Sobel to teacher image
        teacher_sobel = apply_sobel_edge(teacher_image)

        # ── Single-layer path ────────────────────────────────────────
        if self.layers is None:
            teacher_tokens = self._teacher_patch_tokens(teacher_sobel)
            return self._token_loss(student_patch_tokens, teacher_tokens)

        # ── Multi-layer path ─────────────────────────────────────────
        if student_feat_by_layer is None:
            raise ValueError(
                "student_feat_by_layer must be provided when edge_align.layers is set. "
                "Ensure rae.distill_layers includes the edge_align layers."
            )
        teacher_by_layer = self._teacher_patch_tokens_by_layer(teacher_sobel)

        total = None
        assert self.layer_weights is not None
        for layer_i, w in zip(self.layers, self.layer_weights):
            if w == 0.0:
                continue
            if layer_i not in student_feat_by_layer:
                raise ValueError(f"Missing student feature for edge_align layer {layer_i}.")
            if layer_i not in teacher_by_layer:
                raise ValueError(f"Missing teacher feature for edge_align layer {layer_i}.")
            s_patch = student_feat_by_layer[layer_i].float()
            t_patch = teacher_by_layer[layer_i]
            loss = self._token_loss(s_patch, t_patch) * float(w)
            total = loss if total is None else (total + loss)

        if total is None:
            any_tensor = next(iter(teacher_by_layer.values()))
            return any_tensor.new_zeros(())
        return total
