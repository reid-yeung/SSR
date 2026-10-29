"""Spatial-gradient alignment loss for Stage-1 RAE training.

Aligns the *spatial gradient* (finite-difference along H and W axes) of student
patch tokens to those of a frozen DINOv3 teacher.  The gradient captures where
feature values change across the spatial grid — i.e. edges, texture boundaries,
and structural transitions — making this an **explicit** structural alignment
signal that complements the existing distillation (absolute value) and
structural-attention (relational pattern) losses.

Supports:
  * Single-layer (final-layer) and multi-layer alignment.
  * Cosine-similarity and MSE loss types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from stage1.encoders.dinov3 import Dinov3withNorm


# ---------------------------------------------------------------------------
# Config dataclass (mirrors StructuralAttentionAlignmentConfig)
# ---------------------------------------------------------------------------

@dataclass
class SpatialGradientAlignmentConfig:
    teacher_path: str = ""
    enabled: bool = False
    start_epoch: int = 0
    end_epoch: Optional[int] = None
    weight: float = 1.0
    warmup_steps: int = 0
    warmup_schedule: str = "none"          # "none" | "linear" | "cosine"
    normalize: bool = True
    loss_type: str = "cosine"              # "cosine" | "mse"
    layers: Optional[list[int]] = None
    layer_weights: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class SpatialGradientAlignmentLoss(torch.nn.Module):
    """Align spatial gradients of student patch tokens to a frozen DINOv3 teacher.

    Patch tokens of shape ``(B, N, C)`` are reshaped to ``(B, H, W, C)`` where
    ``H = W = sqrt(N)``.  Finite differences along H (vertical) and W
    (horizontal) yield gradient maps of shape ``(B, H-1, W, C)`` and
    ``(B, H, W-1, C)`` respectively.  The loss aligns these gradient maps
    between student and teacher using cosine similarity or MSE.

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
                f"Unsupported spatial_grad loss_type '{loss_type}'. Choose from ['cosine', 'mse']."
            )
        self.loss_type = loss_type_l

        # Frozen teacher
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

        # Multi-layer config
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
                        "spatial_grad.layer_weights must have the same length as spatial_grad.layers; "
                        f"got {len(self.layer_weights)} vs {len(self.layers)}"
                    )
            if any(w < 0 for w in self.layer_weights):
                raise ValueError("spatial_grad.layer_weights must be non-negative.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_teacher_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.image_mean.to(img.device)) / self.image_std.to(img.device)

    @staticmethod
    def _spatial_gradients(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute finite-difference spatial gradients from patch tokens.

        Args:
            tokens: ``(B, N, C)`` where ``N = H * W``.

        Returns:
            grad_h: ``(B, H-1, W, C)`` — vertical gradient.
            grad_w: ``(B, H, W-1, C)`` — horizontal gradient.
        """
        B, N, C = tokens.shape
        H = W = int(sqrt(N))
        assert H * W == N, f"N={N} is not a perfect square"
        grid = tokens.view(B, H, W, C)
        grad_h = grid[:, 1:, :, :] - grid[:, :-1, :, :]   # (B, H-1, W, C)
        grad_w = grid[:, :, 1:, :] - grid[:, :, :-1, :]   # (B, H, W-1, C)
        return grad_h, grad_w

    def _gradient_loss(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial-gradient alignment loss for one pair of token maps."""
        s_gh, s_gw = self._spatial_gradients(student_tokens)
        t_gh, t_gw = self._spatial_gradients(teacher_tokens)

        if self.loss_type == "mse":
            loss_h = F.mse_loss(s_gh, t_gh, reduction="mean")
            loss_w = F.mse_loss(s_gw, t_gw, reduction="mean")
        else:
            # cosine: 1 - cos_sim, averaged over all spatial positions
            # flatten spatial dims → (B, M, C) then compute per-vector cosine
            loss_h = (
                1.0 - F.cosine_similarity(
                    s_gh.flatten(1, 2), t_gh.flatten(1, 2), dim=-1, eps=1e-6
                )
            ).mean()
            loss_w = (
                1.0 - F.cosine_similarity(
                    s_gw.flatten(1, 2), t_gw.flatten(1, 2), dim=-1, eps=1e-6
                )
            ).mean()

        return loss_h + loss_w

    # ------------------------------------------------------------------
    # Teacher feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _teacher_patch_tokens(self, teacher_image: torch.Tensor) -> torch.Tensor:
        """Return teacher patch tokens for the final layer."""
        self.teacher.eval()
        inp = self._normalize_teacher_image(teacher_image)
        return self.teacher(inp).float()

    @torch.no_grad()
    def _teacher_patch_tokens_by_layer(
        self, teacher_image: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Return teacher patch tokens for each requested layer."""
        if self.layers is None:
            raise RuntimeError("_teacher_patch_tokens_by_layer() called but layers are not configured.")
        self.teacher.eval()
        inp = self._normalize_teacher_image(teacher_image)
        tokens = self.teacher.forward_tokens(inp, layers=self.layers)
        patch_by_layer = tokens.get("patch_tokens_by_layer", None)
        if patch_by_layer is None:
            raise ValueError("Teacher encoder did not return patch_tokens_by_layer for requested layers.")
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
        """Compute spatial-gradient alignment loss.

        Args:
            student_patch_tokens: ``(B, N, C)`` — student final-layer patch tokens.
            teacher_image: ``(B, 3, H, W)`` — teacher input image (already resized
                to encoder resolution, **not** normalised).
            student_feat_by_layer: ``{layer_idx: (B, N, C)}`` — student intermediate
                patch tokens.  Required when ``layers`` is set.
        """
        student_patch_tokens = student_patch_tokens.float()

        # ── Single-layer path ────────────────────────────────────────
        if self.layers is None:
            teacher_tokens = self._teacher_patch_tokens(teacher_image)
            return self._gradient_loss(student_patch_tokens, teacher_tokens)

        # ── Multi-layer path ─────────────────────────────────────────
        if student_feat_by_layer is None:
            raise ValueError(
                "student_feat_by_layer must be provided when spatial_grad.layers is set. "
                "Ensure rae.distill_layers includes the spatial_grad layers."
            )
        teacher_by_layer = self._teacher_patch_tokens_by_layer(teacher_image)

        total = None
        assert self.layer_weights is not None
        for layer_i, w in zip(self.layers, self.layer_weights):
            if w == 0.0:
                continue
            if layer_i not in student_feat_by_layer:
                raise ValueError(f"Missing student feature for spatial_grad layer {layer_i}.")
            if layer_i not in teacher_by_layer:
                raise ValueError(f"Missing teacher feature for spatial_grad layer {layer_i}.")
            s_patch = student_feat_by_layer[layer_i].float()
            t_patch = teacher_by_layer[layer_i]
            loss = self._gradient_loss(s_patch, t_patch) * float(w)
            total = loss if total is None else (total + loss)

        if total is None:
            any_tensor = next(iter(teacher_by_layer.values()))
            return any_tensor.new_zeros(())
        return total
