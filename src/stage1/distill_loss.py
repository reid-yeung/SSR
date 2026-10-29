from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from stage1.teacher_utils import build_teacher_encoder
from stage1.fft_filter import apply_abs_high_pass
from stage1.ghpf_filter import apply_ghpf


@dataclass
class DinoV3DistillConfig:
    teacher_path: str
    normalize: bool = True
    weight: float = 1.0
    start_epoch: int = 0
    # "mse" (default) or "cosine"
    loss_type: str = "mse"
    # Scheme-B transformer block indices (1..num_hidden_layers). If None/empty, use last layer only.
    layers: list[int] | None = None
    layer_weights: list[float] | None = None
    fft_filter_enabled: bool = False
    fft_filter_cutoff_ratio: float = 0.0
    fft_filter_method: str = "ft"
    fft_filter_student_enable: bool = False
    fft_filter_orig_weight: float = 0.5
    fft_filter_fft_weight: float = 0.5


def _apply_fft_filter(images: torch.Tensor, cutoff_ratio: float, *, method: str) -> torch.Tensor:
    method_l = str(method).lower()
    if method_l == "ft":
        filtered, _ = apply_abs_high_pass(images, cutoff_ratio)
        return filtered
    if method_l == "ghpf":
        return apply_ghpf(images, cutoff_ratio)
    raise ValueError(f"Unsupported fft_filter method '{method}'. Choose from ['ft', 'ghpf'].")


class DinoV3DistillLoss(torch.nn.Module):
    """Feature distillation loss for DinoV3 patch tokens.

    Student features are expected to be the DinoV3 patch tokens of shape (B, N, C),
    matching `Dinov3withNorm.forward()` output (CLS+register tokens removed).

    The teacher is a frozen DinoV3 model.
    """

    def __init__(
        self,
        teacher_path: str,
        *,
        normalize: bool = True,
        loss_type: str = "mse",
        layers: list[int] | None = None,
        layer_weights: list[float] | None = None,
        fft_filter_enabled: bool = False,
        fft_filter_cutoff_ratio: float = 0.0,
        fft_filter_method: str = "ft",
    ):
        super().__init__()
        self.teacher = build_teacher_encoder(teacher_path, normalize=normalize)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        proc = AutoImageProcessor.from_pretrained(teacher_path)
        self.register_buffer("image_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False)
        self.fft_filter_enabled = bool(fft_filter_enabled)
        self.fft_filter_cutoff_ratio = float(fft_filter_cutoff_ratio)
        self.fft_filter_method = str(fft_filter_method).lower()
        if self.fft_filter_method not in {"ft", "ghpf"}:
            raise ValueError(
                f"Unsupported fft_filter_method '{fft_filter_method}'. Choose from ['ft', 'ghpf']."
            )
        if self.fft_filter_enabled and self.fft_filter_cutoff_ratio <= 0.0:
            raise ValueError("fft_filter_cutoff_ratio must be positive when fft_filter_enabled is True")

        loss_type_l = str(loss_type).lower()
        if loss_type_l not in {"mse", "cosine"}:
            raise ValueError(f"Unsupported distillation loss_type '{loss_type}'. Choose from ['mse', 'cosine'].")
        self.loss_type = loss_type_l

        if layers is None or len(layers) == 0:
            self.layers: list[int] | None = None
            self.layer_weights: list[float] | None = None
        else:
            self.layers = [int(x) for x in list(layers)]
            if layer_weights is None or len(layer_weights) == 0:
                self.layer_weights = [1.0 for _ in self.layers]
            else:
                self.layer_weights = [float(x) for x in list(layer_weights)]
                if len(self.layer_weights) != len(self.layers):
                    raise ValueError(
                        "distill.layer_weights must have the same length as distill.layers; "
                        f"got {len(self.layer_weights)} vs {len(self.layers)}"
                    )

            if any(w < 0 for w in self.layer_weights):
                raise ValueError("distill.layer_weights must be non-negative.")

    def _pairwise_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        if teacher_feat.shape != student_feat.shape:
            raise ValueError(
                f"Distill feature shape mismatch: teacher={tuple(teacher_feat.shape)} vs student={tuple(student_feat.shape)}"
            )

        if self.loss_type == "mse":
            return F.mse_loss(student_feat, teacher_feat, reduction="mean")

        student_norm = F.normalize(student_feat, dim=-1, eps=1e-6)
        teacher_norm = F.normalize(teacher_feat, dim=-1, eps=1e-6)
        cos_sim = (student_norm * teacher_norm).sum(dim=-1)
        return (1.0 - cos_sim).mean()

    @torch.no_grad()
    def teacher_features(
        self,
        teacher_image: torch.Tensor,
        *,
        apply_fft: bool | None = None,
        fft_filter_cutoff_ratio: float | None = None,
    ) -> torch.Tensor:
        self.teacher.eval()
        teacher_input = teacher_image
        use_fft = self.fft_filter_enabled if apply_fft is None else bool(apply_fft)
        if use_fft:
            cutoff_ratio = self.fft_filter_cutoff_ratio if fft_filter_cutoff_ratio is None else float(fft_filter_cutoff_ratio)
            if cutoff_ratio <= 0.0:
                raise ValueError("fft_filter_cutoff_ratio must be positive when FFT is enabled")
            teacher_input = _apply_fft_filter(teacher_input, cutoff_ratio, method=self.fft_filter_method)
        teacher_input = (teacher_input - self.image_mean.to(teacher_input.device)) / self.image_std.to(teacher_input.device)
        return self.teacher(teacher_input).float()

    @torch.no_grad()
    def teacher_features_by_layer(
        self,
        teacher_image: torch.Tensor,
        *,
        apply_fft: bool | None = None,
        fft_filter_cutoff_ratio: float | None = None,
    ) -> dict[int, torch.Tensor]:
        if self.layers is None:
            raise RuntimeError("teacher_features_by_layer() called but layers are not configured.")
        self.teacher.eval()
        teacher_input = teacher_image
        use_fft = self.fft_filter_enabled if apply_fft is None else bool(apply_fft)
        if use_fft:
            cutoff_ratio = self.fft_filter_cutoff_ratio if fft_filter_cutoff_ratio is None else float(fft_filter_cutoff_ratio)
            if cutoff_ratio <= 0.0:
                raise ValueError("fft_filter_cutoff_ratio must be positive when FFT is enabled")
            teacher_input = _apply_fft_filter(teacher_input, cutoff_ratio, method=self.fft_filter_method)
        teacher_input = (teacher_input - self.image_mean.to(teacher_input.device)) / self.image_std.to(teacher_input.device)
        tokens = self.teacher.forward_tokens(teacher_input, layers=self.layers)
        patch_by_layer = tokens.get("patch_tokens_by_layer", None)
        if patch_by_layer is None:
            raise ValueError("Teacher encoder did not return patch_tokens_by_layer for requested layers.")
        # Ensure float32 for stable loss.
        return {int(k): v.float() for k, v in patch_by_layer.items()}

    def forward(
        self,
        *,
        teacher_image: torch.Tensor,
        student_feat: torch.Tensor | None = None,
        student_feat_by_layer: dict[int, torch.Tensor] | None = None,
        apply_fft: bool | None = None,
        fft_filter_cutoff_ratio: float | None = None,
    ) -> torch.Tensor:
        # Backward-compatible single-layer path.
        if self.layers is None:
            if student_feat is None:
                raise ValueError("student_feat must be provided when distill.layers is not set.")
            teacher_feat = self.teacher_features(
                teacher_image,
                apply_fft=apply_fft,
                fft_filter_cutoff_ratio=fft_filter_cutoff_ratio,
            )
            return self._pairwise_loss(student_feat.float(), teacher_feat.float())

        # Multi-layer path.
        if student_feat_by_layer is None:
            raise ValueError("student_feat_by_layer must be provided when distill.layers is set.")
        teacher_by_layer = self.teacher_features_by_layer(
            teacher_image,
            apply_fft=apply_fft,
            fft_filter_cutoff_ratio=fft_filter_cutoff_ratio,
        )
        total = None
        assert self.layer_weights is not None
        for layer_i, w in zip(self.layers, self.layer_weights):
            if w == 0.0:
                continue
            if layer_i not in student_feat_by_layer:
                raise ValueError(f"Missing student feature for distill layer {layer_i}.")
            if layer_i not in teacher_by_layer:
                raise ValueError(f"Missing teacher feature for distill layer {layer_i}.")
            l = self._pairwise_loss(student_feat_by_layer[layer_i].float(), teacher_by_layer[layer_i].float())
            l = l * float(w)
            total = l if total is None else (total + l)
        if total is None:
            # All weights are zero.
            any_tensor = next(iter(teacher_by_layer.values()))
            return any_tensor.new_zeros(())
        return total
