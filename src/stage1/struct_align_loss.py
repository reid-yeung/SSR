from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from stage1.teacher_utils import build_teacher_encoder
from stage1.fft_filter import apply_abs_high_pass
from stage1.ghpf_filter import apply_ghpf


@dataclass
class StructuralAttentionAlignmentConfig:
    teacher_path: str
    enabled: bool = False
    start_epoch: int = 0
    end_epoch: int | None = None
    weight: float = 1.0
    normalize: bool = True
    temperature: float = 1.0
    loss_type: str = "kl"
    use_patch_affinity: bool = False
    layers: list | None = None
    layer_weights: list | None = None
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


class StructuralAttentionAlignmentLoss(torch.nn.Module):
    """Align student structural similarity distribution to a frozen DINOv3 anchor.

    Two modes controlled by ``use_patch_affinity``:

    * ``use_patch_affinity=False`` (default, original behaviour):
      Aligns cls→patch similarity distribution (B, 1, N).
      ``softmax( (cls @ patch^T) / T )``

    * ``use_patch_affinity=True``:
      Aligns patch→patch affinity matrix (B, N, N).
      ``softmax( (patch @ patch^T) / T )``
      This captures richer relational structure between spatial tokens.

    Multi-layer support (``layers`` / ``layer_weights``):
      When ``layers`` is provided, the loss is computed at each specified
      transformer block and summed with ``layer_weights``.  If ``layers`` is
      None the loss is computed on the final-layer tokens only (original
      behaviour).

    Gradient compensation uses the common distillation scaling (T^2) for KL.
    """

    def __init__(
        self,
        teacher_path: str,
        *,
        normalize: bool = True,
        temperature: float = 1.0,
        loss_type: str = "kl",
        use_patch_affinity: bool = False,
        layers: list[int] | None = None,
        layer_weights: list[float] | None = None,
        fft_filter_enabled: bool = False,
        fft_filter_cutoff_ratio: float = 0.0,
        fft_filter_method: str = "ft",
    ) -> None:
        super().__init__()
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")

        loss_type_l = str(loss_type).lower()
        if loss_type_l not in {"kl", "mse", "cosine"}:
            raise ValueError(f"Unsupported struct_align loss_type '{loss_type}'. Choose from ['kl', 'mse', 'cosine'].")

        self.teacher = build_teacher_encoder(teacher_path, normalize=normalize)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        proc = AutoImageProcessor.from_pretrained(teacher_path)
        self.register_buffer("image_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False)

        self.temperature = float(temperature)
        self.loss_type = loss_type_l
        self.use_patch_affinity = bool(use_patch_affinity)

        # Multi-layer config (mirrors DinoV3DistillLoss)
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
                        "struct_align.layer_weights must have the same length as struct_align.layers; "
                        f"got {len(self.layer_weights)} vs {len(self.layers)}"
                    )
            if any(w < 0 for w in self.layer_weights):
                raise ValueError("struct_align.layer_weights must be non-negative.")

        self.fft_filter_enabled = bool(fft_filter_enabled)
        self.fft_filter_cutoff_ratio = float(fft_filter_cutoff_ratio)
        self.fft_filter_method = str(fft_filter_method).lower()
        if self.fft_filter_method not in {"ft", "ghpf"}:
            raise ValueError(
                f"Unsupported fft_filter_method '{fft_filter_method}'. Choose from ['ft', 'ghpf']."
            )
        if self.fft_filter_enabled and self.fft_filter_cutoff_ratio <= 0.0:
            raise ValueError("fft_filter_cutoff_ratio must be positive when fft_filter_enabled is True")

    def _normalize_teacher_image(self, teacher_image: torch.Tensor) -> torch.Tensor:
        return (teacher_image - self.image_mean.to(teacher_image.device)) / self.image_std.to(teacher_image.device)

    def _patch_affinity_sim(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Compute patch→patch affinity matrix (B, N, N) scaled by temperature."""
        normed = F.normalize(patch_tokens, dim=-1)
        return torch.bmm(normed, normed.transpose(1, 2)) / self.temperature

    def _cls_patch_sim(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Compute cls→patch similarity (B, 1, N) scaled by temperature."""
        if self.loss_type in {"mse", "cosine"}:
            cls_token = F.normalize(cls_token, dim=-1)
            patch_tokens = F.normalize(patch_tokens, dim=-1)
        return torch.bmm(cls_token.unsqueeze(1), patch_tokens.transpose(1, 2)) / self.temperature

    def _pairwise_loss(self, student_sim: torch.Tensor, teacher_sim: torch.Tensor) -> torch.Tensor:
        """Compute KL, MSE, or cosine loss between two similarity tensors of the same shape."""
        if student_sim.shape != teacher_sim.shape:
            raise ValueError(
                f"Similarity shape mismatch: student={tuple(student_sim.shape)} vs teacher={tuple(teacher_sim.shape)}"
            )
        if self.loss_type == "mse":
            return F.mse_loss(student_sim, teacher_sim, reduction="mean")
        if self.loss_type == "cosine":
            # Treat each row of the flattened similarity map as a vector and align directions.
            B = student_sim.shape[0]
            s = student_sim.view(B, -1)
            t = teacher_sim.view(B, -1)
            return (1.0 - F.cosine_similarity(s, t, dim=-1, eps=1e-6)).mean()
        # KL: flatten last two dims so batchmean works correctly for (B, N, N)
        B = student_sim.shape[0]
        target_prob = F.softmax(teacher_sim.view(B, -1), dim=-1)
        pred_log_prob = F.log_softmax(student_sim.view(B, -1), dim=-1)
        loss = F.kl_div(pred_log_prob, target_prob, reduction="batchmean")
        return loss * (self.temperature ** 2)

    @torch.no_grad()
    def _teacher_tokens(
        self,
        teacher_image: torch.Tensor,
        *,
        fft_filter_cutoff_ratio: float | None = None,
        apply_fft: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return teacher cls + patch tokens for the final layer."""
        self.teacher.eval()
        teacher_input = teacher_image
        use_fft = self.fft_filter_enabled if apply_fft is None else bool(apply_fft)
        if use_fft:
            cutoff_ratio = self.fft_filter_cutoff_ratio if fft_filter_cutoff_ratio is None else float(fft_filter_cutoff_ratio)
            if cutoff_ratio <= 0.0:
                raise ValueError("fft_filter_cutoff_ratio must be positive when FFT is enabled")
            teacher_input = _apply_fft_filter(teacher_input, cutoff_ratio, method=self.fft_filter_method)
        teacher_input = self._normalize_teacher_image(teacher_input)
        tokens = self.teacher.forward_tokens(teacher_input)
        return {
            "cls_token": tokens["cls_token"].float(),
            "patch_tokens": tokens["patch_tokens"].float(),
        }

    @torch.no_grad()
    def _teacher_tokens_by_layer(
        self,
        teacher_image: torch.Tensor,
        *,
        fft_filter_cutoff_ratio: float | None = None,
        apply_fft: bool | None = None,
    ) -> dict[int, torch.Tensor]:
        """Return teacher patch tokens for each requested layer."""
        if self.layers is None:
            raise RuntimeError("_teacher_tokens_by_layer() called but layers are not configured.")
        self.teacher.eval()
        teacher_input = teacher_image
        use_fft = self.fft_filter_enabled if apply_fft is None else bool(apply_fft)
        if use_fft:
            cutoff_ratio = self.fft_filter_cutoff_ratio if fft_filter_cutoff_ratio is None else float(fft_filter_cutoff_ratio)
            if cutoff_ratio <= 0.0:
                raise ValueError("fft_filter_cutoff_ratio must be positive when FFT is enabled")
            teacher_input = _apply_fft_filter(teacher_input, cutoff_ratio, method=self.fft_filter_method)
        teacher_input = self._normalize_teacher_image(teacher_input)
        tokens = self.teacher.forward_tokens(teacher_input, layers=self.layers)
        patch_by_layer = tokens.get("patch_tokens_by_layer", None)
        if patch_by_layer is None:
            raise ValueError("Teacher encoder did not return patch_tokens_by_layer for requested layers.")
        return {int(k): v.float() for k, v in patch_by_layer.items()}

    def _single_layer_loss(
        self,
        *,
        student_cls_token: torch.Tensor,
        student_patch_tokens: torch.Tensor,
        teacher_cls_token: torch.Tensor,
        teacher_patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_patch_affinity:
            teacher_sim = self._patch_affinity_sim(teacher_patch_tokens)
            student_sim = self._patch_affinity_sim(student_patch_tokens)
        else:
            teacher_sim = self._cls_patch_sim(teacher_cls_token, teacher_patch_tokens)
            student_sim = self._cls_patch_sim(student_cls_token, student_patch_tokens)
        return self._pairwise_loss(student_sim, teacher_sim)

    def forward(
        self,
        *,
        student_cls_token: torch.Tensor,
        student_patch_tokens: torch.Tensor,
        teacher_image: torch.Tensor,
        student_feat_by_layer: dict[int, torch.Tensor] | None = None,
        fft_filter_cutoff_ratio: float | None = None,
        apply_fft: bool | None = None,
    ) -> torch.Tensor:
        # Normalise student cls token shape
        if student_cls_token.dim() == 3:
            if student_cls_token.shape[1] != 1:
                raise ValueError(f"student_cls_token must have shape (B, 1, C) when 3D, got {tuple(student_cls_token.shape)}")
            student_cls = student_cls_token[:, 0, :]
        elif student_cls_token.dim() == 2:
            student_cls = student_cls_token
        else:
            raise ValueError(f"student_cls_token must be 2D or 3D, got {student_cls_token.dim()}D")

        if student_patch_tokens.dim() != 3:
            raise ValueError(f"student_patch_tokens must be 3D (B, N, C), got {tuple(student_patch_tokens.shape)}")

        student_cls = student_cls.float()
        student_patch_tokens = student_patch_tokens.float()

        # ── Single-layer path (original behaviour) ──────────────────────────
        if self.layers is None:
            teacher_tok = self._teacher_tokens(
                teacher_image,
                fft_filter_cutoff_ratio=fft_filter_cutoff_ratio,
                apply_fft=apply_fft,
            )
            return self._single_layer_loss(
                student_cls_token=student_cls,
                student_patch_tokens=student_patch_tokens,
                teacher_cls_token=teacher_tok["cls_token"],
                teacher_patch_tokens=teacher_tok["patch_tokens"],
            )

        # ── Multi-layer path ─────────────────────────────────────────────────
        if student_feat_by_layer is None:
            raise ValueError(
                "student_feat_by_layer must be provided when struct_align.layers is set. "
                "Ensure rae.distill_layers includes the struct_align layers."
            )
        teacher_by_layer = self._teacher_tokens_by_layer(
            teacher_image,
            fft_filter_cutoff_ratio=fft_filter_cutoff_ratio,
            apply_fft=apply_fft,
        )
        # Also get final-layer teacher tokens for cls (needed in cls→patch mode)
        teacher_final = self._teacher_tokens(teacher_image, apply_fft=False)

        total = None
        assert self.layer_weights is not None
        for layer_i, w in zip(self.layers, self.layer_weights):
            if w == 0.0:
                continue
            if layer_i not in student_feat_by_layer:
                raise ValueError(f"Missing student feature for struct_align layer {layer_i}.")
            if layer_i not in teacher_by_layer:
                raise ValueError(f"Missing teacher feature for struct_align layer {layer_i}.")
            s_patch = student_feat_by_layer[layer_i].float()
            t_patch = teacher_by_layer[layer_i]
            # For cls→patch mode we use the final-layer cls token as the query
            # (intermediate cls tokens are not extracted to keep overhead low).
            l = self._single_layer_loss(
                student_cls_token=student_cls,
                student_patch_tokens=s_patch,
                teacher_cls_token=teacher_final["cls_token"],
                teacher_patch_tokens=t_patch,
            )
            l = l * float(w)
            total = l if total is None else (total + l)

        if total is None:
            any_tensor = next(iter(teacher_by_layer.values()))
            return any_tensor.new_zeros(())
        return total
