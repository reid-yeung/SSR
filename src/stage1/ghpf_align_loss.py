"""GHPF-based alignment loss for Stage-1 RAE training.

Student receives the original image; teacher receives a Gaussian high-pass
filtered image. The loss aligns student and teacher patch latent tokens
directly.

Compared to edge_align (Sobel, both student and teacher see filtered input),
ghpf_align keeps the student on the original-image domain while the teacher
provides a frequency-domain structural target.

Supports:
    * Direct patch-token alignment.
    * Optional class-token-only cosine alignment.
    * Single-layer (final-layer) and multi-layer alignment.
    * Cosine and MSE loss types.
    * Teacher-side GHPF with configurable cutoff ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from stage1.teacher_utils import build_teacher_encoder
from stage1.ghpf_filter import apply_ghpf


@dataclass
class GHPFAlignmentConfig:
    teacher_path: str = ""
    enabled: bool = False
    start_epoch: int = 0
    end_epoch: Optional[int] = None
    weight: float = 1.0
    warmup_steps: int = 0
    warmup_schedule: str = "none"  # "none" | "linear" | "cosine"
    normalize: bool = True
    loss_type: str = "cosine"  # "cosine" | "mse"
    class_token_cosine_only: bool = False  # if True, align only cls token with cosine loss
    layers: Optional[list] = None
    layer_weights: Optional[list] = None
    cutoff_ratio: float = 0.05  # GHPF cutoff: d0 = cutoff_ratio * min(H, W)
    use_relaxed_cosine: bool = False  # if True, only penalize cos_sim < (1 - relaxed_margin)
    relaxed_margin: float = 0.2       # margin threshold for relaxed cosine loss
    normalize_hp: bool = False        # if True, per-batch min-max normalize GHPF output to [0,1] before teacher
    use_energy_mask: bool = False     # if True, mask out low-energy patches before computing loss
    energy_mask_quantile: float = 0.3 # patches below this energy quantile are masked out
    use_patch_affinity: bool = False  # if True, align patch→patch affinity matrix (like struct_align)
    temperature: float = 1.0          # temperature for affinity scaling (only used when use_patch_affinity=True)


class GHPFAlignmentLoss(torch.nn.Module):
    """Align student tokens (on original input) to a frozen teacher.

    Student sees the original image; teacher sees a Gaussian high-pass edge map.

    By default, alignment is done on patch tokens. When
    ``class_token_cosine_only=True``, only class-token cosine alignment is used.

    Multi-layer support (``layers`` / ``layer_weights``):
      When ``layers`` is provided, the loss is computed at each specified
      transformer block and summed with ``layer_weights``. If ``layers`` is
      ``None`` the loss is computed on the final-layer tokens only.
    """

    def __init__(
        self,
        teacher_path: str,
        *,
        normalize: bool = True,
        loss_type: str = "cosine",
        class_token_cosine_only: bool = False,
        cutoff_ratio: float = 0.05,
        layers: Optional[list[int]] = None,
        layer_weights: Optional[list[float]] = None,
        use_relaxed_cosine: bool = False,
        relaxed_margin: float = 0.2,
        normalize_hp: bool = False,
        use_energy_mask: bool = False,
        energy_mask_quantile: float = 0.3,
        use_patch_affinity: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        loss_type_l = str(loss_type).lower()
        if loss_type_l not in {"cosine", "mse", "kl"}:
            raise ValueError(
                f"Unsupported ghpf_align loss_type '{loss_type}'. "
                "Choose from ['cosine', 'mse', 'kl']."
            )
        self.class_token_cosine_only = bool(class_token_cosine_only)
        if self.class_token_cosine_only and loss_type_l != "cosine":
            raise ValueError(
                "ghpf_align.class_token_cosine_only=True requires ghpf_align.loss_type='cosine'."
            )
        if loss_type_l == "kl" and not use_patch_affinity:
            raise ValueError(
                "ghpf_align.loss_type='kl' is only supported when use_patch_affinity=True."
            )
        if not (0.0 < float(cutoff_ratio) <= 1.0):
            raise ValueError(
                f"ghpf_align.cutoff_ratio must be in (0, 1], got {cutoff_ratio}"
            )
        self.loss_type = loss_type_l
        self.cutoff_ratio = float(cutoff_ratio)
        self.use_relaxed_cosine = bool(use_relaxed_cosine)
        self.relaxed_margin = float(relaxed_margin)
        self.normalize_hp = bool(normalize_hp)
        self.use_energy_mask = bool(use_energy_mask)
        self.energy_mask_quantile = float(energy_mask_quantile)
        self.use_patch_affinity = bool(use_patch_affinity)
        if float(temperature) <= 0.0:
            raise ValueError("ghpf_align.temperature must be positive")
        self.temperature = float(temperature)
        self.teacher = build_teacher_encoder(teacher_path, normalize=normalize)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        proc = AutoImageProcessor.from_pretrained(teacher_path)
        self.register_buffer(
            "image_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False
        )

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
                        "ghpf_align.layer_weights must have the same length as ghpf_align.layers; "
                        f"got {len(self.layer_weights)} vs {len(self.layers)}"
                    )
            if any(w < 0 for w in self.layer_weights):
                raise ValueError("ghpf_align.layer_weights must be non-negative.")

    def _normalize_teacher_image(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.image_mean.to(img.device)) / self.image_std.to(img.device)

    def _compute_energy_mask(self, teacher_hp: torch.Tensor, n_patches: int) -> torch.Tensor:
        """Compute a per-patch energy mask from the GHPF image.

        Patches whose spatial energy falls below ``energy_mask_quantile`` are
        masked out (False), so the loss ignores flat/textureless regions where
        the teacher token is unreliable.

        Args:
            teacher_hp: (B, C, H, W) GHPF-filtered image (output of _prepare_teacher_image).
            n_patches: total number of patch tokens N = (H/patch_size)^2.
        Returns:
            mask: (B, N) bool tensor, True = keep this patch.
        """
        B, C, H, W = teacher_hp.shape
        grid = int(n_patches ** 0.5)
        patch_size = H // grid
        if grid * grid != n_patches or grid * patch_size != H:
            # Fallback: skip masking rather than producing a wrong-shaped mask.
            return teacher_hp.new_ones(B, n_patches, dtype=torch.bool)

        energy = teacher_hp.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        energy_patches = F.avg_pool2d(energy, kernel_size=patch_size, stride=patch_size)  # (B, 1, grid, grid)
        energy_flat = energy_patches.flatten(1)  # (B, N)

        threshold = energy_flat.quantile(self.energy_mask_quantile, dim=1, keepdim=True)  # (B, 1)
        return energy_flat > threshold  # (B, N) bool

    def _patch_affinity_sim(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Compute patch→patch affinity matrix (B, N, N) scaled by temperature.
        Mirrors struct_align's implementation exactly.
        """
        normed = F.normalize(patch_tokens, dim=-1)
        return torch.bmm(normed, normed.transpose(1, 2)) / self.temperature

    def _pairwise_loss(self, student_sim: torch.Tensor, teacher_sim: torch.Tensor) -> torch.Tensor:
        """KL, MSE, or cosine loss between two affinity matrices.
        Mirrors struct_align's _pairwise_loss exactly.
        """
        if student_sim.shape != teacher_sim.shape:
            raise ValueError(
                f"Affinity shape mismatch: student={tuple(student_sim.shape)} vs teacher={tuple(teacher_sim.shape)}"
            )
        if self.loss_type == "mse":
            return F.mse_loss(student_sim, teacher_sim, reduction="mean")
        if self.loss_type == "cosine":
            B = student_sim.shape[0]
            s = student_sim.view(B, -1)
            t = teacher_sim.view(B, -1)
            return (1.0 - F.cosine_similarity(s, t, dim=-1, eps=1e-6)).mean()
        # KL (default for affinity mode)
        B = student_sim.shape[0]
        target_prob = F.softmax(teacher_sim.view(B, -1), dim=-1)
        pred_log_prob = F.log_softmax(student_sim.view(B, -1), dim=-1)
        loss = F.kl_div(pred_log_prob, target_prob, reduction="batchmean")
        return loss * (self.temperature ** 2)

    def _token_loss(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cosine or MSE loss between two patch-token tensors of the same shape.

        Args:
            student_tokens: (B, N, C)
            teacher_tokens: (B, N, C)
            mask: optional (B, N) bool tensor; if provided, only masked patches
                  contribute to the loss (weighted mean to stay DDP-safe).
        """
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                f"Token shape mismatch: student={tuple(student_tokens.shape)} vs teacher={tuple(teacher_tokens.shape)}"
            )
        if student_tokens.dim() != 3:
            raise ValueError(
                f"Token tensors must be 3D (B, N, C), got student={tuple(student_tokens.shape)}"
            )

        if self.loss_type == "mse":
            loss = F.mse_loss(student_tokens, teacher_tokens, reduction="none").mean(dim=-1)  # (B, N)
        else:
            s = F.normalize(student_tokens, dim=-1)
            t = F.normalize(teacher_tokens, dim=-1)
            cos_sim = F.cosine_similarity(s, t, dim=-1, eps=1e-6)  # (B, N)
            if self.use_relaxed_cosine:
                loss = F.relu((1.0 - self.relaxed_margin) - cos_sim)
            else:
                loss = 1.0 - cos_sim  # (B, N)

        if mask is None:
            return loss.mean()
        # Weighted mean over masked patches — DDP-safe (no variable-length indexing)
        w = mask.float()
        return (loss * w).sum() / w.sum().clamp(min=1.0)

    @staticmethod
    def _to_2d_cls(cls_token: torch.Tensor, *, name: str) -> torch.Tensor:
        if cls_token.dim() == 2:
            return cls_token
        if cls_token.dim() == 3 and cls_token.shape[1] == 1:
            return cls_token[:, 0, :]
        raise ValueError(
            f"{name} must be (B, C) or (B, 1, C), got {tuple(cls_token.shape)}"
        )

    def _cls_cosine_loss(self, student_cls: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        if student_cls.shape != teacher_cls.shape:
            raise ValueError(
                f"CLS shape mismatch: student={tuple(student_cls.shape)} vs teacher={tuple(teacher_cls.shape)}"
            )
        if student_cls.dim() != 2:
            raise ValueError(f"CLS tensors must be 2D (B, C), got {tuple(student_cls.shape)}")

        s = F.normalize(student_cls, dim=-1)
        t = F.normalize(teacher_cls, dim=-1)
        cos_sim = F.cosine_similarity(s, t, dim=-1, eps=1e-6)  # (B,)

        if self.use_relaxed_cosine:
            loss = F.relu((1.0 - self.relaxed_margin) - cos_sim)
        else:
            loss = 1.0 - cos_sim
        return loss.mean()

    def _prepare_teacher_image(self, teacher_image: torch.Tensor) -> torch.Tensor:
        hp = apply_ghpf(teacher_image, self.cutoff_ratio)
        if self.normalize_hp:
            hp_min = hp.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
            hp_max = hp.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
            hp = (hp - hp_min) / (hp_max - hp_min + 1e-6)
        return hp

    def _prepare_teacher_input(self, teacher_hp: torch.Tensor) -> torch.Tensor:
        """Normalize GHPF image for teacher forward pass.

        When normalize_hp=True the image is already in [0, 1] and has a very
        different distribution from natural images, so we skip the ImageNet
        mean/std normalization to avoid feeding the teacher OOD values.
        When normalize_hp=False we apply the standard ImageNet normalization.
        """
        if self.normalize_hp:
            return teacher_hp
        return self._normalize_teacher_image(teacher_hp)

    @torch.no_grad()
    def _teacher_tokens(self, teacher_image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return teacher cls + patch tokens for the final layer."""
        self.teacher.eval()
        inp = self._prepare_teacher_input(teacher_image)
        tokens = self.teacher.forward_tokens(inp)
        return {
            "cls_token": tokens["cls_token"].float(),
            "patch_tokens": tokens["patch_tokens"].float(),
        }

    @torch.no_grad()
    def _teacher_tokens_by_layer(
        self,
        teacher_image: torch.Tensor,
    ) -> dict[str, dict[int, torch.Tensor]]:
        """Return teacher cls + patch tokens for each requested layer."""
        if self.layers is None:
            raise RuntimeError("_teacher_tokens_by_layer() called but layers are not configured.")
        self.teacher.eval()
        inp = self._prepare_teacher_input(teacher_image)
        tokens = self.teacher.forward_tokens(inp, layers=self.layers)
        patch_by_layer = tokens.get("patch_tokens_by_layer", None)
        if patch_by_layer is None:
            raise ValueError("Teacher encoder did not return patch_tokens_by_layer for requested layers.")
        cls_by_layer = tokens.get("cls_tokens_by_layer", None)
        if cls_by_layer is None:
            raise ValueError("Teacher encoder did not return cls_tokens_by_layer for requested layers.")
        return {
            "patch_tokens_by_layer": {int(k): v.float() for k, v in patch_by_layer.items()},
            "cls_tokens_by_layer": {int(k): v.float() for k, v in cls_by_layer.items()},
        }

    def _single_layer_loss(
        self,
        *,
        student_patch_tokens: torch.Tensor,
        teacher_patch_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_patch_affinity:
            # Affinity mode: align patch→patch similarity matrices.
            # energy_mask is not applied here — affinity is a global relational
            # structure and masking individual patches would distort the matrix.
            student_sim = self._patch_affinity_sim(student_patch_tokens)
            teacher_sim = self._patch_affinity_sim(teacher_patch_tokens)
            return self._pairwise_loss(student_sim, teacher_sim)
        return self._token_loss(student_patch_tokens, teacher_patch_tokens, mask=mask)

    def forward(
        self,
        *,
        student_patch_tokens: torch.Tensor,
        teacher_image: torch.Tensor,
        student_cls_token: Optional[torch.Tensor] = None,
        student_feat_by_layer: Optional[dict[int, torch.Tensor]] = None,
        student_cls_by_layer: Optional[dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute GHPF alignment loss."""
        teacher_input = self._prepare_teacher_image(teacher_image)

        # Compute energy mask once from the GHPF image (shared across all layers)
        energy_mask: Optional[torch.Tensor] = None
        if self.use_energy_mask and not self.class_token_cosine_only:
            n_patches = student_patch_tokens.shape[1]
            energy_mask = self._compute_energy_mask(teacher_input, n_patches)

        if self.class_token_cosine_only:
            if student_cls_token is None:
                raise ValueError(
                    "student_cls_token must be provided when ghpf_align.class_token_cosine_only is enabled."
                )
            student_cls = self._to_2d_cls(student_cls_token, name="student_cls_token").float()

            if self.layers is None:
                teacher_cls = self._teacher_tokens(teacher_input)["cls_token"]
                return self._cls_cosine_loss(student_cls, teacher_cls)

            if student_cls_by_layer is None:
                raise ValueError(
                    "student_cls_by_layer must be provided when ghpf_align.layers is set and "
                    "ghpf_align.class_token_cosine_only is enabled."
                )
            teacher_by_layer_all = self._teacher_tokens_by_layer(teacher_input)
            teacher_cls_by_layer = teacher_by_layer_all["cls_tokens_by_layer"]

            total = None
            assert self.layer_weights is not None
            for layer_i, w in zip(self.layers, self.layer_weights):
                if w == 0.0:
                    continue
                if layer_i not in student_cls_by_layer:
                    raise ValueError(f"Missing student cls feature for ghpf_align layer {layer_i}.")
                if layer_i not in teacher_cls_by_layer:
                    raise ValueError(f"Missing teacher cls feature for ghpf_align layer {layer_i}.")
                s_cls = self._to_2d_cls(student_cls_by_layer[layer_i], name=f"student_cls_by_layer[{layer_i}]").float()
                t_cls = self._to_2d_cls(teacher_cls_by_layer[layer_i], name=f"teacher_cls_by_layer[{layer_i}]").float()
                loss = self._cls_cosine_loss(s_cls, t_cls) * float(w)
                total = loss if total is None else (total + loss)

            if total is None:
                any_tensor = next(iter(teacher_cls_by_layer.values()))
                return any_tensor.new_zeros(())
            return total

        if student_patch_tokens.dim() != 3:
            raise ValueError(f"student_patch_tokens must be 3D (B, N, C), got {tuple(student_patch_tokens.shape)}")
        student_patch_tokens = student_patch_tokens.float()

        if self.layers is None:
            teacher_patch_tokens = self._teacher_tokens(teacher_input)["patch_tokens"]
            return self._single_layer_loss(
                student_patch_tokens=student_patch_tokens,
                teacher_patch_tokens=teacher_patch_tokens,
                mask=energy_mask,
            )

        if student_feat_by_layer is None:
            raise ValueError(
                "student_feat_by_layer must be provided when ghpf_align.layers is set. "
                "Ensure rae.distill_layers includes the ghpf_align layers."
            )
        teacher_by_layer = self._teacher_tokens_by_layer(teacher_input)["patch_tokens_by_layer"]

        total = None
        assert self.layer_weights is not None
        for layer_i, w in zip(self.layers, self.layer_weights):
            if w == 0.0:
                continue
            if layer_i not in student_feat_by_layer:
                raise ValueError(f"Missing student feature for ghpf_align layer {layer_i}.")
            if layer_i not in teacher_by_layer:
                raise ValueError(f"Missing teacher feature for ghpf_align layer {layer_i}.")
            s_patch = student_feat_by_layer[layer_i].float()
            t_patch = teacher_by_layer[layer_i]
            loss = self._single_layer_loss(
                student_patch_tokens=s_patch,
                teacher_patch_tokens=t_patch,
                mask=energy_mask,
            ) * float(w)
            total = loss if total is None else (total + loss)

        if total is None:
            any_tensor = next(iter(teacher_by_layer.values()))
            return any_tensor.new_zeros(())
        return total
