# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script with reconstruction, LPIPS, and GAN losses.

This script adapts the training logic from the Kakao Brain VQGAN trainer while
targeting the RAE autoencoder architecture used in this repository.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from glob import glob
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1 import RAE
from stage1.distill_loss import DinoV3DistillLoss
from stage1.struct_align_loss import StructuralAttentionAlignmentLoss
from stage1.spatial_grad_loss import SpatialGradientAlignmentLoss
from stage1.edge_align_loss import EdgeAlignmentLoss, apply_sobel_edge
from stage1.fft_filter import apply_abs_high_pass
from stage1.ghpf_align_loss import GHPFAlignmentLoss
from stage1.fft_align_loss import FFTAlignmentLoss
from stage1.ghpf_recon_loss import GHPFReconLoss
from stage1.ghpf_filter import apply_ghpf

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *
from PIL import Image
import numpy as np


def _strip_lora_from_state_dict(state_dict: dict) -> dict:
    """Remove PEFT/LoRA adapter weights from a module state_dict.

    This keeps the base model weights in the checkpoint, while saving adapters separately.
    """
    filtered = {}
    for k, v in state_dict.items():
        kl = k.lower()
        if ".lora_" in kl or "lora_a" in kl or "lora_b" in kl or "lora_embedding" in kl:
            continue
        if "modules_to_save" in kl:
            continue
        filtered[k] = v
    return filtered


def _get_lora_adapter_dir(checkpoint_path: str) -> str:
    # e.g. ckpts/.../ep-00000012.pt -> ckpts/.../adapters/ep-00000012
    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    parent = os.path.dirname(checkpoint_path)
    return os.path.join(parent, "adapters", base)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging if set.')
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode, rae.forward).")
    parser.add_argument("--benchmark-warmup-steps", type=int, default=0,
                        help="Warmup steps excluded from benchmark timing; disabled unless --benchmark-steps > 0.")
    parser.add_argument("--benchmark-steps", type=int, default=0,
                        help="Measure this many steady-state steps, report CUDA peak memory, then exit without saving.")
    return parser.parse_args()


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def rampup_weight(
    base_weight: float,
    *,
    global_step: int,
    start_step: int,
    warmup_steps: int,
    schedule: str,
) -> float:
    """Ramp a weight from 0 -> base_weight after start_step.

    Args:
        base_weight: Target weight after warmup.
        global_step: Current training step.
        start_step: Step at which ramping starts.
        warmup_steps: Number of steps to ramp up. If <= 0, returns base_weight.
        schedule: one of {"none", "linear", "cosine"}.
    """
    if base_weight <= 0.0:
        return 0.0
    if warmup_steps <= 0:
        return float(base_weight)

    schedule = str(schedule).lower()
    if global_step < start_step:
        return 0.0

    progress = (global_step - start_step) / float(warmup_steps)
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)

    if schedule in ("none", "constant"):
        factor = 1.0
    elif schedule == "linear":
        factor = progress
    elif schedule == "cosine":
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported struct_align warmup_schedule '{schedule}'. Use one of: none, linear, cosine")

    return float(base_weight) * float(factor)


def _parse_ratio_spec(value, *, name: str) -> tuple[float, float]:
    """Parse a scalar or 2-item range spec into (start, end)."""
    if value is None:
        return 0.0, 0.0

    if isinstance(value, (int, float)):
        v = float(value)
        return v, v

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name} must be a float or a 2-item list like [start, end], got len={len(value)}")
        start_v = float(value[0])
        end_v = float(value[1])
        return start_v, end_v

    raise ValueError(f"{name} must be a float or a 2-item list like [start, end], got type={type(value)}")


def schedule_ratio(
    start: float,
    end: float,
    *,
    global_step: int,
    start_step: int,
    warmup_steps: int,
    schedule: str,
) -> float:
    """Schedule a scalar from start -> end after start_step."""
    start = float(start)
    end = float(end)
    if start == end:
        return float(end)

    schedule_l = str(schedule).lower()
    if global_step < start_step:
        return float(start)

    if warmup_steps <= 0:
        return float(end)

    progress = (global_step - start_step) / float(warmup_steps)
    progress = 0.0 if progress < 0.0 else (1.0 if progress > 1.0 else progress)

    if schedule_l in ("none", "constant"):
        factor = 1.0
    elif schedule_l == "linear":
        factor = progress
    elif schedule_l == "cosine":
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(
            f"Unsupported struct_align.fft_filter warmup_schedule '{schedule}'. Use one of: none, linear, cosine"
        )

    return float(start + (end - start) * factor)


def _parse_fft_filter_method(fft_cfg: dict, *, name: str) -> str:
    method = str(fft_cfg.get("method", "ft")).lower()
    if method not in {"ft", "ghpf"}:
        raise ValueError(f"{name}.method must be one of ['ft', 'ghpf'], got '{method}'")
    return method


def _apply_configured_fft_filter(images: torch.Tensor, cutoff_ratio: float, *, method: str) -> torch.Tensor:
    method_l = str(method).lower()
    if method_l == "ft":
        filtered, _ = apply_abs_high_pass(images, cutoff_ratio)
        return filtered
    if method_l == "ghpf":
        return apply_ghpf(images, cutoff_ratio)
    raise ValueError(f"Unsupported fft_filter method '{method}'. Use one of: ft, ghpf")


def _extract_student_distill_tokens(
    rae: RAE,
    teacher_image: torch.Tensor,
    *,
    layers: list[int] | None,
) -> dict:
    """Extract student tokens for alignment losses from a pre-resized image."""
    enc_input = rae._normalize_encoder_input(teacher_image)
    grad_enabled = rae.training and rae._encoder_requires_grad()
    with torch.set_grad_enabled(grad_enabled):
        cls_token = None
        distill_feat_by_layer = None
        distill_cls_by_layer = None
        if hasattr(rae.encoder, "forward_tokens"):
            if layers is not None and len(layers) > 0:
                out = rae.encoder.forward_tokens(enc_input, layers=list(layers))
            else:
                out = rae.encoder.forward_tokens(enc_input)
            distill_feat = out["patch_tokens"]
            cls_token = out["cls_token"]
            if isinstance(out, dict) and "patch_tokens_by_layer" in out:
                distill_feat_by_layer = out["patch_tokens_by_layer"]
            if isinstance(out, dict) and "cls_tokens_by_layer" in out:
                distill_cls_by_layer = out["cls_tokens_by_layer"]
        else:
            distill_feat = rae.encoder(enc_input)

    return {
        "distill_feat": distill_feat,
        "distill_feat_by_layer": distill_feat_by_layer,
        "distill_cls_by_layer": distill_cls_by_layer,
        "distill_cls": cls_token,
    }




def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
    *,
    save_stage2_compatible_merged: bool = False,
    ema_encoder: Optional[torch.nn.Module] = None,
) -> None:
    # If the encoder uses LoRA:
    # - Always save a base+adapter checkpoint for resuming stage1 training.
    # - Optionally (typically for ep-last) also save a stage2-compatible *merged* checkpoint
    #   where LoRA has been merged into the base encoder and no LoRA keys remain.
    has_lora = bool(getattr(getattr(model.module, "encoder", None), "lora_enabled", lambda: False)())

    base_path = path
    merged_path = None
    if has_lora and save_stage2_compatible_merged:
        base_path = path.replace(".pt", "-base.pt") if path.endswith(".pt") else (path + "-base")
        merged_path = path

    model_state = model.module.state_dict()
    if has_lora:
        model_state = _strip_lora_from_state_dict(model_state)
        ema_state = _strip_lora_from_state_dict(ema_model.state_dict())
    else:
        ema_state = ema_model.state_dict()

    state = {
        "step": step,
        "epoch": epoch,
        "model": model_state,
        "ema": ema_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    torch.save(state, base_path)

    if has_lora:
        adapter_root = _get_lora_adapter_dir(base_path)
        model_adapter_dir = os.path.join(adapter_root, "model")
        ema_adapter_dir = os.path.join(adapter_root, "ema")
        model.module.encoder.save_lora_adapter(model_adapter_dir)
        if hasattr(ema_model, "encoder") and hasattr(ema_model.encoder, "save_lora_adapter"):
            ema_model.encoder.save_lora_adapter(ema_adapter_dir)
        if ema_encoder is not None and hasattr(ema_encoder, "save_lora_adapter"):
            ema_teacher_adapter_dir = os.path.join(adapter_root, "ema_encoder")
            ema_encoder.save_lora_adapter(ema_teacher_adapter_dir)

    # Save stage2-compatible merged checkpoint (encoder+decoder) if requested.
    if merged_path is not None:
        # Merge LoRA into the in-memory models (safe here because we only call this for final export).
        if hasattr(model.module, "encoder") and hasattr(model.module.encoder, "merge_lora_and_unload_"):
            model.module.encoder.merge_lora_and_unload_()
        if hasattr(ema_model, "encoder") and hasattr(ema_model.encoder, "merge_lora_and_unload_"):
            ema_model.encoder.merge_lora_and_unload_()

        merged_state = {
            "step": step,
            "epoch": epoch,
            "model": model.module.state_dict(),
            "ema": ema_model.state_dict(),
            # Keep optimizer/disc fields for compatibility, though stage2 typically only uses 'ema'.
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "disc": disc.state_dict(),
            "disc_optimizer": disc_optimizer.state_dict(),
            "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
        }
        os.makedirs(os.path.dirname(merged_path), exist_ok=True)
        torch.save(merged_state, merged_path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    # When LoRA is enabled, main checkpoint contains base weights only.
    has_lora = bool(getattr(getattr(model.module, "encoder", None), "lora_enabled", lambda: False)())
    model.module.load_state_dict(checkpoint["model"], strict=not has_lora)
    ema_model.load_state_dict(checkpoint["ema"], strict=not has_lora)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])

    # Load LoRA adapter weights if present.
    if has_lora:
        adapter_root = _get_lora_adapter_dir(path)
        model_adapter_dir = os.path.join(adapter_root, "model")
        ema_adapter_dir = os.path.join(adapter_root, "ema")
        if not os.path.isdir(model_adapter_dir):
            raise FileNotFoundError(
                "LoRA is enabled but model adapter directory is missing for resume: "
                f"{model_adapter_dir}."
            )
        model.module.encoder.load_lora_adapter(model_adapter_dir)

        if not (hasattr(ema_model, "encoder") and hasattr(ema_model.encoder, "load_lora_adapter")):
            raise RuntimeError(
                "LoRA is enabled but EMA model does not support load_lora_adapter(). "
                "Cannot safely resume EMA LoRA weights."
            )
        if not os.path.isdir(ema_adapter_dir):
            raise FileNotFoundError(
                "LoRA is enabled but EMA adapter directory is missing for resume: "
                f"{ema_adapter_dir}."
            )
        ema_model.encoder.load_lora_adapter(ema_adapter_dir)
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def _build_random_patch_mask(
    B: int,
    N: int,
    device: torch.device,
    *,
    mask_ratio_min_max: tuple[float, float],
    mask_sample_probability: float,
) -> torch.Tensor:
    """Return boolean mask of shape (B, N)."""
    min_r, max_r = float(mask_ratio_min_max[0]), float(mask_ratio_min_max[1])
    min_r = max(0.0, min(min_r, 1.0))
    max_r = max(0.0, min(max_r, 1.0))
    if max_r <= 0.0 or mask_sample_probability <= 0.0:
        return torch.zeros((B, N), device=device, dtype=torch.bool)

    use_mask = torch.rand((B,), device=device) < float(mask_sample_probability)
    ratios = torch.empty((B,), device=device).uniform_(min_r, max_r)
    ratios = torch.where(use_mask, ratios, torch.zeros_like(ratios))
    n_mask = (ratios * N).round().to(torch.long)
    n_mask = torch.clamp(n_mask, min=0, max=N)

    rand = torch.rand((B, N), device=device)
    order = torch.argsort(rand, dim=1)
    masks = torch.zeros((B, N), device=device, dtype=torch.bool)
    for i in range(B):
        nm = int(n_mask[i].item())
        if nm > 0:
            masks[i, order[i, :nm]] = True
    return masks

def main():
    args = parse_args()
    #### Dist Init
    rank, world_size, device = setup_distributed()

    #### Config init
    full_cfg = OmegaConf.load(args.config)
    (rae_config, *_) = parse_configs(full_cfg)
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    disc_cfg = gan_cfg.get("disc", {})
    if not disc_cfg:
        raise ValueError("gan.disc configuration is required for stage-1 training.")
    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))

    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")
    batch_size = int(training_cfg.get("batch_size", 16))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
        batch_size = global_batch_size // world_size
    else:
        global_batch_size = batch_size * world_size
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 1250))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))

    encoder_training_section = full_cfg.get("encoder_training", None)
    encoder_training_cfg = (
        OmegaConf.to_container(encoder_training_section, resolve=True) if encoder_training_section is not None else {}
    )
    encoder_training_cfg = dict(encoder_training_cfg) if isinstance(encoder_training_cfg, dict) else {}
    encoder_train_enabled = bool(encoder_training_cfg.get("enabled", False))
    encoder_train_start_epoch = int(encoder_training_cfg.get("start_epoch", 0))
    encoder_lr_mult = float(encoder_training_cfg.get("lr_mult", 1.0))
    encoder_train_mode = str(encoder_training_cfg.get("mode", "full")).lower()
    if encoder_train_mode not in {"full", "lora"}:
        raise ValueError("encoder_training.mode must be one of: full, lora")
    ema_teacher_cfg = dict(encoder_training_cfg.get("ema_teacher", {}))
    ema_teacher_enabled = bool(ema_teacher_cfg.get("enabled", False))
    ema_teacher_decay = float(ema_teacher_cfg.get("decay", 0.9999))

    distill_section = full_cfg.get("distill", None)
    distill_cfg = OmegaConf.to_container(distill_section, resolve=True) if distill_section is not None else {}
    distill_cfg = dict(distill_cfg) if isinstance(distill_cfg, dict) else {}
    distill_enabled = bool(distill_cfg.get("enabled", False))
    distill_start_epoch = int(distill_cfg.get("start_epoch", 0))
    distill_weight = float(distill_cfg.get("weight", 1.0))
    distill_teacher_path = distill_cfg.get("teacher_path", "")
    distill_normalize = bool(distill_cfg.get("normalize", True))
    distill_loss_type = distill_cfg.get("loss_type", "mse")
    distill_layers_raw = distill_cfg.get("layers", None)
    distill_layers: list[int] | None
    if distill_layers_raw is None:
        distill_layers = None
    else:
        distill_layers = [int(x) for x in list(distill_layers_raw)]
        if len(distill_layers) == 0:
            distill_layers = None
        elif any(x <= 0 for x in distill_layers):
            raise ValueError("distill.layers must be a non-empty list of positive ints (scheme-B: 1..num_hidden_layers).")

    distill_layer_weights_raw = distill_cfg.get("layer_weights", None)
    distill_layer_weights: list[float] | None
    if distill_layer_weights_raw is None:
        distill_layer_weights = None
    else:
        distill_layer_weights = [float(x) for x in list(distill_layer_weights_raw)]
        if len(distill_layer_weights) == 0:
            distill_layer_weights = None
        if distill_layers is None:
            raise ValueError("distill.layer_weights is set but distill.layers is missing.")
        if distill_layer_weights is not None and len(distill_layer_weights) != len(distill_layers):
            raise ValueError(
                "distill.layer_weights must have the same length as distill.layers; "
                f"got {len(distill_layer_weights)} vs {len(distill_layers)}"
            )
    distill_fft_cfg = distill_cfg.get("fft_filter", {})
    distill_fft_cfg = dict(distill_fft_cfg) if isinstance(distill_fft_cfg, dict) else {}
    distill_fft_enabled = bool(distill_fft_cfg.get("enabled", False))
    distill_fft_cutoff_ratio = float(distill_fft_cfg.get("cutoff_ratio", 0.0))
    distill_fft_method = _parse_fft_filter_method(distill_fft_cfg, name="distill.fft_filter")
    distill_fft_student_enable = bool(distill_fft_cfg.get("student_enable", False))
    distill_fft_orig_weight = float(distill_fft_cfg.get("orig_weight", 0.5))
    distill_fft_weight = float(distill_fft_cfg.get("fft_weight", 0.5))
    if distill_fft_orig_weight < 0.0 or distill_fft_weight < 0.0:
        raise ValueError("distill.fft_filter.orig_weight and distill.fft_filter.fft_weight must be non-negative")

    struct_align_section = full_cfg.get("struct_align", None)
    struct_align_cfg = (
        OmegaConf.to_container(struct_align_section, resolve=True) if struct_align_section is not None else {}
    )
    struct_align_cfg = dict(struct_align_cfg) if isinstance(struct_align_cfg, dict) else {}
    struct_align_enabled = bool(struct_align_cfg.get("enabled", False))
    struct_align_start_epoch = int(struct_align_cfg.get("start_epoch", 0))
    struct_align_end_epoch_raw = struct_align_cfg.get("end_epoch", None)
    struct_align_end_epoch = None if struct_align_end_epoch_raw is None else int(struct_align_end_epoch_raw)
    if struct_align_end_epoch is not None and struct_align_end_epoch < struct_align_start_epoch:
        raise ValueError(
            "struct_align.end_epoch must be >= struct_align.start_epoch; "
            f"got end_epoch={struct_align_end_epoch}, start_epoch={struct_align_start_epoch}"
        )
    struct_align_weight = float(struct_align_cfg.get("weight", 1.0))
    struct_align_warmup_steps = int(struct_align_cfg.get("warmup_steps", 0))
    struct_align_warmup_schedule = str(struct_align_cfg.get("warmup_schedule", "none"))
    struct_align_teacher_path = struct_align_cfg.get("teacher_path", "")
    struct_align_normalize = bool(struct_align_cfg.get("normalize", True))
    struct_align_temperature = float(struct_align_cfg.get("temperature", 1.0))
    struct_align_loss_type = str(struct_align_cfg.get("loss_type", "kl")).lower()
    if struct_align_loss_type not in {"kl", "mse", "cosine"}:
        raise ValueError(
            f"Unsupported struct_align.loss_type '{struct_align_loss_type}'. "
            "Choose from ['kl', 'mse', 'cosine']."
        )
    struct_align_fft_cfg = struct_align_cfg.get("fft_filter", {})
    struct_align_fft_cfg = dict(struct_align_fft_cfg) if isinstance(struct_align_fft_cfg, dict) else {}
    struct_align_fft_enabled = bool(struct_align_fft_cfg.get("enabled", False))
    struct_align_fft_method = _parse_fft_filter_method(struct_align_fft_cfg, name="struct_align.fft_filter")
    struct_align_fft_student_enable = bool(struct_align_fft_cfg.get("student_enable", False))
    struct_align_fft_orig_weight = float(struct_align_fft_cfg.get("orig_weight", 0.5))
    struct_align_fft_weight = float(struct_align_fft_cfg.get("fft_weight", 0.5))
    if struct_align_fft_orig_weight < 0.0 or struct_align_fft_weight < 0.0:
        raise ValueError("struct_align.fft_filter.orig_weight and struct_align.fft_filter.fft_weight must be non-negative")
    struct_align_fft_cutoff_start, struct_align_fft_cutoff_end = _parse_ratio_spec(
        struct_align_fft_cfg.get("cutoff_ratio", 0.0),
        name="struct_align.fft_filter.cutoff_ratio",
    )
    # Allow FFT cutoff warmup to differ from struct_align weight warmup.
    struct_align_fft_warmup_steps = int(struct_align_fft_cfg.get("warmup_steps", struct_align_warmup_steps))
    struct_align_fft_warmup_schedule = str(struct_align_fft_cfg.get("warmup_schedule", struct_align_warmup_schedule))
    # Patch-affinity mode and multi-layer config (mirrors distill).
    struct_align_use_patch_affinity = bool(struct_align_cfg.get("use_patch_affinity", False))
    _sa_layers_raw = struct_align_cfg.get("layers", None)
    struct_align_layers: list[int] | None = (
        [int(x) for x in list(_sa_layers_raw)] if _sa_layers_raw is not None and len(_sa_layers_raw) > 0 else None
    )
    _sa_lw_raw = struct_align_cfg.get("layer_weights", None)
    struct_align_layer_weights: list[float] | None = (
        [float(x) for x in list(_sa_lw_raw)] if _sa_lw_raw is not None and len(_sa_lw_raw) > 0 else None
    )

    if struct_align_fft_enabled:
        if struct_align_fft_cutoff_start <= 0.0 or struct_align_fft_cutoff_end <= 0.0:
            raise ValueError(
                "struct_align.fft_filter.cutoff_ratio must be positive when enabled is True; "
                f"got start={struct_align_fft_cutoff_start}, end={struct_align_fft_cutoff_end}"
            )

    # ── struct_align_2 config ────────────────────────────────────────
    struct_align_2_section = full_cfg.get("struct_align_2", None)
    struct_align_2_cfg = (
        OmegaConf.to_container(struct_align_2_section, resolve=True) if struct_align_2_section is not None else {}
    )
    struct_align_2_cfg = dict(struct_align_2_cfg) if isinstance(struct_align_2_cfg, dict) else {}
    struct_align_2_enabled = bool(struct_align_2_cfg.get("enabled", False))
    struct_align_2_start_epoch = int(struct_align_2_cfg.get("start_epoch", 0))
    struct_align_2_end_epoch_raw = struct_align_2_cfg.get("end_epoch", None)
    struct_align_2_end_epoch = None if struct_align_2_end_epoch_raw is None else int(struct_align_2_end_epoch_raw)
    if struct_align_2_end_epoch is not None and struct_align_2_end_epoch < struct_align_2_start_epoch:
        raise ValueError(
            "struct_align_2.end_epoch must be >= struct_align_2.start_epoch; "
            f"got end_epoch={struct_align_2_end_epoch}, start_epoch={struct_align_2_start_epoch}"
        )
    struct_align_2_weight = float(struct_align_2_cfg.get("weight", 1.0))
    struct_align_2_warmup_steps = int(struct_align_2_cfg.get("warmup_steps", 0))
    struct_align_2_warmup_schedule = str(struct_align_2_cfg.get("warmup_schedule", "none"))
    struct_align_2_teacher_path = struct_align_2_cfg.get("teacher_path", "")
    struct_align_2_normalize = bool(struct_align_2_cfg.get("normalize", True))
    struct_align_2_temperature = float(struct_align_2_cfg.get("temperature", 1.0))
    struct_align_2_loss_type = str(struct_align_2_cfg.get("loss_type", "kl")).lower()
    if struct_align_2_loss_type not in {"kl", "mse", "cosine"}:
        raise ValueError(
            f"Unsupported struct_align_2.loss_type '{struct_align_2_loss_type}'. "
            "Choose from ['kl', 'mse', 'cosine']."
        )
    struct_align_2_fft_cfg = struct_align_2_cfg.get("fft_filter", {})
    struct_align_2_fft_cfg = dict(struct_align_2_fft_cfg) if isinstance(struct_align_2_fft_cfg, dict) else {}
    struct_align_2_fft_enabled = bool(struct_align_2_fft_cfg.get("enabled", False))
    struct_align_2_fft_method = _parse_fft_filter_method(struct_align_2_fft_cfg, name="struct_align_2.fft_filter")
    struct_align_2_fft_student_enable = bool(struct_align_2_fft_cfg.get("student_enable", False))
    struct_align_2_fft_orig_weight = float(struct_align_2_fft_cfg.get("orig_weight", 0.5))
    struct_align_2_fft_weight = float(struct_align_2_fft_cfg.get("fft_weight", 0.5))
    if struct_align_2_fft_orig_weight < 0.0 or struct_align_2_fft_weight < 0.0:
        raise ValueError("struct_align_2.fft_filter.orig_weight and struct_align_2.fft_filter.fft_weight must be non-negative")
    struct_align_2_fft_cutoff_start, struct_align_2_fft_cutoff_end = _parse_ratio_spec(
        struct_align_2_fft_cfg.get("cutoff_ratio", 0.0),
        name="struct_align_2.fft_filter.cutoff_ratio",
    )
    struct_align_2_fft_warmup_steps = int(struct_align_2_fft_cfg.get("warmup_steps", struct_align_2_warmup_steps))
    struct_align_2_fft_warmup_schedule = str(struct_align_2_fft_cfg.get("warmup_schedule", struct_align_2_warmup_schedule))
    struct_align_2_use_patch_affinity = bool(struct_align_2_cfg.get("use_patch_affinity", False))
    _sa2_layers_raw = struct_align_2_cfg.get("layers", None)
    struct_align_2_layers: list[int] | None = (
        [int(x) for x in list(_sa2_layers_raw)] if _sa2_layers_raw is not None and len(_sa2_layers_raw) > 0 else None
    )
    _sa2_lw_raw = struct_align_2_cfg.get("layer_weights", None)
    struct_align_2_layer_weights: list[float] | None = (
        [float(x) for x in list(_sa2_lw_raw)] if _sa2_lw_raw is not None and len(_sa2_lw_raw) > 0 else None
    )

    if struct_align_2_fft_enabled:
        if struct_align_2_fft_cutoff_start <= 0.0 or struct_align_2_fft_cutoff_end <= 0.0:
            raise ValueError(
                "struct_align_2.fft_filter.cutoff_ratio must be positive when enabled is True; "
                f"got start={struct_align_2_fft_cutoff_start}, end={struct_align_2_fft_cutoff_end}"
            )

    # ── spatial_grad config ──────────────────────────────────────────
    spatial_grad_section = full_cfg.get("spatial_grad", None)
    spatial_grad_cfg = (
        OmegaConf.to_container(spatial_grad_section, resolve=True) if spatial_grad_section is not None else {}
    )
    spatial_grad_cfg = dict(spatial_grad_cfg) if isinstance(spatial_grad_cfg, dict) else {}
    spatial_grad_enabled = bool(spatial_grad_cfg.get("enabled", False))
    spatial_grad_start_epoch = int(spatial_grad_cfg.get("start_epoch", 0))
    spatial_grad_end_epoch_raw = spatial_grad_cfg.get("end_epoch", None)
    spatial_grad_end_epoch = None if spatial_grad_end_epoch_raw is None else int(spatial_grad_end_epoch_raw)
    if spatial_grad_end_epoch is not None and spatial_grad_end_epoch < spatial_grad_start_epoch:
        raise ValueError(
            "spatial_grad.end_epoch must be >= spatial_grad.start_epoch; "
            f"got end_epoch={spatial_grad_end_epoch}, start_epoch={spatial_grad_start_epoch}"
        )
    spatial_grad_weight = float(spatial_grad_cfg.get("weight", 1.0))
    spatial_grad_warmup_steps = int(spatial_grad_cfg.get("warmup_steps", 0))
    spatial_grad_warmup_schedule = str(spatial_grad_cfg.get("warmup_schedule", "none"))
    spatial_grad_teacher_path = spatial_grad_cfg.get("teacher_path", "")
    spatial_grad_normalize = bool(spatial_grad_cfg.get("normalize", True))
    spatial_grad_loss_type = str(spatial_grad_cfg.get("loss_type", "cosine")).lower()
    if spatial_grad_loss_type not in {"cosine", "mse"}:
        raise ValueError(
            f"Unsupported spatial_grad.loss_type '{spatial_grad_loss_type}'. "
            "Choose from ['cosine', 'mse']."
        )
    _sg_layers_raw = spatial_grad_cfg.get("layers", None)
    spatial_grad_layers: list[int] | None = (
        [int(x) for x in list(_sg_layers_raw)] if _sg_layers_raw is not None and len(_sg_layers_raw) > 0 else None
    )
    _sg_lw_raw = spatial_grad_cfg.get("layer_weights", None)
    spatial_grad_layer_weights: list[float] | None = (
        [float(x) for x in list(_sg_lw_raw)] if _sg_lw_raw is not None and len(_sg_lw_raw) > 0 else None
    )

    # ── edge_align config ────────────────────────────────────────────
    edge_align_section = full_cfg.get("edge_align", None)
    edge_align_cfg = (
        OmegaConf.to_container(edge_align_section, resolve=True) if edge_align_section is not None else {}
    )
    edge_align_cfg = dict(edge_align_cfg) if isinstance(edge_align_cfg, dict) else {}
    edge_align_enabled = bool(edge_align_cfg.get("enabled", False))
    edge_align_start_epoch = int(edge_align_cfg.get("start_epoch", 0))
    edge_align_end_epoch_raw = edge_align_cfg.get("end_epoch", None)
    edge_align_end_epoch = None if edge_align_end_epoch_raw is None else int(edge_align_end_epoch_raw)
    if edge_align_end_epoch is not None and edge_align_end_epoch < edge_align_start_epoch:
        raise ValueError(
            "edge_align.end_epoch must be >= edge_align.start_epoch; "
            f"got end_epoch={edge_align_end_epoch}, start_epoch={edge_align_start_epoch}"
        )
    edge_align_weight = float(edge_align_cfg.get("weight", 1.0))
    edge_align_warmup_steps = int(edge_align_cfg.get("warmup_steps", 0))
    edge_align_warmup_schedule = str(edge_align_cfg.get("warmup_schedule", "none"))
    edge_align_teacher_path = edge_align_cfg.get("teacher_path", "")
    edge_align_normalize = bool(edge_align_cfg.get("normalize", True))
    edge_align_loss_type = str(edge_align_cfg.get("loss_type", "cosine")).lower()
    if edge_align_loss_type not in {"cosine", "mse"}:
        raise ValueError(
            f"Unsupported edge_align.loss_type '{edge_align_loss_type}'. "
            "Choose from ['cosine', 'mse']."
        )
    _ea_layers_raw = edge_align_cfg.get("layers", None)
    edge_align_layers: list[int] | None = (
        [int(x) for x in list(_ea_layers_raw)] if _ea_layers_raw is not None and len(_ea_layers_raw) > 0 else None
    )
    _ea_lw_raw = edge_align_cfg.get("layer_weights", None)
    edge_align_layer_weights: list[float] | None = (
        [float(x) for x in list(_ea_lw_raw)] if _ea_lw_raw is not None and len(_ea_lw_raw) > 0 else None
    )

    # ── ghpf_align config ────────────────────────────────────────────
    ghpf_align_section = full_cfg.get("ghpf_align", None)
    ghpf_align_cfg = (
        OmegaConf.to_container(ghpf_align_section, resolve=True) if ghpf_align_section is not None else {}
    )
    ghpf_align_cfg = dict(ghpf_align_cfg) if isinstance(ghpf_align_cfg, dict) else {}
    ghpf_align_enabled = bool(ghpf_align_cfg.get("enabled", False))
    ghpf_align_start_epoch = int(ghpf_align_cfg.get("start_epoch", 0))
    ghpf_align_end_epoch_raw = ghpf_align_cfg.get("end_epoch", None)
    ghpf_align_end_epoch = None if ghpf_align_end_epoch_raw is None else int(ghpf_align_end_epoch_raw)
    if ghpf_align_end_epoch is not None and ghpf_align_end_epoch < ghpf_align_start_epoch:
        raise ValueError(
            "ghpf_align.end_epoch must be >= ghpf_align.start_epoch; "
            f"got end_epoch={ghpf_align_end_epoch}, start_epoch={ghpf_align_start_epoch}"
        )
    ghpf_align_weight = float(ghpf_align_cfg.get("weight", 1.0))
    ghpf_align_warmup_steps = int(ghpf_align_cfg.get("warmup_steps", 0))
    ghpf_align_warmup_schedule = str(ghpf_align_cfg.get("warmup_schedule", "none"))
    ghpf_align_teacher_path = ghpf_align_cfg.get("teacher_path", "")
    ghpf_align_normalize = bool(ghpf_align_cfg.get("normalize", True))
    ghpf_align_loss_type = str(ghpf_align_cfg.get("loss_type", "cosine")).lower()
    ghpf_align_class_token_cosine_only = bool(ghpf_align_cfg.get("class_token_cosine_only", False))
    if ghpf_align_loss_type not in {"cosine", "mse"}:
        raise ValueError(
            f"Unsupported ghpf_align.loss_type '{ghpf_align_loss_type}'. "
            "Choose from ['cosine', 'mse']."
        )
    if ghpf_align_class_token_cosine_only and ghpf_align_loss_type != "cosine":
        raise ValueError("ghpf_align.class_token_cosine_only=True requires ghpf_align.loss_type='cosine'.")
    ghpf_align_cutoff_ratio = float(ghpf_align_cfg.get("cutoff_ratio", 0.05))
    if ghpf_align_enabled and ghpf_align_cutoff_ratio <= 0.0:
        raise ValueError("ghpf_align.cutoff_ratio must be positive when ghpf_align is enabled.")
    ghpf_align_use_relaxed_cosine = bool(ghpf_align_cfg.get("use_relaxed_cosine", False))
    ghpf_align_relaxed_margin = float(ghpf_align_cfg.get("relaxed_margin", 0.2))
    ghpf_align_normalize_hp = bool(ghpf_align_cfg.get("normalize_hp", False))
    ghpf_align_use_energy_mask = bool(ghpf_align_cfg.get("use_energy_mask", False))
    ghpf_align_energy_mask_quantile = float(ghpf_align_cfg.get("energy_mask_quantile", 0.3))
    ghpf_align_use_patch_affinity = bool(ghpf_align_cfg.get("use_patch_affinity", False))
    ghpf_align_temperature = float(ghpf_align_cfg.get("temperature", 1.0))
    _ga_layers_raw = ghpf_align_cfg.get("layers", None)
    ghpf_align_layers: list[int] | None = (
        [int(x) for x in list(_ga_layers_raw)] if _ga_layers_raw is not None and len(_ga_layers_raw) > 0 else None
    )
    _ga_lw_raw = ghpf_align_cfg.get("layer_weights", None)
    ghpf_align_layer_weights: list[float] | None = (
        [float(x) for x in list(_ga_lw_raw)] if _ga_lw_raw is not None and len(_ga_lw_raw) > 0 else None
    )

    # ── fft_align config ─────────────────────────────────────────────
    fft_align_section = full_cfg.get("fft_align", None)
    fft_align_cfg = (
        OmegaConf.to_container(fft_align_section, resolve=True) if fft_align_section is not None else {}
    )
    fft_align_cfg = dict(fft_align_cfg) if isinstance(fft_align_cfg, dict) else {}
    fft_align_enabled = bool(fft_align_cfg.get("enabled", False))
    fft_align_start_epoch = int(fft_align_cfg.get("start_epoch", 0))
    fft_align_end_epoch_raw = fft_align_cfg.get("end_epoch", None)
    fft_align_end_epoch = None if fft_align_end_epoch_raw is None else int(fft_align_end_epoch_raw)
    if fft_align_end_epoch is not None and fft_align_end_epoch < fft_align_start_epoch:
        raise ValueError(
            "fft_align.end_epoch must be >= fft_align.start_epoch; "
            f"got end_epoch={fft_align_end_epoch}, start_epoch={fft_align_start_epoch}"
        )
    fft_align_weight = float(fft_align_cfg.get("weight", 1.0))
    fft_align_warmup_steps = int(fft_align_cfg.get("warmup_steps", 0))
    fft_align_warmup_schedule = str(fft_align_cfg.get("warmup_schedule", "none"))
    fft_align_teacher_path = fft_align_cfg.get("teacher_path", "")
    fft_align_normalize = bool(fft_align_cfg.get("normalize", True))
    fft_align_loss_type = str(fft_align_cfg.get("loss_type", "cosine")).lower()
    fft_align_class_token_cosine_only = bool(fft_align_cfg.get("class_token_cosine_only", False))
    if fft_align_loss_type not in {"cosine", "mse"}:
        raise ValueError(
            f"Unsupported fft_align.loss_type '{fft_align_loss_type}'. "
            "Choose from ['cosine', 'mse']."
        )
    if fft_align_class_token_cosine_only and fft_align_loss_type != "cosine":
        raise ValueError("fft_align.class_token_cosine_only=True requires fft_align.loss_type='cosine'.")
    fft_align_cutoff_ratio = float(fft_align_cfg.get("cutoff_ratio", 0.05))
    if fft_align_enabled and fft_align_cutoff_ratio <= 0.0:
        raise ValueError("fft_align.cutoff_ratio must be positive when fft_align is enabled.")
    fft_align_use_relaxed_cosine = bool(fft_align_cfg.get("use_relaxed_cosine", False))
    fft_align_relaxed_margin = float(fft_align_cfg.get("relaxed_margin", 0.2))
    fft_align_normalize_hp = bool(fft_align_cfg.get("normalize_hp", False))
    fft_align_use_energy_mask = bool(fft_align_cfg.get("use_energy_mask", False))
    fft_align_energy_mask_quantile = float(fft_align_cfg.get("energy_mask_quantile", 0.3))
    fft_align_use_patch_affinity = bool(fft_align_cfg.get("use_patch_affinity", False))
    fft_align_temperature = float(fft_align_cfg.get("temperature", 1.0))
    _fa_layers_raw = fft_align_cfg.get("layers", None)
    fft_align_layers: list[int] | None = (
        [int(x) for x in list(_fa_layers_raw)] if _fa_layers_raw is not None and len(_fa_layers_raw) > 0 else None
    )
    _fa_lw_raw = fft_align_cfg.get("layer_weights", None)
    fft_align_layer_weights: list[float] | None = (
        [float(x) for x in list(_fa_lw_raw)] if _fa_lw_raw is not None and len(_fa_lw_raw) > 0 else None
    )

    # ── ghpf_recon config ───────────────────────────────────────────
    ghpf_recon_section = full_cfg.get("ghpf_recon", None)
    ghpf_recon_cfg = (
        OmegaConf.to_container(ghpf_recon_section, resolve=True) if ghpf_recon_section is not None else {}
    )
    ghpf_recon_cfg = dict(ghpf_recon_cfg) if isinstance(ghpf_recon_cfg, dict) else {}
    ghpf_recon_enabled = bool(ghpf_recon_cfg.get("enabled", False))
    ghpf_recon_start_epoch = int(ghpf_recon_cfg.get("start_epoch", 0))
    ghpf_recon_end_epoch_raw = ghpf_recon_cfg.get("end_epoch", None)
    ghpf_recon_end_epoch = None if ghpf_recon_end_epoch_raw is None else int(ghpf_recon_end_epoch_raw)
    if ghpf_recon_end_epoch is not None and ghpf_recon_end_epoch < ghpf_recon_start_epoch:
        raise ValueError(
            "ghpf_recon.end_epoch must be >= ghpf_recon.start_epoch; "
            f"got end_epoch={ghpf_recon_end_epoch}, start_epoch={ghpf_recon_start_epoch}"
        )
    ghpf_recon_weight = float(ghpf_recon_cfg.get("weight", 1.0))
    ghpf_recon_warmup_steps = int(ghpf_recon_cfg.get("warmup_steps", 0))
    ghpf_recon_warmup_schedule = str(ghpf_recon_cfg.get("warmup_schedule", "none"))
    ghpf_recon_loss_type = str(ghpf_recon_cfg.get("loss_type", "mse")).lower()
    if ghpf_recon_loss_type not in {"mse", "cosine"}:
        raise ValueError(
            f"Unsupported ghpf_recon.loss_type '{ghpf_recon_loss_type}'. "
            "Choose from ['mse', 'cosine']."
        )
    ghpf_recon_cutoff_ratio = float(ghpf_recon_cfg.get("cutoff_ratio", 0.05))
    if ghpf_recon_enabled and ghpf_recon_cutoff_ratio <= 0.0:
        raise ValueError("ghpf_recon.cutoff_ratio must be positive when ghpf_recon is enabled.")
    ghpf_recon_normalize_map = str(ghpf_recon_cfg.get("normalize_map", "none")).lower()
    if ghpf_recon_normalize_map not in {"none", "l2", "channel_std"}:
        raise ValueError(
            f"Unsupported ghpf_recon.normalize_map '{ghpf_recon_normalize_map}'. "
            "Choose from ['none', 'l2', 'channel_std']."
        )
    ghpf_recon_eps = float(ghpf_recon_cfg.get("eps", 1e-6))
    if ghpf_recon_eps <= 0.0:
        raise ValueError("ghpf_recon.eps must be positive.")

    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    # update args as a dict to full_cfg
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir


    #### Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    # noise_tau_start_epoch: epoch from which noise injection is enabled (default 0 = always on)
    _rae_params = rae_config.get("params", {}) if rae_config is not None else {}
    _rae_params = dict(_rae_params) if isinstance(_rae_params, dict) else {}
    noise_tau_start_epoch = int(_rae_params.get("noise_tau_start_epoch", 0))
    _noise_tau_value = float(_rae_params.get("noise_tau", rae.noise_tau))
    if args.compile:
        rae.encode = torch.compile(rae.encode)
        rae.forward = torch.compile(rae.forward)

    # Multi-layer distillation (scheme-B) uses encoder hidden states for the requested transformer blocks.
    # Merge struct_align and spatial_grad layers so encode_with_distill fetches all needed intermediate states in one pass.
    _combined_layers: list[int] | None = None
    if distill_enabled and distill_layers is not None:
        _combined_layers = list(distill_layers)
    if struct_align_enabled and struct_align_layers is not None:
        _combined_layers = sorted(set((_combined_layers or []) + struct_align_layers))
    if struct_align_2_enabled and struct_align_2_layers is not None:
        _combined_layers = sorted(set((_combined_layers or []) + struct_align_2_layers))
    if spatial_grad_enabled and spatial_grad_layers is not None:
        _combined_layers = sorted(set((_combined_layers or []) + spatial_grad_layers))
    if edge_align_enabled and edge_align_layers is not None:
        _combined_layers = sorted(set((_combined_layers or []) + edge_align_layers))
    if ghpf_align_enabled and ghpf_align_layers is not None:
        _combined_layers = sorted(set((_combined_layers or []) + ghpf_align_layers))
    if fft_align_enabled and fft_align_layers is not None:
        _combined_layers = sorted(set((_combined_layers or []) + fft_align_layers))
    if _combined_layers is not None:
        setattr(rae, "distill_layers", _combined_layers)

    rae.encoder.eval()
    rae.decoder.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)

    # Teacher EMA encoder: same architecture as student encoder, LoRA weights driven by EMA.
    if ema_teacher_enabled:
        if not encoder_train_enabled or encoder_train_mode != "lora":
            raise ValueError(
                "encoder_training.ema_teacher.enabled=true requires "
                "encoder_training.enabled=true and encoder_training.mode='lora'."
            )
        ema_encoder = deepcopy(rae.encoder).to(device).eval()
        ema_encoder.requires_grad_(False)
    else:
        ema_encoder = None

    # Validate LoRA training mode assumptions.
    if encoder_train_enabled and encoder_train_mode == "lora":
        if not hasattr(rae.encoder, "lora_enabled") or not rae.encoder.lora_enabled():
            raise ValueError(
                "encoder_training.mode='lora' is set, but the encoder does not have LoRA enabled. "
                "Set stage_1.params.encoder_params.lora.enabled=true (and related LoRA params)."
            )
        if not distill_enabled:
            logger.warning(
                "encoder_training.mode='lora' is enabled while distill.enabled=false. "
                "This is allowed (e.g. for struct_align-only), but distill loss will be disabled."
            )

    # Default behavior: only train decoder. Encoder training is gated by epoch.
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)
    ddp_model = DDP(rae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)  # type: ignore[arg-type]
    rae = ddp_model.module
    decoder = ddp_model.module.decoder
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)  # type: ignore[arg-type]
    discriminator = ddp_disc.module
    disc_scheduler: LambdaLR | None = None
    disc_sched_msg: Optional[str] = None

    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

    lpips = LPIPS().to(device)
    lpips.eval()

    #### Opt, Schedl init
    if encoder_train_enabled:
        if encoder_train_mode == "lora":
            # Collect LoRA params directly — at this point encoder is frozen (requires_grad=False),
            # so filtering by requires_grad would yield an empty list. Instead grab lora_A/lora_B
            # by name; set_lora_trainable(True) will enable their gradients at start_epoch.
            enc_train_params = [
                p for n, p in rae.encoder.named_parameters()
                if ".lora_" in n.lower() or "lora_a" in n.lower() or "lora_b" in n.lower()
            ]
            if not enc_train_params:
                raise ValueError(
                    "encoder_training.mode='lora' but no LoRA parameters found in encoder. "
                    "Ensure stage_1.params.encoder_params.lora.enabled=true."
                )
        else:
            enc_train_params = list(rae.encoder.parameters())
        aux_params = (
            list(rae.aux_proj.parameters()) + list(rae.aux_norm.parameters())
            if hasattr(rae, "aux_proj") else []
        )
        opt_param_groups = [
            {"params": list(decoder.parameters()) + aux_params, "lr_mult": 1.0},
            {"params": enc_train_params, "lr_mult": encoder_lr_mult},
        ]
        optimizer, optim_msg = build_optimizer(opt_param_groups, training_cfg)
    else:
        aux_params = (
            list(rae.aux_proj.parameters()) + list(rae.aux_norm.parameters())
            if hasattr(rae, "aux_proj") else []
        )
        optimizer, optim_msg = build_optimizer(
            list(decoder.parameters()) + aux_params, training_cfg
        )
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)

    distill_loss_module = None
    if distill_enabled:
        if not distill_teacher_path:
            # fallback to student encoder path
            stage1_params = rae_config.get("params", {}) if rae_config is not None else {}
            enc_params = stage1_params.get("encoder_params", {}) if isinstance(stage1_params, dict) else {}
            distill_teacher_path = enc_params.get("dinov3_path", stage1_params.get("encoder_config_path", ""))
        distill_loss_module = DinoV3DistillLoss(
            distill_teacher_path,
            normalize=distill_normalize,
            loss_type=distill_loss_type,
            layers=distill_layers,
            layer_weights=distill_layer_weights,
            fft_filter_enabled=distill_fft_enabled,
            fft_filter_cutoff_ratio=distill_fft_cutoff_ratio,
            fft_filter_method=distill_fft_method,
        ).to(device)
        if ema_teacher_enabled and ema_encoder is not None:
            # Keep image normalization buffers from the configured teacher path,
            # but route teacher token extraction through EMA encoder.
            distill_loss_module.teacher = ema_encoder
            distill_loss_module.teacher.eval()
            distill_loss_module.teacher.requires_grad_(False)

    struct_align_loss_module = None
    if struct_align_enabled:
        if not struct_align_teacher_path:
            stage1_params = rae_config.get("params", {}) if rae_config is not None else {}
            enc_params = stage1_params.get("encoder_params", {}) if isinstance(stage1_params, dict) else {}
            struct_align_teacher_path = enc_params.get("dinov3_path", stage1_params.get("encoder_config_path", ""))
        struct_align_loss_module = StructuralAttentionAlignmentLoss(
            struct_align_teacher_path,
            normalize=struct_align_normalize,
            temperature=struct_align_temperature,
            loss_type=struct_align_loss_type,
            use_patch_affinity=struct_align_use_patch_affinity,
            layers=struct_align_layers,
            layer_weights=struct_align_layer_weights,
            fft_filter_enabled=struct_align_fft_enabled,
            fft_filter_cutoff_ratio=struct_align_fft_cutoff_end,
            fft_filter_method=struct_align_fft_method,
        ).to(device)
        if ema_teacher_enabled and ema_encoder is not None:
            struct_align_loss_module.teacher = ema_encoder
            struct_align_loss_module.teacher.eval()
            struct_align_loss_module.teacher.requires_grad_(False)

    struct_align_2_loss_module = None
    if struct_align_2_enabled:
        if not struct_align_2_teacher_path:
            stage1_params = rae_config.get("params", {}) if rae_config is not None else {}
            enc_params = stage1_params.get("encoder_params", {}) if isinstance(stage1_params, dict) else {}
            struct_align_2_teacher_path = enc_params.get("dinov3_path", stage1_params.get("encoder_config_path", ""))
        struct_align_2_loss_module = StructuralAttentionAlignmentLoss(
            struct_align_2_teacher_path,
            normalize=struct_align_2_normalize,
            temperature=struct_align_2_temperature,
            loss_type=struct_align_2_loss_type,
            use_patch_affinity=struct_align_2_use_patch_affinity,
            layers=struct_align_2_layers,
            layer_weights=struct_align_2_layer_weights,
            fft_filter_enabled=struct_align_2_fft_enabled,
            fft_filter_cutoff_ratio=struct_align_2_fft_cutoff_end,
            fft_filter_method=struct_align_2_fft_method,
        ).to(device)
        if ema_teacher_enabled and ema_encoder is not None:
            struct_align_2_loss_module.teacher = ema_encoder
            struct_align_2_loss_module.teacher.eval()
            struct_align_2_loss_module.teacher.requires_grad_(False)

    spatial_grad_loss_module = None
    if spatial_grad_enabled:
        if not spatial_grad_teacher_path:
            stage1_params = rae_config.get("params", {}) if rae_config is not None else {}
            enc_params = stage1_params.get("encoder_params", {}) if isinstance(stage1_params, dict) else {}
            spatial_grad_teacher_path = enc_params.get("dinov3_path", stage1_params.get("encoder_config_path", ""))
        spatial_grad_loss_module = SpatialGradientAlignmentLoss(
            spatial_grad_teacher_path,
            normalize=spatial_grad_normalize,
            loss_type=spatial_grad_loss_type,
            layers=spatial_grad_layers,
            layer_weights=spatial_grad_layer_weights,
        ).to(device)

    edge_align_loss_module = None
    if edge_align_enabled:
        if not edge_align_teacher_path:
            stage1_params = rae_config.get("params", {}) if rae_config is not None else {}
            enc_params = stage1_params.get("encoder_params", {}) if isinstance(stage1_params, dict) else {}
            edge_align_teacher_path = enc_params.get("dinov3_path", stage1_params.get("encoder_config_path", ""))
        edge_align_loss_module = EdgeAlignmentLoss(
            edge_align_teacher_path,
            normalize=edge_align_normalize,
            loss_type=edge_align_loss_type,
            layers=edge_align_layers,
            layer_weights=edge_align_layer_weights,
        ).to(device)

    ghpf_align_loss_module = None
    if ghpf_align_enabled:
        if not ghpf_align_teacher_path:
            stage1_params = rae_config.get("params", {}) if rae_config is not None else {}
            enc_params = stage1_params.get("encoder_params", {}) if isinstance(stage1_params, dict) else {}
            ghpf_align_teacher_path = enc_params.get("dinov3_path", stage1_params.get("encoder_config_path", ""))
        ghpf_align_loss_module = GHPFAlignmentLoss(
            ghpf_align_teacher_path,
            normalize=ghpf_align_normalize,
            loss_type=ghpf_align_loss_type,
            class_token_cosine_only=ghpf_align_class_token_cosine_only,
            cutoff_ratio=ghpf_align_cutoff_ratio,
            layers=ghpf_align_layers,
            layer_weights=ghpf_align_layer_weights,
            use_relaxed_cosine=ghpf_align_use_relaxed_cosine,
            relaxed_margin=ghpf_align_relaxed_margin,
            normalize_hp=ghpf_align_normalize_hp,
            use_energy_mask=ghpf_align_use_energy_mask,
            energy_mask_quantile=ghpf_align_energy_mask_quantile,
            use_patch_affinity=ghpf_align_use_patch_affinity,
            temperature=ghpf_align_temperature,
        ).to(device)
        if ema_teacher_enabled and ema_encoder is not None:
            ghpf_align_loss_module.teacher = ema_encoder
            ghpf_align_loss_module.teacher.eval()
            ghpf_align_loss_module.teacher.requires_grad_(False)

    fft_align_loss_module = None
    if fft_align_enabled:
        if not fft_align_teacher_path:
            stage1_params = rae_config.get("params", {}) if rae_config is not None else {}
            enc_params = stage1_params.get("encoder_params", {}) if isinstance(stage1_params, dict) else {}
            fft_align_teacher_path = enc_params.get("dinov3_path", stage1_params.get("encoder_config_path", ""))
        fft_align_loss_module = FFTAlignmentLoss(
            fft_align_teacher_path,
            normalize=fft_align_normalize,
            loss_type=fft_align_loss_type,
            class_token_cosine_only=fft_align_class_token_cosine_only,
            cutoff_ratio=fft_align_cutoff_ratio,
            layers=fft_align_layers,
            layer_weights=fft_align_layer_weights,
            use_relaxed_cosine=fft_align_use_relaxed_cosine,
            relaxed_margin=fft_align_relaxed_margin,
            normalize_hp=fft_align_normalize_hp,
            use_energy_mask=fft_align_use_energy_mask,
            energy_mask_quantile=fft_align_energy_mask_quantile,
            use_patch_affinity=fft_align_use_patch_affinity,
            temperature=fft_align_temperature,
        ).to(device)
        if ema_teacher_enabled and ema_encoder is not None:
            fft_align_loss_module.teacher = ema_encoder
            fft_align_loss_module.teacher.eval()
            fft_align_loss_module.teacher.requires_grad_(False)

    ghpf_recon_loss_module = None
    if ghpf_recon_enabled:
        ghpf_recon_loss_module = GHPFReconLoss(
            loss_type=ghpf_recon_loss_type,
            cutoff_ratio=ghpf_recon_cutoff_ratio,
            normalize_map=ghpf_recon_normalize_map,
            eps=ghpf_recon_eps,
        ).to(device)

    # ── aux_skip config ──────────────────────────────────────────────
    aux_skip_section = full_cfg.get("aux_skip", None)
    aux_skip_cfg = (
        OmegaConf.to_container(aux_skip_section, resolve=True) if aux_skip_section is not None else {}
    )
    aux_skip_cfg = dict(aux_skip_cfg) if isinstance(aux_skip_cfg, dict) else {}
    aux_skip_enabled = bool(aux_skip_cfg.get("enabled", False))
    aux_skip_start_epoch = int(aux_skip_cfg.get("start_epoch", 0))

    #### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)


    #### Data init
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    stage1_transform = transforms.Compose(
        [
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )
    loader, sampler = prepare_dataloader(
        args.data_path, batch_size, num_workers, rank, world_size, transform=stage1_transform
    )
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")

    # Schedl init after knowing dataset length
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)

    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"Stage-1 RAE trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Discriminator architecture:\n{discriminator}")
        num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logger.info(f"Discriminator trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Using {disc_loss_type} discriminator loss and {gen_loss_type} generator loss.")
        logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}, GAN weight: {disc_weight:.6f}")
        logger.info(f"GAN training starts at epoch {gan_start_epoch}, discriminator updates start at epoch {disc_update_epoch}, LPIPS loss starts at epoch {lpips_start_epoch}.")
        if encoder_train_enabled:
            logger.info(
                f"Encoder training enabled: start_epoch={encoder_train_start_epoch}, lr_mult={encoder_lr_mult}, mode={encoder_train_mode}"
            )
            if ema_teacher_enabled:
                logger.info(
                    f"EMA teacher enabled: decay={ema_teacher_decay} (student LoRA weights -> EMA teacher encoder)"
                )
            else:
                logger.info("EMA teacher disabled.")
        else:
            logger.info("Encoder training disabled (default: decoder-only training).")
        if distill_enabled:
            logger.info(
                f"Distill enabled: start_epoch={distill_start_epoch}, weight={distill_weight}, teacher_path={distill_teacher_path}, loss_type={distill_loss_type}, fft_filter_enabled={distill_fft_enabled}, fft_filter_method={distill_fft_method}, fft_filter_cutoff_ratio={distill_fft_cutoff_ratio}, fft_filter_student_enable={distill_fft_student_enable}, fft_filter_orig_weight={distill_fft_orig_weight}, fft_filter_fft_weight={distill_fft_weight}"
            )
            if ema_teacher_enabled:
                logger.info("Distill teacher source: EMA encoder (auto-switched by encoder_training.ema_teacher).")
        else:
            logger.info("Distill disabled.")
        if struct_align_enabled:
            logger.info(
                "Struct-align enabled: "
                f"start_epoch={struct_align_start_epoch}, weight={struct_align_weight}, "
                f"end_epoch={struct_align_end_epoch}, loss_type={struct_align_loss_type}, "
                f"use_patch_affinity={struct_align_use_patch_affinity}, "
                f"layers={struct_align_layers}, layer_weights={struct_align_layer_weights}, "
                f"warmup_steps={struct_align_warmup_steps}, warmup_schedule={struct_align_warmup_schedule}, "
                f"teacher_path={struct_align_teacher_path}, temperature={struct_align_temperature}, "
                f"fft_filter_enabled={struct_align_fft_enabled}, "
                f"fft_filter_method={struct_align_fft_method}, "
                f"fft_filter_student_enable={struct_align_fft_student_enable}, "
                f"fft_filter_orig_weight={struct_align_fft_orig_weight}, "
                f"fft_filter_fft_weight={struct_align_fft_weight}, "
                f"fft_filter_cutoff_ratio=[{struct_align_fft_cutoff_start}, {struct_align_fft_cutoff_end}], "
                f"fft_filter_warmup_steps={struct_align_fft_warmup_steps}, "
                f"fft_filter_warmup_schedule={struct_align_fft_warmup_schedule}"
            )
            if ema_teacher_enabled:
                logger.info("Struct-align teacher source: EMA encoder (auto-switched by encoder_training.ema_teacher).")
        else:
            logger.info("Struct-align disabled.")
        if struct_align_2_enabled:
            logger.info(
                "Struct-align-2 enabled: "
                f"start_epoch={struct_align_2_start_epoch}, weight={struct_align_2_weight}, "
                f"end_epoch={struct_align_2_end_epoch}, loss_type={struct_align_2_loss_type}, "
                f"use_patch_affinity={struct_align_2_use_patch_affinity}, "
                f"layers={struct_align_2_layers}, layer_weights={struct_align_2_layer_weights}, "
                f"warmup_steps={struct_align_2_warmup_steps}, warmup_schedule={struct_align_2_warmup_schedule}, "
                f"teacher_path={struct_align_2_teacher_path}, temperature={struct_align_2_temperature}, "
                f"fft_filter_enabled={struct_align_2_fft_enabled}, "
                f"fft_filter_method={struct_align_2_fft_method}, "
                f"fft_filter_student_enable={struct_align_2_fft_student_enable}, "
                f"fft_filter_orig_weight={struct_align_2_fft_orig_weight}, "
                f"fft_filter_fft_weight={struct_align_2_fft_weight}, "
                f"fft_filter_cutoff_ratio=[{struct_align_2_fft_cutoff_start}, {struct_align_2_fft_cutoff_end}], "
                f"fft_filter_warmup_steps={struct_align_2_fft_warmup_steps}, "
                f"fft_filter_warmup_schedule={struct_align_2_fft_warmup_schedule}"
            )
            if ema_teacher_enabled:
                logger.info("Struct-align-2 teacher source: EMA encoder (auto-switched by encoder_training.ema_teacher).")
        else:
            logger.info("Struct-align-2 disabled.")
        if spatial_grad_enabled:
            logger.info(
                "Spatial-grad enabled: "
                f"start_epoch={spatial_grad_start_epoch}, weight={spatial_grad_weight}, "
                f"end_epoch={spatial_grad_end_epoch}, loss_type={spatial_grad_loss_type}, "
                f"layers={spatial_grad_layers}, layer_weights={spatial_grad_layer_weights}, "
                f"warmup_steps={spatial_grad_warmup_steps}, warmup_schedule={spatial_grad_warmup_schedule}, "
                f"teacher_path={spatial_grad_teacher_path}"
            )
        else:
            logger.info("Spatial-grad disabled.")
        if edge_align_enabled:
            logger.info(
                "Edge-align enabled: "
                f"start_epoch={edge_align_start_epoch}, weight={edge_align_weight}, "
                f"end_epoch={edge_align_end_epoch}, loss_type={edge_align_loss_type}, "
                f"layers={edge_align_layers}, layer_weights={edge_align_layer_weights}, "
                f"warmup_steps={edge_align_warmup_steps}, warmup_schedule={edge_align_warmup_schedule}, "
                f"teacher_path={edge_align_teacher_path}"
            )
        else:
            logger.info("Edge-align disabled.")
        if ghpf_align_enabled:
            logger.info(
                "GHPF-align enabled: "
                f"start_epoch={ghpf_align_start_epoch}, weight={ghpf_align_weight}, "
                f"end_epoch={ghpf_align_end_epoch}, loss_type={ghpf_align_loss_type}, "
                f"cutoff_ratio={ghpf_align_cutoff_ratio}, "
                f"class_token_cosine_only={ghpf_align_class_token_cosine_only}, "
                f"layers={ghpf_align_layers}, layer_weights={ghpf_align_layer_weights}, "
                f"warmup_steps={ghpf_align_warmup_steps}, warmup_schedule={ghpf_align_warmup_schedule}, "
                f"teacher_path={ghpf_align_teacher_path}"
            )
            if ema_teacher_enabled:
                logger.info("GHPF-align teacher source: EMA encoder (auto-switched by encoder_training.ema_teacher).")
        else:
            logger.info("GHPF-align disabled.")
        if fft_align_enabled:
            logger.info(
                "FFT-align enabled: "
                f"start_epoch={fft_align_start_epoch}, weight={fft_align_weight}, "
                f"end_epoch={fft_align_end_epoch}, loss_type={fft_align_loss_type}, "
                f"cutoff_ratio={fft_align_cutoff_ratio}, "
                f"class_token_cosine_only={fft_align_class_token_cosine_only}, "
                f"layers={fft_align_layers}, layer_weights={fft_align_layer_weights}, "
                f"warmup_steps={fft_align_warmup_steps}, warmup_schedule={fft_align_warmup_schedule}, "
                f"teacher_path={fft_align_teacher_path}"
            )
            if ema_teacher_enabled:
                logger.info("FFT-align teacher source: EMA encoder (auto-switched by encoder_training.ema_teacher).")
        else:
            logger.info("FFT-align disabled.")
        if ghpf_recon_enabled:
            logger.info(
                "GHPF-recon enabled: "
                f"start_epoch={ghpf_recon_start_epoch}, weight={ghpf_recon_weight}, "
                f"end_epoch={ghpf_recon_end_epoch}, loss_type={ghpf_recon_loss_type}, "
                f"cutoff_ratio={ghpf_recon_cutoff_ratio}, normalize_map={ghpf_recon_normalize_map}, "
                f"eps={ghpf_recon_eps}, warmup_steps={ghpf_recon_warmup_steps}, "
                f"warmup_schedule={ghpf_recon_warmup_schedule}"
            )
        else:
            logger.info("GHPF-recon disabled.")
        if aux_skip_enabled:
            aux_layer_val = getattr(rae, "aux_skip_layer", None)
            logger.info(f"Aux-skip enabled: layer={aux_layer_val}, start_epoch={aux_skip_start_epoch}")
        else:
            logger.info("Aux-skip disabled.")
        if disc_aug is not None:
            logger.info(f"Using DiffAug with policies: {disc_aug}")
        else:
            logger.info("Not using DiffAug.")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        logger.info(disc_optim_msg)
        print(disc_sched_msg if disc_sched_msg else "No LR scheduler for discriminator.")
        logger.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    last_layer = decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch
    struct_align_start_step = struct_align_start_epoch * steps_per_epoch
    struct_align_2_start_step = struct_align_2_start_epoch * steps_per_epoch
    spatial_grad_start_step = spatial_grad_start_epoch * steps_per_epoch
    edge_align_start_step = edge_align_start_epoch * steps_per_epoch
    ghpf_align_start_step = ghpf_align_start_epoch * steps_per_epoch
    fft_align_start_step = fft_align_start_epoch * steps_per_epoch
    ghpf_recon_start_step = ghpf_recon_start_epoch * steps_per_epoch
    dist.barrier()
    benchmark_enabled = args.benchmark_steps > 0
    benchmark_seen = 0
    benchmark_measured = 0
    benchmark_start: float | None = None
    benchmark_done = False
    if benchmark_enabled and rank == 0:
        logger.info(f"Benchmark enabled: warmup={args.benchmark_warmup_steps}, measured={args.benchmark_steps}.")
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)

        # Epoch-based encoder gating
        encoder_should_train = encoder_train_enabled and epoch >= encoder_train_start_epoch
        if encoder_should_train:
            if encoder_train_mode == "lora" and hasattr(rae.encoder, "set_lora_trainable"):
                rae.encoder.set_lora_trainable(True)
                rae.encoder.train()
            else:
                rae.encoder.requires_grad_(True)
                rae.encoder.train()
        else:
            if encoder_train_mode == "lora" and hasattr(rae.encoder, "set_lora_trainable"):
                rae.encoder.set_lora_trainable(False)
                rae.encoder.eval()
            else:
                rae.encoder.requires_grad_(False)
                rae.encoder.eval()

        # Epoch-based aux_skip gating: enable/disable aux projection params
        aux_skip_active = aux_skip_enabled and epoch >= aux_skip_start_epoch
        if hasattr(rae, "aux_proj"):
            for p in rae.aux_proj.parameters():
                p.requires_grad_(aux_skip_active)
            for p in rae.aux_norm.parameters():
                p.requires_grad_(aux_skip_active)

        # Epoch-based noise_tau gating
        rae.noise_tau = _noise_tau_value if epoch >= noise_tau_start_epoch else 0.0

        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt"
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
                ema_encoder=ema_encoder,
            )
        for step, (images, _) in enumerate(loader):
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            images = images.to(device, non_blocking=True)
            real_normed = images * 2.0 - 1.0
            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()
            struct_align_fft_cur_ratio: float | None = None
            struct_align_2_fft_cur_ratio: float | None = None
            with autocast(**autocast_kwargs):
                use_distill = (
                    distill_enabled
                    and distill_loss_module is not None
                    and encoder_should_train
                    and epoch >= distill_start_epoch
                    and distill_weight > 0.0
                )
                use_distill_dual_fft = use_distill and distill_fft_enabled and distill_fft_student_enable

                struct_align_eff_weight = 0.0
                struct_align_in_epoch_window = (
                    epoch >= struct_align_start_epoch
                    and (struct_align_end_epoch is None or epoch <= struct_align_end_epoch)
                )
                if struct_align_enabled and encoder_should_train and struct_align_in_epoch_window:
                    # Ramp-up weight to avoid early training jitter when KL is large.
                    struct_align_eff_weight = rampup_weight(
                        struct_align_weight,
                        global_step=global_step,
                        start_step=struct_align_start_step,
                        warmup_steps=struct_align_warmup_steps,
                        schedule=struct_align_warmup_schedule,
                    )

                use_struct_align = struct_align_loss_module is not None and struct_align_eff_weight > 0.0
                use_struct_align_dual_fft = use_struct_align and struct_align_fft_enabled and struct_align_fft_student_enable

                struct_align_2_eff_weight = 0.0
                struct_align_2_in_epoch_window = (
                    epoch >= struct_align_2_start_epoch
                    and (struct_align_2_end_epoch is None or epoch <= struct_align_2_end_epoch)
                )
                if struct_align_2_enabled and encoder_should_train and struct_align_2_in_epoch_window:
                    struct_align_2_eff_weight = rampup_weight(
                        struct_align_2_weight,
                        global_step=global_step,
                        start_step=struct_align_2_start_step,
                        warmup_steps=struct_align_2_warmup_steps,
                        schedule=struct_align_2_warmup_schedule,
                    )

                use_struct_align_2 = struct_align_2_loss_module is not None and struct_align_2_eff_weight > 0.0
                use_struct_align_2_dual_fft = use_struct_align_2 and struct_align_2_fft_enabled and struct_align_2_fft_student_enable

                # ── spatial_grad effective weight ──
                spatial_grad_eff_weight = 0.0
                spatial_grad_in_epoch_window = (
                    epoch >= spatial_grad_start_epoch
                    and (spatial_grad_end_epoch is None or epoch <= spatial_grad_end_epoch)
                )
                if spatial_grad_enabled and encoder_should_train and spatial_grad_in_epoch_window:
                    spatial_grad_eff_weight = rampup_weight(
                        spatial_grad_weight,
                        global_step=global_step,
                        start_step=spatial_grad_start_step,
                        warmup_steps=spatial_grad_warmup_steps,
                        schedule=spatial_grad_warmup_schedule,
                    )
                use_spatial_grad = spatial_grad_loss_module is not None and spatial_grad_eff_weight > 0.0

                edge_align_eff_weight = 0.0
                edge_align_in_epoch_window = (
                    epoch >= edge_align_start_epoch
                    and (edge_align_end_epoch is None or epoch <= edge_align_end_epoch)
                )
                if edge_align_enabled and encoder_should_train and edge_align_in_epoch_window:
                    edge_align_eff_weight = rampup_weight(
                        edge_align_weight,
                        global_step=global_step,
                        start_step=edge_align_start_step,
                        warmup_steps=edge_align_warmup_steps,
                        schedule=edge_align_warmup_schedule,
                    )
                use_edge_align = edge_align_loss_module is not None and edge_align_eff_weight > 0.0

                ghpf_align_eff_weight = 0.0
                ghpf_align_in_epoch_window = (
                    epoch >= ghpf_align_start_epoch
                    and (ghpf_align_end_epoch is None or epoch <= ghpf_align_end_epoch)
                )
                if ghpf_align_enabled and encoder_should_train and ghpf_align_in_epoch_window:
                    ghpf_align_eff_weight = rampup_weight(
                        ghpf_align_weight,
                        global_step=global_step,
                        start_step=ghpf_align_start_step,
                        warmup_steps=ghpf_align_warmup_steps,
                        schedule=ghpf_align_warmup_schedule,
                    )
                use_ghpf_align = ghpf_align_loss_module is not None and ghpf_align_eff_weight > 0.0

                fft_align_eff_weight = 0.0
                fft_align_in_epoch_window = (
                    epoch >= fft_align_start_epoch
                    and (fft_align_end_epoch is None or epoch <= fft_align_end_epoch)
                )
                if fft_align_enabled and encoder_should_train and fft_align_in_epoch_window:
                    fft_align_eff_weight = rampup_weight(
                        fft_align_weight,
                        global_step=global_step,
                        start_step=fft_align_start_step,
                        warmup_steps=fft_align_warmup_steps,
                        schedule=fft_align_warmup_schedule,
                    )
                use_fft_align = fft_align_loss_module is not None and fft_align_eff_weight > 0.0

                ghpf_recon_eff_weight = 0.0
                ghpf_recon_in_epoch_window = (
                    epoch >= ghpf_recon_start_epoch
                    and (ghpf_recon_end_epoch is None or epoch <= ghpf_recon_end_epoch)
                )
                if ghpf_recon_enabled and ghpf_recon_in_epoch_window:
                    ghpf_recon_eff_weight = rampup_weight(
                        ghpf_recon_weight,
                        global_step=global_step,
                        start_step=ghpf_recon_start_step,
                        warmup_steps=ghpf_recon_warmup_steps,
                        schedule=ghpf_recon_warmup_schedule,
                    )
                use_ghpf_recon = ghpf_recon_loss_module is not None and ghpf_recon_eff_weight > 0.0

                if use_struct_align and struct_align_fft_enabled:
                    struct_align_fft_cur_ratio = schedule_ratio(
                        struct_align_fft_cutoff_start,
                        struct_align_fft_cutoff_end,
                        global_step=global_step,
                        start_step=struct_align_start_step,
                        warmup_steps=struct_align_fft_warmup_steps,
                        schedule=struct_align_fft_warmup_schedule,
                    )

                if use_struct_align_2 and struct_align_2_fft_enabled:
                    struct_align_2_fft_cur_ratio = schedule_ratio(
                        struct_align_2_fft_cutoff_start,
                        struct_align_2_fft_cutoff_end,
                        global_step=global_step,
                        start_step=struct_align_2_start_step,
                        warmup_steps=struct_align_2_fft_warmup_steps,
                        schedule=struct_align_2_fft_warmup_schedule,
                    )

                need_extra = use_distill or use_struct_align or use_struct_align_2 or use_spatial_grad or use_edge_align or use_ghpf_align or use_fft_align or aux_skip_active

                # When Teacher EMA is active, swap in ema_encoder so that the
                # latent used for decoding comes from the stable Teacher.
                # The Student encoder still receives gradients via distill/struct_align losses.
                _student_encoder = rae.encoder
                if ema_teacher_enabled and encoder_should_train and ema_encoder is not None:
                    rae.encoder = ema_encoder

                if need_extra:
                    recon, extra = ddp_model(images, return_extra=True)  # type: ignore[misc]
                else:
                    recon = ddp_model(images) # keep gradient synced

                # Restore Student encoder after Teacher-swapped forward pass.
                rae.encoder = _student_encoder

                # If alignment teachers are routed to EMA encoder, the `extra` tokens
                # produced above are EMA-side too. Re-extract student tokens here so
                # distill/struct_align/spatial_grad losses remain true student-vs-teacher.
                if (
                    need_extra
                    and ema_teacher_enabled
                    and encoder_should_train
                    and ema_encoder is not None
                    and (use_distill or use_struct_align or use_struct_align_2 or use_spatial_grad)
                ):
                    _align_layers = getattr(rae, "distill_layers", distill_layers)
                    _student_align = _extract_student_distill_tokens(
                        rae,
                        extra["teacher_image"],
                        layers=_align_layers,
                    )
                    extra["distill_feat"] = _student_align["distill_feat"]
                    extra["distill_feat_by_layer"] = _student_align.get("distill_feat_by_layer", None)
                    extra["distill_cls"] = _student_align.get("distill_cls", None)

                distill_fft_student = None
                struct_align_fft_student = None
                struct_align_2_fft_student = None
                if need_extra and (use_distill_dual_fft or use_struct_align_dual_fft or use_struct_align_2_dual_fft):
                    teacher_image = extra["teacher_image"]
                    if use_distill_dual_fft:
                        distill_teacher_fft_image = _apply_configured_fft_filter(
                            teacher_image,
                            distill_fft_cutoff_ratio,
                            method=distill_fft_method,
                        )
                        distill_fft_student = _extract_student_distill_tokens(
                            rae,
                            distill_teacher_fft_image,
                            layers=distill_layers,
                        )
                    if use_struct_align_dual_fft:
                        struct_fft_ratio = struct_align_fft_cur_ratio
                        if struct_fft_ratio is None:
                            struct_fft_ratio = struct_align_fft_cutoff_end
                        struct_teacher_fft_image = _apply_configured_fft_filter(
                            teacher_image,
                            float(struct_fft_ratio),
                            method=struct_align_fft_method,
                        )
                        struct_align_fft_student = _extract_student_distill_tokens(
                            rae,
                            struct_teacher_fft_image,
                            layers=distill_layers,
                        )
                    if use_struct_align_2_dual_fft:
                        struct_2_fft_ratio = struct_align_2_fft_cur_ratio
                        if struct_2_fft_ratio is None:
                            struct_2_fft_ratio = struct_align_2_fft_cutoff_end
                        struct_2_teacher_fft_image = _apply_configured_fft_filter(
                            teacher_image,
                            float(struct_2_fft_ratio),
                            method=struct_align_2_fft_method,
                        )
                        struct_align_2_fft_student = _extract_student_distill_tokens(
                            rae,
                            struct_2_teacher_fft_image,
                            layers=distill_layers,
                        )

                recon_normed = recon * 2.0 - 1.0
                rec_loss = (recon - images).abs().mean() # L1
                if use_lpips:
                    lpips_loss = lpips(real_normed, recon_normed)
                else:
                    lpips_loss = rec_loss.new_zeros(())

                if use_distill:
                    if use_distill_dual_fft:
                        if distill_layers is None:
                            distill_loss_orig = distill_loss_module(
                                teacher_image=extra["teacher_image"],
                                student_feat=extra["distill_feat"],
                                apply_fft=False,
                            )
                            distill_loss_fft = distill_loss_module(
                                teacher_image=extra["teacher_image"],
                                student_feat=distill_fft_student["distill_feat"],
                                apply_fft=True,
                                fft_filter_cutoff_ratio=distill_fft_cutoff_ratio,
                            )
                        else:
                            distill_loss_orig = distill_loss_module(
                                teacher_image=extra["teacher_image"],
                                student_feat_by_layer=extra["distill_feat_by_layer"],
                                apply_fft=False,
                            )
                            distill_loss_fft = distill_loss_module(
                                teacher_image=extra["teacher_image"],
                                student_feat_by_layer=distill_fft_student["distill_feat_by_layer"],
                                apply_fft=True,
                                fft_filter_cutoff_ratio=distill_fft_cutoff_ratio,
                            )
                        distill_loss = (
                            distill_fft_orig_weight * distill_loss_orig
                            + distill_fft_weight * distill_loss_fft
                        )
                    elif distill_layers is None:
                        distill_loss = distill_loss_module(
                            teacher_image=extra["teacher_image"],
                            student_feat=extra["distill_feat"],
                        )
                    else:
                        distill_loss = distill_loss_module(
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=extra["distill_feat_by_layer"],
                        )
                else:
                    distill_loss = rec_loss.new_zeros(())

                if use_struct_align:
                    if extra.get("distill_cls", None) is None:
                        raise ValueError(
                            "struct_align is enabled but student cls token is missing. "
                            "Ensure the encoder supports forward_tokens() (e.g. Dinov3withNorm)."
                        )
                    if use_struct_align_dual_fft:
                        if struct_align_fft_student is None or struct_align_fft_student.get("distill_cls", None) is None:
                            raise ValueError("struct_align fft_filter.student_enable=True requires student cls token for FFT branch.")
                        struct_align_loss_orig = struct_align_loss_module(
                            student_cls_token=extra["distill_cls"],
                            student_patch_tokens=extra["distill_feat"],
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=extra.get("distill_feat_by_layer", None),
                            apply_fft=False,
                        )
                        struct_align_loss_fft = struct_align_loss_module(
                            student_cls_token=struct_align_fft_student["distill_cls"],
                            student_patch_tokens=struct_align_fft_student["distill_feat"],
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=struct_align_fft_student.get("distill_feat_by_layer", None),
                            apply_fft=False,  # TEMP: teacher uses original image; student uses FFT-filtered image
                        )
                        struct_align_loss = (
                            struct_align_fft_orig_weight * struct_align_loss_orig
                            + struct_align_fft_weight * struct_align_loss_fft
                        )
                    else:
                        struct_align_loss = struct_align_loss_module(
                            student_cls_token=extra["distill_cls"],
                            student_patch_tokens=extra["distill_feat"],
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=extra.get("distill_feat_by_layer", None),
                            fft_filter_cutoff_ratio=struct_align_fft_cur_ratio,
                        )
                else:
                    struct_align_loss = rec_loss.new_zeros(())

                if use_struct_align_2:
                    if extra.get("distill_cls", None) is None:
                        raise ValueError(
                            "struct_align_2 is enabled but student cls token is missing. "
                            "Ensure the encoder supports forward_tokens() (e.g. Dinov3withNorm)."
                        )
                    if use_struct_align_2_dual_fft:
                        if struct_align_2_fft_student is None or struct_align_2_fft_student.get("distill_cls", None) is None:
                            raise ValueError("struct_align_2 fft_filter.student_enable=True requires student cls token for FFT branch.")
                        struct_align_2_loss_orig = struct_align_2_loss_module(
                            student_cls_token=extra["distill_cls"],
                            student_patch_tokens=extra["distill_feat"],
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=extra.get("distill_feat_by_layer", None),
                            apply_fft=False,
                        )
                        struct_align_2_loss_fft = struct_align_2_loss_module(
                            student_cls_token=struct_align_2_fft_student["distill_cls"],
                            student_patch_tokens=struct_align_2_fft_student["distill_feat"],
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=struct_align_2_fft_student.get("distill_feat_by_layer", None),
                            apply_fft=False,
                        )
                        struct_align_2_loss = (
                            struct_align_2_fft_orig_weight * struct_align_2_loss_orig
                            + struct_align_2_fft_weight * struct_align_2_loss_fft
                        )
                    else:
                        struct_align_2_loss = struct_align_2_loss_module(
                            student_cls_token=extra["distill_cls"],
                            student_patch_tokens=extra["distill_feat"],
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=extra.get("distill_feat_by_layer", None),
                            fft_filter_cutoff_ratio=struct_align_2_fft_cur_ratio,
                        )
                else:
                    struct_align_2_loss = rec_loss.new_zeros(())

                if use_spatial_grad:
                    if spatial_grad_layers is None:
                        spatial_grad_loss = spatial_grad_loss_module(
                            student_patch_tokens=extra["distill_feat"],
                            teacher_image=extra["teacher_image"],
                        )
                    else:
                        spatial_grad_loss = spatial_grad_loss_module(
                            student_patch_tokens=extra["distill_feat"],
                            teacher_image=extra["teacher_image"],
                            student_feat_by_layer=extra.get("distill_feat_by_layer", None),
                        )
                else:
                    spatial_grad_loss = rec_loss.new_zeros(())

                if use_edge_align:
                    _teacher_image = extra["teacher_image"]
                    _sobel_image = apply_sobel_edge(_teacher_image)
                    edge_align_student = _extract_student_distill_tokens(
                        rae, _sobel_image, layers=edge_align_layers
                    )
                    edge_align_loss = edge_align_loss_module(
                        student_patch_tokens=edge_align_student["distill_feat"],
                        teacher_image=_teacher_image,
                        student_feat_by_layer=edge_align_student.get("distill_feat_by_layer", None),
                    )
                else:
                    edge_align_loss = rec_loss.new_zeros(())

                if use_ghpf_align:
                    _teacher_image = extra["teacher_image"]
                    ghpf_align_student = _extract_student_distill_tokens(
                        rae, _teacher_image, layers=ghpf_align_layers
                    )
                    ghpf_align_loss = ghpf_align_loss_module(
                        student_patch_tokens=ghpf_align_student["distill_feat"],
                        teacher_image=_teacher_image,
                        student_cls_token=ghpf_align_student.get("distill_cls", None),
                        student_feat_by_layer=ghpf_align_student.get("distill_feat_by_layer", None),
                        student_cls_by_layer=ghpf_align_student.get("distill_cls_by_layer", None),
                    )
                else:
                    ghpf_align_loss = rec_loss.new_zeros(())

                if use_fft_align:
                    _teacher_image = extra["teacher_image"]
                    fft_align_student = _extract_student_distill_tokens(
                        rae, _teacher_image, layers=fft_align_layers
                    )
                    fft_align_loss = fft_align_loss_module(
                        student_patch_tokens=fft_align_student["distill_feat"],
                        teacher_image=_teacher_image,
                        student_cls_token=fft_align_student.get("distill_cls", None),
                        student_feat_by_layer=fft_align_student.get("distill_feat_by_layer", None),
                        student_cls_by_layer=fft_align_student.get("distill_cls_by_layer", None),
                    )
                else:
                    fft_align_loss = rec_loss.new_zeros(())

                if use_ghpf_recon:
                    ghpf_recon_loss = ghpf_recon_loss_module(
                        recon_image=recon,
                        target_image=images,
                    )
                else:
                    ghpf_recon_loss = rec_loss.new_zeros(())

                recon_total = (
                    rec_loss
                    + perceptual_weight * lpips_loss
                    + distill_weight * distill_loss
                    + struct_align_eff_weight * struct_align_loss
                    + struct_align_2_eff_weight * struct_align_2_loss
                    + spatial_grad_eff_weight * spatial_grad_loss
                    + edge_align_eff_weight * edge_align_loss
                    + ghpf_align_eff_weight * ghpf_align_loss
                    + fft_align_eff_weight * fft_align_loss
                    + ghpf_recon_eff_weight * ghpf_recon_loss
                )
                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = ddp_disc(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)
            # Calculate adaptive weight outside autocast (autograd operation, not forward pass)
            if use_gan:
                adaptive_weight = calculate_adaptive_weight(
                    recon_total, gan_loss, last_layer, max_d_weight
                )
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total
            total_loss.float()
            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            update_ema(ema_model, ddp_model.module, ema_decay)
            if ema_teacher_enabled and encoder_should_train and ema_encoder is not None:
                update_ema_encoder(ema_encoder, rae.encoder, ema_teacher_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                # Set model to eval mode and get fresh reconstruction with updated weights
                ddp_model.eval()
                ddp_disc.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**autocast_kwargs):
                        # Fresh forward pass with updated model weights (no gradient)
                        with torch.no_grad():
                            recon_disc = ddp_model(images)
                            recon_disc_normed = recon_disc * 2.0 - 1.0
                        # discretize
                        fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                        accuracy = (logits_real > logits_fake).float().mean() # accuracy of the discriminator
                    d_loss.float()
                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                        "disc_accuracy": accuracy.detach(),
                    }
                    epoch_metrics["disc_loss"] += d_loss.detach()
                    epoch_metrics["disc_accuracy"] += accuracy.detach()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                ddp_disc.eval()
                # Set model back to train mode
                ddp_model.train()

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["distill"] += distill_loss.detach()
            epoch_metrics["struct_align"] += struct_align_loss.detach()
            epoch_metrics["struct_align_2"] += struct_align_2_loss.detach()
            epoch_metrics["spatial_grad"] += spatial_grad_loss.detach()
            epoch_metrics["edge_align"] += edge_align_loss.detach()
            epoch_metrics["ghpf_align"] += ghpf_align_loss.detach()
            epoch_metrics["fft_align"] += fft_align_loss.detach()
            epoch_metrics["ghpf_recon"] += ghpf_recon_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/distill": distill_loss.detach().item(),
                    "loss/struct_align": struct_align_loss.detach().item(),
                    "loss/struct_align_2": struct_align_2_loss.detach().item(),
                    "loss/spatial_grad": spatial_grad_loss.detach().item(),
                    "loss/edge_align": edge_align_loss.detach().item(),
                    "loss/ghpf_align": ghpf_align_loss.detach().item(),
                    "loss/fft_align": fft_align_loss.detach().item(),
                    "loss/ghpf_recon": ghpf_recon_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if struct_align_enabled:
                    stats["weight/struct_align"] = float(struct_align_eff_weight)
                    if struct_align_fft_enabled and struct_align_fft_cur_ratio is not None:
                        stats["fft/struct_align_cutoff_ratio"] = float(struct_align_fft_cur_ratio)
                if struct_align_2_enabled:
                    stats["weight/struct_align_2"] = float(struct_align_2_eff_weight)
                    if struct_align_2_fft_enabled and struct_align_2_fft_cur_ratio is not None:
                        stats["fft/struct_align_2_cutoff_ratio"] = float(struct_align_2_fft_cur_ratio)
                if spatial_grad_enabled:
                    stats["weight/spatial_grad"] = float(spatial_grad_eff_weight)
                if edge_align_enabled:
                    stats["weight/edge_align"] = float(edge_align_eff_weight)
                if ghpf_align_enabled:
                    stats["weight/ghpf_align"] = float(ghpf_align_eff_weight)
                if fft_align_enabled:
                    stats["weight/fft_align"] = float(fft_align_eff_weight)
                if ghpf_recon_enabled:
                    stats["weight/ghpf_recon"] = float(ghpf_recon_eff_weight)
                if aux_skip_enabled and hasattr(rae, "aux_proj"):
                    _aux_w = rae.aux_proj.weight
                    stats["aux_skip/active"] = float(aux_skip_active)
                    stats["aux_skip/proj_weight_norm"] = float(_aux_w.norm().item())
                    if _aux_w.grad is not None:
                        stats["aux_skip/proj_grad_norm"] = float(_aux_w.grad.norm().item())
                if len(optimizer.param_groups) > 1:
                    stats["lr/encoder"] = optimizer.param_groups[1]["lr"]
                if disc_metrics:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                            "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                            "disc/weight": adaptive_weight.item(),
                        }
                    )
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
            if global_step % sample_every == 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    # only keep first 4 sample
                    sample_images = images[:4]
                    samples = ema_model.decode(ema_model.encode(sample_images))
                    # also concat input and reconstruction
                    comparison = torch.cat([sample_images, samples], dim=0).cpu().float()
                    # reshape to grid (sample at first row, recon at second row)
                    n = sample_images.size(0)
                    grid = make_grid(comparison, nrow=n)
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step)
                logger.info("Generating EMA samples done.")
            global_step += 1
            if benchmark_enabled:
                torch.cuda.synchronize(device)
                benchmark_seen += 1
                if benchmark_seen == args.benchmark_warmup_steps:
                    torch.cuda.reset_peak_memory_stats(device)
                    benchmark_start = time.perf_counter()
                elif benchmark_seen > args.benchmark_warmup_steps:
                    if benchmark_start is None:
                        torch.cuda.reset_peak_memory_stats(device)
                        benchmark_start = time.perf_counter()
                    benchmark_measured += 1
                    if benchmark_measured >= args.benchmark_steps:
                        torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - benchmark_start
                        metrics = torch.tensor(
                            [
                                elapsed,
                                float(torch.cuda.max_memory_allocated(device)),
                                float(torch.cuda.max_memory_reserved(device)),
                            ],
                            dtype=torch.float64,
                            device=device,
                        )
                        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
                        if rank == 0:
                            avg_step = metrics[0].item() / args.benchmark_steps
                            peak_alloc_gib = metrics[1].item() / (1024 ** 3)
                            peak_reserved_gib = metrics[2].item() / (1024 ** 3)
                            logger.info(
                                "BENCHMARK_RESULT "
                                f"warmup_steps={args.benchmark_warmup_steps} "
                                f"measured_steps={args.benchmark_steps} "
                                f"elapsed_sec={metrics[0].item():.6f} "
                                f"avg_step_sec={avg_step:.6f} "
                                f"peak_allocated_gib={peak_alloc_gib:.6f} "
                                f"peak_reserved_gib={peak_reserved_gib:.6f}"
                            )
                        benchmark_done = True
                        break
        if benchmark_done:
            break
        if rank == 0 and num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_gan": avg_gan,
                "epoch/loss_struct_align": (epoch_metrics["struct_align"] / num_batches).item(),
                "epoch/loss_struct_align_2": (epoch_metrics["struct_align_2"] / num_batches).item(),
                "epoch/loss_spatial_grad": (epoch_metrics["spatial_grad"] / num_batches).item(),
                "epoch/loss_ghpf_align": (epoch_metrics["ghpf_align"] / num_batches).item(),
                "epoch/loss_fft_align": (epoch_metrics["fft_align"] / num_batches).item(),
                "epoch/loss_ghpf_recon": (epoch_metrics["ghpf_recon"] / num_batches).item(),
            }
            if disc_metrics:
                epoch_stats.update(
                    {
                        "epoch/loss_disc": (epoch_metrics["disc_loss"] / num_batches).item(),
                        "epoch/disc_logits_real": disc_metrics["logits_real"].item(),
                        "epoch/disc_logits_fake": disc_metrics["logits_fake"].item(),
                        "epoch/disc_accuracy": (epoch_metrics["disc_accuracy"] / num_batches).item(),
                    }
                )
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
    # save the final ckpt
    if rank == 0 and not benchmark_enabled:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt"
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
            discriminator,
            disc_optimizer,
            disc_scheduler,
            save_stage2_compatible_merged=True,
            ema_encoder=ema_encoder,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
