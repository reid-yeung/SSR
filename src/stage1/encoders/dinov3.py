import os
from collections.abc import Mapping
from typing import Any, Optional
import re

import torch
from torch import nn
from transformers import AutoModel

from . import register_encoder


@register_encoder()
class Dinov3withNorm(nn.Module):
    def __init__(
        self,
        dinov3_path: str,
        normalize: bool = True,
        lora: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        hf_offline = os.environ.get("RAE_HF_OFFLINE", "0") == "1"
        try:
            self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            if hf_offline:
                raise
            self.encoder = AutoModel.from_pretrained(dinov3_path, local_files_only=False)

        self.encoder.requires_grad_(False)

        # Note: when instantiated from OmegaConf, `lora` may be a DictConfig (Mapping but not a dict).
        self._lora_cfg = dict(lora) if isinstance(lora, Mapping) else {}
        self._lora_enabled = bool(self._lora_cfg.get("enabled", False))
        if self._lora_enabled:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "LoRA is enabled but 'peft' is not available. Install it with: pip install peft"
                ) from e

            r = int(self._lora_cfg.get("r", 16))
            lora_alpha = int(self._lora_cfg.get("lora_alpha", r))
            target_modules_raw = self._lora_cfg.get("target_modules", ["q_proj", "v_proj"])
            # OmegaConf may provide ListConfig which is not JSON-serializable; PEFT saves config as JSON.
            if isinstance(target_modules_raw, str):
                target_modules = target_modules_raw
            else:
                target_modules = [str(x) for x in list(target_modules_raw)]
            lora_dropout = float(self._lora_cfg.get("lora_dropout", 0.0))
            bias = str(self._lora_cfg.get("bias", "none"))

            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            # get_peft_model modifies the model in-place and returns a PeftModel wrapper.
            self.encoder = get_peft_model(self.encoder, peft_config)
            # Keep base weights frozen; gate LoRA weights via set_lora_trainable().
            self.encoder.requires_grad_(False)
            self.set_lora_trainable(True)

        if normalize:
            for attr_name in ["layernorm", "norm"]:
                layer = getattr(self.encoder, attr_name, None)
                if layer is not None and hasattr(layer, "elementwise_affine"):
                    layer.elementwise_affine = False
                    layer.weight = None
                    layer.bias = None
                    break

        cfg = self.encoder.config
        self.patch_size = getattr(cfg, "patch_size", None)
        self.hidden_size = getattr(cfg, "hidden_size", None)
        self.num_hidden_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
        if self.patch_size is None or self.hidden_size is None:
            raise ValueError("Dinov3withNorm encoder requires config.patch_size and config.hidden_size.")

    def lora_enabled(self) -> bool:
        return bool(self._lora_enabled)

    def _iter_lora_named_parameters(self):
        for name, param in self.encoder.named_parameters():
            # PEFT uses names like: ...q_proj.lora_A.default.weight / lora_B...
            if ".lora_" in name or "lora_A" in name or "lora_B" in name:
                yield name, param

    def set_lora_trainable(self, enabled: bool) -> None:
        """Enable/disable gradients for LoRA weights only.

        Base model weights always stay frozen in this wrapper.
        """
        if not self._lora_enabled:
            return
        enabled = bool(enabled)

        # Ensure base weights are frozen.
        for _, param in self.encoder.named_parameters():
            param.requires_grad = False

        for _, param in self._iter_lora_named_parameters():
            param.requires_grad = enabled

    def save_lora_adapter(self, output_dir: str) -> None:
        if not self._lora_enabled:
            raise RuntimeError("save_lora_adapter() called but LoRA is not enabled for this encoder.")
        if not hasattr(self.encoder, "save_pretrained"):
            raise RuntimeError("LoRA is enabled but encoder does not support save_pretrained().")
        os.makedirs(output_dir, exist_ok=True)
        # Save in PEFT format (adapter_config.json + adapter_model.bin).
        self.encoder.save_pretrained(output_dir, safe_serialization=False)

    def load_lora_adapter(self, adapter_dir: str) -> None:
        if not self._lora_enabled:
            raise RuntimeError("load_lora_adapter() called but LoRA is not enabled for this encoder.")
        # We expect adapter_model.bin produced by save_lora_adapter(safe_serialization=False).
        adapter_bin = os.path.join(adapter_dir, "adapter_model.bin")
        adapter_st = os.path.join(adapter_dir, "adapter_model.safetensors")
        if os.path.isfile(adapter_bin):
            state = torch.load(adapter_bin, map_location="cpu")
        elif os.path.isfile(adapter_st):
            try:
                from safetensors.torch import load_file
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    f"Found '{adapter_st}' but safetensors is not available. Install it with: pip install safetensors"
                ) from e
            state = load_file(adapter_st)
        else:
            raise FileNotFoundError(
                f"No adapter weights found in {adapter_dir}. Expected adapter_model.bin or adapter_model.safetensors"
            )

        if not isinstance(state, dict) or len(state) == 0:
            raise RuntimeError(f"Adapter state is empty in {adapter_dir}.")

        lora_markers = (".lora_", "lora_A", "lora_B", "lora_embedding")
        adapter_keys = set(state.keys())
        adapter_lora_keys = {k for k in adapter_keys if any(m in k for m in lora_markers)}
        if len(adapter_lora_keys) == 0:
            raise RuntimeError(
                "Adapter state does not contain LoRA keys. "
                f"Found {len(adapter_keys)} keys in {adapter_dir}."
            )

        model_lora_keys = {
            k for k in self.encoder.state_dict().keys()
            if any(m in k for m in lora_markers)
        }
        if len(model_lora_keys) == 0:
            raise RuntimeError("Current encoder has no LoRA keys, but adapter loading was requested.")

        def _normalize_lora_key_for_compare(k: str) -> str:
            # PEFT naming may differ across versions/adapters:
            # - ...lora_A.weight
            # - ...lora_A.default.weight
            # Normalize them into a comparable key-space.
            k = re.sub(r"\.default(?=\.weight$)", "", k)
            return k

        adapter_norm = {_normalize_lora_key_for_compare(k) for k in adapter_lora_keys}
        model_norm = {_normalize_lora_key_for_compare(k) for k in model_lora_keys}

        # Fail fast for obvious key-space mismatches (e.g. wrong backbone/adapter format).
        if adapter_norm.isdisjoint(model_norm):
            raise RuntimeError(
                "LoRA adapter keys do not match current encoder LoRA key-space. "
                f"Sample adapter key: {next(iter(adapter_lora_keys))}\n"
                f"Sample model key: {next(iter(model_lora_keys))}"
            )

        # Use PEFT's set_peft_model_state_dict to correctly map adapter keys
        # (e.g. base_model.model.X.lora_A.weight) into the PEFT-wrapped model.
        # Plain load_state_dict(strict=False) silently skips all keys due to key
        # format mismatch, leaving LoRA weights at their initialized (near-zero) values.
        from peft import set_peft_model_state_dict
        load_result = set_peft_model_state_dict(self.encoder, state)

        # Different PEFT versions may return _IncompatibleKeys or None.
        if load_result is not None:
            missing = list(getattr(load_result, "missing_keys", []) or [])
            unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
            missing_lora = [k for k in missing if any(m in k for m in lora_markers)]
            unexpected_lora = [k for k in unexpected if any(m in k for m in lora_markers)]
            if len(missing_lora) > 0 or len(unexpected_lora) > 0:
                raise RuntimeError(
                    "LoRA adapter key mismatch while loading. "
                    f"missing_lora={missing_lora[:10]}, unexpected_lora={unexpected_lora[:10]}"
                )

    def merge_lora_and_unload_(self) -> None:
        """Merge LoRA weights into the base encoder and remove PEFT modules.

        After this call, the encoder becomes a plain HF model (no LoRA keys in state_dict),
        which is required for downstream consumers that expect vanilla DINOv3 weights.
        """
        if not self._lora_enabled:
            return
        if not hasattr(self.encoder, "merge_and_unload"):
            raise RuntimeError("LoRA is enabled but encoder does not support merge_and_unload().")
        self.encoder = self.encoder.merge_and_unload()
        self.encoder.requires_grad_(False)
        self._lora_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        return tokens["patch_tokens"]

    def forward_tokens(self, x: torch.Tensor, *, layers: Optional[list[int]] = None) -> dict[str, torch.Tensor]:
        """Return CLS + patch tokens for SSL-style losses.

        Returns:
            {
              "cls_token": (B, C),
              "patch_tokens": (B, N, C),
                            "cls_tokens_by_layer": Dict[layer, (B, C)] (when ``layers`` is set),
                            "patch_tokens_by_layer": Dict[layer, (B, N, C)] (when ``layers`` is set),
            }
        """
        need_hidden_states = layers is not None and len(layers) > 0

        # HF vision models expect `pixel_values=...`. When wrapped by PEFT, positional calls may break.
        # We only request hidden states when multi-layer tokens are needed to avoid extra overhead.
        try:
            if need_hidden_states:
                outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
            else:
                outputs = self.encoder(pixel_values=x)
        except TypeError:
            if need_hidden_states:
                outputs = self.encoder(x, output_hidden_states=True, return_dict=True)
            else:
                outputs = self.encoder(x)
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError("DINOv3 encoder output does not have last_hidden_state.")
        hidden = outputs.last_hidden_state  # (B, 1 + num_register_tokens + N_patches, C)
        cls_token = hidden[:, 0, :]
        unused_token_num = 1 + self.num_register_tokens
        patch_tokens = hidden[:, unused_token_num:, :]

        out: dict[str, torch.Tensor] = {"cls_token": cls_token, "patch_tokens": patch_tokens}
        if need_hidden_states:
            if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
                raise ValueError("output_hidden_states=True was requested but encoder did not return hidden_states.")
            if self.num_hidden_layers <= 0:
                raise ValueError("Encoder config.num_hidden_layers is missing; cannot validate requested layers.")

            patch_tokens_by_layer: dict[int, torch.Tensor] = {}
            cls_tokens_by_layer: dict[int, torch.Tensor] = {}
            hidden_states = outputs.hidden_states
            for layer in layers:
                layer_i = int(layer)
                if layer_i < 1 or layer_i > self.num_hidden_layers:
                    raise ValueError(
                        f"Invalid layer={layer_i}. Scheme-B expects transformer block index in [1, {self.num_hidden_layers}]."
                    )
                h = hidden_states[layer_i]
                cls_tokens_by_layer[layer_i] = h[:, 0, :]
                patch_tokens_by_layer[layer_i] = h[:, unused_token_num:, :]

            # NOTE: keys are ints to make downstream dict lookups stable.
            out["cls_tokens_by_layer"] = cls_tokens_by_layer  # type: ignore[assignment]
            out["patch_tokens_by_layer"] = patch_tokens_by_layer  # type: ignore[assignment]
        return out
