import random
import torch
import torch.nn as nn
from .decoders import GeneralDecoder
from .decoders.utils import ViTMAEConfig
from .encoders import ARCHS
from transformers import AutoConfig, AutoImageProcessor
from omegaconf import OmegaConf
from typing import Optional
from math import sqrt
from typing import Protocol

class Stage1Protocal(Protocol):
    # must have patch size attribute
    patch_size: int
    hidden_size: int
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

class RAE(nn.Module):
    def __init__(self,
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov3withNorm',
        encoder_config_path: str = 'models/dinov3b',
        encoder_input_size: int = 224,
        encoder_params: dict = {},
        # ---- decoder configs ----
        decoder_config_path: str = 'configs/decoder/ViTXL/config.yaml',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.8,
        noise_tau_start_epoch: int = 0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
        # ---- aux skip connection ----
        aux_skip_layer: Optional[int] = None,
        aux_drop_prob: float = 0.0,
    ):
        super().__init__()
        encoder_cls = ARCHS[encoder_cls]
        self.encoder: Stage1Protocal = encoder_cls(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent

        # decoder
        decoder_config = ViTMAEConfig(**OmegaConf.to_container(OmegaConf.load(decoder_config_path), resolve=True))
        decoder_config.hidden_size = self.latent_dim # set the hidden size of the decoder to be the same as the encoder's output
        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)
        # load pretrained decoder weights
        if pretrained_decoder_path is not None:
            print(f"Loading pretrained decoder from {pretrained_decoder_path}")
            state_dict = torch.load(pretrained_decoder_path, map_location='cpu')
            keys = self.decoder.load_state_dict(state_dict, strict=False)
            if len(keys.missing_keys) > 0:
                print(f"Missing keys when loading pretrained decoder: {keys.missing_keys}")
        self.noise_tau = noise_tau
        # Kept for config compatibility; epoch-based gating is handled in train_stage1.py.
        self.noise_tau_start_epoch = noise_tau_start_epoch
        self.reshape_to_2d = reshape_to_2d

        # aux skip: inject an intermediate encoder layer into the decoder input
        self.aux_skip_layer = aux_skip_layer
        self.aux_drop_prob = aux_drop_prob
        if aux_skip_layer is not None:
            decoder_hidden_size = decoder_config.decoder_hidden_size
            self.aux_norm = nn.LayerNorm(self.latent_dim)
            self.aux_proj = nn.Linear(self.latent_dim, decoder_hidden_size, bias=True)
            nn.init.zeros_(self.aux_proj.weight)
            nn.init.zeros_(self.aux_proj.bias)
            print(f"Aux-skip enabled: encoder layer {aux_skip_layer} -> decoder (zero-init)")

        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location='cpu')
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False

    def _encoder_requires_grad(self) -> bool:
        return any(p.requires_grad for p in self.encoder.parameters())

    def _resize_for_encoder(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode='bicubic',
                align_corners=False,
            )
        return x

    def _normalize_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        return x

    def _prepare_encoder_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self._resize_for_encoder(x)
        x = self._normalize_encoder_input(x)
        return x
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc_input = self._prepare_encoder_input(x)
        grad_enabled = self.training and self._encoder_requires_grad()
        with torch.set_grad_enabled(grad_enabled):
            if self.aux_skip_layer is not None and hasattr(self.encoder, "forward_tokens"):
                out = self.encoder.forward_tokens(enc_input, layers=[self.aux_skip_layer])
                z = out["patch_tokens"]
                self._aux_feat = out["patch_tokens_by_layer"][self.aux_skip_layer].detach()
            else:
                z = self.encoder(enc_input)
                self._aux_feat = None
            if self.training and self.noise_tau > 0:
                z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def encode_with_distill(self, x: torch.Tensor) -> dict:
        teacher_image = self._resize_for_encoder(x)
        enc_input = self._normalize_encoder_input(teacher_image)
        grad_enabled = self.training and self._encoder_requires_grad()
        with torch.set_grad_enabled(grad_enabled):
            cls_token = None
            distill_feat_by_layer = None
            if hasattr(self.encoder, "forward_tokens"):
                layers = getattr(self, "distill_layers", None)
                # merge aux_skip_layer into the layers we request
                aux_layer = self.aux_skip_layer
                if aux_layer is not None:
                    layers_to_fetch = sorted(set((list(layers) if layers else []) + [aux_layer]))
                else:
                    layers_to_fetch = list(layers) if layers else []
                if layers_to_fetch:
                    out = self.encoder.forward_tokens(enc_input, layers=layers_to_fetch)
                else:
                    out = self.encoder.forward_tokens(enc_input)
                tokens = out["patch_tokens"]
                cls_token = out["cls_token"]
                if isinstance(out, dict) and "patch_tokens_by_layer" in out:
                    distill_feat_by_layer = out["patch_tokens_by_layer"]
            else:
                tokens = self.encoder(enc_input)
                aux_layer = None

            distill_feat = tokens
            if self.training and self.noise_tau > 0:
                tokens = self.noising(tokens)

        z = tokens
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)

        aux_feat = None
        if aux_layer is not None and distill_feat_by_layer is not None and aux_layer in distill_feat_by_layer:
            aux_feat = distill_feat_by_layer[aux_layer].detach()

        return {
            "z": z,
            "enc_input": enc_input,
            "teacher_image": teacher_image,
            "distill_feat": distill_feat,
            "distill_feat_by_layer": distill_feat_by_layer,
            "distill_cls": cls_token,
            "aux_feat": aux_feat,
        }

    def decode(self, z: torch.Tensor, aux_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        if self.reshape_to_2d:
            b, c, h, w = z.shape
            n = h * w
            z = z.view(b, c, n).transpose(1, 2)
        # project aux features if available
        aux_proj = None
        if aux_feat is not None and self.aux_skip_layer is not None:
            aux_proj = self.aux_proj(self.aux_norm(aux_feat))  # [B, N, decoder_hidden_size]
        output = self.decoder(z, aux_hidden_states=aux_proj, drop_cls_token=False).logits
        x_rec = self.decoder.unpatchify(output)
        x_rec = x_rec * self.encoder_std.to(x_rec.device) + self.encoder_mean.to(x_rec.device)
        return x_rec

    def _maybe_drop_aux(self, aux_feat: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Drop aux_feat during eval (always) or training (with aux_drop_prob)."""
        if not self.training:
            return None
        if aux_feat is not None and self.aux_drop_prob > 0.0:
            # use Python random to avoid tensor.item() which breaks torch.compile
            if random.random() < self.aux_drop_prob:
                return None
        return aux_feat

    def forward(self, x: torch.Tensor, *, return_extra: bool = False):
        if return_extra:
            enc_out = self.encode_with_distill(x)
            aux_feat = self._maybe_drop_aux(enc_out.get("aux_feat"))
            x_rec = self.decode(enc_out["z"], aux_feat=aux_feat)
            extra = {
                "enc_input": enc_out["enc_input"],
                "teacher_image": enc_out["teacher_image"],
                "distill_feat": enc_out["distill_feat"],
                "distill_feat_by_layer": enc_out.get("distill_feat_by_layer", None),
                "distill_cls": enc_out["distill_cls"],
            }
            return x_rec, extra
        z = self.encode(x)
        aux_feat = self._maybe_drop_aux(getattr(self, "_aux_feat", None))
        x_rec = self.decode(z, aux_feat=aux_feat)
        return x_rec
