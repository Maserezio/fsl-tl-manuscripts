"""Backbone loaders + a uniform wrapper interface.

Every backbone is exposed through a :class:`BackboneWrapper` that guarantees::

    .patch_size : int
    .embed_dim  : int
    .num_blocks : int
    .num_prefix : int                      # class + register tokens to strip
    .supports_windowing : bool
    .forward_features(x, windowed, window, global_idx)
        -> List[(B, N_spatial, C)]         # one per block, prefix stripped

Loaders for DINOv2 / DINOv3 / RADIO try the real weights first and *fall back*
to a randomly-initialised ViT stub with matching dims when the network or the
weights are unreachable (VSC cluster / offline). This keeps the whole library
runnable without internet -- the smoke test relies on it.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import EncoderConfig
from .windowed_attn import window_partition_tokens, window_unpartition_tokens


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
@dataclass
class BackboneSpec:
    name: str
    kind: str  # "dinov2" | "dinov3" | "radio"
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    num_reg: int  # register tokens when use_registers
    has_cls: bool
    hub_id: str = ""  # torch.hub / HF identifier


_REGISTRY = {
    # DINOv2 (patch 14)
    "dinov2_vits14": BackboneSpec("dinov2_vits14", "dinov2", 14, 384, 12, 6, 0, True, "dinov2_vits14"),
    "dinov2_vitb14": BackboneSpec("dinov2_vitb14", "dinov2", 14, 768, 12, 12, 0, True, "dinov2_vitb14"),
    "dinov2_vitl14": BackboneSpec("dinov2_vitl14", "dinov2", 14, 1024, 24, 16, 0, True, "dinov2_vitl14"),
    "dinov2_vits14_reg": BackboneSpec("dinov2_vits14_reg", "dinov2", 14, 384, 12, 6, 4, True, "dinov2_vits14_reg"),
    "dinov2_vitb14_reg": BackboneSpec("dinov2_vitb14_reg", "dinov2", 14, 768, 12, 12, 4, True, "dinov2_vitb14_reg"),
    "dinov2_vitl14_reg": BackboneSpec("dinov2_vitl14_reg", "dinov2", 14, 1024, 24, 16, 4, True, "dinov2_vitl14_reg"),
    # DINOv3 (patch 16, registers by default)
    "dinov3_vits16": BackboneSpec("dinov3_vits16", "dinov3", 16, 384, 12, 6, 4, True, "dinov3_vits16"),
    "dinov3_vitb16": BackboneSpec("dinov3_vitb16", "dinov3", 16, 768, 12, 12, 4, True, "dinov3_vitb16"),
    "dinov3_vitl16": BackboneSpec("dinov3_vitl16", "dinov3", 16, 1024, 24, 16, 4, True, "dinov3_vitl16"),
    # NVIDIA RADIO (patch 16; one summary token, no registers)
    "radio_v2.5-b": BackboneSpec("radio_v2.5-b", "radio", 16, 768, 12, 12, 0, True, "radio_v2.5-b"),
    "radio_v2.5-l": BackboneSpec("radio_v2.5-l", "radio", 16, 1024, 24, 16, 0, True, "radio_v2.5-l"),
    # Plain supervised timm ViT (patch 16, CLS only, no registers) -- loaded via
    # timm rather than torch.hub/HF; used to put a genuinely-supervised (non-SSL)
    # ViT baseline through the same SFP-hierarchy path as DINOv2/v3/RADIO.
    "vit_small_patch16_224.augreg_in21k": BackboneSpec(
        "vit_small_patch16_224.augreg_in21k", "timm_vit", 16, 384, 12, 6, 0, True,
        "vit_small_patch16_224.augreg_in21k",
    ),
    # XS bracket partner for the size axis. 5.5M in the backbone proper; timm
    # reports 9.7M for the checkpoint because augreg_in21k carries a 21843-class
    # head (192 x 21843 = 4.2M) that the SFP path never uses.
    "vit_tiny_patch16_224.augreg_in21k": BackboneSpec(
        "vit_tiny_patch16_224.augreg_in21k", "timm_vit", 16, 192, 12, 3, 0, True,
        "vit_tiny_patch16_224.augreg_in21k",
    ),
    # DINOv3 ViT-S/16 via timm instead of the "dinov3" torch.hub loader. Meta gates
    # the hub weights, so the hub path fails and hier_encoder (correctly) refuses to
    # fall back to random init; timm serves the same lvd1689m checkpoint ungated.
    # 4 register tokens + CLS = 5 prefix tokens, which the timm_vit reader strips.
    "dinov3_vits16_timm": BackboneSpec(
        "dinov3_vits16_timm", "timm_vit", 16, 384, 12, 6, 4, True,
        "vit_small_patch16_dinov3.lvd1689m",
    ),
}


def list_backbones() -> List[str]:
    return sorted(_REGISTRY)


def get_spec(name: str) -> BackboneSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backbone {name!r}; available: {list_backbones()}"
        )
    return _REGISTRY[name]


# --------------------------------------------------------------------------- #
#  Fallback ViT (also serves as the offline / "radio-like" stub)
# --------------------------------------------------------------------------- #
class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BackboneWrapper(nn.Module):
    """Base class fixing the public interface; subclasses set attributes."""

    patch_size: int
    embed_dim: int
    num_blocks: int
    num_prefix: int
    supports_windowing: bool = False

    def forward_features(
        self,
        x: torch.Tensor,
        windowed: bool = False,
        window: int = 14,
        global_idx: Tuple[int, ...] = (),
    ) -> List[torch.Tensor]:
        raise NotImplementedError


class FallbackViTBackbone(BackboneWrapper):
    """Random-init ViT with a controllable prefix; supports windowed attention."""

    supports_windowing = True

    def __init__(self, spec: BackboneSpec, cfg: EncoderConfig):
        super().__init__()
        self.patch_size = spec.patch_size
        self.embed_dim = spec.embed_dim
        self.num_blocks = spec.depth
        self.num_reg = spec.num_reg if cfg.use_registers else 0
        self.has_cls = spec.has_cls
        self.num_prefix = (1 if self.has_cls else 0) + self.num_reg

        self.patch_embed = nn.Conv2d(3, self.embed_dim, spec.patch_size, spec.patch_size)
        if self.has_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if self.num_reg:
            self.reg_tokens = nn.Parameter(torch.zeros(1, self.num_reg, self.embed_dim))
        base = cfg.img_size or (448, 448)
        self.base_grid = (base[0] // spec.patch_size, base[1] // spec.patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.base_grid[0] * self.base_grid[1], self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [_Block(self.embed_dim, spec.num_heads) for _ in range(spec.depth)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self._pos_cache = {}

    def _interp_pos(self, hp: int, wp: int, device, dtype) -> torch.Tensor:
        key = (hp, wp)
        if key not in self._pos_cache:
            gh, gw = self.base_grid
            pos = self.pos_embed.reshape(1, gh, gw, self.embed_dim).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(hp, wp), mode="bicubic", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, hp * wp, self.embed_dim)
            self._pos_cache[key] = pos.detach()
        return self._pos_cache[key].to(device=device, dtype=dtype)

    def forward_features(self, x, windowed=False, window=14, global_idx=()):
        b = x.shape[0]
        hp, wp = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        tok = self.patch_embed(x)  # (B, C, Hp, Wp)
        tok = rearrange(tok, "b c hp wp -> b (hp wp) c")
        tok = tok + self._interp_pos(hp, wp, tok.device, tok.dtype)

        prefix = []
        if self.has_cls:
            prefix.append(self.cls_token.expand(b, -1, -1))
        if self.num_reg:
            prefix.append(self.reg_tokens.expand(b, -1, -1))
        if prefix:
            x = torch.cat(prefix + [tok], dim=1)
        else:
            x = tok

        feats: List[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            if windowed and i not in global_idx:
                pre, sp = x[:, : self.num_prefix], x[:, self.num_prefix :]
                sp_w, pad_hw = window_partition_tokens(sp, hp, wp, window)
                sp_w = blk(sp_w)
                sp = window_unpartition_tokens(sp_w, window, pad_hw, hp, wp)
                x = torch.cat([pre, sp], dim=1) if self.num_prefix else sp
            else:
                x = blk(x)
            feats.append(x[:, self.num_prefix :])
        if self.norm is not None:
            feats[-1] = self.norm(feats[-1])
        return feats


# --------------------------------------------------------------------------- #
#  Real backbone loaders (best-effort; fall back on failure)
# --------------------------------------------------------------------------- #
class _HubDinoBackbone(BackboneWrapper):
    """Wraps a torch.hub / HF DINO model using its intermediate-layer API."""

    supports_windowing = False

    def __init__(self, model: nn.Module, spec: BackboneSpec, cfg: EncoderConfig):
        super().__init__()
        self.model = model
        self.patch_size = spec.patch_size
        self.embed_dim = spec.embed_dim
        self.num_blocks = spec.depth
        self.num_prefix = (1 if spec.has_cls else 0) + (
            spec.num_reg if cfg.use_registers else 0
        )
        self._norm = cfg.norm_features

    def forward_features(self, x, windowed=False, window=14, global_idx=()):
        if windowed:
            warnings.warn(
                "windowed_attn is only applied to the fallback ViT; the real "
                f"{type(self.model).__name__} runs full attention.",
                stacklevel=2,
            )
        n = list(range(self.num_blocks))
        outs = self.model.get_intermediate_layers(
            x, n=n, reshape=False, return_class_token=False, norm=self._norm
        )
        return list(outs)


def _try_load_dinov2(spec: BackboneSpec, cfg: EncoderConfig) -> BackboneWrapper:
    wdir = cfg.weights_dir("dinov2")
    if wdir:
        os.environ.setdefault("TORCH_HOME", wdir)
    model = torch.hub.load("facebookresearch/dinov2", spec.hub_id, pretrained=cfg.pretrained)
    return _HubDinoBackbone(model, spec, cfg)


def _try_load_dinov3(spec: BackboneSpec, cfg: EncoderConfig) -> BackboneWrapper:
    wdir = cfg.weights_dir("dinov3")
    # Preferred: HuggingFace transformers
    try:
        from transformers import AutoModel  # noqa

        repo = os.path.join(wdir, spec.hub_id) if wdir else f"facebook/{spec.hub_id}"
        model = AutoModel.from_pretrained(repo)
        return _HFDinoV3Backbone(model, spec, cfg)
    except Exception:
        if wdir:
            os.environ.setdefault("TORCH_HOME", wdir)
        model = torch.hub.load("facebookresearch/dinov3", spec.hub_id, pretrained=cfg.pretrained)
        return _HubDinoBackbone(model, spec, cfg)


class _HFDinoV3Backbone(BackboneWrapper):
    supports_windowing = False

    def __init__(self, model: nn.Module, spec: BackboneSpec, cfg: EncoderConfig):
        super().__init__()
        self.model = model
        self.patch_size = spec.patch_size
        self.embed_dim = spec.embed_dim
        self.num_blocks = spec.depth
        self.num_prefix = (1 if spec.has_cls else 0) + (
            spec.num_reg if cfg.use_registers else 0
        )

    def forward_features(self, x, windowed=False, window=14, global_idx=()):
        out = self.model(x, output_hidden_states=True)
        hs = out.hidden_states[1:]  # drop embedding layer; keep per-block
        return [h[:, self.num_prefix :] for h in hs]


class _RadioBackbone(BackboneWrapper):
    """Wraps NVIDIA RADIO; keeps only spatial tokens (drops the summary token)."""

    supports_windowing = False

    def __init__(self, model: nn.Module, spec: BackboneSpec, cfg: EncoderConfig):
        super().__init__()
        self.model = model
        self.patch_size = spec.patch_size
        self.embed_dim = spec.embed_dim
        self.num_blocks = spec.depth
        self.num_prefix = 0  # RADIO returns spatial tokens already split out

    def forward_features(self, x, windowed=False, window=14, global_idx=()):
        # RADIO returns (summary, spatial_features); we only want spatial.
        summary, spatial = self.model(x)
        return [spatial]  # single last-block map -> only "sfp" is meaningful


class _TimmViTBackbone(BackboneWrapper):
    """Wraps a plain supervised timm ViT via its own get_intermediate_layers.

    Same per-block feature-extraction contract as _HubDinoBackbone (timm's
    VisionTransformer exposes get_intermediate_layers with an equivalent
    signature -- return_prefix_tokens instead of return_class_token, prefix
    stripped either way when False).
    """

    supports_windowing = False

    def __init__(self, model: nn.Module, spec: BackboneSpec, cfg: EncoderConfig):
        super().__init__()
        self.model = model
        self.patch_size = spec.patch_size
        self.embed_dim = spec.embed_dim
        self.num_blocks = spec.depth
        self.num_prefix = 1 if spec.has_cls else 0  # timm ViT here: CLS only, no registers
        self._norm = cfg.norm_features

    def forward_features(self, x, windowed=False, window=14, global_idx=()):
        if windowed:
            warnings.warn(
                "windowed_attn is only applied to the fallback ViT; the real "
                f"{type(self.model).__name__} runs full attention.",
                stacklevel=2,
            )
        n = list(range(self.num_blocks))
        if hasattr(self.model, "get_intermediate_layers"):
            outs = self.model.get_intermediate_layers(
                x, n=n, reshape=False, return_prefix_tokens=False, norm=self._norm
            )
        else:
            # timm builds the DINOv3 ViTs on its `Eva` class, which has no
            # get_intermediate_layers. forward_intermediates is the equivalent and
            # already drops the prefix (return_prefix_tokens=False by default):
            # verified at 224px / patch 16 it returns 196 = 14x14 tokens, not 201.
            outs = self.model.forward_intermediates(
                x, indices=n, return_prefix_tokens=False, norm=self._norm,
                output_fmt="NLC", intermediates_only=True,
            )
        return list(outs)


def _try_load_timm_vit(spec: BackboneSpec, cfg: EncoderConfig) -> BackboneWrapper:
    import timm

    model = timm.create_model(spec.hub_id, pretrained=cfg.pretrained, dynamic_img_size=True)
    return _TimmViTBackbone(model, spec, cfg)


def _try_load_radio(spec: BackboneSpec, cfg: EncoderConfig) -> BackboneWrapper:
    wdir = cfg.weights_dir("radio")
    if wdir:
        os.environ.setdefault("TORCH_HOME", wdir)
    model = torch.hub.load(
        "NVlabs/RADIO", "radio_model", version=spec.hub_id, progress=True
    )
    return _RadioBackbone(model, spec, cfg)


# --------------------------------------------------------------------------- #
#  Public loader
# --------------------------------------------------------------------------- #
def load_backbone(cfg: EncoderConfig) -> BackboneWrapper:
    spec = get_spec(cfg.backbone)
    if not cfg.pretrained:
        # timm can instantiate the real architecture with random weights, so a
        # deliberate random-init arm gets the genuine model rather than the stub.
        # Without this, `pretrained=False` silently trains FallbackViTBackbone --
        # a different architecture -- and any random-vs-pretrained comparison
        # built on it is measuring the wrong thing.
        # The hub-backed families (dinov2/dinov3/radio) have no offline
        # architecture to build, so they still fall back.
        if spec.kind == "timm_vit":
            return _try_load_timm_vit(spec, cfg)
        return FallbackViTBackbone(spec, cfg)
    loaders = {
        "dinov2": _try_load_dinov2,
        "dinov3": _try_load_dinov3,
        "radio": _try_load_radio,
        "timm_vit": _try_load_timm_vit,
    }
    try:
        return loaders[spec.kind](spec, cfg)
    except Exception as e:  # offline / restricted network -> random-init stub
        warnings.warn(
            f"could not load pretrained {cfg.backbone!r} ({type(e).__name__}: {e}); "
            f"using a random-init fallback ViT with matching dims. Set "
            f"{spec.kind.upper()}_WEIGHTS_DIR or check the network for real weights.",
            stacklevel=2,
        )
        return FallbackViTBackbone(spec, cfg)
