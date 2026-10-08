"""``build_hierarchical_encoder(cfg) -> HierarchicalEncoder``.

Turns a vision foundation backbone into a hierarchical feature encoder whose
output is a stride-keyed ``OrderedDict`` of channels-first maps -- drop-in for a
U-Net decoder or YOLOv8's PANet neck.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .backbones import BackboneWrapper, get_spec, load_backbone
from .config import EncoderConfig
from .lora import inject_lora, lora_parameters
from .neck_convstem import ConvStemFusion
from .neck_multiblock import MultiBlockPyramid, default_block_indexes
from .neck_sfp import SimpleFeaturePyramid, _groupnorm


class Stride2Stem(nn.Module):
    """Fresh learnable stem producing a stride-2 feature from the raw image."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1),
            _groupnorm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            _groupnorm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class HierarchicalEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone: BackboneWrapper = load_backbone(cfg)
        self.patch_size = self.backbone.patch_size
        self._levels = cfg.levels  # ascending incl. 2 if include_stride2

        # input normalization: DINO models were pretrained on ImageNet-normalized
        # input; the encoder itself expects raw [0,1] RGB and normalizes here.
        norm = cfg.input_norm
        if norm == "auto":
            norm = "imagenet" if get_spec(cfg.backbone).kind in ("dinov2", "dinov3", "timm_vit") else "none"
        if norm == "imagenet":
            self.register_buffer(
                "_in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "_in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
        elif norm == "none":
            self._in_mean = None
            self._in_std = None
        else:
            raise ValueError(f"input_norm must be 'auto'|'imagenet'|'none', got {norm!r}")
        self._channels = cfg.resolved_channels()

        # strides handled by the neck (everything except the stride-2 stem)
        self._neck_strides = tuple(s for s in self._levels if s != 2 or not cfg.include_stride2)
        neck_channels = tuple(
            self._channels[self._levels.index(s)] for s in self._neck_strides
        )

        # --- freeze / PEFT on the backbone ---------------------------------
        if cfg.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self._lora = 0
        if cfg.lora_rank > 0:
            self._lora = inject_lora(self.backbone, cfg.lora_rank, cfg.lora_alpha)
            lora_parameters(self.backbone)  # ensure trainable

        # --- windowing ------------------------------------------------------
        self._windowed = cfg.windowed_attn and self.backbone.supports_windowing

        # --- neck -----------------------------------------------------------
        src_stride = self.patch_size
        if cfg.feature_strategy == "sfp":
            self.neck = SimpleFeaturePyramid(
                self.backbone.embed_dim, src_stride, self._neck_strides, neck_channels
            )
            self._block_idx: Optional[Tuple[int, ...]] = None
        elif cfg.feature_strategy == "multiblock":
            k = len(self._neck_strides)
            self._block_idx = (
                tuple(cfg.multiblock_indexes)
                if cfg.multiblock_indexes is not None
                else default_block_indexes(self.backbone.num_blocks, k)
            )
            if len(self._block_idx) != k:
                raise ValueError(
                    f"multiblock_indexes must have {k} entries (one per neck "
                    f"stride), got {self._block_idx}"
                )
            self.neck = MultiBlockPyramid(
                self.backbone.embed_dim, src_stride, self._neck_strides, neck_channels
            )
        else:
            raise ValueError(
                f"feature_strategy must be 'sfp' or 'multiblock', got "
                f"{cfg.feature_strategy!r}"
            )

        # --- optional stride-2 stem ----------------------------------------
        self.stem: Optional[Stride2Stem] = None
        if cfg.include_stride2:
            self.stem = Stride2Stem(self._channels[self._levels.index(2)])

        # --- optional conv-stem branch fused into P2 (stride 4) -------------
        self.conv_stem: Optional[ConvStemFusion] = None
        if cfg.conv_stem_fusion:
            if 4 not in self._neck_strides:
                raise ValueError(
                    "conv_stem_fusion=True requires stride 4 in out_strides, "
                    f"got {self._neck_strides}"
                )
            p2_ch = self._channels[self._levels.index(4)]
            self.conv_stem = ConvStemFusion(p2_ch, stem_channels=p2_ch)

        self.last_padding: Tuple[int, int] = (0, 0)  # (pad_h, pad_w)

    # -------------------------------------------------------------- properties
    @property
    def strides(self) -> Tuple[int, ...]:
        return self._levels

    @property
    def out_channels(self) -> Tuple[int, ...]:
        return self._channels

    @property
    def num_features(self) -> int:
        return len(self._levels)

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    # -------------------------------------------------------------- internals
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        self.last_padding = (ph, pw)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x

    def _tokens_to_map(self, tokens: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        b, n, c = tokens.shape
        assert n == hp * wp, (
            f"token/grid mismatch: N={n} but Hp*Wp={hp}*{wp}={hp * wp}. "
            f"patch_size={self.patch_size}. Check padding to a patch multiple."
        )
        return rearrange(tokens, "b (hp wp) c -> b c hp wp", hp=hp, wp=wp)

    # --------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> "OrderedDict[int, torch.Tensor]":
        # getattr defaults: stay loadable for checkpoints pickled before these
        # attributes existed (unpickling skips __init__).
        in_mean = getattr(self, "_in_mean", None)
        if in_mean is not None:
            x = (x - in_mean) / self._in_std
        xp = self._pad(x)
        hp = xp.shape[-2] // self.patch_size
        wp = xp.shape[-1] // self.patch_size

        feats = self.backbone.forward_features(
            xp,
            windowed=self._windowed,
            window=self.cfg.window_size,
            global_idx=self.cfg.global_attn_indexes,
        )

        if self.cfg.feature_strategy == "sfp":
            src = self._tokens_to_map(feats[-1], hp, wp)
            pyr = self.neck(src)
        else:
            maps = [self._tokens_to_map(feats[i], hp, wp) for i in self._block_idx]
            pyr = self.neck(maps)

        conv_stem = getattr(self, "conv_stem", None)
        if conv_stem is not None:
            pyr[4] = conv_stem(xp, pyr[4])  # xp: normalized + padded image

        out: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        for s in self._levels:
            if s == 2 and self.stem is not None:
                out[2] = self.stem(xp)
            else:
                out[s] = pyr[s]
        return out


def build_hierarchical_encoder(cfg: EncoderConfig) -> HierarchicalEncoder:
    return HierarchicalEncoder(cfg)
