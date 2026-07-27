"""Hybrid conv-stem branch for the P2 (stride-4) level.

ViT patch embeddings destroy sub-patch detail; a text line ~2 patches tall
leaves little high-frequency signal in the tokens. This small conv branch runs
on the RAW input image (after the encoder's input normalization) and is fused
into the P2 pyramid map, restoring stroke-level detail on a cheap path.

The stem is a fresh module and stays trainable regardless of backbone freezing.
Total parameters stay well under 0.5M (printed at build time).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .neck_sfp import _groupnorm


class ConvStemP2(nn.Module):
    """Conv3x3 s2 (32) -> GN -> GELU -> Conv3x3 s2 (64) -> GN -> GELU ->
    Conv3x3 s1 (out_channels). Output at stride 4 of the input image."""

    def __init__(self, out_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            _groupnorm(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            _groupnorm(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ConvStemFusion(nn.Module):
    """Fuse the conv-stem map into the pyramid P2 map.

    Concat + 1x1 conv (not add -- the two branches have different channel
    statistics). If spatial sizes differ (patch-14 backbones produce H*4/14 at
    the nominal "4" level, the stem produces H/4), the stem map is bilinearly
    resized to the pyramid grid.
    """

    def __init__(self, p2_channels: int, stem_channels: int):
        super().__init__()
        self.stem = ConvStemP2(stem_channels)
        self.fuse = nn.Conv2d(p2_channels + stem_channels, p2_channels, 1)
        n = sum(p.numel() for p in self.parameters())
        assert n < 500_000, f"conv-stem branch too large: {n} params"
        print(f"[hier_encoder] conv-stem P2 branch: {n / 1e3:.1f}K params (trainable)")

    def forward(self, image: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        s = self.stem(image)
        if s.shape[-2:] != p2.shape[-2:]:
            s = F.interpolate(s, size=p2.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.fuse(torch.cat([p2, s], dim=1))
