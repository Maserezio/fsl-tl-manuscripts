"""ViTDet Simple Feature Pyramid (arXiv:2203.16527).

Takes a *single* last-block feature map at ``stride == patch_size`` (nominally
labelled "16") and builds a pyramid by up/down-sampling with transposed convs
and max-pools. Stride labels are nominal: a patch-14 backbone produces H/14 at
the "16" level, not H/16 -- documented, and the caller can crop with the padding
info returned by the encoder.

Ladder relative to the source stride (16):
    /4  = ConvTranspose2d(x2) -> GN -> GELU -> ConvTranspose2d(x2)
    /8  = ConvTranspose2d(x2) -> GN -> GELU
    /16 = identity (1x1 proj)
    /32 = MaxPool2d(2) -> 1x1 proj
Each level then applies 3x3 conv -> LayerNorm2d -> 1x1 conv to out_channels[i].
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm over a (B, C, H, W) tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


def _groupnorm(dim: int) -> nn.GroupNorm:
    groups = 32
    while dim % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, dim)


def build_resampler(dim: int, factor: float) -> nn.Sequential:
    """Resample a (B, dim, H, W) map by ``factor`` (2, 4 up; 0.5, 0.25 down; 1 id).

    Channel count is preserved; a following head projects to the target width.
    """
    layers: List[nn.Module] = []
    if factor > 1:
        n = int(round(math.log2(factor)))
        for i in range(n):
            layers.append(nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2))
            if i < n - 1:
                layers += [_groupnorm(dim), nn.GELU()]
    elif factor < 1:
        m = int(round(math.log2(1.0 / factor)))
        for _ in range(m):
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.Conv2d(dim, dim, kernel_size=1))  # 1x1 proj
    else:
        layers.append(nn.Conv2d(dim, dim, kernel_size=1))  # identity 1x1 proj
    return nn.Sequential(*layers)


class _Head(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = LayerNorm2d(dim)
        self.proj = nn.Conv2d(dim, out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(self.conv(x)))


class SimpleFeaturePyramid(nn.Module):
    """Single-source SFP: one map (stride = source_stride) -> multi-scale pyramid."""

    def __init__(
        self,
        embed_dim: int,
        source_stride: int,
        out_strides: Tuple[int, ...],
        out_channels: Tuple[int, ...],
    ):
        super().__init__()
        assert len(out_strides) == len(out_channels)
        self.out_strides = tuple(out_strides)
        self.resamplers = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for s, c in zip(out_strides, out_channels):
            factor = source_stride / s  # >1 upsample, <1 downsample
            self.resamplers[str(s)] = build_resampler(embed_dim, factor)
            self.heads[str(s)] = _Head(embed_dim, c)

    def forward(self, feat: torch.Tensor) -> "Dict[int, torch.Tensor]":
        out: Dict[int, torch.Tensor] = {}
        for s in self.out_strides:
            y = self.resamplers[str(s)](feat)
            out[s] = self.heads[str(s)](y)
        return out
