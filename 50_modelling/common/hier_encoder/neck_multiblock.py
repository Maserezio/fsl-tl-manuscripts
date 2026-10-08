"""Multi-depth token sampling neck.

Instead of building the whole pyramid from the last block (SFP), we sample four
evenly-spaced blocks (default [N/4, N/2, 3N/4, N-1]), reshape each to a spatial
map, and route each source to a target stride with the same up/down resamplers
as the SFP. This mirrors how DINOv2's own segmentor consumes multi-scale
readouts -- earlier blocks feed the high-resolution levels, later blocks the
coarse ones.

All source maps arrive at ``source_stride`` (== patch size); the resampler for
each target stride handles the rescale. One source is consumed per target
stride, so ``len(block_indexes) == len(out_strides)``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .neck_sfp import _Head, build_resampler


class MultiBlockPyramid(nn.Module):
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
        self.resamplers = nn.ModuleList()
        self.heads = nn.ModuleList()
        for s, c in zip(out_strides, out_channels):
            factor = source_stride / s
            self.resamplers.append(build_resampler(embed_dim, factor))
            self.heads.append(_Head(embed_dim, c))

    def forward(self, feats: List[torch.Tensor]) -> "Dict[int, torch.Tensor]":
        assert len(feats) == len(self.out_strides), (
            f"multiblock expects {len(self.out_strides)} source maps, "
            f"got {len(feats)}"
        )
        out: Dict[int, torch.Tensor] = {}
        for i, s in enumerate(self.out_strides):
            y = self.resamplers[i](feats[i])
            out[s] = self.heads[i](y)
        return out


def default_block_indexes(num_blocks: int, k: int) -> Tuple[int, ...]:
    """k block indexes at [N/k, 2N/k, ...] with the last pinned to block N-1.

    For ViT-B (12 blocks) and k=4 this is (3, 6, 9, 11): P2<-3, P3<-6, P4<-9,
    P5<-11 -- earlier blocks feed the higher-resolution levels.
    """
    if k == 1:
        return (num_blocks - 1,)
    idx = [num_blocks * (i + 1) // k for i in range(k - 1)] + [num_blocks - 1]
    # ensure strictly increasing & valid
    idx = sorted(set(max(0, min(num_blocks - 1, j)) for j in idx))
    while len(idx) < k:  # pad if collisions collapsed entries
        idx.append(min(num_blocks - 1, idx[-1] + 1))
        idx = sorted(set(idx))
    return tuple(idx[:k])
