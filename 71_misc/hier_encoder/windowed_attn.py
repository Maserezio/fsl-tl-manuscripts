"""ViTDet-style windowed attention helpers (arXiv:2203.16527).

These operate on *token* tensors of shape ``(B, N, C)`` where ``N == Hp * Wp``
(no class / register tokens -- strip those first). Non-overlapping windows are
padded to the window boundary, attention runs inside each window, then the map
is un-partitioned and cropped back.

The partition/un-partition functions are backbone-agnostic. For the bundled
fallback ViT we call these inside the block loop so that blocks *not* in
``global_attn_indexes`` only attend within ``window_size`` x ``window_size``
patch windows -- this is what keeps memory bounded on ~2000x3000 pages.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


def window_partition_tokens(
    tokens: torch.Tensor, hp: int, wp: int, window: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """(B, Hp*Wp, C) -> (B*nW, window*window, C), plus padded (Hpad, Wpad)."""
    b, n, c = tokens.shape
    assert n == hp * wp, (
        f"window_partition: N={n} != Hp*Wp={hp}*{wp}={hp * wp}"
    )
    x = tokens.view(b, hp, wp, c)
    pad_h = (window - hp % window) % window
    pad_w = (window - wp % window) % window
    if pad_h or pad_w:
        # pad on the (Wp, Hp) spatial dims of a channels-last token grid
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hpad, wpad = hp + pad_h, wp + pad_w
    x = rearrange(
        x, "b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c", wh=window, ww=window
    )
    return x, (hpad, wpad)


def window_unpartition_tokens(
    windows: torch.Tensor,
    window: int,
    pad_hw: Tuple[int, int],
    hp: int,
    wp: int,
) -> torch.Tensor:
    """Inverse of :func:`window_partition_tokens`, cropping padding away."""
    hpad, wpad = pad_hw
    nh, nw = hpad // window, wpad // window
    x = rearrange(
        windows,
        "(b nh nw) (wh ww) c -> b (nh wh) (nw ww) c",
        nh=nh,
        nw=nw,
        wh=window,
        ww=window,
    )
    if hpad != hp or wpad != wp:
        x = x[:, :hp, :wp, :].contiguous()
    return x.view(x.shape[0], hp * wp, x.shape[-1])
