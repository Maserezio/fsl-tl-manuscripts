"""Minimal, self-contained LoRA adapters for ViT attention.

No ``peft`` dependency. We inject low-rank updates into the *query* and *value*
projections of every attention block. Both the fused (``qkv``) and the split
(``q_proj`` / ``v_proj``) layouts are supported. LoRA parameters stay trainable
even when the backbone is frozen.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` and adds ``scaling * B(A(x))``."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        assert rank > 0
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)  # start as identity (no-op)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, self.A), self.B)
        return out + self.scaling * delta


class LoRAQKV(nn.Module):
    """LoRA on the q and v slices of a fused ``qkv`` linear (out = 3 * dim)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        assert base.out_features % 3 == 0, "fused qkv must have out=3*dim"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.dim = base.out_features // 3
        self.rank = rank
        self.scaling = alpha / rank
        self.Aq = nn.Parameter(torch.zeros(rank, base.in_features))
        self.Bq = nn.Parameter(torch.zeros(self.dim, rank))
        self.Av = nn.Parameter(torch.zeros(rank, base.in_features))
        self.Bv = nn.Parameter(torch.zeros(self.dim, rank))
        nn.init.kaiming_uniform_(self.Aq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Av, a=math.sqrt(5))
        nn.init.zeros_(self.Bq)
        nn.init.zeros_(self.Bv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.base(x)
        dq = torch.nn.functional.linear(torch.nn.functional.linear(x, self.Aq), self.Bq)
        dv = torch.nn.functional.linear(torch.nn.functional.linear(x, self.Av), self.Bv)
        # build an additive tensor over the full 3*dim width (no in-place)
        pad = torch.zeros_like(qkv[..., : self.dim])
        delta = torch.cat([dq, pad, dv], dim=-1)
        return qkv + self.scaling * delta


def inject_lora(module: nn.Module, rank: int, alpha: float) -> int:
    """Recursively replace attention q/v projections with LoRA-wrapped ones.

    Returns the number of injected adapters. Safe to call on frozen backbones;
    the wrapped base stays frozen while the LoRA factors are trainable.
    """
    if rank <= 0:
        return 0
    n = 0
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if name == "qkv" and child.out_features == 3 * child.in_features:
                setattr(parent, name, LoRAQKV(child, rank, alpha))
                n += 1
            elif name in ("q_proj", "v_proj", "query", "value"):
                setattr(parent, name, LoRALinear(child, rank, alpha))
                n += 1
    return n


def lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    ps: List[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, (LoRALinear, LoRAQKV)):
            for name, p in m.named_parameters(recurse=False):
                if name != "base":
                    p.requires_grad_(True)
                    ps.append(p)
    return ps
