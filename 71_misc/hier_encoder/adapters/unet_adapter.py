"""SMP-compatible encoder wrapper.

``segmentation_models_pytorch`` expects an encoder that:
  * exposes ``_out_channels`` (input, stride2, stride4, ...), ``_depth``,
    ``_in_channels``;
  * returns a list ``[x, s2, s4, s8, s16, s32]`` from ``forward`` where ``x`` is
    the raw input tensor (the Unet decoder ignores it but the interface wants
    it).

We force ``include_stride2=True`` at registration so the Unet decoder gets its 5
skip connections; if the user asks for ``encoder_depth=4`` we drop the stem.

``smp`` is imported lazily so the core library works without it.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import List

import torch
import torch.nn as nn

from ..build import build_hierarchical_encoder
from ..config import EncoderConfig


class HierSMPEncoder(nn.Module):
    def __init__(
        self,
        cfg: EncoderConfig,
        depth: int = 5,
        output_stride: int = 32,
        in_channels: int = 3,
        **kwargs,
    ):
        super().__init__()
        if output_stride != 32:
            raise NotImplementedError(
                "HierSMPEncoder only supports output_stride=32 (no dilation)."
            )
        cfg = replace(cfg, include_stride2=(depth == 5))
        # SMP Unet default wants strides 4,8,16,32 (+2 for depth 5)
        cfg = replace(cfg, out_strides=(4, 8, 16, 32))
        self.encoder = build_hierarchical_encoder(cfg)
        self._depth = depth
        self._in_channels = in_channels

        ch = dict(zip(self.encoder.strides, self.encoder.out_channels))
        if depth == 5:
            self._out_channels = (in_channels, ch[2], ch[4], ch[8], ch[16], ch[32])
            self._ordered = [2, 4, 8, 16, 32]
        else:
            self._out_channels = (in_channels, ch[4], ch[8], ch[16], ch[32])
            self._ordered = [4, 8, 16, 32]

        self.output_stride = 32

    # SMP-required attributes / hooks -----------------------------------------
    @property
    def out_channels(self):
        return self._out_channels

    def set_in_channels(self, in_channels: int, pretrained: bool = True) -> None:
        if in_channels != 3:
            raise NotImplementedError(
                "HierSMPEncoder only supports in_channels=3 (RGB foundation models)."
            )

    def make_dilated(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Dilated / output_stride<32 is not supported by the ViT-based encoder."
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.encoder(x)
        return [x] + [feats[s] for s in self._ordered]

    def load_state_dict(self, *args, **kwargs):  # tolerate encoder_weights=None
        return super().load_state_dict(*args, **kwargs)


def register_smp_encoder(name: str, cfg: EncoderConfig, depth: int = 5) -> None:
    """Register ``name`` in SMP's encoder registry (side-effect).

    After this, ``smp.Unet(encoder_name=name, encoder_weights=None, ...)`` builds
    a Unet whose encoder is the hierarchical foundation-model encoder.
    """
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import encoders

    frozen_cfg = copy.deepcopy(cfg)

    encoders[name] = {
        "encoder": HierSMPEncoder,
        "pretrained_settings": {},
        "params": {"cfg": frozen_cfg, "depth": depth},
    }
