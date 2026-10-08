"""Configuration for the hierarchical encoder.

All knobs live here. No argparse anywhere in the package -- callers construct an
``EncoderConfig`` and pass it to :func:`hier_encoder.build_hierarchical_encoder`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


@dataclass
class EncoderConfig:
    # --- backbone selection -------------------------------------------------
    backbone: str = "dinov3_vitb16"
    """One of the names in ``backbones.list_backbones()`` e.g.
    ``dinov2_vitb14``, ``dinov2_vitb14_reg``, ``dinov3_vitb16``,
    ``radio_v2.5-b`` ..."""
    pretrained: bool = True
    freeze_backbone: bool = True  # default regime: frozen backbone, few-shot

    # --- feature strategy ---------------------------------------------------
    feature_strategy: str = "sfp"  # "sfp" | "multiblock"
    out_strides: Tuple[int, ...] = (4, 8, 16, 32)
    out_channels: Union[int, Tuple[int, ...]] = (64, 128, 256, 512)
    """Per-level channel counts (YOLO-style) or a single int broadcast to all
    levels. Order matches ``sorted(out_strides)`` (ascending)."""

    # blocks to sample for the "multiblock" strategy; ``None`` -> evenly spaced
    multiblock_indexes: Optional[Tuple[int, ...]] = None

    # --- input handling -----------------------------------------------------
    img_size: Optional[Tuple[int, int]] = (1024, 1024)
    """Hint for position-embedding pre-interpolation / stub grids. ``None`` ->
    fully dynamic (interpolate lazily per unseen (H, W))."""

    # --- windowed attention (ViTDet) for high-resolution pages --------------
    windowed_attn: bool = True
    window_size: int = 14  # in *patches*
    global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11)

    # --- PEFT ---------------------------------------------------------------
    lora_rank: int = 0  # 0 disables LoRA; 4 / 8 / 16 for PEFT
    lora_alpha: float = 8.0

    # --- input normalization --------------------------------------------------
    input_norm: str = "auto"
    """Normalization applied to the input image inside the encoder. The encoder
    expects raw [0,1] RGB (what Ultralytics / ToTensor produce).
    "auto"     -> ImageNet mean/std for DINOv2/DINOv3 (how they were pretrained),
                  nothing for RADIO (it conditions its own input).
    "imagenet" -> always apply ImageNet mean/std.
    "none"     -> pass through (use when your dataloader already normalizes)."""

    # --- token layout -------------------------------------------------------
    use_registers: bool = True  # DINOv2-reg / DINOv3 default carry register tokens
    norm_features: bool = True  # apply the backbone final LN to sampled tokens

    # --- optional stride-2 stem (SMP Unet wants 5 skips) --------------------
    include_stride2: bool = False

    # --- hybrid conv-stem branch fused into P2 (stride 4) -------------------
    conv_stem_fusion: bool = False
    """Small conv branch on the raw (normalized) image, concat+1x1-fused into
    the stride-4 pyramid map. Restores sub-patch high-frequency detail for
    thin objects. Always trainable; <0.5M params. Requires 4 in out_strides."""

    # ------------------------------------------------------------------ utils
    @property
    def levels(self) -> Tuple[int, ...]:
        """All output strides in ascending order, incl. stride-2 stem if on."""
        s = set(self.out_strides)
        if self.include_stride2:
            s.add(2)
        return tuple(sorted(s))

    def resolved_channels(self) -> Tuple[int, ...]:
        """Channel count per stride in :pyattr:`levels` order."""
        lv = self.levels
        oc = self.out_channels
        if isinstance(oc, int):
            return tuple(oc for _ in lv)
        oc = tuple(oc)
        if len(oc) == len(lv):
            return oc
        # out_channels given for out_strides only; prepend stride-2 channel.
        if self.include_stride2 and len(oc) == len(self.out_strides):
            return (oc[0],) + oc
        raise ValueError(
            f"out_channels has length {len(oc)} but there are {len(lv)} output "
            f"levels {lv}. Pass an int, one value per level, or one value per "
            f"out_stride (stride-2 channel is then copied from level-0)."
        )

    def weights_dir(self, kind: str) -> Optional[str]:
        """Cluster weight dir from ``{DINOV2,DINOV3,RADIO}_WEIGHTS_DIR`` env."""
        return os.environ.get(f"{kind.upper()}_WEIGHTS_DIR")
