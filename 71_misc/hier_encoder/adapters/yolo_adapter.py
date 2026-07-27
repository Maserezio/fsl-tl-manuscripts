"""Ultralytics-compatible backbone wrapper.

APPROACH (documented choice): Ultralytics' ``parse_model`` tracks a *single*
output-channel count per layer and assumes each layer emits one tensor, so a
single custom backbone module that emits three maps (P3/P4/P5) does not slot
into the YAML/``parse_model`` pipeline cleanly (channel bookkeeping + list
outputs break). We therefore build the detection model **programmatically**:
we keep ``HierEncoderYOLO`` as the backbone and assemble YOLOv8's stock PANet
neck + ``Detect`` head from Ultralytics' own building blocks (``C2f``, ``Conv``,
``Concat``, ``Detect``) with channels we compute ourselves. This is robust
across Ultralytics versions -- it only depends on those module classes existing.

``make_yolo_yaml`` is kept as a documented reference of the equivalent YAML.
``ultralytics`` is imported lazily so the core library works without it.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

import torch
import torch.nn as nn

from ..build import build_hierarchical_encoder
from ..config import EncoderConfig

# YOLOv8 scale -> (P3, P4, P5) channels its neck consumes.
YOLOV8_NECK_CHANNELS = {
    "n": (64, 128, 256),
    "s": (128, 256, 512),
    "m": (192, 384, 576),
    "l": (256, 512, 512),
    "x": (320, 640, 640),
}
# number of C2f repeats per neck stage for each scale (depth multiple).
_C2F_N = {"n": 1, "s": 1, "m": 2, "l": 3, "x": 3}


class HierEncoderYOLO(nn.Module):
    """Custom Ultralytics backbone: returns one map per stride (default P3-P5).

    Pass ``strides=(4, 8, 16, 32)`` for a P2 variant -- thin objects like
    manuscript text lines (~35 px tall) need the stride-4 level.
    """

    def __init__(
        self,
        cfg: EncoderConfig,
        neck_channels: Tuple[int, ...],
        strides: Tuple[int, ...] = (8, 16, 32),
    ):
        super().__init__()
        assert len(neck_channels) == len(strides)
        # encoder emits a uniform width; proj re-projects each level to the
        # channel count the YOLOv8 neck expects for this scale.
        cfg = replace(
            cfg, include_stride2=False, out_strides=tuple(strides), out_channels=256
        )
        self.encoder = build_hierarchical_encoder(cfg)
        self.strides = tuple(strides)
        self.neck_channels = tuple(neck_channels)
        enc_ch = dict(zip(self.encoder.strides, self.encoder.out_channels))
        self.proj = nn.ModuleList(
            [nn.Conv2d(enc_ch[s], nc, 1) for s, nc in zip(strides, neck_channels)]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.encoder(x)
        return [self.proj[i](feats[s]) for i, s in enumerate(self.strides)]


class HierYOLOv8Detection(nn.Module):
    """HierEncoderYOLO backbone + stock YOLOv8 PANet neck + Detect head.

    In ``train()`` mode ``forward`` returns the list of three raw Detect feature
    maps (shape ``(B, 4*reg_max + nc, H_i, W_i)``) -- exactly what YOLOv8's loss
    consumes -- which is what the example prints.
    """

    def __init__(self, cfg: EncoderConfig, nc: int, scale: str = "s"):
        super().__init__()
        from ultralytics.nn.modules import C2f, Concat, Conv, Detect  # lazy

        c3, c4, c5 = YOLOV8_NECK_CHANNELS[scale]
        n = _C2F_N[scale]

        self.backbone = HierEncoderYOLO(cfg, (c3, c4, c5))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat = Concat(1)

        # top-down
        self.c2f_p4 = C2f(c5 + c4, c4, n, shortcut=False)   # after up(P5)+P4
        self.c2f_p3 = C2f(c4 + c3, c3, n, shortcut=False)   # after up(->)+P3  => N3
        # bottom-up
        self.down_p3 = Conv(c3, c3, 3, 2)
        self.c2f_n4 = C2f(c3 + c4, c4, n, shortcut=False)   # => N4
        self.down_p4 = Conv(c4, c4, 3, 2)
        self.c2f_n5 = C2f(c4 + c5, c5, n, shortcut=False)   # => N5

        self.detect = Detect(nc, ch=(c3, c4, c5))
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.detect.training = True  # return raw maps from forward

    def forward(self, x: torch.Tensor):
        p3, p4, p5 = self.backbone(x)
        t4 = self.c2f_p4(self.concat([self.up(p5), p4]))
        n3 = self.c2f_p3(self.concat([self.up(t4), p3]))
        n4 = self.c2f_n4(self.concat([self.down_p3(n3), t4]))
        n5 = self.c2f_n5(self.concat([self.down_p4(n4), p5]))
        return self.detect([n3, n4, n5])


def make_yolo_yaml(nc: int, scale: str = "s") -> str:
    """Reference YAML equivalent of the programmatic model (documentation)."""
    c3, c4, c5 = YOLOV8_NECK_CHANNELS[scale]
    return f"""# Reference only -- we build this programmatically (see module docstring).
nc: {nc}
backbone:
  - [-1, 1, HierEncoderYOLO, ['{scale}']]   # returns P3/P4/P5
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 'P4'], 1, Concat, [1]]
  - [-1, 3, C2f, [{c4}]]
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 'P3'], 1, Concat, [1]]
  - [-1, 3, C2f, [{c3}]]
  - [-1, 1, Conv, [{c3}, 3, 2]]
  - [[-1, 't4'], 1, Concat, [1]]
  - [-1, 3, C2f, [{c4}]]
  - [-1, 1, Conv, [{c4}, 3, 2]]
  - [[-1, 'P5'], 1, Concat, [1]]
  - [-1, 3, C2f, [{c5}]]
  - [['N3', 'N4', 'N5'], 1, Detect, [nc]]
"""


def build_yolo_detection_model(
    cfg: EncoderConfig, nc: int, scale: str = "s"
) -> HierYOLOv8Detection:
    """Build a YOLOv8-``scale``-shaped detection model with the hier backbone.

    Standalone module for quick forward tests. For *training* with the full
    Ultralytics pipeline (augmentation, logging, val), use
    :func:`build_trainable_yolo`, which returns a real ``ultralytics.YOLO``.
    """
    return HierYOLOv8Detection(cfg, nc, scale)


# --------------------------------------------------------------------------- #
#  Trainable path: register HierEncoderYOLO into Ultralytics + build a YOLO()
#  via a YAML that uses the stock TorchVision/Index multi-output pattern, so the
#  standard `.train()` trainer (data aug, EMA, val, logging) can be used.
# --------------------------------------------------------------------------- #
# Module-level (picklable) parse-friendly wrapper. It reads the active config
# from a module global so ``parse_model`` can call it as ``HierEncoderYOLO(scale)``
# while the checkpoint can still be pickled (a local/closure class cannot).
_ACTIVE_YOLO_CFG: "EncoderConfig | None" = None


def _p2_channels(scale: str) -> Tuple[int, int, int, int]:
    c3, c4, c5 = YOLOV8_NECK_CHANNELS[scale]
    return (c3 // 2, c3, c4, c5)


class HierEncoderYOLOParse(HierEncoderYOLO):
    def __init__(self, scale: str = "s", p2: bool = False):
        if _ACTIVE_YOLO_CFG is None:
            raise RuntimeError(
                "call register_hier_yolo_backbone(cfg) before building the YOLO model"
            )
        # YAML args pass scale as 'sc_<letter>': parse_model eval's bare strings
        # against its own locals, so 'n'/'m' would silently become ints there.
        if isinstance(scale, str) and scale.startswith("sc_"):
            scale = scale[3:]
        if scale not in YOLOV8_NECK_CHANNELS:
            raise ValueError(f"bad scale {scale!r}; use one of {list(YOLOV8_NECK_CHANNELS)}")
        if p2:
            super().__init__(_ACTIVE_YOLO_CFG, _p2_channels(scale), (4, 8, 16, 32))
        else:
            super().__init__(_ACTIVE_YOLO_CFG, YOLOV8_NECK_CHANNELS[scale])


def register_hier_yolo_backbone(cfg: EncoderConfig):
    """Inject the parse-friendly ``HierEncoderYOLO`` (returns [P3,P4,P5]) into the
    ``ultralytics.nn.tasks`` namespace so ``parse_model`` can resolve it by name.
    Binds ``cfg`` as the active config; the YAML passes only the scale letter."""
    import ultralytics.nn.tasks as tasks

    global _ACTIVE_YOLO_CFG
    _ACTIVE_YOLO_CFG = cfg
    tasks.HierEncoderYOLO = HierEncoderYOLOParse
    return HierEncoderYOLOParse


def make_trainable_yolo_yaml(nc: int, scale: str = "s", p2: bool = False) -> str:
    """YOLOv8 detection YAML: HierEncoderYOLO -> Index -> stock PAN + Detect.

    ``p2=True`` adds a stride-4 detection level (thin objects, e.g. text lines)
    -- same recipe as the proven yolov8-resnet34-p2 config.
    """
    c3, c4, c5 = YOLOV8_NECK_CHANNELS[scale]
    n = _C2F_N[scale]
    if p2:
        c2 = c3 // 2
        return f"""# HierEncoderYOLO backbone (DINO/RADIO) + YOLOv8 P2/P3/P4/P5 head (thin objects).
nc: {nc}
depth_multiple: 1.0
width_multiple: 1.0

backbone:
  - [-1, 1, HierEncoderYOLO, ['sc_{scale}', True]]  # 0  returns [P2, P3, P4, P5]
  - [0, 1, Index, [{c2}, 0]]                     # 1  P2/4
  - [0, 1, Index, [{c3}, 1]]                     # 2  P3/8
  - [0, 1, Index, [{c4}, 2]]                     # 3  P4/16
  - [0, 1, Index, [{c5}, 3]]                     # 4  P5/32
  - [4, 1, SPPF, [{c5}, 5]]                      # 5  SPPF(P5)

head:
  - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 6
  - [[-1, 3], 1, Concat, [1]]                 # 7   cat P4
  - [-1, {n}, C2f, [{c4}]]                     # 8
  - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 9
  - [[-1, 2], 1, Concat, [1]]                 # 10  cat P3
  - [-1, {n}, C2f, [{c3}]]                     # 11
  - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 12
  - [[-1, 1], 1, Concat, [1]]                 # 13  cat P2
  - [-1, {n}, C2f, [{c2}]]                     # 14  (P2/4 out)
  - [-1, 1, Conv, [{c2}, 3, 2]]               # 15
  - [[-1, 11], 1, Concat, [1]]                # 16
  - [-1, {n}, C2f, [{c3}]]                     # 17  (P3/8 out)
  - [-1, 1, Conv, [{c3}, 3, 2]]               # 18
  - [[-1, 8], 1, Concat, [1]]                 # 19
  - [-1, {n}, C2f, [{c4}]]                     # 20  (P4/16 out)
  - [-1, 1, Conv, [{c4}, 3, 2]]               # 21
  - [[-1, 5], 1, Concat, [1]]                 # 22  cat P5(SPPF)
  - [-1, {n}, C2f, [{c5}]]                     # 23  (P5/32 out)
  - [[14, 17, 20, 23], 1, Detect, [nc]]       # 24  Detect(P2, P3, P4, P5)
"""
    return f"""# HierEncoderYOLO backbone (DINO/RADIO) + stock YOLOv8 P3/P4/P5 head.
nc: {nc}
depth_multiple: 1.0
width_multiple: 1.0

backbone:
  - [-1, 1, HierEncoderYOLO, ['sc_{scale}']]  # 0  returns [P3, P4, P5]
  - [0, 1, Index, [{c3}, 0]]               # 1  P3/8
  - [0, 1, Index, [{c4}, 1]]               # 2  P4/16
  - [0, 1, Index, [{c5}, 2]]               # 3  P5/32
  - [3, 1, SPPF, [{c5}, 5]]                # 4  SPPF(P5)

head:
  - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 5
  - [[-1, 2], 1, Concat, [1]]                 # 6  cat P4
  - [-1, {n}, C2f, [{c4}]]                     # 7
  - [-1, 1, nn.Upsample, [None, 2, nearest]]  # 8
  - [[-1, 1], 1, Concat, [1]]                 # 9  cat P3
  - [-1, {n}, C2f, [{c3}]]                     # 10 (P3 out)
  - [-1, 1, Conv, [{c3}, 3, 2]]               # 11
  - [[-1, 7], 1, Concat, [1]]                 # 12 cat head-P4
  - [-1, {n}, C2f, [{c4}]]                     # 13 (P4 out)
  - [-1, 1, Conv, [{c4}, 3, 2]]               # 14
  - [[-1, 4], 1, Concat, [1]]                 # 15 cat P5(SPPF)
  - [-1, {n}, C2f, [{c5}]]                     # 16 (P5 out)
  - [[10, 13, 16], 1, Detect, [nc]]           # 17 Detect(P3, P4, P5)
"""


def build_trainable_yolo(cfg: EncoderConfig, nc: int, scale: str = "s", p2: bool = False):
    """Return a real ``ultralytics.YOLO`` whose backbone is the hier encoder.

    ``p2=True`` adds a stride-4 detection level for thin objects (text lines).

    Usage::
        model = build_trainable_yolo(cfg, nc=1, scale='s', p2=True)
        model.train(data='diva.yaml', epochs=100, imgsz=1568)
    """
    import tempfile

    from ultralytics import YOLO

    register_hier_yolo_backbone(cfg)
    yaml_str = make_trainable_yolo_yaml(nc, scale, p2)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(yaml_str)
    f.close()
    return YOLO(f.name, task="detect")
