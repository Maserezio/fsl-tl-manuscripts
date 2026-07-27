"""Integration of hier_encoder ViT/foundation backbones into YOLO detection via the SFP neck.

Reuses `hier_encoder.adapters.yolo_adapter` but wraps its `HierEncoderYOLO` so every
pyramid level is snapped to exactly (imgsz//stride) -- see `_ExactStrideHierEncoderYOLO`.
`HIER_VIT_BACKBONES` mirrors `01_simple_segmentation/models/vit_hier.py` so the U-Net
(simple segmentation) and YOLO (2-stage) sides feed the identical set of backbones.

Usage:
    from hier_encoder_yolo import build_trainable_yolo
    from hier_encoder import EncoderConfig

    cfg = EncoderConfig(backbone='dinov2_vits14', pretrained=True,
                        freeze_backbone=True, feature_strategy='sfp')
    model = build_trainable_yolo(cfg, nc=1, scale='s')
    model.train(data=..., epochs=...)
"""
import sys
import tempfile
from pathlib import Path

# hier_encoder lives in the shared 71_misc/ package dir; ensure it's importable even when
# this module is used standalone (not only via train_detector, which also adds the path).
_HIER_PARENT = Path(__file__).resolve().parents[2] / "71_misc"
if str(_HIER_PARENT) not in sys.path:
    sys.path.insert(0, str(_HIER_PARENT))

import torch.nn.functional as F
from hier_encoder.adapters.yolo_adapter import (
    HierEncoderYOLOParse as _HierEncoderYOLOParse,
    make_trainable_yolo_yaml,
    register_hier_yolo_backbone as _register_hier_yolo_backbone,
)

# Mirrors 01_simple_segmentation/models/vit_hier.py HIER_VIT_BACKBONES so the
# transformer backbones fed through hier_encoder's SFP neck are identical on both
# the U-Net (simple segmentation) and YOLO (2-stage detection) sides.
HIER_VIT_BACKBONES = {
    "vit_small_patch16_224.augreg_in21k": "vit_small_patch16_224.augreg_in21k",
    "vit_small_patch16_dinov3": "dinov3_vits16",
    # Native foundation backbones (small variants unless the family only exposes a
    # base model, as for RADIO v2.5).
    "dinov2": "dinov2_vits14",
    "dinov2_reg": "dinov2_vits14_reg",
    "dinov3": "dinov3_vits16",
    "am-radio": "radio_v2.5-b",
}


class _ExactStrideHierEncoderYOLO(_HierEncoderYOLOParse):
    """Force each pyramid level to exactly (H//stride, W//stride).

    hier_encoder's SFP derives level sizes from the ViT patch grid (imgsz/patch), so a
    patch-14 backbone (DINOv2) yields [2g, g, g/2] with g=imgsz/14 -- not a clean
    power-of-2 pyramid unless imgsz is a multiple of 28. Ultralytics' hardcoded 256
    stride-probe (256/14 -> odd grid 19 -> level 9) then breaks the PANet Upsample+Concat
    (19 up-2x = 38 != 9's neighbour). Snapping every level to imgsz//{8,16,32} makes the
    neck align at ANY input size -- construction probe or real batch -- for the price of
    one cheap bilinear resample, standard practice for a ViTDet SFP neck. Patch-16
    backbones (DINOv3/RADIO/augreg ViT) already satisfy this, so the resample is a no-op.

    Module-level (not a closure) so torch.save can pickle checkpoints referencing it.
    """

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = super().forward(x)
        # Build a COHERENT pyramid: pick the coarsest grid n5 = round(imgsz/32), then
        # force P4=2*n5 and P3=4*n5. This guarantees each level is an exact 2x of the
        # next (what PANet's Upsample+Concat requires) for ANY input size -- snapping to
        # plain imgsz//stride would not (e.g. 252: //32=7, //16=15, and 2*7 != 15).
        n5h, n5w = max(1, round(h / 32)), max(1, round(w / 32))
        mult = {8: 4, 16: 2, 32: 1}
        fixed = []
        for o, s in zip(outs, self.strides):
            th, tw = mult[s] * n5h, mult[s] * n5w
            if o.shape[-2:] != (th, tw):
                o = F.interpolate(o, size=(th, tw), mode="bilinear", align_corners=False)
            fixed.append(o)
        return fixed


def register_hier_yolo_backbone(cfg):
    """Bind `cfg` as the active hier_encoder backbone and inject the exact-stride wrapper
    into ultralytics.nn.tasks so parse_model resolves `HierEncoderYOLO` by name."""
    _register_hier_yolo_backbone(cfg)          # sets yolo_adapter._ACTIVE_YOLO_CFG
    import ultralytics.nn.tasks as tasks
    tasks.HierEncoderYOLO = _ExactStrideHierEncoderYOLO
    return _ExactStrideHierEncoderYOLO


def build_trainable_yolo(cfg, nc, scale="s", p2=False):
    """Return a real ultralytics.YOLO whose backbone is the hier_encoder SFP encoder.

    `p2=True` adds a stride-4 detection level for thin objects (manuscript text lines)."""
    from ultralytics import YOLO

    register_hier_yolo_backbone(cfg)
    yaml_str = make_trainable_yolo_yaml(nc, scale, p2)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(yaml_str)
    f.close()
    return YOLO(f.name, task="detect")


__all__ = (
    "build_trainable_yolo",
    "register_hier_yolo_backbone",
    "HIER_VIT_BACKBONES",
)
