"""Train one 2-stage detector from the unified backbone matrix, with a fixed recipe.

Mirrors 01_simple_segmentation's `--encoder <name>` convention: a single BACKBONES
registry maps each encoder name to how its model is built, so the detection matrix and
the U-Net segmentation matrix train exactly the same backbones.

Two routing mechanisms, identical to 01_simple_segmentation/models/vit_hier.py:
  * hierarchical CNNs (ResNet, ConvNeXt) -> a stock YOLOv8 detection YAML in
    configs/detection/ whose native [P3,P4,P5] pyramid feeds the PANet neck.
  * transformer backbones (supervised ViT, DINOv2/v3, AM-RADIO) -> hier_encoder's
    SFP ("Simple Feature Pyramid", ViTDet) neck, built via
    hier_encoder_yolo.build_trainable_yolo(EncoderConfig(..., feature_strategy="sfp")).

An escape hatch remains for stock checkpoints / RT-DETR via --model (e.g. a yolov8n.pt
baseline). Exactly one of --encoder / --model must be given.
"""
import argparse
import sys
import tempfile
from pathlib import Path

# `hier_encoder` is the shared SFP package under 71_misc/. Put 71_misc on sys.path so
# this module imports from any cwd / Jupyter kernel / Colab even without the editable
# install active in that interpreter.
_HIER_PARENT = Path(__file__).resolve().parents[2] / "71_misc"
if str(_HIER_PARENT) not in sys.path:
    sys.path.insert(0, str(_HIER_PARENT))
# This module's own directory too, so `import timm_backbone` / `hier_encoder_yolo` work
# when imported (not just when run as a script from here).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import yaml
from ultralytics import YOLO, RTDETR
from ultralytics import settings as _ul_settings
from hier_encoder import EncoderConfig

import timm_backbone
import hier_encoder_yolo

# Ultralytics ships an MLflow logging callback that fires whenever `mlflow` is importable;
# recent MLflow hard-errors on its file-store backend ("maintenance mode"), crashing
# training at on_pretrain_routine_end. We don't use MLflow tracking here, so disable that
# (and other noisy loggers) once, centrally -- covers the notebooks and run_train_all.sh.
try:
    _ul_settings.update({"mlflow": False})
except Exception:
    pass

HERE = Path(__file__).resolve().parent
_DET = HERE / "configs" / "detection"

# name -> ("yolo", ckpt)          stock YOLOv8 CSPDarknet baseline (COCO-pretrained)
#      -> ("yaml", path)          hierarchical CNN, native pyramid
#      -> ("sfp",  hier_backbone) transformer via hier_encoder SFP neck
# The transformer entries reuse hier_encoder_yolo.HIER_VIT_BACKBONES (which itself
# mirrors 01_simple_segmentation's vit_hier.py) so the two matrices stay in lockstep.
BACKBONES = {
    "yolov8n":                             ("yolo", "yolov8n.pt"),
    "resnet34":                            ("yaml", str(_DET / "yolov8-resnet34.yaml")),
    "resnet50":                            ("yaml", str(_DET / "yolov8-resnet50.yaml")),
    "convnext":                            ("yaml", str(_DET / "yolov8-convnext.yaml")),
    "convnext_dinov3":                     ("yaml", str(_DET / "yolov8-convnext-dinov3.yaml")),
    "pvt_v2":                              ("yaml", str(_DET / "yolov8-pvt_v2.yaml")),
    "vit_small_patch16_224.augreg_in21k":  ("sfp",  "vit_small_patch16_224.augreg_in21k"),
    "vit_small_patch16_dinov3":            ("sfp",  "dinov3_vits16"),
    "dinov2":                              ("sfp",  "dinov2_vits14"),
    "dinov2_reg":                          ("sfp",  "dinov2_vits14_reg"),
    "am-radio":                            ("sfp",  "radio_v2.5-b"),
}


def resolve_data(data_cfg) -> str:
    """Return a data YAML whose `path:` is absolute, resolved against the CONFIG FILE.

    Ultralytics resolves a relative `path:` against its `datasets_dir` setting (e.g.
    ~/Thesis/datasets), NOT the YAML's own location -- so `../../../../00_data/...`
    (correct relative to configs/data/) climbs to `/00_data` and the split is "not found".
    We resolve `path` relative to the config file, make it absolute, and write a temp
    copy. Absolute paths already pass through unchanged. Portable across local + Colab
    (both compute the absolute path from wherever the repo actually sits)."""
    data_cfg = Path(data_cfg)
    d = yaml.safe_load(open(data_cfg))
    p = Path(d.get("path", "."))
    if not p.is_absolute():
        p = (data_cfg.parent / p).resolve()
    d["path"] = str(p)
    f = tempfile.NamedTemporaryFile("w", suffix=f"_{data_cfg.stem}.yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    return f.name


def build_model(encoder: str, nc: int, freeze: bool, scale: str = "s"):
    """Return an ultralytics YOLO whose backbone is `encoder` from BACKBONES."""
    if encoder not in BACKBONES:
        raise SystemExit(
            f"unknown --encoder {encoder!r}. Choose from:\n  " + "\n  ".join(BACKBONES)
        )
    kind, ref = BACKBONES[encoder]
    if kind == "yolo":
        # Prefer the canonical checkpoint in 80_models; fall back to the bare name so
        # Ultralytics auto-downloads it (avoids re-cluttering this dir with a loose .pt).
        canonical = HERE / "../../80_models/02_2stage/pretrained" / ref
        return YOLO(str(canonical) if canonical.exists() else ref)  # stock CSPDarknet, COCO-pretrained
    if kind == "yaml":
        timm_backbone.register()          # ConvNeXt keys resolve through TimmBackbone
        return YOLO(ref)
    # kind == "sfp": bind the hier_encoder backbone, then build the SFP+PANet YOLO.
    cfg = EncoderConfig(
        backbone=ref,
        pretrained=True,
        freeze_backbone=freeze,
        feature_strategy="sfp",
    )
    return hier_encoder_yolo.build_trainable_yolo(cfg, nc=nc, scale=scale)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--encoder", help="backbone name from the matrix: " + ", ".join(BACKBONES))
    g.add_argument("--model", help="stock checkpoint/YAML escape hatch, e.g. yolov8n.pt | rtdetr-l.pt")
    ap.add_argument("--data", required=True, help="path to dataset.yaml")
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--scale", default="s", help="YOLOv8 neck scale for SFP backbones (n/s/m/l/x)")
    ap.add_argument("--no-freeze", action="store_true", help="train the transformer backbone (SFP path only)")
    ap.add_argument("--warmup-epochs", type=float, default=3.0, dest="warmup_epochs")
    args = ap.parse_args()

    if args.encoder:
        model = build_model(args.encoder, nc=1, freeze=not args.no_freeze, scale=args.scale)
        is_rtdetr = False
    else:
        is_rtdetr = "rtdetr" in args.model.lower()
        model = (RTDETR if is_rtdetr else YOLO)(args.model)

    # RT-DETR (transformer) needs its own recipe: lower LR + long warmup.
    lr0, warmup = args.lr0, args.warmup_epochs
    if is_rtdetr:
        lr0 = min(args.lr0, 1e-4)
        warmup = max(args.warmup_epochs, 15.0)
        print(f"[RT-DETR recipe] lr0={lr0} warmup_epochs={warmup}")

    model.train(
        data=resolve_data(args.data),
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        optimizer="AdamW", lr0=lr0, cos_lr=True, patience=args.patience,
        warmup_epochs=warmup,
        box=10.0, dfl=2.0,               # YOLO loss gains (ignored by RT-DETR)
        mosaic=0.0, fliplr=0.0,          # text must not mirror / stitch
        project=args.project, name=args.name, exist_ok=True,
    )
    print(f"DONE -> {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
