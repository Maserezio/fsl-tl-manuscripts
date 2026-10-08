#!/usr/bin/env python3
"""Distill DINOv3 ViT-B/16 into a random-init ViT-S/16 on CATMuS images.

The exported checkpoint is a plain timm backbone state dict and can therefore be
loaded into the same ViT-S/16 used by ``hier_encoder``/Mask R-CNN.  CATMuS labels
are deliberately ignored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightly_train
import timm
import torch


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "00_data/CATMuS/medieval-segmentation/source/data"
DEFAULT_OUT = ROOT / "80_models/instance_segmentation/mask_rcnn/vit_catmus_dinov3_distillation"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# A complete DINOv3 ViT-S/16 has 21,586,944 parameters with its classifier
# removed.  This guard catches accidentally truncated/wrapped backbones.
MIN_STUDENT_PARAMS = 20_000_000
MAX_STUDENT_PARAMS = 23_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch in a separate output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)

    images = sorted(p for p in DATA.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found below {DATA}")

    # No .pt checkpoint and pretrained=False: the student really starts random.
    # num_classes=0 also keeps the exported state dict backbone-only.
    student = timm.create_model(
        "vit_small_patch16_dinov3",
        pretrained=False,
        dynamic_img_size=True,
        num_classes=0,
    )
    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"CATMuS images: {len(images)}")
    print(f"Student parameters: total={total:,}, trainable={trainable:,}")
    if not (MIN_STUDENT_PARAMS <= trainable <= MAX_STUDENT_PARAMS):
        raise RuntimeError(
            f"Refusing to train truncated/unexpected student: {trainable:,} trainable "
            f"parameters, expected {MIN_STUDENT_PARAMS:,}..{MAX_STUDENT_PARAMS:,}."
        )

    # Confirm that the dense token map, rather than only a classifier head, is present.
    with torch.inference_mode():
        features = student.forward_features(torch.zeros(1, 3, 224, 224))
    print(f"Student dense feature shape at 224 px: {tuple(features.shape)}")

    out = args.out
    epochs = args.epochs
    if args.smoke:
        out = args.out.with_name(args.out.name + "_smoke")
        epochs = 1

    lightly_train.pretrain(
        out=out,
        data=DATA,
        model=student,
        method="distillation",  # LightlyTrain >=0.15: distillationv3 alias.
        method_args={"teacher": "dinov3/vitb16"},
        batch_size=args.batch_size,
        epochs=epochs,
        num_workers=args.workers,
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
        optim="adamw",
        trainer_args={"log_every_n_steps": 10},
        overwrite=args.smoke,
    )


if __name__ == "__main__":
    main()
