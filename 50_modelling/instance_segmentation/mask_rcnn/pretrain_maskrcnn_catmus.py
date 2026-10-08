#!/usr/bin/env python3
"""Supervised CATMuS pretraining of the backbone-matrix Mask R-CNN arms.

This is the "CATMuS" initialisation of the RQ1 backbone tables: the same model
that train_maskrcnn_catmus_diva.py pretrains for the stock ResNet-50 arm, but
built through maskrcnn_diva.build_model so the encoder is one of the matrix
backbones (ConvNeXt-Tiny / PVTv2-B2 / ViT-S+SFP). Everything starts from random
weights and is trained on the 1335 CATMuS Medieval pages; the resulting best.pt
(full Mask R-CNN state) is the source checkpoint for fine-tuning on a target
subset. Recipe, loop, checkpoints and resume logic are train_run's, unchanged.

    python pretrain_maskrcnn_catmus.py --arm convnext_tiny_random --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_maskrcnn_catmus_diva as transfer  # noqa: E402
from maskrcnn_diva import ARMS, build_model as build_arm_model, seed_everything  # noqa: E402

REPO = HERE.parents[2]
OUT_ROOT = REPO / "80_models/instance_segmentation/mask_rcnn/catmus_pretrain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
                        choices=sorted(a for a in ARMS if a.endswith("_random")))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(42)

    # train_run builds its model through the module-level build_model; point it at
    # the matrix builder so `initialization` is read as an arm name.
    transfer.build_model = lambda arm, image_size: build_arm_model(arm, image_size)

    loaders = transfer.make_loaders(transfer.CATMUS_COCO, transfer.CATMUS_IMAGES, args.image_size)
    backbone = args.arm[: -len("_random")]
    run_dir = OUT_ROOT / f"maskrcnn_{backbone}_{args.image_size}{'_smoke' if args.smoke else ''}"
    checkpoint = transfer.train_run(
        run_dir, args.arm, loaders, args.epochs, args.image_size, smoke=args.smoke,
    )
    print(f"CATMuS pretrain done -> {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
