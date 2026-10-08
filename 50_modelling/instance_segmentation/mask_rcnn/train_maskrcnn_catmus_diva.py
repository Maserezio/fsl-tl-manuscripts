#!/usr/bin/env python3
"""Sequential Mask R-CNN random/COCO/CATMuS transfer experiment."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead

from maskrcnn_diva import (
    DivaCocoLines,
    collate,
    evaluate_coco,
    metric_value,
    seed_everything,
    train_one_epoch,
)


REPO = Path(__file__).resolve().parents[3]
MODEL_ROOT = REPO / "80_models/instance_segmentation/mask_rcnn/maskrcnn_catmus_transfer"
CATMUS_COCO = REPO / "00_data/CATMuS/medieval-segmentation/coco_instances"
CATMUS_IMAGES = REPO / "00_data/CATMuS/medieval-segmentation/yolo_seg_dataset/images"
SUBSETS = ("CB55", "CS18", "CS863")
ARMS = ("random", "coco", "catmus")


def build_model(initialization: str, image_size: int = 704):
    weights = (
        MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        if initialization == "coco" else None
    )
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=image_size,
        max_size=image_size,
        box_detections_per_img=300,
        trainable_backbone_layers=5,
    )
    ratios = (0.05, 0.1, 0.2, 0.5, 1.0)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(ratios,) * 5,
    )
    model.rpn.head = RPNHead(model.backbone.out_channels, len(ratios), conv_depth=1)
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    model.roi_heads.detections_per_img = 300
    model.roi_heads.score_thresh = 0.0
    return model


def make_loaders(coco_dir: Path, image_root: Path, image_size: int):
    loaders = {}
    for split, augment, shuffle, workers in (
        ("train", True, True, 2), ("val", False, False, 1)
    ):
        dataset = DivaCocoLines(coco_dir, image_root, split, image_size, augment)
        loaders[split] = DataLoader(
            dataset, batch_size=1, shuffle=shuffle, num_workers=workers,
            pin_memory=True, collate_fn=collate,
        )
    return loaders


def complete(run_dir: Path, epochs: int) -> bool:
    history_path = run_dir / "history.json"
    if not (run_dir / "best.pt").is_file() or not history_path.is_file():
        return False
    return len(json.loads(history_path.read_text(encoding="utf-8"))) >= epochs


def train_run(
    run_dir: Path,
    initialization: str,
    loaders: dict,
    epochs: int,
    image_size: int,
    source_checkpoint: Path | None = None,
    smoke: bool = False,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    target_epochs = 1 if smoke else epochs
    if complete(run_dir, target_epochs):
        print(f"[{run_dir.name}] complete; reusing best.pt", flush=True)
        return run_dir / "best.pt"
    device = torch.device("cuda")
    model = build_model(initialization, image_size)
    if source_checkpoint is not None:
        state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{run_dir.name}] params total={total:,} trainable={trainable:,} "
        f"pages={len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}",
        flush=True,
    )
    if trainable < 40_000_000:
        raise RuntimeError(f"truncated Mask R-CNN: only {trainable:,} trainable params")
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone}
    other = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": 1e-4}, {"params": other, "lr": 1e-3}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Training uses BF16 autocast in train_one_epoch; gradient scaling is an
    # FP16 workaround and is unnecessary for BF16.
    scaler = GradScaler("cuda", enabled=False)
    history = []
    best_map = -1.0
    start_epoch = 1
    last_path = run_dir / "last.pt"
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint.get("best_map", -1.0))
        print(f"[{run_dir.name}] resume at epoch {start_epoch}", flush=True)
    started = time.time()
    for epoch in range(start_epoch, target_epochs + 1):
        loss, components = train_one_epoch(
            model, loaders["train"], optimizer, scaler, device,
            max_steps=1 if smoke else 0,
        )
        scheduler.step()
        row = {"epoch": epoch, "loss": loss, **components}
        should_eval = smoke or epoch % 5 == 0 or epoch == target_epochs
        if should_eval:
            metrics = evaluate_coco(
                model, loaders["val"], device,
                max_pages=1 if smoke else 0, max_detections=300,
            )
            row.update({f"val_{k}": v for k, v in metrics.items()})
            # COCO's aggregate AP is undefined (-1) when its conventional
            # maxDet=100 is replaced by 300 for dense manuscript pages. AP50 is
            # still well-defined and uses all 300 predictions.
            score = metric_value(metrics, "map_50")
            if not (run_dir / "best.pt").is_file() or score > best_map:
                best_map = score
                torch.save({"model": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
            print(
                f"[{run_dir.name}] epoch={epoch:03d}/{target_epochs} loss={loss:.4f} "
                f"val_segm_mAP50={score:.4f} best={best_map:.4f} "
                f"elapsed={(time.time()-started)/60:.1f}m",
                flush=True,
            )
        else:
            print(
                f"[{run_dir.name}] epoch={epoch:03d}/{target_epochs} loss={loss:.4f} "
                f"elapsed={(time.time()-started)/60:.1f}m", flush=True,
            )
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_map": best_map,
        }, last_path)
    if not (run_dir / "best.pt").is_file():
        torch.save({"model": model.state_dict(), "epoch": target_epochs}, run_dir / "best.pt")
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return run_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("pretrain", "finetune"),
                        default=["pretrain", "finetune"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--run-suffix", default="",
                        help="suffix for distinct fine-tuning runs")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(42)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    cat_loaders = make_loaders(CATMUS_COCO, CATMUS_IMAGES, args.image_size)
    pretrain_dir = MODEL_ROOT / ("catmus_random_pretrain_smoke" if args.smoke else "catmus_random_pretrain")
    cat_checkpoint = pretrain_dir / "best.pt"
    if "pretrain" in args.stages:
        cat_checkpoint = train_run(
            pretrain_dir, "random", cat_loaders, args.pretrain_epochs,
            args.image_size, smoke=args.smoke,
        )
    if "finetune" not in args.stages:
        return
    if not cat_checkpoint.is_file():
        raise FileNotFoundError(cat_checkpoint)
    for subset in args.subsets:
        coco_dir = REPO / f"00_data/DIVA-HisDB/coco_task2_{subset}"
        image_root = REPO / f"00_data/DIVA-HisDB/yolo_dataset_{subset}/images"
        loaders = make_loaders(coco_dir, image_root, args.image_size)
        for arm in args.arms:
            source = cat_checkpoint if arm == "catmus" else None
            run_name = f"{subset.lower()}_{arm}_finetune{args.finetune_epochs}"
            if args.run_suffix:
                run_name += f"_{args.run_suffix}"
            if args.smoke:
                run_name += "_smoke"
            train_run(
                MODEL_ROOT / run_name, arm, loaders, args.finetune_epochs,
                args.image_size, source_checkpoint=source, smoke=args.smoke,
            )


if __name__ == "__main__":
    main()
