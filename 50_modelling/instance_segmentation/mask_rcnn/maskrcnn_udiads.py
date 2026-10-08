#!/usr/bin/env python3
"""Train PVTv2-B2 Mask R-CNN on U-DIADS-TL COCO annotations.

The official Zottin metric receives the binary ground-truth masks and a
non-overlapping predicted instance-ID map. For reproducibility this script
also writes a 0/255 binary union mask for every validation/test page.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mask2former_udiads import resolve_split_dir, zottin_metrics  # noqa: E402
from maskrcnn_diva import (  # noqa: E402
    ARMS,
    DivaCocoLines,
    load_init_checkpoint,
    resolve_init_checkpoint,
    build_model,
    collate,
    configure_mask_loss,
    seed_everything,
    train_one_epoch,
)


SUBSETS = ("Latin14396", "Latin2", "Syr341")
SCORE_GRID = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
MASK_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

# FIXED: optional line post-processing for RQ3 cross-collection robustness. All rules are off
# by default (previous behaviour). Thresholds are fixed a priori and never tuned on targets.
# RQ3_POST=1 switches on all three rules; RQ3_POST_<RULE>=1 switches on a single rule.
_POST_ALL = os.environ.get("RQ3_POST", "0") == "1"
POST_MERGE = _POST_ALL or os.environ.get("RQ3_POST_MERGE", "0") == "1"        # rule a
POST_MIN_AREA = _POST_ALL or os.environ.get("RQ3_POST_MIN_AREA", "0") == "1"  # rule b
POST_SHAPE_REGION = _POST_ALL or os.environ.get("RQ3_POST_SHAPE", "0") == "1"  # rule c
POST_Y_IOU = 0.5              # a: 1-D IoU of the y-extents
POST_GAP_FACTOR = 1.0         # a: horizontal gap < factor x median line height
POST_MIN_AREA_FRAC = 0.2      # b: area < frac x median mask area
POST_MAX_HEIGHT_FACTOR = 3.0  # c: height > factor x median line height
POST_CONFIDENT_SCORE = 0.7    # c: masks defining the main text block
POST_REGION_DILATE = 2.0      # c: text-block box grown by this x median line height
POST_REGION_MIN_INSIDE = 0.5  # c: share of a mask that must lie inside the text block
POST_LOG: list[dict] = []     # per-page counts of every rule, read by the caller


def make_loader(subset: str, split: str, image_size: int, augment: bool,
                max_images: int = 0):
    data_root = REPO / "00_data/U-DIADS-TL"
    dataset = DivaCocoLines(
        data_root / f"coco_dataset_{subset.lower()}",
        data_root / f"yolo_dataset_{subset.lower()}/images",
        split,
        image_size,
        augment=augment,
    )
    if max_images:
        dataset.records = dataset.records[:max_images]
    return dataset, DataLoader(
        dataset,
        batch_size=1,
        shuffle=augment,
        num_workers=1 if not augment else 2,
        pin_memory=True,
        collate_fn=collate,
    )


def configure_dense_inference(model, max_detections: int, replace_anchors: bool = True) -> None:
    # U-DIADS pages contain up to ~200 text-line instances. The Torchvision
    # COCO default of 100 silently truncates Latin2 and Syr341 predictions.
    model.roi_heads.detections_per_img = max_detections
    model.roi_heads.score_thresh = min(SCORE_GRID)
    model.rpn._post_nms_top_n["testing"] = max(1000, max_detections * 2)
    if not replace_anchors:
        # Zero-shot scoring of a pretrained model keeps its own (trained) RPN;
        # swapping anchors would re-initialise the head.
        return
    # U-DIADS COCO boxes have median h/w near 0.10; square and 0.5 anchors
    # only inflate the dense GT x anchor matching matrix on Syr341.
    ratios = (0.05, 0.1, 0.2)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=(ratios,) * 5,
    )
    model.rpn.head = RPNHead(model.backbone.out_channels, len(ratios), conv_depth=1)


def _boxes(canvas: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    boxes = {}
    for label in np.unique(canvas):
        if label == 0:
            continue
        ys, xs = np.nonzero(canvas == label)
        boxes[int(label)] = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    return boxes


def postprocess_lines(canvas: np.ndarray, scores: dict[int, float], meta: dict) -> np.ndarray:
    """FIXED: RQ3 line post-processing (rules a-c, each behind its constant).

    a) merge fragments whose y-extents overlap by 1-D IoU > POST_Y_IOU and whose horizontal
       gap is below POST_GAP_FACTOR x the page's median line height, until stable; merged
       fragments are joined through their convex hull so that one contour is exported;
    b) drop masks smaller than POST_MIN_AREA_FRAC x the page's median mask area;
    c) drop masks taller than POST_MAX_HEIGHT_FACTOR x the median line height or lying
       mostly outside the main text block (box around masks scoring >= POST_CONFIDENT_SCORE,
       grown by POST_REGION_DILATE x the median line height).
    """
    log = {"page": Path(str(meta.get("path", ""))).name, "before": len(scores),
           "merged": 0, "small": 0, "tall": 0, "outside": 0}
    boxes = _boxes(canvas)
    if not boxes:
        POST_LOG.append({**log, "after": 0})
        return canvas
    median_h = float(np.median([b[3] - b[2] + 1 for b in boxes.values()]))
    if POST_MERGE:
        changed = True
        while changed:
            changed = False
            ids = sorted(boxes)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    if a not in boxes or b not in boxes:
                        continue
                    ax0, ax1, ay0, ay1 = boxes[a]
                    bx0, bx1, by0, by1 = boxes[b]
                    inter = max(0, min(ay1, by1) - max(ay0, by0) + 1)
                    union = max(ay1, by1) - min(ay0, by0) + 1
                    gap = max(bx0 - ax1, ax0 - bx1, 0)
                    if inter / union > POST_Y_IOU and gap < POST_GAP_FACTOR * median_h:
                        joined = ((canvas == a) | (canvas == b)).astype(np.uint8)
                        hull = cv2.convexHull(cv2.findNonZero(joined))
                        fill = np.zeros_like(joined)
                        cv2.fillPoly(fill, [hull], 1)
                        canvas[(fill > 0) & ((canvas == 0) | (canvas == b))] = a
                        canvas[canvas == b] = a
                        scores[a] = max(scores.get(a, 0.0), scores.pop(b, 0.0))
                        boxes = _boxes(canvas)
                        log["merged"] += 1
                        changed = True
    if POST_MIN_AREA:
        areas = {k: int((canvas == k).sum()) for k in boxes}
        median_area = float(np.median(list(areas.values())))
        for k, area in areas.items():
            if area < POST_MIN_AREA_FRAC * median_area:
                canvas[canvas == k] = 0
                log["small"] += 1
        boxes = _boxes(canvas)
    if POST_SHAPE_REGION and boxes:
        median_h = float(np.median([b[3] - b[2] + 1 for b in boxes.values()]))
        for k, (x0, x1, y0, y1) in list(boxes.items()):
            if y1 - y0 + 1 > POST_MAX_HEIGHT_FACTOR * median_h:
                canvas[canvas == k] = 0
                log["tall"] += 1
                del boxes[k]
        confident = [boxes[k] for k in boxes if scores.get(k, 0.0) >= POST_CONFIDENT_SCORE]
        if confident:
            pad = POST_REGION_DILATE * median_h
            rx0 = max(0, min(b[0] for b in confident) - pad)
            rx1 = max(b[1] for b in confident) + pad
            ry0 = max(0, min(b[2] for b in confident) - pad)
            ry1 = max(b[3] for b in confident) + pad
            region = np.zeros(canvas.shape, bool)
            region[int(ry0):int(ry1) + 1, int(rx0):int(rx1) + 1] = True
            for k in list(boxes):
                mask = canvas == k
                if (mask & region).sum() < POST_REGION_MIN_INSIDE * mask.sum():
                    canvas[mask] = 0
                    log["outside"] += 1
    # Relabel consecutively so downstream code sees 1..N.
    out = np.zeros_like(canvas)
    for new_id, k in enumerate([k for k in np.unique(canvas) if k != 0], start=1):
        out[canvas == k] = new_id
    POST_LOG.append({**log, "after": int(out.max())})
    return out


# Exploratory (2026-10-05): optional DP-seam export of the ROI masks (RQ3 transfer check).
SEAM_EXPORT = os.environ.get("RQ3_SEAM", "0") == "1"
SEAM_THR = float(os.environ.get("RQ3_SEAM_THR", "0.5"))
SEAM_K = int(os.environ.get("RQ3_SEAM_K", "64"))
if SEAM_EXPORT:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dp_seam"))
    from seam_core import seam_polygon  # noqa: E402


def prediction_to_instances(output: dict, meta: dict, score_threshold: float,
                            mask_threshold: float) -> np.ndarray:
    """Make a native-resolution non-overlapping uint16 instance map."""
    height, width = int(meta["new_h"]), int(meta["new_w"])
    canvas = np.zeros((height, width), dtype=np.uint16)
    next_id = 1
    scores = {}
    # Torchvision returns detections in descending score order, so overlap goes
    # to the most confident instance, matching the other U-DIADS pipelines.
    for idx, (score, mask) in enumerate(zip(output["scores"], output["masks"][:, 0])):
        if float(score) < score_threshold:
            continue
        foreground = mask[:height, :width].detach().float().cpu().numpy() >= mask_threshold
        if SEAM_EXPORT:  # exploratory: DP seam on the ROI-mask probability inside the box
            prob = mask[:height, :width].detach().float().cpu().numpy()
            x0, y0, x1, y1 = [int(round(float(v))) for v in output["boxes"][idx]]
            x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
            x1, y1 = min(width, x1 + 2), min(height, y1 + 2)
            poly = seam_polygon(prob[y0:y1, x0:x1], SEAM_THR, SEAM_K) if x1 - x0 > 8 and y1 - y0 > 1 else None
            if poly is not None and len(poly) >= 3:
                region = np.zeros((height, width), np.uint8)
                cv2.fillPoly(region, [poly.astype(np.int32) + [x0, y0]], 1)
                foreground = region > 0
        writable = foreground & (canvas == 0)
        if not writable.any():
            continue
        canvas[writable] = next_id
        scores[next_id] = float(score)
        next_id += 1
    if POST_MERGE or POST_MIN_AREA or POST_SHAPE_REGION:  # FIXED: optional post-processing
        canvas = postprocess_lines(canvas, scores, meta)
    return cv2.resize(
        canvas,
        (int(meta["orig_w"]), int(meta["orig_h"])),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint16, copy=False)


def ground_truth_path(subset: str, split: str, stem: str) -> Path:
    base = REPO / "00_data/U-DIADS-TL" / subset / f"text-line-gt-{subset}"
    return resolve_split_dir(base, split) / f"{stem}.png"


@torch.inference_mode()
def score_model(model, loader, device, subset: str, split: str,
                score_threshold: float, mask_threshold: float,
                output_dir: Path | None = None) -> tuple[dict, list[dict]]:
    model.eval()
    if output_dir is not None:
        instance_dir = output_dir / "instances"
        binary_dir = output_dir / "binary_masks"
        instance_dir.mkdir(parents=True, exist_ok=True)
        binary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for page, (images, _, metadata) in enumerate(loader, start=1):
        image = images[0].to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model([image])[0]
        meta = metadata[0]
        stem = Path(meta["path"]).stem
        instances = prediction_to_instances(
            output, meta, score_threshold=score_threshold, mask_threshold=mask_threshold
        )
        gt_binary = np.asarray(Image.open(ground_truth_path(subset, split, stem)).convert("L")) > 0
        metrics = zottin_metrics(gt_binary, instances)
        row = {
            "image": f"{stem}.png",
            "instances": int(len(np.unique(instances)) - 1),
            **metrics,
        }
        rows.append(row)
        if output_dir is not None:
            Image.fromarray(instances, mode="I;16").save(instance_dir / f"{stem}.png")
            Image.fromarray((instances > 0).astype(np.uint8) * 255, mode="L").save(
                binary_dir / f"{stem}.png"
            )
        print(
            f"[{subset}/{split}] {page:02d}/{len(loader):02d} {stem} "
            f"n={row['instances']} FM={metrics['FM']:.4f} PixelIU={metrics['Pixel_IU']:.4f}",
            flush=True,
        )
        del output, image
    keys = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
    summary = {
        "pages": len(rows),
        "mean_instances": float(np.mean([row["instances"] for row in rows])),
        **{key: float(np.mean([row[key] for row in rows])) for key in keys},
    }
    if output_dir is not None:
        with (output_dir / "per_page_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary, rows


@torch.inference_mode()
def calibrate_score(model, loader, device, subset: str) -> tuple[float, list[dict]]:
    """Tune confidence by validation line-count error without touching test."""
    model.eval()
    errors = {threshold: [] for threshold in SCORE_GRID}
    for images, _, metadata in loader:
        image = images[0].to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model([image])[0]
        stem = Path(metadata[0]["path"]).stem
        gt = np.asarray(Image.open(ground_truth_path(subset, "val", stem)).convert("L")) > 0
        gt_components, _ = cv2.connectedComponents(gt.astype(np.uint8))
        target_count = int(gt_components - 1)
        scores = output["scores"].detach().float().cpu().numpy()
        for threshold in SCORE_GRID:
            errors[threshold].append(abs(int((scores >= threshold).sum()) - target_count))
        del output, image
    rows = [
        {"score_threshold": threshold, "mean_absolute_count_error": float(np.mean(values))}
        for threshold, values in errors.items()
    ]
    selected = min(rows, key=lambda row: (row["mean_absolute_count_error"], -row["score_threshold"]))
    return float(selected["score_threshold"]), rows


@torch.inference_mode()
def calibrate_masks(model, loader, device, subset: str,
                    score_threshold: float) -> tuple[float, list[dict]]:
    """Evaluate every mask cutoff in one validation inference pass."""
    model.eval()
    collected = {threshold: [] for threshold in MASK_GRID}
    for images, _, metadata in loader:
        image = images[0].to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model([image])[0]
        meta = metadata[0]
        stem = Path(meta["path"]).stem
        gt_binary = np.asarray(Image.open(ground_truth_path(subset, "val", stem)).convert("L")) > 0
        for threshold in MASK_GRID:
            instances = prediction_to_instances(output, meta, score_threshold, threshold)
            collected[threshold].append(zottin_metrics(gt_binary, instances))
        del output, image
    metric_names = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
    rows = [
        {
            "mask_threshold": threshold,
            **{
                name: float(np.mean([metrics[name] for metrics in pages]))
                for name in metric_names
            },
        }
        for threshold, pages in collected.items()
    ]
    selected = max(rows, key=lambda row: (row["FM"], row["Pixel_IU"]))
    return float(selected["mask_threshold"]), rows


def save_evaluation(output_dir: Path, summary: dict, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with (output_dir / "per_page_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_subset(args, subset: str, device: torch.device) -> dict:
    run_name = f"maskrcnn_{args.arm}_{args.image_size}_{args.epochs}ep"
    if args.zero_shot:
        run_name = f"maskrcnn_{args.arm}_{args.image_size}_zeroshot"
    if args.smoke:
        run_name += "_smoke"
    model_dir = args.model_root / subset / run_name
    eval_dir = args.evaluation_root / subset / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    limit = 1 if args.smoke else 0
    train_data, train_loader = make_loader(subset, "train", args.image_size, True, limit)
    _, val_loader = make_loader(subset, "val", args.image_size, False, limit)
    _, test_loader = make_loader(subset, "test", args.image_size, False, limit)
    model = build_model(args.arm, args.image_size)
    init_checkpoint = resolve_init_checkpoint(args.arm, args.init_checkpoint)
    if init_checkpoint is not None:
        init_epoch = load_init_checkpoint(model, init_checkpoint)
    configure_dense_inference(model, args.max_detections, replace_anchors=not args.zero_shot)
    if args.zero_shot:
        # The evaluation block reloads model_dir/best.pt; make the init weights
        # that checkpoint so zero-shot goes through the same scoring path.
        torch.save({"model": model.state_dict(), "epoch": init_epoch, "subset": subset},
                   model_dir / "best.pt")
        args.evaluate_only = True
    model.to(device)
    print(
        f"[{subset}] train/val/test={len(train_data)}/{len(val_loader.dataset)}/{len(test_loader.dataset)} "
        f"size={args.image_size} max_det={args.max_detections}", flush=True,
    )

    history = []
    started = time.time()
    epochs = 1 if args.smoke else args.epochs
    if not args.evaluate_only:
        backbone = list(model.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone}
        downstream = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone, "lr": args.lr_backbone},
            {"params": downstream, "lr": args.lr},
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        best_fm = -math.inf
        best_pixel_iu = -math.inf
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            loss, losses = train_one_epoch(model, train_loader, optimizer, scaler, device)
            scheduler.step()
            row = {"epoch": epoch, "loss": loss, **losses}
            should_eval = args.smoke or epoch % args.eval_every == 0 or epoch == epochs
            if should_eval:
                val, _ = score_model(
                    model, val_loader, device, subset, "val",
                    score_threshold=args.checkpoint_score,
                    mask_threshold=args.checkpoint_mask,
                )
                row.update({f"val_{key}": value for key, value in val.items()})
                print(
                    f"[{subset}] epoch={epoch:03d} loss={loss:.4f} "
                    f"val_FM={val['FM']:.4f} val_PixelIU={val['Pixel_IU']:.4f}", flush=True,
                )
                if (val["FM"], val["Pixel_IU"]) > (best_fm, best_pixel_iu):
                    best_fm, best_pixel_iu, best_epoch = val["FM"], val["Pixel_IU"], epoch
                    serializable_args = {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    }
                    torch.save({
                        "model": model.state_dict(), "epoch": epoch,
                        "args": serializable_args, "subset": subset,
                    }, model_dir / "best.pt")
            else:
                print(f"[{subset}] epoch={epoch:03d} loss={loss:.4f}", flush=True)
            history.append(row)
            (model_dir / "history.json").write_text(json.dumps(history, indent=2))
    else:
        best_epoch = int(torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)["epoch"])

    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    best_epoch = int(checkpoint["epoch"])

    score_threshold, score_rows = calibrate_score(model, val_loader, device, subset)
    selected_mask, mask_trials = calibrate_masks(
        model, val_loader, device, subset, score_threshold
    )

    validation_dir = eval_dir / "validation"
    test_dir = eval_dir / "test"
    validation, validation_rows = score_model(
        model, val_loader, device, subset, "val", score_threshold, selected_mask,
        output_dir=validation_dir,
    )
    test, test_rows = score_model(
        model, test_loader, device, subset, "test", score_threshold, selected_mask,
        output_dir=test_dir,
    )
    calibration = {"score_grid": score_rows, "mask_grid": mask_trials}
    (eval_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    summary = {
        "model": "Mask R-CNN", "backbone": args.arm,
        "annotation_format": "COCO polygons", "subset": subset,
        "image_size": args.image_size, "epochs": epochs, "best_epoch": best_epoch,
        "max_detections": args.max_detections,
        "score_threshold": score_threshold, "mask_threshold": selected_mask,
        "threshold_selection": "validation only; count error then Zottin FM",
        "validation": validation, "test": test,
        "runtime_seconds": time.time() - started,
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_evaluation(validation_dir, {**summary, "split": "validation", **validation}, validation_rows)
    save_evaluation(test_dir, {**summary, "split": "test", **test}, test_rows)
    print(json.dumps(summary, indent=2), flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["all"])
    # Same backbone arms as maskrcnn_diva.py; the default keeps the run name of
    # the runs that already exist (maskrcnn_pvt_v2_b2_imagenet_<size>_<ep>ep).
    parser.add_argument("--arm", choices=sorted(ARMS), default="pvt_v2_b2_imagenet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-score", type=float, default=0.5)
    parser.add_argument("--checkpoint-mask", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="full Mask R-CNN state to start from (default for *_catmus arms)")
    parser.add_argument("--zero-shot", action="store_true",
                        help="skip training: score the init checkpoint as is")
    parser.add_argument(
        "--model-root", type=Path,
        default=REPO / "80_models/instance_segmentation/mask_rcnn/u-diads-tl",
    )
    parser.add_argument(
        "--evaluation-root", type=Path,
        default=REPO / "99_evaluation/instance_segmentation/mask_rcnn/u-diads-tl",
    )
    args = parser.parse_args()
    subsets = list(SUBSETS) if args.subsets == ["all"] else args.subsets
    unknown = set(subsets) - set(SUBSETS)
    if unknown:
        parser.error(f"unknown subsets: {sorted(unknown)}")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_mask_loss("bce", device)
    summaries = [run_subset(args, subset, device) for subset in subsets]
    aggregate = {
        "model": "Mask R-CNN", "backbone": args.arm,
        "subsets": summaries,
        "macro_test": {
            key: float(np.mean([summary["test"][key] for summary in summaries]))
            for key in ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
        },
    }
    args.evaluation_root.mkdir(parents=True, exist_ok=True)
    aggregate_name = ("maskrcnn_pvt_v2_b2_summary.json" if args.arm == "pvt_v2_b2_imagenet"
                      else f"maskrcnn_{args.arm}_summary.json")
    (args.evaluation_root / aggregate_name).write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
