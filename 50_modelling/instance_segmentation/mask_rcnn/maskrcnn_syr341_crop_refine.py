#!/usr/bin/env python3
"""Validation-selected Mask R-CNN + Tversky refinement on U-DIADS-TL.

Mask R-CNN is the sole source of instance proposals, scores, and identities.
The fixed crop U-Net replaces only its coarse ROI masks. All post-processing
parameters are selected on validation before the test split is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"))

import evaluate_loss_ablation_detr_udiads as crop_base  # noqa: E402
from maskrcnn_diva import build_model  # noqa: E402
from maskrcnn_udiads import configure_dense_inference, make_loader  # noqa: E402


SUBSETS = ("Latin14396", "Latin2", "Syr341")
SUBSET = "Syr341"
METRICS = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
SCORE_VALUES = tuple(np.round(np.arange(0.01, 0.52, 0.02), 2))
MASK_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
MINIMUM_AREAS = (0, 100, 250, 500, 1000)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def ground_truth_dir(split: str) -> Path:
    root = REPO / "00_data/U-DIADS-TL" / SUBSET / f"text-line-gt-{SUBSET}"
    candidates = ("validation", "val") if split == "val" else ("test", "public-test")
    return next(path for name in candidates if (path := root / name).is_dir())


@torch.inference_mode()
def cache_detections(model, loader, device: torch.device, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    detections = {}
    model.eval()
    for index, (images, _, metadata) in enumerate(loader, start=1):
        meta = metadata[0]
        stem = Path(meta["path"]).stem
        cache_path = cache_dir / f"{stem}.npz"
        if cache_path.exists():
            saved = np.load(cache_path)
            boxes, scores = saved["boxes"], saved["scores"]
        else:
            image = images[0].to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model([image])[0]
            scale_x = float(meta["new_w"]) / float(meta["orig_w"])
            scale_y = float(meta["new_h"]) / float(meta["orig_h"])
            boxes = output["boxes"].detach().float().cpu().numpy().astype(np.float32)
            boxes[:, [0, 2]] /= scale_x
            boxes[:, [1, 3]] /= scale_y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, int(meta["orig_w"]))
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, int(meta["orig_h"]))
            scores = output["scores"].detach().float().cpu().numpy().astype(np.float32)
            valid = (
                (boxes[:, 2] > boxes[:, 0])
                & (boxes[:, 3] > boxes[:, 1])
                & np.isfinite(boxes).all(axis=1)
            )
            boxes, scores = boxes[valid], scores[valid]
            np.savez_compressed(cache_path, boxes=boxes, scores=scores)
            del output, image
        detections[stem] = (boxes, scores)
        print(f"[Mask R-CNN] {index:02d}/{len(loader):02d} {stem}: {len(boxes)} raw boxes", flush=True)
    return detections


def select_detector(raw: dict, gt_dir: Path) -> tuple[float, list[dict]]:
    rows = []
    for score_threshold in SCORE_VALUES:
        tp = fp = fn = 0
        counts = []
        for stem, (boxes, scores) in raw.items():
            predicted = boxes[scores >= score_threshold]
            truth = crop_base.component_boxes(gt_dir / f"{stem}.png")
            one_tp, one_fp, one_fn = crop_base.detection_counts(predicted, truth)
            tp += one_tp
            fp += one_fp
            fn += one_fn
            counts.append(len(predicted))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append({
            "score_threshold": score_threshold,
            "box_precision@0.5": precision,
            "box_recall@0.5": recall,
            "box_F1@0.5": f1,
            "mean_boxes": float(np.mean(counts)),
        })
    best = max(rows, key=lambda row: (row["box_F1@0.5"], row["box_precision@0.5"]))
    return float(best["score_threshold"]), rows


@torch.inference_mode()
def crop_probability(crop_bgr: np.ndarray, model: torch.nn.Module,
                     crop_width: int, crop_height: int) -> np.ndarray:
    native_height, native_width = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)
    tensor = tensor.float().unsqueeze(0).to(crop_base.DEVICE).div_(255.0)
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return cv2.resize(probability, (native_width, native_height), interpolation=cv2.INTER_LINEAR)


def render_thresholds(page: np.ndarray, boxes: np.ndarray, model: torch.nn.Module,
                      thresholds: tuple[float, ...], pad: int, args) -> dict[float, np.ndarray]:
    height, width = page.shape[:2]
    canvases = {value: np.zeros((height, width), dtype=np.uint16) for value in thresholds}
    next_ids = {value: 1 for value in thresholds}
    for box in boxes:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        probability = crop_probability(page[y1:y2, x1:x2], model, args.crop_width, args.crop_height)
        for threshold in thresholds:
            mask = crop_base.connect_line(probability >= threshold, args.close_fraction).astype(bool)
            if not mask.any():
                continue
            region = canvases[threshold][y1:y2, x1:x2]
            writable = mask & (region == 0)
            if writable.any():
                region[writable] = next_ids[threshold]
                next_ids[threshold] += 1
    return canvases


def filter_instance_area(instances: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area <= 0:
        return instances
    counts = np.bincount(instances.ravel())
    keep = counts >= minimum_area
    keep[0] = False
    mapping = np.zeros(len(counts), dtype=np.uint16)
    mapping[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.uint16)
    return mapping[instances]


def evaluate_contiguous(gt_img: np.ndarray, pred_img: np.ndarray,
                        threshold: float = 0.75) -> tuple[float, ...]:
    """Zottin metric without sorting contiguous predicted IDs per pixel."""
    num_gt, gt_components = cv2.connectedComponents(gt_img)
    num_pred = int(pred_img.max()) + 1
    n_gt, n_pred = num_gt - 1, num_pred - 1
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    contingency = np.bincount(
        gt_components.ravel().astype(np.int64) * num_pred + pred_img.ravel(),
        minlength=num_gt * num_pred,
    ).reshape(num_gt, num_pred)
    intersections = contingency[1:, 1:].astype(np.float64)
    gt_area = contingency[1:, :].sum(axis=1).astype(np.float64)
    pred_area = contingency[:, 1:].sum(axis=0).astype(np.float64)
    unions = gt_area[:, None] + pred_area[None, :] - intersections
    ious = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    matched_all = int((ious >= threshold).sum())
    if n_pred:
        best_pred = np.argmax(ious, axis=1)
        best_iou = ious[np.arange(n_gt), best_pred]
        matched_gt = np.flatnonzero(best_iou > 0)
    else:
        best_pred = np.empty((n_gt,), dtype=np.int64)
        matched_gt = np.empty((0,), dtype=np.int64)
    if len(matched_gt):
        chosen = best_pred[matched_gt]
        tp = intersections[matched_gt, chosen]
        fp = pred_area[chosen] - tp
        fn = gt_area[matched_gt] - tp
        denominator = float((tp + fp + fn).sum())
        pixel_iu = 0.0 if denominator == 0 else float(tp.sum() / denominator)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        line_iu = float(((precision >= threshold) & (recall >= threshold)).mean())
    else:
        pixel_iu = line_iu = 0.0
    detection_rate = matched_all / n_gt
    recognition_accuracy = 0.0 if n_pred == 0 else matched_all / n_pred
    fm = 0.0 if detection_rate + recognition_accuracy == 0 else (
        2 * detection_rate * recognition_accuracy / (detection_rate + recognition_accuracy)
    )
    return pixel_iu, line_iu, detection_rate, recognition_accuracy, fm


def main() -> None:
    global SUBSET
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=SUBSETS, default="Syr341")
    parser.add_argument("--arm", default="pvt_v2_b2_imagenet",
                        help="Mask R-CNN arm (maskrcnn_diva.ARMS key) supplying the proposals")
    parser.add_argument("--crop-arm", choices=("bce", "tversky", "supervoxel"), default="tversky",
                        help="loss arm of the fixed two-stage crop U-Net")
    parser.add_argument("--fixed-postproc", action="store_true",
                        help="no validation grid for the mask stage: pad 15, mask threshold 0.5, "
                             "no minimum area (the detector score is still validation-selected)")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--pads", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--close-fraction", type=float, default=0.0)
    parser.add_argument("--max-detections", type=int, default=300)
    args = parser.parse_args()
    SUBSET = args.subset
    if args.output_root is None:
        args.output_root = (
            REPO / "99_evaluation/instance_segmentation/mask_rcnn/u-diads-tl" / SUBSET
            / (f"maskrcnn_{args.arm}_1024_200ep_croprefine"
               + ("" if args.crop_arm == "tversky" and not args.fixed_postproc
                  else f"_{args.crop_arm}" + ("_fixed" if args.fixed_postproc else "")))
        )
    if args.model_dir is None:
        args.model_dir = (
            REPO / "80_models/instance_segmentation/mask_rcnn/u-diads-tl" / SUBSET
            / f"maskrcnn_{args.arm}_1024_200ep"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(crop_base.DEVICE)
    checkpoint = torch.load(args.model_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["args"]
    model = build_model(args.arm, int(cfg["image_size"]))
    configure_dense_inference(model, args.max_detections)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    _, val_loader = make_loader(SUBSET, "val", int(cfg["image_size"]), False)
    _, test_loader = make_loader(SUBSET, "test", int(cfg["image_size"]), False)

    val_raw = cache_detections(model, val_loader, device, args.output_root / "detections/validation")
    score_threshold, detector_grid = select_detector(val_raw, ground_truth_dir("val"))
    write_rows(args.output_root / "validation_detector_grid.csv", detector_grid)
    selected_detector = next(row for row in detector_grid if row["score_threshold"] == score_threshold)
    print(f"[validation] detector selected: {selected_detector}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    segmenter_path = (
        REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl"
        / "crop_seg_loss_ablation_components_1024x256" / SUBSET
        / f"{args.crop_arm}/best.pth"
    )
    segmenter, segmenter_checkpoint = crop_base.load_segmenter(segmenter_path, crop_base.DEVICE)
    pads, mask_thresholds, minimum_areas = args.pads, MASK_THRESHOLDS, MINIMUM_AREAS
    if args.fixed_postproc:
        pads, mask_thresholds, minimum_areas = [15], (0.5,), (0,)
    validation = {
        (pad, threshold, area): []
        for pad in pads for threshold in mask_thresholds for area in minimum_areas
    }
    val_gt = ground_truth_dir("val")
    for page_index, (images, _, metadata) in enumerate(val_loader, start=1):
        meta = metadata[0]
        stem = Path(meta["path"]).stem
        page = cv2.imread(meta["path"], cv2.IMREAD_COLOR)
        boxes, scores = val_raw[stem]
        boxes = boxes[scores >= score_threshold]
        gt = cv2.imread(str(val_gt / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        for pad in pads:
            rendered = render_thresholds(page, boxes, segmenter, mask_thresholds, pad, args)
            for threshold in mask_thresholds:
                for area in minimum_areas:
                    instances = filter_instance_area(rendered[threshold], area)
                    values = evaluate_contiguous(gt, instances)
                    validation[(pad, threshold, area)].append(dict(zip(METRICS, values)))
        print(f"[validation] {page_index:02d}/{len(val_loader):02d} {stem}: {len(boxes)} boxes", flush=True)

    mask_grid = []
    for (pad, threshold, area), rows in validation.items():
        mask_grid.append({"pad": pad, "mask_threshold": threshold, "minimum_instance_area": area, **aggregate(rows)})
    write_rows(args.output_root / "validation_mask_grid.csv", mask_grid)
    best = max(mask_grid, key=lambda row: (row["FM"], row["Pixel_IU"]))
    print(f"[validation] mask selected: {best}", flush=True)

    # Test is touched once, after the complete validation selection.
    model = build_model(args.arm, int(cfg["image_size"]))
    configure_dense_inference(model, args.max_detections)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    test_raw = cache_detections(model, test_loader, device, args.output_root / "detections/test")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    instance_dir = args.output_root / "test/instances"
    binary_dir = args.output_root / "test/binary_masks"
    instance_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    test_gt = ground_truth_dir("test")
    test_rows = []
    for page_index, (images, _, metadata) in enumerate(test_loader, start=1):
        meta = metadata[0]
        stem = Path(meta["path"]).stem
        page = cv2.imread(meta["path"], cv2.IMREAD_COLOR)
        boxes, scores = test_raw[stem]
        boxes = boxes[scores >= score_threshold]
        instances = render_thresholds(
            page, boxes, segmenter, (float(best["mask_threshold"]),), int(best["pad"]), args
        )[float(best["mask_threshold"])]
        instances = filter_instance_area(instances, int(best["minimum_instance_area"]))
        gt = cv2.imread(str(test_gt / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        values = evaluate_contiguous(gt, instances)
        row = {"page": stem, "boxes": len(boxes), "instances": len(np.unique(instances)) - 1, **dict(zip(METRICS, values))}
        test_rows.append(row)
        cv2.imwrite(str(instance_dir / f"{stem}.png"), instances)
        cv2.imwrite(str(binary_dir / f"{stem}.png"), (instances > 0).astype(np.uint8) * 255)
        print(f"[test] {page_index:02d}/{len(test_loader):02d} {stem}: FM={row['FM']:.4f} PixelIU={row['Pixel_IU']:.4f}", flush=True)
    write_rows(args.output_root / "test/per_page_zottin.csv", test_rows)

    summary = {
        "subset": SUBSET,
        "model": f"Mask R-CNN + fixed {args.crop_arm} crop mask refiner",
        "crop_arm": args.crop_arm,
        "fixed_postproc": bool(args.fixed_postproc),
        "proposal_model": "Mask R-CNN",
        "proposal_arm": args.arm,
        "proposal_checkpoint": str(args.model_dir / "best.pt"),
        "proposal_checkpoint_epoch": int(checkpoint["epoch"]),
        "mask_refiner": str(segmenter_path),
        "mask_refiner_epoch": int(segmenter_checkpoint["epoch"]),
        "selection": "all proposal and mask post-processing parameters selected on validation only",
        "selected": {
            "score_threshold": score_threshold,
            "pad": int(best["pad"]),
            "mask_threshold": float(best["mask_threshold"]),
            "minimum_instance_area": int(best["minimum_instance_area"]),
            "close_fraction": args.close_fraction,
        },
        "validation_detector": selected_detector,
        "validation": {key: float(best[key]) for key in METRICS},
        "test": {"pages": len(test_rows), **aggregate(test_rows)},
        "output_dir": str(args.output_root),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
