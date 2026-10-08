#!/usr/bin/env python3
"""Validation-selected RT-DETR + crop U-Net refinement for Syr341.

The earlier post-processing sweep evaluated mask thresholds directly on test.
This script keeps the experiment usable in the thesis by selecting both detector
deduplication/confidence and crop-mask threshold on validation only.  Crop
probabilities are computed once and reused for every validation threshold.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import evaluate_loss_ablation_detr_udiads as base  # noqa: E402


SUBSET = "Syr341"
DETECTOR = "rtdetr_stock_random_v2"
ARM = "tversky"
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
MIN_INSTANCE_AREAS = (0, 100, 250, 500, 1000)
METRICS = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")


def split_dir(base_dir: Path, *names: str) -> Path:
    for name in names:
        candidate = base_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"None of {names} found below {base_dir}")


@torch.no_grad()
def crop_probability(crop_bgr: np.ndarray, model: torch.nn.Module,
                     crop_width: int, crop_height: int) -> np.ndarray:
    native_height, native_width = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
    tensor = (
        torch.from_numpy(np.ascontiguousarray(resized))
        .permute(2, 0, 1).float().unsqueeze(0).to(base.DEVICE) / 255.0
    )
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return cv2.resize(
        probability, (native_width, native_height), interpolation=cv2.INTER_LINEAR
    )


def render_thresholds(page: np.ndarray, boxes: np.ndarray, model: torch.nn.Module,
                      thresholds: tuple[float, ...], args) -> dict[float, np.ndarray]:
    """Render all thresholds from one set of crop-model forward passes."""
    height, width = page.shape[:2]
    canvases = {threshold: np.zeros((height, width), dtype=np.uint16)
                for threshold in thresholds}
    next_ids = {threshold: 1 for threshold in thresholds}

    for box in boxes:
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1 = max(0, x1 - args.pad), max(0, y1 - args.pad)
        x2, y2 = min(width, x2 + args.pad), min(height, y2 + args.pad)
        if x2 <= x1 or y2 <= y1:
            continue
        probability = crop_probability(
            page[y1:y2, x1:x2], model, args.crop_width, args.crop_height
        )
        for threshold in thresholds:
            crop_mask = base.connect_line(probability >= threshold, args.close_fraction).astype(bool)
            if not crop_mask.any():
                continue
            region = canvases[threshold][y1:y2, x1:x2]
            writable = crop_mask & (region == 0)
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
    filtered = np.where(keep[instances], instances, 0)
    _, relabeled = np.unique(filtered, return_inverse=True)
    return relabeled.reshape(instances.shape).astype(np.uint16)


def aggregate(rows: list[dict]) -> dict:
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def select_detector(raw: dict, paths: list[Path], gt_dir: Path) -> tuple[float, float, list[dict]]:
    rows = []
    confidence_values = np.round(np.arange(0.01, 0.32, 0.02), 2)
    for dedup_iou in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        confidence, frame = base.select_confidence(
            raw, paths, gt_dir, dedup_iou, confidence_values
        )
        selected = frame.loc[frame.confidence == confidence].iloc[0]
        rows.append({"dedup_iou": dedup_iou, **selected.to_dict()})
    best = max(rows, key=lambda row: (row["box_F1@0.5"], row["box_precision@0.5"]))
    return float(best["dedup_iou"]), float(best["confidence"]), rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "99_evaluation/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/syr341_val_selected_fm_target",
    )
    parser.add_argument("--crop-width", type=int, default=1024)
    parser.add_argument("--crop-height", type=int, default=256)
    parser.add_argument("--pad", type=int, default=15)
    parser.add_argument("--close-fraction", type=float, default=0.0)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    subset_root = REPO / "00_data/U-DIADS-TL" / SUBSET
    image_root = subset_root / f"img-{SUBSET}"
    gt_root = subset_root / f"text-line-gt-{SUBSET}"
    val_paths = base.image_paths(split_dir(image_root, "validation", "val"))
    test_paths = base.image_paths(split_dir(image_root, "test", "public-test"))
    val_gt = split_dir(gt_root, "validation", "val")
    test_gt = split_dir(gt_root, "test", "public-test")

    detector_dir = (
        REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/detection/rtdetr_hf"
        / SUBSET / DETECTOR / "best_model"
    )
    print("[detector] validation inference", flush=True)
    val_raw = base.raw_detections(detector_dir, val_paths)
    dedup_iou, confidence, detector_grid = select_detector(val_raw, val_paths, val_gt)
    write_rows(args.output_root / "validation_detector_grid.csv", detector_grid)
    print(
        f"[detector] selected dedup={dedup_iou:.2f}, confidence={confidence:.2f}",
        flush=True,
    )
    val_detections = base.finalized_detections(
        val_raw, val_paths, confidence, dedup_iou
    )
    del val_raw

    segmenter_path = (
        REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl"
        / "crop_seg_loss_ablation_components_1024x256" / SUBSET / ARM / "best.pth"
    )
    segmenter, checkpoint = base.load_segmenter(segmenter_path, base.DEVICE)
    validation_by_setting = {
        (threshold, minimum_area): []
        for threshold in THRESHOLDS for minimum_area in MIN_INSTANCE_AREAS
    }
    for page_path in val_paths:
        page = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        boxes, _ = val_detections[page_path.stem]
        rendered = render_thresholds(page, boxes, segmenter, THRESHOLDS, args)
        gt = cv2.imread(str(val_gt / f"{page_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        for threshold in THRESHOLDS:
            for minimum_area in MIN_INSTANCE_AREAS:
                instances = filter_instance_area(rendered[threshold], minimum_area)
                values = base.evaluate_metrics(gt, instances.astype(np.int32))
                validation_by_setting[(threshold, minimum_area)].append(
                    dict(zip(METRICS, values))
                )
        print(f"[validation] {page_path.stem}", flush=True)

    validation_grid = []
    for (threshold, minimum_area), page_rows in validation_by_setting.items():
        validation_grid.append({
            "segmenter_threshold": threshold,
            "minimum_instance_area": minimum_area,
            **aggregate(page_rows),
        })
    write_rows(args.output_root / "validation_mask_grid.csv", validation_grid)
    best = max(validation_grid, key=lambda row: (row["FM"], row["Pixel_IU"]))
    selected_threshold = float(best["segmenter_threshold"])
    selected_minimum_area = int(best["minimum_instance_area"])
    print(f"[validation] selected {best}", flush=True)

    del val_detections
    gc.collect()
    print("[detector] test inference", flush=True)
    test_raw = base.raw_detections(detector_dir, test_paths)
    test_detections = base.finalized_detections(
        test_raw, test_paths, confidence, dedup_iou
    )
    del test_raw

    instance_dir = args.output_root / "test/instances"
    binary_dir = args.output_root / "test/binary_masks"
    instance_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    test_rows = []
    for page_path in test_paths:
        page = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        boxes, _ = test_detections[page_path.stem]
        rendered = render_thresholds(
            page, boxes, segmenter, (selected_threshold,), args
        )[selected_threshold]
        instances = filter_instance_area(rendered, selected_minimum_area)
        gt = cv2.imread(str(test_gt / f"{page_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        values = base.evaluate_metrics(gt, instances.astype(np.int32))
        row = {
            "page": page_path.stem,
            "boxes": len(boxes),
            "instances": len(np.unique(instances)) - 1,
            **dict(zip(METRICS, values)),
        }
        test_rows.append(row)
        cv2.imwrite(str(instance_dir / f"{page_path.stem}.png"), instances)
        cv2.imwrite(
            str(binary_dir / f"{page_path.stem}.png"),
            (instances > 0).astype(np.uint8) * 255,
        )
        print(
            f"[test] {page_path.stem}: PixelIU={row['Pixel_IU']:.3f}, "
            f"FM={row['FM']:.3f}", flush=True,
        )
    write_rows(args.output_root / "test/per_page_zottin.csv", test_rows)

    summary = {
        "subset": SUBSET,
        "detector": DETECTOR,
        "segmenter": ARM,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection": "detector and mask post-processing selected on validation only",
        "selected": {
            "dedup_iou": dedup_iou,
            "confidence": confidence,
            "segmenter_threshold": selected_threshold,
            "minimum_instance_area": selected_minimum_area,
            "close_fraction": args.close_fraction,
        },
        "validation": {metric: float(best[metric]) for metric in METRICS},
        "test": {"pages": len(test_rows), **aggregate(test_rows)},
        "output_dir": str(args.output_root),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
