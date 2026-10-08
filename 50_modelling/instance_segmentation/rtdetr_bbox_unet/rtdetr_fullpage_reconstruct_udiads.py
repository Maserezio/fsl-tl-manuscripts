#!/usr/bin/env python3
"""Reconstruct U-DIADS line instances from full-page U-Net masks and RT-DETR boxes.

RT-DETR supplies line identities only.  A full-page PVT-B2 U-Net probability
map, optionally fused with thin ARU-Net support, supplies the pixel geometry.
Compact markers projected from validation-selected RT-DETR boxes partition the
semantic body with watershed.  Detector and reconstruction settings are chosen
on validation pages before the test split is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.segmentation import watershed


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INSTANCE_DIR = REPO / "50_modelling/instance_segmentation/mask_rcnn"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INSTANCE_DIR))

import evaluate_loss_ablation_detr_udiads as detector_base  # noqa: E402
from mask2former_udiads import zottin_metrics  # noqa: E402
from pixelwise_reconstruct_udiads import (  # noqa: E402
    Config as BodyConfig,
    METRICS,
    build_body,
    load_page,
    stems,
)


SUBSETS = ("Latin14396", "Latin2", "Syr341")
# Keep the detector controlled across subsets and avoid scratch/random weights.
DETECTOR_RUN = "rtdetr_convnext_tiny_imagenet"


@dataclass(frozen=True)
class ReconstructionConfig:
    threshold: float
    body: str
    baseline_erode: int
    seed_radius: int
    fallback_components: bool
    minimum_instance_area: int

    @property
    def name(self) -> str:
        return (
            f"body-{self.body}_t{self.threshold:.2f}_e{self.baseline_erode}_"
            f"r{self.seed_radius}_fallback-{int(self.fallback_components)}_"
            f"a{self.minimum_instance_area}"
        )


def split_dir(root: Path, split: str) -> Path:
    names = ("validation", "val") if split == "val" else ("test", "public-test")
    return next(path for name in names if (path := root / name).is_dir())


def image_paths(subset: str, split: str) -> list[Path]:
    root = REPO / "00_data/U-DIADS-TL" / subset / f"img-{subset}"
    return detector_base.image_paths(split_dir(root, split))


def gt_dir(subset: str, split: str) -> Path:
    root = REPO / "00_data/U-DIADS-TL" / subset / f"text-line-gt-{subset}"
    return split_dir(root, split)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def cached_raw_detections(subset: str, split: str, paths: list[Path],
                          cache_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache_dir = cache_root / "detections" / subset / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    missing = []
    for path in paths:
        cached = cache_dir / f"{path.stem}.npz"
        if cached.exists():
            values = np.load(cached)
            result[path.stem] = (values["boxes"], values["scores"])
        else:
            missing.append(path)
    if missing:
        model_dir = (
            REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/detection/rtdetr_hf"
            / subset / DETECTOR_RUN / "best_model"
        )
        inferred = detector_base.raw_detections(model_dir, missing)
        for stem, (boxes, scores) in inferred.items():
            np.savez_compressed(cache_dir / f"{stem}.npz", boxes=boxes, scores=scores)
            result[stem] = boxes, scores
    return result


def select_detector(raw: dict, paths: list[Path], truth_dir: Path) -> tuple[float, float, list[dict]]:
    rows = []
    confidence_values = np.round(np.arange(0.01, 0.32, 0.02), 2)
    for dedup_iou in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        confidence, frame = detector_base.select_confidence(
            raw, paths, truth_dir, dedup_iou, confidence_values
        )
        selected = frame.loc[frame.confidence == confidence].iloc[0]
        rows.append({"dedup_iou": dedup_iou, **selected.to_dict()})
    best = max(rows, key=lambda row: (row["box_F1@0.5"], row["box_precision@0.5"]))
    return float(best["dedup_iou"]), float(best["confidence"]), rows


def marker_map(body: np.ndarray, probability: np.ndarray, boxes: np.ndarray,
               radius: int) -> np.ndarray:
    """Project one compact, unique marker from every detection onto the body."""
    height, width = body.shape
    markers = np.zeros(body.shape, dtype=np.int32)
    for label, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        candidate = body[y1:y2, x1:x2] & (markers[y1:y2, x1:x2] == 0)
        ys, xs = np.nonzero(candidate)
        if not len(xs):
            continue
        center_x = 0.5 * (x1 + x2) - x1
        center_y = 0.5 * (y1 + y2) - y1
        scale_x = max(1.0, 0.5 * (x2 - x1))
        scale_y = max(1.0, 0.5 * (y2 - y1))
        spatial = ((xs - center_x) / scale_x) ** 2 + ((ys - center_y) / scale_y) ** 2
        confidence_penalty = 0.20 * (1.0 - probability[y1 + ys, x1 + xs])
        order = np.argsort(spatial + confidence_penalty)
        seed_y, seed_x = int(y1 + ys[order[0]]), int(x1 + xs[order[0]])
        # Draw locally: allocating a full-page disk for every one of ~200 boxes
        # dominates the validation sweep on dense Syr341 pages.
        local_disk = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.circle(local_disk, (seed_x - x1, seed_y - y1), radius, 1, thickness=-1)
        local_markers = markers[y1:y2, x1:x2]
        writable = (local_disk > 0) & body[y1:y2, x1:x2] & (local_markers == 0)
        if not writable.any():
            markers[seed_y, seed_x] = label
        else:
            local_markers[writable] = label
    return markers


def reconstruct(body: np.ndarray, probability: np.ndarray, baseline: np.ndarray,
                boxes: np.ndarray, config: ReconstructionConfig) -> np.ndarray:
    markers = marker_map(body, probability, boxes, config.seed_radius)
    n_body, body_cc, stats, _ = cv2.connectedComponentsWithStats(
        body.astype(np.uint8), connectivity=8
    )
    output = np.zeros(body.shape, dtype=np.int32)
    next_fallback = int(markers.max()) + 1
    confidence = np.clip(0.8 * probability + 0.2 * baseline, 0.0, 1.0)
    for component_id in range(1, n_body):
        if stats[component_id, cv2.CC_STAT_AREA] < 20:
            continue
        x, y, width, height = (int(value) for value in stats[component_id, :4])
        component = body_cc[y:y + height, x:x + width] == component_id
        local_markers = np.where(component, markers[y:y + height, x:x + width], 0)
        labels = np.unique(local_markers)
        labels = labels[labels > 0]
        target = output[y:y + height, x:x + width]
        if len(labels) == 0:
            if config.fallback_components:
                target[component] = next_fallback
                next_fallback += 1
            continue
        if len(labels) == 1:
            target[component] = int(labels[0])
            continue
        partition = watershed(
            1.0 - confidence[y:y + height, x:x + width],
            markers=local_markers,
            mask=component,
            watershed_line=True,
        )
        target[component] = partition[component]

    labels, counts = np.unique(output, return_counts=True)
    keep = labels[(labels > 0) & (counts >= config.minimum_instance_area)]
    filtered = np.where(np.isin(output, keep), output, 0)
    _, compact = np.unique(filtered, return_inverse=True)
    return compact.reshape(output.shape).astype(np.uint16)


def candidates() -> list[ReconstructionConfig]:
    result = []
    for threshold in (0.40, 0.50, 0.60):
        for body in ("semantic_seam", "semantic_aru_seam"):
            erosions = (8,) if body == "semantic_seam" else (4, 8, 12)
            for erosion in erosions:
                for radius in (1, 3):
                    for fallback in (False, True):
                        for area in (20, 100, 250):
                            result.append(ReconstructionConfig(
                                threshold, body, erosion, radius, fallback, area
                            ))
    return result


def body_for(probability: np.ndarray, baseline: np.ndarray, subset: str,
             config: ReconstructionConfig) -> np.ndarray:
    body_config = BodyConfig(
        threshold=config.threshold,
        body=config.body,
        instances="components",
        baseline_erode=config.baseline_erode,
    )
    body, _ = build_body(probability, baseline, subset, body_config)
    return body


def run_subset(subset: str, output_root: Path) -> dict:
    val_paths = image_paths(subset, "val")
    test_paths = image_paths(subset, "test")
    val_raw = cached_raw_detections(subset, "val", val_paths, output_root)
    dedup_iou, confidence, detector_grid = select_detector(
        val_raw, val_paths, gt_dir(subset, "val")
    )
    write_rows(output_root / subset / "validation_detector_grid.csv", detector_grid)
    val_boxes = detector_base.finalized_detections(
        val_raw, val_paths, confidence, dedup_iou
    )
    val_pages = {stem: load_page(subset, "val", stem) for stem in stems(subset, "val")}

    rows = []
    configs = candidates()
    body_cache = {}
    body_keys = {
        (config.threshold, config.body, config.baseline_erode) for config in configs
    }
    for threshold, body_name, erosion in body_keys:
        body_config = ReconstructionConfig(
            threshold, body_name, erosion, seed_radius=1,
            fallback_components=False, minimum_instance_area=20,
        )
        body_cache[(threshold, body_name, erosion)] = {
            stem: body_for(probability, baseline, subset, body_config)
            for stem, (probability, baseline, _maskrcnn, _gt) in val_pages.items()
        }
    for index, config in enumerate(configs, start=1):
        page_scores = []
        for stem, (probability, baseline, _maskrcnn, gt) in val_pages.items():
            body = body_cache[(config.threshold, config.body, config.baseline_erode)][stem]
            pred = reconstruct(body, probability, baseline, val_boxes[stem][0], config)
            page_scores.append(zottin_metrics(gt, pred))
        mean = aggregate(page_scores)
        balanced = float(np.sqrt(max(0.0, mean["Pixel_IU"] * mean["FM"])))
        rows.append({"subset": subset, **asdict(config), **mean, "balanced": balanced})
        if index % 24 == 0:
            print(f"[{subset}] validation {index}/{len(configs)}", flush=True)
    write_rows(output_root / subset / "validation_reconstruction_grid.csv", rows)
    best_row = max(rows, key=lambda row: (row["balanced"], row["FM"], row["Pixel_IU"]))
    keys = tuple(asdict(configs[0]))
    selected = ReconstructionConfig(**{key: best_row[key] for key in keys})
    del val_pages, val_raw, val_boxes, body_cache
    gc.collect()

    test_raw = cached_raw_detections(subset, "test", test_paths, output_root)
    test_boxes = detector_base.finalized_detections(
        test_raw, test_paths, confidence, dedup_iou
    )
    instance_dir = output_root / subset / selected.name / "test/instances"
    binary_dir = output_root / subset / selected.name / "test/binary_masks"
    instance_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    test_rows = []
    for stem in stems(subset, "test"):
        probability, baseline, _maskrcnn, gt = load_page(subset, "test", stem)
        body = body_for(probability, baseline, subset, selected)
        pred = reconstruct(body, probability, baseline, test_boxes[stem][0], selected)
        metrics = zottin_metrics(gt, pred)
        test_rows.append({
            "page": stem,
            "detections": len(test_boxes[stem][0]),
            "instances": int(pred.max()),
            **metrics,
        })
        cv2.imwrite(str(instance_dir / f"{stem}.png"), pred)
        cv2.imwrite(str(binary_dir / f"{stem}.png"), (pred > 0).astype(np.uint8) * 255)
    write_rows(output_root / subset / selected.name / "test/per_page_zottin.csv", test_rows)
    result = {
        "subset": subset,
        "detector": DETECTOR_RUN,
        "selection": "detector and reconstruction parameters selected on validation only",
        "detector_selected": {"dedup_iou": dedup_iou, "confidence": confidence},
        "reconstruction_selected": asdict(selected),
        "validation": {key: float(best_row[key]) for key in (*METRICS, "balanced")},
        "test": {"pages": len(test_rows), **aggregate(test_rows)},
        "output_dir": str(instance_dir.parent),
    }
    summary_path = output_root / subset / selected.name / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=["all"])
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "99_evaluation/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/rtdetr_fullpage_reconstruction",
    )
    args = parser.parse_args()
    subsets = list(SUBSETS) if args.subsets == ["all"] else args.subsets
    unknown = set(subsets) - set(SUBSETS)
    if unknown:
        parser.error(f"unknown subsets: {sorted(unknown)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = [run_subset(subset, args.output_root) for subset in subsets]
    macro = {
        key: float(np.mean([result["test"][key] for result in results]))
        for key in METRICS
    }
    summary = {
        "method": "RT-DETR identities + full-page PVT-B2 U-Net reconstruction",
        "detector": DETECTOR_RUN,
        "subsets": results,
        "macro_test": macro,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
