#!/usr/bin/env python3
"""Recover U-DIADS line instances from binary PVT-B2 predictions.

The COCO annotations in this repository were generated from U-DIADS' native
binary masks, so converting the polygons back to a binary target adds no new
training signal.  This experiment instead uses the information that *is*
different between the two representations:

* the PVT-B2 U-Net probability map supplies the full text-line body;
* pixel-wise subtraction identifies body pixels missed by Mask R-CNN;
* Mask R-CNN instance masks supply identity markers for those residual pixels;
* ARU-Net supplies thin support and fallback markers for completely missed lines.

All parameter selection is performed on validation pages.  Test pages are read
only after the per-subset configuration has been selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEG_DIR = REPO / "50_modelling/semantic_segmentation/unet"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SEG_DIR))

from mask2former_udiads import zottin_metrics  # noqa: E402
from postproc import PER_MS_PARAMS, remove_small_objects, run_pipeline  # noqa: E402


SUBSETS = ("Latin14396", "Latin2", "Syr341")
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
MASKRCNN_RUN = "maskrcnn_pvt_v2_b2_imagenet_1024_200ep"
SEMANTIC_RUN = "unet_tu-pvt_v2_b2_sz_udiads_{subset}"
METRICS = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")


@dataclass(frozen=True)
class Config:
    threshold: float
    body: str
    instances: str
    baseline_erode: int = 8
    minimum_marker_area: int = 20

    @property
    def name(self) -> str:
        name = (
            f"body-{self.body}_instances-{self.instances}_"
            f"t{self.threshold:.2f}_e{self.baseline_erode}"
        )
        if self.instances == "maskrcnn_global_seeds":
            name += f"_a{self.minimum_marker_area}"
        return name


def split_alias(split: str) -> str:
    return "validation" if split == "val" else split


def gt_dir(subset: str, split: str) -> Path:
    return (
        REPO / "00_data/U-DIADS-TL" / subset / f"text-line-gt-{subset}"
        / split_alias(split)
    )


def stems(subset: str, split: str) -> list[str]:
    return sorted(path.stem for path in gt_dir(subset, split).iterdir()
                  if path.suffix.lower() in EXTENSIONS)


def probability_path(subset: str, stem: str) -> Path:
    return (
        REPO / "99_evaluation/semantic_segmentation/unet/u-diads-tl/prob_cache"
        / SEMANTIC_RUN.format(subset=subset) / f"{stem}.npy"
    )


def baseline_path(subset: str, split: str, stem: str) -> Path:
    cache = "arunet_baseline_cache_validation" if split == "val" else "arunet_baseline_cache"
    return (
        REPO / "99_evaluation/semantic_segmentation/unet/u-diads-tl"
        / cache / subset / f"{stem}.npy"
    )


def maskrcnn_path(subset: str, split: str, stem: str) -> Path:
    return (
        REPO / "99_evaluation/instance_segmentation/mask_rcnn/u-diads-tl"
        / subset / MASKRCNN_RUN / split_alias(split) / "instances" / f"{stem}.png"
    )


def load_page(subset: str, split: str, stem: str) -> tuple[np.ndarray, ...]:
    probability = np.load(probability_path(subset, stem)).astype(np.float32)
    baseline = np.load(baseline_path(subset, split, stem)).astype(np.float32)
    maskrcnn = cv2.imread(str(maskrcnn_path(subset, split, stem)), cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(str(gt_dir(subset, split) / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if maskrcnn is None or gt is None:
        raise FileNotFoundError(f"Missing prediction or GT for {subset}/{split}/{stem}")
    if not (probability.shape == baseline.shape == maskrcnn.shape == gt.shape):
        raise ValueError(
            f"Shape mismatch for {subset}/{split}/{stem}: "
            f"prob={probability.shape}, baseline={baseline.shape}, "
            f"maskrcnn={maskrcnn.shape}, gt={gt.shape}"
        )
    return probability, baseline, maskrcnn.astype(np.int32), gt > 0


def thin_baseline(baseline: np.ndarray, erosion: int) -> np.ndarray:
    baseline_u8 = np.clip(baseline * 255.0, 0, 255).astype(np.uint8)
    otsu = cv2.threshold(baseline_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, erosion)))
    return cv2.erode(otsu, kernel, iterations=1) > 0


def build_body(probability: np.ndarray, baseline: np.ndarray, subset: str,
               config: Config) -> tuple[np.ndarray, np.ndarray]:
    semantic = remove_small_objects(probability > config.threshold, min_size=50).astype(bool)
    baseline_thin = thin_baseline(baseline, config.baseline_erode)

    if config.body == "semantic":
        body = semantic
    elif config.body == "semantic_seam":
        body = run_pipeline(semantic, PER_MS_PARAMS[subset]).astype(bool)
    elif config.body == "semantic_aru_seam":
        fused = remove_small_objects(semantic | baseline_thin, min_size=500)
        body = run_pipeline(fused, PER_MS_PARAMS[subset]).astype(bool)
    else:
        raise ValueError(f"Unknown body mode: {config.body}")
    return body, baseline_thin


def _markers_for_component(
    component: np.ndarray,
    maskrcnn: np.ndarray,
    baseline_thin: np.ndarray,
    include_baseline_fallback: bool,
    minimum_marker_area: int = 20,
) -> np.ndarray:
    """Create compact, unique watershed markers inside one semantic component."""
    markers = np.zeros(component.shape, dtype=np.int32)
    next_marker = 1

    labels, counts = np.unique(maskrcnn[component], return_counts=True)
    for label, count in zip(labels, counts):
        if label == 0 or count < minimum_marker_area:
            continue
        seed = component & (maskrcnn == label)
        # One detector label can occur in disconnected pieces after restriction;
        # keep its largest piece as a stable identity marker.
        n_seed, seed_cc, stats, _ = cv2.connectedComponentsWithStats(
            seed.astype(np.uint8), connectivity=8
        )
        if n_seed <= 1:
            continue
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        markers[seed_cc == largest] = next_marker
        next_marker += 1

    if next_marker == 1 and include_baseline_fallback:
        # Mask R-CNN missed this entire body component.  ARU-Net can still
        # provide multiple line markers when the semantic body merged lines.
        fallback = component & baseline_thin
        n_base, base_cc, stats, _ = cv2.connectedComponentsWithStats(
            fallback.astype(np.uint8), connectivity=8
        )
        for label in range(1, n_base):
            if stats[label, cv2.CC_STAT_AREA] < minimum_marker_area:
                continue
            markers[base_cc == label] = next_marker
            next_marker += 1

    if next_marker == 1:
        # Last-resort marker at the component's medial maximum.
        distance = ndimage.distance_transform_edt(component)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        markers[y, x] = 1
    return markers


def seeded_reconstruction(
    body: np.ndarray,
    probability: np.ndarray,
    baseline: np.ndarray,
    baseline_thin: np.ndarray,
    maskrcnn: np.ndarray,
    include_baseline_fallback: bool,
) -> np.ndarray:
    """Assign semantic residual pixels to detector/baseline identity markers."""
    n_body, body_cc, stats, _ = cv2.connectedComponentsWithStats(
        body.astype(np.uint8), connectivity=8
    )
    output = np.zeros(body.shape, dtype=np.int32)
    next_output = 1
    confidence = np.clip(0.8 * probability + 0.2 * baseline, 0.0, 1.0)

    for component_id in range(1, n_body):
        if stats[component_id, cv2.CC_STAT_AREA] < 20:
            continue
        x = stats[component_id, cv2.CC_STAT_LEFT]
        y = stats[component_id, cv2.CC_STAT_TOP]
        w = stats[component_id, cv2.CC_STAT_WIDTH]
        h = stats[component_id, cv2.CC_STAT_HEIGHT]
        local_component = body_cc[y:y + h, x:x + w] == component_id
        local_markers = _markers_for_component(
            local_component,
            maskrcnn[y:y + h, x:x + w],
            baseline_thin[y:y + h, x:x + w],
            include_baseline_fallback,
        )
        local_labels = np.unique(local_markers)
        local_labels = local_labels[local_labels > 0]
        if len(local_labels) == 1:
            output[y:y + h, x:x + w][local_component] = next_output
            next_output += 1
            continue

        partition = watershed(
            1.0 - confidence[y:y + h, x:x + w],
            markers=local_markers,
            mask=local_component,
            watershed_line=True,
        )
        for label in local_labels:
            region = partition == label
            if region.sum() < 20:
                continue
            output[y:y + h, x:x + w][region] = next_output
            next_output += 1
    return output


def global_seeded_reconstruction(
    body: np.ndarray,
    probability: np.ndarray,
    baseline: np.ndarray,
    maskrcnn: np.ndarray,
    minimum_marker_area: int,
) -> np.ndarray:
    """Expand detector identities without duplicating them across body fragments.

    The first reconstruction version assigned a fresh ID inside every connected
    semantic component.  A single Mask R-CNN instance crossing several broken
    semantic components therefore became several predictions.  Here the
    detector label is kept globally: disconnected residual pieces attached to
    the same detector mask remain one instance for the Zottin evaluator.
    """
    n_body, body_cc, stats, _ = cv2.connectedComponentsWithStats(
        body.astype(np.uint8), connectivity=8
    )
    output = np.zeros(body.shape, dtype=np.int32)
    next_output = int(maskrcnn.max()) + 1
    confidence = np.clip(0.8 * probability + 0.2 * baseline, 0.0, 1.0)

    for component_id in range(1, n_body):
        if stats[component_id, cv2.CC_STAT_AREA] < 20:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        local_component = body_cc[y:y + h, x:x + w] == component_id
        local_detector = maskrcnn[y:y + h, x:x + w]
        markers = np.where(local_component, local_detector, 0).astype(np.int32)

        labels, counts = np.unique(markers[markers > 0], return_counts=True)
        keep = labels[counts >= minimum_marker_area]
        markers[~np.isin(markers, keep)] = 0
        if len(keep) == 0:
            distance = ndimage.distance_transform_edt(local_component)
            marker_y, marker_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            markers[marker_y, marker_x] = next_output
            keep = np.array([next_output], dtype=np.int32)
            next_output += 1

        target = output[y:y + h, x:x + w]
        if len(keep) == 1:
            target[local_component] = int(keep[0])
        else:
            partition = watershed(
                1.0 - confidence[y:y + h, x:x + w],
                markers=markers,
                mask=local_component,
                watershed_line=True,
            )
            target[local_component] = partition[local_component]
    return output


def predict_instances(probability: np.ndarray, baseline: np.ndarray,
                      maskrcnn: np.ndarray, subset: str, config: Config) -> np.ndarray:
    body, baseline_thin = build_body(probability, baseline, subset, config)
    if config.instances == "components":
        _, labels = cv2.connectedComponents(body.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32)
    if config.instances == "maskrcnn_seeds":
        return seeded_reconstruction(
            body, probability, baseline, baseline_thin, maskrcnn,
            include_baseline_fallback=False,
        )
    if config.instances == "maskrcnn_arunet_seeds":
        return seeded_reconstruction(
            body, probability, baseline, baseline_thin, maskrcnn,
            include_baseline_fallback=True,
        )
    if config.instances == "maskrcnn_global_seeds":
        return global_seeded_reconstruction(
            body, probability, baseline, maskrcnn, config.minimum_marker_area,
        )
    raise ValueError(f"Unknown instance mode: {config.instances}")


def candidate_configs() -> list[Config]:
    return [
        Config(threshold, body, instances, erosion)
        for threshold in (0.40, 0.50, 0.60)
        for body in ("semantic", "semantic_seam", "semantic_aru_seam")
        for instances in ("components", "maskrcnn_seeds", "maskrcnn_arunet_seeds")
        for erosion in ((4, 8, 12) if body == "semantic_aru_seam" else (8,))
    ]


def aggregate(page_metrics: list[dict]) -> dict:
    return {metric: float(np.mean([row[metric] for row in page_metrics])) for metric in METRICS}


def validation_grid(subset: str) -> tuple[Config, Config, list[dict]]:
    pages = {stem: load_page(subset, "val", stem) for stem in stems(subset, "val")}
    rows = []
    for config in candidate_configs():
        scores = []
        for probability, baseline, maskrcnn, gt in pages.values():
            pred = predict_instances(probability, baseline, maskrcnn, subset, config)
            scores.append(zottin_metrics(gt, pred))
        mean = aggregate(scores)
        # Instance segmentation needs both good shapes and good identities.
        balanced = float(np.sqrt(max(0.0, mean["Pixel_IU"] * mean["FM"])))
        rows.append({"subset": subset, **asdict(config), **mean, "balanced": balanced})

    best_balanced_row = max(rows, key=lambda row: (row["balanced"], row["Pixel_IU"], row["FM"]))
    best_pixel_row = max(rows, key=lambda row: (row["Pixel_IU"], row["FM"]))
    keys = ("threshold", "body", "instances", "baseline_erode")
    best_balanced = Config(**{key: best_balanced_row[key] for key in keys})
    best_pixel = Config(**{key: best_pixel_row[key] for key in keys})
    return best_balanced, best_pixel, rows


def evaluate_and_save(subset: str, config: Config, output_root: Path,
                      selection: str, semantic_run: str | None = None) -> dict:
    output_dir = output_root / subset / config.name / "test"
    instance_dir = output_dir / "instances"
    binary_dir = output_dir / "binary_masks"
    residual_dir = output_dir / "recovered_residual"
    for directory in (instance_dir, binary_dir, residual_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for stem in stems(subset, "test"):
        probability, baseline, maskrcnn, gt = load_page(subset, "test", stem)
        if semantic_run is not None:
            probability = np.load(
                REPO / "99_evaluation/semantic_segmentation/unet/u-diads-tl/prob_cache"
                / semantic_run / f"{stem}.npy"
            ).astype(np.float32)
        pred = predict_instances(probability, baseline, maskrcnn, subset, config)
        binary = pred > 0
        residual = binary & (maskrcnn == 0)
        metrics = zottin_metrics(gt, pred)
        rows.append({
            "page": stem,
            "instances": int(pred.max()),
            "recovered_pixels": int(residual.sum()),
            **metrics,
        })
        cv2.imwrite(str(instance_dir / f"{stem}.png"), pred.astype(np.uint16))
        cv2.imwrite(str(binary_dir / f"{stem}.png"), binary.astype(np.uint8) * 255)
        cv2.imwrite(str(residual_dir / f"{stem}.png"), residual.astype(np.uint8) * 255)

    with (output_dir / "zottin_per_page.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "subset": subset,
        "selection": selection,
        "config": asdict(config),
        "semantic_run": semantic_run or SEMANTIC_RUN.format(subset=subset),
        "test": {"pages": len(rows), **aggregate(rows)},
        "mean_recovered_pixels": float(np.mean([row["recovered_pixels"] for row in rows])),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_validation_grid(path: Path) -> list[dict]:
    """Read a prior validation sweep so test evaluation can be resumed cheaply."""
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("threshold", "Pixel_IU", "Line_IU", "DR", "RA", "FM", "balanced"):
                row[key] = float(row[key])
            row["baseline_erode"] = int(row["baseline_erode"])
            row["minimum_marker_area"] = int(row.get("minimum_marker_area", 20))
            rows.append(row)
    return rows


def selected_configs(rows: list[dict]) -> tuple[Config, Config]:
    best_balanced_row = max(rows, key=lambda row: (row["balanced"], row["Pixel_IU"], row["FM"]))
    best_pixel_row = max(rows, key=lambda row: (row["Pixel_IU"], row["FM"]))
    keys = ("threshold", "body", "instances", "baseline_erode", "minimum_marker_area")
    best_balanced = Config(**{key: best_balanced_row[key] for key in keys})
    best_pixel = Config(**{key: best_pixel_row[key] for key in keys})
    return best_balanced, best_pixel


def run_syr341_fm_target(
    output_root: Path,
    semantic_run: str | None = None,
) -> None:
    """Validation-only search for the identity-preserving Syr341 variant."""
    subset = "Syr341"
    pages = {stem: load_page(subset, "val", stem) for stem in stems(subset, "val")}
    if semantic_run is not None:
        for stem, (_probability, baseline, maskrcnn, gt) in list(pages.items()):
            probability = np.load(
                REPO / "99_evaluation/semantic_segmentation/unet/u-diads-tl/prob_cache"
                / semantic_run / f"{stem}.npy"
            ).astype(np.float32)
            pages[stem] = probability, baseline, maskrcnn, gt
    rows = []
    for threshold in (0.40, 0.425, 0.45, 0.475, 0.50):
        for marker_area in (50, 75, 100, 125, 150, 200):
            config = Config(
                threshold=threshold,
                body="semantic_aru_seam",
                instances="maskrcnn_global_seeds",
                baseline_erode=8,
                minimum_marker_area=marker_area,
            )
            scores = []
            for probability, baseline, maskrcnn, gt in pages.values():
                pred = predict_instances(probability, baseline, maskrcnn, subset, config)
                scores.append(zottin_metrics(gt, pred))
            mean = aggregate(scores)
            row = {"subset": subset, **asdict(config), **mean}
            rows.append(row)
            print(
                f"[{config.name}] val PixelIU={mean['Pixel_IU']:.4f} "
                f"FM={mean['FM']:.4f}", flush=True,
            )

    grid_path = output_root / "Syr341_global_identity_validation_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_row = max(rows, key=lambda row: (row["FM"], row["Pixel_IU"]))
    keys = ("threshold", "body", "instances", "baseline_erode", "minimum_marker_area")
    best = Config(**{key: best_row[key] for key in keys})
    test = evaluate_and_save(
        subset, best, output_root, "Syr341 validation FM (identity-preserving seeds)",
        semantic_run=semantic_run,
    )
    result = {
        "method": "semantic + ARU-Net body, identity-preserving PVT-B2 Mask R-CNN seeds",
        "semantic_run": semantic_run or SEMANTIC_RUN.format(subset=subset),
        "validation": {key: best_row[key] for key in (*keys, *METRICS)},
        "test": test,
        "validation_grid": str(grid_path),
    }
    summary_path = output_root / "syr341_global_identity_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO / "99_evaluation/instance_segmentation/mask_rcnn/u-diads-tl/pixelwise_reconstruction",
    )
    parser.add_argument(
        "--reuse-validation-grid", action="store_true",
        help="Reuse existing per-subset validation CSVs and only regenerate test outputs.",
    )
    parser.add_argument(
        "--syr341-fm-target", action="store_true",
        help="Run the focused validation sweep for identity-preserving Syr341 seeds.",
    )
    parser.add_argument(
        "--syr341-maskrcnn-baseline-target", action="store_true",
        help="Tune Mask R-CNN completion using the validation-best ConvNeXt semantic baseline.",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.syr341_fm_target:
        run_syr341_fm_target(args.output_root)
        return
    if args.syr341_maskrcnn_baseline_target:
        semantic_run = "unet_tu-convnext_tiny_sz_udiads_Syr341"
        target_root = args.output_root / "maskrcnn_convnext_semantic_baseline"
        target_root.mkdir(parents=True, exist_ok=True)
        run_syr341_fm_target(target_root, semantic_run=semantic_run)
        return

    summaries = []
    for subset in args.subsets:
        grid_path = args.output_root / f"{subset}_validation_grid.csv"
        if args.reuse_validation_grid and grid_path.exists():
            grid = read_validation_grid(grid_path)
            best, best_pixel = selected_configs(grid)
        else:
            best, best_pixel, grid = validation_grid(subset)
            with grid_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(grid[0]))
                writer.writeheader()
                writer.writerows(grid)
        print(f"[{subset}] balanced={best.name}; PixelIU-best={best_pixel.name}", flush=True)
        balanced_result = evaluate_and_save(
            subset, best, args.output_root,
            "validation geometric mean of Pixel_IU and FM",
        )
        pixel_result = (
            balanced_result if best_pixel == best else evaluate_and_save(
                subset, best_pixel, args.output_root, "validation Pixel_IU",
            )
        )
        summaries.append({
            "subset": subset,
            "balanced_selection": balanced_result,
            "pixel_iou_selection": pixel_result,
        })

    macro_balanced = {
        metric: float(np.mean([summary["balanced_selection"]["test"][metric]
                               for summary in summaries]))
        for metric in METRICS
    }
    macro_pixel = {
        metric: float(np.mean([summary["pixel_iou_selection"]["test"][metric]
                               for summary in summaries]))
        for metric in METRICS
    }
    result = {
        "method": "PVT-B2 semantic body + pixelwise residual instance reconstruction",
        "subsets": summaries,
        "macro_test_balanced_selection": macro_balanced,
        "macro_test_pixel_iou_selection": macro_pixel,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
