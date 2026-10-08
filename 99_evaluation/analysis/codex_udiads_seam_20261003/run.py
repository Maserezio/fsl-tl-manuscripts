#!/usr/bin/env python3
"""Paired U-DIADS-TL seam experiment on cached Mask R-CNN proposals.

Detector score, crop model, pad, and mask threshold are fixed to the published
Mask R-CNN + crop-refinement run. Only mask reconstruction changes. Select K
on validation separately for filled and ink-gated seams; visit test afterward.
"""
from __future__ import annotations
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "50_modelling/instance_segmentation/mask_rcnn"),
                str(REPO / "50_modelling/instance_segmentation/dp_seam")]
import maskrcnn_syr341_crop_refine as base  # noqa: E402
from seam_in_box import seam_polygon  # noqa: E402

SUBSETS = ("Latin14396", "Latin2", "Syr341")
KS = (2, 8, 64)
MODES = ("baseline", "filled", "ink_gated")
METRICS = base.METRICS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paths(sub: str):
    eval_root = (REPO / "99_evaluation/instance_segmentation/mask_rcnn/u-diads-tl" / sub
                 / "maskrcnn_convnext_tiny_catmus_1024_200ep_croprefine_tversky_fixed")
    ckpt = (REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/crop_seg_loss_ablation_components_1024x256"
            / sub / "tversky/best.pth")
    data_root = REPO / "00_data/U-DIADS-TL" / sub
    return eval_root, ckpt, data_root


def available_stems(eval_root: Path, split: str):
    dirname = "validation" if split == "val" else "test"
    return sorted(p.stem for p in (eval_root / "detections" / dirname).glob("*.npz"))


def render(page: np.ndarray, detections: list[tuple[np.ndarray, float]], model,
           pad: int, threshold: float, choice: dict | None = None):
    """Render all modes together to share precisely the same crop probabilities."""
    h, w = page.shape[:2]
    keys = ([("baseline", 0)] + [(mode, k) for mode in MODES[1:] for k in KS]
            if choice is None else [(mode, choice[mode]) for mode in MODES])
    canvases = {key: np.zeros((h, w), dtype=np.uint16) for key in keys}
    next_ids = {key: 1 for key in keys}
    for box, _score in detections:
        x0, y0, x1, y1 = np.rint(box).astype(int)
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        probability = base.crop_probability(page[y0:y1, x0:x1], model, 1024, 256)
        ink = probability >= threshold
        masks = {("baseline", 0): base.crop_base.connect_line(ink, 0.0).astype(bool)}
        for k in sorted({k for mode, k in keys if mode != "baseline"}):
            polygon = seam_polygon(probability, threshold, k)
            band = np.zeros(probability.shape, dtype=np.uint8)
            if polygon is not None and len(polygon) >= 3:
                cv2.fillPoly(band, [polygon], 1)
            if ("filled", k) in canvases:
                masks[("filled", k)] = band.astype(bool)
            if ("ink_gated", k) in canvases:
                masks[("ink_gated", k)] = ink & band.astype(bool)
        for key, mask in masks.items():
            if not mask.any():
                continue
            region = canvases[key][y0:y1, x0:x1]
            writable = mask & (region == 0)
            if writable.any():
                region[writable] = next_ids[key]
                next_ids[key] += 1
    return canvases


def evaluate_split(sub: str, split: str, model, eval_root: Path, data_root: Path,
                   selected: dict, output: Path, save_examples: bool = False,
                   choice: dict | None = None):
    image_root = data_root / f"img-{sub}"
    image_split = (next(name for name in ("val", "validation") if (image_root / name).is_dir())
                   if split == "val" else "test")
    gt_split = "validation" if split == "val" else "test"
    rows = []
    stems = available_stems(eval_root, split)
    for index, stem in enumerate(stems, start=1):
        page = cv2.imread(str(image_root / image_split / f"{stem}.jpg"))
        gt = cv2.imread(str(data_root / f"text-line-gt-{sub}" / gt_split / f"{stem}.png"),
                        cv2.IMREAD_GRAYSCALE)
        if page is None or gt is None:
            raise FileNotFoundError(f"Missing image or GT for {sub}/{split}/{stem}")
        saved = np.load(eval_root / "detections" / gt_split / f"{stem}.npz")
        detections = list(zip(saved["boxes"][saved["scores"] >= selected["score_threshold"]],
                              saved["scores"][saved["scores"] >= selected["score_threshold"]]))
        canvases = render(page, detections, model, selected["pad"], selected["mask_threshold"], choice)
        for (mode, k), instances in canvases.items():
            values = base.evaluate_contiguous(gt, instances)
            rows.append({"subset": sub, "split": split, "page": stem, "mode": mode, "K": k,
                         "boxes": len(detections), "instances": int(instances.max()),
                         **dict(zip(METRICS, values))})
            if save_examples and index <= 2:
                d = output / "examples" / sub / split
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{stem}_{mode}_k{k}.png"), instances)
        print(f"[{sub}/{split}] {index:02d}/{len(stems):02d} {stem}: {len(detections)} boxes", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    return {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite results: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"device": base.crop_base.DEVICE, "ks": KS, "subsets": {},
                "protocol": "Saved detector boxes and original crop model; detector score, pad and mask "
                            "threshold fixed from prior validation. K selected on validation. Test once."}
    for sub in args.subsets:
        base.SUBSET = sub
        eval_root, ckpt, data_root = paths(sub)
        prior = json.loads((eval_root / "summary.json").read_text())
        selected = prior["selected"]
        model, _ = base.crop_base.load_segmenter(ckpt, base.crop_base.DEVICE)
        val_rows = evaluate_split(sub, "val", model, eval_root, data_root, selected, args.output)
        write_csv(args.output / sub / "validation_per_page.csv", val_rows)
        choice = {}
        val_scores = {}
        for mode in MODES:
            options = [0] if mode == "baseline" else KS
            entries = []
            for k in options:
                subset_rows = [r for r in val_rows if r["mode"] == mode and r["K"] == k]
                entries.append((k, summarize(subset_rows)))
            k, scores = max(entries, key=lambda item: (item[1]["FM"], item[1]["Pixel_IU"], -item[0]))
            choice[mode] = k
            val_scores[mode] = {str(kk): ss for kk, ss in entries}
        print(f"[{sub}] validation choice {choice}", flush=True)
        test_rows = evaluate_split(sub, "test", model, eval_root, data_root, selected,
                                   args.output, save_examples=True, choice=choice)
        write_csv(args.output / sub / "test_per_page_selected.csv", test_rows)
        test_scores = {mode: summarize([r for r in test_rows if r["mode"] == mode])
                       for mode in MODES}
        prior_scores = prior["test"]
        for metric in METRICS:
            if abs(test_scores["baseline"][metric] - prior_scores[metric]) > 1e-9:
                raise AssertionError(f"Baseline mismatch {sub} {metric}: "
                                     f"{test_scores['baseline'][metric]} vs {prior_scores[metric]}")
        metadata["subsets"][sub] = {
            "prior_summary": str(eval_root / "summary.json"), "crop_checkpoint": str(ckpt),
            "crop_checkpoint_sha256": sha256(ckpt), "fixed_parameters": selected,
            "val_choice": choice, "validation": val_scores, "test": test_scores,
            "n_validation": len(available_stems(eval_root, "val")),
            "n_test": len(available_stems(eval_root, "test")),
        }
        (args.output / "summary.json").write_text(json.dumps(metadata, indent=2))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{sub}] test FM " + ", ".join(f"{m}: {test_scores[m]['FM']:.4f}" for m in MODES),
              flush=True)


if __name__ == "__main__":
    main()
