#!/usr/bin/env python3
"""Refine Mask R-CNN line proposals with the fixed two-stage crop segmenter."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "50_modelling/instance_segmentation/rtdetr_bbox_unet"))

from dataset import mask_to_polygon, polygon_to_baseline  # noqa: E402
from evaluate import load_segm_model, segment_crop  # noqa: E402
from maskrcnn_diva import (  # noqa: E402
    PAGE_NS,
    XSI_NS,
    DivaCocoLines,
    build_model,
    collate,
    region_polygon,
)


def make_page(meta: dict, region_points: np.ndarray):
    ET.register_namespace("", PAGE_NS)
    ET.register_namespace("xsi", XSI_NS)
    root = ET.Element(f"{{{PAGE_NS}}}PcGts", {
        f"{{{XSI_NS}}}schemaLocation": f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"
    })
    metadata = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator").text = "Mask R-CNN + crop refinement"
    now = datetime.now(timezone.utc).isoformat()
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Created").text = now
    ET.SubElement(metadata, f"{{{PAGE_NS}}}LastChange").text = now
    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page", {
        "imageFilename": Path(meta["path"]).name,
        "imageWidth": str(meta["orig_w"]),
        "imageHeight": str(meta["orig_h"]),
    })
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", {"id": "region_textline"})
    ET.SubElement(region, f"{{{PAGE_NS}}}Coords", {
        "points": " ".join(f"{x},{y}" for x, y in region_points)
    })
    return root, region


SUBSET = "CB55"


def evaluate_java(pred_dir: Path, split: str):
    subset = REPO / "00_data/DIVA-HisDB" / SUBSET
    split_name = {"val": "validation", "test": "public-test"}[split]
    gt_pixel = subset / f"pixel-level-gt-{SUBSET}/pixel-level-gt/{split_name}"
    gt_page = subset / f"PAGE-gt-{SUBSET}-TASK-2/TASK-2/{split_name}"
    images = subset / f"img-{SUBSET}/img/{split_name}"
    results = pred_dir / "results.csv"
    if results.exists():
        results.unlink()
    jar = Path.home() / "Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
    for xml_path in sorted(pred_dir.glob("*.xml")):
        stem = xml_path.stem
        run = subprocess.run([
            "java", "-cp", f"/usr/share/openjfx/lib/*:{jar}",
            "ch.unifr.LineSegmentationEvaluatorTool",
            "-igt", str(gt_pixel / f"{stem}.png"),
            "-xgt", str(gt_page / f"{stem}.xml"),
            "-xp", str(xml_path), "-overlap", str(images / f"{stem}.jpg"), "-csv",
        ], cwd=pred_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr[-500:])
    rows = list(csv.DictReader(results.open(newline="")))
    rows = list({row["filename"]: row for row in rows}.values())
    keys = ["PixelIU", "LinesIU", "LinesRecall", "LinesPrecision", "LinesFMeasure",
            "PixelPrecision", "PixelRecall", "MatchedPixelIU"]
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def main():
    global SUBSET
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("CB55", "CS18", "CS863"), default="CB55")
    parser.add_argument("--model-run", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--box-pad", type=int, default=15)
    parser.add_argument("--crop-threshold", type=float, default=0.5)
    parser.add_argument("--crop-arm", choices=("bce", "tversky", "supervoxel"),
                        default="bce")
    parser.add_argument("--select", action="store_true",
                        help="ignore --score-threshold/--crop-threshold: pick both on the "
                             "validation split (grid SCORE_GRID x CROP_GRID by FM, then "
                             "Pixel IU), then evaluate the requested split once")
    args = parser.parse_args()
    SUBSET = args.subset
    if args.select:
        run_selected(args)
        return
    run_one(args, args.split, args.score_threshold, args.crop_threshold)


SCORE_GRID = (0.3, 0.5, 0.7, 0.9)
CROP_GRID = (0.5, 0.7, 0.9)


def run_selected(args):
    rows = []
    for score in SCORE_GRID:
        for crop_thr in CROP_GRID:
            summary = run_one(args, "val", score, crop_thr, quiet=True)
            rows.append({"score_threshold": score, "crop_threshold": crop_thr,
                         **{k: summary[k] for k in ("LinesFMeasure", "PixelIU", "LinesIU")}})
            print(f"[val] score={score} crop={crop_thr} FM={summary['LinesFMeasure']:.4f} "
                  f"PIU={summary['PixelIU']:.4f}", flush=True)
    best = max(rows, key=lambda r: (r["LinesFMeasure"], r["PixelIU"]))
    print(f"[val] selected {best}", flush=True)
    summary = run_one(args, args.split, best["score_threshold"], best["crop_threshold"])
    out_dir = Path(summary["out_dir"])
    with (out_dir / "validation_grid.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["selection"] = "score and crop thresholds selected on validation by FM, then Pixel IU"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    selected_dir = out_dir.parent / f"{args.split}_crop_refine_{args.crop_arm}_selected"
    if selected_dir.is_symlink() or selected_dir.exists():
        selected_dir.unlink() if selected_dir.is_symlink() else None
    if not selected_dir.exists():
        selected_dir.symlink_to(out_dir.name)


def run_one(args, split, score_threshold, crop_threshold, quiet=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = REPO / "80_models/instance_segmentation/mask_rcnn/diva-hisdb" / SUBSET / args.model_run
    checkpoint = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = checkpoint["args"]
    model = build_model(
        cfg["arm"], cfg["image_size"], cfg.get("mask_roi_size", 14),
        cfg.get("mask_roi_width", 0), cfg.get("roi_batch_size", 512),
        cfg.get("checkpoint_mask_head", False),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    coco_dir = REPO / f"00_data/DIVA-HisDB/coco_task2_{SUBSET}"
    image_root = REPO / f"00_data/DIVA-HisDB/yolo_dataset_{SUBSET}/images"
    dataset = DivaCocoLines(coco_dir, image_root, split, cfg["image_size"], False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                        pin_memory=True, collate_fn=collate)
    seg_weights = (REPO / "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/"
                   f"crop_seg_loss_ablation_components_1024x256/{SUBSET}/{args.crop_arm}/best.pth")
    segmenter = load_segm_model(str(seg_weights), "resnet34", "unet", str(device))

    split_name = {"val": "validation", "test": "public-test"}[split]
    gt_xml_dir = (REPO / f"00_data/DIVA-HisDB/{SUBSET}/"
                  f"PAGE-gt-{SUBSET}-TASK-2/TASK-2/{split_name}")
    suffix = (f"crop_refine_{args.crop_arm}_s{score_threshold:g}_"
              f"p{args.box_pad}_m{crop_threshold:g}")
    out_dir = (REPO / "99_evaluation/instance_segmentation/mask_rcnn/diva-hisdb" / SUBSET /
               args.model_run / f"{split}_{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = []
    with torch.no_grad():
        for images, _, metadata in loader:
            with autocast(device_type=device.type, dtype=torch.float16,
                          enabled=device.type == "cuda"):
                output = model([images[0].to(device)])[0]
            meta = metadata[0]
            stem = Path(meta["path"]).stem
            scale = meta["new_w"] / meta["orig_w"]
            keep = output["scores"] >= score_threshold
            boxes = output["boxes"][keep].detach().float().cpu().numpy() / scale
            image = cv2.imread(meta["path"])
            height, width = image.shape[:2]
            region_points = region_polygon(gt_xml_dir / f"{stem}.xml")
            boxes = np.asarray([
                box for box in boxes
                if cv2.pointPolygonTest(
                    region_points,
                    (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)), False
                ) >= 0
            ], dtype=np.float32).reshape(-1, 4)
            root, region = make_page(meta, region_points)
            written = 0
            for box in boxes:
                x1 = max(0, int(np.floor(box[0])) - args.box_pad)
                y1 = max(0, int(np.floor(box[1])) - args.box_pad)
                x2 = min(width, int(np.ceil(box[2])) + args.box_pad)
                y2 = min(height, int(np.ceil(box[3])) + args.box_pad)
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                mask = segment_crop(crop, segmenter, 1024, 256, str(device), crop_threshold)
                foreground = float((mask > 0).mean())
                if foreground < 0.005 or foreground > 0.95:
                    continue
                polygon = mask_to_polygon(mask, x1, y1)
                if polygon is None or len(polygon) < 3:
                    continue
                baseline = polygon_to_baseline(polygon)
                if len(baseline) < 2:
                    continue
                line = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", {"id": f"line_{written}"})
                ET.SubElement(line, f"{{{PAGE_NS}}}Coords", {
                    "points": " ".join(f"{x},{y}" for x, y in polygon)
                })
                ET.SubElement(line, f"{{{PAGE_NS}}}Baseline", {
                    "points": " ".join(f"{x},{y}" for x, y in baseline)
                })
                written += 1
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(out_dir / f"{stem}.xml", encoding="utf-8", xml_declaration=True)
            counts.append(written)
            if not quiet:
                print(f"{stem}: {len(boxes)} boxes -> {written} polygons", flush=True)

    metrics = evaluate_java(out_dir, split)
    summary = {
        "subset": SUBSET, "model_run": args.model_run, "split": split,
        "score_threshold": score_threshold, "box_pad": args.box_pad,
        "crop_threshold": crop_threshold,
        "crop_arm": args.crop_arm,
        "crop_segmenter": str(seg_weights), "mean_predictions": float(np.mean(counts)),
        "out_dir": str(out_dir),
        **metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if not quiet:
        print(json.dumps(summary, indent=2))
    del model, segmenter
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    main()
