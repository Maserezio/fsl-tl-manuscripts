#!/usr/bin/env python3
"""Evaluate the three polygon crop-segmenter losses with CB55 RF-/RT-DETR.

The detector outputs are computed once per detector and then held fixed across
the BCE, Tversky and SuperVoxel stage-2 arms.  Each stage-2 mask is converted to
a PAGE TextLine polygon/baseline and scored by the official DIVA Java evaluator.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dataset import create_page_xml, mask_to_polygon, polygon_to_baseline  # noqa: E402
from evaluate import _remove_overlapping_rows, load_segm_model, segment_crop  # noqa: E402


PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
JAVA_CP = (
    "/usr/share/openjfx/lib/*:"
    "/home/artur/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/"
    "LineSegmentationEvaluator.jar"
)
JAVA_MAIN = "ch.unifr.LineSegmentationEvaluatorTool"

TEST_IMG_DIR = REPO / "00_data/DIVA-HisDB/CB55/img-CB55/img/public-test"
GT_PIXEL_DIR = REPO / "00_data/DIVA-HisDB/CB55/pixel-level-gt-CB55/pixel-level-gt/public-test"
GT_PAGE_DIR = REPO / "00_data/DIVA-HisDB/CB55/PAGE-gt-CB55-TASK-2/TASK-2/public-test"

RF_WEIGHTS = REPO / "20_detection/rfdetr/output/checkpoint_best_regular.pth"
RT_MODEL_DIR = REPO / "20_detection/rtdetr/output/best_model"
DEFAULT_SEG_ROOT = (
    REPO
    / "80_models/02_2stage/diva-hisdb/crop_seg_loss_ablation_polygon/CB55"
)
DEFAULT_OUT_ROOT = (
    REPO
    / "99_evaluation/02_2stage/diva-hisdb/cb55_polygon_loss_ablation_detr"
)

ARMS = ("bce", "tversky", "supervoxel")
DETECTOR_THRESHOLDS = {"rf_detr": 0.5, "rt_detr": 0.1}
SEG_THRESHOLD = 0.5
RESIZE_W, RESIZE_H = 1088, 128
SEG_ROOT = DEFAULT_SEG_ROOT
OUT_ROOT = DEFAULT_OUT_ROOT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_region_polygon(gt_xml_path: Path) -> np.ndarray | None:
    element = ET.parse(gt_xml_path).getroot().find(
        f".//{{{PAGE_NS}}}TextRegion/{{{PAGE_NS}}}Coords"
    )
    if element is None:
        return None
    return np.asarray(
        [list(map(int, xy.split(","))) for xy in element.attrib["points"].split()],
        dtype=np.int32,
    ).reshape(-1, 2)


def region_filter(boxes: np.ndarray, gt_xml_path: Path) -> np.ndarray:
    if not len(boxes) or not gt_xml_path.exists():
        return boxes
    polygon = load_region_polygon(gt_xml_path)
    if polygon is None:
        return boxes
    keep = [
        cv2.pointPolygonTest(
            polygon,
            (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)),
            False,
        )
        >= 0
        for box in boxes
    ]
    return boxes[np.asarray(keep, dtype=bool)]


def finalize_boxes(boxes: np.ndarray, stem: str) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    boxes = region_filter(boxes, GT_PAGE_DIR / f"{stem}.xml")
    return _remove_overlapping_rows(boxes)


def detect_rf() -> dict[str, np.ndarray]:
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=str(RF_WEIGHTS), num_classes=1)
    threshold = DETECTOR_THRESHOLDS["rf_detr"]
    result = {}
    for image_path in sorted(TEST_IMG_DIR.glob("*.jpg")):
        detections = model.predict(str(image_path), threshold=threshold)
        result[image_path.stem] = finalize_boxes(detections.xyxy, image_path.stem)
        print(f"[rf_detr] {image_path.stem}: {len(result[image_path.stem])} boxes", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def detect_rt() -> dict[str, np.ndarray]:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model = AutoModelForObjectDetection.from_pretrained(str(RT_MODEL_DIR)).eval().to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(str(RT_MODEL_DIR))
    threshold = DETECTOR_THRESHOLDS["rt_detr"]
    result = {}
    for image_path in sorted(TEST_IMG_DIR.glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        max_dim = max(width, height)
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        processed = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(max_dim, max_dim)]
        )[0]
        boxes = processed["boxes"].cpu().numpy().reshape(-1, 4)
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
        result[image_path.stem] = finalize_boxes(boxes, image_path.stem)
        print(f"[rt_detr] {image_path.stem}: {len(result[image_path.stem])} boxes", flush=True)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def clean_prediction_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.xml", "*.csv", "*-overlap.png", "*-visualization.png"):
        for old_path in path.glob(pattern):
            old_path.unlink()


def write_page_xmls(
    detector: str,
    arm: str,
    boxes_by_stem: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, int]:
    checkpoint = SEG_ROOT / arm / "best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model = load_segm_model(str(checkpoint), "resnet34", "unet", DEVICE)
    clean_prediction_dir(out_dir)
    written_by_stem = {}

    for image_path in sorted(TEST_IMG_DIR.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        root, region = create_page_xml(str(image_path), width, height)
        written = 0
        for index, box in enumerate(boxes_by_stem[image_path.stem]):
            x1, y1, x2, y2 = (int(value) for value in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            mask = segment_crop(
                crop, model, RESIZE_W, RESIZE_H, DEVICE, SEG_THRESHOLD
            )
            foreground_ratio = float((mask > 0).mean())
            if foreground_ratio < 0.005 or foreground_ratio > 0.95:
                continue
            polygon = mask_to_polygon(mask, x1, y1)
            if polygon is None or len(polygon) < 3:
                continue
            baseline = polygon_to_baseline(polygon)
            if len(baseline) < 2:
                continue

            line = ET.SubElement(
                region, "TextLine", {"id": f"textline_{index}", "custom": "0"}
            )
            ET.SubElement(
                line,
                "Coords",
                {"points": " ".join(f"{x},{y}" for x, y in polygon)},
            )
            ET.SubElement(
                line,
                "Baseline",
                {"points": " ".join(f"{x},{y}" for x, y in baseline)},
            )
            ET.SubElement(ET.SubElement(line, "TextEquiv"), "Unicode").text = ""
            written += 1

        xml_text = minidom.parseString(
            ET.tostring(root, encoding="utf-8")
        ).toprettyxml(indent="  ")
        (out_dir / f"{image_path.stem}.xml").write_text(xml_text, encoding="utf-8")
        written_by_stem[image_path.stem] = written
        print(
            f"[{detector}/{arm}] {image_path.stem}: "
            f"{len(boxes_by_stem[image_path.stem])} boxes -> {written} polygons",
            flush=True,
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return written_by_stem


def run_diva(out_dir: Path) -> pd.DataFrame:
    results_path = out_dir / "results.csv"
    if results_path.exists():
        results_path.unlink()
    for xml_path in sorted(out_dir.glob("*.xml")):
        stem = xml_path.stem
        command = [
            "java",
            "-cp",
            JAVA_CP,
            JAVA_MAIN,
            "-igt",
            str(GT_PIXEL_DIR / f"{stem}.png"),
            "-xgt",
            str(GT_PAGE_DIR / f"{stem}.xml"),
            "-xp",
            str(xml_path),
            "-overlap",
            str(TEST_IMG_DIR / f"{stem}.jpg"),
            "-csv",
        ]
        completed = subprocess.run(
            command,
            cwd=out_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"DIVA evaluator failed for {stem}:\n{completed.stderr[-4000:]}"
            )
    if not results_path.exists():
        raise RuntimeError(f"DIVA evaluator did not create {results_path}")
    frame = pd.read_csv(results_path)
    frame.to_csv(out_dir / "diva_results.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=("rf_detr", "rt_detr"),
        default=("rf_detr", "rt_detr"),
    )
    parser.add_argument("--width", type=int, default=1088)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--segmenter-root", type=Path, default=DEFAULT_SEG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    global RESIZE_W, RESIZE_H, SEG_ROOT, OUT_ROOT
    RESIZE_W, RESIZE_H = args.width, args.height
    SEG_ROOT = args.segmenter_root.resolve()
    OUT_ROOT = args.output_root.resolve()

    required = [TEST_IMG_DIR, GT_PIXEL_DIR, GT_PAGE_DIR, SEG_ROOT]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if "rf_detr" in args.detectors and not RF_WEIGHTS.exists():
        raise FileNotFoundError(RF_WEIGHTS)
    if "rt_detr" in args.detectors and not RT_MODEL_DIR.exists():
        raise FileNotFoundError(RT_MODEL_DIR)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    detector_boxes = {}
    if "rf_detr" in args.detectors:
        detector_boxes["rf_detr"] = detect_rf()
    if "rt_detr" in args.detectors:
        detector_boxes["rt_detr"] = detect_rt()

    summary = []
    metric_columns = (
        "PixelIU",
        "LinesIU",
        "LinesPrecision",
        "LinesRecall",
        "LinesFMeasure",
    )
    for detector, boxes_by_stem in detector_boxes.items():
        for arm in ARMS:
            out_dir = OUT_ROOT / f"{detector}_{arm}" / "pred_xml"
            written = write_page_xmls(detector, arm, boxes_by_stem, out_dir)
            frame = run_diva(out_dir)
            row = {
                "detector": detector,
                "loss": arm,
                "detector_threshold": DETECTOR_THRESHOLDS[detector],
                "segmenter_threshold": SEG_THRESHOLD,
                "n_pages": len(frame),
                "n_polygons": sum(written.values()),
            }
            row.update({column: float(frame[column].mean()) for column in metric_columns})
            summary.append(row)
            pd.DataFrame(summary).to_csv(OUT_ROOT / "diva_metrics.csv", index=False)
            print(
                f"RESULT {detector}/{arm}: LinesFMeasure={row['LinesFMeasure']:.5f} "
                f"LinesIU={row['LinesIU']:.5f} PixelIU={row['PixelIU']:.5f}",
                flush=True,
            )

    metadata = {
        "rf_detr_weights": str(RF_WEIGHTS),
        "rt_detr_model_dir": str(RT_MODEL_DIR),
        "segmenter_root": str(SEG_ROOT),
        "detector_thresholds": DETECTOR_THRESHOLDS,
        "segmenter_threshold": SEG_THRESHOLD,
        "resize": [RESIZE_W, RESIZE_H],
    }
    (OUT_ROOT / "evaluation_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\n" + pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"\nSaved summary to {OUT_ROOT / 'diva_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
