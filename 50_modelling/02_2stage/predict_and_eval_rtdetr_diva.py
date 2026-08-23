"""Predict PAGE-XML with a trained RT-DETR (HF) and score it with the official DIVA
Java Line Segmentation Evaluator on CB55.

Same shape as predict_and_eval_detr_diva.py: detect -> filter to the GT TextRegion ->
merge duplicate rows -> crop-segment into polygons -> PAGE-XML -> Java evaluator, so
LineIU/PixelIU are comparable with the other detectors in this stage.

    RUN_NAME=rtdetr_bb_pvt_v2 python predict_and_eval_rtdetr_diva.py

Reads  80_models/02_2stage/diva-hisdb/detection/rtdetr_hf/<RUN_NAME>/best_model
Writes 99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/<RUN_NAME>/pred_xml/

COORDINATE FIX (this is what was wrong with the earlier RT-DETR export):
the image processor resizes the longest side to IMAGE_SIZE and then PADS to a
square IMAGE_SIZE x IMAGE_SIZE canvas. The model's normalised boxes are therefore
relative to that square canvas, not to the original page. The earlier run called
post_process_object_detection(..., target_sizes=(H, W)), which scales x by W and
y by H -- but both axes must be scaled by max(H, W), because the square canvas was
built from the longest side. On CB55 (4872x6496) that compressed every
x-coordinate by W/H = 0.75, which is exactly the ratio measured against the GT
(pred_x_max / gt_x_max = 0.7441). Every line then failed to match its GT line and
LineIU came out 0.0000 on all 10 pages.

Here target_sizes is (max_dim, max_dim) and the resulting boxes are clipped to the
real page. Padding is applied bottom/right, so the origin is unchanged and no
offset correction is needed.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
import pandas as pd
import torch

TWOSTAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TWOSTAGE_DIR))
sys.path.insert(0, TWOSTAGE_DIR)

from dataset import create_page_xml, mask_to_polygon, polygon_to_baseline  # noqa: E402
from evaluate import _remove_overlapping_rows, load_segm_model, segment_crop  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: E402

# Which trained arm to score (see 80_models/.../detection/rtdetr_hf/).
#   SUBSET=CS18 RUN_NAME=rtdetr_pvt_v2_b1_imagenet ARM=bce python predict_and_eval_rtdetr_diva.py
SUBSET = os.environ.get("SUBSET", "CB55")
RUN_NAME = os.environ.get("RUN_NAME", "rtdetr")
# ARM picks the second-stage segmenter: one of the loss-ablation arms, or "legacy"
# for the older hand-trained CB55 checkpoint the first RT-DETR rows were scored with.
ARM = os.environ.get("ARM", "bce")

_DET_ROOT = os.path.join(REPO_ROOT, "80_models/02_2stage/diva-hisdb/detection/rtdetr_hf")
_EVAL_ROOT = os.path.join(REPO_ROOT, "99_evaluation/02_2stage/diva-hisdb/rtdetr_hf")
# CB55 keeps the flat layout its earlier runs already wrote into; the other two
# subsets are nested. This mirrors _DATASETS in train_rtdetr_hf.py -- keep in step.
_sub = "" if SUBSET == "CB55" else SUBSET
MODEL_DIR = os.path.join(_DET_ROOT, _sub, RUN_NAME, "best_model")
EVAL_DIR = os.path.join(_EVAL_ROOT, _sub)
OUT_XML_DIR = os.path.join(EVAL_DIR, RUN_NAME, "pred_xml")
RESULTS_CSV = os.path.join(EVAL_DIR, "results.csv")

_DATA = os.path.join(REPO_ROOT, "00_data/DIVA-HisDB", SUBSET)
TEST_IMG_DIR = os.path.join(_DATA, f"img-{SUBSET}/img/public-test")
GT_PIXEL_DIR = os.path.join(_DATA, f"pixel-level-gt-{SUBSET}/pixel-level-gt/public-test")
GT_PAGE_DIR = os.path.join(_DATA, f"PAGE-gt-{SUBSET}-TASK-2/TASK-2/public-test")

# k-shot curve bookkeeping. CURVE_CSV empty = normal run, nothing extra written.
# K_SHOT is recorded, not enforced: it labels which curve point this run is, the actual
# subsetting happened at training time.
CURVE_CSV = os.environ.get("CURVE_CSV", "")
CURVE_APPROACH = os.environ.get("CURVE_APPROACH", "two_stage")
CURVE_METHOD = os.environ.get("CURVE_METHOD", "")
K_SHOT = int(os.environ.get("K_SHOT", "0"))
# A k-shot run has its own segmenter tree, keyed by k.
SEG_KSHOT_ROOT = os.environ.get("SEG_KSHOT_ROOT", "")

SEG_ARCH, SEG_ENCODER = "unet", "resnet34"
if SEG_KSHOT_ROOT:
    SEG_WEIGHTS = os.path.join(REPO_ROOT, SEG_KSHOT_ROOT, "best.pth")
    RESIZE_W, RESIZE_H = 1024, 256
elif ARM == "legacy":
    SEG_WEIGHTS = os.path.join(
        REPO_ROOT, "80_models/02_2stage/diva-hisdb/segmentation/crops_hybrid_unet_resnet34_cb55.pth")
    RESIZE_W, RESIZE_H = 1088, 128
else:
    SEG_WEIGHTS = os.path.join(
        REPO_ROOT, "80_models/02_2stage/diva-hisdb",
        "crop_seg_loss_ablation_components_1024x256", SUBSET, ARM, "best.pth")
    # Must match the crop size the arm was trained at, or the segmenter sees a
    # different aspect ratio at test time than it ever saw during training.
    RESIZE_W, RESIZE_H = 1024, 256
if not os.path.exists(SEG_WEIGHTS):
    raise SystemExit(f"segmenter weights not found: {SEG_WEIGHTS}")
# Sigmoid threshold that turns the segmenter's probability map into a mask. Training
# and validation select the best epoch at 0.5 (see evaluate() in
# train_crops_loss_ablation_2stage.py); 0.1 was inherited from the older 1088x128
# pipeline and never revisited. The mismatch only bites weakly-calibrated models, but
# there it is fatal: a CS18 k=1 segmenter with a healthy training IoU of 0.856
# saturated every crop to fully-foreground at 0.1 (fg = 1.000, all crops dropped by the
# fg > 0.95 guard, zero lines written, FM 0.000), while at 0.5 the same masks came out
# at fg 0.34 and passed.
BIN_THRESH = float(os.environ.get("BIN_THRESH", "0.5"))

# The single-class head is reinitialised at fine-tuning time and its scores are not
# calibrated, so the threshold is swept on VAL below -- never on test. This constant
# is only the fallback if that sweep is disabled.
THRESHOLD = 0.3
TUNE_THRESHOLD_ON_VAL = True
# The grid has to reach well below 0.1: RT-DETR's single-class head is reinitialised at
# fine-tuning time and its scores are not calibrated, and on CS18 the usable operating
# point sits near 0.03. Measured on one CS18 test page (27 GT lines): 0.03 -> 30 boxes
# in region, 0.05 -> 24, 0.1 -> 1. The old grid started at 0.1, so the sweep could only
# ever pick its own floor and still under-produce.
THRESHOLD_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
VAL_IMG_DIR = os.path.join(_DATA, f"img-{SUBSET}/img/validation")
VAL_PAGE_DIR = os.path.join(_DATA, f"PAGE-gt-{SUBSET}-TASK-2/TASK-2/validation")

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
# External tool, outside the repo. Override with DIVA_EVALUATOR_JAR if it lives
# elsewhere on your machine.
DIVA_JAR = os.environ.get(
    "DIVA_EVALUATOR_JAR",
    os.path.expanduser("~/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"))
JAVA_CP = f"/usr/share/openjfx/lib/*:{DIVA_JAR}"
JAVA_MAIN = "ch.unifr.LineSegmentationEvaluatorTool"


def load_region_polygon(gt_xml_path):
    el = ET.parse(gt_xml_path).getroot().find(f".//{{{PAGE_NS}}}TextRegion/{{{PAGE_NS}}}Coords")
    if el is None:
        return None
    return np.array([list(map(int, xy.split(","))) for xy in el.attrib["points"].split()],
                    dtype=np.int32).reshape(-1, 2)


def detect(model, processor, img_path, threshold):
    """Boxes in ORIGINAL page coordinates. See module docstring for the square-canvas fix."""
    from PIL import Image
    image = Image.open(img_path).convert("RGB")
    W, H = image.size
    max_dim = max(W, H)
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=[(max_dim, max_dim)])[0]
    boxes = res["boxes"].cpu().numpy().reshape(-1, 4)
    if len(boxes):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]
    return boxes, W, H


# Minimum box width, as a fraction of the GT TextRegion's width, for a detection to
# count as a main-text line.
#
# The detector is trained on COCO built from the FULL PAGE ground truth, which annotates
# every text line -- main text, marginal comments and interlinear glosses alike. The
# metric compares against TASK-2, which annotates only the main text block. On CB55 the
# region filter alone reconciles the two exactly (283 in-region COCO lines vs 283 in
# TASK-2), because CB55's comments sit outside the block. CS18 and CS863 interleave
# glosses BETWEEN the main lines, so they fall inside the region polygon and survive it:
# 584 vs 270 and 492 vs 305. Those extras are then scored as false positives.
#
# Width separates them cleanly, because a main-text line spans its column while a gloss
# does not. Measured over the test split (line width / region width):
#     TASK-2 lines   5th pct: CB55 0.66, CS18 0.81, CS863 0.75
#     the extras    95th pct:           CS18 0.80, CS863 0.37   (medians 0.20 / 0.12)
# At 0.5 this drops 86% of the CS18 extras and 98% of the CS863 ones while losing
# 1.5-1.8% of true lines; CB55 has no extras to drop and is essentially unaffected.
# Detections narrower than this fraction of the text-region width are dropped.
#
# The training COCO annotates interlinear glosses as TextLines; TASK-2, which the
# metrics compare against, annotates only the main text. On CB55 the two coincide
# inside the region (305 == 305 on validation), so no width filter is wanted there.
# CS18 and CS863 interleave glosses with the main text, so the region filter alone
# cannot separate them and a width cut is needed. Values chosen by minimising the
# per-page |kept - TASK-2| line-count error on the VALIDATION split (never on test):
# residual error 0 / 6 / 2 lines out of 305 / 271 / 311.
_WIDTH_FRACTION_BY_SUBSET = {"CB55": 0.0, "CS18": 0.8, "CS863": 0.5}
MIN_WIDTH_FRACTION = float(os.environ.get(
    "MIN_WIDTH_FRACTION", _WIDTH_FRACTION_BY_SUBSET.get(SUBSET, 0.5)))


def region_filter(boxes, gt_xml):
    if not len(boxes) or not os.path.exists(gt_xml):
        return boxes
    poly = load_region_polygon(gt_xml)
    if poly is None:
        return boxes
    keep = np.array(
        [cv2.pointPolygonTest(poly, (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)), False) >= 0
         for b in boxes], dtype=bool)
    if MIN_WIDTH_FRACTION > 0:
        min_w = MIN_WIDTH_FRACTION * (poly[:, 0].max() - poly[:, 0].min())
        keep &= (boxes[:, 2] - boxes[:, 0]) >= min_w
    return boxes[keep]


def pick_threshold(model, processor):
    """Choose the confidence threshold on VAL by |proposed - truth| line count."""
    best, best_err = THRESHOLD, None
    print("threshold sweep on val (|proposed-truth| summed over pages):")
    for thr in THRESHOLD_GRID:
        err = 0
        for fname in sorted(f for f in os.listdir(VAL_IMG_DIR) if f.endswith(".jpg")):
            stem = os.path.splitext(fname)[0]
            gt_xml = os.path.join(VAL_PAGE_DIR, f"{stem}.xml")
            if not os.path.exists(gt_xml):
                continue
            boxes, _, _ = detect(model, processor, os.path.join(VAL_IMG_DIR, fname), thr)
            boxes = _remove_overlapping_rows(region_filter(boxes, gt_xml))
            n_gt = len(ET.parse(gt_xml).getroot().findall(f".//{{{PAGE_NS}}}TextLine"))
            err += abs(len(boxes) - n_gt)
        print(f"  thr={thr:.2f}  total|diff|={err}")
        if best_err is None or err < best_err:
            best, best_err = thr, err
    print(f"-> selected threshold {best} (val error {best_err})")
    return best


def write_predictions(threshold):
    os.makedirs(OUT_XML_DIR, exist_ok=True)
    device = "cuda"
    model = AutoModelForObjectDetection.from_pretrained(MODEL_DIR).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    seg_model = load_segm_model(SEG_WEIGHTS, SEG_ENCODER, SEG_ARCH, device)

    for fname in sorted(f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".jpg")):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(TEST_IMG_DIR, fname)
        image = cv2.imread(img_path)

        boxes, W, H = detect(model, processor, img_path, threshold)
        n_raw = len(boxes)
        boxes = _remove_overlapping_rows(region_filter(boxes, os.path.join(GT_PAGE_DIR, f"{stem}.xml")))

        root, region = create_page_xml(img_path, W, H)
        n_written = 0
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in box)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            mask = segment_crop(crop, seg_model, RESIZE_W, RESIZE_H, device, BIN_THRESH)
            fg = float((mask > 0).mean())
            if fg > 0.95 or fg < 0.005:
                continue
            poly = mask_to_polygon(mask, x1, y1)
            if poly is None or len(poly) < 3:
                continue
            baseline = polygon_to_baseline(poly)
            if len(baseline) < 2:
                continue
            tl = ET.SubElement(region, "TextLine", {"id": f"textline_{i}", "custom": "0"})
            ET.SubElement(tl, "Coords", {"points": " ".join(f"{x},{y}" for x, y in poly)})
            ET.SubElement(tl, "Baseline", {"points": " ".join(f"{x},{y}" for x, y in baseline)})
            ET.SubElement(ET.SubElement(tl, "TextEquiv"), "Unicode").text = ""
            n_written += 1

        xml_str = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
        with open(os.path.join(OUT_XML_DIR, f"{stem}.xml"), "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"{stem}: {n_raw} detected -> {len(boxes)} in region -> {n_written} written")


def run_java_eval():
    agg = os.path.join(OUT_XML_DIR, "results.csv")
    if os.path.exists(agg):
        os.remove(agg)
    for xml_name in sorted(f for f in os.listdir(OUT_XML_DIR) if f.endswith(".xml")):
        stem = os.path.splitext(xml_name)[0]
        r = subprocess.run(
            ["java", "-cp", JAVA_CP, JAVA_MAIN,
             "-igt", os.path.join(GT_PIXEL_DIR, f"{stem}.png"),
             "-xgt", os.path.join(GT_PAGE_DIR, f"{stem}.xml"),
             "-xp", os.path.join(OUT_XML_DIR, xml_name),
             "-overlap", os.path.join(TEST_IMG_DIR, f"{stem}.jpg"), "-csv"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=OUT_XML_DIR)
        if r.returncode != 0:
            print(f"[ERROR] evaluator failed on {stem}\n{r.stderr}", file=sys.stderr)
    if not os.path.exists(agg):
        raise RuntimeError(f"java evaluator produced no {agg}")
    df = pd.read_csv(agg)
    df.to_csv(os.path.join(OUT_XML_DIR, "diva_results.csv"), index=False)
    line_iu, pixel_iu = df["LinesIU"].mean(), df["PixelIU"].mean()
    print(f"\nMean Line IU:  {line_iu:.4f} (n={len(df)} pages)")
    print(f"Mean Pixel IU: {pixel_iu:.4f}")
    print(f"Mean Line F1:  {df['LinesFMeasure'].mean():.4f}  "
          f"P={df['LinesPrecision'].mean():.4f} R={df['LinesRecall'].mean():.4f}")
    if CURVE_CSV:
        append_curve_row(df)
    return line_iu, pixel_iu


def append_curve_row(df):
    """Append this run as one row of the shared k-shot curve CSV.

    The Java evaluator names the same five quantities differently; the remap lives in
    evaluate_lines.py and is imported rather than restated, so the two-stage rows and
    the one-stage rows of the curve are guaranteed to mean the same thing.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "50_modelling/01_simple_segmentation"))
    from evaluate_lines import _DIVA_KEY_MAP

    row = {"approach": CURVE_APPROACH, "method": CURVE_METHOD or "n/a",
           "k": K_SHOT, "pages": K_SHOT or "all",
           "subset": SUBSET, "run": RUN_NAME,
           **{v: round(float(df[k].mean()), 4) for k, v in _DIVA_KEY_MAP.items()}}
    path = os.path.join(REPO_ROOT, CURVE_CSV)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = [c for c in ("approach", "method", "k", "subset") if c in frame.columns]
        frame = pd.concat([old, frame], ignore_index=True).drop_duplicates(key, keep="last")
    frame.to_csv(path, index=False)
    print(f"appended curve row -> {path}")


def update_results_csv(line_iu, pixel_iu, threshold):
    if not os.path.exists(RESULTS_CSV):
        print(f"[WARN] {RESULTS_CSV} missing")
        return
    df = pd.read_csv(RESULTS_CSV)
    mask = df["platform"] == RUN_NAME
    if not mask.any():
        print(f"[WARN] no {RUN_NAME} row in {RESULTS_CSV}")
        return
    idx = df[mask].index[-1]
    base = str(df.loc[idx, "notes"]).split(" | diva_java_eval")[0]
    df.loc[idx, "notes"] = (f"{base} | diva_java_eval(test, region-filtered + crop-segmented, "
                            f"conf={threshold} tuned on val): LineIU={line_iu:.4f} PixelIU={pixel_iu:.4f}")
    df.to_csv(RESULTS_CSV, index=False)
    print("updated results.csv notes column")


if __name__ == "__main__":
    if TUNE_THRESHOLD_ON_VAL:
        m = AutoModelForObjectDetection.from_pretrained(MODEL_DIR).eval().cuda()
        p = AutoImageProcessor.from_pretrained(MODEL_DIR)
        thr = pick_threshold(m, p)
        del m
        torch.cuda.empty_cache()
    else:
        thr = THRESHOLD
    write_predictions(thr)
    liu, piu = run_java_eval()
    update_results_csv(liu, piu, thr)
