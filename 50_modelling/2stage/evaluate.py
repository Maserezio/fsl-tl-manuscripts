"""
evaluate.py — run the full YOLO+segm prediction pipeline and evaluate results.

Two sub-commands:

  predict   Run YOLO detection + crop segmentation on a test image directory,
            writing PAGE-XML output files.

  score     Score existing PAGE-XML predictions using HSCP (polygon) metrics
            and/or the DIVA Java evaluator.

Usage examples
--------------
# Predict and write PAGE-XMLs
python evaluate.py predict --config configs/segm_cb55.yaml \\
    --yolo-weights ../71_misc/runs/yolo_cs863_coco/exp/weights/best.pt \\
    --segm-weights ../80_models/best.pth \\
    --test-img-dir  ../00_data/DIVA-HisDB/CS863/img-CS863/img/public-test/ \\
    --test-xml-dir  ../00_data/DIVA-HisDB/CS863/PAGE-gt-CS863-TASK-2/TASK-2/public-test/ \\
    --out-xml-dir   ../99_evaluation/pred_cs863/

# Score predictions with HSCP
python evaluate.py score --mode hscp \\
    --pred-xml-dir  ../99_evaluation/pred_cs863/ \\
    --gt-xml-dir    ../00_data/DIVA-HisDB/CS863/PAGE-gt-CS863-TASK-2/TASK-2/public-test/ \\
    --test-img-dir  ../00_data/DIVA-HisDB/CS863/img-CS863/img/public-test/

# Score with DIVA Java evaluator
python evaluate.py score --mode diva \\
    --pred-xml-dir  ../99_evaluation/pred_cs863/ \\
    --gt-pixel-dir  ~/Thesis/fsl-tl-manuscripts/00_data/DIVA-HisDB/CS863/pixel-level-gt-CS863/pixel-level-gt/public-test/ \\
    --gt-page-dir   ~/Thesis/fsl-tl-manuscripts/00_data/DIVA-HisDB/CS863/PAGE-gt-CS863-TASK-2/TASK-2/public-test/ \\
    --img-dir       ~/Thesis/fsl-tl-manuscripts/00_data/DIVA-HisDB/CS863/img-CS863/img/public-test/
"""

import argparse
import os
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import segmentation_models_pytorch as smp

from dataset import (
    create_page_xml,
    mask_to_polygon,
    parse_xml_polygons,
    polygon_to_baseline,
    polygon_to_mask,
)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_segm_model(weights_path: str, encoder: str, arch: str, device: str):
    build_fn = getattr(smp, arch.capitalize() if arch != "segformer" else "Segformer", smp.Unet)
    model = build_fn(encoder_name=encoder, encoder_weights=None,
                     in_channels=3, classes=1).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def segment_crop(crop_bgr: np.ndarray, model, resize_w: int, resize_h: int,
                 device: str, bin_thresh: float) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = crop_rgb.shape[:2]
    resized = cv2.resize(crop_rgb, (resize_w, resize_h))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    probs = cv2.resize(probs, (w0, h0))
    pred_mask = (probs >= bin_thresh).astype(np.uint8) * 255

    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pred_mask
    largest = max(contours, key=cv2.contourArea)
    single_mask = np.zeros_like(pred_mask)
    cv2.drawContours(single_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return single_mask


def _remove_overlapping_rows(boxes: np.ndarray, iou_thresh: float = 0.95) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    boxes = boxes[boxes[:, 1].argsort()]
    keep = []
    for b in boxes:
        replaced = False
        for i, k in enumerate(keep):
            overlap = (min(b[3], k[3]) - max(b[1], k[1])) / (b[3] - b[1] + 1e-6)
            if overlap > iou_thresh:
                if (b[3] - b[1]) > (k[3] - k[1]):
                    keep[i] = b
                replaced = True
                break
        if not replaced:
            keep.append(b)
    return np.array(keep)


# ---------------------------------------------------------------------------
# Predict command
# ---------------------------------------------------------------------------

def cmd_predict(args):
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    print("[INFO] Loading YOLO...")
    yolo = YOLO(args.yolo_weights)

    print("[INFO] Loading segmentation model...")
    seg_model = load_segm_model(
        args.segm_weights,
        encoder=args.encoder,
        arch=args.arch,
        device=device,
    )
    print("[INFO] Models ready")

    os.makedirs(args.out_xml_dir, exist_ok=True)
    resize_w, resize_h = args.resize_w, args.resize_h

    img_files = sorted(f for f in os.listdir(args.test_img_dir)
                       if f.lower().endswith((".jpg", ".png")))
    print(f"[INFO] Found {len(img_files)} images")

    for img_name in tqdm(img_files, desc="Predicting"):
        img_path = os.path.join(args.test_img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Cannot read {img_name}")
            continue

        H, W = image.shape[:2]

        # Detection
        results = yolo(image, conf=args.conf, verbose=False)[0]
        boxes = (results.boxes.xyxy.cpu().numpy()
                 if results.boxes is not None
                 else np.empty((0, 4), dtype=np.float32))

        # Filter boxes to text region from GT XML (if available)
        gt_xml = os.path.join(args.test_xml_dir, os.path.splitext(img_name)[0] + ".xml")
        if os.path.exists(gt_xml) and len(boxes) > 0:
            region_coords_el = ET.parse(gt_xml).getroot().find(
                ".//{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
                "TextRegion/"
                "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Coords"
            )
            if region_coords_el is not None:
                region_poly = np.array(
                    [list(map(int, xy.split(",")))
                     for xy in region_coords_el.attrib["points"].split()],
                    dtype=np.int32
                ).reshape(-1, 2)
                keep = [cv2.pointPolygonTest(
                            region_poly,
                            (float((b[0] + b[2]) // 2), float((b[1] + b[3]) // 2)),
                            False) >= 0
                        for b in boxes]
                boxes = boxes[np.array(keep)]

        boxes = _remove_overlapping_rows(boxes)

        root, region = create_page_xml(img_path, W, H)
        n_written = 0

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            mask = segment_crop(crop, seg_model, resize_w, resize_h, device, args.bin_thresh)

            fg_ratio = float((mask > 0).mean())
            if fg_ratio > 0.95 or fg_ratio < 0.005:
                continue

            poly = mask_to_polygon(mask, x1, y1)
            if poly is None or len(poly) < 3:
                continue

            baseline = polygon_to_baseline(poly)
            if len(baseline) < 2:
                continue

            tl = ET.SubElement(region, "TextLine", {"id": f"textline_{i}", "custom": "0"})
            ET.SubElement(tl, "Coords",   {"points": " ".join(f"{x},{y}" for x, y in poly)})
            ET.SubElement(tl, "Baseline", {"points": " ".join(f"{x},{y}" for x, y in baseline)})
            ET.SubElement(ET.SubElement(tl, "TextEquiv"), "Unicode").text = ""
            n_written += 1

        xml_str = minidom.parseString(
            ET.tostring(root, encoding="utf-8")
        ).toprettyxml(indent="  ")
        out_path = os.path.join(args.out_xml_dir, os.path.splitext(img_name)[0] + ".xml")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        if n_written == 0:
            print(f"[WARN] {img_name}: 0 polygons written")

    print("[INFO] Done")


# ---------------------------------------------------------------------------
# HSCP scoring
# ---------------------------------------------------------------------------

def match_score(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    inter = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt,  mask_pred).sum()
    return inter / union if union > 0 else 0.0


def hscp_polygon_eval(gt_xml: str, pred_xml: str, H: int, W: int,
                      iou_thresh: float = 0.75) -> dict:
    gt_polys   = parse_xml_polygons(gt_xml)
    pred_polys = parse_xml_polygons(pred_xml)
    N1, N2 = len(gt_polys), len(pred_polys)

    if N1 == 0 or N2 == 0:
        return {"DR": 0.0, "RA": 0.0, "FM": 0.0, "M": 0, "N_gt": N1, "N_pred": N2}

    gt_masks   = [polygon_to_mask(p, H, W) for p in gt_polys]
    pred_masks = [polygon_to_mask(p, H, W) for p in pred_polys]

    used_pred = set()
    M = 0
    for gt_mask in gt_masks:
        best_j, best_score = -1, 0.0
        for j, pr_mask in enumerate(pred_masks):
            if j in used_pred:
                continue
            s = match_score(gt_mask, pr_mask)
            if s > best_score:
                best_score, best_j = s, j
        if best_score >= iou_thresh and best_j >= 0:
            M += 1
            used_pred.add(best_j)

    DR = M / N1 if N1 else 0.0
    RA = M / N2 if N2 else 0.0
    FM = (2 * DR * RA) / (DR + RA) if (DR + RA) > 0 else 0.0
    return {"DR": round(DR, 4), "RA": round(RA, 4), "FM": round(FM, 4),
            "M": M, "N_gt": N1, "N_pred": N2}


def cmd_score_hscp(args):
    img_files = sorted(f for f in os.listdir(args.test_img_dir)
                       if f.lower().endswith((".jpg", ".png")))
    results = []
    for img_name in tqdm(img_files, desc="Scoring (HSCP)"):
        stem = os.path.splitext(img_name)[0]
        pred_xml = os.path.join(args.pred_xml_dir, f"{stem}.xml")
        gt_xml   = os.path.join(args.gt_xml_dir,   f"{stem}.xml")
        if not os.path.exists(pred_xml) or not os.path.exists(gt_xml):
            print(f"[WARN] Missing XML for {stem}")
            continue
        img = cv2.imread(os.path.join(args.test_img_dir, img_name))
        H, W = img.shape[:2]
        row = hscp_polygon_eval(gt_xml, pred_xml, H, W, iou_thresh=args.iou_thresh)
        row["image"] = img_name
        results.append(row)

    df = pd.DataFrame(results)
    print(f"\nMean DR: {df['DR'].mean():.4f}  RA: {df['RA'].mean():.4f}  FM: {df['FM'].mean():.4f}")
    out_csv = os.path.join(args.pred_xml_dir, "hscp_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")


# ---------------------------------------------------------------------------
# DIVA Java evaluator
# ---------------------------------------------------------------------------

def cmd_score_diva(args):
    java_cp = os.path.expanduser(
        "/usr/share/openjfx/lib/*:"
        "/home/artur/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
    )
    java_main = "ch.unifr.LineSegmentationEvaluatorTool"
    gt_pixel_dir = os.path.expanduser(args.gt_pixel_dir)
    gt_page_dir  = os.path.expanduser(args.gt_page_dir)
    img_dir      = os.path.expanduser(args.img_dir)
    out_csv_dir  = os.path.join(args.pred_xml_dir, "eval_csv")
    os.makedirs(out_csv_dir, exist_ok=True)

    pred_xmls = sorted(f for f in os.listdir(args.pred_xml_dir) if f.endswith(".xml"))
    for xml_name in tqdm(pred_xmls, desc="DIVA evaluator"):
        stem = os.path.splitext(xml_name)[0]
        cmd = [
            "java", "-cp", java_cp, java_main,
            "-igt", os.path.join(gt_pixel_dir, f"{stem}.png"),
            "-xgt", os.path.join(gt_page_dir,  f"{stem}.xml"),
            "-xp",  os.path.join(args.pred_xml_dir, xml_name),
            "-overlap", os.path.join(img_dir, f"{stem}.jpg"),
            "-csv",
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, check=True)
            with open(os.path.join(out_csv_dir, f"{stem}.csv"), "w") as f:
                f.write(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Evaluator failed on {stem}\n{e.stderr}")

    # Aggregate
    rows = []
    for csv_file in sorted(os.listdir(out_csv_dir)):
        if csv_file.endswith(".csv"):
            try:
                rows.append(pd.read_csv(os.path.join(out_csv_dir, csv_file)))
            except Exception:
                pass
    if rows:
        df = pd.concat(rows, ignore_index=True)
        print(f"\nMean Line IU:  {df['LinesIU'].mean():.4f}")
        print(f"Mean Pixel IU: {df['PixelIU'].mean():.4f}")
        df.to_csv(os.path.join(args.pred_xml_dir, "diva_results.csv"), index=False)
        print(f"Saved to {args.pred_xml_dir}/diva_results.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    # ---- predict ----
    pr = sub.add_parser("predict", help="Run YOLO+segm pipeline and write PAGE-XMLs")
    pr.add_argument("--yolo-weights",  required=True)
    pr.add_argument("--segm-weights",  required=True)
    pr.add_argument("--test-img-dir",  required=True)
    pr.add_argument("--test-xml-dir",  required=True,
                    help="GT PAGE-XML dir (used to filter boxes to text region)")
    pr.add_argument("--out-xml-dir",   required=True)
    pr.add_argument("--encoder",       default="resnet34")
    pr.add_argument("--arch",          default="unet")
    pr.add_argument("--resize-w",      type=int, default=1080)
    pr.add_argument("--resize-h",      type=int, default=128)
    pr.add_argument("--bin-thresh",    type=float, default=0.1)
    pr.add_argument("--conf",          type=float, default=0.5)

    # ---- score ----
    sc = sub.add_parser("score", help="Score existing PAGE-XML predictions")
    sc.add_argument("--mode", choices=["hscp", "diva"], required=True)
    sc.add_argument("--pred-xml-dir", required=True)
    # hscp
    sc.add_argument("--gt-xml-dir",   help="GT PAGE-XML dir (HSCP mode)")
    sc.add_argument("--test-img-dir", help="Test images dir (HSCP mode)")
    sc.add_argument("--iou-thresh",   type=float, default=0.75)
    # diva
    sc.add_argument("--gt-pixel-dir", help="GT pixel masks dir (DIVA mode)")
    sc.add_argument("--gt-page-dir",  help="GT PAGE-XML dir (DIVA mode)")
    sc.add_argument("--img-dir",      help="Original images dir (DIVA mode)")

    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "score":
        if args.mode == "hscp":
            cmd_score_hscp(args)
        else:
            cmd_score_diva(args)


if __name__ == "__main__":
    main()
