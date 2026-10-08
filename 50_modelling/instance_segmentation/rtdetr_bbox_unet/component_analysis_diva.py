#!/usr/bin/env python3
"""Component-level vs. end-to-end analysis of the two-stage pipeline on the DIVA-HisDB test pages
(ConvNeXt-Tiny RT-DETR with CATMuS initialization, BCE crop U-Net, the thresholds selected on validation
for the thesis rows). Five conditions per subset, all scored with the DIVA Line Segmentation Evaluator:

  full       predicted boxes + crop U-Net masks (the thesis pipeline)
  gtbox      ground-truth line boxes + crop U-Net masks            -> isolates the detector
  oraclemask predicted boxes + ground-truth polygon inside the box  -> isolates the crop segmenter
             (boxes without a matching ground-truth line keep their crop U-Net mask)
  gt         ground-truth polygons                                   -> evaluator upper bound

Component metrics: box precision/recall/F1 at IoU 0.5 and 0.75 and AP@0.5 against the Task-2 line boxes
(after the region/width filter and duplicate-row removal), and the pixel IoU of the crop U-Net mask with the
ground-truth polygon inside each ground-truth box.

    SUBSETS="CB55 CS18 CS863" ../../.venv/bin/python component_analysis_diva.py
Results: 99_evaluation/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/component_analysis/{summary.json, <subset>/<condition>/*.xml}
"""
import json, os, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from xml.dom import minidom

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = next(str(p) for p in __import__("pathlib").Path(os.path.abspath(__file__)).parents if (p / "50_modelling").is_dir())  # repo root
sys.path.insert(0, HERE)
from dataset import create_page_xml, mask_to_polygon, polygon_to_baseline  # noqa: E402
from evaluate import _remove_overlapping_rows, load_segm_model, segment_crop  # noqa: E402
from rtdetr_load import load_detector  # noqa: E402
from transformers import AutoImageProcessor  # noqa: E402

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
JAR = os.path.expanduser("~/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar")
RUN = os.environ.get("RUN_NAME", "rtdetr_convnext_tiny_catmus")
ARM = os.environ.get("ARM", "bce")
# confidence and width fraction selected on validation for the thesis rows (99_evaluation/.../rtdetr_hf/results.csv)
OPERATING = {"CB55": (0.07, 0.0), "CS18": (0.01, 0.8), "CS863": (0.03, 0.6)}
OUT = os.path.join(REPO, "99_evaluation/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/component_analysis")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def paths(sub):
    d = os.path.join(REPO, "00_data/DIVA-HisDB", sub)
    det = os.path.join(REPO, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/detection/rtdetr_hf", "" if sub == "CB55" else sub, RUN, "best_model")
    seg = os.path.join(REPO, "80_models/instance_segmentation/rtdetr_bbox_unet/diva-hisdb/crop_seg_loss_ablation_components_1024x256", sub, ARM, "best.pth")
    return dict(img=os.path.join(d, f"img-{sub}/img/public-test"), page=os.path.join(d, f"PAGE-gt-{sub}-TASK-2/TASK-2/public-test"),
                pix=os.path.join(d, f"pixel-level-gt-{sub}/pixel-level-gt/public-test"), det=det, seg=seg)


def gt_lines(xml):
    root = ET.parse(xml).getroot()
    region = root.find(f".//{{{PAGE_NS}}}TextRegion/{{{PAGE_NS}}}Coords")
    region = None if region is None else np.array([list(map(int, p.split(","))) for p in region.attrib["points"].split()])
    polys = [np.array([list(map(int, p.split(","))) for p in tl.find(f"{{{PAGE_NS}}}Coords").attrib["points"].split()])
             for tl in root.findall(f".//{{{PAGE_NS}}}TextLine")]
    return region, polys


def detect(model, proc, img_rgb, thr):
    H, W = img_rgb.shape[:2]
    m = max(H, W)
    inputs = proc(images=img_rgb, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(**inputs)
    r = proc.post_process_object_detection(out, threshold=thr, target_sizes=[(m, m)])[0]
    b = r["boxes"].cpu().numpy().reshape(-1, 4); s = r["scores"].cpu().numpy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return np.concatenate([b[keep], s[keep, None]], 1)


def region_filter(boxes, region, frac):
    if region is None or not len(boxes):
        return boxes
    keep = np.array([cv2.pointPolygonTest(region.astype(np.int32), (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)), False) >= 0
                     for b in boxes], bool)
    if frac > 0:
        keep &= (boxes[:, 2] - boxes[:, 0]) >= frac * (region[:, 0].max() - region[:, 0].min())
    return boxes[keep]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def match(boxes, gtb, thr):
    """greedy by score -> (tp flags in score order, matched gt index per box)"""
    order = np.argsort(-boxes[:, 4]) if len(boxes) else []
    used, tp, idx = set(), [], {}
    for k in order:
        best, bi = 0.0, -1
        for i, g in enumerate(gtb):
            if i not in used:
                v = iou(boxes[k], g)
                if v > best:
                    best, bi = v, i
        ok = best >= thr
        tp.append(ok)
        if ok:
            used.add(bi); idx[k] = bi
    return np.array(tp, bool), idx


def ap50(tp, n_gt):
    if not len(tp):
        return 0.0
    ctp = np.cumsum(tp); fp = np.cumsum(~tp)
    rec = ctp / max(n_gt, 1); prec = ctp / (ctp + fp)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([env[rec >= r].max() if (rec >= r).any() else 0.0 for r in np.linspace(0, 1, 101)]))


def write_xml(path, img_path, W, H, polys):
    root, region = create_page_xml(img_path, W, H)
    for i, poly in enumerate(polys):
        bl = polygon_to_baseline(poly)
        if len(poly) < 3 or len(bl) < 2:
            continue
        tl = ET.SubElement(region, "TextLine", {"id": f"textline_{i}", "custom": "0"})
        ET.SubElement(tl, "Coords", {"points": " ".join(f"{x},{y}" for x, y in poly)})
        ET.SubElement(tl, "Baseline", {"points": " ".join(f"{x},{y}" for x, y in bl)})
        ET.SubElement(ET.SubElement(tl, "TextEquiv"), "Unicode").text = ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  "))


def crop_poly(seg, img, box, fallback=None):
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    mask = segment_crop(crop, seg, 1024, 256, DEV, 0.5)
    fg = float((mask > 0).mean())
    if fg > 0.95 or fg < 0.005:
        return None, mask
    return mask_to_polygon(mask, x1, y1), mask


def diva(p, stem, xml):
    with tempfile.TemporaryDirectory() as cwd:
        shutil.copy(xml, os.path.join(cwd, f"{stem}.xml"))
        subprocess.run(["java", "-Djava.awt.headless=true", "-cp", f"/usr/share/openjfx/lib/*:{JAR}", "ch.unifr.LineSegmentationEvaluatorTool",
                        "-igt", os.path.join(p["pix"], f"{stem}.png"), "-xgt", os.path.join(p["page"], f"{stem}.xml"),
                        "-xp", os.path.join(cwd, f"{stem}.xml"), "-csv"], cwd=cwd, capture_output=True, text=True, check=True)
        lines = open(os.path.join(cwd, "results.csv")).read().splitlines()
        head, vals = lines[0].split(","), lines[1].split(",")
        return dict(zip(head[1:], [float(v) for v in vals[-(len(head) - 1):]]))


def run_subset(sub):
    p = paths(sub)
    thr, frac = OPERATING[sub]
    det = load_detector(p["det"]).to(DEV).eval()
    proc = AutoImageProcessor.from_pretrained(p["det"])
    seg = load_segm_model(p["seg"], "resnet34", "unet", DEV)
    conds = ["full", "gtbox", "oraclemask", "gt"]
    for c in conds:
        os.makedirs(os.path.join(OUT, sub, c), exist_ok=True)
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(p["pix"]) if f.endswith(".png"))
    all_tp50, all_tp75, n_gt, n_pred, ap_tp, ious = [], [], 0, 0, [], []
    for stem in stems:
        img_path = os.path.join(p["img"], f"{stem}.jpg")
        img = cv2.imread(img_path); H, W = img.shape[:2]
        region, polys = gt_lines(os.path.join(p["page"], f"{stem}.xml"))
        gtb = [(q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()) for q in polys]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detector at the operating point (as in the thesis pipeline) and at the lowest grid value for AP
        boxes = _remove_overlapping_rows(region_filter(detect(det, proc, rgb, thr), region, frac))
        low = _remove_overlapping_rows(region_filter(detect(det, proc, rgb, 0.01), region, frac))
        tp_low, _ = match(low, gtb, 0.5)
        ap_tp.append((low[np.argsort(-low[:, 4])][:, 4] if len(low) else np.zeros(0), tp_low))
        tp50, idx = match(boxes, gtb, 0.5)
        tp75, _ = match(boxes, gtb, 0.75)
        all_tp50.append(int(tp50.sum())); all_tp75.append(int(tp75.sum())); n_gt += len(gtb); n_pred += len(boxes)
        # full and oracle-mask conditions
        full, orc = [], []
        for k, b in enumerate(boxes):
            poly, _ = crop_poly(seg, img, b)
            if poly is not None and len(poly) >= 3:
                full.append(poly)
            if k in idx:      # matched box: ground-truth polygon clipped to the box
                m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [polys[idx[k]].astype(np.int32)], 1)
                x1, y1, x2, y2 = (int(v) for v in b[:4]); clip = np.zeros_like(m); clip[y1:y2, x1:x2] = m[y1:y2, x1:x2]
                q = mask_to_polygon(clip * 255, 0, 0)
                if q is not None and len(q) >= 3:
                    orc.append(q)
            elif poly is not None and len(poly) >= 3:
                orc.append(poly)
        # ground-truth boxes + crop U-Net, and crop IoU
        gtc = []
        for q, g in zip(polys, gtb):
            poly, mask = crop_poly(seg, img, g)
            if mask is not None:
                x1, y1, x2, y2 = (int(v) for v in g)
                ref = np.zeros((y2 - y1, x2 - x1), np.uint8); cv2.fillPoly(ref, [(q - [x1, y1]).astype(np.int32)], 1)
                pm = mask[:ref.shape[0], :ref.shape[1]] > 0
                ious.append(float((pm & (ref > 0)).sum() / max((pm | (ref > 0)).sum(), 1)))
            if poly is not None and len(poly) >= 3:
                gtc.append(poly)
        for c, ps in (("full", full), ("gtbox", gtc), ("oraclemask", orc), ("gt", [[tuple(v) for v in q] for q in polys])):
            write_xml(os.path.join(OUT, sub, c, f"{stem}.xml"), img_path, W, H, ps)
        print(sub, stem, "boxes", len(boxes), "gt", len(gtb), "tp50", int(tp50.sum()), flush=True)
    # AP@0.5 over all pages of the subset
    scores = np.concatenate([s for s, _ in ap_tp]); tps = np.concatenate([t for _, t in ap_tp])
    order = np.argsort(-scores)
    res = {"box_precision50": sum(all_tp50) / max(n_pred, 1), "box_recall50": sum(all_tp50) / max(n_gt, 1),
           "box_precision75": sum(all_tp75) / max(n_pred, 1), "box_recall75": sum(all_tp75) / max(n_gt, 1),
           "box_ap50": ap50(tps[order], n_gt), "crop_iou_mean": float(np.mean(ious)), "crop_iou_median": float(np.median(ious)),
           "gt_lines": n_gt, "pred_boxes": n_pred, "threshold": thr, "width_fraction": frac}
    for k in ("50", "75"):
        pr, rc = res[f"box_precision{k}"], res[f"box_recall{k}"]
        res[f"box_f1{k}"] = 2 * pr * rc / max(pr + rc, 1e-9)
    with ThreadPoolExecutor(int(os.environ.get("DIVA_WORKERS", "6"))) as ex:
        for c in conds:
            rows = list(ex.map(lambda st: diva(p, st, os.path.join(OUT, sub, c, f"{st}.xml")), stems))
            res[f"fm_{c}"] = float(np.mean([r["LinesFMeasure"] for r in rows]))
            res[f"lineiu_{c}"] = float(np.mean([r["LinesIU"] for r in rows]))
            res[f"pixeliu_{c}"] = float(np.nanmean([r["PixelIU"] for r in rows]))
    del det, seg; torch.cuda.empty_cache()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(OUT, "summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {}
    for sub in os.environ.get("SUBSETS", "CB55 CS18 CS863").split():
        if sub in summary:
            continue
        summary[sub] = run_subset(sub)
        json.dump(summary, open(sp, "w"), indent=1)
        print(sub, {k: round(v, 4) if isinstance(v, float) else v for k, v in summary[sub].items()}, flush=True)


if __name__ == "__main__":
    main()
