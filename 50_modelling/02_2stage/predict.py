"""PREDICT (2-stage, DIVA) — YOLO text-line segmentation -> PAGE-XML -> Java evaluator.

Two modes:
  single    one YOLO-seg model on the full page predicts line boxes + masks jointly
  twostage  stage-1 YOLO detector proposes line boxes, stage-2 YOLO-seg segments the
            mask inside each (padded) box crop

Both write one TextLine polygon per predicted line (mask -> region-clip -> contour),
then score with the official DIVA Java evaluator (Pixel IU, Line IU, Line P/R/F).

  python predict.py --mode single   --manuscript CB55
  python predict.py --mode twostage --manuscript CB55
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SS = REPO / "50_modelling" / "01_simple_segmentation"
sys.path.insert(0, str(SS))                                     # single source for XML + Java helpers
from predict_diva_seamcarve import (create_page_xml, load_main_region_polygon,   # noqa: E402
                                    run_diva_evaluator, PAGE_NS)
from data.diva_dataset import _resolve_image_path, _resolve_layout               # noqa: E402

SEG_DEFAULT = REPO / "80_models" / "02_2stage" / "diva-hisdb" / "segmentation" / "yolo_seg_cb55" / "weights" / "best.pt"
DET_DEFAULT = REPO / "80_models" / "02_2stage" / "diva-hisdb" / "detection" / "yolo_cb55_finetune_500" / "train" / "weights" / "best.pt"


def mask_to_polygon(mask, region_mask, min_area=800, approx_ratio=0.005):
    """Instance mask -> region-clipped outer-contour polygon (Nx2) or None."""
    m = mask.astype(np.uint8)
    if region_mask is not None:
        m = m & region_mask
    if m.sum() < min_area:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, approx_ratio * cv2.arcLength(c, True), True)
    return approx.reshape(-1, 2) if len(approx) >= 3 else None


def instance_masks_single(seg_model, img, conf, imgsz):
    """Full-page YOLO-seg -> list of full-res boolean masks."""
    H, W = img.shape[:2]
    r = seg_model.predict(img, imgsz=imgsz, conf=conf, verbose=False, retina_masks=False)[0]
    out = []
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy():          # model-space; page letterboxes to exact stride
            out.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5)
    return out


def instance_masks_twostage(det_model, seg_model, img, conf, det_imgsz, crop_pad, crop_imgsz):
    """YOLO det boxes -> YOLO-seg per padded crop -> masks pasted at page scale."""
    H, W = img.shape[:2]
    det = det_model.predict(img, imgsz=det_imgsz, conf=conf, verbose=False)[0]
    out = []
    for box in det.boxes.xyxy.cpu().numpy().astype(int):
        x0, y0, x1, y1 = box
        x0, y0 = max(0, x0 - crop_pad), max(0, y0 - crop_pad)
        x1, y1 = min(W, x1 + crop_pad), min(H, y1 + crop_pad)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        r = seg_model.predict(crop, imgsz=crop_imgsz, conf=0.1, verbose=False, retina_masks=True)[0]
        if r.masks is None or len(r.masks.data) == 0:
            continue
        masks = r.masks.data.cpu().numpy()                       # retina_masks -> crop-native shape
        best = (masks > 0.5).any(axis=0)          # UNION of instances: model may split one line into
        if best.shape != (y1 - y0, x1 - x0):      # fragments; largest-only drops ink and kills recall
            best = cv2.resize(best.astype(np.uint8), (x1 - x0, y1 - y0),
                              interpolation=cv2.INTER_NEAREST) > 0
        # connect word fragments into ONE component (mask_to_polygon keeps the largest contour,
        # so an unconnected mask would export only one word group) — same role as U-Net's connect_line
        best = cv2.morphologyEx(best.astype(np.uint8), cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (75, 13))) > 0
        canvas = np.zeros((H, W), bool)
        canvas[y0:y1, x0:x1] = best
        out.append(canvas)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["single", "twostage"], required=True)
    ap.add_argument("--manuscript", default="CB55")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seg-weights", default=str(SEG_DEFAULT))
    ap.add_argument("--det-weights", default=str(DET_DEFAULT))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024, help="single-stage / detector inference size")
    ap.add_argument("--crop-imgsz", type=int, default=640, help="stage-2 seg size per crop")
    ap.add_argument("--crop-pad", type=int, default=15)
    ap.add_argument("--min-area", type=int, default=800)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    seg_model = YOLO(args.seg_weights)
    det_model = YOLO(args.det_weights) if args.mode == "twostage" else None

    MS = args.manuscript
    d = _resolve_layout(str(REPO / "00_data" / "DIVA-HisDB"), MS, "diva")[args.split]
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO / "99_evaluation" / "02_2stage" / "diva-hisdb" / f"yolo_{args.mode}_{MS}_{args.split}")
    xml_dir = out_dir / "pred_xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted({p.stem for p in Path(d["img"]).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")})
    total = 0
    for stem in tqdm(stems, desc=f"yolo {args.mode}"):
        img = cv2.cvtColor(cv2.imread(_resolve_image_path(d["img"], stem)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        region_pts = load_main_region_polygon(Path(d["xml"]) / f"{stem}.xml")
        region_mask = None
        if region_pts is not None:
            region_mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(region_mask, [np.array(region_pts, np.int32)], 1)

        if args.mode == "single":
            masks = instance_masks_single(seg_model, img, args.conf, args.imgsz)
        else:
            masks = instance_masks_twostage(det_model, seg_model, img, args.conf,
                                            args.imgsz, args.crop_pad, args.crop_imgsz)

        tree, reg = create_page_xml(f"{stem}.jpg", W, H, f"YOLO-{args.mode}", region_pts)
        kept = 0
        for mask in masks:
            poly = mask_to_polygon(mask, region_mask, args.min_area)
            if poly is None:
                continue
            ymax = int(poly[:, 1].max()); xmin = int(poly[:, 0].min()); xmax = int(poly[:, 0].max())
            tl = ET.SubElement(reg, f"{{{PAGE_NS}}}TextLine", {"id": f"textline_{kept}", "custom": "0"})
            ET.SubElement(tl, f"{{{PAGE_NS}}}Coords",
                          {"points": " ".join(f"{int(x)},{int(y)}" for x, y in poly)})
            ET.SubElement(tl, f"{{{PAGE_NS}}}Baseline", {"points": f"{xmin},{ymax} {xmax},{ymax}"})
            ET.SubElement(ET.SubElement(tl, f"{{{PAGE_NS}}}TextEquiv"), f"{{{PAGE_NS}}}Unicode").text = ""
            kept += 1
        ET.indent(tree, space="  ")
        tree.write(str(xml_dir / f"{stem}.xml"), encoding="utf-8", xml_declaration=True)
        total += kept

    print(f"Wrote {len(stems)} PAGE XML files ({total} text lines) -> {xml_dir}")

    if not args.no_eval:
        summary = run_diva_evaluator(
            xml_dir, d["gt"], d["xml"], d["img"], out_dir / "diva_csv",
            java_cp="/usr/share/openjfx/lib/*:/home/artur/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar",
            java_main="ch.unifr.LineSegmentationEvaluatorTool")
        with (out_dir / "diva_summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary.keys()))
            w.writeheader(); w.writerow(summary)
        print(f"\n=== yolo {args.mode} — {MS} {args.split} (DIVA Java) ===")
        for k in ("PixelIU", "LinesIU", "LinesPrecision", "LinesRecall", "LinesFMeasure"):
            print(f"  {k}: {summary.get(k, float('nan')):.4f}")


if __name__ == "__main__":
    main()
