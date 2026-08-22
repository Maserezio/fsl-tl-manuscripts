#!/usr/bin/env python3
"""ABLATE POST-PROCESSING (2-stage, DIVA) — single-page sweep, existing weights, no training.

Isolates the three suspects that drop ascenders / descenders / abbreviation marks:

  A. crop padding      fixed 15 px vs. fraction-of-box-height
  B. closing kernel    (75, 13) flat vs. vertically-bridging kernels
  C. polygon export    largest-contour-only vs. satellite-component merging

Detection is run ONCE per page. Stage-2 segmentation is cached per crop-pad setting
(it is the only expensive thing that depends on padding). Everything downstream —
closing + polygon export — is pure OpenCV and is swept cheaply on those cached masks.

Outputs, per config:
  <out>/<cfg>/pred_xml/<stem>.xml    PAGE-XML
  <out>/<cfg>/overlay.jpg            page + polygons (satellite pixels in magenta)
  <out>/results.csv                  DIVA metrics + diagnostics, sorted by FMeasure

  python ablate_postproc.py --manuscript CB55 --stem e-codices_fmb-cb-0055_0116r_max
  python ablate_postproc.py --manuscript CB55                 # first test page
  python ablate_postproc.py --manuscript CB55 --no-eval       # overlays only, no Java
"""
import argparse
import csv
import itertools
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SS = REPO / "50_modelling" / "01_simple_segmentation"
sys.path.insert(0, str(SS))
from predict_diva_seamcarve import (create_page_xml, load_main_region_polygon,   # noqa: E402
                                    run_diva_evaluator, PAGE_NS)
from data.diva_dataset import _resolve_image_path, _resolve_layout               # noqa: E402

SEG_DEFAULT = REPO / "80_models" / "02_2stage" / "diva-hisdb" / "segmentation" / "yolo_seg_cb55" / "weights" / "best.pt"
DET_DEFAULT = REPO / "80_models" / "02_2stage" / "diva-hisdb" / "detection" / "yolo_cb55_finetune_500" / "train" / "weights" / "best.pt"
JAVA_CP = ("/usr/share/openjfx/lib/*:"
           "/home/artur/Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar")
JAVA_MAIN = "ch.unifr.LineSegmentationEvaluatorTool"


# --------------------------------------------------------------------------------------
# polygon export — the variant under test
# --------------------------------------------------------------------------------------
def mask_to_polygon(mask, region_mask, min_area=800, approx_ratio=0.005,
                    satellites=False, satellite_min=20, link_dist=30):
    """Instance mask -> region-clipped polygon (Nx2), and #pixels contributed by satellites.

    satellites=False reproduces the current production behaviour: keep only the largest
    external contour, which silently deletes every detached diacritic / abbreviation mark.
    satellites=True re-attaches nearby detached components before contour extraction.

    Returns (poly_or_None, satellite_pixel_count).
    """
    m = mask.astype(np.uint8)
    if region_mask is not None:
        m = m & region_mask
    if int(m.sum()) < min_area:
        return None, 0

    sat_px = 0
    if satellites:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            return None, 0
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = stats[main, :4]
        keep = (lab == main).astype(np.uint8)
        for i in range(1, n):
            if i == main or stats[i, cv2.CC_STAT_AREA] < satellite_min:
                continue
            xi, yi, wi, hi = stats[i, :4]
            if xi > x + w + link_dist or xi + wi < x - link_dist:
                continue
            if yi > y + h + link_dist or yi + hi < y - link_dist:
                continue
            keep[lab == i] = 1
            sat_px += int(stats[i, cv2.CC_STAT_AREA])
        # vertical bridge so satellites land inside ONE external contour
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_RECT, (15, 2 * link_dist + 1)))
        m = keep

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, approx_ratio * cv2.arcLength(c, True), True)
    return (approx.reshape(-1, 2) if len(approx) >= 3 else None), sat_px


# --------------------------------------------------------------------------------------
# stage 1 + stage 2, cached
# --------------------------------------------------------------------------------------
def detect_boxes(det_model, img, conf, det_imgsz):
    r = det_model.predict(img, imgsz=det_imgsz, conf=conf, verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy().astype(int)


def raw_masks_for_pad(seg_model, img, boxes, pad_spec, crop_imgsz, center_filter):
    """Stage-2 per padded crop -> list of full-page boolean masks, WITHOUT closing.

    pad_spec: ("abs", px) or ("frac", f) — vertical pad = f * box_height, horizontal = 0.05 * w.
    center_filter: if True, keep only stage-2 instances overlapping the central band of the
        ORIGINAL (unpadded) box. Guards against neighbouring lines entering a large crop.
    """
    H, W = img.shape[:2]
    out = []
    for x0, y0, x1, y1 in boxes:
        bh, bw = y1 - y0, x1 - x0
        if pad_spec[0] == "abs":
            ph = pw = int(pad_spec[1])
        else:
            ph, pw = int(pad_spec[1] * bh), int(0.05 * bw)
        cx0, cy0 = max(0, x0 - pw), max(0, y0 - ph)
        cx1, cy1 = min(W, x1 + pw), min(H, y1 + ph)
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        r = seg_model.predict(crop, imgsz=crop_imgsz, conf=0.1, verbose=False, retina_masks=True)[0]
        if r.masks is None or len(r.masks.data) == 0:
            continue
        masks = (r.masks.data.cpu().numpy() > 0.5)
        if masks.shape[1:] != (cy1 - cy0, cx1 - cx0):
            masks = np.stack([cv2.resize(m.astype(np.uint8), (cx1 - cx0, cy1 - cy0),
                                         interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
        if center_filter and len(masks) > 1:
            # central 50% band of the original box, in crop coordinates
            b0 = (y0 - cy0) + int(0.25 * bh)
            b1 = (y0 - cy0) + int(0.75 * bh)
            sel = [m for m in masks if m[max(0, b0):max(1, b1), :].any()]
            masks = np.stack(sel) if sel else masks
        union = masks.any(axis=0)
        canvas = np.zeros((H, W), bool)
        canvas[cy0:cy1, cx0:cx1] = union
        out.append(canvas)
    return out


# --------------------------------------------------------------------------------------
def write_xml(path, stem, W, H, region_pts, polys, tag):
    tree, reg = create_page_xml(f"{stem}.jpg", W, H, tag, region_pts)
    for i, poly in enumerate(polys):
        ymax = int(poly[:, 1].max()); xmin = int(poly[:, 0].min()); xmax = int(poly[:, 0].max())
        tl = ET.SubElement(reg, f"{{{PAGE_NS}}}TextLine", {"id": f"textline_{i}", "custom": "0"})
        ET.SubElement(tl, f"{{{PAGE_NS}}}Coords",
                      {"points": " ".join(f"{int(x)},{int(y)}" for x, y in poly)})
        ET.SubElement(tl, f"{{{PAGE_NS}}}Baseline", {"points": f"{xmin},{ymax} {xmax},{ymax}"})
        ET.SubElement(ET.SubElement(tl, f"{{{PAGE_NS}}}TextEquiv"), f"{{{PAGE_NS}}}Unicode").text = ""
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="utf-8", xml_declaration=True)


def save_overlay(path, img, polys, base_masks, final_masks):
    """Green = exported polygon outline. Magenta = pixels recovered vs. the largest-only path."""
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    gained = (final_masks & ~base_masks)
    vis[gained] = (255, 0, 255)
    for poly in polys:
        cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 0), 3)
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manuscript", default="CB55")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stem", default=None, help="page stem; default = first page of split")
    ap.add_argument("--seg-weights", default=str(SEG_DEFAULT))
    ap.add_argument("--det-weights", default=str(DET_DEFAULT))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--crop-imgsz", type=int, default=640)
    ap.add_argument("--min-area", type=int, default=800)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--java-cp", default=JAVA_CP)
    # ---- sweep axes -------------------------------------------------------------------
    ap.add_argument("--pads", default="abs:15,frac:0.25,frac:0.40,frac:0.60",
                    help="comma list of abs:<px> / frac:<fraction of box height>")
    ap.add_argument("--kernels", default="75x13,75x41,45x61",
                    help="comma list of WxH closing kernels; 'none' to skip closing")
    ap.add_argument("--link-dists", default="20,30,50",
                    help="comma list of satellite link distances (px)")
    ap.add_argument("--center-filter", default="0,1", help="comma list of 0/1")
    args = ap.parse_args()

    pads = []
    for tok in args.pads.split(","):
        k, v = tok.split(":")
        pads.append((k, float(v) if k == "frac" else int(v)))
    kernels = [None if t == "none" else tuple(int(z) for z in t.split("x"))
               for t in args.kernels.split(",")]
    link_dists = [int(t) for t in args.link_dists.split(",")]
    cfilters = [bool(int(t)) for t in args.center_filter.split(",")]

    from ultralytics import YOLO
    det_model = YOLO(args.det_weights)
    seg_model = YOLO(args.seg_weights)

    MS = args.manuscript
    d = _resolve_layout(str(REPO / "00_data" / "DIVA-HisDB"), MS, "diva")[args.split]
    stems = sorted({p.stem for p in Path(d["img"]).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")})
    stem = args.stem or stems[0]
    if stem not in stems:
        sys.exit(f"stem {stem!r} not in {args.split}; available: {stems[:5]} ...")

    out_root = Path(args.output_dir) if args.output_dir else (
        REPO / "99_evaluation" / "02_2stage" / "diva-hisdb" / f"ablate_postproc_{MS}_{stem}")
    out_root.mkdir(parents=True, exist_ok=True)

    img = cv2.cvtColor(cv2.imread(_resolve_image_path(d["img"], stem)), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    region_pts = load_main_region_polygon(Path(d["xml"]) / f"{stem}.xml")
    region_mask = None
    if region_pts is not None:
        region_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(region_mask, [np.array(region_pts, np.int32)], 1)

    t0 = time.time()
    boxes = detect_boxes(det_model, img, args.conf, args.imgsz)
    print(f"[{stem}] {W}x{H}, {len(boxes)} detections in {time.time()-t0:.1f}s")
    med_h = float(np.median(boxes[:, 3] - boxes[:, 1])) if len(boxes) else 0.0
    print(f"  median box height = {med_h:.0f}px  ->  frac:0.40 == {0.40*med_h:.0f}px pad")

    # cache stage-2 once per (pad, center_filter)
    cache = {}
    for pad, cf in itertools.product(pads, cfilters):
        t = time.time()
        cache[(pad, cf)] = raw_masks_for_pad(seg_model, img, boxes, pad, args.crop_imgsz, cf)
        print(f"  stage-2 pad={pad[0]}:{pad[1]} center_filter={int(cf)} "
              f"-> {len(cache[(pad, cf)])} masks ({time.time()-t:.1f}s)")

    rows = []
    for (pad, cf), kern, sat_mode in itertools.product(cache.keys(), kernels,
                                                       [("off", 0)] + [("on", ld) for ld in link_dists]):
        name = (f"pad-{pad[0]}{pad[1]}_cf{int(cf)}"
                f"_k{'none' if kern is None else f'{kern[0]}x{kern[1]}'}"
                f"_sat-{sat_mode[0]}{sat_mode[1] or ''}")
        cfg_dir = out_root / name
        (cfg_dir / "pred_xml").mkdir(parents=True, exist_ok=True)

        polys, sat_total = [], 0
        base_acc = np.zeros((H, W), bool)   # largest-contour-only reference, for the overlay
        final_acc = np.zeros((H, W), bool)
        for raw in cache[(pad, cf)]:
            m = raw
            if kern is not None:
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, kern)) > 0
            poly, spx = mask_to_polygon(m, region_mask, args.min_area,
                                        satellites=(sat_mode[0] == "on"), link_dist=sat_mode[1] or 30)
            if poly is None:
                continue
            polys.append(poly)
            sat_total += spx
            f = np.zeros((H, W), np.uint8); cv2.fillPoly(f, [poly.astype(np.int32)], 1)
            final_acc |= f.astype(bool)
            bpoly, _ = mask_to_polygon(m, region_mask, args.min_area, satellites=False)
            if bpoly is not None:
                b = np.zeros((H, W), np.uint8); cv2.fillPoly(b, [bpoly.astype(np.int32)], 1)
                base_acc |= b.astype(bool)

        write_xml(cfg_dir / "pred_xml" / f"{stem}.xml", stem, W, H, region_pts, polys, name)
        save_overlay(cfg_dir / "overlay.jpg", img, polys, base_acc, final_acc)

        row = {"config": name, "pad": f"{pad[0]}:{pad[1]}", "center_filter": int(cf),
               "kernel": "none" if kern is None else f"{kern[0]}x{kern[1]}",
               "satellites": sat_mode[0], "link_dist": sat_mode[1],
               "n_lines": len(polys), "sat_px": sat_total,
               "gained_px": int((final_acc & ~base_acc).sum())}

        if not args.no_eval:
            try:
                s = run_diva_evaluator(cfg_dir / "pred_xml", d["gt"], d["xml"], d["img"],
                                       cfg_dir / "diva_csv",
                                       java_cp=args.java_cp, java_main=JAVA_MAIN)
                row.update({k: float(s.get(k, float("nan"))) for k in
                            ("PixelIU", "LinesIU", "LinesPrecision", "LinesRecall", "LinesFMeasure")})
            except Exception as e:                                   # noqa: BLE001
                row["error"] = str(e)[:200]
        rows.append(row)
        print(f"  {name:<52} lines={row['n_lines']:>3} gained_px={row['gained_px']:>7} "
              f"FM={row.get('LinesFMeasure', float('nan')):.4f}")

    keys = sorted({k for r in rows for k in r})
    with (out_root / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config"] + [k for k in keys if k != "config"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_root / "args.json").write_text(json.dumps(vars(args), indent=2))

    rows.sort(key=lambda r: r.get("LinesFMeasure", -1), reverse=True)
    print(f"\n=== top 10 by LinesFMeasure — {MS}/{stem} ===")
    hdr = f"{'config':<52} {'FM':>7} {'PixIU':>7} {'LineIU':>7} {'R':>7} {'lines':>6} {'gained':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows[:10]:
        print(f"{r['config']:<52} {r.get('LinesFMeasure', float('nan')):>7.4f} "
              f"{r.get('PixelIU', float('nan')):>7.4f} {r.get('LinesIU', float('nan')):>7.4f} "
              f"{r.get('LinesRecall', float('nan')):>7.4f} {r['n_lines']:>6} {r['gained_px']:>8}")
    print(f"\n-> {out_root/'results.csv'}")


if __name__ == "__main__":
    main()
