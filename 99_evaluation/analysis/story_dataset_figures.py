#!/usr/bin/env python3
"""Dataset figures for the thesis, all as native-resolution crops of training pages (no rescaling):
  gtfmt_*      ground-truth formats: U-DIADS-TL Latin2 image + binary line mask, DIVA-HisDB CS18 image with
               PAGE XML (TASK-2) line polygons
  rq3_*        one crop per cross-collection dataset with its ground-truth line polygons
Output: 99_evaluation/analysis/story_figures/ (copied into the thesis graphics folder by hand).

    .venv/bin/python 99_evaluation/analysis/story_dataset_figures.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "99_evaluation/analysis/story_figures"
PAL = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189), (23, 190, 207)]
Image.MAX_IMAGE_PIXELS = None


def save(arr, name, jpg=True):
    OUT.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr)
    im.save(OUT / name, quality=92) if jpg else im.save(OUT / name)
    print(name, im.size)


def draw_polys(img, polys, lw):
    out = img.copy()
    for i, p in enumerate(sorted(polys, key=lambda q: q[:, 1].mean())):
        cv2.polylines(out, [np.rint(p).astype(np.int32)], True, PAL[i % len(PAL)], lw)
    return out


def crop_box(polys, W, H, wfrac, hfrac, xpos=0.0, ypos=0.5):
    allp = np.concatenate(polys)
    x0, y0 = allp.min(0); x1, y1 = allp.max(0)
    bw, bh = x1 - x0, y1 - y0
    cw, chh = wfrac * bw, hfrac * bh
    cx0 = x0 + xpos * (bw - cw) - 0.02 * bw
    cy0 = y0 + ypos * (bh - chh)
    return int(max(0, cx0)), int(max(0, cy0)), int(min(W, cx0 + cw + 0.04 * bw)), int(min(H, cy0 + chh))


def gt_formats():
    # U-DIADS-TL Latin2: image and binary mask
    d = ROOT / "00_data/U-DIADS-TL/Latin2"
    stem = "083"
    img = np.asarray(Image.open(next((d / "img-Latin2/training").glob(stem + ".*"))).convert("RGB"))
    gt = np.asarray(Image.open(d / "text-line-gt-Latin2/training" / f"{stem}.png").convert("L")) > 127
    ys, xs = np.where(gt)
    H, W = gt.shape
    x0, x1 = xs.min() - 20, xs.min() + int(0.55 * (xs.max() - xs.min()))
    y0 = ys.min() + int(0.30 * (ys.max() - ys.min())); y1 = y0 + int(0.18 * (ys.max() - ys.min()))
    save(img[y0:y1, x0:x1], "gtfmt_udiads_img.jpg")
    save(np.where(gt[y0:y1, x0:x1, None], 0, 255).astype(np.uint8).repeat(3, 2), "gtfmt_udiads_mask.png", jpg=False)
    # DIVA-HisDB CS18 training page with TASK-2 polygons
    d = ROOT / "00_data/DIVA-HisDB/CS18"
    xmls = sorted((d / "PAGE-gt-CS18-TASK-2/TASK-2/training").glob("*.xml"))
    xml = xmls[0]
    ns = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}"
    polys = [np.array([[float(v) for v in pt.split(",")] for pt in c.get("points").split()])
             for c in ET.parse(xml).getroot().iter(f"{ns}Coords") if c.get("points")]
    region = polys[0]
    lines = [np.array([[float(v) for v in pt.split(",")] for pt in tl.find(f"{ns}Coords").get("points").split()])
             for tl in ET.parse(xml).getroot().iter(f"{ns}TextLine")]
    img = np.asarray(Image.open(next((d / "img-CS18/img/training").glob(xml.stem + ".*"))).convert("RGB"))
    H, W = img.shape[:2]
    x0, y0, x1, y1 = crop_box(lines, W, H, 0.55, 0.22, xpos=0.0, ypos=0.45)
    y0 = max(0, y0 - 45); x1 = min(W, x0 + 1060)
    y1 = min(H, y0 + int(round((x1 - x0) / (614 / 277))))  # same aspect as the U-DIADS-TL crop (2x2 figure)
    save(img[y0:y1, x0:x1], "gtfmt_diva_img.jpg")
    save(draw_polys(img, lines, 5)[y0:y1, x0:x1], "gtfmt_diva_polys.jpg")
    print("DIVA page", xml.stem, "region pts", len(region))


def rq3():
    names = {"Pinkas": "Pinkas", "ONB": "ONB", "RASAM": "RASAM", "RASM": "RASM", "Phil_gr_130": "Phil_gr_130",
             "GRPOLY": "GRPOLY", "NorHand_v3": "NorHand_v3"}
    for key in names:
        root = ROOT / ("00_data/RQ3/final_roots/NorHand_v3" if key == "NorHand_v3" else f"00_data/RQ3/matrix/{key}")
        coco = json.loads((root / "coco_instances/train.json").read_text())
        ims = sorted(coco["images"], key=lambda x: x["file_name"])
        best = max(ims, key=lambda im: sum(a["image_id"] == im["id"] for a in coco["annotations"]))
        polys = [np.array(a["segmentation"][0], float).reshape(-1, 2) for a in coco["annotations"] if a["image_id"] == best["id"]]
        img = np.asarray(Image.open(root / "images/train" / best["file_name"]).convert("RGB"))
        H, W = img.shape[:2]
        lw = max(3, int(round(max(H, W) / 900)))
        x0, y0, x1, y1 = crop_box(polys, W, H, 0.6, min(0.45, 8.0 / max(len(polys), 1) * 1.0), xpos=0.0, ypos=0.4)
        save(draw_polys(img, polys, lw)[y0:y1, x0:x1], f"rq3_{key}.jpg")


if __name__ == "__main__":
    gt_formats()
    rq3()
