#!/usr/bin/env python3
"""Merge / split / miss rates of predicted line instances, stratified by page density (dev only).

Reads the raw Mask R-CNN detections cached by rq3_bench.py (dense models, scale 1, dev pages of
Phil. gr. 130, Pinkas, NorHand) and compares the instance map with the ground-truth polygons on
ink pixels. For GT line j and predicted instance i, c_ij = |P_i & G_j & ink| / |G_j & ink|.
  merged : a prediction with c >= TAU on this line also has c >= TAU on another GT line
  split  : two or more predictions have c >= TAU on this line
  missed : no prediction reaches c >= TAU
Page density = mean fraction of a GT line's bounding box covered by other GT polygons.
No model is run and no test page is read.

    nice -n 19 .venv/bin/python 50_modelling/semantic_segmentation/center_line/rq3_merge_split.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, pickle, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_bench as B  # noqa: E402

TAU = 0.25
ROOT = Path(__file__).resolve().parents[3]
OUT = B.OUT / "merge_split"


def gt_masks(coll, meta):
    coco = GT.setdefault(coll, json.loads((ROOT / f"00_data/RQ3/dev/{coll}/coco_instances/dev.json").read_text()))
    name = Path(meta["path"]).name
    img = next(im for im in coco["images"] if im["file_name"] == name)
    fit = meta["new_w"] / meta["orig_w"]
    h, w = meta["new_h"], meta["new_w"]
    masks = []
    for a in coco["annotations"]:
        if a["image_id"] != img["id"]:
            continue
        m = np.zeros((h, w), np.uint8)
        for seg in a["segmentation"]:
            cv2.fillPoly(m, [np.round(np.asarray(seg, np.float32).reshape(-1, 2) * fit).astype(np.int32)], 1)
        masks.append(m.astype(bool))
    return masks


def ink_mask(gray):
    g = gray.astype(np.float32)
    norm = g / np.maximum(cv2.GaussianBlur(g, (0, 0), 25), 1)
    u8 = np.clip(norm * 170, 0, 255).astype(np.uint8)
    _, ink = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink.astype(bool)


def density(masks):
    union_others = np.zeros_like(masks[0], np.uint16)
    for m in masks:
        union_others += m
    fr = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        others = (union_others[box] - m[box]) > 0
        fr.append(others.mean())
    return float(np.mean(fr)) if fr else np.nan


def rates(canvas, masks, ink):
    n_pred = int(canvas.max())
    cov = np.zeros((n_pred + 1, len(masks)))
    for j, g in enumerate(masks):
        px = g & ink
        tot = px.sum()
        if tot == 0:
            continue
        cov[:, j] = np.bincount(canvas[px].astype(np.int64), minlength=n_pred + 1)[:n_pred + 1] / tot
    hit = cov[1:] >= TAU                       # drop background label 0
    lines_per_pred = hit.sum(1)
    merged = np.array([(hit[:, j] & (lines_per_pred >= 2)).any() for j in range(len(masks))])
    split = hit.sum(0) >= 2
    missed = hit.sum(0) == 0
    return merged.mean(), split.mean(), missed.mean(), int((lines_per_pred > 0).sum())


GT = {}


def main():
    rows = []
    for f in sorted((B.OUT / "cache").glob("*_dense.pkl")):
        src = f.name[: -len("_dense.pkl")]
        for page in pickle.load(open(f, "rb")):
            masks = gt_masks(page["coll"], page["meta"])
            ink = ink_mask(page["gray"])
            dens = density(masks)
            for assign, canvas in (("score", B.canvas_score(page, page["mt"])),
                                   ("pixel", B.canvas_pixel(page, page["mt"]))):
                mr, sr, xr, n_used = rates(canvas, masks, ink)
                rows.append({"source": src, "target": page["coll"], "page": Path(page["meta"]["path"]).stem,
                             "assignment": assign, "n_gt": len(masks), "n_pred": n_used, "density": dens,
                             "merged": mr, "split": sr, "missed": xr})
        print("done", src, flush=True)
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "pages.csv", index=False)
    pd.set_option("display.width", 200)
    pct = ["merged", "split", "missed"]
    print("\nper target, score assignment (mean over sources and pages, %):")
    print((d[d.assignment == "score"].groupby("target")[pct + ["density"]].mean() * 100).round(1).to_string())
    print("\nscore vs pixel assignment per target (%):")
    print((d.groupby(["target", "assignment"])[pct].mean() * 100).round(1).unstack().to_string())
    pg = d[d.assignment == "score"].drop_duplicates(["target", "page"])
    edges = pg.density.quantile([0, 1 / 3, 2 / 3, 1]).values
    d["stratum"] = pd.cut(d.density, edges, labels=["low", "mid", "high"], include_lowest=True)
    print("\ndensity tertiles (GT box overlap):", np.round(edges * 100, 1))
    print((d.groupby(["stratum", "assignment"], observed=True)[pct].mean() * 100).round(1).unstack().to_string())
    print(d.groupby("stratum", observed=True).target.agg(lambda s: dict(s.value_counts() // 2)).to_string())


if __name__ == "__main__":
    main()
