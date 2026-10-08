#!/usr/bin/env python3
"""Overlapping ground-truth line polygons and how the line representations handle them.

DIVA-HisDB (Task 2, public test): the four representations of story_diva_pages.py (full training partition).
RQ3 collections (protocol A, test pages): Mask R-CNN "ab" (predictions re-exported by overlap_rq3_preds.sh)
and the CATMuS-pretrained center-line model (WiSE-FT 0.5, thesis decoding row:1.0).

Ink = main-text pixels of the pixel-level GT (blue bit 0x08), as in the DIVA evaluator. Shared ink of a GT
line = its ink that also lies inside another GT polygon. Matching: pixel IoU on ink, matched at IoU >= 0.75.

    .venv/bin/python 99_evaluation/analysis/overlap_lines_diva.py                      statistics -> story_figures/overlap_lines.csv
    .venv/bin/python 99_evaluation/analysis/overlap_lines_diva.py render DS STEM x0 y0 x1 y1
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import csv, sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_diva_pages as S  # noqa: E402

THR = 0.75
OUT = S.OUT
ROOT = S.ROOT
METHODS = ["unet", "twostage", "maskrcnn", "maskrcnn_crop", "centerline"]
RQ3 = ["NorHand_v3", "Phil_gr_130", "Pinkas", "ONB", "RASAM"]
BINS = [(0, 0), (1e-9, 0.05), (0.05, 0.15), (0.15, 1.01)]


def rq3_root(ds):
    return ROOT / ("00_data/RQ3/final_roots/NorHand_v3" if ds == "NorHand_v3" else f"00_data/RQ3/matrix/{ds}")


def spec(ds):
    """stems, gt xml, pixel gt, image, {method: prediction dir}"""
    if ds in S.SUBS:
        base = S.DIVA / ds
        stems = sorted(p.stem for p in (base / f"pixel-level-gt-{ds}/pixel-level-gt/public-test").glob("*.png"))
        return (stems, lambda s: base / f"PAGE-gt-{ds}-TASK-2/TASK-2/public-test/{s}.xml",
                lambda s: base / f"pixel-level-gt-{ds}/pixel-level-gt/public-test/{s}.png",
                lambda s: next((base / f"img-{ds}/img/public-test").glob(s + ".*")),
                {**{m: S.pred_dir(m, ds) for m in S.METHODS},
                 # Mask R-CNN boxes + Tversky crop U-Net, one polygon per box, overlaps kept (overlap_diva_croprefine.sh)
                 "maskrcnn_crop": ROOT / f"99_evaluation/instance_segmentation/mask_rcnn/diva-hisdb/{ds}/maskrcnn_convnext_tiny_catmus_704/test_crop_refine_tversky_selected"})
    r = rq3_root(ds)
    stems = sorted(p.stem for p in (r / "page-gt/test").glob("*.xml"))
    return (stems, lambda s: r / f"page-gt/test/{s}.xml",
            lambda s: next((r / "pixel-gt/test").glob(s + ".*")),
            lambda s: next((r / "images/test").glob(s + ".*")),
            {"maskrcnn": ROOT / f"99_evaluation/instance_segmentation/mask_rcnn/cross_collection/{ds}_maskrcnn_convnext_tiny_catmus_1152_k3_overlap/test_pred_xml",
             "centerline": ROOT / f"99_evaluation/semantic_segmentation/center_line/pred_lf/{ds}_A_w50_test/row_1.0/{ds}"})


def polys(xml):
    out = []
    for el in ET.parse(xml).getroot().iter():
        if el.tag.endswith("TextLine"):
            c = next(ch for ch in el if ch.tag.endswith("Coords"))
            p = np.array([[int(float(v)) for v in xy.split(",")] for xy in c.get("points").split()], np.int32)
            if len(p) >= 3:
                out.append(p)
    return out


def bbox(p):
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max() + 1, p[:, 1].max() + 1


def rast(p, box):
    x0, y0, x1, y1 = box
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [p - [x0, y0]], 1)
    return m.astype(bool)


def inter(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def union(a, b):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def analyse(job):
    ds, stem = job
    stems, gtx, pix, _, preds = spec(ds)
    fg = (cv2.imread(str(pix(stem)))[:, :, 0] & 0x08) > 0
    H, W = fg.shape
    g = [np.clip(p, 0, [W - 1, H - 1]) for p in polys(gtx(stem))]
    gb = [bbox(p) for p in g]
    cnt = np.zeros(fg.shape, np.uint8)
    for p in g:
        m = np.zeros(fg.shape, np.uint8)
        cv2.fillPoly(m, [p], 1)
        cnt += m
    shared = fg & (cnt >= 2)
    rows = []
    P = {}
    anyp = {}
    for m, d in preds.items():
        f = d / f"{stem}.xml"
        P[m] = [np.clip(p, 0, [W - 1, H - 1]) for p in polys(f)] if f.exists() else None
        if P[m] is not None:
            anyp[m] = np.zeros(fg.shape, np.uint8)
            cv2.fillPoly(anyp[m], P[m], 1)
    for i, p in enumerate(g):
        x0, y0, x1, y1 = gb[i]
        a = rast(p, gb[i]) & fg[y0:y1, x0:x1]
        sh = rast(p, gb[i]) & shared[y0:y1, x0:x1]
        r = {"ds": ds, "page": stem, "line": i, "ink": int(a.sum()), "shared": int(sh.sum())}
        for m, pr in P.items():
            if pr is None:
                r[f"{m}_iou"] = r[f"{m}_cov"] = r[f"{m}_inkrec"] = r[f"{m}_dropped"] = float("nan")
                continue
            r[f"{m}_dropped"] = int((a & (anyp[m][y0:y1, x0:x1] == 0)).sum())    # ink of this line in no prediction
            bi, cov, rec = 0.0, 0, 0.0
            for q in pr:
                qb = bbox(q)
                if not inter(gb[i], qb):
                    continue
                u = union(gb[i], qb)
                ux0, uy0, ux1, uy1 = u
                f = fg[uy0:uy1, ux0:ux1]
                gm, qm = rast(p, u) & f, rast(q, u) & f
                un = (gm | qm).sum()
                iou = (gm & qm).sum() / un if un else 0.0
                if iou > bi:
                    shu = rast(p, u) & shared[uy0:uy1, ux0:ux1]
                    bi, cov, rec = iou, int((shu & qm).sum()), (gm & qm).sum() / max(gm.sum(), 1)
            r[f"{m}_iou"], r[f"{m}_cov"], r[f"{m}_inkrec"] = round(float(bi), 4), cov, round(float(rec), 4)
        rows.append(r)
    return rows


def stats():
    dss = S.SUBS + RQ3
    jobs = [(ds, s) for ds in dss for s in spec(ds)[0]]
    with ProcessPoolExecutor(8) as ex:
        R = [r for rs in ex.map(analyse, jobs, chunksize=4) for r in rs]
    keys = sorted({k for r in R for k in r}, key=lambda k: list(R[0]).index(k) if k in R[0] else 99)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "overlap_lines.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys([k for r in R for k in r])))
        w.writeheader(); w.writerows(R)
    summarize(OUT / "overlap_lines.csv")


def summarize(path):
    import pandas as pd
    d = pd.read_csv(path)
    d["share"] = d.shared / d.ink.clip(lower=1)
    methods = [m for m in METHODS if f"{m}_iou" in d]
    for ds, g in d.groupby("ds", sort=False):
        ms = [m for m in methods if g[f"{m}_iou"].notna().any()]
        print(f"\n{ds}: {len(g)} lines, {100 * (g.shared > 0).mean():.1f}% with shared ink, "
              f"{100 * g.shared.sum() / g.ink.sum():.2f}% of all ink shared")
        print(f"  {'shared share':14s} {'n':>5s}  " + "  ".join(f"{m + ' rec / cov':>22s}" for m in ms))
        for lo, hi in BINS:
            b = g[(g.share >= lo) & (g.share <= hi)] if hi == 0 else g[(g.share > lo) & (g.share <= hi)]
            if not len(b):
                continue
            cells = []
            for m in ms:
                rec = 100 * (b[f"{m}_iou"] >= THR).mean()
                cov = 100 * b[f"{m}_cov"].sum() / b.shared.sum() if b.shared.sum() else float("nan")
                cells.append(f"{rec:6.1f} / {cov:5.1f}")
            lab = "none" if hi == 0 else f"{100 * lo:.0f}-{min(100, 100 * hi):.0f}%"
            print(f"  {lab:14s} {len(b):5d}  " + "  ".join(f"{c:>22s}" for c in cells))
        print(f"  {'ink in no pred.':14s} {'':5s}  " + "  ".join(
            f"{100 * g[f'{m}_dropped'].sum() / g.ink.sum():21.1f}%" for m in ms))


PAL = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189), (23, 190, 207)]


def render(ds, stem, x0, y0, x1, y1):
    _, gtx, pix, img_of, preds = spec(ds)
    img = cv2.cvtColor(cv2.imread(str(img_of(stem))), cv2.COLOR_BGR2RGB)
    fg = (cv2.imread(str(pix(stem)))[:, :, 0] & 0x08) > 0
    box = (x0, y0, x1, y1)
    sets = [("gt", polys(gtx(stem)))] + [(m, polys(d / f"{stem}.xml")) for m, d in preds.items()]
    for name, ps in sets:
        ps = [p for p in ps if inter(bbox(p), box)]
        ps.sort(key=lambda p: p[:, 1].mean())
        base = (0.55 * img[y0:y1, x0:x1] + 0.45 * 255).astype(np.float32)
        cnt = np.zeros(base.shape[:2], np.uint8)
        for k, p in enumerate(ps):
            m = rast(p, box)
            cnt += m
            sel = m & fg[y0:y1, x0:x1]
            base[sel] = 0.35 * base[sel] + 0.65 * np.array(PAL[k % len(PAL)])
        twice = (cnt >= 2) & fg[y0:y1, x0:x1]           # ink claimed by two polygons: black
        base[twice] = (20, 20, 20)
        o = base.astype(np.uint8).copy()
        for k, p in enumerate(ps):
            cv2.polylines(o, [p - [x0, y0]], True, PAL[k % len(PAL)], 3)
        cv2.imwrite(str(OUT / f"overlap_{ds}_{name}.png"), cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
        print(name, len(ps), "polygons, ink in two polygons:", int(twice.sum()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        render(sys.argv[2], sys.argv[3], *map(int, sys.argv[4:8]))
    elif len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(OUT / "overlap_lines.csv")
    else:
        stats()
