#!/usr/bin/env python3
"""Chapter 6 figures for the failure analysis of 2026-10-06 (see 99_evaluation/analysis/diva_error_taxonomy.py):
  fragments  CB55 test page 0108v: GT polygons that cover only a line-initial abbreviation sign, and Mask R-CNN
  rtmiss     CS863 test page 013: lines that RT-DETR + BBox U-Net misses completely
  numerals   U-DIADS-TL Latin2 test page 230: small GT components (marginal chapter numbers) and BBox U-Net output
Writes PNGs to 99_evaluation/analysis/story_figures and the thesis graphics folder.
    .venv/bin/python 99_evaluation/analysis/fig_ch6_findings.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import shutil, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402
import story_udiads as U  # noqa: E402

OUT, THESIS = Q.OUT, Q.THESIS


def save(name, rgb):
    cv2.imwrite(str(OUT / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    shutil.copy(OUT / name, THESIS / name)
    print(name, rgb.shape[1], "x", rgb.shape[0])


def diva_pair(sub, stem, box, model, tag):
    gtx, pix, imf = Q.files(sub, stem)
    img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
    fg = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
    lw = max(2, int(round((box[2] - box[0]) / 450)))
    save(f"{tag}_gt.png", Q.panel(img, fg, O.polys(gtx), box, lw))
    save(f"{tag}_{model}.png", Q.panel(img, fg, O.polys(Q.diva_preds(sub)[model] / f"{stem}.xml"), box, lw))


def numerals():
    ms, stem = "Latin2", "230"
    f = next((U.DATA / ms / f"text-line-gt-{ms}/test").glob(f"*{stem}*.png"))
    g = U.gt(ms, f.stem)
    img = cv2.cvtColor(cv2.imread(str(next((U.DATA / ms / f"img-{ms}/test").glob(f.stem + ".*")))), cv2.COLOR_BGR2RGB)
    n, comp, st, _ = cv2.connectedComponentsWithStats(g)
    med = np.median(st[1:, 4])
    small = np.zeros(n, bool); small[1:] = st[1:, 4] < 0.2 * med
    lab = U.labels("maskrcnn", ms, f.stem).astype(np.int64)
    # best IoU of every GT component with one predicted instance (FEST matching threshold 0.75)
    npred = int(lab.max()) + 1
    c = np.bincount(comp.ravel().astype(np.int64) * npred + lab.ravel(), minlength=n * npred).reshape(n, npred).astype(float)
    inter = c[1:, 1:]; ga = c[1:].sum(1, keepdims=True); pa = c[:, 1:].sum(0, keepdims=True)
    best = np.r_[0, (inter / np.maximum(ga + pa - inter, 1)).max(1)]
    ys, xs = np.nonzero(small[comp])
    x0, x1 = 0, int(min(g.shape[1], xs.max() + 650))
    y0 = int(np.percentile(ys, 20)); y1 = min(g.shape[0], y0 + 520)
    base = (0.55 * img + 0.45 * 255).astype(np.float32)
    a = base.copy()
    a[(comp > 0) & ~small[comp]] = 0.35 * a[(comp > 0) & ~small[comp]] + 0.65 * np.array((31, 119, 180))
    a[small[comp]] = 0.2 * a[small[comp]] + 0.8 * np.array((214, 39, 40))
    b = base.copy()
    ok = (best >= 0.75)[comp] & (comp > 0)
    b[ok] = 0.35 * b[ok] + 0.65 * np.array((44, 160, 44))
    miss = (best < 0.75)[comp] & (comp > 0)
    b[miss] = 0.2 * b[miss] + 0.8 * np.array((214, 39, 40))
    save("numerals_latin2_230_gt.png", a[y0:y1, x0:x1].astype(np.uint8))
    save("numerals_latin2_230_bbox.png", b[y0:y1, x0:x1].astype(np.uint8))
    print("small components on page:", int(small.sum()), "of", n - 1, "matched:", int((best[small] >= 0.75).sum()))


if __name__ == "__main__":
    diva_pair("CB55", "e-codices_fmb-cb-0055_0108v_max", (1450, 3100, 3500, 3800), "maskrcnn", "fragments_cb55_0108v")
    diva_pair("CS863", "e-codices_csg-0863_013_max", (65, 690, 2714, 1538), "twostage", "rtmiss_cs863_013")
    numerals()
