#!/usr/bin/env python3
"""Zero-shot CATMuS Mask R-CNN on the RQ3 collections with saved predictions (all seven RQ3 collections): for every
ground-truth line, the number of predicted instances that cover at least 10% of its ink, against the widest
gap between ink columns inside the line (in line heights) and the line length (in line heights). Quarter scale.
    .venv/bin/python 99_evaluation/analysis/zs_split_wordgaps.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402

EV = Q.ROOT / "99_evaluation/instance_segmentation/mask_rcnn/cross_collection"
PRED = {ds: f"{ds}_maskrcnn_convnext_tiny_catmus_1152_zeroshot_story_zeroshot"
        for ds in ("Pinkas", "ONB", "RASAM", "Phil_gr_130", "GRPOLY", "NorHand_v3")}
PRED["RASM"] = "RASM_maskrcnn_convnext_tiny_catmus_1152_zeroshot_story_zs"
SC = 0.25
rows = []
for ds, run in PRED.items():
    for x in sorted((EV / run / "test_pred_xml").glob("*.xml")):
        gtx, pix, _ = Q.files(ds, x.stem)
        fg = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
        H, W = fg.shape
        fg = cv2.resize(fg.astype(np.uint8), (int(W * SC), int(H * SC)), interpolation=cv2.INTER_AREA) > 0
        h, w = fg.shape
        preds = []
        for p in O.polys(x):
            m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.rint(p * SC).astype(np.int32)], 1); preds.append(m > 0)
        for g in O.polys(gtx):
            gp = np.rint(g * SC).astype(np.int32)
            m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [gp], 1); ink = (m > 0) & fg
            n = int(ink.sum())
            if n < 20:
                continue
            area = cv2.contourArea(gp.astype(np.float32)); x0, x1 = gp[:, 0].min(), gp[:, 0].max()
            lh = max(area / max(x1 - x0, 1), 1.0)
            cols = np.flatnonzero(ink.any(0))
            gaps = np.diff(cols) - 1
            frag = sum(int((ink & pm).sum()) >= 0.1 * n for pm in preds)
            rows.append({"ds": ds, "page": x.stem, "length": (cols[-1] - cols[0] + 1) / lh,
                         "max_gap": (gaps.max() if len(gaps) else 0) / lh, "frag": frag})
    print(ds, "done", flush=True)
d = pd.DataFrame(rows)
d.to_csv(Q.ROOT / "99_evaluation/analysis/zs_split_wordgaps_all7.csv", index=False)
d["split"] = d.frag >= 2
print(d.groupby("ds").agg(lines=("frag", "size"), frag=("frag", "mean"), split=("split", "mean"),
                          gap=("max_gap", "median"), length=("length", "median")).round(2))
d["gap_bin"] = pd.cut(d.max_gap, [-0.01, 0.5, 1, 1.5, 2, 99])
d["len_bin"] = pd.cut(d.length, [0, 10, 20, 30, 999])
print(d.groupby("gap_bin", observed=True).agg(lines=("frag", "size"), frag=("frag", "mean"), split=("split", "mean")).round(2))
print(d.groupby("len_bin", observed=True).agg(lines=("frag", "size"), frag=("frag", "mean"), split=("split", "mean")).round(2))
for ds, g in d.groupby("ds"):
    print(ds, "spearman frag~gap %.2f frag~length %.2f" % (g.frag.corr(g.max_gap, method="spearman"), g.frag.corr(g.length, method="spearman")))
print("all spearman frag~gap %.2f frag~length %.2f" % (d.frag.corr(d.max_gap, method="spearman"), d.frag.corr(d.length, method="spearman")))

s = d.frag >= 2
print("ALL7 gap>0.5 split %.1f%% (n=%d) | gap<=0.5 split %.1f%% (n=%d)" % (100 * s[d.max_gap > 0.5].mean(), (d.max_gap > 0.5).sum(),
      100 * s[d.max_gap <= 0.5].mean(), (d.max_gap <= 0.5).sum()))
