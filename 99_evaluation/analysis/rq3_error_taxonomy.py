#!/usr/bin/env python3
"""Line-level error taxonomy: Mask R-CNN (ab) vs the center-line model (CATMuS + WiSE-FT), dev pages.

Pages: the dev pages of Phil. gr. 130, Pinkas, and NorHand v3, predicted by the Protocol-A models of
the five bench sources (rq3_bench.py, BENCH_MODEL=base, and rq3_centre_lf.py caches *_A_w50.pkl).
Both instance maps are compared with the GT polygons on ink pixels at the same canvas
(rq3_merge_split.rates, TAU = 0.25). Per GT line: merged / split / missed, its height h (median
polygon thickness) and its pitch ratio r = (vertical distance to the nearest horizontally overlapping
neighbour) / h. Per predicted line: spurious when < TAU of its ink lies in any GT line.
Logistic regression merged ~ log r + C(target) per model.

    nice -n 19 .venv/bin/python 99_evaluation/analysis/rq3_error_taxonomy.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, pickle, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq3_bench as B  # noqa: E402
import rq3_centre_lf as CLF  # noqa: E402
import rq3_merge_split as MS  # noqa: E402
from rq3_linefield import line_geometry, page_polys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "99_evaluation/semantic_segmentation/center_line/taxonomy_v3"
TAU = MS.TAU


def line_stats(masks):
    """Per GT mask: thickness h and pitch ratio r (nan without a horizontally overlapping neighbour)."""
    geo = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            geo.append(None); continue
        cols = np.unique(xs)
        top = np.array([ys[xs == c].min() for c in cols[::4]]); bot = np.array([ys[xs == c].max() for c in cols[::4]])
        geo.append((float(np.median(bot - top + 1)), float(np.median((top + bot) / 2)), cols.min(), cols.max()))
    out = []
    for i, g in enumerate(geo):
        if g is None:
            out.append((np.nan, np.nan)); continue
        h, cy, a1, a2 = g
        d = [abs(o[1] - cy) for j, o in enumerate(geo) if j != i and o is not None
             and min(a2, o[3]) - max(a1, o[2]) > 0.3 * min(a2 - a1, o[3] - o[2]) and abs(o[1] - cy) > 0.25 * h]
        out.append((h, min(d) / h if d else np.nan))
    return out


def per_line(canvas, masks, ink):
    n_pred = int(canvas.max())
    cov = np.zeros((n_pred + 1, len(masks)))
    for j, g in enumerate(masks):
        px = g & ink
        if px.sum():
            cov[:, j] = np.bincount(canvas[px].astype(np.int64), minlength=n_pred + 1)[:n_pred + 1] / px.sum()
    hit = cov[1:] >= TAU
    lpp = hit.sum(1)
    merged = [(hit[:, j] & (lpp >= 2)).any() for j in range(len(masks))]
    split = list(hit.sum(0) >= 2)
    missed = list(hit.sum(0) == 0)
    anygt = np.zeros(canvas.shape, bool)
    for g in masks:
        anygt |= g
    spurious, n_real = 0, 0
    for i in range(1, n_pred + 1):
        pi = (canvas == i) & ink
        if pi.sum() < 20:
            continue
        n_real += 1
        if (pi & anygt).sum() / pi.sum() < TAU:
            spurious += 1
    return merged, split, missed, spurious, n_real


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return report(pd.read_csv(OUT / "lines.csv"), pd.read_csv(OUT / "preds.csv"))
    lines, preds = [], []
    for f in sorted((B.OUT / "cache").glob("*_base.pkl")):
        src = f.name[: -len("_base.pkl")]
        lf = {(p["coll"], p["im"]["file_name"]): p for p in pickle.load(open(CLF.OUT / "cache" / f"{src}_A_w50.pkl", "rb"))}
        for page in pickle.load(open(f, "rb")):
            meta = page["meta"]
            masks = MS.gt_masks(page["coll"], meta)
            ink = MS.ink_mask(page["gray"])
            stats = line_stats(masks)
            dens = MS.density(masks)
            lpage = lf[(page["coll"], Path(meta["path"]).name)]
            inst_lf = cv2.resize(CLF.page_instances(lpage, "row", 1.0), (meta["new_w"], meta["new_h"]),
                                 interpolation=cv2.INTER_NEAREST)
            # decoder v3 (graph + whole strokes + traced ends): overlapping masks -> instance map, first come
            v3masks, sc = CLF.page_instances(lpage, "graph", 1.0, stroke="safe", ext="trace")
            inst_v3 = np.zeros(v3masks[0].shape if v3masks else (1, 1), np.uint16)
            for k, mk in enumerate(v3masks):
                inst_v3[mk & (inst_v3 == 0)] = k + 1
            inst_v3 = cv2.resize(inst_v3, (meta["new_w"], meta["new_h"]), interpolation=cv2.INTER_NEAREST)
            for model, canvas in (("Mask R-CNN", B.canvas_score(page, page["mt"])), ("Center-line", inst_lf),
                                  ("Center-line v3", inst_v3)):
                mg, sp, ms, spur, n = per_line(canvas, masks, ink)
                for j in range(len(masks)):
                    lines.append({"model": model, "source": src, "target": page["coll"], "page": Path(meta["path"]).stem,
                                  "density": dens, "h": stats[j][0], "r": stats[j][1],
                                  "merged": mg[j], "split": sp[j], "missed": ms[j]})
                preds.append({"model": model, "source": src, "target": page["coll"], "spurious": spur, "pred": n})
        print("done", src, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(lines); p = pd.DataFrame(preds)
    d.to_csv(OUT / "lines.csv", index=False); p.to_csv(OUT / "preds.csv", index=False)
    report(d, p)


def report(d, p):
    pd.set_option("display.width", 200)
    k = ["merged", "split", "missed"]
    print("\nGT-line error rates per target (%):")
    print((d.groupby(["target", "model"])[k].mean() * 100).round(1).unstack().to_string())
    sp = p.groupby(["target", "model"])[["spurious", "pred"]].sum()
    print("\nspurious predictions (% of predicted lines):")
    print((sp.spurious / sp.pred * 100).round(1).unstack().to_string())
    print("\nall pages (%):", (d.groupby("model")[k].mean() * 100).round(1).to_dict("index"))
    rr = d.dropna(subset=["r"]).copy()
    rr["r_bin"] = pd.cut(rr.r, [0, 1.25, 1.5, 2.0, 3.0, 99], labels=["<1.25", "1.25-1.5", "1.5-2", "2-3", ">3"])
    print("\nmerge rate (%) by pitch ratio r = neighbour distance / line height:")
    t = rr.groupby(["r_bin", "model"], observed=True).merged.agg(["mean", "size"])
    print((t["mean"] * 100).round(1).unstack().join(t["size"].unstack().iloc[:, 0].rename("lines")).to_string())
    import statsmodels.formula.api as smf
    rr["logr"] = np.log(rr.r); rr["m"] = rr.merged.astype(int)
    for model, g in rr.groupby("model"):
        fit = smf.logit("m ~ logr + C(target)", data=g).fit(disp=0)
        lo, hi = fit.conf_int().loc["logr"]
        print(f"{model}: logit(merge) slope on log r = {fit.params['logr']:.2f} [{lo:.2f}, {hi:.2f}], n = {len(g)}")


if __name__ == "__main__":
    main()
