#!/usr/bin/env python3
"""Line-level error taxonomy on the DIVA-HisDB test pages (thesis models), the counterpart of rq3_error_taxonomy.py.

Per GT line (Task-2 polygon, main-text ink = pixel-GT bit 0x08) and pipeline: merged / split / missed with the
same rule as rq3_merge_split (coverage c >= TAU = 0.25 of the line's ink), its width, height h, pitch ratio r,
and whether it is the first or last line of its region. Per predicted line: spurious when < TAU of its main-text
ink lies in a GT line; a spurious line on comment ink (bit 0x02) is counted as a gloss detection.
Predicted polygons are rasterised in reading order; a pixel covered twice keeps the first line.

    nice -n 19 .venv/bin/python 99_evaluation/analysis/diva_error_taxonomy.py          # compute + report
    nice -n 19 .venv/bin/python 99_evaluation/analysis/diva_error_taxonomy.py report   # report from the CSVs
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

TAU = 0.25
OUT = Q.ROOT / "99_evaluation/analysis/diva_taxonomy"
NAMES = {"unet": "U-Net", "twostage": "RT-DETR + BBox U-Net", "maskrcnn": "Mask R-CNN", "seam": "Mask R-CNN + DP seam",
         "centerline": "Center-line"}


def canvas_of(polys, shape):
    c = np.zeros(shape, np.uint16)
    for k, p in enumerate(sorted(polys, key=lambda p: p[:, 1].mean())):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [p], 1)
        c[(m > 0) & (c == 0)] = k + 1
    return c


def geometry(gt, shape):
    """width, height h, pitch ratio r, first/last line of the page column (by vertical order among overlapping lines)."""
    geo = []
    for p in gt:
        x0, y0, x1, y1 = O.bbox(p)
        m = O.rast(p, (x0, y0, x1, y1))
        cols = np.nonzero(m.any(0))[0][::4]
        top = np.array([np.argmax(m[:, c]) for c in cols]) + y0
        bot = np.array([m.shape[0] - 1 - np.argmax(m[::-1, c]) for c in cols]) + y0
        geo.append((x1 - x0, float(np.median(bot - top + 1)), float(np.median((top + bot) / 2)), x0, x1))
    out = []
    for i, (w, h, cy, a1, a2) in enumerate(geo):
        nb = [o for j, o in enumerate(geo) if j != i and min(a2, o[4]) - max(a1, o[3]) > 0.3 * min(a2 - a1, o[4] - o[3])]
        d = [abs(o[2] - cy) for o in nb if abs(o[2] - cy) > 0.25 * h]
        first = not any(o[2] < cy - 0.25 * h for o in nb)
        last = not any(o[2] > cy + 0.25 * h for o in nb)
        out.append({"width": w, "h": h, "r": min(d) / h if d else np.nan, "edge": first or last})
    return out


def page(sub, stem, preds):
    gtx, pix, _ = Q.files(sub, stem)
    pg = cv2.imread(str(pix))
    ink, comment = (pg[:, :, 0] & 0x08) > 0, (pg[:, :, 0] & 0x02) > 0
    shape = ink.shape
    gt = O.polys(gtx)
    gtc = np.zeros(shape, np.uint8)
    cv2.fillPoly(gtc, gt, 1)
    gtc = gtc > 0
    geo = geometry(gt, shape)
    lines, predrows = [], []
    for key, d in preds.items():
        pp = O.polys(d / f"{stem}.xml")
        can = canvas_of(pp, shape)
        n = int(can.max())
        cov = np.zeros((n + 1, len(gt)))
        for j, p in enumerate(gt):
            box = O.bbox(p)
            x0, y0, x1, y1 = box
            sel = O.rast(p, box) & ink[y0:y1, x0:x1]
            if sel.sum():
                cov[:, j] = np.bincount(can[y0:y1, x0:x1][sel].astype(np.int64), minlength=n + 1)[:n + 1] / sel.sum()
        hit = cov[1:] >= TAU
        lpp = hit.sum(1)
        for j in range(len(gt)):
            lines.append({"sub": sub, "page": stem, "model": NAMES[key], "merged": bool((hit[:, j] & (lpp >= 2)).any()),
                          "split": bool(hit[:, j].sum() >= 2), "missed": bool(hit[:, j].sum() == 0), **geo[j]})
        lab_ink = np.bincount(can[ink].astype(np.int64), minlength=n + 1)
        lab_ink_gt = np.bincount(can[ink & gtc].astype(np.int64), minlength=n + 1)
        lab_com = np.bincount(can[comment].astype(np.int64), minlength=n + 1)
        for i in range(1, n + 1):
            tot = lab_ink[i] + lab_com[i]
            if tot < 20:
                continue
            spur = lab_ink_gt[i] < TAU * max(lab_ink[i], 1)
            predrows.append({"sub": sub, "page": stem, "model": NAMES[key], "spurious": bool(spur),
                             "gloss": bool(spur and lab_com[i] > lab_ink[i])})
    return lines, predrows


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        return report(pd.read_csv(OUT / "lines.csv"), pd.read_csv(OUT / "preds.csv"))
    lines, preds = [], []
    for sub in ("CB55", "CS18", "CS863"):
        pr = Q.diva_preds(sub)
        for f in sorted(pr["maskrcnn"].glob("*.xml")):
            l, p = page(sub, f.stem, pr)
            lines += l; preds += p
            print("done", sub, f.stem, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    d, p = pd.DataFrame(lines), pd.DataFrame(preds)
    d.to_csv(OUT / "lines.csv", index=False); p.to_csv(OUT / "preds.csv", index=False)
    report(d, p)


def report(d, p):
    pd.set_option("display.width", 220)
    k = ["merged", "split", "missed"]
    d["any"] = d[k].any(axis=1)
    print("GT lines per subset:", d[d.model == "U-Net"].groupby("sub").size().to_dict())
    print("\nGT-line error rates (%), all test pages:")
    print((d.groupby("model")[k + ["any"]].mean() * 100).round(1).to_string())
    print("\nper subset (% any error):")
    print((d.groupby(["sub", "model"])["any"].mean() * 100).round(1).unstack().to_string())
    s = p.groupby("model")[["spurious", "gloss"]].agg(["sum", "size"])
    print("\nspurious predicted lines (count, of which on comment ink):")
    print(pd.DataFrame({"pred": s[("spurious", "size")], "spurious": s[("spurious", "sum")], "gloss": s[("gloss", "sum")]}).to_string())
    d["wq"] = pd.qcut(d.width, 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
    print("\nany-error rate (%) by line-width quartile:", d.groupby("wq", observed=True).width.agg(["min", "max"]).to_dict("index"))
    print((d.groupby(["wq", "model"], observed=True)["any"].mean() * 100).round(1).unstack().to_string())
    print("\nany-error rate (%) first/last line of column vs inner:")
    print((d.groupby(["edge", "model"])["any"].mean() * 100).round(1).unstack().to_string())
    rr = d.dropna(subset=["r"]).copy()
    rr["r_bin"] = pd.cut(rr.r, [0, 1.5, 2.0, 3.0, 99], labels=["<1.5", "1.5-2", "2-3", ">3"])
    print("\nmerge rate (%) by pitch ratio r:")
    t = rr.groupby(["r_bin", "model"], observed=True).merged.agg(["mean", "size"])
    print((t["mean"] * 100).round(1).unstack().join(t["size"].unstack().iloc[:, 0].rename("lines")).to_string())
    print("\nerror concentration: share of all erroneous lines on the worst page per model")
    e = d[d["any"]].groupby(["model", "page"]).size()
    print((e.groupby("model").max() / e.groupby("model").sum() * 100).round(1).to_string(),
          "\n", e.groupby("model").idxmax().to_string())


if __name__ == "__main__":
    main()
