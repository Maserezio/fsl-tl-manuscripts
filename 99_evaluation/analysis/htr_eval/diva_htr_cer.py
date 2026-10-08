#!/usr/bin/env python3
"""Segmentation-induced HTR error on DIVA-HisDB test pages (no transcriptions exist for DIVA-HisDB).

Reference: TrOCR (medieval-data/trocr-medieval-base) on the ground-truth line polygons.
Hypothesis: the same recogniser on the predicted line polygons of each pipeline.
Page CER in reading order: the line texts are sorted by the vertical center of their polygon (then by x) and
joined with spaces, for the GT polygons and for each pipeline; merges, splits, cuts, missed lines, and spurious
lines (e.g. glosses) all show up as edits. Control "gt_dil": the GT polygons dilated by 8 px, which measures how
much the recogniser output changes under a small change of the polygon alone.
Speed-up: a predicted line whose main-text ink equals that of one GT line (Jaccard >= 0.98) reuses the GT text.

    nice -n 5 ../../.venv/bin/python diva_htr_cer.py   (cwd 99_evaluation/analysis/htr_eval) -> diva_htr_lines.csv, diva_htr_pages.csv
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import fig_qualitative_v2 as Q  # noqa: E402
import overlap_lines_diva as O  # noqa: E402
import htr_common as H  # noqa: E402

PAGES = {"CB55": ["0108v", "0160v", "0162v", "0105r"], "CS18": ["096", "164", "098", "061"],
         "CS863": ["013", "025", "017", "050"]}
ALL = __import__("os").environ.get("HTR_ALL") == "1"
SUFFIX = "_all30" if ALL else ""
MODELS = ["gt_dil", "unet", "twostage", "maskrcnn", "seam", "centerline"]
TAU, SAME = 0.25, 1.01   # SAME > 1: every predicted line is recognised (no reuse)


def ink_sets(polys, ink):
    out = []
    for p in polys:
        x0, y0, x1, y1 = O.bbox(p)
        x0, y0 = max(x0, 0), max(y0, 0)
        m = O.rast(p, (x0, y0, x1, y1))[: ink.shape[0] - y0, : ink.shape[1] - x0] & ink[y0:y1, x0:x1]
        ys, xs = np.nonzero(m)
        out.append(set(((ys + y0) * 100000 + xs + x0).tolist()))
    return out


def main():
    torch.set_num_threads(12)
    proc, model = H.load("medieval-data/trocr-medieval-base")
    lines, pages = [], []
    for sub, ids in PAGES.items():
        pr = Q.diva_preds(sub)
        # thesis center-line model: CATMuS pretraining, 20 pages, bandseamstroke decoder
        pr["centerline"] = Q.ROOT / f"99_evaluation/semantic_segmentation/center_line/diva_cbad/pred_cm_cellseam/test/{sub}_full/bandseamstroke_inside"
        if ALL:
            ids = sorted(f.stem for f in pr["maskrcnn"].glob("*.xml"))
        for pid in ids:
            stem = next(f.stem for f in pr["maskrcnn"].glob("*.xml") if pid in f.stem)
            gtx, pix, imf = Q.files(sub, stem)
            img = cv2.cvtColor(cv2.imread(str(imf)), cv2.COLOR_BGR2RGB)
            ink = (cv2.imread(str(pix))[:, :, 0] & 0x08) > 0
            gt = sorted(O.polys(gtx), key=lambda p: p[:, 1].mean())
            gink = ink_sets(gt, ink)
            ref = H.recognise(proc, model, [H.line_image(img, p) for p in gt])
            for j, p in enumerate(gt):
                lines.append({"sub": sub, "page": pid, "model": "gt", "line": j, "y": float(p[:, 1].mean()), "text": ref[j]})
            for key in MODELS:
                if key == "gt_dil":
                    pp = [cv2.findContours(cv2.dilate(cv2.fillPoly(np.zeros(ink.shape, np.uint8), [p], 1), np.ones((17, 17), np.uint8)),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0][:, 0] for p in gt]
                else:
                    pp = sorted(O.polys(pr[key] / f"{stem}.xml"), key=lambda p: p[:, 1].mean())
                pink = ink_sets(pp, ink)
                # reuse GT text when the predicted line holds exactly one GT line's ink
                hyp, todo = [None] * len(pp), []
                for i, s in enumerate(pink if key != "gt_dil" else []):
                    for j, g in enumerate(gink):
                        if s and g and len(s & g) / len(s | g) >= SAME:
                            hyp[i] = ref[j]; break
                    else:
                        todo.append(i)
                if key == "gt_dil":
                    todo = list(range(len(pp)))
                rec = H.recognise(proc, model, [H.line_image(img, pp[i]) for i in todo]) if todo else []
                for i, t in zip(todo, rec):
                    hyp[i] = t
                order = lambda ps: sorted(range(len(ps)), key=lambda i: (round(ps[i][:, 1].mean() / 40), ps[i][:, 0].mean()))
                ref_page = " ".join(ref[j] for j in order(gt)); hyp_page = " ".join(hyp[i] for i in order(pp))
                for i in range(len(pp)):
                    lines.append({"sub": sub, "page": pid, "model": key, "line": i, "y": float(pp[i][:, 1].mean()), "text": hyp[i]})
                pages.append({"sub": sub, "page": pid, "model": key, "cer": 100 * H.cer(ref_page, hyp_page),
                              "ref_chars": len(ref_page), "n_pred": len(pp), "n_gt": len(gt), "recognised": len(todo)})
                print(sub, pid, key, "CER %.2f" % pages[-1]["cer"], "recognised", len(todo), flush=True)
            pd.DataFrame(lines).to_csv(HERE / f"diva_htr_lines{SUFFIX}.csv", index=False)
            pd.DataFrame(pages).to_csv(HERE / f"diva_htr_pages{SUFFIX}.csv", index=False)
    d = pd.DataFrame(pages)
    print(d.pivot_table(index="model", columns="sub", values="cer").round(2))


if __name__ == "__main__":
    main()
