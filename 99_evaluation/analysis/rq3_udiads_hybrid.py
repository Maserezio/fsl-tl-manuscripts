#!/usr/bin/env python3
"""U-DIADS-TL (FEST metric): line separation of the RQ1 U-Net foreground by center lines vs seam carving.

Same U-Net probability maps (ConvNeXt-Tiny, ImageNet, 3 training pages; prob_cache of
eval_udiads_matrix.py), two ways to split the foreground into lines:
  rq1     : the RQ1 pipeline (ARU-Net baseline fusion, seam-carve disconnection, cleanup, components)
  center  : U-Net foreground (prob >= 0.5) cut by the cells of the CATMuS center-line model
            (zero-shot, rq3_centre_lf.page_instances row:1.0); connected components within each cell.
Test pages of Latin2, Latin14396, Syr341; paired per page.

    .venv/bin/python 99_evaluation/analysis/rq3_udiads_hybrid.py
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "50_modelling").is_dir())  # repo root
SEG = ROOT / "50_modelling/semantic_segmentation/unet"
for p in (HERE, ROOT / "50_modelling/common", SEG):
    sys.path.insert(0, str(p))
DATA = ROOT / "00_data/U-DIADS-TL"
PROB = ROOT / "99_evaluation/semantic_segmentation/unet/u-diads-tl/prob_cache"
BASE = ROOT / "99_evaluation/semantic_segmentation/unet/u-diads-tl/arunet_baseline_cache"
OUT = ROOT / "99_evaluation/semantic_segmentation/center_line/udiads_hybrid"
SUBSETS = ["Latin2", "Latin14396", "Syr341"]
RUN = "unet_tu-convnext_tiny_sz_udiads_{}"


def cells_for_tests():
    import torch
    from PIL import Image
    import rq3_linefield as LF
    import rq3_centre_lf as CL
    net = LF.build_model()
    net.load_state_dict(torch.load(LF.MODELS / "catmus_pretrain.pt", map_location="cpu", weights_only=False)["model"])
    net = net.to(LF.DEV).eval()
    with torch.inference_mode():
        for sub in SUBSETS:
            (OUT / "cells" / sub).mkdir(parents=True, exist_ok=True)
            for f in sorted((DATA / sub / f"img-{sub}" / "test").glob("*.jpg")):
                dst = OUT / "cells" / sub / f"{f.stem}.png"
                if dst.exists():
                    continue
                img = np.asarray(Image.open(f).convert("RGB"))
                H, W = img.shape[:2]
                s1 = 1152 / max(H, W)
                cen, _, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s1), int(H * s1)), interpolation=cv2.INTER_AREA))
                sel = cen > 0.5
                thick = float(np.median(up[sel] + dn[sel])) if sel.sum() > 50 else LF.T0
                s2 = min(s1 * LF.T0 / max(thick, 1.0), 3200 / max(H, W))
                cen, end, up, dn = LF.run_net(net, cv2.resize(img, (int(W * s2), int(H * s2)),
                                                              interpolation=cv2.INTER_AREA if s2 < 1 else cv2.INTER_LINEAR))
                frags = [{"xs": fr["xs"] / s2, "cy": fr["cy"] / s2} for fr in LF._fragments(cen, end, 0.4)]
                page = {"coll": sub, "im": {"height": H, "width": W, "file_name": f.name}, "frags": frags, "T0": LF.T0 / s2}
                cv2.imwrite(str(dst), CL.page_instances(page, "row", 1.0).astype(np.uint16))
            print("cells", sub, flush=True)


def score(args):
    from evaluate_util import evaluate_metrics
    from postproc import PER_MS_PARAMS, fuse_masks, run_pipeline
    sub, stem = args
    prob = np.load(PROB / RUN.format(sub) / f"{stem}.npy").astype(np.float32)
    gt = (cv2.imread(str(DATA / sub / f"text-line-gt-{sub}" / "test" / f"{stem}.png")).sum(axis=-1) > 0).astype(np.uint8)
    params = PER_MS_PARAMS.get(sub, PER_MS_PARAMS["Latin14396"])
    disc = {k: params[k] for k in ("line_sigma", "min_line_distance", "peak_frac", "valley_ratio")}
    whole = os.environ.get("HYBRID_MODE") == "whole"     # one label per cell; RQ1 pipeline skipped
    if not whole:
        base = np.load(BASE / sub / f"{stem}.npy").astype(np.float32)
        refined = run_pipeline(fuse_masks(base, prob), disc)
        _, rq1 = cv2.connectedComponents(refined.astype(np.uint8))
    cells = cv2.imread(str(OUT / "cells" / sub / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
    if cells.shape != prob.shape:
        cells = cv2.resize(cells.astype(np.float32), prob.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(np.int32)
    fg = prob >= 0.5
    center, nxt = np.zeros(prob.shape, np.int32), 1
    for lab in np.unique(cells[fg]):
        if lab == 0:
            continue
        if whole:
            m = (cells == lab) & fg
            if m.sum() >= 50:
                center[m] = nxt; nxt += 1
            continue
        n, cc = cv2.connectedComponents(((cells == lab) & fg).astype(np.uint8))
        for c in range(1, n):
            m = cc == c
            if m.sum() >= 50:                      # same order of magnitude as the RQ1 small-object cleanup
                center[m] = nxt; nxt += 1
    c = tuple(evaluate_metrics(gt, center))
    return sub, stem, (c if whole else tuple(evaluate_metrics(gt, rq1))), c


def main():
    cells_for_tests()
    jobs = [(s, f.stem) for s in SUBSETS for f in sorted((DATA / s / f"img-{s}" / "test").glob("*.jpg"))]
    with ProcessPoolExecutor(int(os.environ.get("EVAL_WORKERS", "4"))) as ex:
        res = list(ex.map(score, jobs))
    keys = ["PixelIU", "LineIU", "DR", "RA", "FM"]
    rows = [{"subset": s, "page": p, **{f"rq1_{k}": a[i] for i, k in enumerate(keys)},
             **{f"center_{k}": b[i] for i, k in enumerate(keys)}} for s, p, a, b in res]
    (OUT / ("pages_whole.json" if os.environ.get("HYBRID_MODE") == "whole" else "pages.json")).write_text(json.dumps(rows, indent=1))
    import pandas as pd
    d = pd.DataFrame(rows)
    t = d.groupby("subset")[[f"{m}_{k}" for m in ("rq1", "center") for k in keys]].mean().mul(100).round(2)
    t.loc["Mean"] = t.mean().round(2)
    print(t.T.to_string())
    print("pages where center FM > rq1 FM: %d / %d" % ((d.center_FM > d.rq1_FM).sum(), len(d)))


if __name__ == "__main__":
    main()
