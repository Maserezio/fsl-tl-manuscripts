#!/usr/bin/env python3
"""U-DIADS-TL qualitative figure: line instances of the three RQ1 pipelines (ConvNeXt-Tiny) on one subset.
  U-Net        ImageNet init, cached probability maps + the thesis postprocessing (ARU-Net baseline fusion,
               seam-carve disconnection, per-subset parameters), as in eval_udiads_matrix.py
  two-stage    RT-DETR CATMuS + Tversky crop U-Net instance maps (99_evaluation/instance_segmentation/rtdetr_bbox_unet/u-diads-tl/..._matrix_m_1024x256)
  Mask R-CNN   CATMuS proposals + crop U-Net refinement instance maps (99_evaluation/instance_segmentation/mask_rcnn/u-diads-tl/...)
Per-page FEST metric (evaluate_metrics) for all three, then a colored-instance crop of a chosen page.

    .venv/bin/python 99_evaluation/analysis/story_udiads.py SUBSET            per-page FM table
    .venv/bin/python 99_evaluation/analysis/story_udiads.py SUBSET page PAGE              metrics of one page
    .venv/bin/python 99_evaluation/analysis/story_udiads.py SUBSET render PAGE x0 y0 x1 y1
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SS = REPO / "50_modelling/semantic_segmentation/unet"
sys.path.insert(0, str(SS)); sys.path.insert(0, str(REPO / "50_modelling/common"))
from postproc import PER_MS_PARAMS, fuse_masks, run_pipeline  # noqa: E402
from evaluate_util import evaluate_metrics  # noqa: E402

DATA = REPO / "00_data/U-DIADS-TL"
EV = REPO / "99_evaluation"
OUT = REPO / "99_evaluation/analysis/story_figures"
PAL = np.array([(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189), (23, 190, 207)])


def labels(method, ms, stem):
    if method == "unet":
        prob = np.load(EV / f"semantic_segmentation/unet/u-diads-tl/prob_cache/unet_tu-convnext_tiny.in12k_ft_in1k_udiads_{ms}/{stem}.npy").astype(np.float32)
        base = np.load(EV / f"semantic_segmentation/unet/u-diads-tl/arunet_baseline_cache/{ms}/{stem}.npy").astype(np.float32)
        p = PER_MS_PARAMS.get(ms, PER_MS_PARAMS["Latin14396"])
        refined = run_pipeline(fuse_masks(base, prob), {k: p[k] for k in ("line_sigma", "min_line_distance", "peak_frac", "valley_ratio")})
        return cv2.connectedComponents(refined.astype(np.uint8))[1]
    if method == "twostage":
        f = EV / f"instance_segmentation/rtdetr_bbox_unet/u-diads-tl/{ms.lower()}_matrix_m_1024x256/rtdetr_convnext_tiny_catmus/rtdetr_convnext_tiny_catmus_tversky/instance/{stem}.png"
    else:
        f = EV / f"instance_segmentation/mask_rcnn/u-diads-tl/{ms}/maskrcnn_convnext_tiny_catmus_1024_200ep_croprefine/test/instances/{stem}.png"
    im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:                       # color-coded instances -> integer labels
        flat = im.reshape(-1, im.shape[2]).astype(np.int64)
        key = flat[:, 0] * 65536 + flat[:, 1] * 256 + flat[:, 2]
        _, lab = np.unique(key, return_inverse=True)
        lab = lab.reshape(im.shape[:2])
        bg = lab[np.all(im == 0, axis=2)]
        if bg.size:
            lab[lab == bg[0]] = -1
        return lab + 1
    return im.astype(np.int64)


def gt(ms, stem):
    return (cv2.imread(str(DATA / ms / f"text-line-gt-{ms}/test/{stem}.png")).sum(axis=-1) > 0).astype(np.uint8)


def score(args):
    method, ms, stem = args
    return (method, stem) + tuple(evaluate_metrics(gt(ms, stem), labels(method, ms, stem)))


def main():
    ms = sys.argv[1]
    stems = sorted(p.stem for p in (DATA / ms / f"text-line-gt-{ms}/test").glob("*.png"))
    if len(sys.argv) > 2 and sys.argv[2] == "render":
        stem = sys.argv[3]; x0, y0, x1, y1 = map(int, sys.argv[4:8])
        img = cv2.cvtColor(cv2.imread(str(next((DATA / ms / f"img-{ms}/test").glob(stem + ".*")))), cv2.COLOR_BGR2RGB)
        g = cv2.connectedComponents(gt(ms, stem))[1]
        for name, lab in [("gt", g)] + [(m, labels(m, ms, stem)) for m in ("unet", "twostage", "maskrcnn")]:
            o = img.copy().astype(np.float32)
            ids = [i for i in np.unique(lab) if i > 0]
            # order instances top to bottom so that neighbors get different colors
            ids.sort(key=lambda i: np.where(lab == i)[0].mean())
            for k, i in enumerate(ids):
                m = lab == i
                o[m] = 0.45 * o[m] + 0.55 * PAL[k % len(PAL)]
            cv2.imwrite(str(OUT / f"udiads_{ms}_{stem}_{name}.png"), cv2.cvtColor(o[y0:y1, x0:x1].astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(name, len(ids))
        return
    if len(sys.argv) > 2 and sys.argv[2] == "page":
        stems = [sys.argv[3]]
    jobs = [(m, ms, s) for m in ("unet", "twostage", "maskrcnn") for s in stems]
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(score, jobs))
    tab = {}
    for r in res:
        tab.setdefault(r[1], {})[r[0]] = r[2:]
    names = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
    for s, d in tab.items():
        print(s, " ".join(f"{m}:{100 * d[m][4]:.1f}" for m in ("unet", "twostage", "maskrcnn")))
    for m in ("unet", "twostage", "maskrcnn"):
        print("mean", m, {n: round(100 * np.mean([tab[s][m][i] for s in tab]), 2) for i, n in enumerate(names)})


if __name__ == "__main__":
    main()
