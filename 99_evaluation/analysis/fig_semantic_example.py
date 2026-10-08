#!/usr/bin/env python3
"""Figure 3.2: semantic postprocessing on the best U-DIADS-TL test page (FEST FM).

ConvNeXt-Tiny U-Net, ImageNet init (size-axis runs = thesis rows). Uses the cached
probability maps and ARU-Net baselines; same fuse -> disconnect -> cleanup as
50_modelling/semantic_segmentation/unet/eval_udiads_matrix.py.
Writes graphics/unetpp_{a_prob,b_baseline,c_fused,d_instances}.png into the thesis repo.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "50_modelling/semantic_segmentation/unet"))
sys.path.insert(0, str(REPO / "50_modelling/common"))
from postproc import PER_MS_PARAMS, fuse_masks, run_pipeline, remove_small_objects  # noqa: E402
from evaluate_util import evaluate_metrics  # noqa: E402

PROB = REPO / "99_evaluation/semantic_segmentation/unet/u-diads-tl/prob_cache"
BASE = REPO / "99_evaluation/semantic_segmentation/unet/u-diads-tl/arunet_baseline_cache"
DATA = REPO / "00_data/U-DIADS-TL"
OUT = Path.home() / "Thesis/fsl-tl-manuscripts-thesis/graphics"


def pipeline(ms, stem):
    prob = np.load(PROB / f"unet_tu-convnext_tiny_sz_udiads_{ms}" / f"{stem}.npy").astype(np.float32)
    base = np.load(BASE / ms / f"{stem}.npy").astype(np.float32)
    p = PER_MS_PARAMS[ms]
    disc = {k: p[k] for k in ("line_sigma", "min_line_distance", "peak_frac", "valley_ratio")}
    fused = fuse_masks(base, prob)
    refined = run_pipeline(fused, disc)
    _, labels = cv2.connectedComponents(refined.astype(np.uint8))
    return prob, base, fused, labels


def score(job):
    ms, g = job
    gt = (cv2.imread(str(g)).sum(axis=-1) > 0).astype(np.uint8)
    *_, labels = pipeline(ms, g.stem)
    return (evaluate_metrics(gt, labels)[-1], ms, g.stem)


def main():
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(ms, g) for ms in ("Latin14396", "Latin2", "Syr341")
            for g in sorted((DATA / ms / f"text-line-gt-{ms}" / "test").glob("*.png"))]
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(score, jobs))
    rows.sort(reverse=True)
    for r in rows[:5]:
        print("FM %.4f %s %s" % r)
    fm, ms, stem = rows[0]
    prob, base, fused, labels = pipeline(ms, stem)
    img = cv2.imread(str(next((DATA / ms / f"img-{ms}" / "test").glob(stem + ".*"))))
    # crop: upper part of the text block, about ten lines, without rescaling
    ys, xs = np.nonzero(labels)
    y0, x1 = int(ys.min()), int(xs.max())
    x0 = int(xs.min())
    h = min(int(0.32 * (ys.max() - ys.min())), 900)
    sl = (slice(max(0, y0 - 20), y0 + h), slice(max(0, x0 - 20), min(img.shape[1], x1 + 20)))
    baseline_bin = cv2.threshold((base * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thin = cv2.erode(baseline_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))) > 0
    rng = np.random.default_rng(3)
    pal = rng.integers(40, 230, size=(labels.max() + 1, 3)).astype(np.uint8); pal[0] = 255
    panels = {
        "a_prob": cv2.applyColorMap(255 - (np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_BONE),
        "b_baseline": np.where(thin[..., None], np.array([0, 0, 200], np.uint8), (img * 0.45 + 140).astype(np.uint8)),
        "c_fused": np.where(fused[..., None] > 0, np.uint8(30), np.uint8(255)).repeat(3, axis=2),
        "d_instances": pal[labels],
    }
    for k, v in panels.items():
        cv2.imwrite(str(OUT / f"unetpp_{k}.png"), v[sl])
    print("chosen", ms, stem, "FM %.4f" % fm, "crop", sl)


if __name__ == "__main__":
    main()
