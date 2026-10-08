#!/usr/bin/env python3
"""Figure 3.1 v2: semantic postprocessing on a high-FM U-DIADS-TL test page where the postprocessing
visibly works (removed small components and projection cuts), with the removed pixels shown in red.

Per page: FEST FM, number of removed small components (thresholded U-Net mask and fused mask filters),
number of cuts (components after disconnect minus before). Picks the page with FM >= 0.95 and the most
removed components + cuts, then the 560x300 window with the most removed pixels and cut lines.
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fig_semantic_example as F  # noqa: E402
from postproc import remove_small_objects, PER_MS_PARAMS, run_pipeline  # noqa: E402
from evaluate_util import evaluate_metrics  # noqa: E402

WIN = (300, 560)


def stages(ms, stem):
    prob = np.load(F.PROB / f"unet_tu-convnext_tiny_sz_udiads_{ms}" / f"{stem}.npy").astype(np.float32)
    base = np.load(F.BASE / ms / f"{stem}.npy").astype(np.float32)
    bb = cv2.threshold((base * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    raw = (prob > 0.5).astype(np.uint8)
    pred = remove_small_objects(raw, min_size=50)
    thin = (cv2.erode(bb, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))) > 0).astype(np.uint8)
    union = np.maximum(pred, thin)
    fused = remove_small_objects(union, min_size=500)
    removed = ((raw > 0) & (pred == 0)) | ((union > 0) & (fused == 0))
    p = PER_MS_PARAMS[ms]
    refined = run_pipeline(fused, {k: p[k] for k in ("line_sigma", "min_line_distance", "peak_frac", "valley_ratio")})
    _, labels = cv2.connectedComponents(refined.astype(np.uint8))
    cut = (fused > 0) & (refined == 0) & ~removed
    return prob, thin, fused, removed, cut, labels


def score(job):
    ms, g = job
    gt = (cv2.imread(str(g)).sum(axis=-1) > 0).astype(np.uint8)
    prob, thin, fused, removed, cut, labels = stages(ms, g.stem)
    nrem = cv2.connectedComponents(removed.astype(np.uint8))[0] - 1
    ncut = labels.max() - (cv2.connectedComponents(fused.astype(np.uint8))[0] - 1)
    return (evaluate_metrics(gt, labels)[-1], nrem, ncut, ms, g.stem)


def main():
    jobs = [(ms, g) for ms in ("Latin14396", "Latin2", "Syr341")
            for g in sorted((F.DATA / ms / f"text-line-gt-{ms}" / "test").glob("*.png"))]
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(score, jobs))
    good = sorted([r for r in rows if r[0] >= 0.95], key=lambda r: -(r[1] + 3 * max(r[2], 0)))
    for r in good[:6]:
        print("FM %.4f removed %d cuts %d %s %s" % r)
    fm, nrem, ncut, ms, stem = good[0]
    prob, thin, fused, removed, cut, labels = stages(ms, stem)
    img = cv2.imread(str(next((F.DATA / ms / f"img-{ms}" / "test").glob(stem + ".*"))))
    # window with most removed pixels + dilated cut pixels, inside the text block
    w = removed.astype(np.float32) + 5 * cv2.dilate(cut.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    w *= (cv2.blur((labels > 0).astype(np.float32), (101, 101)) > 0.05)
    ii = cv2.integral(w)
    H, W = w.shape
    best, pos = -1, (0, 0)
    for y in range(0, H - WIN[0], 20):
        for x in range(0, W - WIN[1], 20):
            v = ii[y + WIN[0], x + WIN[1]] - ii[y, x + WIN[1]] - ii[y + WIN[0], x] + ii[y, x]
            if v > best:
                best, pos = v, (y, x)
    sl = (slice(pos[0], pos[0] + WIN[0]), slice(pos[1], pos[1] + WIN[1]))
    rng = np.random.default_rng(3)
    pal = rng.integers(40, 230, size=(labels.max() + 1, 3)).astype(np.uint8); pal[0] = 255
    c = np.where(fused[..., None] > 0, np.uint8(30), np.uint8(255)).repeat(3, axis=2)
    c[removed] = (40, 40, 230)                     # BGR red: removed small components
    c[cut & (fused > 0)] = (230, 120, 0)           # BGR blue: projection cuts
    panels = {
        "a_prob": cv2.applyColorMap(255 - (np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_BONE),
        "b_baseline": np.where(thin[..., None] > 0, np.array([0, 0, 200], np.uint8), (img * 0.45 + 140).astype(np.uint8)),
        "c_fused": c,
        "d_instances": pal[labels],
    }
    for k, v in panels.items():
        cv2.imwrite(str(F.OUT / f"unetpp_{k}.png"), v[sl])
    inwin = lambda m: int(cv2.connectedComponents(m[sl].astype(np.uint8))[0] - 1)
    print("chosen", ms, stem, "FM %.4f" % fm, "crop", sl, "removed comps in crop", inwin(removed), "cut comps", inwin(cut))


if __name__ == "__main__":
    main()
