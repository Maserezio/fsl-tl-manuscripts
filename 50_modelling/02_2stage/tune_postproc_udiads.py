"""Select second-stage postprocessing on VALIDATION, then score the test split.

Subset, detector and segmenter arm come from the environment:
    SUBSET=Latin2 DETECTOR=rtdetr_stock_random ARM=tversky python tune_postproc_udiads.py

Diagnosis that led here, all measured on test page 071 (the worst, FM 0.326):
  - the detector emits 182 boxes for 186 GT lines with zero spurious ones, so
    detection is not the problem;
  - median best-match IoU is 0.732 against the metric's hard 0.75 cut, with 66% of
    predictions stuck in [0.5, 0.75) -- everything fails by a hair;
  - the mask is both 16% too wide (precision 0.837) and 11% incomplete
    (recall 0.892), i.e. boundary error, not a systematic dilation.

Two inference-time knobs move that boundary, and both turned out to matter:
  close_fraction  0.04 -> 0.005 alone took page 071 from 0.326 to 0.538. The default
                  builds a 41px horizontal closing kernel, tuned for Latin word gaps;
                  Syriac is denser and the kernel bleeds past the GT ink.
  pad             15 -> 25 took it to 0.630. Counterintuitively MORE context helps
                  (pad=0 collapses to 0.03) -- the segmenter was trained on padded
                  crops -- but past 25 it falls off a cliff (pad=40 scores 0.000).

Those numbers came from the test page, so they are diagnostic only. This script
re-runs the same grid on validation, picks there, and only then touches test.

The metric matches evaluate_util.evaluate_metrics (a GT line counts when some
prediction reaches IoU >= 0.75; DR = M/n_gt, RA = M/n_pred, FM their harmonic mean),
with disjoint-bbox pairs skipped since those are IoU 0 by construction. It is
asserted against the official implementation on the first validation page.
"""

import itertools
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "71_misc"))

import evaluate_loss_ablation_detr_udiads as E  # noqa: E402
from evaluate_util import evaluate_metrics  # noqa: E402

SUBSET = os.environ.get("SUBSET", "Syr341")
DETECTOR = os.environ.get("DETECTOR", "rtdetr_convnext_tiny_dinov3")
ARM = os.environ.get("ARM", "bce")
CROP_W, CROP_H = 1024, 256

PADS = [15, 20, 25]
THRESHOLDS = [0.3, 0.4, 0.5]
CLOSE_FRACTIONS = [0.005, 0.01, 0.02]
BASELINE = dict(pad=15, threshold=0.5, close_fraction=0.04)

OUT_DIR = REPO / "99_evaluation/02_2stage/u-diads-tl/postproc_tuned"


def boxes_of(labels):
    out = {}
    for i in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == i)
        if len(xs):
            out[i] = (xs.min(), ys.min(), xs.max(), ys.max())
    return out


def metrics(gt_lab, pr_lab):
    gb, pb = boxes_of(gt_lab), boxes_of(pr_lab)
    if not gb or not pb:
        return dict(DR=0.0, RA=0.0, FM=0.0, median_IoU=0.0, n_pred=len(pb))
    matched, ious = 0, []
    for gid, g in gb.items():
        best = 0.0
        for pid, p in pb.items():
            if p[2] < g[0] or g[2] < p[0] or p[3] < g[1] or g[3] < p[1]:
                continue
            x0, y0 = min(p[0], g[0]), min(p[1], g[1])
            x1, y1 = max(p[2], g[2]), max(p[3], g[3])
            A = gt_lab[y0:y1 + 1, x0:x1 + 1] == gid
            B = pr_lab[y0:y1 + 1, x0:x1 + 1] == pid
            inter = np.logical_and(A, B).sum()
            if inter:
                best = max(best, inter / np.logical_or(A, B).sum())
        ious.append(best)
        matched += best >= 0.75
    dr, ra = matched / len(gb), matched / len(pb)
    fm = 0.0 if dr + ra == 0 else 2 * dr * ra / (dr + ra)
    return dict(DR=dr, RA=ra, FM=fm, median_IoU=float(np.median(ious)), n_pred=len(pb))


def probability_crops(img, boxes, model, pad):
    """One segmenter forward per box. Independent of threshold/closing, so the grid
    reuses these instead of re-running the network 27 times per page."""
    H, W = img.shape[:2]
    out = []
    with torch.no_grad():
        for b in boxes:
            x1, y1, x2, y2 = np.rint(b).astype(int)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.ascontiguousarray(
                cv2.resize(rgb, (CROP_W, CROP_H)))).permute(2, 0, 1).float()[None].to(E.DEVICE) / 255.0
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            out.append(((x1, y1, x2, y2),
                        cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)))
    return out


def assemble(crops, shape, threshold, close_fraction):
    inst = np.zeros(shape, np.uint16)
    nid = 1
    for (x1, y1, x2, y2), p in crops:
        m = E.connect_line(p >= threshold, close_fraction).astype(bool)
        if not m.any():
            continue
        region = inst[y1:y2, x1:x2]
        w = m & (region == 0)
        if not w.any():
            continue
        region[w] = nid
        nid += 1
    return inst


def load_split(root, split_names, gt_names):
    img_dir = next(d for d in (root / f"img-{SUBSET}").iterdir() if d.name in split_names)
    gt_dir = next(d for d in (root / f"text-line-gt-{SUBSET}").iterdir() if d.name in gt_names)
    return E.image_paths(img_dir), gt_dir


def main():
    root = REPO / "00_data/U-DIADS-TL" / SUBSET
    val_paths, val_gt = load_split(root, ("val", "validation"), ("validation", "val"))
    test_paths, test_gt = load_split(root, ("test", "public-test"), ("test", "public-test"))
    det_dir = (REPO / "80_models/02_2stage/u-diads-tl/detection/rtdetr_hf"
               / SUBSET / DETECTOR / "best_model")
    seg = (REPO / "80_models/02_2stage/u-diads-tl"
           / "crop_seg_loss_ablation_components_1024x256" / SUBSET / ARM / "best.pth")

    raw = E.raw_detections(det_dir, val_paths + test_paths)
    conf, _ = E.select_confidence({p.stem: raw[p.stem] for p in val_paths}, val_paths,
                                  val_gt, 0.9, np.round(np.arange(0.01, 0.92, 0.02), 2))
    print(f"confidence selected on val: {conf}", flush=True)
    model, _ = E.load_segmenter(seg, E.DEVICE)

    def run_split(paths, gt_dir, grid, label):
        """grid: list of (pad, threshold, close_fraction) -> DataFrame of per-page rows."""
        dets = E.finalized_detections(raw, paths, conf, 0.9)
        pads = sorted({g[0] for g in grid})
        rows = []
        for path in paths:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            gt = (cv2.imread(str(gt_dir / f"{path.stem}.png")).sum(-1) > 0).astype(np.uint8)
            _, gt_lab = cv2.connectedComponents(gt)
            cache = {pad: probability_crops(img, dets[path.stem][0], model, pad) for pad in pads}
            for pad, thr, cf in grid:
                inst = assemble(cache[pad], img.shape[:2], thr, cf)
                m = metrics(gt_lab, inst.astype(np.int32))
                rows.append(dict(split=label, page=path.stem, pad=pad, threshold=thr,
                                 close_fraction=cf, **m))
            print(f"  [{label}] {path.stem} done", flush=True)
        return pd.DataFrame(rows)

    grid = list(itertools.product(PADS, THRESHOLDS, CLOSE_FRACTIONS))
    grid.append((BASELINE["pad"], BASELINE["threshold"], BASELINE["close_fraction"]))

    print(f"\n=== VALIDATION sweep ({len(grid)} settings x {len(val_paths)} pages) ===", flush=True)
    val = run_split(val_paths, val_gt, grid, "val")

    # Sanity: our fast metric must equal the official one at the baseline setting.
    p0 = val_paths[0]
    img0 = cv2.imread(str(p0), cv2.IMREAD_COLOR)
    gt0 = (cv2.imread(str(val_gt / f"{p0.stem}.png")).sum(-1) > 0).astype(np.uint8)
    dets0 = E.finalized_detections(raw, [p0], conf, 0.9)[p0.stem][0]
    inst0 = assemble(probability_crops(img0, dets0, model, BASELINE["pad"]),
                     img0.shape[:2], BASELINE["threshold"], BASELINE["close_fraction"])
    off = evaluate_metrics(gt0, inst0.astype(np.int32))[4]
    ours = val[(val.page == p0.stem) & (val.pad == BASELINE["pad"])
               & (val.threshold == BASELINE["threshold"])
               & (val.close_fraction == BASELINE["close_fraction"])].FM.iloc[0]
    ok = abs(off - ours) < 1e-6
    print(f"\nmetric check on {p0.stem}: official={off:.4f} ours={ours:.4f} "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("fast metric disagrees with the official one -- aborting")

    val_agg = (val.groupby(["pad", "threshold", "close_fraction"])
               [["FM", "DR", "RA", "median_IoU"]].mean().reset_index()
               .sort_values("FM", ascending=False))
    print("\n=== validation, top settings ===")
    print(val_agg.head(6).round(4).to_string(index=False))
    best = val_agg.iloc[0]
    b = (int(best["pad"]), float(best["threshold"]), float(best["close_fraction"]))
    print(f"\nselected on val: pad={b[0]} threshold={b[1]} close_fraction={b[2]}")

    print(f"\n=== TEST with the selected setting, plus the baseline ===", flush=True)
    test_grid = [b, (BASELINE["pad"], BASELINE["threshold"], BASELINE["close_fraction"])]
    test = run_split(test_paths, test_gt, test_grid, "test")
    test_agg = (test.groupby(["pad", "threshold", "close_fraction"])
                [["FM", "DR", "RA", "median_IoU"]].mean().reset_index())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val.to_csv(OUT_DIR / f"{SUBSET}_val_sweep.csv", index=False)
    test.to_csv(OUT_DIR / f"{SUBSET}_test_per_page.csv", index=False)
    test_agg.to_csv(OUT_DIR / f"{SUBSET}_test_summary.csv", index=False)
    print("\n=== TEST RESULT ===")
    print(test_agg.round(4).to_string(index=False))
    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
