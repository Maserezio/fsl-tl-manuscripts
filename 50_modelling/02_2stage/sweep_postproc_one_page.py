"""Sweep the second-stage postprocessing on a single page, without re-running the net.

Diagnosis for Syr341/071 (FM 0.326): the detector emits 182 boxes for 186 GT lines
with zero spurious ones, and the median best-match IoU is 0.732 against a hard 0.75
threshold -- 66% of predictions sit in [0.5, 0.75). Predicted components carry ~12%
more area than GT at ~0.95x the height, i.e. they are too WIDE, which points at
connect_line's horizontal closing kernel (width = close_fraction * crop_width, 41px
at the default 0.04) rather than at the detector.

Two knobs decide that width: `segmenter_threshold` (probability -> mask) and
`close_fraction` (how aggressively word gaps get bridged). They interact: shrink the
kernel too far and a line breaks into words, at which point connect_line's
"keep the largest connected component" step throws the rest of the line away.

The segmenter forward pass does not depend on either knob, so probabilities are
computed once per box and cached; the grid then runs on CPU over those arrays.

The FM here is recomputed with the same rule as evaluate_util.evaluate_metrics
(a GT line counts when some prediction reaches IoU >= 0.75; DR = M/n_gt,
RA = M/n_pred), only with disjoint bounding boxes skipped -- they are IoU 0 by
definition. The script asserts it reproduces the official number at the default
settings before trusting any of the swept ones.

THIS SWEEPS ON A TEST PAGE. It is a diagnostic, not a tuning run: whatever it
suggests has to be re-selected on validation before it goes into a reported number.
"""

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
PAGE = os.environ.get("PAGE", "071")
DETECTOR = os.environ.get("DETECTOR", "rtdetr_convnext_tiny_dinov3")
ARM = os.environ.get("ARM", "bce")

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CLOSE_FRACTIONS = [0.005, 0.01, 0.02, 0.04, 0.06]
PAD, CROP_W, CROP_H = 15, 1024, 256
OUT = REPO / "99_evaluation/02_2stage/u-diads-tl/postproc_sweep_one_page.csv"


class Args:
    crop_w, crop_h = CROP_W, CROP_H
    pad = PAD
    segmenter_threshold = 0.5
    close_fraction = 0.04
    min_area = 50
    dedup_iou = 0.9


def boxes_of(labels):
    out = {}
    for i in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == i)
        if len(xs):
            out[i] = (xs.min(), ys.min(), xs.max(), ys.max())
    return out


def fast_metrics(gt_lab, pr_lab):
    """DR / RA / FM with the official rule, disjoint-box pairs skipped."""
    gb, pb = boxes_of(gt_lab), boxes_of(pr_lab)
    n_gt, n_pred = len(gb), len(pb)
    if not n_gt or not n_pred:
        return 0.0, 0.0, 0.0, []
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
    dr, ra = matched / n_gt, matched / n_pred
    fm = 0.0 if dr + ra == 0 else 2 * dr * ra / (dr + ra)
    return dr, ra, fm, ious


def assemble(prob_crops, shape, threshold, close_fraction):
    """Rebuild the page instance map from cached per-box probabilities."""
    inst = np.zeros(shape, np.uint16)
    next_id = 1
    for (x1, y1, x2, y2), prob in prob_crops:
        mask = E.connect_line(prob >= threshold, close_fraction).astype(bool)
        if not mask.any():
            continue
        region = inst[y1:y2, x1:x2]
        writable = mask & (region == 0)
        if not writable.any():
            continue
        region[writable] = next_id
        next_id += 1
    return inst


def main():
    root = REPO / "00_data/U-DIADS-TL" / SUBSET
    img = cv2.imread(str(next((root / f"img-{SUBSET}" / "test").glob(f"{PAGE}.*"))), cv2.IMREAD_COLOR)
    gt = (cv2.imread(str(root / f"text-line-gt-{SUBSET}" / "test" / f"{PAGE}.png")).sum(-1) > 0)
    _, gt_lab = cv2.connectedComponents(gt.astype(np.uint8))

    # Detections: same confidence the scoring run selected on validation.
    det_dir = (REPO / "80_models/02_2stage/u-diads-tl/detection/rtdetr_hf"
               / SUBSET / DETECTOR / "best_model")
    val_dir = next(d for d in (root / f"img-{SUBSET}").iterdir()
                   if d.name in ("val", "validation"))
    val_gt = next(d for d in (root / f"text-line-gt-{SUBSET}").iterdir()
                  if d.name in ("val", "validation"))
    val_paths = E.image_paths(val_dir)
    page_path = next((root / f"img-{SUBSET}" / "test").glob(f"{PAGE}.*"))
    raw = E.raw_detections(det_dir, val_paths + [page_path])
    conf, _ = E.select_confidence({p.stem: raw[p.stem] for p in val_paths}, val_paths,
                                  val_gt, Args.dedup_iou,
                                  np.round(np.arange(0.01, 0.92, 0.02), 2))
    boxes = E.finalized_detections(raw, [page_path], conf, Args.dedup_iou)[page_path.stem][0]
    print(f"conf={conf}  boxes={len(boxes)}  GT lines={gt_lab.max()}", flush=True)

    # One forward pass per box; threshold/closing are applied afterwards.
    seg = (REPO / "80_models/02_2stage/u-diads-tl"
           / "crop_seg_loss_ablation_components_1024x256" / SUBSET / ARM / "best.pth")
    model, _ = E.load_segmenter(seg, E.DEVICE)
    H, W = img.shape[:2]
    prob_crops = []
    with torch.no_grad():
        for b in boxes:
            x1, y1, x2, y2 = np.rint(b).astype(int)
            x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
            x2, y2 = min(W, x2 + PAD), min(H, y2 + PAD)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.ascontiguousarray(
                cv2.resize(rgb, (CROP_W, CROP_H)))).permute(2, 0, 1).float()[None].to(E.DEVICE) / 255.0
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            prob_crops.append(((x1, y1, x2, y2),
                               cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)))
    print(f"cached {len(prob_crops)} probability crops\n", flush=True)

    # Sanity: the defaults must reproduce the official metric on this page.
    base = assemble(prob_crops, (H, W), Args.segmenter_threshold, Args.close_fraction)
    off = evaluate_metrics(gt.astype(np.uint8), base.astype(np.int32))
    mine = fast_metrics(gt_lab, base.astype(np.int32))
    print(f"check: official FM={off[4]:.4f}  fast FM={mine[2]:.4f}  "
          f"{'OK' if abs(off[4] - mine[2]) < 1e-6 else 'MISMATCH -- sweep not trustworthy'}\n",
          flush=True)

    rows = []
    for thr in THRESHOLDS:
        for cf in CLOSE_FRACTIONS:
            inst = assemble(prob_crops, (H, W), thr, cf)
            dr, ra, fm, ious = fast_metrics(gt_lab, inst.astype(np.int32))
            rows.append(dict(threshold=thr, close_fraction=cf, DR=round(dr, 4),
                             RA=round(ra, 4), FM=round(fm, 4),
                             median_IoU=round(float(np.median(ious)), 4),
                             instances=int(inst.max())))
            print(f"  thr={thr:.1f} close={cf:.3f}  FM={fm:.4f}  "
                  f"medIoU={np.median(ious):.3f}  inst={inst.max()}", flush=True)

    df = pd.DataFrame(rows).sort_values("FM", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nbest:\n{df.head(5).to_string(index=False)}")
    print(f"\nbaseline (thr=0.5, close=0.04): "
          f"FM={df[(df.threshold == 0.5) & (df.close_fraction == 0.04)].FM.iloc[0]:.4f}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
