"""Render GT-vs-prediction overlays for the worst-scoring pages of a two-stage run.

Three panels per page:
  1. original page
  2. GT instances, one colour each
  3. prediction, colour-coded by what the metric did with it:
       green  = predicted line matched a GT line at IoU >= 0.75 (counts toward DR/RA)
       orange = predicted line overlaps some GT line but below 0.75 (found, not counted)
       blue   = predicted line overlaps no GT line at all (spurious)
       red    = GT line with no prediction above 0.75 (missed), drawn as outline

The colour split is the point: Syr341 keeps Pixel_IU ~0.77 and Line_IU ~0.93 while FM
collapses, which means lines are being found but not matched. Orange vs blue vs red
says which of those three failure modes actually dominates on a given page.

Matching uses the same IoU >= 0.75 rule as evaluate_util.evaluate_metrics, but pairs
are prefiltered by bounding-box overlap first -- two components whose boxes are
disjoint have IoU 0 by definition, so skipping them changes nothing and turns a
187x172 full-page scan into a few hundred small ones.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

SUBSET = os.environ.get("SUBSET", "Syr341")
RUN = os.environ.get("RUN", "rtdetr_convnext_tiny_dinov3_bce")
N_WORST = int(os.environ.get("N_WORST", "3"))
IOU_MATCH = 0.75

INST_DIR = REPO / "99_evaluation/02_2stage/u-diads-tl/_full_cross_instances" / SUBSET / RUN
PER_PAGE = REPO / "99_evaluation/02_2stage/u-diads-tl/full_cross_zottin_per_page.csv"
IMG_DIR = REPO / "00_data/U-DIADS-TL" / SUBSET / f"img-{SUBSET}" / "test"
GT_DIR = REPO / "00_data/U-DIADS-TL" / SUBSET / f"text-line-gt-{SUBSET}" / "test"
OUT_DIR = REPO / "99_evaluation/02_2stage/u-diads-tl/viz_worst" / f"{SUBSET}_{RUN}"

GREEN, ORANGE, BLUE, RED = (60, 200, 60), (0, 165, 255), (230, 120, 40), (40, 40, 230)


def components(mask_or_labels, from_mask):
    """-> (labels, {id: (x0, y0, x1, y1)}) so pairs can be prefiltered by box."""
    labels = (cv2.connectedComponents(mask_or_labels.astype(np.uint8))[1]
              if from_mask else mask_or_labels.astype(np.int32))
    boxes = {}
    for i in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == i)
        if len(xs):
            boxes[i] = (xs.min(), ys.min(), xs.max(), ys.max())
    return labels, boxes


def boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def iou(lab_a, ia, lab_b, ib, box):
    """IoU of two components, evaluated only inside their combined bounding box."""
    x0, y0, x1, y1 = box
    A = lab_a[y0:y1 + 1, x0:x1 + 1] == ia
    B = lab_b[y0:y1 + 1, x0:x1 + 1] == ib
    inter = np.logical_and(A, B).sum()
    if not inter:
        return 0.0
    return inter / np.logical_or(A, B).sum()


def classify(gt_lab, gt_boxes, pr_lab, pr_boxes):
    """-> (pred_id -> colour, set of missed gt ids)."""
    matched_gt, pred_colour = set(), {}
    for pid, pb in pr_boxes.items():
        best, best_gid = 0.0, None
        for gid, gb in gt_boxes.items():
            if not boxes_overlap(pb, gb):
                continue
            union = (min(pb[0], gb[0]), min(pb[1], gb[1]), max(pb[2], gb[2]), max(pb[3], gb[3]))
            v = iou(gt_lab, gid, pr_lab, pid, union)
            if v > best:
                best, best_gid = v, gid
        if best >= IOU_MATCH:
            pred_colour[pid] = GREEN
            matched_gt.add(best_gid)
        elif best > 0:
            pred_colour[pid] = ORANGE
        else:
            pred_colour[pid] = BLUE
    return pred_colour, set(gt_boxes) - matched_gt


def paint(labels, colours, shape):
    out = np.zeros((*shape, 3), np.uint8)
    for i, c in colours.items():
        out[labels == i] = c
    return out


def main():
    # page kept as str: the CSV column reads as int and drops the leading
    # zeros the filenames actually carry (071.jpg, not 71.jpg).
    per = pd.read_csv(PER_PAGE, dtype={"page": str})
    per = per[(per.subset == SUBSET) & (per.detector + "_" + per.loss == RUN)]
    if per.empty:
        raise SystemExit(f"no per-page rows for {SUBSET}/{RUN} in {PER_PAGE}")
    worst = per.nsmallest(N_WORST, "FM")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"worst {N_WORST} pages of {SUBSET}/{RUN}:")

    for r in worst.itertuples():
        stem = str(r.page)
        matches = sorted(IMG_DIR.glob(f"{stem}.*")) or sorted(IMG_DIR.glob(f"*{stem}.*"))
        if not matches:
            raise SystemExit(f"no image for page {stem!r} in {IMG_DIR}")
        img = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
        stem = matches[0].stem          # canonical, zero-padded
        gt_lab, gt_boxes = components(
            (cv2.imread(str(GT_DIR / f"{stem}.png")).sum(-1) > 0), True)
        pr_lab, pr_boxes = components(
            cv2.imread(str(INST_DIR / f"{stem}.png"), cv2.IMREAD_UNCHANGED), False)

        pred_colour, missed = classify(gt_lab, gt_boxes, pr_lab, pr_boxes)
        n = {"matched": sum(1 for c in pred_colour.values() if c == GREEN),
             "partial": sum(1 for c in pred_colour.values() if c == ORANGE),
             "spurious": sum(1 for c in pred_colour.values() if c == BLUE),
             "missed_gt": len(missed)}

        gt_vis = paint(gt_lab, {i: tuple(int(v) for v in np.random.default_rng(i).integers(60, 255, 3))
                                for i in gt_boxes}, gt_lab.shape)
        pr_vis = paint(pr_lab, pred_colour, pr_lab.shape)
        for gid in missed:                       # missed GT as red outline on the pred panel
            m = (gt_lab == gid).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pr_vis, cnts, -1, RED, 3)

        blend = lambda v: cv2.addWeighted(img, 0.45, v, 0.55, 0)
        panel = np.hstack([img, blend(gt_vis), blend(pr_vis)])
        cv2.putText(panel, f"{stem}  FM={r.FM:.3f}  GT={len(gt_boxes)}  pred={len(pr_boxes)}  "
                    f"matched={n['matched']} partial={n['partial']} spurious={n['spurious']} "
                    f"missed={n['missed_gt']}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.imwrite(str(OUT_DIR / f"{stem}_FM{r.FM:.3f}.jpg"), panel,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {stem}: FM={r.FM:.3f} GT={len(gt_boxes)} pred={len(pr_boxes)} "
              f"matched={n['matched']} partial={n['partial']} spurious={n['spurious']} "
              f"missed={n['missed_gt']}")

    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    main()
