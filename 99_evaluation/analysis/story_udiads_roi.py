#!/usr/bin/env python3
"""Why the native Mask R-CNN ROI masks fail on U-DIADS-TL although the boxes are good: same CATMuS ConvNeXt-Tiny
Mask R-CNN, test page, (a) ROI-mask instances vs (b) the same detector's boxes refined by the crop U-Net.
Panels per method: instances in colour, and a pixel error map against the binary ink ground truth
(green: GT ink covered by the prediction of its line, orange: predicted pixels that are not GT ink,
blue: GT ink not covered). Per-line IoU at the FEST threshold 0.75 printed.

    .venv/bin/python 99_evaluation/analysis/story_udiads_roi.py SUBSET STEM x0 y0 x1 y1
"""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import story_udiads as U  # noqa: E402

EV = U.EV / "instance_segmentation/mask_rcnn/u-diads-tl"


def roi_labels(ms, stem):
    return cv2.imread(str(EV / f"{ms}/maskrcnn_convnext_tiny_catmus_1024_200ep/test/instances/{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.int64)


def line_iou(gt, lab):
    """best IoU per GT component, and the predicted label it matches"""
    n, comp = cv2.connectedComponents(gt)
    out = []
    for i in range(1, n):
        g = comp == i
        ids, cnt = np.unique(lab[g], return_counts=True)
        cnt, ids = cnt[ids > 0], ids[ids > 0]
        if not len(ids):
            out.append((i, 0, 0.0)); continue
        best = 0.0, 0
        for j in ids[np.argsort(-cnt)][:3]:
            p = lab == j
            best = max(best, ((g & p).sum() / (g | p).sum(), j))
        out.append((i, best[1], best[0]))
    return comp, out


def panels(img, gt, lab, box, name, ms, stem):
    x0, y0, x1, y1 = box
    comp, m = line_iou(gt, lab)
    owner = {j: i for i, j, _ in m}
    base = (0.55 * img + 0.45 * 255).astype(np.float32)
    inst = base.copy()
    ids = [i for i in np.unique(lab[y0:y1, x0:x1]) if i > 0]
    ids.sort(key=lambda i: np.where(lab == i)[0].mean())
    for k, i in enumerate(ids):
        sel = lab == i
        inst[sel] = 0.45 * inst[sel] + 0.55 * U.PAL[k % len(U.PAL)]
    err = base.copy()
    g = gt > 0
    covered = np.zeros_like(g)
    for i, j, _ in m:
        if j:
            covered |= (comp == i) & (lab == j)
    err[g & covered] = (44, 160, 44)
    err[g & ~covered] = (31, 119, 180)
    err[(lab > 0) & ~g] = (255, 127, 14)
    for a, tag in ((inst, "inst"), (err, "err")):
        cv2.imwrite(str(U.OUT / f"udiads_roi_{ms}_{stem}_{name}_{tag}.png"), cv2.cvtColor(a[y0:y1, x0:x1].astype(np.uint8), cv2.COLOR_RGB2BGR))
    ious = np.array([v for _, _, v in m])
    fm = U.evaluate_metrics(gt, lab)[4]
    print(f"{name:9s} page FM {100 * fm:5.1f}  lines {len(ious)}  matched@0.75 {(ious >= 0.75).sum()}  median line IoU {np.median(ious):.2f}"
          f"  pred px not ink {100 * ((lab > 0) & ~g).sum() / max((lab > 0).sum(), 1):.0f}%")


def main():
    ms, stem = sys.argv[1], sys.argv[2]
    box = tuple(map(int, sys.argv[3:7]))
    img = cv2.cvtColor(cv2.imread(str(next((U.DATA / ms / f"img-{ms}/test").glob(stem + ".*")))), cv2.COLOR_BGR2RGB).astype(np.float32)
    gt = U.gt(ms, stem)
    panels(img, gt, roi_labels(ms, stem), box, "roi", ms, stem)
    panels(img, gt, U.labels("maskrcnn", ms, stem), box, "crop", ms, stem)
    x0, y0, x1, y1 = box
    g = cv2.connectedComponents(gt)[1]
    gi = (0.55 * img + 0.45 * 255).astype(np.float32)
    ids = sorted([i for i in np.unique(g[y0:y1, x0:x1]) if i > 0], key=lambda i: np.where(g == i)[0].mean())
    for k, i in enumerate(ids):
        sel = g == i
        gi[sel] = 0.45 * gi[sel] + 0.55 * U.PAL[k % len(U.PAL)]
    cv2.imwrite(str(U.OUT / f"udiads_roi_{ms}_{stem}_gt.png"), cv2.cvtColor(gi[y0:y1, x0:x1].astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
