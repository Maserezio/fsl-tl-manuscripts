"""Locate the two-stage bottleneck on Syr341 by replacing the detector with an oracle.

Syr341's best two-stage run keeps Pixel_IU 0.762 and Line_IU 0.911 -- essentially
Latin14396's 0.780 / 0.934 -- yet FM collapses from 0.870 to 0.687. The whole loss
is in DR/RA, i.e. the count of GT lines reaching IoU >= 0.75 against a predicted
instance. That is consistent with two very different causes:

  (a) the detector misses or mis-frames lines, or
  (b) the crop segmenter cannot separate lines inside a correctly framed crop.

Feeding GT-derived boxes into the same segmenter separates them. If oracle boxes
restore FM, the detector is the bottleneck; if FM stays low, the second stage is.

Boxes come from the GT instance map's connected components, so they are exactly the
boxes a perfect detector would emit. Everything downstream -- crop geometry, padding,
segmenter, re-assembly, metric -- is the unmodified pipeline.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "71_misc"))

import evaluate_loss_ablation_detr_udiads as E  # noqa: E402
from evaluate_util import evaluate_metrics  # noqa: E402

SUBSETS = ["Syr341", "Latin14396"]     # Latin14396 as the control
ARM = "bce"                            # best second-stage loss on Syr341
PAD = 15
CROP_W, CROP_H = 1024, 256
SEG_THRESHOLD = 0.5
MIN_AREA = 50


class Args:
    """Mirrors the argparse namespace the pipeline functions expect."""
    crop_w, crop_h = CROP_W, CROP_H
    pad = PAD
    segmenter_threshold = SEG_THRESHOLD
    min_area = MIN_AREA
    close_fraction = 0.04          # argparse default in the eval script
    dedup_iou = 0.9


def gt_boxes(gt_path):
    """Boxes a perfect detector would emit: bounding box of every GT component."""
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    n, labels = cv2.connectedComponents((gt > 0).astype(np.uint8))
    boxes = []
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        if len(xs) < MIN_AREA:
            continue
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(boxes, dtype=np.float32), gt


def main():
    rows = []
    for subset in SUBSETS:
        root = REPO / "00_data/U-DIADS-TL" / subset
        img_dir = next(d for d in (root / f"img-{subset}").iterdir()
                       if d.name in ("test", "public-test"))
        gt_dir = next(d for d in (root / f"text-line-gt-{subset}").iterdir()
                      if d.name in ("test", "public-test"))
        seg = (REPO / "80_models/02_2stage/u-diads-tl"
               / "crop_seg_loss_ablation_components_1024x256" / subset / ARM / "best.pth")
        model, _ = E.load_segmenter(seg, E.DEVICE)

        for img in sorted(p for p in img_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif")):
            boxes, gt = gt_boxes(gt_dir / f"{img.stem}.png")
            page = cv2.imread(str(img), cv2.IMREAD_COLOR)
            inst = E.predict_instances(page, boxes, model, Args)
            pix, line, dr, ra, fm = evaluate_metrics(gt, inst.astype(np.int32))
            rows.append(dict(subset=subset, page=img.stem, gt_lines=len(boxes),
                             instances=int(inst.max()), Pixel_IU=pix, Line_IU=line,
                             DR=dr, RA=ra, FM=fm))
            print(f"  [{subset}] {img.stem}: gt_boxes={len(boxes)} "
                  f"instances={inst.max()} FM={fm:.3f}", flush=True)

    df = pd.DataFrame(rows)
    out = REPO / "99_evaluation/02_2stage/u-diads-tl/oracle_boxes_probe.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}\n")
    agg = df.groupby("subset")[["gt_lines", "instances", "Pixel_IU", "Line_IU", "DR", "RA", "FM"]].mean()
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
