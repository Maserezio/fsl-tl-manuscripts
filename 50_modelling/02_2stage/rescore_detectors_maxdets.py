"""Re-score trained detectors with a detection cap that fits the data.

    FAMILY=diva python rescore_detectors_maxdets.py

torchmetrics' MeanAveragePrecision defaults to max_detection_thresholds=[1,10,100].
Syr341 averages 175 GT lines per page, so mAR@100 cannot exceed 100/175 = 0.571 and
AP is truncated at the same point: the metric runs out of detection slots before the
detector runs out of lines. Latin14396 (78 lines) and Latin2 (98) are under the cap,
so only Syr341 was being measured against a lower ceiling -- which made its detector
look far worse than it is.

The same cap bites DIVA: CS18 carries 141-283 GT lines per test page and CS863 up
to 174 (CB55 tops out at 90, so it is unaffected and is rescored only so that every
reported number has the same provenance).

Recomputed here with max_detection_thresholds=[1, 10, 300]; 300 is RT-DETR's
num_queries, i.e. the most it can emit. Everything else -- checkpoints, images,
box decoding -- is unchanged, so the delta is purely the cap.

BACKEND: torchmetrics defaults to pycocotools, whose summarize() only ever looks up
maxDets == 100 and so returns map = -1 for any other threshold list -- which is why
the mAP5095_cap300 column of the first run of this script was entirely -1. Verified
on a toy case that faster_coco_eval computes it correctly and agrees with
pycocotools exactly at the default cap.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import evaluate_loss_ablation_detr_udiads as E  # noqa: E402

MAX_DETS = [1, 10, 300]
BACKEND = "faster_coco_eval"

FAMILY = os.environ.get("FAMILY", "udiads")
if FAMILY == "diva":
    SUBSETS = ["CB55", "CS863", "CS18"]
    DATA_ROOT = REPO / "00_data/DIVA-HisDB"
    DET_ROOT = REPO / "80_models/02_2stage/diva-hisdb/detection/rtdetr_hf"
    IMG_SPLIT = "public-test"
    OUT = REPO / "99_evaluation/02_2stage/diva-hisdb/detector_maxdets300.csv"
    coco_dir = lambda sub: DATA_ROOT / f"coco_dataset_{sub}"
    img_dir = lambda sub: DATA_ROOT / sub / f"img-{sub}" / "img" / IMG_SPLIT
    # CB55 kept the flat layout its first runs wrote into; the others are nested.
    det_dir = lambda sub: DET_ROOT if sub == "CB55" else DET_ROOT / sub
else:
    SUBSETS = ["Syr341", "Latin14396", "Latin2"]
    DATA_ROOT = REPO / "00_data/U-DIADS-TL"
    DET_ROOT = REPO / "80_models/02_2stage/u-diads-tl/detection/rtdetr_hf"
    OUT = REPO / "99_evaluation/02_2stage/u-diads-tl/detector_maxdets300.csv"
    coco_dir = lambda sub: DATA_ROOT / f"coco_dataset_{sub.lower()}"
    img_dir = lambda sub: next(d for d in (DATA_ROOT / sub / f"img-{sub}").iterdir()
                               if d.name in ("test", "public-test"))
    det_dir = lambda sub: DET_ROOT / sub


def gt_targets(subset, stems):
    """COCO test annotations as torchmetrics targets, keyed by file stem."""
    coco = json.loads((coco_dir(subset) / "test.json").read_text())
    by_id = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    boxes = {s: [] for s in by_id.values()}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        boxes[by_id[a["image_id"]]].append([x, y, x + w, y + h])
    return {s: torch.tensor(v, dtype=torch.float32).reshape(-1, 4) for s, v in boxes.items()}


def main():
    rows = []
    for subset in SUBSETS:
        paths = E.image_paths(img_dir(subset))
        targets = gt_targets(subset, [p.stem for p in paths])
        det_root = det_dir(subset)

        for run_dir in sorted(det_root.iterdir()):
            model_dir = run_dir / "best_model"
            if not model_dir.exists():
                continue
            raw = E.raw_detections(model_dir, paths)
            preds, tgts = [], []
            for p in paths:
                b, s = raw[p.stem]
                preds.append({"boxes": torch.as_tensor(b, dtype=torch.float32).reshape(-1, 4),
                              "scores": torch.as_tensor(s, dtype=torch.float32).reshape(-1),
                              "labels": torch.zeros(len(s), dtype=torch.long)})
                t = targets[p.stem]
                tgts.append({"boxes": t, "labels": torch.zeros(len(t), dtype=torch.long)})

            out = {}
            for tag, caps in (("cap100", [1, 10, 100]), ("cap300", MAX_DETS)):
                m = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                         class_metrics=False, max_detection_thresholds=caps,
                                         backend=BACKEND)
                m.update(preds, tgts)
                r = m.compute()
                out[tag] = (float(r["map_50"]), float(r["map"]),
                            float(r[f"mar_{caps[-1]}"]))
            rows.append(dict(subset=subset, run=run_dir.name,
                             mAP50_cap100=out["cap100"][0], mAP50_cap300=out["cap300"][0],
                             mAP5095_cap100=out["cap100"][1], mAP5095_cap300=out["cap300"][1],
                             mAR_cap100=out["cap100"][2], mAR_cap300=out["cap300"][2],
                             mean_preds=float(np.mean([len(p["scores"]) for p in preds]))))
            r = rows[-1]
            print(f"  [{subset}] {run_dir.name:32} "
                  f"mAP50 {r['mAP50_cap100']:.3f}->{r['mAP50_cap300']:.3f}  "
                  f"mAR {r['mAR_cap100']:.3f}->{r['mAR_cap300']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
