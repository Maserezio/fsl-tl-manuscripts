"""PREDICT (U-DIADS-TL) — single-stage YOLO-seg -> instance label map -> OFFICIAL Zottin metric.

The page-level YOLO-seg predicts one mask per text line; masks become an instance label
map which is scored directly with evaluate_metrics (Pixel_IU, Line_IU, DR, RA, FM) against
the binary line GT — same evaluator as the U-Net comparison, so rows are comparable.

  python predict_yolo_udiads.py --manuscript Latin14396
  python predict_yolo_udiads.py --manuscript Latin14396 --weights path/to/best.pt
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "50_modelling" / "01_simple_segmentation"))
sys.path.append(str(REPO / "71_misc"))
from data.diva_dataset import _resolve_layout          # noqa: E402  (u_diads family supported)
from evaluate_util import evaluate_metrics             # noqa: E402  (official Zottin 5-tuple)

_EXTS = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manuscript", required=True, help="Latin14396 | Latin2 | Syr341")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=None,
                    help="default: 80_models/02_2stage/u-diads-tl/segmentation/yolo_seg_<ms>/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960, help="must match training imgsz")
    ap.add_argument("--min-area", type=int, default=100)
    args = ap.parse_args()

    MS = args.manuscript
    weights = Path(args.weights) if args.weights else (
        REPO / "80_models" / "02_2stage" / "u-diads-tl" / "segmentation" /
        f"yolo_seg_{MS.lower()}" / "weights" / "best.pt")
    if not weights.exists():
        raise SystemExit(f"no weights at {weights} — train first (run_yolo_seg_udiads.sh)")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    layout = _resolve_layout(str(REPO / "00_data" / "U-DIADS-TL"), MS, "u_diads")[args.split]
    img_dir, gt_dir = Path(layout["img"]), Path(layout["gt"])
    out_dir = REPO / "99_evaluation" / "02_2stage" / "u-diads-tl" / f"yolo_single_{MS}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in _EXTS)
    rows, agg = [], {k: [] for k in ("Pixel_IU", "Line_IU", "DR", "RA", "FM")}
    for stem in tqdm(stems, desc=f"yolo {MS}"):
        imgp = next(p for p in img_dir.iterdir() if p.stem == stem and p.suffix.lower() in _EXTS)
        img = cv2.cvtColor(cv2.imread(str(imgp)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        gt = (cv2.imread(str(gt_dir / f"{stem}.png")).sum(axis=-1) > 0).astype(np.uint8)

        r = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False, retina_masks=False)[0]
        label_map = np.zeros((H, W), np.int32)
        nxt = 1
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():                 # model-space -> page
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                if m.sum() < args.min_area:
                    continue
                label_map[m] = nxt
                nxt += 1

        pix, line, DR, RA, FM = evaluate_metrics(gt, label_map)
        rows.append(dict(page=stem, lines=nxt - 1, Pixel_IU=round(pix, 4), Line_IU=round(line, 4),
                         DR=round(DR, 4), RA=round(RA, 4), FM=round(FM, 4)))
        for k, v in zip(agg, (pix, line, DR, RA, FM)):
            agg[k].append(v)
        print(f"  {stem}: lines={nxt - 1}  Line_IU={line:.3f}  FM={FM:.3f}", flush=True)

    means = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
    with (out_dir / "zottin_metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
        w.writerow(dict(page="MEAN", lines="", **{k: v for k, v in means.items()}))

    print(f"\n=== yolo single-stage — {MS} {args.split} (official Zottin, {len(stems)} pages) ===")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))
    print(f"saved -> {out_dir / 'zottin_metrics.csv'}")


if __name__ == "__main__":
    main()
