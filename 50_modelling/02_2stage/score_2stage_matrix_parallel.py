"""Full detector x loss cross for U-DIADS two-stage, with the metric parallelised.

Same computation as evaluate_loss_ablation_detr_udiads.py -- same detections, same
confidence sweep on val, same crop segmenter, same Zottin metric -- reorganised so a
90-cell cross is hours instead of a day:

  1. GPU phase, serial: run each detector once per subset, sweep its confidence on
     val, then render instance maps for every (detector, loss) pair. Detections are
     computed once per detector and reused across the three losses, which the
     original script also does, but here the instance maps are written out instead
     of being scored inline.
  2. CPU phase, pooled: evaluate_metrics is ~23 s per page and dominates everything
     else. Pages are independent, so they go to a process pool. Workers read the
     instance map and the GT from disk and never touch the GPU or transformers.

Incremental: cells already present in the output CSV are skipped, so this can be
re-run to fill gaps. RESCORE_ALL=1 forces a full pass.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

SUBSETS = [s.strip() for s in os.environ.get(
    "SUBSETS", "Latin14396,Latin2,Syr341").split(",") if s.strip()]
ARMS = tuple(a.strip() for a in os.environ.get("ARMS", "bce,tversky,supervoxel").split(",") if a.strip())
# Optional filter: score only these detector run names (default: every run found).
ONLY = {d.strip() for d in os.environ.get("DETECTORS", "").split(",") if d.strip()}
N_WORKERS = max(1, (os.cpu_count() or 4) - 2)
OUT_CSV = REPO / "99_evaluation/02_2stage/u-diads-tl/full_cross_zottin.csv"
INST_ROOT = REPO / "99_evaluation/02_2stage/u-diads-tl/_full_cross_instances"


class Args:
    """The argparse namespace the pipeline helpers expect (defaults from the script)."""
    crop_w, crop_h = 1024, 256
    pad = 15
    segmenter_threshold = 0.5
    min_area = 50
    close_fraction = 0.04
    dedup_iou = 0.9


def split_dir(base, *names):
    for n in names:
        if (base / n).is_dir():
            return base / n
    raise FileNotFoundError(f"none of {names} under {base}")


def score_page(job):
    """Worker: Zottin metric for one rendered instance map. No GPU, no transformers."""
    subset, detector, arm, stem, inst_path, gt_path = job
    inst = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    pix, line, dr, ra, fm = evaluate_metrics(gt, inst.astype(np.int32))
    return dict(subset=subset, detector=detector, loss=arm, page=stem,
                Pixel_IU=pix, Line_IU=line, DR=dr, RA=ra, FM=fm,
                instances=int(inst.max()))


def main():
    done = set()
    previous = pd.DataFrame()
    if OUT_CSV.exists() and not os.environ.get("RESCORE_ALL"):
        previous = pd.read_csv(OUT_CSV)
        done = set(zip(previous.subset, previous.detector, previous.loss))
        print(f"[incremental] {len(done)} cell(s) already scored")

    jobs = []
    for subset in SUBSETS:
        root = REPO / "00_data/U-DIADS-TL" / subset
        img_dir = split_dir(root / f"img-{subset}", "test", "public-test")
        val_dir = split_dir(root / f"img-{subset}", "val", "validation")
        gt_dir = split_dir(root / f"text-line-gt-{subset}", "test", "public-test")
        val_gt = split_dir(root / f"text-line-gt-{subset}", "validation", "val")
        test_paths, val_paths = E.image_paths(img_dir), E.image_paths(val_dir)
        det_root = REPO / "80_models/02_2stage/u-diads-tl/detection/rtdetr_hf" / subset
        seg_root = (REPO / "80_models/02_2stage/u-diads-tl"
                    / "crop_seg_loss_ablation_components_1024x256" / subset)

        for run_dir in sorted(det_root.iterdir()):
            model_dir = run_dir / "best_model"
            if not model_dir.exists():
                continue
            detector = run_dir.name
            if ONLY and detector not in ONLY:
                continue
            todo = [a for a in ARMS if (subset, detector, a) not in done
                    and (seg_root / a / "best.pth").exists()]
            if not todo:
                continue

            # Detections once per detector, reused by every loss arm.
            raw = E.raw_detections(model_dir, val_paths + test_paths)
            conf, _ = E.select_confidence({k: raw[k] for k in (p.stem for p in val_paths)},
                                          val_paths, val_gt, Args.dedup_iou,
                                          np.round(np.arange(0.01, 0.92, 0.02), 2))
            dets = E.finalized_detections(raw, test_paths, conf, Args.dedup_iou)
            print(f"[{subset}/{detector}] conf={conf} arms={todo}", flush=True)

            for arm in todo:
                model, _ = E.load_segmenter(seg_root / arm / "best.pth", E.DEVICE)
                out_dir = INST_ROOT / subset / f"{detector}_{arm}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for p in test_paths:
                    dst = out_dir / f"{p.stem}.png"
                    if not dst.exists():
                        page = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        inst = E.predict_instances(page, dets[p.stem][0], model, Args)
                        cv2.imwrite(str(dst), inst)
                    jobs.append((subset, detector, arm, p.stem, dst,
                                 gt_dir / f"{p.stem}.png"))
                del model

    if not jobs:
        print("nothing to score")
        return
    print(f"\n[score] {len(jobs)} page-jobs across {N_WORKERS} workers\n", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(score_page, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} pages", flush=True)

    per_page = pd.DataFrame(rows)
    agg = (per_page.groupby(["subset", "detector", "loss"])
           [["Pixel_IU", "Line_IU", "DR", "RA", "FM", "instances"]]
           .mean().round(4).reset_index())
    out = pd.concat([previous, agg], ignore_index=True) if len(previous) else agg
    out = out.drop_duplicates(["subset", "detector", "loss"], keep="last")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    per_page.to_csv(OUT_CSV.with_name("full_cross_zottin_per_page.csv"), index=False)
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
