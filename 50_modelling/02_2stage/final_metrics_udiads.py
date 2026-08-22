"""All five official metrics on test, tuned postprocessing vs the default.

SUBSET / DETECTOR / ARM / TUNED (as "pad,threshold,close_fraction") come from the
environment so the same script serves every subset.

tune_postproc_syr341.py only needed DR/RA/FM to rank settings, so its fast matcher
skipped Pixel_IU and Line_IU -- those come from calculate_pixel_and_line_IU, a
separate part of the official metric. This rebuilds the instance maps for both
settings and runs evaluate_util.evaluate_metrics unmodified, so every number here is
the official one.

Setting chosen on VALIDATION (see Syr341_val_sweep.csv), not on test:
    pad=15, threshold=0.4, close_fraction=0.005   vs default 15 / 0.5 / 0.04

The metric is O(n_gt x n_pred) full-page mask ops and Syr341 runs ~175 lines per
page, so it goes in a process pool; the GPU part (one segmenter forward per box)
happens once per setting up front.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# TUNED is whatever the validation sweep picked for this subset; the default is the
# shipped setting, kept alongside so the delta is visible.
_t = [float(x) for x in os.environ.get("TUNED", "15,0.4,0.005").split(",")]
SETTINGS = {
    "tuned":   (int(_t[0]), _t[1], _t[2]),
    "default": (15, 0.5, 0.04),
}
INST_ROOT = REPO / "99_evaluation/02_2stage/u-diads-tl/postproc_tuned/_instances"
OUT = REPO / "99_evaluation/02_2stage/u-diads-tl/postproc_tuned"
N_WORKERS = max(1, (os.cpu_count() or 4) - 2)


def score(job):
    """Worker: the official five-tuple for one rendered page."""
    label, stem, inst_path, gt_path = job
    inst = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
    gt = (cv2.imread(str(gt_path)).sum(-1) > 0).astype(np.uint8)
    pix, line, dr, ra, fm = evaluate_metrics(gt, inst.astype(np.int32))
    return dict(setting=label, page=stem, Pixel_IU=pix, Line_IU=line,
                DR=dr, RA=ra, FM=fm, instances=int(inst.max()))


def main():
    root = REPO / "00_data/U-DIADS-TL" / SUBSET
    img_dir = next(d for d in (root / f"img-{SUBSET}").iterdir() if d.name in ("test", "public-test"))
    gt_dir = next(d for d in (root / f"text-line-gt-{SUBSET}").iterdir() if d.name in ("test", "public-test"))
    val_dir = next(d for d in (root / f"img-{SUBSET}").iterdir() if d.name in ("val", "validation"))
    val_gt = next(d for d in (root / f"text-line-gt-{SUBSET}").iterdir() if d.name in ("validation", "val"))
    test_paths, val_paths = E.image_paths(img_dir), E.image_paths(val_dir)

    det = REPO / "80_models/02_2stage/u-diads-tl/detection/rtdetr_hf" / SUBSET / DETECTOR / "best_model"
    raw = E.raw_detections(det, val_paths + test_paths)
    conf, _ = E.select_confidence({p.stem: raw[p.stem] for p in val_paths}, val_paths,
                                  val_gt, 0.9, np.round(np.arange(0.01, 0.92, 0.02), 2))
    dets = E.finalized_detections(raw, test_paths, conf, 0.9)
    model, _ = E.load_segmenter(
        REPO / "80_models/02_2stage/u-diads-tl"
        / "crop_seg_loss_ablation_components_1024x256" / SUBSET / ARM / "best.pth", E.DEVICE)
    print(f"conf={conf}, {len(test_paths)} test pages", flush=True)

    jobs = []
    for label, (pad, thr, cf) in SETTINGS.items():
        # Subset in the path: page stems collide across subsets (Latin14396 and
        # Latin2 both have 251/252), and the cache below skips existing files,
        # so a flat directory silently scores one subset against another's map.
        out_dir = INST_ROOT / SUBSET / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in test_paths:
            dst = out_dir / f"{path.stem}.png"
            if not dst.exists():
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                H, W = img.shape[:2]
                inst = np.zeros((H, W), np.uint16)
                nid = 1
                with torch.no_grad():
                    for b in dets[path.stem][0]:
                        x1, y1, x2, y2 = np.rint(b).astype(int)
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                        t = torch.from_numpy(np.ascontiguousarray(
                            cv2.resize(rgb, (CROP_W, CROP_H)))).permute(2, 0, 1).float()[None].to(E.DEVICE) / 255.0
                        p = cv2.resize(torch.sigmoid(model(t))[0, 0].cpu().numpy(),
                                       (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                        m = E.connect_line(p >= thr, cf).astype(bool)
                        if not m.any():
                            continue
                        region = inst[y1:y2, x1:x2]
                        w = m & (region == 0)
                        if w.any():
                            region[w] = nid
                            nid += 1
                cv2.imwrite(str(dst), inst)
            jobs.append((label, path.stem, dst, gt_dir / f"{path.stem}.png"))
        print(f"  [{label}] instances ready", flush=True)

    print(f"\nscoring {len(jobs)} page-settings across {N_WORKERS} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(score, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 5 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    per = pd.DataFrame(rows)
    agg = per.groupby("setting")[["Pixel_IU", "Line_IU", "DR", "RA", "FM"]].mean().round(4)
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / f"{SUBSET}_final_per_page.csv", index=False)
    agg.to_csv(OUT / f"{SUBSET}_final_summary.csv")
    print(f"\n=== {SUBSET} test, all five official metrics ===")
    print(agg.to_string())
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
