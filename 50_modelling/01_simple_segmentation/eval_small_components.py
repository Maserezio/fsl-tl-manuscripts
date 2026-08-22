"""Recall on *small* GT connected components — the fine-detail probe for the
skip-connection ablation (run_skip_ablation.sh).

Rationale: Pixel IU / Line IU / FM are dominated by the bulk of the text body,
which a coarse (stride-32-only) decoder can still cover approximately. What the
high-resolution skips are supposed to carry is high-frequency detail: descenders,
diacritics, isolated punctuation, thin stroke fragments — the tiny connected
components of the pixel GT. This script measures exactly those.

Measured on the RAW model output (sliding-window prob map thresholded at 0.5),
deliberately *before* any post-processing: the DIVA export pipeline
(predict_diva_seamcarve.py) removes components below `--clean-min-size` 200 px,
which would zero this metric by construction for every model. So this number is
a property of the segmentation head, not of the line-export heuristics.

A GT component counts as detected when at least `--cover` (default 0.5) of its
pixels are predicted foreground. Don't-care pixels (DIVA red==128) are excluded
from the GT mask before labelling, matching the training loss.

Usage:
  python eval_small_components.py --runs unet_resnet34_diva_CB55 ...
  python eval_small_components.py --runs-glob 'skipabl_*_CB55' --out small_cc.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from data.diva_dataset import _load_mask, _resolve_image_path, _resolve_layout  # noqa: E402
from evaluate import _resolve_data_root, sliding_window_inference               # noqa: E402
from models import build_model                                                  # noqa: E402

DEFAULT_CKPT_ROOT = REPO / "80_models/01_simple_segmentation/diva-hisdb/segmentation/simple_segmentation"


def component_recall(gt_mask: np.ndarray, pred_mask: np.ndarray, max_area: int, cover: float) -> dict:
    """Split GT components at `max_area` and score detection on each bucket."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask.astype(np.uint8), connectivity=8)
    out = {"small_total": 0, "small_hit": 0, "large_total": 0, "large_hit": 0,
           "small_gt_px": 0, "small_tp_px": 0}

    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x, y, w, h = (stats[label_id, cv2.CC_STAT_LEFT], stats[label_id, cv2.CC_STAT_TOP],
                      stats[label_id, cv2.CC_STAT_WIDTH], stats[label_id, cv2.CC_STAT_HEIGHT])
        # score inside the component's bbox only — full-page masking per component
        # is what makes the naive version O(n_components * page) and unusably slow
        comp = labels[y:y + h, x:x + w] == label_id
        covered = int(np.count_nonzero(pred_mask[y:y + h, x:x + w][comp]))

        bucket = "small" if area < max_area else "large"
        out[f"{bucket}_total"] += 1
        out[f"{bucket}_hit"] += int(covered >= cover * area)
        if bucket == "small":
            out["small_gt_px"] += area
            out["small_tp_px"] += covered

    return out


def eval_run(ckpt: Path, split: str, max_area: int, cover: float, threshold: float, device) -> dict | None:
    blob = torch.load(ckpt, map_location=device)
    cfg = blob.get("cfg")
    if cfg is None:
        print(f"[skip] {ckpt}: no embedded cfg")
        return None

    model = build_model(cfg).to(device)
    state = {("net." + k[len("unet."):] if k.startswith("unet.") else k): v
             for k, v in blob["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()

    manuscript = cfg["data"]["manuscript"]
    layout = _resolve_layout(_resolve_data_root(cfg["data"]["data_root"]), manuscript, "diva")[split]
    img_dir, gt_dir = layout["img"], layout["gt"]

    stems = sorted(p.stem for p in Path(gt_dir).iterdir() if p.suffix.lower() == ".png")
    agg = {k: 0 for k in ("small_total", "small_hit", "large_total", "large_hit",
                          "small_gt_px", "small_tp_px")}

    for stem in tqdm(stems, desc=ckpt.parent.name, leave=False):
        img_rgb = cv2.cvtColor(cv2.imread(_resolve_image_path(img_dir, stem)), cv2.COLOR_BGR2RGB)
        gt_mask, dont_care = _load_mask(str(Path(gt_dir) / f"{stem}.png"), "diva")
        gt_mask = gt_mask & ~dont_care          # same exclusion the training loss uses

        prob = sliding_window_inference(
            model, img_rgb, crop_size=cfg["training"]["crop_size"],
            device=device, use_amp=cfg["training"].get("amp", True),
        )
        pred = prob > threshold

        for key, value in component_recall(gt_mask, pred, max_area, cover).items():
            agg[key] += value

    del model
    torch.cuda.empty_cache()

    return {
        "run": ckpt.parent.name,
        "encoder": cfg["model"].get("encoder_name"),
        "n_skips": cfg["model"].get("n_skips", 4),
        "pages": len(stems),
        "small_components": agg["small_total"],
        "small_cc_recall": round(agg["small_hit"] / max(agg["small_total"], 1), 4),
        "small_cc_pixel_recall": round(agg["small_tp_px"] / max(agg["small_gt_px"], 1), 4),
        "large_components": agg["large_total"],
        "large_cc_recall": round(agg["large_hit"] / max(agg["large_total"], 1), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=None, help="run folder names under --ckpt-root")
    parser.add_argument("--runs-glob", default=None, help="glob over --ckpt-root instead of --runs")
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-area", type=int, default=200, help="GT components below this are 'small'")
    parser.add_argument("--cover", type=float, default=0.5, help="fraction of a GT component that must be predicted")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", default=None, help="CSV output path")
    args = parser.parse_args()

    root = Path(args.ckpt_root)
    if args.runs_glob:
        names = sorted(p.name for p in root.glob(args.runs_glob) if (p / "best.pth").exists())
    elif args.runs:
        names = args.runs
    else:
        parser.error("pass --runs or --runs-glob")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in names:
        ckpt = root / name / "best.pth"
        if not ckpt.exists():
            print(f"[skip] {name}: no best.pth")
            continue
        row = eval_run(ckpt, args.split, args.max_area, args.cover, args.threshold, device)
        if row is None:
            continue
        rows.append(row)
        print(f"{row['run']:44s} n_skips={row['n_skips']}  "
              f"small_cc_recall={row['small_cc_recall']:.4f} "
              f"({row['small_components']} comps)  "
              f"small_cc_pixel_recall={row['small_cc_pixel_recall']:.4f}  "
              f"large_cc_recall={row['large_cc_recall']:.4f}")

    if args.out and rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
