"""Unified line-segmentation evaluation for the 6-backbone DIVA-HisDB + U-DIADS-TL
comparison (see run_train_all.sh) -- one combined table/CSV, same 5-tuple output for
both families:

    Pixel_IU  Line_IU  DR  RA  FM

Note: the two families are NOT evaluated with the same tool, despite the common
output schema:
  - **DIVA-HisDB**: prediction is exported to PAGE-XML (predict_diva_seamcarve.py),
    then scored with the official DIVA Line Segmentation Evaluator (Java). Its own
    output uses different names for the same concepts (PixelIU/LinesIU/LinesRecall/
    LinesPrecision/LinesFMeasure) -- remapped here to the common schema
    (DR=recall="detection rate", RA=precision="recognition accuracy", FM=F-measure).
  - **U-DIADS-TL**: prediction stays a binary mask (no PAGE-XML, no region GT);
    scored with the Python "Zottin" metric (71_misc/evaluate_util.py) after ARU-Net
    fusion + seam-carve (postproc.py), reading from preprocessing/cache_predictions.py's
    cached probability maps when present.
Both ultimately compute the same underlying quantities (pixel IoU after best-match
line assignment; line-level detection rate / recognition accuracy / F-measure) --
the output rows are directly comparable across families despite the different tools.

Usage:
  python evaluate_lines.py                                    # all 6 backbones, both families, all subsets
  python evaluate_lines.py --family udiads --encoders resnet34
  python evaluate_lines.py --family diva --subsets CB55
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(REPO / "71_misc"))

from data.diva_dataset import _resolve_layout       # noqa: E402
from evaluate_util import evaluate_metrics           # noqa: E402
from models.arunet import ARUNetWrapper               # noqa: E402
from postproc import build_fused, run_pipeline, PER_MS_PARAMS  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_METRIC_KEYS = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")

DEFAULT_ENCODERS = [
    "resnet34",
    "resnet50",
    "tu-convnext_tiny.in12k_ft_in1k",
    "tu-convnext_tiny.dinov3_lvd1689m",
    "vit_small_patch16_224.augreg_in21k",
    "vit_small_patch16_dinov3",
]

# DIVA Java evaluator's own key names -> the common 5-tuple schema.
_DIVA_KEY_MAP = {
    "PixelIU": "Pixel_IU",
    "LinesIU": "Line_IU",
    "LinesRecall": "DR",
    "LinesPrecision": "RA",
    "LinesFMeasure": "FM",
}

FAMILIES = {
    "diva": dict(
        data_root="00_data/DIVA-HisDB",
        dataset_family="diva",
        subsets=["CB55", "CS18", "CS863"],
        ckpt_root="80_models/01_simple_segmentation/diva-hisdb/segmentation/simple_segmentation",
        out_root="99_evaluation/01_simple_segmentation/diva-hisdb/lines_eval",
    ),
    "udiads": dict(
        data_root="00_data/U-DIADS-TL",
        dataset_family="u_diads",
        subsets=["Latin14396", "Latin2", "Syr341"],
        ckpt_root="80_models/01_simple_segmentation/u-diads-tl/segmentation/simple_segmentation",
        cache_root="99_evaluation/01_simple_segmentation/u-diads-tl/prob_cache",
    ),
}


def img_stems(img_dir):
    return sorted({p.stem for p in Path(img_dir).iterdir() if p.suffix.lower() in _EXTS})


def read_rgb(img_dir, stem):
    for e in _EXTS:
        p = Path(img_dir) / f"{stem}{e}"
        if p.exists():
            return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f"{stem} in {img_dir}")


def to_label_map(binary):
    _, lbl = cv2.connectedComponents(binary.astype(np.uint8))
    return lbl.astype(np.int32)


def eval_diva_one(encoder: str, ms: str, split: str, run: str | None = None) -> dict | None:
    spec = FAMILIES["diva"]
    run = run or f"unet_{encoder}_diva_{ms}"
    ckpt = REPO / spec["ckpt_root"] / run / "best.pth"
    if not ckpt.exists():
        print(f"[skip] {run}: no checkpoint at {ckpt}")
        return None

    out_dir = REPO / spec["out_root"] / run
    cmd = [
        sys.executable, str(HERE / "predict_diva_seamcarve.py"),
        "--checkpoint", str(ckpt), "--manuscript", ms, "--split", split,
        "--output-dir", str(out_dir), "--approx-ratio", "0.001",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {run}: {e}")
        return None

    summary_csv = out_dir / "diva_summary.csv"
    if not summary_csv.exists():
        print(f"[skip] {run}: no diva_summary.csv produced (Java evaluator unavailable?)")
        return None
    with summary_csv.open() as f:
        raw = next(csv.DictReader(f))
    return {v: float(raw[k]) for k, v in _DIVA_KEY_MAP.items() if k in raw}


def eval_udiads_one(encoder: str, ms: str, split: str, arunet) -> dict | None:
    spec = FAMILIES["udiads"]
    run = f"unet_{encoder}_udiads_{ms}"
    cache_dir = REPO / spec["cache_root"] / run
    if not cache_dir.exists() or not any(cache_dir.glob("*.npy")):
        print(f"[skip] {run}: no cached predictions at {cache_dir} "
              f"(run preprocessing/cache_predictions.py first)")
        return None

    layout = _resolve_layout(str(REPO / spec["data_root"]), ms, spec["dataset_family"])[split]
    img_dir, gt_dir = layout["img"], layout["gt"]
    params = PER_MS_PARAMS.get(ms, PER_MS_PARAMS["Latin14396"])
    disc = {k: params[k] for k in ("line_sigma", "min_line_distance", "peak_frac", "valley_ratio")}
    stems = img_stems(img_dir)

    agg = {k: [] for k in _METRIC_KEYS}
    for stem in tqdm(stems, desc=run, leave=False):
        cache_f = cache_dir / f"{stem}.npy"
        if not cache_f.exists():
            continue
        img = read_rgb(img_dir, stem)
        gt = (cv2.imread(str(Path(gt_dir) / f"{stem}.png")).sum(axis=-1) > 0).astype(np.uint8)
        seg_prob = np.load(cache_f).astype(np.float32)

        fused = build_fused(img, seg_prob, arunet)
        refined = run_pipeline(fused, disc)
        pix, line, DR, RA, FM = evaluate_metrics(gt, to_label_map(refined))
        for k, v in zip(agg, (pix, line, DR, RA, FM)):
            agg[k].append(v)

    if not agg["Pixel_IU"]:
        return None
    return {k: round(float(np.mean(v)), 4) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["diva", "udiads"], default=None, help="default: both")
    ap.add_argument("--encoders", nargs="+", default=None, help="default: all 6 backbones")
    ap.add_argument("--subsets", nargs="+", default=None, help="default: all subsets for the family")
    ap.add_argument("--runs", nargs="+", default=None,
                    help="DIVA only: explicit checkpoint folder names under ckpt_root, instead of the "
                         "unet_<encoder>_diva_<subset> naming (used by run_skip_ablation.sh, whose runs "
                         "are named skipabl_<encoder>_diva_<subset>_s<n_skips>)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(REPO / "99_evaluation" / "01_simple_segmentation" / "lines_eval_summary.csv"))
    args = ap.parse_args()

    encoders = args.encoders or DEFAULT_ENCODERS
    families = [args.family] if args.family else list(FAMILIES)

    rows = []
    if args.runs:
        if families != ["diva"]:
            raise SystemExit("--runs is DIVA-only; pass --family diva")
        for run in args.runs:
            ms = args.subsets[0] if args.subsets else FAMILIES["diva"]["subsets"][0]
            metrics = eval_diva_one(encoder=run, ms=ms, split=args.split, run=run)
            if metrics is None:
                continue
            row = {"family": "diva", "encoder": run, "subset": ms, **metrics}
            rows.append(row)
            print(f"diva    {run:44s} {ms:12s}  " +
                  "  ".join(f"{k}={row.get(k, float('nan')):.3f}" for k in _METRIC_KEYS))
        _write(rows, args.out)
        return

    arunet = None
    if "udiads" in families:
        arunet_pb = REPO / "80_models" / "01_simple_segmentation" / "u-diads-tl" / "pretrained" / "arunet" / "model100_ema.pb"
        arunet = ARUNetWrapper(str(arunet_pb), device="cpu", scale=0.33)

    for family in families:
        subsets = args.subsets or FAMILIES[family]["subsets"]
        for ms in subsets:
            for enc in encoders:
                if family == "diva":
                    metrics = eval_diva_one(enc, ms, args.split)
                else:
                    metrics = eval_udiads_one(enc, ms, args.split, arunet)
                if metrics is None:
                    continue
                row = {"family": family, "encoder": enc, "subset": ms, **metrics}
                rows.append(row)
                print(f"{family:7s} {enc:38s} {ms:12s}  " +
                      "  ".join(f"{k}={row.get(k, float('nan')):.3f}" for k in _METRIC_KEYS))

    _write(rows, args.out)


def _write(rows: list[dict], out_path: str) -> None:
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + "=" * 100)
    print("LINE SEGMENTATION EVALUATION -- DIVA-HisDB (Java evaluator) + U-DIADS-TL (Zottin), common 5-tuple")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
