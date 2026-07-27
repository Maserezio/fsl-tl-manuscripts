"""Cache raw sliding-window segmentation probability maps for the 6-backbone
DIVA-HisDB + U-DIADS-TL comparison (see ../run_train_all.sh), one .npy per
(encoder, family, subset, page) -- GPU-only, no ARU-Net (CPU) involved, so this
phase runs much faster than a full eval and only needs to pay the inference
cost once. Resumable: skips any page whose .npy already exists.

The unified eval script reads from this cache when present instead of
re-running inference.

Cache layout:
  99_evaluation/01_simple_segmentation/<family>/prob_cache/
    unet_<encoder>_<family>_<subset>/<stem>.npy

Usage:
  python cache_predictions.py                          # all 6 backbones, both families, all subsets
  python cache_predictions.py --family udiads --encoders resnet34
  python cache_predictions.py --family diva --subsets CB55
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
SEG_DIR = HERE.parent  # 01_simple_segmentation/ -- evaluate.py, models/, data/ live here
REPO = HERE.parents[2]
sys.path.insert(0, str(SEG_DIR))

from evaluate import sliding_window_inference   # noqa: E402
from models import build_model                   # noqa: E402
from data.diva_dataset import _resolve_layout      # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

DEFAULT_ENCODERS = [
    "resnet34",
    "resnet50",
    "tu-convnext_tiny.in12k_ft_in1k",
    "tu-convnext_tiny.dinov3_lvd1689m",
    "vit_small_patch16_224.augreg_in21k",
    "vit_small_patch16_dinov3",
]

FAMILIES = {
    "diva": dict(
        data_root="00_data/DIVA-HisDB",
        dataset_family="diva",
        subsets=["CB55", "CS18", "CS863"],
        ckpt_root="80_models/01_simple_segmentation/diva-hisdb/segmentation/simple_segmentation",
        cache_root="99_evaluation/01_simple_segmentation/diva-hisdb/prob_cache",
    ),
    "udiads": dict(
        data_root="00_data/U-DIADS-TL",
        dataset_family="u_diads",
        subsets=["Latin14396", "Latin2", "Syr341"],
        ckpt_root="80_models/01_simple_segmentation/u-diads-tl/segmentation/simple_segmentation",
        cache_root="99_evaluation/01_simple_segmentation/u-diads-tl/prob_cache",
    ),
}


def find_ckpt(ckpt_root: Path, run: str) -> Path:
    p = ckpt_root / run / "best.pth"
    if not p.exists():
        raise FileNotFoundError(f"no checkpoint for {run!r} at {p}")
    return p


def read_rgb(img_dir, stem):
    for e in _EXTS:
        p = Path(img_dir) / f"{stem}{e}"
        if p.exists():
            return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f"{stem} in {img_dir}")


def img_stems(img_dir):
    return sorted({p.stem for p in Path(img_dir).iterdir() if p.suffix.lower() in _EXTS})


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = build_model(ck["cfg"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    crop = ck["cfg"]["training"].get("crop_size", 448)
    return model, crop


def cache_family(family: str, encoders, subsets, split: str) -> None:
    spec = FAMILIES[family]
    data_root = REPO / spec["data_root"]
    ckpt_root = REPO / spec["ckpt_root"]
    cache_root = REPO / spec["cache_root"]

    for ms in subsets:
        layout = _resolve_layout(str(data_root), ms, spec["dataset_family"])[split]
        img_dir = layout["img"]
        stems = img_stems(img_dir)

        for enc in encoders:
            run = f"unet_{enc}_{family}_{ms}"
            cache_dir = cache_root / run
            cache_dir.mkdir(parents=True, exist_ok=True)

            todo = [s for s in stems if not (cache_dir / f"{s}.npy").exists()]
            if not todo:
                print(f"[skip] {run}: all {len(stems)} pages already cached")
                continue

            try:
                ckpt = find_ckpt(ckpt_root, run)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            model, crop = load_model(ckpt)

            for stem in tqdm(todo, desc=run, leave=False):
                img = read_rgb(img_dir, stem)
                seg_prob = sliding_window_inference(model, img, crop, device=DEVICE)
                np.save(cache_dir / f"{stem}.npy", seg_prob.astype(np.float32))
            del model
            torch.cuda.empty_cache()
            print(f"[done] {run}: cached {len(todo)} new pages ({len(stems) - len(todo)} already had cache)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["diva", "udiads"], default=None,
                     help="default: both families")
    ap.add_argument("--encoders", nargs="+", default=None, help="default: all 6 backbones")
    ap.add_argument("--subsets", nargs="+", default=None, help="default: all subsets for the family")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    encoders = args.encoders or DEFAULT_ENCODERS
    families = [args.family] if args.family else list(FAMILIES)
    for family in families:
        subsets = args.subsets or FAMILIES[family]["subsets"]
        cache_family(family, encoders, subsets, args.split)

    print("ALL CACHING DONE")


if __name__ == "__main__":
    main()
