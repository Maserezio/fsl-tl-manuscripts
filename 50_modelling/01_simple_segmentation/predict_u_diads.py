"""Cache sliding-window prob maps for the best U-DIADS-TL checkpoints.

For each manuscript, loads best.pth, runs sliding-window inference on every
image in the chosen split, saves:
  - prob map  : <out>/<manuscript>/<split>/prob/<stem>.npy   (float32, [H, W])
  - 8-bit png : <out>/<manuscript>/<split>/prob_png/<stem>.png  (0..255)

No metrics. Apply morphological closing + IoU/HSCP downstream on the cached
arrays — that's seconds per kernel size instead of re-running inference.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.diva_dataset import _resolve_image_path, _resolve_layout
from evaluate import sliding_window_inference
from models import build_model


REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_TEMPLATE = (
    REPO_ROOT
    / "80_models"
    / "01_simple_segmentation"
    / "u-diads-tl"
    / "segmentation"
    / "simple_segmentation"
    / "simple_segmentation_u_diads_{manuscript}_unet_resnet_u_diads_k3"
    / "best.pth"
)
DEFAULT_OUT = REPO_ROOT / "99_evaluation" / "01_simple_segmentation" / "u-diads-tl" / "pred_simple_unet_u_diads"


def _resolve_data_root(data_root: str) -> str:
    path = Path(data_root)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parents[1] / path).resolve())


def predict_manuscript(manuscript: str, split: str, out_root: Path) -> None:
    ckpt_path = Path(str(CKPT_TEMPLATE).format(manuscript=manuscript))
    if not ckpt_path.exists():
        print(f"[SKIP] {manuscript}: checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_root = _resolve_data_root(cfg["data"]["data_root"])
    split_dirs = _resolve_layout(
        data_root, manuscript, cfg["data"].get("dataset_family", "u_diads")
    )
    img_dir = split_dirs[split]["img"]

    out_dir = out_root / manuscript / split
    npy_dir = out_dir / "prob"
    png_dir = out_dir / "prob_png"
    otsu_dir = out_dir / "otsu"
    masked_npy_dir = out_dir / "prob_otsu"
    masked_png_dir = out_dir / "prob_otsu_png"
    for d in (npy_dir, png_dir, otsu_dir, masked_npy_dir, masked_png_dir):
        d.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    )

    print(
        f"\n=== {manuscript} ({split}) — ckpt epoch {ckpt.get('epoch', '?')}, "
        f"val_iou {ckpt.get('val_iou', 0.0):.4f} — {len(img_files)} pages ==="
    )

    for filename in img_files:
        stem = os.path.splitext(filename)[0]
        img_path = _resolve_image_path(img_dir, stem)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cached = npy_dir / f"{stem}.npy"
        if cached.exists():
            prob_map = np.load(cached).astype(np.float32)
        else:
            prob_map = sliding_window_inference(
                model,
                img_rgb,
                crop_size=cfg["training"]["crop_size"],
                device=device,
                use_amp=cfg["training"].get("amp", True),
            ).astype(np.float32)

        np.save(npy_dir / f"{stem}.npy", prob_map)
        png = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(png_dir / f"{stem}.png"), png)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = (otsu > 0).astype(np.float32)
        prob_masked = prob_map * ink

        cv2.imwrite(str(otsu_dir / f"{stem}.png"), otsu)
        np.save(masked_npy_dir / f"{stem}.npy", prob_masked.astype(np.float32))
        masked_png = np.clip(prob_masked * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(masked_png_dir / f"{stem}.png"), masked_png)

        print(f"  {stem}: prob shape {prob_map.shape} -> saved (+ otsu mask)")

    print(f"Output dir: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manuscripts",
        nargs="+",
        default=["Latin14396", "Latin2", "Syr341"],
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for ms in args.manuscripts:
        predict_manuscript(ms, args.split, out_root)


if __name__ == "__main__":
    main()
