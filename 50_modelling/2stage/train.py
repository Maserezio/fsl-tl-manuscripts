"""
train.py — unified training entry point for bbox detection (YOLO) and
           text-line segmentation (U-Net / Segformer).

Usage examples
--------------
# Train all YOLO init variants defined in the config
python train.py --task yolo --config configs/yolo_cs863.yaml

# Train segmentation on full CB55 dataset
python train.py --task segm --config configs/segm_cb55.yaml

# Train segmentation in FSL mode — inline page selection, one model per k value
python train.py --task segm --config configs/segm_cb55.yaml --mode fsl
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from dataset import TextLineSegDataset, parse_xml_polygons, polygon_to_bbox
from few_shot_sampler import select_labeled_pages


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def loss_fn(logits: torch.Tensor, targets: torch.Tensor):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return 0.5 * bce + 0.5 * dice_loss(logits, targets)


def otsu_batch(probs: torch.Tensor) -> torch.Tensor:
    probs_np = probs.detach().cpu().numpy()[:, 0]
    bin_masks = np.zeros_like(probs_np, dtype=np.uint8)
    for i in range(probs_np.shape[0]):
        p = (probs_np[i] * 255).astype(np.uint8)
        _, bin_masks[i] = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return torch.from_numpy(bin_masks).unsqueeze(1).to(probs.device).float() / 255.0


# ---------------------------------------------------------------------------
# YOLO training
# ---------------------------------------------------------------------------

def train_yolo(cfg: dict):
    from ultralytics import YOLO

    train_args = {k: cfg[k] for k in ("data", "epochs", "batch", "imgsz", "lr0", "patience")
                  if k in cfg}

    results = {}
    for name, weights in cfg["variants"].items():
        print(f"\n{'='*60}\nTraining YOLO variant: {name}\n{'='*60}")
        out_dir = os.path.join(cfg.get("out_dir", "../71_misc/runs"), f"yolo_{name}")
        metrics = YOLO(weights).train(**train_args, project=out_dir, name="exp")
        results[name] = metrics

    rows = []
    for name, m in results.items():
        rows.append({
            "model":     name,
            "mAP50":     round(m.box.map50, 4),
            "mAP50-95":  round(m.box.map,   4),
            "precision": round(m.box.mp,     4),
            "recall":    round(m.box.mr,     4),
        })
    df = pd.DataFrame(rows).set_index("model").sort_values("mAP50-95", ascending=False)
    print("\n" + "="*60)
    print("YOLO variant comparison")
    print("="*60)
    print(df.to_string())

    out_dir = cfg.get("out_dir", "../71_misc/runs")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "yolo_comparison.csv"))
    print(f"\nSaved comparison to {out_dir}/yolo_comparison.csv")


# ---------------------------------------------------------------------------
# Segmentation — single training run
# ---------------------------------------------------------------------------

def build_segm_model(cfg: dict, device: str):
    arch = cfg.get("architecture", "unet").lower()
    build_fn = getattr(smp, arch.capitalize() if arch != "segformer" else "Segformer",
                       smp.Unet)
    return build_fn(
        encoder_name=cfg.get("encoder", "resnet34"),
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_channels=3,
        classes=1,
        decoder_dropout=cfg.get("dropout", 0.1),
    ).to(device)


def run_segm_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_dice = 0.0, 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            bin_masks = otsu_batch(probs)
            inter = (bin_masks * masks).sum(dim=(2, 3))
            union = bin_masks.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            total_dice += ((2 * inter + 1e-6) / (union + 1e-6)).mean().item()

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


def train_segm_full(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize = tuple(cfg["resize"])
    pad    = cfg.get("pad", 15)
    epochs = cfg.get("epochs", 20)
    batch  = cfg.get("batch", 16)
    lr     = cfg.get("lr", 1e-3)
    out_dir = cfg.get("out_dir", "../71_misc/runs/segm")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {device}")
    train_ds = TextLineSegDataset(cfg["train_img_dir"], cfg["train_xml_dir"], pad=pad, resize=resize)
    val_ds   = TextLineSegDataset(cfg["val_img_dir"],   cfg["val_xml_dir"],   pad=pad, resize=resize)
    test_ds  = TextLineSegDataset(cfg["test_img_dir"],  cfg["test_xml_dir"],  pad=pad, resize=resize)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4)

    model     = build_segm_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                            factor=0.5, patience=3)

    best_dice = float("-inf")
    ckpt_path = os.path.join(out_dir, "best.pth")

    for epoch in range(epochs):
        train_loss, _ = run_segm_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_dice = run_segm_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_dice)
        lr_cur = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{epochs} | lr={lr_cur:.2e} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({"model_state_dict": model.state_dict(),
                        "best_val_dice": best_dice}, ckpt_path)
            print(f"  -> saved best model (val_dice={best_dice:.4f})")

    print(f"\nTraining done. Best val dice: {best_dice:.4f}")
    print(f"Checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Segmentation — FSL mode (one model per k value, inline page selection)
# ---------------------------------------------------------------------------

def train_segm_fsl(cfg: dict):
    """Train one segmentation model per k-shot value using select_labeled_pages().

    Config keys used (in addition to the full-dataset keys):
        fsl_k_values        : list of k values, e.g. [1, 2, 5, 10]
        fsl_method          : selection method (default: grayscale_variance)
        fsl_precomputed     : path to pre-computed diversity file (optional)
        fsl_seed            : random seed for method=random (default: 42)
    """
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    resize  = tuple(cfg["resize"])
    pad     = cfg.get("pad", 15)
    epochs  = cfg.get("epochs", 20)
    batch   = cfg.get("batch", 8)
    lr      = cfg.get("lr", 1e-3)
    out_dir = cfg.get("out_dir", "../71_misc/runs/segm_fsl")
    os.makedirs(out_dir, exist_ok=True)

    k_values   = cfg.get("fsl_k_values", [1, 2, 5, 10])
    method     = cfg.get("fsl_method", "grayscale_variance")
    precomp    = cfg.get("fsl_precomputed", None)
    seed       = cfg.get("fsl_seed", 42)

    # Full val/test datasets stay fixed across all k runs
    val_ds  = TextLineSegDataset(cfg["val_img_dir"],  cfg["val_xml_dir"],  pad=pad, resize=resize)
    test_ds = TextLineSegDataset(cfg["test_img_dir"], cfg["test_xml_dir"], pad=pad, resize=resize)
    val_loader  = DataLoader(val_ds,  batch_size=batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=2)

    results_rows = []

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"FSL  k={k}  method={method}")
        print(f"{'='*60}")

        labeled_stems = select_labeled_pages(
            img_dir=cfg["train_img_dir"],
            k=k,
            method=method,
            precomputed_path=precomp,
            seed=seed,
        )
        print(f"Selected pages: {labeled_stems}")

        # Build train dataset from only the selected pages
        train_ds = TextLineSegDataset(
            cfg["train_img_dir"], cfg["train_xml_dir"],
            pad=pad, resize=resize,
            include_stems=set(labeled_stems),
        )
        print(f"Train crops: {len(train_ds)}  Val crops: {len(val_ds)}  Test crops: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)

        model     = build_segm_model(cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss, _ = run_segm_epoch(model, train_loader, optimizer, device, train=True)
            val_loss, val_dice = run_segm_epoch(model, val_loader, optimizer, device, train=False)
            print(f"  epoch {epoch+1:02d} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        # Evaluate on test set
        model.eval()
        ious = []
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                probs = torch.sigmoid(model(imgs))
                bin_masks = otsu_batch(probs)
                inter = (bin_masks * masks).sum(dim=(2, 3))
                union = bin_masks.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                ious.append(((2 * inter + 1e-6) / (union + 1e-6)).mean().item())

        mean_iou = float(np.mean(ious)) if ious else 0.0
        results_rows.append({"Method": method, "Shots": k, "MeanIoU": mean_iou})

        save_dir = os.path.join(out_dir, f"{method}_{k}shot")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        print(f"  -> saved {save_dir}/model.pth  (mean_iou={mean_iou:.4f})")

    df = pd.DataFrame(results_rows)
    print("\nFSL results:")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "fsl_results.csv"), index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",   choices=["yolo", "segm"], required=True)
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--mode",   choices=["full", "fsl"], default="full",
                   help="Segmentation mode: full dataset or FSL splits (default: full)")
    # k-shot overrides (apply on top of config values when --mode fsl)
    p.add_argument("--k-shot",             type=int,   default=None, dest="k_shot",
                   help="Single k value; overrides fsl_k_values in config")
    p.add_argument("--k-shot-method",      default=None, dest="k_shot_method",
                   choices=["grayscale_variance", "random",
                            "pca_max_distance", "pca_centroid",
                            "ica_max_distance", "ica_centroid"])
    p.add_argument("--k-shot-precomputed", default=None, dest="k_shot_precomputed",
                   help="Path to pre-computed diversity file")
    p.add_argument("--k-shot-seed",        type=int, default=None, dest="k_shot_seed")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI k-shot overrides onto config
    if args.k_shot is not None:
        cfg["fsl_k_values"] = [args.k_shot]
    if args.k_shot_method is not None:
        cfg["fsl_method"] = args.k_shot_method
    if args.k_shot_precomputed is not None:
        cfg["fsl_precomputed"] = args.k_shot_precomputed
    if args.k_shot_seed is not None:
        cfg["fsl_seed"] = args.k_shot_seed

    if args.task == "yolo":
        train_yolo(cfg)
    elif args.task == "segm":
        if args.mode == "fsl":
            train_segm_fsl(cfg)
        else:
            train_segm_full(cfg)


if __name__ == "__main__":
    main()
