import argparse
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data import get_splits, select_labeled_pages
from data.diva_dataset import _resolve_layout
from models import build_model


def _resolve_data_root(data_root: str) -> str:
    path = Path(data_root)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parents[1] / path).resolve())


def dice_loss(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    inter = (prob * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (prob.sum() + target.sum() + smooth)


def seg_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dont_care: torch.Tensor,
) -> torch.Tensor:
    valid = ~dont_care
    logits_v = logits[valid]
    mask_v = mask[valid]
    if logits_v.numel() == 0:
        return logits.sum() * 0.0

    bce = F.binary_cross_entropy_with_logits(logits_v, mask_v)
    dice = dice_loss(torch.sigmoid(logits_v), mask_v)
    return bce + dice


def boundary_loss(logits: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    boundary_mask = boundary.bool()
    if not boundary_mask.any():
        return logits.sum() * 0.0

    logits_b = logits[boundary_mask]
    target_b = torch.zeros_like(logits_b)
    return F.binary_cross_entropy_with_logits(logits_b, target_b)


def pixel_iou(logits: torch.Tensor, mask: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits.detach()) > 0.5).long()
    target = mask.long()
    inter = (pred & target).sum().item()
    union = (pred | target).sum().item()
    return inter / (union + 1e-6)


_VIS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_VIS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _unnormalise(img_t: torch.Tensor) -> np.ndarray:
    img = img_t.cpu().float().permute(1, 2, 0).numpy()
    img = img * _VIS_STD + _VIS_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


@torch.no_grad()
def save_visualisations(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    crop_size: int,
    vis_dir: str,
    epoch: int,
    n_images: int = 4,
) -> None:
    model.eval()
    os.makedirs(vis_dir, exist_ok=True)

    for index, (img, mask, _dc, _bnd, _sp) in enumerate(val_loader):
        if index >= n_images:
            break

        img = img.to(device)
        mask = mask.to(device)

        img_r = F.interpolate(img, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        mask_r = F.interpolate(mask, size=(crop_size, crop_size), mode="nearest")
        prob = model.predict(img_r)

        img_np = _unnormalise(img_r[0])
        gt_np = (mask_r[0, 0].cpu().numpy() * 255).astype(np.uint8)
        prob_np = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
        pred_np = ((prob[0, 0].cpu().numpy() > 0.5) * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(prob_np, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = img_np.copy()
        pred_mask_3c = np.stack([np.zeros_like(pred_np), pred_np, np.zeros_like(pred_np)], axis=-1)
        overlay = np.clip(overlay.astype(np.int32) + pred_mask_3c // 3, 0, 255).astype(np.uint8)

        gt_rgb = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        gt_rgb[:, :, 1] = gt_np

        separator = np.ones((crop_size, 4, 3), dtype=np.uint8) * 80
        panel = np.concatenate([img_np, separator, gt_rgb, separator, heatmap, separator, overlay], axis=1)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        cv2.putText(
            panel_bgr,
            f"Epoch {epoch:03d}  |  img  |  GT  |  prob  |  pred",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(vis_dir, f"epoch_{epoch:03d}_img{index:02d}.png"), panel_bgr)

    model.train()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    lambda_boundary: float,
    use_amp: bool,
) -> dict:
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    totals = {"loss": 0.0, "loss_seg": 0.0, "loss_bnd": 0.0, "iou": 0.0}
    n = 0

    for img_l, mask_l, dc_l, bnd_l, _sp_l in loader:
        img_l = img_l.to(device)
        mask_l = mask_l.to(device)
        dc_l = dc_l.to(device)
        bnd_l = bnd_l.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            logits_l = model(img_l)
            loss_seg_v = seg_loss(logits_l, mask_l, dc_l)
            loss_bnd_v = boundary_loss(logits_l, bnd_l)
            loss = loss_seg_v + lambda_boundary * loss_bnd_v

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = img_l.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["loss_seg"] += loss_seg_v.item() * batch_size
        totals["loss_bnd"] += loss_bnd_v.item() * batch_size
        totals["iou"] += pixel_iou(logits_l, mask_l) * batch_size
        n += batch_size

    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    crop_size: int,
) -> dict:
    model.eval()
    total_iou = 0.0
    n = 0
    stride = crop_size // 2

    for img, mask, _dc, _bnd, _sp in loader:
        img = img.to(device)
        mask = mask.to(device)

        _, _, height, width = img.shape
        pad_h = max(0, crop_size - height)
        pad_w = max(0, crop_size - width)
        if pad_h or pad_w:
            img_p = F.pad(img, (0, pad_w, 0, pad_h))
        else:
            img_p = img
        _, _, h_p, w_p = img_p.shape

        prob_sum = torch.zeros((1, 1, h_p, w_p), device=device)
        count_map = torch.zeros((1, 1, h_p, w_p), device=device)

        ys = list(range(0, max(1, h_p - crop_size + 1), stride))
        xs = list(range(0, max(1, w_p - crop_size + 1), stride))
        if h_p > crop_size and ys[-1] + crop_size < h_p:
            ys.append(h_p - crop_size)
        if w_p > crop_size and xs[-1] + crop_size < w_p:
            xs.append(w_p - crop_size)

        for y0 in ys:
            for x0 in xs:
                patch = img_p[:, :, y0:y0 + crop_size, x0:x0 + crop_size]
                prob = model.predict(patch)
                prob_sum[:, :, y0:y0 + crop_size, x0:x0 + crop_size] += prob
                count_map[:, :, y0:y0 + crop_size, x0:x0 + crop_size] += 1.0

        prob_full = (prob_sum / count_map.clamp(min=1e-6))[:, :, :height, :width]
        logits = torch.logit(prob_full.clamp(1e-6, 1 - 1e-6))

        total_iou += pixel_iou(logits, mask) * img.size(0)
        n += img.size(0)

    return {"iou": total_iou / n}


def main(cfg_path: str, overrides: dict | None = None, run_name_override: str | None = None,
         out_dir_override: str | None = None):
    with open(cfg_path) as handle:
        cfg = yaml.safe_load(handle)
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            current = cfg
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value

    seed = cfg.get("training", {}).get("seed")
    if seed is not None:
        # Opt-in only: unset (the default, and what every earlier run used) leaves
        # torch's global RNG untouched so existing recipes reproduce as before.
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Device: {device}  ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"Device: {device}  (no CUDA found)")

    dcfg = cfg["data"]
    tcfg = cfg["training"]
    manuscript = dcfg["manuscript"]
    dataset_family = dcfg.get("dataset_family", "diva")
    data_root = _resolve_data_root(dcfg["data_root"])

    split_dirs = _resolve_layout(data_root, manuscript, dataset_family)
    base_img_train = split_dirs["train"]["img"]
    repo_root = Path(__file__).resolve().parents[2]
    precomp_dir = os.path.join(str(repo_root), "71_misc")
    precomp = None
    if dataset_family == "diva":
        candidate = os.path.join(
            precomp_dir,
            f"diva_{manuscript.lower()}_{dcfg['k_shot']}_diverse_images.txt",
        )
        if os.path.exists(candidate):
            precomp = candidate
    labeled_stems = select_labeled_pages(
        img_dir=base_img_train,
        k=dcfg["k_shot"],
        method=dcfg.get("selection_method", "grayscale_variance"),
        precomputed_path=precomp,
    )
    print(f"[k={dcfg['k_shot']}] Labeled pages: {labeled_stems}")

    _train_paired, val_ds, _test_ds, train_l = get_splits(
        data_root=data_root,
        manuscript=manuscript,
        labeled_stems=labeled_stems,
        dataset_family=dataset_family,
        crop_size=tcfg["crop_size"],
        sv_n_segments=0,   # supervoxel loss removed -> no superpixel maps needed
    )

    train_loader = DataLoader(
        train_l,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=len(train_l) >= tcfg["batch_size"],
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")
    print(f"Model: {model.creator_name}  (skips kept: {getattr(model, 'n_skips', 'n/a')})")

    backbone_params = list(model.backbone.parameters())
    other_params = [parameter for parameter in model.parameters() if not any(parameter is bp for bp in backbone_params)]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": tcfg.get("lr_backbone", 1e-5)},
            {"params": other_params, "lr": tcfg["lr"]},
        ],
        weight_decay=tcfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    scaler = GradScaler("cuda", enabled=tcfg.get("amp", True) and device.type == "cuda")

    project_root = str(repo_root)
    run_name = run_name_override or f"simple_segmentation_{dataset_family}_{manuscript}_{os.path.splitext(os.path.basename(cfg_path))[0]}_k{dcfg['k_shot']}"
    if out_dir_override:
        # --out-dir: parent dir (abs, or repo-relative) that holds the run_name folder
        parent = out_dir_override if os.path.isabs(out_dir_override) else os.path.join(project_root, out_dir_override)
        ckpt_dir = os.path.join(parent, run_name)
    else:
        ckpt_dir = os.path.join(
            project_root,
            "80_models",
            "01_simple_segmentation",
            "segmentation",
            "simple_segmentation",
            run_name,
        )
    os.makedirs(ckpt_dir, exist_ok=True)

    vis_every = int(tcfg.get("vis_every", 0) or 0)
    vis_dir = os.path.join(ckpt_dir, "visualisations")
    best_iou = 0.0
    patience = int(tcfg.get("early_stopping_patience", 0) or 0)
    epochs_since_improvement = 0

    last_epoch = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        last_epoch = epoch
        start = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            lambda_boundary=tcfg.get("lambda_boundary", cfg.get("supervoxel", {}).get("lambda_boundary", 0.0)),
            use_amp=tcfg.get("amp", True),
        )
        val_metrics = validate(model, val_loader, device, tcfg["crop_size"])
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']} [{elapsed:.0f}s]  "
            f"loss={train_metrics['loss']:.4f}  "
            f"seg={train_metrics['loss_seg']:.4f}  "
            f"bnd={train_metrics['loss_bnd']:.4f}  "
            f"train_iou={train_metrics['iou']:.3f}  "
            f"val_iou={val_metrics['iou']:.3f}"
        )

        if vis_every > 0 and (epoch % vis_every == 0 or epoch == 1):
            save_visualisations(model, val_loader, device, tcfg["crop_size"], vis_dir, epoch)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            epochs_since_improvement = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_iou": best_iou, "cfg": cfg},
                os.path.join(ckpt_dir, "best.pth"),
            )
            print(f"  Saved best checkpoint (val_iou={best_iou:.4f})")
        else:
            epochs_since_improvement += 1
            if patience > 0 and epochs_since_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    torch.save(
        {"epoch": last_epoch, "state_dict": model.state_dict(), "cfg": cfg},
        os.path.join(ckpt_dir, "final.pth"),
    )
    print(f"Training complete. Best val IoU = {best_iou:.4f}")
    print(f"Checkpoint dir: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--k-shot", type=int, default=None, dest="k_shot")
    parser.add_argument(
        "--k-shot-method",
        default=None,
        dest="k_shot_method",
        choices=["grayscale_variance", "random", "pca_max_distance", "pca_centroid", "ica_max_distance", "ica_centroid"],
    )
    parser.add_argument("--k-shot-precomputed", default=None, dest="k_shot_precomputed")
    parser.add_argument("--k-shot-seed", type=int, default=None, dest="k_shot_seed")
    parser.add_argument("--manuscript", type=str, default=None)
    parser.add_argument("--arch", type=str, default=None, choices=["unet", "segformer"],
                        help="segmentation architecture (overrides model.arch)")
    parser.add_argument("--encoder", type=str, default=None,
                        help="encoder/backbone name (overrides model.encoder_name)")
    parser.add_argument("--n-skips", type=int, default=None, dest="n_skips", choices=[0, 1, 2, 3, 4],
                        help="skip-connection ablation: keep only the N deepest U-Net skips and zero "
                             "the higher-resolution ones (4 = untouched baseline). Param count unchanged.")
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                        help="overrides training.batch_size (lower for big transformer backbones)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="overrides training.epochs")
    parser.add_argument("--patience", type=int, default=None,
                        help="overrides training.early_stopping_patience (0 disables it)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed python/numpy/torch RNGs (default: unseeded, as in all earlier runs)")
    parser.add_argument("--lr", type=float, default=None, help="overrides training.lr")
    parser.add_argument("--lr-backbone", type=float, default=None, dest="lr_backbone",
                        help="overrides training.lr_backbone")
    parser.add_argument("--weight-decay", type=float, default=None, dest="weight_decay",
                        help="overrides training.weight_decay")
    parser.add_argument("--lambda-boundary", type=float, default=None, dest="lambda_boundary",
                        help="overrides training.lambda_boundary (boundary-loss weight)")
    parser.add_argument("--run-name", type=str, default=None, dest="run_name",
                        help="checkpoint folder name under 80_models/01_simple_segmentation/<dataset>/segmentation/simple_segmentation/")
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir",
                        help="parent dir (abs or repo-relative) to hold the run_name folder; "
                             "overrides the default 80_models/... location")
    args = parser.parse_args()

    overrides = {}
    if args.k_shot is not None:
        overrides["data.k_shot"] = args.k_shot
    if args.k_shot_method is not None:
        overrides["data.selection_method"] = args.k_shot_method
    if args.k_shot_precomputed is not None:
        overrides["data.precomputed_path"] = args.k_shot_precomputed
    if args.k_shot_seed is not None:
        overrides["data.seed"] = args.k_shot_seed
    if args.manuscript is not None:
        overrides["data.manuscript"] = args.manuscript
    if args.arch is not None:
        overrides["model.arch"] = args.arch
    if args.encoder is not None:
        overrides["model.encoder_name"] = args.encoder
    if args.n_skips is not None:
        overrides["model.n_skips"] = args.n_skips
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.epochs is not None:
        overrides["training.epochs"] = args.epochs
    if args.seed is not None:
        overrides["training.seed"] = args.seed
    if args.patience is not None:
        overrides["training.early_stopping_patience"] = args.patience
    if args.lr is not None:
        overrides["training.lr"] = args.lr
    if args.lr_backbone is not None:
        overrides["training.lr_backbone"] = args.lr_backbone
    if args.weight_decay is not None:
        overrides["training.weight_decay"] = args.weight_decay
    if args.lambda_boundary is not None:
        overrides["training.lambda_boundary"] = args.lambda_boundary

    main(args.config, overrides or None, run_name_override=args.run_name, out_dir_override=args.out_dir)
