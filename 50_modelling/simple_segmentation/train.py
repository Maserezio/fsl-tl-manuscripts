import argparse
import os
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
from losses import supervoxel_variance_loss
from models import UnifiedSegmenter


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


def pseudo_loss(logits_u: torch.Tensor, conf_thresh: float = 0.9) -> torch.Tensor:
    prob = torch.sigmoid(logits_u.detach())
    conf_mask = prob.max(1, keepdim=True).values > conf_thresh
    pseudo_mask = (prob > 0.5).float()

    logits_c = logits_u[conf_mask.expand_as(logits_u)]
    mask_c = pseudo_mask[conf_mask.expand_as(logits_u)]
    if logits_c.numel() == 0:
        return logits_u.sum() * 0.0

    bce = F.binary_cross_entropy_with_logits(logits_c, mask_c)
    dice = dice_loss(torch.sigmoid(logits_c), mask_c)
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
    model: UnifiedSegmenter,
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
    model: UnifiedSegmenter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    conf_thresh: float,
    lambda_boundary: float,
    lambda_sv: float,
    lambda_sv_u: float,
    use_amp: bool,
) -> dict:
    model.train()
    amp_enabled = use_amp and device.type == "cuda"
    totals = {"loss": 0.0, "loss_seg": 0.0, "loss_bnd": 0.0, "iou": 0.0}
    if model.use_allspark:
        totals["loss_u"] = 0.0
    else:
        totals["loss_sv_l"] = 0.0
        totals["loss_sv_u"] = 0.0

    n = 0

    for (img_l, mask_l, dc_l, bnd_l, sp_l), (img_u, _mask_u, _dc_u, _bnd_u, sp_u) in loader:
        img_l = img_l.to(device)
        mask_l = mask_l.to(device)
        dc_l = dc_l.to(device)
        bnd_l = bnd_l.to(device)
        sp_l = sp_l.to(device)
        img_u = img_u.to(device)
        sp_u = sp_u.to(device)

        if model.use_allspark:
            model.update_pseudo_labels(img_u)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            if model.use_allspark:
                logits_l, logits_u = model.forward_train(img_l, img_u)
                loss_seg_v = seg_loss(logits_l, mask_l, dc_l)
                loss_bnd_v = boundary_loss(logits_l, bnd_l)
                loss_u_v = pseudo_loss(logits_u, conf_thresh)
                loss = loss_seg_v + lambda_boundary * loss_bnd_v + loss_u_v
            else:
                logits_l = model(img_l)
                logits_u = model(img_u)
                loss_seg_v = seg_loss(logits_l, mask_l, dc_l)
                loss_bnd_v = boundary_loss(logits_l, bnd_l)
                loss_sv_l_v = supervoxel_variance_loss(logits_l, sp_l)
                loss_sv_u_v = supervoxel_variance_loss(logits_u, sp_u)
                loss = loss_seg_v + lambda_boundary * loss_bnd_v + lambda_sv * loss_sv_l_v + lambda_sv_u * loss_sv_u_v

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
        if model.use_allspark:
            totals["loss_u"] += loss_u_v.item() * batch_size
        else:
            totals["loss_sv_l"] += loss_sv_l_v.item() * batch_size
            totals["loss_sv_u"] += loss_sv_u_v.item() * batch_size
        n += batch_size

    return {key: value / n for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: UnifiedSegmenter,
    loader: DataLoader,
    device: torch.device,
    crop_size: int,
) -> dict:
    model.eval()
    total_iou = 0.0
    n = 0

    for img, mask, _dc, _bnd, _sp in loader:
        img = img.to(device)
        mask = mask.to(device)

        img_r = F.interpolate(img, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        mask_r = F.interpolate(mask, size=(crop_size, crop_size), mode="nearest")
        probs = model.predict(img_r)
        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))

        total_iou += pixel_iou(logits, mask_r) * img.size(0)
        n += img.size(0)

    return {"iou": total_iou / n}


def main(cfg_path: str, overrides: dict | None = None):
    with open(cfg_path) as handle:
        cfg = yaml.safe_load(handle)
    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            current = cfg
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"Device: {device}  ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"Device: {device}  (no CUDA found)")

    dcfg = cfg["data"]
    tcfg = cfg["training"]
    acfg = cfg.get("allspark", {})
    svcfg = cfg.get("supervoxel", {})
    use_allspark = acfg.get("enabled", False)
    manuscript = dcfg["manuscript"]

    base_img_train = os.path.join(dcfg["data_root"], manuscript, f"img-{manuscript}", "img", "training")
    repo_root = Path(__file__).resolve().parents[2]
    precomp_dir = os.path.join(str(repo_root), "10_img_preprocessing")
    precomp = os.path.join(precomp_dir, f"diva_{manuscript.lower()}_{dcfg['k_shot']}_diverse_images.txt")
    labeled_stems = select_labeled_pages(
        img_dir=base_img_train,
        k=dcfg["k_shot"],
        method=dcfg.get("selection_method", "grayscale_variance"),
        precomputed_path=precomp if os.path.exists(precomp) else None,
    )
    print(f"[k={dcfg['k_shot']}] Labeled pages: {labeled_stems}")

    train_paired, val_ds, _test_ds, _train_l = get_splits(
        data_root=dcfg["data_root"],
        manuscript=manuscript,
        labeled_stems=labeled_stems,
        crop_size=tcfg["crop_size"],
        sv_n_segments=svcfg.get("n_segments", 0 if use_allspark else 200),
        sv_compactness=svcfg.get("compactness", 10.0),
        sv_sigma=svcfg.get("sigma", 1.0),
    )

    train_loader = DataLoader(
        train_paired,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UnifiedSegmenter(cfg).to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")

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

    project_root = str(Path(__file__).resolve().parent)
    ckpt_dir = os.path.join(
        project_root,
        "checkpoints",
        f"{manuscript}_{os.path.splitext(os.path.basename(cfg_path))[0]}_k{dcfg['k_shot']}",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    vis_every = int(tcfg.get("vis_every", 0) or 0)
    vis_dir = os.path.join(ckpt_dir, "visualisations")
    best_iou = 0.0

    for epoch in range(1, tcfg["epochs"] + 1):
        if use_allspark and epoch == acfg.get("warmup_epoch", 0) + 1:
            model.enable_smem()
            print(f"[Epoch {epoch}] Semantic Memory activated.")

        start = time.time()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            conf_thresh=tcfg.get("conf_threshold", 0.9),
            lambda_boundary=svcfg.get("lambda_boundary", 0.5),
            lambda_sv=svcfg.get("lambda_sv", 0.1),
            lambda_sv_u=svcfg.get("lambda_sv_u", 0.05),
            use_amp=tcfg.get("amp", True),
        )
        val_metrics = validate(model, val_loader, device, tcfg["crop_size"])
        scheduler.step()

        elapsed = time.time() - start
        if use_allspark:
            smem = "S-Mem" if model.using_smem else "batch"
            print(
                f"Epoch {epoch:3d}/{tcfg['epochs']} [{elapsed:.0f}s]  "
                f"loss={train_metrics['loss']:.4f}  "
                f"seg={train_metrics['loss_seg']:.4f}  "
                f"bnd={train_metrics['loss_bnd']:.4f}  "
                f"pseudo={train_metrics['loss_u']:.4f}  "
                f"train_iou={train_metrics['iou']:.3f}  "
                f"val_iou={val_metrics['iou']:.3f}  [{smem}]"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{tcfg['epochs']} [{elapsed:.0f}s]  "
                f"loss={train_metrics['loss']:.4f}  "
                f"seg={train_metrics['loss_seg']:.4f}  "
                f"bnd={train_metrics['loss_bnd']:.4f}  "
                f"sv_l={train_metrics['loss_sv_l']:.4f}  "
                f"sv_u={train_metrics['loss_sv_u']:.4f}  "
                f"train_iou={train_metrics['iou']:.3f}  "
                f"val_iou={val_metrics['iou']:.3f}"
            )

        if vis_every > 0 and (epoch % vis_every == 0 or epoch == 1):
            save_visualisations(model, val_loader, device, tcfg["crop_size"], vis_dir, epoch)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_iou": best_iou, "cfg": cfg},
                os.path.join(ckpt_dir, "best.pth"),
            )
            print(f"  Saved best checkpoint (val_iou={best_iou:.4f})")

    torch.save(
        {"epoch": tcfg["epochs"], "state_dict": model.state_dict(), "cfg": cfg},
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

    main(args.config, overrides or None)
