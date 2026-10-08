#!/usr/bin/env python3
"""Loss-only stage-2 crop-segmentation ablation on one U-DIADS-TL subset.

Each U-DIADS binary GT connected component is one text-line instance.  A crop
is built around exactly that component, padded in native page coordinates, and
then resized to 1024x256.  BCE, Tversky, and SuperVoxel arms share every other
choice, including a ten-epoch BCE warm-up and a deterministic seed reset.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import TverskyLoss
from supervoxel_loss.loss import SuperVoxelLoss2D
from dataset import parse_xml_polygons, polygon_to_bbox
from torch.utils.data import DataLoader, Dataset, get_worker_info

# Python >=3.11 no longer lets random.sample consume a set.  Upstream
# supervoxel-loss imports sample directly, so patch that local alias only.
import supervoxel_loss.critical_detection_2d as _sv_critical_2d


def _sample_sequence(population, k):
    return random.sample(tuple(population), k)


_sv_critical_2d.sample = _sample_sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
# Shared few-shot page selector, so this segmenter, the detector and the one-stage
# U-Net all pick the identical pages for a given k.
sys.path.insert(0, str(REPO / "50_modelling/common"))
from few_shot_sampler import select_labeled_pages  # noqa: E402
# Two families, one training loop. They differ only in where the per-line
# supervision comes from: U-DIADS ships an instance mask PNG, DIVA ships PAGE
# polygons, so each gets its own Dataset and its own split-directory layout.
FAMILIES = {
    "udiads": dict(
        data=REPO / "00_data" / "U-DIADS-TL",
        models=REPO / "80_models/instance_segmentation/rtdetr_bbox_unet" / "u-diads-tl",
        subsets=("Latin14396", "Latin2", "Syr341"),
    ),
    "diva": dict(
        data=REPO / "00_data" / "DIVA-HisDB",
        models=REPO / "80_models/instance_segmentation/rtdetr_bbox_unet" / "diva-hisdb",
        subsets=("CB55", "CS18", "CS863"),
    ),
    # CATMuS Medieval line polygons for pretraining the BBox U-Net (pages resized to a long side of 2400 px,
    # one PAGE XML per page next to its image, see make_colab_catmus_pretrain.py)
    "catmus": dict(
        data=REPO / "00_data" / "CATMuS" / "crops_pagexml",
        models=REPO / "80_models/instance_segmentation/rtdetr_bbox_unet" / "catmus_pretrain",
        subsets=("CATMuS",),
    ),
    "pinkas": dict(
        data=REPO.parent / "RQ_3_datasets" / "Pinkas" / "pinkas_dataset_images_and_xmls",
        models=REPO / "80_models/instance_segmentation/rtdetr_bbox_unet" / "external-pagexml" / "pinkas",
        subsets=("Pinkas",),
    ),
}

LOSS_ARMS = ("bce", "tversky", "supervoxel")
ARMS = LOSS_ARMS
# Candidate directory names per split, resolved independently for images and for
# masks. U-DIADS is not consistent: Latin14396 keeps images in img-*/train but
# masks in text-line-gt-*/training, while Latin2 and Syr341 use "training" on both
# sides. The previous fixed (image_name, mask_name) pair matched only Latin14396
# and made the other two subsets die with FileNotFoundError on img-*/train.
SPLIT_NAMES = {
    "train": ("train", "training"),
    "val": ("val", "validation"),
}


_DIVA_SPLITS = {"train": ("training",), "val": ("validation",)}


def _diva_split(base, split):
    for name in _DIVA_SPLITS[split]:
        if (base / name).is_dir():
            return base / name
    raise FileNotFoundError(f"no {split} directory under {base}: tried {_DIVA_SPLITS[split]}")


def resolve_split_dir(base, split):
    for name in SPLIT_NAMES[split]:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"no {split} directory under {base}: tried {SPLIT_NAMES[split]}")

TRAIN_AUG = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent=(-0.02, 0.02),
            rotate=(-2, 2),
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            fill_mask=0,
            p=0.5,
        ),
        A.GaussNoise(p=0.2),
    ]
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@lru_cache(maxsize=32)
def _load_page(image_path: str, mask_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(mask_path)
    binary = (gray > 0).astype(np.uint8)
    _, labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return image, labels


class ComponentCropDataset(Dataset):
    """Lazy crops where one U-DIADS connected component is the only foreground."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        crop_w: int,
        crop_h: int,
        pad: int,
        min_area: int,
        augment: bool,
    ):
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.pad = pad
        self.augment = augment
        self.aug = TRAIN_AUG if augment else None
        self.samples = []
        # Foreground area per sample. Exposed by name because save_results needs it and
        # the two families' sample tuples do not share a layout.
        self.areas = []

        image_by_stem = {
            p.stem: p
            for p in image_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        }
        for stem, image_path in sorted(image_by_stem.items()):
            mask_path = mask_dir / f"{stem}.png"
            if not mask_path.exists():
                continue
            gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            n, _, stats, _ = cv2.connectedComponentsWithStats(
                (gray > 0).astype(np.uint8), connectivity=8
            )
            for label_id in range(1, n):
                area = int(stats[label_id, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                x = int(stats[label_id, cv2.CC_STAT_LEFT])
                y = int(stats[label_id, cv2.CC_STAT_TOP])
                w = int(stats[label_id, cv2.CC_STAT_WIDTH])
                h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                self.samples.append(
                    (str(image_path), str(mask_path), label_id, (x, y, x + w, y + h), area, stem)
                )
                self.areas.append(int(area))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label_id, bbox, area, stem = self.samples[index]
        image, labels = _load_page(image_path, mask_path)
        height, width = labels.shape
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - self.pad), max(0, y1 - self.pad)
        x2, y2 = min(width, x2 + self.pad), min(height, y2 + self.pad)
        crop = image[y1:y2, x1:x2]
        target = (labels[y1:y2, x1:x2] == label_id).astype(np.uint8)
        crop = cv2.resize(crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(
            target, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST
        )
        if self.aug is not None:
            transformed = self.aug(image=crop, mask=target)
            crop, target = transformed["image"], transformed["mask"]
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0).float()
        return x, y, index


class PolygonCropDataset(Dataset):
    """Lazy crops where one DIVA PAGE TextLine polygon is the only foreground.

    DIVA supervision is a polygon per line in PAGE XML, not a connected component
    in a mask PNG, so the sample index stores the polygon itself. Otherwise the
    contract matches ComponentCropDataset exactly -- same crop geometry, padding,
    resize interpolation and augmentation -- so both families train through one loop.
    """

    def __init__(self, image_dir, xml_dir, crop_w, crop_h, pad, min_area, augment,
                 include_stems=None):
        self.crop_w, self.crop_h, self.pad = crop_w, crop_h, pad
        self.augment = augment
        self.aug = TRAIN_AUG if augment else None
        self.samples = []
        # Foreground area per sample. Exposed by name because save_results needs it and
        # the two families' sample tuples do not share a layout.
        self.areas = []

        images = {p.stem: p for p in image_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")}
        if include_stems is not None:
            missing = set(include_stems) - set(images)
            if missing:
                raise FileNotFoundError(f"pages not in {image_dir}: {sorted(missing)}")
            images = {s: p for s, p in images.items() if s in include_stems}
        for stem, image_path in sorted(images.items()):
            xml_path = xml_dir / f"{stem}.xml"
            if not xml_path.exists():
                continue
            for poly in parse_xml_polygons(str(xml_path)):
                x1, y1, x2, y2 = polygon_to_bbox(poly)
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue
                self.samples.append((str(image_path), np.asarray(poly, np.int32),
                                     (x1, y1, x2, y2), stem))
                self.areas.append(int(abs(cv2.contourArea(np.asarray(poly, np.int32)))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, poly, bbox, stem = self.samples[index]
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - self.pad), max(0, y1 - self.pad)
        x2, y2 = min(width, x2 + self.pad), min(height, y2 + self.pad)
        crop = image[y1:y2, x1:x2]
        target = np.zeros((y2 - y1, x2 - x1), np.uint8)
        cv2.fillPoly(target, [poly - [x1, y1]], 1)
        crop = cv2.resize(crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST)
        if self.aug is not None:
            t = self.aug(image=crop, mask=target)
            crop, target = t["image"], t["mask"]
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0).float()
        return x, y, index


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info = get_worker_info()
    aug = getattr(info.dataset, "aug", None) if info is not None else None
    if hasattr(aug, "set_random_seed"):
        aug.set_random_seed(worker_seed)


@torch.no_grad()
def evaluate(model, loader, device: str):
    model.eval()
    tp = fp = fn = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        predicted = (torch.sigmoid(model(images)) >= 0.5).cpu().numpy()[:, 0]
        truth = targets.numpy()[:, 0].astype(bool)
        tp += int((predicted & truth).sum())
        fp += int((predicted & ~truth).sum())
        fn += int((~predicted & truth).sum())
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
    }


def build_model(backbone: str, device: str):
    return smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)


def run_arm(arm, args, train_ds, val_ds, output_root, device):
    seed_everything(args.seed)
    if hasattr(train_ds.aug, "set_random_seed"):
        train_ds.aug.set_random_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        drop_last=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        worker_init_fn=seed_worker,
    )

    model = build_model(args.backbone, device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial.get("state_dict", initial))
        print(f"[{arm}] initialized from {args.init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()
    if arm == "bce":
        main_loss = bce
    elif arm == "tversky":
        main_loss = TverskyLoss(
            mode="binary", alpha=0.3, beta=0.7, from_logits=True
        )
    elif arm == "supervoxel":
        main_loss = SuperVoxelLoss2D(
            alpha=args.sv_alpha, beta=args.sv_beta, device=device
        )
    else:
        raise ValueError(arm)

    arm_dir = output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_dice = -1.0
    start_epoch = 1
    last_path = arm_dir / "last.pth"
    history_path = arm_dir / "history.csv"
    if last_path.exists() and history_path.exists() and not args.force:
        last = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(last["state_dict"])
        optimizer.load_state_dict(last["optimizer_state_dict"])
        history = pd.read_csv(history_path).to_dict("records")
        start_epoch = int(last["epoch"]) + 1
        history = [row for row in history if int(row["epoch"]) < start_epoch]
        best_dice = float(last["best_dice"])
        random.setstate(last["python_rng_state"])
        np.random.set_state(last["numpy_rng_state"])
        # map_location=device moved every tensor in the checkpoint onto the GPU,
        # RNG states included, but set_rng_state/set_state require CPU ByteTensors.
        torch.set_rng_state(last["torch_rng_state"].cpu())
        if device == "cuda" and last.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in last["cuda_rng_state"]])
        generator.set_state(last["loader_generator_state"].cpu())
        print(f"[{arm}] resuming at epoch {start_epoch} from {last_path}", flush=True)
    start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        warmup = epoch <= args.warmup_epochs
        loss_fn = bce if warmup else main_loss
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        for batch_index, (images, targets, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if batch_index % 20 == 0 or batch_index == len(train_loader):
                print(
                    f"[{arm:10s}] ep {epoch:02d} train {batch_index:03d}/{len(train_loader):03d} "
                    f"({time.time()-epoch_start:.1f}s)",
                    flush=True,
                )

        # Validation walks the whole val split and costs the same no matter how few
        # training pages there are, so at small k it dominates the run -- measured at
        # 23-41 min per arm with every-epoch validation. Evaluating every Nth epoch
        # (always including the last, and never before the warm-up ends, or the arm
        # could finish with no eligible checkpoint) cuts that proportionally.
        due = (epoch % args.eval_every == 0 or epoch == args.epochs
               or epoch == args.warmup_epochs + 1)
        if due:
            metrics = evaluate(model, val_loader, device)
            metrics.update(
                arm=arm,
                epoch=epoch,
                phase="warmup(bce)" if warmup else arm,
                train_loss=total_loss / max(1, len(train_loader)),
                elapsed_minutes=(time.time() - start) / 60.0,
            )
            history.append(metrics)
            pd.DataFrame(history).to_csv(history_path, index=False)
            print(
                f"[{arm:10s}] ep {epoch:02d}/{args.epochs} ({metrics['phase']:12s}) "
                f"loss {metrics['train_loss']:.4f} | IoU {metrics['iou']:.4f} "
                f"Dice {metrics['dice']:.4f} precision {metrics['precision']:.4f} "
                f"recall {metrics['recall']:.4f}",
                flush=True,
            )
            eligible = arm == "bce" or epoch > args.warmup_epochs
            if eligible and metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                        "arm": arm,
                        "backbone": args.backbone,
                        "resize": [args.crop_w, args.crop_h],
                        "subset": args.subset,
                        "sv_alpha": args.sv_alpha if arm == "supervoxel" else None,
                        "sv_beta": args.sv_beta if arm == "supervoxel" else None,
                    },
                    arm_dir / "best.pth",
                )
        else:
            print(f"[{arm:10s}] ep {epoch:02d}/{args.epochs} "
                  f"loss {total_loss / max(1, len(train_loader)):.4f} (no val)", flush=True)
        last_tmp = arm_dir / "last.pth.tmp"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                "loader_generator_state": generator.get_state(),
            },
            last_tmp,
        )
        last_tmp.replace(last_path)
    return pd.DataFrame(history)


def completed_arm(arm_dir: Path, arm: str, args) -> pd.DataFrame | None:
    history_path, best_path = arm_dir / "history.csv", arm_dir / "best.pth"
    if args.force or not history_path.exists() or not best_path.exists():
        return None
    history = pd.read_csv(history_path).sort_values("epoch")
    # Row count is not the test: with --eval-every N the history holds ~epochs/N rows
    # by design. What matters is that training reached the last epoch.
    if int(history.epoch.max()) < args.epochs:
        return None
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if arm != "bce" and int(checkpoint["epoch"]) <= args.warmup_epochs:
        return None
    return history.iloc[: args.epochs].reset_index(drop=True)


def save_results(histories, val_ds, args, output_root, device):
    all_history = pd.concat(histories.values(), ignore_index=True)
    all_history.to_csv(output_root / "history_all_arms.csv", index=False)
    rows = []
    for arm, history in histories.items():
        eligible = history if arm == "bce" else history[history.epoch > args.warmup_epochs]
        best_index = eligible.dice.idxmax()
        rows.append(
            {
                "arm": arm,
                "sv_alpha": args.sv_alpha if arm == "supervoxel" else None,
                "sv_beta": args.sv_beta if arm == "supervoxel" else None,
                "best_epoch": int(history.loc[best_index, "epoch"]),
                **{
                    f"{metric}@bestDice": float(history.loc[best_index, metric])
                    for metric in ("iou", "dice", "precision", "recall")
                },
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False), flush=True)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    for axis, metric in zip(axes, ("iou", "dice", "recall")):
        for arm, history in histories.items():
            axis.plot(history.epoch, history[metric], label=arm)
        axis.axvline(args.warmup_epochs + 0.5, color="gray", ls="--", lw=1)
        axis.set_title(f"validation {metric}")
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(
        f"{args.family} {args.subset}: U-Net/{args.backbone}, {args.crop_w}x{args.crop_h}"
    )
    figure.tight_layout()
    figure.savefig(output_root / "curves.png", dpi=150)
    plt.close(figure)

    areas = np.asarray(val_ds.areas)
    quantile_targets = np.quantile(areas, [0.25, 0.5, 0.75])
    shown = [int(np.argmin(np.abs(areas - target))) for target in quantile_targets]
    images = []
    targets = []
    for index in shown:
        image, target, _ = val_ds[index]
        images.append(image)
        targets.append(target[0].numpy())
    batch = torch.stack(images).to(device)
    predictions = {}
    for arm in histories:
        checkpoint = torch.load(
            output_root / arm / "best.pth", map_location=device, weights_only=False
        )
        model = build_model(args.backbone, device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            predictions[arm] = (
                torch.sigmoid(model(batch)).cpu().numpy()[:, 0] >= 0.5
            )
        del model

    ncols = 2 + len(ARMS)
    figure, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 5.8))
    for row, index in enumerate(shown):
        panels = [("image", images[row].permute(1, 2, 0).numpy()), ("GT line", targets[row])]
        panels += [(arm, predictions[arm][row]) for arm in ARMS]
        for column, (title, panel) in enumerate(panels):
            axes[row, column].imshow(panel, cmap=None if panel.ndim == 3 else "gray")
            axes[row, column].set_axis_off()
            if row == 0:
                axes[row, column].set_title(title)
    figure.suptitle(f"Same three validation components — {args.subset}")
    figure.tight_layout()
    figure.savefig(output_root / "qualitative_3crops.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="udiads", choices=tuple(FAMILIES))
    parser.add_argument("--subset", default="Latin14396",
                        help="Latin14396|Latin2|Syr341 for udiads, CB55|CS18|CS863 for diva")
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--crop-width", type=int, default=1024, dest="crop_w")
    parser.add_argument("--crop-height", type=int, default=256, dest="crop_h")
    parser.add_argument("--pad", type=int, default=15)
    parser.add_argument("--min-area", type=int, default=50, dest="min_area")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=10, dest="warmup_epochs")
    parser.add_argument("--eval-every", type=int, default=1, dest="eval_every",
                        help="validate every Nth epoch (1 = every epoch, the default "
                             "the loss-ablation runs were trained with)")
    parser.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    parser.add_argument("--val-batch-size", type=int, default=12, dest="val_batch_size")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    # Few-shot: train on K labeled pages instead of the whole train split, selected by
    # the same few_shot_sampler the detector and the one-stage U-Net use, so all three
    # see identical pages at a given k. 0 = every page. Val is never subsetted -- the
    # points of a k-curve have to be selected against one fixed validation set.
    parser.add_argument("--k-shot", type=int, default=0, dest="k_shot")
    parser.add_argument("--k-shot-method", default="grayscale_variance", dest="k_shot_method")
    # PCA/ICA methods live in a file produced by make_shot_selection.py; without it only
    # grayscale_variance and random can be computed on the fly.
    parser.add_argument("--k-shot-precomputed", default=None, dest="k_shot_precomputed")
    parser.add_argument("--k-shot-seed", type=int, default=42, dest="k_shot_seed")
    # Selection method in the output path: two methods at the same k are different runs
    # and must not overwrite one another.
    parser.add_argument("--tag", default="", help="extra path component under the subset")
    parser.add_argument("--arms", nargs="+", default=list(LOSS_ARMS),
                        help=f"loss arms to train (default: all of {list(LOSS_ARMS)})")
    parser.add_argument("--sv-alpha", type=float, default=0.5,
                        help="SuperVoxel topology weight (default: 0.5)")
    parser.add_argument("--sv-beta", type=float, default=0.5,
                        help="SuperVoxel split/merge balance (default: 0.5)")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="optional common model checkpoint before loss fine-tuning")
    return parser.parse_args()


def main():
    global ARMS
    args = parse_args()
    ARMS = tuple(args.arms)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = FAMILIES[args.family]
    if args.subset not in spec["subsets"]:
        raise SystemExit(f"subset {args.subset!r} is not in family {args.family!r} "
                         f"{spec['subsets']}")
    data_root, model_root = spec["data"], spec["models"]
    subset_root = data_root / args.subset
    image_base = subset_root / f"img-{args.subset}"
    # The size belongs in the path: a 1536x128 run used to land in the 1024x256
    # directory, find that history complete and exit without training anything.
    output_root = (model_root / f"crop_seg_loss_ablation_components_{args.crop_w}x{args.crop_h}"
                   / args.subset)
    if args.arms == ["supervoxel"] and (args.sv_alpha, args.sv_beta) != (0.5, 0.5):
        sv_tag = f"a{args.sv_alpha:g}_b{args.sv_beta:g}".replace(".", "p")
        output_root = output_root / "supervoxel_sweep" / sv_tag
    if args.k_shot:
        # Separate tree, or a k-shot run would overwrite the full-data segmenter that
        # the backbone matrix was scored against.
        output_root = (model_root / "crop_seg_kshot_1024x256" / args.subset
                       / (args.tag or "default") / f"k{args.k_shot}")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.family == "diva":
        # DIVA nests the split one level deeper (img-<S>/img/<split>) and supervises
        # from PAGE-gt-<S>-TASK-2/TASK-2/<split>, with only the long split names.
        image_base = image_base / "img"
        label_base = subset_root / f"PAGE-gt-{args.subset}-TASK-2" / "TASK-2"
        split_dir = lambda base, split: _diva_split(base, split)
        make_dataset = PolygonCropDataset
    elif args.family == "catmus":
        image_base = label_base = data_root
        split_dir = lambda base, split: base / split
        make_dataset = PolygonCropDataset
    elif args.family == "pinkas":
        # Pinkas stores paired images and PAGE XML files in one directory without
        # an official split. Keep whole folios together: the first 80% of sorted
        # page stems are training data and the remaining 20% validation data.
        subset_root = data_root
        image_base = label_base = data_root
        stems = sorted(
            p.stem for p in data_root.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            and (data_root / f"{p.stem}.xml").exists()
        )
        if len(stems) < 5:
            raise RuntimeError(f"not enough paired Pinkas pages under {data_root}")
        cut = max(1, min(len(stems) - 1, round(0.8 * len(stems))))
        pinkas_split = {"train": set(stems[:cut]), "val": set(stems[cut:])}
        split_dir = lambda base, split: base
        make_dataset = PolygonCropDataset
    else:
        label_base = subset_root / f"text-line-gt-{args.subset}"
        split_dir = resolve_split_dir
        make_dataset = ComponentCropDataset

    include = {"train": None, "val": None}
    if args.family == "pinkas":
        include = pinkas_split
    if args.k_shot:
        if make_dataset is not PolygonCropDataset:
            raise SystemExit("--k-shot is implemented for --family diva only; "
                             "ComponentCropDataset has no include_stems filter")
        train_images = split_dir(image_base, "train")
        include["train"] = set(select_labeled_pages(
            img_dir=str(train_images), k=args.k_shot, method=args.k_shot_method,
            precomputed_path=args.k_shot_precomputed, seed=args.k_shot_seed))
        if len(include["train"]) != args.k_shot:
            raise RuntimeError(f"asked for {args.k_shot} pages, selector returned "
                               f"{len(include['train'])} from {train_images}")
        print(f"[k={args.k_shot}] Labeled pages: {sorted(include['train'])}", flush=True)

    datasets = {}
    for split in SPLIT_NAMES:
        kwargs = {} if make_dataset is ComponentCropDataset else {
            "include_stems": include[split]}
        datasets[split] = make_dataset(
            split_dir(image_base, split),
            split_dir(label_base, split),
            args.crop_w,
            args.crop_h,
            args.pad,
            args.min_area,
            augment=split == "train",
            **kwargs,
        )
    print(
        f"device={device} subset={args.subset} resize={args.crop_w}x{args.crop_h} "
        f"train_components={len(datasets['train'])} val_components={len(datasets['val'])} "
        f"arms={list(ARMS)} k_shot={args.k_shot or 'all'} output={output_root}",
        flush=True,
    )
    if not datasets["train"] or not datasets["val"]:
        raise RuntimeError("No component crops found")

    histories = {}
    for arm in ARMS:
        saved = completed_arm(output_root / arm, arm, args)
        if saved is not None:
            histories[arm] = saved
            print(f"[{arm}] already complete; loaded saved history", flush=True)
        else:
            histories[arm] = run_arm(
                arm, args, datasets["train"], datasets["val"], output_root, device
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # A targeted rerun such as ``--arms bce`` must not replace the canonical
    # three-row summary with a one-row file. Pull the other already-complete arms
    # into reporting even though they were not requested for training.
    for arm in LOSS_ARMS:
        if arm in histories:
            continue
        force = args.force
        args.force = False
        saved = completed_arm(output_root / arm, arm, args)
        args.force = force
        if saved is not None:
            histories[arm] = saved
    save_results(histories, datasets["val"], args, output_root, device)
    print(f"DONE -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
