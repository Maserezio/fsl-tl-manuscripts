#!/usr/bin/env python3
"""Fine-tune and evaluate Hugging Face Mask2Former on U-DIADS-TL.

The U-DIADS-TL COCO files contain one polygon instance per connected text
line.  This script rasterizes those polygons into the instance maps expected
by ``Mask2FormerImageProcessor`` and supports the complete three-subset run.

Examples
--------
Smoke test one subset::

    python mask2former_udiads.py run --subsets Latin14396 --epochs 1 \
        --max-train-images 1 --max-eval-images 1 --run-name smoke

Train and test one model per subset::

    python mask2former_udiads.py run --subsets all --epochs 200

Evaluate one existing checkpoint on all subsets::

    python mask2former_udiads.py evaluate --subsets all \
        --checkpoint ../../../80_models/03_Mask2Former/u-diads-tl/Latin14396/m2f_latin14396_swin-t
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, Dataset
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUBSETS = ("Latin14396", "Latin2", "Syr341")
MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_aliases(split: str) -> tuple[str, ...]:
    return {
        "train": ("train", "training"),
        "val": ("val", "validation"),
        "test": ("test",),
    }[split]


def resolve_split_dir(base: Path, split: str) -> Path:
    for alias in split_aliases(split):
        candidate = base / alias
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No {split!r} directory under {base}")


class CocoTextLineDataset(Dataset):
    """Small, dependency-free COCO polygon dataset for U-DIADS-TL."""

    def __init__(
        self,
        data_root: Path,
        subset: str,
        split: str,
        processor: Mask2FormerImageProcessor | None = None,
        shortest_edge: int = 512,
        longest_edge: int = 768,
        mask_label_stride: int = 1,
        max_images: int | None = None,
        horizontal_flip: bool = False,
        vertical_tiles: int = 1,
    ) -> None:
        self.subset = subset
        self.split = split
        self.processor = processor
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge
        self.mask_label_stride = mask_label_stride
        self.horizontal_flip = horizontal_flip
        if vertical_tiles < 1:
            raise ValueError("vertical_tiles must be positive")
        self.vertical_tiles = vertical_tiles

        annotation_path = data_root / f"coco_dataset_{subset.lower()}" / f"{split}.json"
        with annotation_path.open(encoding="utf-8") as handle:
            coco = json.load(handle)

        image_base = data_root / subset / f"img-{subset}"
        self.image_dir = resolve_split_dir(image_base, split)
        self.images = sorted(coco["images"], key=lambda item: item["id"])
        if max_images is not None:
            self.images = self.images[:max_images]

        annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        self.annotations = annotations

    def __len__(self) -> int:
        return len(self.images) * self.vertical_tiles

    def _image_path(self, item: dict) -> Path:
        direct = self.image_dir / item["file_name"]
        if direct.exists():
            return direct
        stem = Path(item["file_name"]).stem
        for extension in EXTENSIONS:
            candidate = self.image_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image {item['file_name']} not found in {self.image_dir}")

    def raw_item(self, index: int) -> tuple[Image.Image, np.ndarray, dict]:
        image_index, tile_index = divmod(index, self.vertical_tiles)
        item = self.images[image_index]
        image = Image.open(self._image_path(item)).convert("RGB")

        # Zero is background and is passed as ignore_index to the processor.
        instance_map = Image.new("I", image.size, color=0)
        draw = ImageDraw.Draw(instance_map)
        instance_to_class: dict[int, int] = {}
        for instance_id, annotation in enumerate(self.annotations[item["id"]], start=1):
            for polygon in annotation.get("segmentation", []):
                if len(polygon) < 6:
                    continue
                points = list(zip(polygon[0::2], polygon[1::2]))
                draw.polygon(points, fill=instance_id)
            instance_to_class[instance_id] = 0  # the sole class: TextLine

        instance_array = np.array(instance_map, dtype=np.int32, copy=True)
        if self.vertical_tiles > 1:
            boundaries = np.linspace(0, image.width, self.vertical_tiles + 1, dtype=int)
            left, right = int(boundaries[tile_index]), int(boundaries[tile_index + 1])
            image = image.crop((left, 0, right, image.height))
            instance_array = np.ascontiguousarray(instance_array[:, left:right])
        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            instance_array = np.ascontiguousarray(instance_array[:, ::-1])
        metadata = {
            "stem": Path(item["file_name"]).stem,
            "height": image.height,
            "width": image.width,
            "instance_to_class": instance_to_class,
        }
        return image, instance_array, metadata

    def __getitem__(self, index: int) -> dict:
        image, instance_map, metadata = self.raw_item(index)
        if self.processor is None:
            return {"image": image, "instance_map": instance_map, "metadata": metadata}

        encoded = self.processor(
            images=image,
            segmentation_maps=instance_map,
            instance_id_to_semantic_id=metadata["instance_to_class"],
            ignore_index=0,
            do_reduce_labels=False,
            size={"shortest_edge": self.shortest_edge, "longest_edge": self.longest_edge},
            return_tensors="pt",
        )
        mask_labels = encoded["mask_labels"][0]
        if self.mask_label_stride > 1:
            # The criterion samples normalized points from target masks, so
            # targets need not occupy the full image resolution. Keeping them
            # at the 1/4 mask-head resolution avoids a ~700 MB [N,H,W] tensor
            # on dense Syr341 pages without changing instance coordinates.
            mask_labels = torch.nn.functional.interpolate(
                mask_labels.unsqueeze(0),
                scale_factor=1.0 / self.mask_label_stride,
                mode="nearest",
                recompute_scale_factor=False,
            )[0]
        return {
            "pixel_values": encoded["pixel_values"][0],
            "pixel_mask": encoded["pixel_mask"][0],
            "mask_labels": mask_labels,
            "class_labels": encoded["class_labels"][0],
            "metadata": metadata,
        }


def collate_mask2former(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "pixel_mask": torch.stack([item["pixel_mask"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def move_training_batch(batch: dict, device: torch.device, mask_label_bf16: bool = False) -> dict:
    mask_dtype = torch.bfloat16 if mask_label_bf16 and device.type == "cuda" else None
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "pixel_mask": batch["pixel_mask"].to(device, non_blocking=True),
        "mask_labels": [
            value.to(device, dtype=mask_dtype, non_blocking=True)
            if mask_dtype is not None
            else value.to(device, non_blocking=True)
            for value in batch["mask_labels"]
        ],
        "class_labels": [value.to(device, non_blocking=True) for value in batch["class_labels"]],
    }


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_finetune_model(
    num_queries: int,
    train_num_points: int = 4096,
    init_checkpoint: Path | None = None,
) -> Mask2FormerForUniversalSegmentation:
    # Load the stock 100 COCO queries first so their learned embeddings survive.
    # Asking from_pretrained for 200 directly would randomly reinitialize every
    # query, including the useful original hundred.
    source = init_checkpoint or MODEL_ID
    config = Mask2FormerConfig.from_pretrained(source)
    config.num_labels = 1
    config.id2label = {0: "TextLine"}
    config.label2id = {"TextLine": 0}
    config.train_num_points = train_num_points
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        source, config=config, ignore_mismatched_sizes=True
    )
    if init_checkpoint is not None:
        print(f"[model] initialized from task checkpoint: {init_checkpoint}")
    query_module = model.model.transformer_module
    old_count, hidden_size = query_module.queries_embedder.weight.shape
    if num_queries != old_count:
        if num_queries < 1:
            raise ValueError("num_queries must be positive")
        old_position = query_module.queries_embedder.weight.detach().clone()
        old_features = query_module.queries_features.weight.detach().clone()
        new_position = torch.nn.Embedding(num_queries, hidden_size)
        new_features = torch.nn.Embedding(num_queries, hidden_size)
        with torch.no_grad():
            keep = min(num_queries, old_count)
            new_position.weight[:keep].copy_(old_position[:keep])
            new_features.weight[:keep].copy_(old_features[:keep])
            # nn.Embedding initializes the extra rows independently with N(0, 1),
            # matching the scale of Mask2Former's learned COCO queries. Copying
            # existing rows here creates q/q+100 twins which remain duplicate
            # text-line predictions even after tens of fine-tuning epochs.
        query_module.queries_embedder = new_position
        query_module.queries_features = new_features
        model.config.num_queries = num_queries
        print(f"[model] expanded learned COCO queries {old_count} -> {num_queries}")
    return model


def load_checkpoint_model(checkpoint: Path) -> Mask2FormerForUniversalSegmentation:
    """Load current checkpoints and repair the extra backbone nesting in legacy ones.

    The pre-existing 5.12.1 checkpoints in ``80_models/03_Mask2Former`` were
    saved after wrapping the Swin model in an extra ``backbone`` module. Their
    config describes ordinary HF Swin, so a vanilla ``from_pretrained`` call
    silently leaves all 227 backbone tensors random. The key transformation
    below is exact: after removing that wrapper and its eight unused projection
    tensors, every saved tensor matches the configured model by name and shape.
    """
    weights_path = checkpoint / "model.safetensors"
    if weights_path.exists():
        state = load_safetensors(weights_path)
        legacy_prefix = "model.pixel_level_module.encoder.backbone."
        if any(key.startswith(legacy_prefix) for key in state):
            config = Mask2FormerConfig.from_pretrained(checkpoint)
            model = Mask2FormerForUniversalSegmentation(config)
            converted = {
                key.replace(legacy_prefix, "model.pixel_level_module.encoder.", 1): value
                for key, value in state.items()
                if ".encoder.proj." not in key
            }
            result = model.load_state_dict(converted, strict=False)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    "Legacy checkpoint conversion was not exact: "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
                )
            print(f"[checkpoint] converted legacy wrapped-backbone keys: {checkpoint}")
            return model
    return Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)


def mean_validation_loss(
    model: Mask2FormerForUniversalSegmentation,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    mask_label_bf16: bool = False,
) -> float:
    model.eval()
    values = []
    with torch.inference_mode():
        for batch in loader:
            inputs = move_training_batch(batch, device, mask_label_bf16)
            with autocast_context(device, amp):
                values.append(float(model(**inputs).loss.detach().cpu()))
            del inputs
    model.train()
    # Dense full-resolution target masks can leave hundreds of MiB in the CUDA
    # caching allocator. Releasing that cache here prevents the first shuffled
    # training page after validation from failing despite fitting in isolation.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.mean(values)) if values else math.nan


def train_subset(args: argparse.Namespace, subset: str, run_name: str) -> Path:
    device = torch.device(args.device)
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_ID)
    train_data = CocoTextLineDataset(
        args.data_root,
        subset,
        "train",
        processor=processor,
        shortest_edge=args.shortest_edge,
        longest_edge=args.longest_edge,
        mask_label_stride=args.mask_label_stride,
        max_images=args.max_train_images,
        horizontal_flip=True,
        vertical_tiles=args.train_vertical_tiles,
    )
    val_data = CocoTextLineDataset(
        args.data_root,
        subset,
        "val",
        processor=processor,
        shortest_edge=args.shortest_edge,
        longest_edge=args.longest_edge,
        mask_label_stride=args.mask_label_stride,
        max_images=args.max_eval_images,
        vertical_tiles=args.train_vertical_tiles,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_mask2former,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_mask2former,
    )

    model = build_finetune_model(
        args.num_queries,
        args.train_num_points,
        init_checkpoint=args.init_checkpoint,
    ).to(device)
    use_fp16_scaler = bool(
        args.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16_scaler)
    print(
        f"[amp] {'fp16 + GradScaler' if use_fp16_scaler else 'bf16' if args.amp and device.type == 'cuda' else 'disabled'}"
    )
    if args.gradient_checkpointing:
        if getattr(model, "supports_gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
        else:
            print("[WARN] This Transformers Mask2Former does not support gradient checkpointing; continuing without it")

    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.learning_rate,
    )
    total_updates = max(1, math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs)
    warmup_updates = int(total_updates * args.warmup_ratio)

    def lr_factor(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(1e-3, step / warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    output_dir = args.model_root / subset / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.csv"
    best_value = math.inf if args.checkpoint_metric == "val_loss" else -math.inf
    update_step = 0
    started = time.monotonic()

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_FM", "lr", "seconds"],
        )
        writer.writeheader()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, args.epochs + 1):
            losses = []
            for batch_index, batch in enumerate(train_loader, start=1):
                inputs = move_training_batch(batch, device, args.mask_label_bf16)
                # Scale a final incomplete accumulation window by its actual
                # size. With three pages and accumulation=2, the old fixed
                # divisor halved the third page's gradient every epoch.
                window_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
                window_size = min(args.gradient_accumulation, len(train_loader) - window_start)
                with autocast_context(device, args.amp):
                    loss = model(**inputs).loss / window_size
                scaler.scale(loss).backward()
                losses.append(float(loss.detach().cpu()) * window_size)
                del loss, inputs

                if batch_index % args.gradient_accumulation == 0 or batch_index == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    update_step += 1

            should_validate = epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0
            val_loss = (
                mean_validation_loss(model, val_loader, device, args.amp, args.mask_label_bf16)
                if should_validate
                else math.nan
            )
            val_metrics = (
                validation_instance_metrics(model, processor, args, subset, device)
                if should_validate and args.checkpoint_metric == "val_fm"
                else {}
            )
            val_fm = val_metrics.get("FM", math.nan)
            train_loss = float(np.mean(losses))
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_FM": val_fm,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.monotonic() - started, 1),
            }
            writer.writerow(row)
            history_file.flush()
            print(
                f"[{subset}] epoch {epoch:04d}/{args.epochs} "
                f"train={train_loss:.4f} val={val_loss:.4f} val_FM={val_fm:.4f} "
                f"updates={update_step}/{total_updates}"
            )

            candidate = val_loss if args.checkpoint_metric == "val_loss" else val_fm
            improved = candidate < best_value if args.checkpoint_metric == "val_loss" else candidate > best_value
            if should_validate and improved:
                best_value = candidate
                model.save_pretrained(output_dir)
                processor.size = {"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge}
                processor.save_pretrained(output_dir)
                (output_dir / "run_config.json").write_text(
                    json.dumps(
                        {
                            "subset": subset,
                            "base_model": MODEL_ID,
                            "epochs": args.epochs,
                            "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay,
                            "num_queries": args.num_queries,
                            "train_num_points": args.train_num_points,
                            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                            "shortest_edge": args.shortest_edge,
                            "longest_edge": args.longest_edge,
                            "mask_label_stride": args.mask_label_stride,
                            "mask_label_bf16": args.mask_label_bf16,
                            "train_vertical_tiles": args.train_vertical_tiles,
                            "checkpoint_metric": args.checkpoint_metric,
                            "best_checkpoint_value": best_value,
                            "best_val_loss": val_loss,
                            "best_val_FM": val_fm,
                            "checkpoint_score_threshold": args.checkpoint_score_threshold,
                            "checkpoint_mask_threshold": args.checkpoint_mask_threshold,
                            "checkpoint_nms_threshold": args.checkpoint_nms_threshold,
                            "stitch_tile_instances": args.stitch_tile_instances,
                            "stitch_seam_band_ratio": args.stitch_seam_band_ratio,
                            "stitch_min_vertical_overlap": args.stitch_min_vertical_overlap,
                            "seed": args.seed,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    return output_dir


def zottin_metrics(gt_binary: np.ndarray, pred_instances: np.ndarray, threshold: float = 0.75) -> dict:
    """Vectorized equivalent of ``71_misc/evaluate_util.py:evaluate_metrics``."""
    structure = np.ones((3, 3), dtype=np.uint8)
    gt_labels, n_gt = ndimage.label(gt_binary.astype(bool), structure=structure)
    _, pred_labels = np.unique(pred_instances, return_inverse=True)
    pred_labels = pred_labels.reshape(pred_instances.shape)
    # np.unique maps background zero to zero because zero is the smallest value.
    n_pred = int(pred_labels.max())
    if n_gt == 0:
        return {"Pixel_IU": 0.0, "Line_IU": 0.0, "DR": 0.0, "RA": 0.0, "FM": 0.0}

    contingency = np.bincount(
        gt_labels.ravel().astype(np.int64) * (n_pred + 1) + pred_labels.ravel(),
        minlength=(n_gt + 1) * (n_pred + 1),
    ).reshape(n_gt + 1, n_pred + 1)
    intersections = contingency[1:, 1:].astype(np.float64)
    gt_area = contingency[1:, :].sum(axis=1).astype(np.float64)
    pred_area = contingency[:, 1:].sum(axis=0).astype(np.float64)
    unions = gt_area[:, None] + pred_area[None, :] - intersections
    ious = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)

    matched_all = int((ious >= threshold).sum())
    if n_pred:
        best_pred = np.argmax(ious, axis=1)
        best_iou = ious[np.arange(n_gt), best_pred]
        matched_gt = np.flatnonzero(best_iou > 0)
    else:
        best_pred = np.empty(n_gt, dtype=np.int64)
        matched_gt = np.empty(0, dtype=np.int64)

    if len(matched_gt):
        selected = best_pred[matched_gt]
        true_positive = intersections[matched_gt, selected]
        false_positive = pred_area[selected] - true_positive
        false_negative = gt_area[matched_gt] - true_positive
        denominator = float((true_positive + false_positive + false_negative).sum())
        pixel_iu = float(true_positive.sum() / denominator) if denominator else 0.0
        precision = np.divide(
            true_positive,
            true_positive + false_positive,
            out=np.zeros_like(true_positive),
            where=(true_positive + false_positive) > 0,
        )
        recall = np.divide(
            true_positive,
            true_positive + false_negative,
            out=np.zeros_like(true_positive),
            where=(true_positive + false_negative) > 0,
        )
        line_iu = float(((precision >= threshold) & (recall >= threshold)).mean())
    else:
        pixel_iu = line_iu = 0.0

    detection_rate = matched_all / n_gt
    recognition_accuracy = matched_all / n_pred if n_pred else 0.0
    f_measure = (
        0.0
        if detection_rate + recognition_accuracy == 0
        else 2 * detection_rate * recognition_accuracy / (detection_rate + recognition_accuracy)
    )
    return {
        "Pixel_IU": pixel_iu,
        "Line_IU": line_iu,
        "DR": detection_rate,
        "RA": recognition_accuracy,
        "FM": f_measure,
    }


def post_process_text_line_instances(
    outputs,
    processed_size: tuple[int, int],
    target_size: tuple[int, int],
    score_threshold: float,
    mask_threshold: float,
    nms_threshold: float,
) -> np.ndarray:
    """Convert Mask2Former queries to a non-overlapping text-line label map.

    Transformers' generic instance postprocessor first forces every mask to a
    square 384x384 canvas, even when inference used an aspect-preserving page
    resolution. That severely distorts document lines. Here masks are resized
    to the actual processor canvas, duplicate queries are removed by mask IoU,
    and overlap pixels are assigned to the highest-scoring query.
    """
    class_scores = outputs.class_queries_logits[0].float().softmax(dim=-1)[:, 0]
    mask_probabilities = torch.nn.functional.interpolate(
        outputs.masks_queries_logits.float(),
        size=processed_size,
        mode="bilinear",
        align_corners=False,
    )[0].sigmoid()
    binary_masks = mask_probabilities >= mask_threshold
    areas = binary_masks.flatten(1).sum(dim=1)
    mask_quality = (
        (mask_probabilities * binary_masks).flatten(1).sum(dim=1)
        / areas.clamp_min(1)
    )
    scores = class_scores * mask_quality
    candidates = torch.nonzero((scores >= score_threshold) & (areas > 0), as_tuple=False).flatten()
    candidates = candidates[torch.argsort(scores[candidates], descending=True)]

    kept: list[int] = []
    for candidate_tensor in candidates:
        candidate = int(candidate_tensor)
        if kept:
            intersections = (binary_masks[kept] & binary_masks[candidate]).flatten(1).sum(dim=1)
            unions = areas[kept] + areas[candidate] - intersections
            ious = intersections.float() / unions.clamp_min(1).float()
            if bool(torch.any(ious >= nms_threshold)):
                continue
        kept.append(candidate)

    if not kept:
        return np.zeros(target_size, dtype=np.uint16)

    kept_indices = torch.tensor(kept, device=mask_probabilities.device)
    weighted_masks = mask_probabilities[kept_indices] * scores[kept_indices, None, None]
    best_values, best_indices = weighted_masks.max(dim=0)
    best_mask_probabilities = mask_probabilities[kept_indices].gather(
        0, best_indices.unsqueeze(0)
    )[0]
    segmentation = best_indices.to(torch.int32) + 1
    segmentation[(best_values <= 0) | (best_mask_probabilities < mask_threshold)] = 0
    segmentation = torch.nn.functional.interpolate(
        segmentation[None, None].float(), size=target_size, mode="nearest"
    )[0, 0].to(torch.uint16)
    return segmentation.cpu().numpy()


def stitch_vertical_tile_instances(
    instance_map: np.ndarray,
    vertical_tiles: int,
    seam_band_ratio: float = 0.02,
    min_vertical_overlap: float = 0.5,
) -> np.ndarray:
    """Merge text-line fragments split by vertical tile seams.

    Candidates must end/start near the same seam. Pairs are matched greedily
    by overlap of their vertical bounding-box intervals, which is appropriate
    for horizontal text lines and avoids using annotations at inference time.
    The masks themselves are not dilated or otherwise changed; only their
    instance IDs are unified.
    """
    if vertical_tiles <= 1 or instance_map.max() == 0:
        return instance_map
    if not 0 <= seam_band_ratio <= 0.5:
        raise ValueError("seam_band_ratio must be between 0 and 0.5")
    if not 0 <= min_vertical_overlap <= 1:
        raise ValueError("min_vertical_overlap must be between 0 and 1")

    max_label = int(instance_map.max())
    parents = np.arange(max_label + 1, dtype=np.int32)

    def find(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = int(parents[label])
        return label

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    boxes = {}
    for label, slices in enumerate(ndimage.find_objects(instance_map), start=1):
        if slices is not None:
            boxes[label] = (
                slices[1].start,
                slices[0].start,
                slices[1].stop,
                slices[0].stop,
            )

    _, width = instance_map.shape
    band = max(2, int(round(width * seam_band_ratio)))
    seams = [int(width * index / vertical_tiles) for index in range(1, vertical_tiles)]
    for seam in seams:
        left = [
            label
            for label, box in boxes.items()
            if box[0] < seam and box[2] >= seam - band
        ]
        right = [
            label
            for label, box in boxes.items()
            if box[0] >= seam and box[0] <= seam + band
        ]
        candidates = []
        for left_label in left:
            left_box = boxes[left_label]
            for right_label in right:
                right_box = boxes[right_label]
                overlap = max(
                    0,
                    min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]),
                )
                smaller_height = max(
                    1,
                    min(left_box[3] - left_box[1], right_box[3] - right_box[1]),
                )
                score = overlap / smaller_height
                if score >= min_vertical_overlap:
                    candidates.append((score, left_label, right_label))

        used_left, used_right = set(), set()
        for _, left_label, right_label in sorted(candidates, reverse=True):
            if left_label not in used_left and right_label not in used_right:
                union(left_label, right_label)
                used_left.add(left_label)
                used_right.add(right_label)

    lookup = np.arange(max_label + 1, dtype=np.int32)
    for label in range(1, max_label + 1):
        lookup[label] = find(label)
    merged = lookup[instance_map]
    _, dense_labels = np.unique(merged, return_inverse=True)
    return dense_labels.reshape(instance_map.shape).astype(np.uint16)


def validation_instance_metrics(
    model: Mask2FormerForUniversalSegmentation,
    processor: Mask2FormerImageProcessor,
    args: argparse.Namespace,
    subset: str,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the live model on full validation pages for checkpoint selection."""
    val_pages = CocoTextLineDataset(
        args.data_root,
        subset,
        "val",
        processor=None,
        max_images=args.max_eval_images,
    )
    rows = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(val_pages)):
            image, _, metadata = val_pages.raw_item(index)
            prediction = np.zeros((metadata["height"], metadata["width"]), dtype=np.uint16)
            boundaries = np.linspace(
                0, metadata["width"], args.train_vertical_tiles + 1, dtype=int
            )
            instance_offset = 0
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                tile = image.crop((int(left), 0, int(right), metadata["height"]))
                encoded = processor(
                    images=tile,
                    size={"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge},
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(device)
                    for key, value in encoded.items()
                    if torch.is_tensor(value)
                }
                with autocast_context(device, args.amp):
                    outputs = model(**inputs)
                tile_map = post_process_text_line_instances(
                    outputs,
                    processed_size=tuple(encoded["pixel_values"].shape[-2:]),
                    target_size=(tile.height, tile.width),
                    score_threshold=args.checkpoint_score_threshold,
                    mask_threshold=args.checkpoint_mask_threshold,
                    nms_threshold=args.checkpoint_nms_threshold,
                )
                foreground = tile_map > 0
                tile_map[foreground] += instance_offset
                prediction[:, int(left):int(right)] = tile_map
                instance_offset = int(prediction.max())
                del inputs, outputs

            gt_path = resolve_split_dir(
                args.data_root / subset / f"text-line-gt-{subset}", "val"
            ) / f"{metadata['stem']}.png"
            if args.stitch_tile_instances:
                prediction = stitch_vertical_tile_instances(
                    prediction,
                    args.train_vertical_tiles,
                    args.stitch_seam_band_ratio,
                    args.stitch_min_vertical_overlap,
                )
            gt_binary = np.asarray(Image.open(gt_path).convert("L")) > 0
            rows.append(zottin_metrics(gt_binary, prediction))

    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metric_names = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
    means = {name: float(np.mean([row[name] for row in rows])) for name in metric_names}
    print(
        f"[{subset}] checkpoint validation: FM={means['FM']:.4f} "
        f"LineIU={means['Line_IU']:.4f} PixelIU={means['Pixel_IU']:.4f}"
    )
    return means


def predict_subset(
    args: argparse.Namespace,
    subset: str,
    checkpoint: Path,
    run_name: str,
) -> dict:
    device = torch.device(args.device)
    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)
    model = load_checkpoint_model(checkpoint).to(device).eval()
    test_data = CocoTextLineDataset(
        args.data_root,
        subset,
        args.eval_split,
        processor=None,
        max_images=args.max_eval_images,
    )
    output_dir = args.evaluation_root / subset / run_name
    instance_dir = output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def infer_image(tile: Image.Image) -> np.ndarray:
        encoded = processor(
            images=tile,
            size={"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge},
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
        with autocast_context(device, args.amp):
            outputs = model(**inputs)
        if args.postprocess == "textline":
            result_map = post_process_text_line_instances(
                outputs,
                processed_size=tuple(encoded["pixel_values"].shape[-2:]),
                target_size=(tile.height, tile.width),
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                nms_threshold=args.nms_threshold,
            )
        else:
            result = processor.post_process_instance_segmentation(
                outputs,
                threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                target_sizes=[(tile.height, tile.width)],
            )[0]
            segmentation = result["segmentation"]
            if segmentation is None:
                result_map = np.zeros((tile.height, tile.width), dtype=np.uint16)
            else:
                result_map = (
                    segmentation.to(torch.int32).cpu().numpy() + 1
                ).clip(min=0).astype(np.uint16)
        del inputs, outputs
        return result_map

    with torch.inference_mode():
        for index in range(len(test_data)):
            image, _, metadata = test_data.raw_item(index)
            if args.vertical_tiles == 1:
                pred_instances = infer_image(image)
            else:
                pred_instances = np.zeros((metadata["height"], metadata["width"]), dtype=np.uint16)
                boundaries = np.linspace(0, metadata["width"], args.vertical_tiles + 1, dtype=int)
                instance_offset = 0
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    tile_map = infer_image(image.crop((left, 0, right, metadata["height"])))
                    foreground = tile_map > 0
                    tile_map[foreground] += instance_offset
                    pred_instances[:, left:right] = tile_map
                    instance_offset = int(pred_instances.max())
                if args.stitch_tile_instances:
                    pred_instances = stitch_vertical_tile_instances(
                        pred_instances,
                        args.vertical_tiles,
                        args.stitch_seam_band_ratio,
                        args.stitch_min_vertical_overlap,
                    )
            Image.fromarray(pred_instances, mode="I;16").save(instance_dir / f"{metadata['stem']}.png")
            instance_count = int(np.count_nonzero(np.unique(pred_instances)))

            gt_path = resolve_split_dir(
                args.data_root / subset / f"text-line-gt-{subset}", args.eval_split
            ) / f"{metadata['stem']}.png"
            gt_binary = np.asarray(Image.open(gt_path).convert("L")) > 0
            metrics = zottin_metrics(gt_binary, pred_instances)
            rows.append(
                {
                    "image": f"{metadata['stem']}.png",
                    "instances": instance_count,
                    **metrics,
                }
            )
            print(
                f"[{subset}] {index + 1:02d}/{len(test_data):02d} {metadata['stem']} "
                f"instances={instance_count} PixelIU={metrics['Pixel_IU']:.4f} "
                f"LineIU={metrics['Line_IU']:.4f}"
            )

    metrics_path = output_dir / "per_page_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    metric_names = ("Pixel_IU", "Line_IU", "DR", "RA", "FM")
    summary = {
        "subset": subset,
        "split": args.eval_split,
        "checkpoint": str(checkpoint),
        "pages": len(rows),
        "postprocess": args.postprocess,
        "vertical_tiles": args.vertical_tiles,
        "stitch_tile_instances": args.stitch_tile_instances,
        "stitch_seam_band_ratio": args.stitch_seam_band_ratio,
        "stitch_min_vertical_overlap": args.stitch_min_vertical_overlap,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "nms_threshold": args.nms_threshold if args.postprocess == "textline" else None,
        "mean_instances": float(np.mean([row["instances"] for row in rows])) if rows else math.nan,
        **{
            name: float(np.mean([row[name] for row in rows])) if rows else math.nan
            for name in metric_names
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{subset}] summary: {json.dumps(summary, indent=2)}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def selected_subsets(values: Iterable[str]) -> list[str]:
    values = list(values)
    if values == ["all"]:
        return list(SUBSETS)
    invalid = set(values) - set(SUBSETS)
    if invalid:
        raise ValueError(f"Unknown subsets: {sorted(invalid)}")
    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subsets", nargs="+", default=["all"])
    parser.add_argument("--data-root", type=Path, default=REPO / "00_data/U-DIADS-TL")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=REPO / "80_models/03_instance_segmentation/u-diads-tl",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=REPO / "99_evaluation/03_instance_segmentation/u-diads-tl",
    )
    parser.add_argument("--run-name", default="mask2former_swin_tiny")
    parser.add_argument("--shortest-edge", type=int, default=512)
    parser.add_argument("--longest-edge", type=int, default=768)
    parser.add_argument("--max-eval-images", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stitch-tile-instances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Merge horizontally aligned instance fragments across vertical tile seams",
    )
    parser.add_argument(
        "--stitch-seam-band-ratio",
        type=float,
        default=0.02,
        help="Fraction of page width searched on each side of a vertical tile seam",
    )
    parser.add_argument(
        "--stitch-min-vertical-overlap",
        type=float,
        default=0.5,
        help="Minimum y-interval overlap over the smaller fragment height for stitching",
    )


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--train-num-points", type=int, default=4096)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize from a previously fine-tuned task checkpoint instead of the COCO model",
    )
    parser.add_argument("--mask-label-stride", type=int, default=4)
    parser.add_argument(
        "--mask-label-bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store dense target masks as BF16 on CUDA to reduce criterion memory",
    )
    parser.add_argument(
        "--train-vertical-tiles",
        type=int,
        default=1,
        help="Train and validate on this many non-overlapping vertical column tiles per page",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-metric",
        choices=("val_loss", "val_fm"),
        default="val_loss",
        help="Select the saved checkpoint by minimum validation loss or maximum instance FM",
    )
    parser.add_argument("--checkpoint-score-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint-mask-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint-nms-threshold", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)


def add_evaluate_arguments(parser: argparse.ArgumentParser, require_checkpoint: bool = True) -> None:
    parser.add_argument("--checkpoint", type=Path, required=require_checkpoint)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--postprocess", choices=("textline", "hf"), default="textline")
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument(
        "--vertical-tiles",
        type=int,
        default=1,
        help="Split every page into this many non-overlapping vertical column tiles",
    )
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune one checkpoint per selected subset")
    add_common_arguments(train_parser)
    add_train_arguments(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on selected test subsets")
    add_common_arguments(evaluate_parser)
    add_evaluate_arguments(evaluate_parser)

    run_parser = subparsers.add_parser("run", help="Train and immediately evaluate each selected subset")
    add_common_arguments(run_parser)
    add_train_arguments(run_parser)
    add_evaluate_arguments(run_parser, require_checkpoint=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    subsets = selected_subsets(args.subsets)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    summaries = []
    for subset in subsets:
        if args.command in ("train", "run"):
            checkpoint = train_subset(args, subset, args.run_name)
        else:
            checkpoint = args.checkpoint
        if args.command in ("evaluate", "run"):
            summaries.append(predict_subset(args, subset, checkpoint, args.run_name))

    if summaries:
        args.evaluation_root.mkdir(parents=True, exist_ok=True)
        summary_path = args.evaluation_root / f"{args.run_name}_all_subsets.json"
        # Preserve other subsets when separately evaluated subset-specific
        # checkpoints use the same run name.
        merged = {}
        if summary_path.exists():
            try:
                merged = {row["subset"]: row for row in json.loads(summary_path.read_text(encoding="utf-8"))}
            except (json.JSONDecodeError, KeyError, TypeError):
                merged = {}
        merged.update({row["subset"]: row for row in summaries})
        ordered = [merged[name] for name in SUBSETS if name in merged]
        summary_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
        print(f"All-subset summary saved to {summary_path}")


if __name__ == "__main__":
    main()
