#!/usr/bin/env python3
"""Fine-tune RF-DETR and RT-DETR on one U-DIADS-TL COCO split.

Both detectors receive the same 1408-square resize/pad geometry, color jitter,
AdamW/cosine recipe, effective batch size, seed, 100 epochs, validation split,
and validation mAP checkpoint selection.  The held-out test set is scored only
after the best validation checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torchvision.transforms
from datasets import Dataset, DatasetDict, Image as HFImage
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.image_transforms import center_to_corners_format
from transformers.models.rt_detr.modeling_rt_detr import replace_batch_norm
from transformers.trainer_utils import get_last_checkpoint

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL_CHECKPOINTS = {
    "RF-DETR": "Roboflow/rf-detr-medium",
    "RT-DETR": "PekingU/rtdetr_r50vd",
}


def load_coco_split(coco_root, image_root, split, category_id_to_label):
    payload = json.loads((coco_root / f"{split}.json").read_text(encoding="utf-8"))
    annotations_by_image = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    records = []
    for info in sorted(payload["images"], key=lambda item: item["id"]):
        width, height = int(info["width"]), int(info["height"])
        image_path = image_root / split / info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        boxes, labels, areas, ids = [], [], [], []
        for annotation in annotations_by_image[info["id"]]:
            x, y, w, h = map(float, annotation["bbox"])
            x, y = max(0.0, x), max(0.0, y)
            w, h = min(w, width - x), min(h, height - y)
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, w, h])
            labels.append(category_id_to_label[int(annotation["category_id"])])
            areas.append(w * h)
            ids.append(int(annotation["id"]))
        if not boxes:
            raise ValueError(f"No boxes in {image_path}")
        records.append(
            {
                "image_id": int(info["id"]),
                "image": str(image_path),
                "width": width,
                "height": height,
                "objects": {"id": ids, "bbox": boxes, "category": labels, "area": areas},
            }
        )
    return Dataset.from_list(records).cast_column("image", HFImage())


def format_coco(image_id, objects):
    return {
        "image_id": image_id,
        "annotations": [
            {
                "image_id": image_id,
                "category_id": category,
                "iscrowd": 0,
                "area": area,
                "bbox": list(box),
            }
            for category, area, box in zip(
                objects["category"], objects["area"], objects["bbox"]
            )
        ],
    }


def transform_batch(examples, processor, jitter, augment):
    images, annotations = [], []
    for image_id, image, objects in zip(
        examples["image_id"], examples["image"], examples["objects"]
    ):
        image = image.convert("RGB")
        if augment:
            image = jitter(image)
        images.append(np.array(image, copy=True))
        annotations.append(format_coco(image_id, objects))
    return processor(images=images, annotations=annotations, return_tensors="pt")


def collate_fn(batch):
    result = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }
    if "pixel_mask" in batch[0]:
        result["pixel_mask"] = torch.stack([item["pixel_mask"] for item in batch])
    return result


@dataclass
class DetectionOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def original_size(target):
    values = np.atleast_1d(np.asarray(target["orig_size"])).flatten()
    return int(values[0]), int(values[1])


def normalized_to_absolute(boxes, image_size):
    height, width = image_size
    corners = center_to_corners_format(boxes)
    # The processor preserves aspect ratio and pads at bottom/right to a square.
    # Normalized labels therefore live in square-canvas coordinates, not in an
    # independently x/y-normalized original rectangle.
    side = max(height, width)
    corners = corners * torch.tensor([[side, side, side, side]], dtype=corners.dtype)
    corners[:, [0, 2]] = corners[:, [0, 2]].clamp(0, width)
    corners[:, [1, 3]] = corners[:, [1, 3]].clamp(0, height)
    return corners


@torch.no_grad()
def compute_metrics(evaluation_results, processor):
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    image_sizes, processed_targets, processed_predictions = [], [], []
    for target_batch in targets:
        batch_sizes = []
        for target in target_batch:
            size = original_size(target)
            batch_sizes.append(size)
            processed_targets.append(
                {
                    "boxes": normalized_to_absolute(torch.as_tensor(target["boxes"]), size),
                    "labels": torch.as_tensor(target["class_labels"], dtype=torch.int64),
                }
            )
        image_sizes.append(torch.tensor(batch_sizes))
    for prediction_batch, original_sizes in zip(predictions, image_sizes):
        outputs = DetectionOutput(
            logits=torch.as_tensor(prediction_batch[1]),
            pred_boxes=torch.as_tensor(prediction_batch[2]),
        )
        square_sizes = torch.tensor(
            [[max(int(h), int(w)), max(int(h), int(w))] for h, w in original_sizes]
        )
        batch_predictions = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=square_sizes
        )
        for prediction, (height, width) in zip(batch_predictions, original_sizes):
            prediction["boxes"][:, [0, 2]] = prediction["boxes"][:, [0, 2]].clamp(0, int(width))
            prediction["boxes"][:, [1, 3]] = prediction["boxes"][:, [1, 3]].clamp(0, int(height))
        processed_predictions.extend(batch_predictions)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
    metric.update(processed_predictions, processed_targets)
    return {key: round(value.item(), 6) for key, value in metric.compute().items()}


def make_processor(checkpoint, image_size):
    return AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"max_height": image_size, "max_width": image_size},
        do_pad=True,
        pad_size={"height": image_size, "width": image_size},
    )


def train_one(model_name, checkpoint, raw_datasets, labels, args, output_root):
    run_name = model_name.lower().replace("-", "_")
    output_dir = output_root / run_name
    summary_path = output_dir / "metrics_summary.json"
    best_model_dir = output_dir / "best_model"
    if summary_path.exists() and best_model_dir.exists() and not args.force:
        print(f"[{model_name}] already complete -> {best_model_dir}", flush=True)
        return json.loads(summary_path.read_text())["summary"]

    set_seed(args.seed)
    processor = make_processor(checkpoint, args.image_size)
    jitter = torchvision.transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015
    )
    datasets = DatasetDict(
        {
            split: dataset.with_transform(
                partial(
                    transform_batch,
                    processor=processor,
                    jitter=jitter,
                    augment=split == "train",
                )
            )
            for split, dataset in raw_datasets.items()
        }
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=labels["id2label"],
        label2id=labels["label2id"],
        ignore_mismatched_sizes=True,
    )
    before = sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    replace_batch_norm(model)
    after = sum(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    print(f"[{model_name}] froze {before-after} live BatchNorm2d layers", flush=True)

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = bool(torch.cuda.is_available() and not bf16)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"udiads-{args.subset.lower()}-{run_name}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        max_grad_norm=0.1,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, processor=processor),
    )
    last_checkpoint = (
        get_last_checkpoint(str(output_dir)) if output_dir.exists() and not args.force else None
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint or None)
    trainer.save_model(str(best_model_dir))
    processor.save_pretrained(str(best_model_dir))
    validation = trainer.evaluate(datasets["val"], metric_key_prefix="validation")
    test = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    record = {
        "model": model_name,
        "checkpoint": checkpoint,
        "validation_mAP@50": validation["validation_map_50"],
        "validation_mAP@50-95": validation["validation_map"],
        "test_mAP@50": test["test_map_50"],
        "test_mAP@50-95": test["test_map"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
    }
    summary_path.write_text(
        json.dumps({"summary": record, "validation": validation, "test": test}, indent=2),
        encoding="utf-8",
    )
    del trainer, model, datasets, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="Latin14396", choices=("Latin14396", "Latin2", "Syr341"))
    parser.add_argument("--models", nargs="+", default=list(MODEL_CHECKPOINTS), choices=list(MODEL_CHECKPOINTS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=1408, dest="image_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    coco_root = REPO / "00_data" / "U-DIADS-TL" / f"coco_dataset_{args.subset.lower()}"
    image_root = REPO / "00_data" / "U-DIADS-TL" / f"yolo_dataset_{args.subset.lower()}" / "images"
    output_root = (
        REPO
        / "80_models/02_2stage/u-diads-tl/detection"
        / f"hf_detr_{args.subset.lower()}_square_geometry"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    category_payload = json.loads((coco_root / "train.json").read_text(encoding="utf-8"))
    categories = sorted(category_payload["categories"], key=lambda item: item["id"])
    category_id_to_label = {int(category["id"]): i for i, category in enumerate(categories)}
    labels = {
        "id2label": {i: category["name"] for i, category in enumerate(categories)},
    }
    labels["label2id"] = {name: i for i, name in labels["id2label"].items()}
    raw_datasets = DatasetDict(
        {
            split: load_coco_split(coco_root, image_root, split, category_id_to_label)
            for split in ("train", "val", "test")
        }
    )
    print(
        f"subset={args.subset} pages="
        + ", ".join(f"{split}:{len(dataset)}" for split, dataset in raw_datasets.items()),
        flush=True,
    )
    records = []
    for model_name in args.models:
        records.append(
            train_one(
                model_name,
                MODEL_CHECKPOINTS[model_name],
                raw_datasets,
                labels,
                args,
                output_root,
            )
        )
    import pandas as pd

    report = pd.DataFrame(records)
    report.to_csv(output_root / f"rf_detr_vs_rt_detr_{args.subset.lower()}_metrics.csv", index=False)
    print("\n" + report.to_string(index=False), flush=True)
    print(f"DONE -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
