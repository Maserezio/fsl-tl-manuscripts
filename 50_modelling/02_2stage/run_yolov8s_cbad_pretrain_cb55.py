"""Train the two additional COCO -> cBAD -> CB55 experiment arms.

Stage 1 produces the checkpoint used for zero-shot CB55 evaluation. Stage 2 starts
from that exact checkpoint and fine-tunes on CB55 with the fixed recipe used by
run_yolov8s_init_ablation_cb55.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from ultralytics import YOLO, settings


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from run_yolov8s_init_ablation_cb55 import RECIPE as CB55_RECIPE  # noqa: E402
from train_detector import resolve_data  # noqa: E402


COCO = REPO / "80_models/02_2stage/pretrained/yolov8s.pt"
CBAD_DATA = REPO / "00_data/cBAD/yolo_dataset_cbad_detect/dataset.yaml"
CB55_DATA = HERE / "configs/data/diva_cb55_detect.yaml"

CBAD_PROJECT = REPO / "80_models/02_2stage/cbad/detection"
CBAD_RUN = "yolov8s_coco_pretrain_300ep"
CBAD_BEST = CBAD_PROJECT / CBAD_RUN / "weights/best.pt"
CBAD_LAST = CBAD_PROJECT / CBAD_RUN / "weights/last.pt"

CB55_PROJECT = REPO / "80_models/02_2stage/diva-hisdb/detection/yolov8s_init_ablation_cb55"
CB55_RUN = "cbad_pretrain_finetune"
CB55_BEST = CB55_PROJECT / CB55_RUN / "weights/best.pt"

REQUESTED_CBAD_BATCH = 8
# batch=8 OOMs on a clean 8 GB RTX 4060. Batch 4 is faster than batch 2 even
# with occasional CPU assigner fallback on cBAD pages containing thousands of
# mosaicked boxes. Ultralytics retains nbs=64 through gradient accumulation.
HARDWARE_CBAD_BATCH = 4

CBAD_RECIPE = {
    "epochs": 300,
    "imgsz": 1280,
    "batch": HARDWARE_CBAD_BATCH,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
    "patience": 50,
    "box": 10.0,
    "cls": 0.5,
    "dfl": 2.0,
    "mosaic": 0.5,
    "seed": 0,
    "deterministic": True,
    "workers": 4,
}


def parameter_count(model: YOLO) -> int:
    return sum(parameter.numel() for parameter in model.model.parameters())


def validate_full_yolov8s(model: YOLO, label: str) -> None:
    model.model.requires_grad_(True)
    total = parameter_count(model)
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"[{label}] parameters: total={total:,}, trainable={trainable:,}", flush=True)
    if total < 10_000_000 or trainable < 10_000_000:
        raise RuntimeError(
            f"{label}: incomplete YOLOv8s ({total:,} total, {trainable:,} trainable); aborting"
        )


def release(*models: YOLO) -> None:
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_cbad(force: bool, resume: bool) -> None:
    if resume:
        if not CBAD_LAST.is_file():
            raise FileNotFoundError(f"cannot resume; missing {CBAD_LAST}")
        print(f"[cBAD] resuming interrupted run from {CBAD_LAST}")
        YOLO(str(CBAD_LAST)).train(resume=True)
        if not CBAD_BEST.is_file():
            raise RuntimeError(f"resumed cBAD training did not produce {CBAD_BEST}")
        return
    if CBAD_BEST.is_file() and not force:
        existing = YOLO(str(CBAD_BEST))
        print(f"[cBAD] reusing {CBAD_BEST} ({parameter_count(existing):,} parameters)")
        release(existing)
        return
    model = YOLO(str(COCO))
    validate_full_yolov8s(model, "COCO -> cBAD")
    model.train(
        data=resolve_data(CBAD_DATA),
        project=str(CBAD_PROJECT),
        name=CBAD_RUN,
        exist_ok=True,
        **CBAD_RECIPE,
    )
    if not CBAD_BEST.is_file():
        raise RuntimeError(f"cBAD training did not produce {CBAD_BEST}")
    trained = YOLO(str(CBAD_BEST))
    if parameter_count(trained) < 10_000_000:
        raise RuntimeError("cBAD best checkpoint is not a complete YOLOv8s")
    release(model, trained)


def train_cb55(force: bool) -> None:
    if not CBAD_BEST.is_file():
        raise FileNotFoundError(CBAD_BEST)
    if CB55_BEST.is_file() and not force:
        existing = YOLO(str(CB55_BEST))
        print(f"[cBAD -> CB55] reusing {CB55_BEST} ({parameter_count(existing):,} parameters)")
        release(existing)
        return
    model = YOLO(str(CBAD_BEST))
    validate_full_yolov8s(model, "cBAD -> CB55")
    model.train(
        data=resolve_data(CB55_DATA),
        project=str(CB55_PROJECT),
        name=CB55_RUN,
        exist_ok=True,
        **CB55_RECIPE,
    )
    if not CB55_BEST.is_file():
        raise RuntimeError(f"CB55 fine-tuning did not produce {CB55_BEST}")
    trained = YOLO(str(CB55_BEST))
    if parameter_count(trained) < 10_000_000:
        raise RuntimeError("CB55 fine-tuned checkpoint is not a complete YOLOv8s")
    release(model, trained)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "cbad", "cb55"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume-cbad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.update({"mlflow": False})
    for required in (COCO, CBAD_DATA, CB55_DATA):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.stage in ("all", "cbad"):
        train_cbad(args.force, args.resume_cbad)
    if args.stage in ("all", "cb55"):
        train_cb55(args.force)
    manifest = {
        "stages": [
            {
                "name": "COCO -> cBAD (also CB55 zero-shot)",
                "initialization": str(COCO),
                "data": str(CBAD_DATA),
                "recipe": CBAD_RECIPE,
                "requested_batch": REQUESTED_CBAD_BATCH,
                "hardware_batch": HARDWARE_CBAD_BATCH,
                "batch_note": "batch=8 OOM on a clean 8 GB GPU; batch=4 is faster than benchmarked batch=2; nbs=64 accumulation retained",
                "best": str(CBAD_BEST),
            },
            {
                "name": "COCO -> cBAD -> CB55",
                "initialization": str(CBAD_BEST),
                "data": str(CB55_DATA),
                "recipe": CB55_RECIPE,
                "best": str(CB55_BEST),
            },
        ]
    }
    CB55_PROJECT.mkdir(parents=True, exist_ok=True)
    (CB55_PROJECT / "cbad_pretrain_experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"DONE: zero-shot={CBAD_BEST}\nfine-tuned={CB55_BEST}")


if __name__ == "__main__":
    main()
