"""Train the CB55 YOLOv8s initialization ablation with one fixed recipe.

Only the initialization differs between arms:
  scratch     yolov8s.yaml
  coco        stock COCO yolov8s.pt
  cbad_distill Lightly-distilled cBAD checkpoint

The recipe is copied from the best existing YOLO/CB55 run in
80_models/02_2stage/diva-hisdb/detection (mAP50 checkpoint selection on CB55 val).
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
from train_detector import resolve_data  # noqa: E402


DATA = HERE / "configs/data/diva_cb55_detect.yaml"
PROJECT = REPO / "80_models/02_2stage/diva-hisdb/detection/yolov8s_init_ablation_cb55"
COCO = REPO / "80_models/02_2stage/pretrained/yolov8s.pt"
CBAD_DISTILL = REPO / "yolov8s_cbad_distilled_backbone.pt"

INITIALIZATIONS = {
    "scratch": "yolov8s.yaml",
    "coco": str(COCO),
    "cbad_distill": str(CBAD_DISTILL),
}

RECIPE = {
    "epochs": 100,
    "imgsz": 1280,
    "batch": 2,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "cos_lr": True,
    "patience": 50,
    "warmup_epochs": 3.0,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "box": 10.0,
    "cls": 0.5,
    "dfl": 2.0,
    "mosaic": 0.0,
    "fliplr": 0.0,
    "seed": 0,
    "deterministic": True,
    "workers": 4,
}


def parameter_counts(model: YOLO) -> tuple[int, int]:
    total = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    return total, trainable


def validate_initialization(name: str, reference: str) -> YOLO:
    if reference != "yolov8s.yaml" and not Path(reference).is_file():
        raise FileNotFoundError(f"{name}: initialization not found: {reference}")
    model = YOLO(reference)
    # Ultralytics loads serialized .pt models in inference mode with requires_grad=False;
    # its Trainer would unfreeze them later. Make that transition explicit so this guard
    # reports the actual fine-tuned student rather than the checkpoint loader state.
    model.model.requires_grad_(True)
    total, trainable = parameter_counts(model)
    print(
        f"[{name}] initialization={reference}\n"
        f"[{name}] student parameters: total={total:,}, trainable={trainable:,}",
        flush=True,
    )
    if total < 10_000_000 or trainable < 10_000_000:
        raise RuntimeError(
            f"{name}: only {total:,} total / {trainable:,} trainable parameters; "
            "expected a complete YOLOv8s (~11M). Aborting before GPU training."
        )
    return model


def train_arm(name: str, reference: str, force: bool) -> dict:
    run_dir = PROJECT / name
    best = run_dir / "weights/best.pt"
    if best.is_file() and not force:
        checked = YOLO(str(best))
        total, trainable = parameter_counts(checked)
        print(f"[{name}] already complete: {best} ({total:,} parameters)")
        return {"arm": name, "status": "existing", "best": str(best), "parameters": total}

    model = validate_initialization(name, reference)
    result = model.train(
        data=resolve_data(DATA),
        project=str(PROJECT),
        name=name,
        exist_ok=True,
        **RECIPE,
    )
    if not best.is_file():
        raise RuntimeError(f"{name}: training returned but {best} was not produced: {result}")
    trained = YOLO(str(best))
    total, trainable = parameter_counts(trained)
    if total < 10_000_000:
        raise RuntimeError(f"{name}: trained checkpoint unexpectedly has only {total:,} parameters")
    del model, trained
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"arm": name, "status": "trained", "best": str(best), "parameters": total}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms", nargs="+", choices=tuple(INITIALIZATIONS),
        default=list(INITIALIZATIONS),
    )
    parser.add_argument("--force", action="store_true", help="overwrite completed arm directories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings.update({"mlflow": False})
    PROJECT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data": str(DATA),
        "project": str(PROJECT),
        "initializations": INITIALIZATIONS,
        "recipe": RECIPE,
        "runs": [],
    }
    for arm in args.arms:
        manifest["runs"].append(train_arm(arm, INITIALIZATIONS[arm], args.force))
        (PROJECT / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(f"DONE: {PROJECT}")


if __name__ == "__main__":
    main()
