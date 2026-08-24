"""Retrain RT-DETR on CB55 with the schedule fixes diagnosed for the earlier run.

The earlier HF run (80_models/.../hf_detr_cb55/rt_detr) reached only val mAP50
0.864 and stalled. Its trainer_state shows why, and this script changes exactly
those three things -- everything else is kept identical to
50_modelling/02_2stage/rf_detr_vs_rt_detr_cb55.ipynb so the comparison stays honest:

  1. OPTIMIZER STEPS. batch 1 x grad_accum 16 over 20 train images gave 20/16 ->
     2 optimizer steps per epoch, 200 total for the whole run. grad_accum drops
     to 2 here -> 10 steps/epoch, 1000 total (5x). The number of forward/backward
     passes per epoch is unchanged (20), so wall-clock is ~unchanged; only the
     update count goes up. per_device batch stays 1: at IMAGE_SIZE=1408 the
     earlier run measured a 4.3GB peak and this GPU has 8GB.
  2. LR SCHEDULE. Was cosine, which annealed LR to 6e-9 by epoch 100 -- val mAP50
     went flat from epoch ~66 onward while eval_loss was STILL falling
     (6.47 -> 6.12), i.e. it ran out of learning rate, not headroom. Switched to
     constant, matching RF-DETR's step schedule with lr_drop=100 (= no drop
     inside a 100-epoch run), which is what RF-DETR got to 0.968 val mAP50 on.
  3. EMA. RF-DETR used EMA (decay 0.993) and its EMA weights beat its raw weights
     (0.971 vs 0.968 val mAP50). The earlier RT-DETR run had none. Added below;
     raw-best and EMA are both evaluated on val and the better one is saved.

Not changed (so any remaining gap is not explained away by them): IMAGE_SIZE,
lr, weight decay, grad clipping, seed, colour-jitter augmentation, the
BatchNorm freezing, and the val/test COCO mAP computation.
"""

import gc
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

# torch 2.9 renamed this; setting only the old name is silently ignored (it warns
# "PYTORCH_CUDA_ALLOC_CONF is deprecated"), so the anti-fragmentation setting the
# original notebook intended was never actually applied. Set both.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torchvision.transforms
from datasets import Dataset, DatasetDict, Image as HFImage
from torchmetrics.detection.mean_ap import MeanAveragePrecision

MAX_DETS = [1, 10, 300]
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.image_transforms import center_to_corners_format
from transformers.models.rt_detr.modeling_rt_detr import replace_batch_norm as _rtdetr_replace_batch_norm

REPO_ROOT = Path(__file__).resolve().parents[2]
# Shared few-shot page selector, so this detector and the one-stage U-Net pick the
# identical pages for a given k.
sys.path.insert(0, str(REPO_ROOT / "50_modelling"))
from few_shot_sampler import select_labeled_pages  # noqa: E402
# DATASET picks both the data and where results land. CB55 keeps the original
# paths so existing runs stay addressable; the U-DIADS subsets follow their own
# family layout.
DATASET = os.environ.get("DATASET", "CB55")
_DATASETS = {
    # CB55 keeps the flat model/eval paths its earlier runs already use; the other
    # DIVA subsets get a per-subset directory like the U-DIADS ones.
    "CB55": dict(
        coco="00_data/DIVA-HisDB/coco_dataset_CB55",
        images="00_data/DIVA-HisDB/yolo_dataset_CB55/images",
        models="80_models/02_2stage/diva-hisdb/detection/rtdetr_hf",
        eval="99_evaluation/02_2stage/diva-hisdb/rtdetr_hf",
    ),
}
for _ds in ("CS18", "CS863"):
    _DATASETS[_ds] = dict(
        coco=f"00_data/DIVA-HisDB/coco_dataset_{_ds}",
        images=f"00_data/DIVA-HisDB/yolo_dataset_{_ds}/images",
        models=f"80_models/02_2stage/diva-hisdb/detection/rtdetr_hf/{_ds}",
        eval=f"99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/{_ds}",
    )
for _ms in ("Latin14396", "Latin2", "Syr341"):
    _DATASETS[_ms] = dict(
        coco=f"00_data/U-DIADS-TL/coco_dataset_{_ms.lower()}",
        images=f"00_data/U-DIADS-TL/yolo_dataset_{_ms.lower()}/images",
        models=f"80_models/02_2stage/u-diads-tl/detection/rtdetr_hf/{_ms}",
        eval=f"99_evaluation/02_2stage/u-diads-tl/rtdetr_hf/{_ms}",
    )
if DATASET not in _DATASETS:
    raise SystemExit(f"unknown DATASET {DATASET!r}; expected one of {list(_DATASETS)}")
_SPEC = _DATASETS[DATASET]

COCO_ROOT = REPO_ROOT / _SPEC["coco"]
IMAGE_ROOT = REPO_ROOT / _SPEC["images"]
EVAL_DIR = REPO_ROOT / _SPEC["eval"]
RESULTS_CSV = EVAL_DIR / "results.csv"

# This script is the single training loop for BOTH HF detectors, so that an
# RF-DETR vs RT-DETR comparison run through it differs only in the checkpoint --
# same Trainer, resolution, schedule, augmentation, seed and metric code. Override
# via env to train the other one:
#   RUN_NAME=rfdetr_hf HF_CHECKPOINT=Roboflow/rf-detr-medium python <this>
# (An earlier RF-DETR comparison used Roboflow's native `rfdetr` package with its own
# recipe -- 560px, multi-scale, ViT layer-wise LR decay -- so it was never controlled
# against RT-DETR. That line of work is archived under
# 80_models/02_2stage/diva-hisdb/detection/_archive/; this path is the controlled one.)
CHECKPOINT = os.environ.get("HF_CHECKPOINT", "PekingU/rtdetr_r50vd")
RUN_NAME = os.environ.get("RUN_NAME", "rtdetr")
OUTPUT_DIR = REPO_ROOT / _SPEC["models"] / RUN_NAME

# Backbone ablation. "" = stock checkpoint untouched; otherwise the backbone is
# swapped for an HF-native one (use_timm_backbone=False, num_feature_levels=3)
# while KEEPING the COCO-pretrained encoder/decoder from CHECKPOINT -- only the
# backbone, the encoder_input_proj (channel counts change) and the single-class
# heads are reinitialised.
#   BACKBONE=pvt_v2_b1 INIT=imagenet RUN_NAME=... python <this>
#
# out_indices selects the last three stages (strides 8/16/32) in both families, but
# the literal list differs: ConvNextConfig.stage_names is ['stem','stage1'..'stage4']
# so [2,3,4] means stage2/3/4, while PvtV2Config.stage_names has no 'stem' entry and
# [2,3,4] raises ValueError -- [1,2,3] gives the same three stages there.
#
# ConvNeXt femto/pico are built from an explicit ConvNextConfig because HF publishes
# no checkpoint below tiny (facebook/convnext-{femto,pico}-224 do not exist), so they
# are random-only. Everything else has an HF checkpoint whose weights are grafted in.
BACKBONE = os.environ.get("BACKBONE", "")
INIT = os.environ.get("INIT", "random")     # random | imagenet | dinov3

_CONVNEXT_ARCH = {                           # timm's depths/dims for each size
    "convnext_femto": dict(depths=[2, 2, 6, 2], hidden_sizes=[48, 96, 192, 384]),
    "convnext_pico":  dict(depths=[2, 2, 6, 2], hidden_sizes=[64, 128, 256, 512]),
    "convnext_tiny":  dict(depths=[3, 3, 9, 3], hidden_sizes=[96, 192, 384, 768]),
}
_PVT_ARCH = {                                # architecture source, weights optional
    "pvt_v2_b0": "OpenGVLab/pvt_v2_b0",
    "pvt_v2_b1": "OpenGVLab/pvt_v2_b1",
    "pvt_v2_b2": "OpenGVLab/pvt_v2_b2",
}
_PRETRAINED = {                              # (backbone, init) -> HF checkpoint
    ("convnext_tiny", "imagenet"): "facebook/convnext-tiny-224",
    ("convnext_tiny", "dinov3"):   "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    ("pvt_v2_b0", "imagenet"):     "OpenGVLab/pvt_v2_b0",
    ("pvt_v2_b1", "imagenet"):     "OpenGVLab/pvt_v2_b1",
    ("pvt_v2_b2", "imagenet"):     "OpenGVLab/pvt_v2_b2",
}
# HF publishes no ConvNeXt below `tiny`, so the ImageNet arms for femto/pico can only
# come from timm. Same graph, different names -- _timm_convnext_to_hf renames the keys.
_TIMM_PRETRAINED = {
    ("convnext_femto", "imagenet"): "convnext_femto.d1_in1k",
    ("convnext_pico", "imagenet"):  "convnext_pico.d1_in1k",
}


# Plain ViTs have no feature pyramid, so RT-DETR cannot take them directly. The
# hier_encoder package under 71_misc/ bolts a ViTDet-style Simple Feature Pyramid
# onto them, producing real 8/16/32 strides. Weights come from timm -- the direct
# Meta download for dinov3 is gated (HTTP 403) and silently falls back to random.
_SFP_BACKBONES = {                          # (backbone, init) -> hier_encoder name
    ("vit_tiny", "random"):     "vit_tiny_patch16_224.augreg_in21k",
    ("vit_tiny", "imagenet"):   "vit_tiny_patch16_224.augreg_in21k",
    ("vit_small", "random"):    "vit_small_patch16_224.augreg_in21k",
    ("vit_small", "imagenet"):  "vit_small_patch16_224.augreg_in21k",
    ("vit_small", "dinov3"):    "dinov3_vits16_timm",
}
_SFP_CHANNELS = 256                          # SFP emits this on every level


def _build_sfp_model(backbone, init, id2label, label2id):
    """RT-DETR with a flat ViT + SFP neck in place of the hierarchical backbone."""
    import torch.nn.functional as F
    from transformers import AutoModelForObjectDetection, ResNetConfig, RTDetrConfig

    sys.path.insert(0, str(REPO_ROOT / "71_misc"))
    from hier_encoder import EncoderConfig, build_hierarchical_encoder

    encoder = build_hierarchical_encoder(EncoderConfig(
        backbone=_SFP_BACKBONES[(backbone, init)],
        pretrained=(init != "random"),
        freeze_backbone=False,
        feature_strategy="sfp",
        out_strides=(8, 16, 32),
        out_channels=(_SFP_CHANNELS,) * 3,
        include_stride2=False,
        img_size=(IMAGE_SIZE, IMAGE_SIZE)))

    class SFPBackbone(torch.nn.Module):
        """hier_encoder's stride-keyed dict -> RT-DETR's (feature_map, mask) pairs."""

        def __init__(self, enc):
            super().__init__()
            self.encoder = enc
            self.intermediate_channel_sizes = [_SFP_CHANNELS] * 3

        def forward(self, pixel_values, pixel_mask=None):
            out = []
            for feature_map in self.encoder(pixel_values).values():
                if pixel_mask is None:
                    mask = torch.ones(feature_map.shape[0], *feature_map.shape[-2:],
                                      dtype=torch.bool, device=feature_map.device)
                else:
                    mask = F.interpolate(pixel_mask[None].float(),
                                         size=feature_map.shape[-2:]).to(torch.bool)[0]
                out.append((feature_map, mask))
            return out

    # The COCO encoder/decoder are kept; only the backbone and the input projections
    # (channel counts change to SFP's uniform 256) are replaced.
    base = RTDetrConfig.from_pretrained(CHECKPOINT).to_dict()
    for key in ("backbone_config", "backbone", "use_timm_backbone", "use_pretrained_backbone",
                "backbone_kwargs", "num_feature_levels", "num_labels", "id2label",
                "label2id", "architectures", "model_type", "transformers_version"):
        base.pop(key, None)
    cfg = RTDetrConfig(**base, backbone_config=ResNetConfig(out_indices=[2, 3, 4]),
                       use_timm_backbone=False, num_feature_levels=3,
                       num_labels=len(id2label), id2label=id2label, label2id=label2id)
    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT, config=cfg, ignore_mismatched_sizes=True)
    model.model.backbone = SFPBackbone(encoder)
    model.model.encoder_input_proj = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Conv2d(_SFP_CHANNELS, cfg.d_model, 1, bias=False),
                            torch.nn.BatchNorm2d(cfg.d_model)) for _ in range(3)])
    print(f"  SFP backbone {_SFP_BACKBONES[(backbone, init)]} "
          f"({'pretrained' if init != 'random' else 'random'})")
    return model


def _timm_convnext_to_hf(timm_sd):
    """timm ConvNeXt state_dict -> HF ConvNextBackbone naming (pure rename)."""
    import re
    out = {}
    for key, value in timm_sd.items():
        if key.startswith("head."):               # classifier, not part of the backbone
            continue
        if key.startswith("stem."):
            index, tail = key.split(".")[1], key.split(".", 2)[2]
            name = "patch_embeddings" if index == "0" else "layernorm"
            out[f"embeddings.{name}.{tail}"] = value
            continue
        block = re.match(r"stages\.(\d+)\.blocks\.(\d+)\.(.+)", key)
        if block:
            stage, layer, tail = block.groups()
            tail = (tail.replace("gamma", "layer_scale_parameter")
                        .replace("conv_dw", "dwconv")
                        .replace("mlp.fc1", "pwconv1")
                        .replace("mlp.fc2", "pwconv2"))
            if tail.startswith("norm."):
                tail = "layernorm." + tail[len("norm."):]
            # timm keeps the pointwise convs as 1x1 Conv2d (out, in, 1, 1); HF uses
            # Linear (out, in). Identical operation, so drop the singleton dims.
            if tail.startswith(("pwconv1.weight", "pwconv2.weight")) and value.dim() == 4:
                value = value.squeeze(-1).squeeze(-1)
            out[f"encoder.stages.{stage}.layers.{layer}.{tail}"] = value
            continue
        down = re.match(r"stages\.(\d+)\.downsample\.(\d+)\.(.+)", key)
        if down:
            stage, index, tail = down.groups()
            out[f"encoder.stages.{stage}.downsampling_layer.{index}.{tail}"] = value
    return out
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "1408"))

# Few-shot: train on only K labeled pages instead of the whole train split.
# 0 = every page (the default for all the backbone-matrix runs). The pages come from
# few_shot_sampler, the same selector the one-stage pipeline uses, so a k-shot curve
# measured here is comparable with one measured there -- both see the identical pages.
# Validation and test are never subsetted.
K_SHOT = int(os.environ.get("K_SHOT", "0"))
# Validation runs on the full val split and costs the same no matter how few training
# pages there are, so at small K_SHOT it dominates the run. Raising this evaluates (and
# checkpoints) every N epochs instead of every one. Default 1 keeps the backbone-matrix
# runs bit-for-bit reproducible.
EVAL_EVERY_EPOCHS = int(os.environ.get("EVAL_EVERY_EPOCHS", "1"))
K_SHOT_METHOD = os.environ.get("K_SHOT_METHOD", "grayscale_variance")
# PCA/ICA methods cannot be recomputed here -- they need ResNet18 features over the whole
# split. make_shot_selection.py precomputes them into a text file; point at it and
# few_shot_sampler reads the block for K_SHOT_METHOD out of it.
K_SHOT_PRECOMPUTED = os.environ.get("K_SHOT_PRECOMPUTED", "") or None
K_SHOT_SEED = int(os.environ.get("K_SHOT_SEED", "42"))

NUM_EPOCHS = 100
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 2      # was 16 -> 5x more optimizer steps
LR_SCHEDULER_TYPE = "constant"       # was "cosine"
USE_EMA = True                       # was absent
EMA_DECAY = 0.993
# HF's RTDetrForObjectDetection does not implement gradient checkpointing
# (`does not support gradient checkpointing`), so activation memory cannot be
# traded for compute here -- left off.
GRADIENT_CHECKPOINTING = False
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.0
MAX_GRAD_NORM = 0.1
SEED = 42
NUM_WORKERS = 2

COLOR_JITTER = dict(brightness=0.4, contrast=0.4, saturation=0.7, hue=0.015)

SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:
    NUM_EPOCHS = 2
    # Per-run name: OUTPUT_DIR.parent is now shared by every arm, so a bare
    # "output_smoke" would make concurrent smoke runs overwrite each other.
    OUTPUT_DIR = OUTPUT_DIR.parent / f"{RUN_NAME}_smoke"

BF16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
FP16 = bool(torch.cuda.is_available() and not BF16)


# --------------------------------------------------------------------------- #
# Data (identical to the notebook)
# --------------------------------------------------------------------------- #
def load_coco_split(split, category_id_to_label):
    payload = json.loads((COCO_ROOT / f"{split}.json").read_text(encoding="utf-8"))
    annotations_by_image = defaultdict(list)
    for a in payload["annotations"]:
        annotations_by_image[a["image_id"]].append(a)

    images = payload["images"]
    if split == "train" and K_SHOT:
        # Subset the image list, not the annotations: records are built per surviving
        # image below, so dropping images drops their boxes with them.
        keep = set(select_labeled_pages(img_dir=str(IMAGE_ROOT / split),
                                        k=K_SHOT, method=K_SHOT_METHOD,
                                        precomputed_path=K_SHOT_PRECOMPUTED,
                                        seed=K_SHOT_SEED))
        images = [i for i in images if Path(i["file_name"]).stem in keep]
        if len(images) != K_SHOT:
            raise RuntimeError(
                f"K_SHOT={K_SHOT} but {len(images)} of the selected pages are in "
                f"{COCO_ROOT / 'train.json'}. Selected: {sorted(keep)}")
        print(f"[k={K_SHOT}] Labeled pages: {sorted(Path(i['file_name']).stem for i in images)}",
              flush=True)

    records = []
    for info in sorted(images, key=lambda i: i["id"]):
        width, height = int(info["width"]), int(info["height"])
        image_path = IMAGE_ROOT / split / info["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        boxes, labels, areas, ids = [], [], [], []
        for a in annotations_by_image[info["id"]]:
            x, y, bw, bh = map(float, a["bbox"])
            x, y = min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))
            bw, bh = min(bw, width - x), min(bh, height - y)
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, bw, bh])
            labels.append(category_id_to_label[int(a["category_id"])])
            areas.append(bw * bh)
            ids.append(int(a["id"]))
        if not boxes:
            raise ValueError(f"{image_path} has no valid boxes")
        records.append({
            "image_id": int(info["id"]), "image": str(image_path),
            "width": width, "height": height,
            "objects": {"id": ids, "bbox": boxes, "category": labels, "area": areas},
        })
    return Dataset.from_list(records).cast_column("image", HFImage())


def format_annotations_as_coco(image_id, categories, areas, boxes):
    return {
        "image_id": image_id,
        "annotations": [
            {"image_id": image_id, "category_id": c, "iscrowd": 0, "area": ar, "bbox": list(b)}
            for c, ar, b in zip(categories, areas, boxes)
        ],
    }


TRAIN_COLOR_JITTER = torchvision.transforms.ColorJitter(**COLOR_JITTER)


def transform_batch(examples, image_processor, augment):
    images, annotations = [], []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = image.convert("RGB")
        if augment:
            image = TRAIN_COLOR_JITTER(image)
        images.append(np.array(image, copy=True))
        annotations.append(format_annotations_as_coco(
            image_id, objects["category"], objects["area"], objects["bbox"]))
    return image_processor(images=images, annotations=annotations, return_tensors="pt")


def collate_fn(batch):
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": [b["labels"] for b in batch],
    }
    if "pixel_mask" in batch[0]:
        out["pixel_mask"] = torch.stack([b["pixel_mask"] for b in batch])
    return out


# --------------------------------------------------------------------------- #
# Metrics (identical to the notebook)
# --------------------------------------------------------------------------- #
@dataclass
class DetectionOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


def original_size(target):
    values = np.atleast_1d(np.asarray(target["orig_size"])).flatten()
    if len(values) < 2:
        raise ValueError(f"Invalid orig_size: {target['orig_size']}")
    return int(values[0]), int(values[1])


def normalized_cxcywh_to_absolute_xyxy(boxes, image_size):
    height, width = image_size
    boxes = center_to_corners_format(boxes)
    return boxes * torch.tensor([[width, height, width, height]], dtype=boxes.dtype)


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor):
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids
    image_sizes, processed_targets, processed_predictions = [], [], []
    for target_batch in targets:
        batch_sizes = []
        for target in target_batch:
            size = original_size(target)
            batch_sizes.append(size)
            processed_targets.append({
                "boxes": normalized_cxcywh_to_absolute_xyxy(torch.as_tensor(target["boxes"]), size),
                "labels": torch.as_tensor(target["class_labels"], dtype=torch.int64),
            })
        image_sizes.append(torch.tensor(batch_sizes))
    for prediction_batch, target_sizes in zip(predictions, image_sizes):
        outputs = DetectionOutput(
            logits=torch.as_tensor(prediction_batch[1]),
            pred_boxes=torch.as_tensor(prediction_batch[2]),
        )
        processed_predictions.extend(image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes))
    # max_detection_thresholds: the default [1, 10, 100] caps AP and AR at 100
    # detections per image. CS18 carries 141-283 GT lines per test page and CS863 up
    # to 174, so the metric would run out of detection slots before the detector runs
    # out of lines and every arm would look far worse than it is. 300 is RT-DETR's
    # num_queries, i.e. the most it can emit.
    #
    # backend: pycocotools returns map = -1 for ANY non-default threshold list -- its
    # summarize() only ever looks up maxDets == 100 -- which silently voids the
    # mAP50-95 column. faster_coco_eval handles it, and agrees with pycocotools
    # exactly at the default cap. Both verified on a toy case before this change.
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False,
                                  max_detection_thresholds=MAX_DETS,
                                  backend="faster_coco_eval")
    metric.update(processed_predictions, processed_targets)
    return {k: round(v.item(), 6) for k, v in metric.compute().items()}


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #
class EMACallback(TrainerCallback):
    """Keeps an exponential moving average of the float params, updated once per
    optimizer step (not per micro-batch)."""

    def __init__(self, model, decay):
        self.decay = decay
        # Built lazily on the first optimizer step: at __init__ time the model is
        # still on CPU, and Trainer moves it to the accelerator afterwards.
        # Kept on CPU -- a GPU-resident fp32 copy of RT-DETR is ~170MB, which does
        # not fit alongside training at IMAGE_SIZE=1408 on this 8GB card. The
        # transfer cost is negligible at ~10 optimizer steps per epoch.
        self.shadow = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        with torch.no_grad():
            msd = model.state_dict()
            if self.shadow is None:
                self.shadow = {
                    k: v.detach().float().cpu().clone()
                    for k, v in msd.items() if v.is_floating_point()
                }
                return
            for k, shadow in self.shadow.items():
                shadow.mul_(self.decay).add_(msd[k].detach().float().cpu(), alpha=1.0 - self.decay)

    def copy_to(self, model):
        if self.shadow is None:
            raise RuntimeError("EMA shadow was never initialised -- no optimizer step ran")
        with torch.no_grad():
            msd = model.state_dict()
            for k, shadow in self.shadow.items():
                msd[k].copy_(shadow.to(device=msd[k].device, dtype=msd[k].dtype))


def build_model(id2label, label2id):
    """Stock checkpoint, or the same checkpoint with its backbone swapped.

    from_pretrained(config=...) keeps the COCO encoder/decoder and reinitialises
    only the shape-mismatched parts. When INIT is not "random" the backbone weights
    are then grafted from an HF checkpoint, asserting every key matched -- silently
    training a random backbone while reporting it as pretrained is the failure this
    guards against.
    """
    if not BACKBONE:
        if INIT != "random":
            raise SystemExit("stock backbone has no INIT arm; unset INIT or set BACKBONE")
        return AutoModelForObjectDetection.from_pretrained(
            CHECKPOINT, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

    from transformers import AutoBackbone, AutoConfig, ConvNextConfig, PvtV2Config, RTDetrConfig

    if (BACKBONE, INIT) in _SFP_BACKBONES:
        return _build_sfp_model(BACKBONE, INIT, id2label, label2id)

    ckpt = _PRETRAINED.get((BACKBONE, INIT))
    timm_ckpt = _TIMM_PRETRAINED.get((BACKBONE, INIT))
    if INIT != "random" and ckpt is None and timm_ckpt is None:
        raise SystemExit(
            f"no checkpoint for {BACKBONE!r}+{INIT!r}. Known HF: {sorted(_PRETRAINED)}; "
            f"known timm: {sorted(_TIMM_PRETRAINED)}.")

    if BACKBONE in _CONVNEXT_ARCH:
        # hidden_sizes forced to a list: ConvNextBackbone does
        # `[config.hidden_sizes[0]] + config.hidden_sizes`, which raises
        # "can only concatenate list (not tuple) to list" on the tuple default.
        bb = (AutoConfig.from_pretrained(ckpt) if ckpt
              else ConvNextConfig(**_CONVNEXT_ARCH[BACKBONE]))
        bb.out_indices = [2, 3, 4]
    elif BACKBONE.startswith("pvt_v2_"):
        # ARCHITECTURE always comes from the size-specific HF config, weights only
        # when ckpt is set. PvtV2Config() defaults to b0 (hidden_sizes 32/64/160/256,
        # depths 2/2/2/2), so using it for the random arm silently trained a b0 and
        # reported it as b1 -- 3.41M params where b1 has 13.50M.
        bb = AutoConfig.from_pretrained(_PVT_ARCH[BACKBONE])
        bb.out_indices = [1, 2, 3]
    else:
        raise SystemExit(f"unknown BACKBONE {BACKBONE!r}; expected '' or one of "
                         f"{sorted(set(_CONVNEXT_ARCH) | {'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2'})}")
    if isinstance(getattr(bb, "hidden_sizes", None), tuple):
        bb.hidden_sizes = list(bb.hidden_sizes)

    cfg = RTDetrConfig.from_pretrained(
        CHECKPOINT, backbone_config=bb, use_timm_backbone=False, num_feature_levels=3,
        num_labels=len(id2label), id2label=id2label, label2id=label2id)
    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT, config=cfg, ignore_mismatched_sizes=True)

    if timm_ckpt:
        import timm
        source = timm.create_model(timm_ckpt, pretrained=True).state_dict()
        mapped = _timm_convnext_to_hf(source)
        inner = model.model.backbone
        inner = inner.model if hasattr(inner, "model") else inner
        missing, unexpected = inner.load_state_dict(mapped, strict=False)
        # hidden_states_norms have no timm counterpart; they are freshly initialised
        # for the HF-native grafts too (facebook/convnext-tiny-224 ships them at 1.0/0.0).
        leftover = [k for k in missing if "hidden_states_norm" not in k]
        if leftover or unexpected:
            raise RuntimeError(
                f"timm graft mismatch for {timm_ckpt}: {len(leftover)} missing, "
                f"{len(unexpected)} unexpected (first missing: {leftover[:3]})")
        key = next(k for k, v in mapped.items() if v.dim() > 1)
        assert torch.allclose(inner.state_dict()[key], mapped[key]), "timm graft did not take"
        print(f"  grafted {INIT} weights from timm:{timm_ckpt}")
    elif ckpt:
        ref = AutoBackbone.from_pretrained(ckpt, out_indices=bb.out_indices)
        inner = model.model.backbone
        inner = inner.model if hasattr(inner, "model") else inner
        missing, unexpected = inner.load_state_dict(ref.state_dict(), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"backbone graft mismatch for {ckpt}: {len(missing)} missing, "
                f"{len(unexpected)} unexpected (first missing: {list(missing)[:3]})")
        k = next(k for k, v in ref.state_dict().items() if v.dim() > 1)
        assert torch.allclose(inner.state_dict()[k], ref.state_dict()[k]), "graft did not take"
        print(f"  grafted {INIT} weights from {ckpt}")
        del ref
    return model


def freeze_stray_batch_norms(model):
    """See the notebook: with per-device batch 1, live BatchNorm2d layers recompute
    statistics from a single sample every step. Swap them for the frozen variant."""
    before = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    _rtdetr_replace_batch_norm(model)
    after = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    return before - after


def main():
    set_seed(SEED)
    print(f"RT-DETR CB55 | epochs={NUM_EPOCHS} accum={GRADIENT_ACCUMULATION_STEPS} "
          f"sched={LR_SCHEDULER_TYPE} ema={USE_EMA} bf16={BF16} smoke={SMOKE}")

    cat_payload = json.loads((COCO_ROOT / "train.json").read_text(encoding="utf-8"))
    categories = sorted(cat_payload["categories"], key=lambda c: c["id"])
    category_id_to_label = {int(c["id"]): i for i, c in enumerate(categories)}
    id2label = {i: c["name"] for i, c in enumerate(categories)}
    label2id = {n: i for i, n in id2label.items()}

    raw = DatasetDict({s: load_coco_split(s, category_id_to_label) for s in ("train", "val", "test")})
    for split, ds in raw.items():
        print(f"  {split}: {len(ds)} pages, {sum(len(o['bbox']) for o in ds['objects'])} boxes")

    image_processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT, do_resize=True,
        size={"max_height": IMAGE_SIZE, "max_width": IMAGE_SIZE},
        do_pad=True, pad_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    )
    datasets = DatasetDict({
        s: ds.with_transform(partial(transform_batch, image_processor=image_processor,
                                     augment=(s == "train")))
        for s, ds in raw.items()
    })

    model = build_model(id2label, label2id)
    print(f"  froze {freeze_stray_batch_norms(model)} live BatchNorm2d layer(s)")
    n_bb = sum(p.numel() for p in model.model.backbone.parameters())
    print(f"  backbone={BACKBONE or 'stock r50vd'}  backbone_params={n_bb:,}  "
          f"total={sum(p.numel() for p in model.parameters()):,}")

    steps_per_epoch = max(1, len(raw["train"]) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
    print(f"  ~{steps_per_epoch} optimizer steps/epoch -> ~{steps_per_epoch * NUM_EPOCHS} total")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim="adamw_torch",
        max_grad_norm=MAX_GRAD_NORM,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        bf16=BF16, fp16=FP16,
        dataloader_num_workers=NUM_WORKERS,
        # load_best_model_at_end requires eval and save strategies to match, so both
        # switch together.
        **(dict(eval_strategy="epoch", save_strategy="epoch")
           if EVAL_EVERY_EPOCHS <= 1 else
           dict(eval_strategy="steps", save_strategy="steps",
                eval_steps=steps_per_epoch * EVAL_EVERY_EPOCHS,
                save_steps=steps_per_epoch * EVAL_EVERY_EPOCHS)),
        logging_strategy="epoch",
        metric_for_best_model="eval_map", greater_is_better=True,
        load_best_model_at_end=True, save_total_limit=2,
        remove_unused_columns=False, eval_do_concat_batches=False,
        report_to="none", seed=SEED, data_seed=SEED,
    )

    ema_cb = EMACallback(model, EMA_DECAY) if USE_EMA else None
    trainer = Trainer(
        model=model, args=args,
        train_dataset=datasets["train"], eval_dataset=datasets["val"],
        processing_class=image_processor, data_collator=collate_fn,
        compute_metrics=partial(compute_metrics, image_processor=image_processor),
        callbacks=[ema_cb] if ema_cb else None,
    )

    train_result = trainer.train()

    # load_best_model_at_end already restored the best raw checkpoint.
    raw_val = trainer.evaluate(datasets["val"], metric_key_prefix="validation")
    print(f"  raw best  val mAP50={raw_val['validation_map_50']:.4f} "
          f"mAP50-95={raw_val['validation_map']:.4f}")

    use_ema_weights = False
    if ema_cb is not None:
        raw_state = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        ema_cb.copy_to(trainer.model)
        ema_val = trainer.evaluate(datasets["val"], metric_key_prefix="validation")
        print(f"  EMA       val mAP50={ema_val['validation_map_50']:.4f} "
              f"mAP50-95={ema_val['validation_map']:.4f}")
        if ema_val["validation_map"] > raw_val["validation_map"]:
            use_ema_weights, val_metrics = True, ema_val
            print("  -> EMA weights win, keeping them")
        else:
            trainer.model.load_state_dict(raw_state)
            val_metrics = raw_val
            print("  -> raw weights win, restoring them")
    else:
        val_metrics = raw_val

    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    print(f"  TEST mAP50={test_metrics['test_map_50']:.4f} "
          f"mAP50-95={test_metrics['test_map']:.4f}")

    best_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(str(best_dir))
    image_processor.save_pretrained(str(best_dir))

    n_params = sum(p.numel() for p in trainer.model.parameters())
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_backbone = sum(p.numel() for p in trainer.model.model.backbone.parameters())
    summary = {
        "model": "RT-DETR", "checkpoint": CHECKPOINT,
        "run_name": RUN_NAME,
        "dataset": DATASET,
        "backbone": BACKBONE or "stock r50vd",
        "init": INIT,
        "params_backbone": n_backbone,
        # torchmetrics names the recall key after the largest cap, so raising
        # max_detection_thresholds renames mar_100 -> mar_300. Keyed off the constant
        # rather than hardcoded, or the field silently turns into None.
        f"test_mAR@{MAX_DETS[-1]}": test_metrics.get(f"test_mar_{MAX_DETS[-1]}"),
        f"validation_mAR@{MAX_DETS[-1]}": val_metrics.get(f"validation_mar_{MAX_DETS[-1]}"),
        "weights_used": "ema" if use_ema_weights else "raw",
        "validation_mAP@50": val_metrics["validation_map_50"],
        "validation_mAP@50-95": val_metrics["validation_map"],
        "test_mAP@50": test_metrics["test_map_50"],
        "test_mAP@50-95": test_metrics["test_map"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "epochs": NUM_EPOCHS,
        "optimizer_steps": trainer.state.global_step,
        "lr_scheduler": LR_SCHEDULER_TYPE,
        "grad_accum": GRADIENT_ACCUMULATION_STEPS,
        "params_total": n_params, "params_trainable": n_trainable,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
    }
    (OUTPUT_DIR / "metrics_summary.json").write_text(json.dumps(
        {"summary": summary, "validation": val_metrics, "test": test_metrics}, indent=2))
    print(json.dumps(summary, indent=2))

    if not SMOKE:
        append_results_row(summary, trainer.state)

    del trainer, model, datasets
    gc.collect()
    torch.cuda.empty_cache()


def append_results_row(summary, state):
    import csv
    header = ["platform", "model", "backbone", "init", "run", "pretrain", "resolution", "epochs_to_best",
              "val_mAP50", "val_mAP50_95", "test_mAP50", "test_mAP50_95",
              "params_total", "params_trainable", "notes"]
    best_epoch = ""
    for rec in state.log_history:
        if rec.get("eval_map") == summary["validation_mAP@50-95"]:
            best_epoch = int(rec.get("epoch", 0))
            break
    row = {
        "platform": RUN_NAME, "model": CHECKPOINT.split("/")[-1],
        "backbone": f"{summary['backbone']} ({summary['params_backbone']:,} params)",
        "init": summary["init"],
        "run": summary["run_name"],
        "pretrain": "COCO full detector", "resolution": IMAGE_SIZE,
        "epochs_to_best": best_epoch,
        "val_mAP50": summary["validation_mAP@50"],
        "val_mAP50_95": summary["validation_mAP@50-95"],
        "test_mAP50": summary["test_mAP@50"],
        "test_mAP50_95": summary["test_mAP@50-95"],
        "params_total": summary["params_total"],
        "params_trainable": summary["params_trainable"],
        "notes": (f"HF Trainer controlled loop (shared with the other HF row): "
                  f"accum={GRADIENT_ACCUMULATION_STEPS} ({summary['optimizer_steps']} steps "
                  f"vs 200 in the earlier run), lr_sched=constant (was cosine->0), EMA on; "
                  f"weights_used={summary['weights_used']}"),
    }
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    if not write_header:
        # DictWriter appends positionally and will not notice that an existing file
        # was written under a different schema -- every later row silently shifts by
        # the number of added columns. Widening `header` above must be matched by
        # migrating the file, so fail here instead of corrupting it.
        with open(RESULTS_CSV, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != header:
            raise RuntimeError(
                f"{RESULTS_CSV} has header {existing}, script writes {header}. "
                "Migrate the file before appending.")
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"appended results row to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
