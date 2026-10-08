#!/usr/bin/env python3
"""Train/evaluate a controlled Mask R-CNN backbone pilot on DIVA-HisDB.

The experimental boundary mirrors the other thesis pipelines:

    image -> encoder -> P2/P3/P4/P5 -> common 256-channel FPN
          -> fixed RPN + ROI box head + ROI mask head

Hierarchical timm encoders expose their native four stages. Flat ViTs reuse the
shared ``hier_encoder`` Simple Feature Pyramid used by the U-Net and RT-DETR
experiments. The FPN and every downstream module have the same architecture for
all arms. COCO TASK-2 is used because it describes exactly the main-text lines
scored by the official DIVA evaluator.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import torchvision.transforms.functional as TF
import torchvision.models.detection.roi_heads as tv_roi_heads
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign, roi_align
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from timm.models._features import FeatureListNet


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HIER_PARENT = REPO / "50_modelling/common"
if str(HIER_PARENT) not in sys.path:
    sys.path.insert(0, str(HIER_PARENT))
MODELLING_ROOT = REPO / "50_modelling"
if str(MODELLING_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELLING_ROOT))

from few_shot_sampler import select_labeled_pages

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"


@dataclass(frozen=True)
class Arm:
    model_name: str | None
    kind: str
    pretrained: bool
    out_indices: tuple[int, ...] = (0, 1, 2, 3)
    backbone_state: str | None = None


ARMS = {
    # Stock COCO model is an external baseline, matching the treatment of stock
    # R50-vd in the RT-DETR matrix.
    "stock": Arm(None, "stock", True),
    "convnext_tiny_imagenet": Arm("convnext_tiny.in12k_ft_in1k", "timm", True),
    "convnext_tiny_random": Arm("convnext_tiny.in12k_ft_in1k", "timm", False),
    "convnext_tiny_dinov3": Arm("convnext_tiny.dinov3_lvd1689m", "timm", True),
    "pvt_v2_b2_imagenet": Arm("pvt_v2_b2.in1k", "timm", True),
    "pvt_v2_b2_random": Arm("pvt_v2_b2.in1k", "timm", False),
    "vit_small_imagenet": Arm("vit_small_patch16_224.augreg_in21k", "hier", True),
    "vit_small_random": Arm("vit_small_patch16_224.augreg_in21k", "hier", False),
    "vit_small_dinov3": Arm("dinov3_vits16_timm", "hier", True),
    # Encoder-size study (XS/S brackets), same recipe as the M-bracket arms above.
    "convnext_femto_imagenet": Arm("convnext_femto.d1_in1k", "timm", True),
    "convnext_femto_random": Arm("convnext_femto.d1_in1k", "timm", False),
    "convnext_pico_imagenet": Arm("convnext_pico.d1_in1k", "timm", True),
    "convnext_pico_random": Arm("convnext_pico.d1_in1k", "timm", False),
    "pvt_v2_b0_imagenet": Arm("pvt_v2_b0.in1k", "timm", True),
    "pvt_v2_b0_random": Arm("pvt_v2_b0.in1k", "timm", False),
    "pvt_v2_b1_imagenet": Arm("pvt_v2_b1.in1k", "timm", True),
    "pvt_v2_b1_random": Arm("pvt_v2_b1.in1k", "timm", False),
    "vit_tiny_imagenet": Arm("vit_tiny_patch16_224.augreg_in21k", "hier", True),
    "vit_tiny_random": Arm("vit_tiny_patch16_224.augreg_in21k", "hier", False),
    # DINOv3-teacher distillation on unlabeled cBAD pages.  These paths contain
    # plain backbone state dicts exported by the three Colab notebooks.  They
    # are loaded strictly into the original timm graph before an FPN/SFP is
    # attached; accepting missing keys would invalidate the comparison.
    "convnext_tiny_cbad": Arm(
        "convnext_tiny", "timm", False, backbone_state=
        "convnext_tiny_cbad_dinov3_epoch1100.pt",
    ),
    "pvt_v2_b2_cbad": Arm(
        "pvt_v2_b2", "timm", False, backbone_state=
        "pvtv2_b2_cbad_dinov3_epoch300.pt",
    ),
    "vit_small_cbad": Arm(
        "dinov3_vits16_timm", "hier", False, backbone_state=
        "vit_small_cbad_dinov3_epoch300.pt",
    ),
    # "CATMuS" initialisation: the same graph as the random arm, then the full
    # Mask R-CNN state from pretrain_maskrcnn_catmus.py is loaded on top
    # (see resolve_init_checkpoint / --init-checkpoint).
    "convnext_tiny_catmus": Arm("convnext_tiny.in12k_ft_in1k", "timm", False),
    "pvt_v2_b2_catmus": Arm("pvt_v2_b2.in1k", "timm", False),
    "vit_small_catmus": Arm("vit_small_patch16_224.augreg_in21k", "hier", False),
}

CATMUS_PRETRAIN_ROOT = REPO / "80_models/instance_segmentation/mask_rcnn/catmus_pretrain"
CBAD_BACKBONE_ROOT = REPO / "80_models/instance_segmentation/mask_rcnn/cbad_distilled_backbones"
# FIXED: train-only scale jitter (RQ3 robustness ablation). Default off = previous behaviour.
# With probability 0.5 the page content is enlarged by s ~ U(1.0, 1.6) and a random
# size x size crop is taken (instances keep >= 50 % of their mask area or are dropped);
# otherwise it is shrunk by s ~ U(0.8, 1.0) and fitted into the canvas. Never used at eval.
SCALE_JITTER = os.environ.get("RQ3_SCALE_JITTER", "0") == "1"
# FIXED: upper bound of the enlargement factor (default 1.6; the 2.0 variant is an ablation).
SCALE_JITTER_UP = (1.0, float(os.environ.get("RQ3_JITTER_UP_MAX", "1.6")))
SCALE_JITTER_DOWN = (0.8, 1.0)
SCALE_JITTER_MIN_KEEP = 0.5


def serializable_args(args) -> dict:
    """Checkpoints are reloaded with weights_only=True, which rejects pathlib objects."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def resolve_init_checkpoint(arm: str, explicit: Path | None) -> Path | None:
    """The CATMuS arms default to their pretrain's best.pt; other arms need none."""
    if explicit is not None:
        return explicit
    if arm.endswith("_catmus"):
        return CATMUS_PRETRAIN_ROOT / f"maskrcnn_{arm[: -len('_catmus')]}_704" / "best.pt"
    return None


def load_init_checkpoint(model, path: Path) -> int:
    """Load a full Mask R-CNN state (same graph) and return its epoch."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    print(f"  init from {path} (epoch {state.get('epoch')})", flush=True)
    return int(state.get("epoch", 0))


_TORCHVISION_MASKRCNN_LOSS = tv_roi_heads.maskrcnn_loss


def configure_mask_loss(name: str, device: torch.device) -> None:
    """Replace only Torchvision's ROI mask objective for this process."""
    if name == "bce":
        criterion = None
    elif name == "tversky":
        from segmentation_models_pytorch.losses import TverskyLoss

        criterion = TverskyLoss(
            mode="binary", alpha=0.3, beta=0.7, from_logits=True
        )
    elif name == "supervoxel":
        # Python >=3.11 rejects sets in random.sample. Upstream imports sample
        # directly, so use the same compatibility shim as the crop segmenter.
        import supervoxel_loss.critical_detection_2d as critical_detection_2d
        from supervoxel_loss.loss import SuperVoxelLoss2D

        critical_detection_2d.sample = lambda population, k: random.sample(
            tuple(population), k
        )

        criterion = SuperVoxelLoss2D(alpha=0.5, beta=0.5, device=str(device))
    else:
        raise ValueError(f"unknown mask loss: {name}")

    def custom_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        discretization_size = tuple(mask_logits.shape[-2:])
        labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
        mask_targets = []
        for masks, boxes, indices in zip(gt_masks, proposals, mask_matched_idxs):
            indices = indices.to(boxes)
            rois = torch.cat((indices[:, None], boxes), dim=1)
            mask_targets.append(
                roi_align(masks[:, None].to(rois), rois, discretization_size, 1.0)[:, 0]
            )
        labels = torch.cat(labels, dim=0)
        mask_targets = torch.cat(mask_targets, dim=0)
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0
        selected_logits = mask_logits[
            torch.arange(labels.shape[0], device=labels.device), labels
        ]
        if criterion is None:
            return F.binary_cross_entropy_with_logits(selected_logits, mask_targets)
        # Both repository losses expect NCHW binary masks.
        return criterion(selected_logits[:, None], mask_targets[:, None])

    tv_roi_heads.maskrcnn_loss = custom_maskrcnn_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DivaCocoLines(Dataset):
    """Rasterize COCO polygons directly at the square model resolution."""

    def __init__(
        self,
        coco_dir: Path,
        image_root: Path,
        split: str,
        size: int,
        augment: bool,
        include_stems: set[str] | None = None,
    ):
        payload = json.loads((coco_dir / f"{split}.json").read_text(encoding="utf-8"))
        grouped: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            grouped[int(annotation["image_id"])].append(annotation)
        self.records = [
            (image_root / split / item["file_name"], item, grouped[int(item["id"])])
            for item in sorted(payload["images"], key=lambda value: int(value["id"]))
            if include_stems is None or Path(item["file_name"]).stem in include_stems
        ]
        if include_stems is not None:
            found = {path.stem for path, _, _ in self.records}
            missing = include_stems - found
            if missing:
                raise ValueError(f"selected pages missing from COCO {split} split: {sorted(missing)}")
        self.size = int(size)
        self.augment = augment
        # PageForge (train only): with probability RQ3_FORGE a page is replaced by a page
        # recomposed from the lines of all training pages. 0 (default) = previous behaviour.
        self.forge_p = float(os.environ.get("RQ3_FORGE", "0")) if augment else 0.0
        self.forge = None
        self.forge_deg_p = 0.0
        if augment and os.environ.get("RQ3_FORGE_CFG"):
            from page_forge import CFG
            self.forge_deg_p = float(CFG["deg_p"])
        if self.forge_p > 0:
            from page_forge import PageForge
            self.forge = PageForge(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, info, annotations = self.records[index]
        if self.forge is not None and random.random() < self.forge_p:
            image, annotations = self.forge.sample(random)
        else:
            image = Image.open(path).convert("RGB")
        orig_w, orig_h = image.size
        scale = min(self.size / orig_w, self.size / orig_h)
        # FIXED: optional train-only scale jitter; without it scale/offsets equal the plain fit.
        jitter = self.augment and SCALE_JITTER
        cropped = False
        if jitter:
            up = random.random() < 0.5
            scale *= random.uniform(*(SCALE_JITTER_UP if up else SCALE_JITTER_DOWN))
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        image_tensor = TF.pil_to_tensor(image).float().div_(255.0)
        image_tensor = TF.resize(image_tensor, [new_h, new_w], antialias=True)
        off_x = off_y = 0
        if jitter and (new_w > self.size or new_h > self.size):
            off_x = random.randint(0, max(0, new_w - self.size))
            off_y = random.randint(0, max(0, new_h - self.size))
            image_tensor = image_tensor[:, off_y:off_y + self.size, off_x:off_x + self.size]
            new_h, new_w = image_tensor.shape[1:]
            cropped = True
        canvas = torch.ones((3, self.size, self.size), dtype=torch.float32)
        canvas[:, :new_h, :new_w] = image_tensor

        # Photometric augmentation only: geometry and instance masks stay aligned.
        if self.augment:
            brightness = random.uniform(0.6, 1.4)
            contrast = random.uniform(0.6, 1.4)
            saturation = random.uniform(0.3, 1.7)
            hue = random.uniform(-0.015, 0.015)
            canvas = TF.adjust_brightness(canvas, brightness)
            canvas = TF.adjust_contrast(canvas, contrast)
            canvas = TF.adjust_saturation(canvas, saturation)
            canvas = TF.adjust_hue(canvas, hue).clamp_(0, 1)
            if self.forge_deg_p and random.random() < self.forge_deg_p:  # PageForge degradations
                from page_forge import degrade
                arr = degrade(canvas.permute(1, 2, 0).numpy().copy(), random)
                canvas = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        masks = []
        boxes = []
        for annotation in annotations:
            mask = np.zeros((self.size, self.size), dtype=np.uint8)
            full_area = 0.0
            segmentation = annotation.get("segmentation", [])
            if isinstance(segmentation, list):
                for polygon in segmentation:
                    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                    if len(points) < 3:
                        continue
                    points[:, 0] *= scale
                    points[:, 1] *= scale
                    if cropped:  # FIXED: shift into the random crop
                        points[:, 0] -= off_x
                        points[:, 1] -= off_y
                        full_area += abs(float(cv2.contourArea(points)))
                    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            # FIXED: after cropping, keep only instances with enough of their mask inside.
            if cropped and mask.sum() < SCALE_JITTER_MIN_KEEP * full_area:
                continue
            x1, y1 = float(xs.min()), float(ys.min())
            # Torchvision boxes are half-open at the right/bottom boundary.
            x2 = float(min(self.size, xs.max() + 1))
            y2 = float(min(self.size, ys.max() + 1))
            if x2 <= x1 or y2 <= y1:
                continue
            masks.append(torch.from_numpy(mask))
            boxes.append([x1, y1, x2, y2])

        masks_tensor = (
            torch.stack(masks).to(torch.uint8)
            if masks else torch.zeros((0, self.size, self.size), dtype=torch.uint8)
        )
        target = {
            "boxes": (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": masks_tensor,
            "image_id": torch.tensor(int(info["id"]), dtype=torch.int64),
            "area": torch.tensor([float(mask.sum()) for mask in masks], dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        meta = {
            "path": str(path),
            "orig_w": orig_w,
            "orig_h": orig_h,
            "new_w": new_w,
            "new_h": new_h,
        }
        return canvas, target, meta


def collate(batch):
    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


class CommonFPNBackbone(nn.Module):
    """Adapt timm or the shared SFP encoder to Mask R-CNN's backbone API."""

    def __init__(self, arm: Arm):
        super().__init__()
        self.kind = arm.kind
        if arm.kind == "timm":
            if arm.backbone_state:
                base = timm.create_model(
                    arm.model_name, pretrained=False, num_classes=0,
                )
                state_path = CBAD_BACKBONE_ROOT / arm.backbone_state
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                base.load_state_dict(state, strict=True)
                loaded = sum(tensor.numel() for tensor in state.values())
                expected = sum(parameter.numel() for parameter in base.parameters())
                if loaded != expected:
                    raise RuntimeError(
                        f"cBAD backbone tensor count mismatch: {loaded:,} != {expected:,}"
                    )
                # timm's normal features_only path rewrites module names.  Wrap
                # only after the exact original state has been verified/loaded.
                self.encoder = FeatureListNet(
                    base, out_indices=arm.out_indices, flatten_sequential=True,
                )
                print(f"  strict cBAD backbone load: {state_path} ({loaded:,} params)",
                      flush=True)
            else:
                self.encoder = timm.create_model(
                    arm.model_name,
                    pretrained=arm.pretrained,
                    features_only=True,
                    out_indices=arm.out_indices,
                )
            in_channels = list(self.encoder.feature_info.channels())
            reductions = tuple(self.encoder.feature_info.reduction())
        elif arm.kind == "hier":
            from hier_encoder import EncoderConfig, build_hierarchical_encoder

            self.encoder = build_hierarchical_encoder(EncoderConfig(
                backbone=arm.model_name,
                pretrained=arm.pretrained,
                freeze_backbone=False,
                feature_strategy="sfp",
                out_strides=(4, 8, 16, 32),
                out_channels=(256, 256, 256, 256),
                include_stride2=False,
                input_norm="none",  # GeneralizedRCNNTransform already normalizes.
                img_size=None,
            ))
            if arm.backbone_state:
                state_path = CBAD_BACKBONE_ROOT / arm.backbone_state
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                inner = self.encoder.backbone
                inner = getattr(inner, "model", inner)
                inner.load_state_dict(state, strict=True)
                loaded = sum(tensor.numel() for tensor in state.values())
                expected = sum(parameter.numel() for parameter in inner.parameters())
                if loaded != expected:
                    raise RuntimeError(
                        f"cBAD backbone tensor count mismatch: {loaded:,} != {expected:,}"
                    )
                print(f"  strict cBAD backbone load: {state_path} ({loaded:,} params)",
                      flush=True)
            in_channels = list(self.encoder.out_channels)
            reductions = tuple(self.encoder.strides)
            inner = self.encoder.backbone
            inner = getattr(inner, "model", inner)
            if hasattr(inner, "set_grad_checkpointing"):
                inner.set_grad_checkpointing(True)
        else:
            raise ValueError(f"unsupported arm kind: {arm.kind}")
        if reductions != (4, 8, 16, 32):
            raise ValueError(f"expected reductions (4,8,16,32), got {reductions}")

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = 256
        self.input_channels = tuple(in_channels)
        self.reductions = reductions

    def forward(self, images: torch.Tensor):
        if self.kind == "timm":
            values = self.encoder(images)
        else:
            values = list(self.encoder(images).values())
        features = OrderedDict((str(i), value) for i, value in enumerate(values))
        return self.fpn(features)


class CheckpointedMaskHead(nn.Module):
    """Activation-checkpoint an unchanged Torchvision mask feature head."""

    def __init__(self, head: nn.Sequential):
        super().__init__()
        self.head = head

    def forward(self, features: torch.Tensor):
        if self.training and torch.is_grad_enabled():
            return checkpoint_sequential(
                self.head, segments=4, input=features, use_reentrant=False
            )
        return self.head(features)


def build_model(arm_name: str, image_size: int, mask_roi_size: int = 14,
                mask_roi_width: int = 0, roi_batch_size: int = 512,
                checkpoint_mask_head: bool = False):
    arm = ARMS[arm_name]
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=image_size,
        max_size=image_size,
        box_detections_per_img=100,
    )

    downstream_before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("backbone.")
    }
    if arm.kind != "stock":
        model.backbone = CommonFPNBackbone(arm)
        after = model.state_dict()
        changed = [
            key for key, value in downstream_before.items()
            if key not in after or not torch.equal(value, after[key].cpu())
        ]
        if changed:
            raise RuntimeError(f"backbone swap changed downstream tensors: {changed[:5]}")

    # All arms use the same line-oriented anchors and a newly initialized RPN
    # predictor. Ratios are height/width in Torchvision.
    ratios = (0.05, 0.1, 0.2, 0.5, 1.0)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=(ratios,) * 5,
    )
    model.rpn.head = RPNHead(model.backbone.out_channels, len(ratios), conv_depth=1)

    # Background + one TextLine class. Only the task-specific output predictors
    # are reset; the COCO-pretrained ROI box and mask feature heads are retained.
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, 2)
    if checkpoint_mask_head:
        model.roi_heads.mask_head = CheckpointedMaskHead(model.roi_heads.mask_head)
    # Torchvision's default 14x14 ROI crop produces only 28x28 mask logits.
    # Larger crops retain substantially more boundary/detail information while
    # leaving the shared mask feature head and its pretrained weights unchanged.
    mask_roi_shape = (mask_roi_size, mask_roi_width or mask_roi_size)
    if mask_roi_shape != (14, 14):
        model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=mask_roi_shape,
            sampling_ratio=2,
        )
    model.roi_heads.detections_per_img = 100
    model.roi_heads.batch_size_per_image = roi_batch_size
    model.roi_heads.score_thresh = 0.0  # retain low scores for AP/threshold tuning
    return model


@torch.no_grad()
def evaluate_coco(model, loader, device, max_pages: int = 0, max_detections: int = 100):
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type=("bbox", "segm"),
        max_detection_thresholds=[1, 10, max_detections],
        backend="pycocotools",
    )
    for page_index, (images, targets, _) in enumerate(loader):
        if max_pages and page_index >= max_pages:
            break
        images_device = [image.to(device, non_blocking=True) for image in images]
        with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            predictions = model(images_device)
        pred_cpu = []
        target_cpu = []
        for prediction, target in zip(predictions, targets):
            pred_cpu.append({
                "boxes": prediction["boxes"].detach().float().cpu(),
                "scores": prediction["scores"].detach().float().cpu(),
                "labels": prediction["labels"].detach().cpu(),
                "masks": (prediction["masks"].detach().cpu()[:, 0] >= 0.5),
            })
            target_cpu.append({
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
                "masks": target["masks"].bool().cpu(),
            })
        metric.update(pred_cpu, target_cpu)
        del predictions, images_device, pred_cpu, target_cpu
    result = metric.compute()
    return {
        key: float(value.item())
        for key, value in result.items()
        if value.numel() == 1
    }


def train_one_epoch(model, loader, optimizer, scaler, device, max_steps: int = 0):
    model.train()
    total = 0.0
    components: dict[str, float] = defaultdict(float)
    steps = 0
    skipped_nonfinite = 0
    for step, (images, targets, _) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [
            {key: value.to(device, non_blocking=True) for key, value in target.items()}
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        # BF16 has FP32-like exponent range and avoids the ROI-classifier
        # overflows observed during long CATMuS runs, at the same memory cost
        # as FP16 on supported NVIDIA GPUs.
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            # Rarely a very dense/high-resolution page can still destabilize an
            # ROI operation. The checkpoint remains finite; skip
            # that page instead of discarding the complete multi-hour epoch.
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"WARNING: skipping non-finite training batch {step}: "
                f"{loss.item()} / {loss_dict}", flush=True,
            )
            if skipped_nonfinite > 5:
                raise RuntimeError("more than 5 non-finite batches in one epoch")
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        for key, value in loss_dict.items():
            components[key] += float(value.detach())
        steps += 1
    averaged = {key: value / max(steps, 1) for key, value in components.items()}
    averaged["skipped_nonfinite"] = skipped_nonfinite
    return total / max(steps, 1), averaged


def collect_predictions(model, loader, device):
    model.eval()
    collected = []
    with torch.no_grad():
        for images, targets, metadata in loader:
            images_device = [image.to(device, non_blocking=True) for image in images]
            with autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(images_device)
            for output, target, meta in zip(outputs, targets, metadata):
                collected.append({
                    "meta": meta,
                    "scores": output["scores"].detach().float().cpu(),
                    "masks": output["masks"].detach().float().cpu()[:, 0],
                    "num_targets": int(target["labels"].numel()),
                })
            del outputs, images_device
    return collected


def region_polygon(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    coords = root.find(f".//{{{PAGE_NS}}}TextRegion/{{{PAGE_NS}}}Coords")
    if coords is None:
        return None
    return np.asarray(
        [[int(value) for value in pair.split(",")] for pair in coords.attrib["points"].split()],
        dtype=np.int32,
    )


def chaikin_closed_polygon(points: np.ndarray, passes: int) -> np.ndarray:
    """Round polygon corners while retaining a closed PAGE-compatible point list."""
    smoothed = points.astype(np.float32)
    for _ in range(passes):
        following = np.roll(smoothed, -1, axis=0)
        first = 0.75 * smoothed + 0.25 * following
        second = 0.25 * smoothed + 0.75 * following
        smoothed = np.stack((first, second), axis=1).reshape(-1, 2)
    rounded = np.rint(smoothed).astype(np.int32)
    if len(rounded) > 1:
        rounded = rounded[np.r_[True, np.any(rounded[1:] != rounded[:-1], axis=1)]]
    return rounded


def write_prediction_xml(
    item,
    output_path: Path,
    gt_xml_dir: Path,
    score_threshold: float,
    mask_threshold: float,
    contour_smooth_passes: int = 0,
    contour_mode: str = "approx",
):
    meta = item["meta"]
    stem = Path(meta["path"]).stem
    region_points = region_polygon(gt_xml_dir / f"{stem}.xml")
    orig_w, orig_h = int(meta["orig_w"]), int(meta["orig_h"])
    new_w, new_h = int(meta["new_w"]), int(meta["new_h"])

    ET.register_namespace("", PAGE_NS)
    ET.register_namespace("xsi", XSI_NS)
    root = ET.Element(f"{{{PAGE_NS}}}PcGts", {
        f"{{{XSI_NS}}}schemaLocation": (
            f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"
        )
    })
    metadata = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Creator").text = "MaskRCNN DIVA pilot"
    now = datetime.now(timezone.utc).isoformat()
    ET.SubElement(metadata, f"{{{PAGE_NS}}}Created").text = now
    ET.SubElement(metadata, f"{{{PAGE_NS}}}LastChange").text = now
    page = ET.SubElement(root, f"{{{PAGE_NS}}}Page", {
        "imageFilename": Path(meta["path"]).name,
        "imageWidth": str(orig_w),
        "imageHeight": str(orig_h),
    })
    region = ET.SubElement(page, f"{{{PAGE_NS}}}TextRegion", {"id": "region_textline"})
    if region_points is None:
        region_points = np.asarray([[0, 0], [orig_w - 1, 0], [orig_w - 1, orig_h - 1], [0, orig_h - 1]])
    ET.SubElement(region, f"{{{PAGE_NS}}}Coords", {
        "points": " ".join(f"{x},{y}" for x, y in region_points)
    })
    region_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    cv2.fillPoly(region_mask, [region_points.astype(np.int32)], 1)

    kept = 0
    for score, mask in zip(item["scores"], item["masks"]):
        if float(score) < score_threshold:
            continue
        content = mask[:new_h, :new_w].numpy()
        restored = cv2.resize(content, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = ((restored >= mask_threshold).astype(np.uint8) * region_mask)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 100:
            continue
        if contour_mode == "two_stage":
            # Match 02_2stage/dataset.py::mask_to_polygon exactly: the contour
            # returned by CHAIN_APPROX_SIMPLE is the PAGE polygon.
            polygon = contour.reshape(-1, 2)
        else:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if contour_smooth_passes:
                polygon = chaikin_closed_polygon(polygon, contour_smooth_passes)
                polygon[:, 0] = np.clip(polygon[:, 0], 0, orig_w - 1)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, orig_h - 1)
        if len(polygon) < 3:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        line = ET.SubElement(region, f"{{{PAGE_NS}}}TextLine", {"id": f"line_{kept}"})
        ET.SubElement(line, f"{{{PAGE_NS}}}Coords", {
            "points": " ".join(f"{int(px)},{int(py)}" for px, py in polygon)
        })
        ET.SubElement(line, f"{{{PAGE_NS}}}Baseline", {
            "points": f"{x},{y + height - 1} {x + width - 1},{y + height - 1}"
        })
        kept += 1
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return kept


def run_diva(predictions, split: str, evaluation_dir: Path, score_threshold: float,
             mask_threshold: float, contour_smooth_passes: int = 0,
             contour_mode: str = "approx", subset: str = "CB55"):
    subset_root = REPO / f"00_data/DIVA-HisDB/{subset}"
    split_name = {"val": "validation", "test": "public-test"}[split]
    gt_xml = subset_root / f"PAGE-gt-{subset}-TASK-2/TASK-2/{split_name}"
    gt_pixel = subset_root / f"pixel-level-gt-{subset}/pixel-level-gt/{split_name}"
    images = subset_root / f"img-{subset}/img/{split_name}"
    if contour_mode == "two_stage":
        suffix = "_two_stage_contour"
    else:
        suffix = f"_smooth{contour_smooth_passes}" if contour_smooth_passes else ""
    pred_dir = evaluation_dir / (
        f"diva_{split}_s{score_threshold:g}_m{mask_threshold:g}{suffix}"
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    stale = pred_dir / "results.csv"
    if stale.exists():
        stale.unlink()
    counts = []
    for item in predictions:
        stem = Path(item["meta"]["path"]).stem
        counts.append(write_prediction_xml(
            item, pred_dir / f"{stem}.xml", gt_xml, score_threshold, mask_threshold,
            contour_smooth_passes=contour_smooth_passes,
            contour_mode=contour_mode,
        ))

    jar = Path.home() / "Thesis/DIVA_Line_Segmentation_Evaluator/out/artifacts/LineSegmentationEvaluator.jar"
    java_cp = f"/usr/share/openjfx/lib/*:{jar}"
    for item in predictions:
        stem = Path(item["meta"]["path"]).stem
        process = subprocess.run([
            "java", "-cp", java_cp, "ch.unifr.LineSegmentationEvaluatorTool",
            "-igt", str(gt_pixel / f"{stem}.png"),
            "-xgt", str(gt_xml / f"{stem}.xml"),
            "-xp", str(pred_dir / f"{stem}.xml"),
            "-overlap", str(images / f"{stem}.jpg"),
            "-csv",
        ], cwd=pred_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode:
            raise RuntimeError(f"DIVA evaluator failed for {stem}: {process.stderr[-500:]}")
    results_path = pred_dir / "results.csv"
    if not results_path.exists():
        raise RuntimeError(f"DIVA evaluator did not produce {results_path}")
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The Java tool can append duplicate entries; one page, last entry wins.
    rows = list({row["filename"]: row for row in rows}.values())
    keys = ["PixelIU", "LinesIU", "LinesRecall", "LinesPrecision", "LinesFMeasure"]
    result = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    result["mean_predictions"] = float(np.mean(counts))
    result["score_threshold"] = score_threshold
    result["mask_threshold"] = mask_threshold
    result["contour_smooth_passes"] = contour_smooth_passes
    result["contour_mode"] = contour_mode
    return result


def pick_score_threshold(predictions):
    """Choose confidence on validation without repeatedly running Java.

    The criterion is the per-page absolute line-count error against the validation
    annotations. It is independent of test and is the same calibration principle
    used by the DIVA two-stage detector export.
    """
    candidates = (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    ranked = []
    for threshold in candidates:
        errors = [
            abs(int((item["scores"] >= threshold).sum()) - item["num_targets"])
            for item in predictions
        ]
        ranked.append((float(np.mean(errors)), float(threshold)))
    error, threshold = min(ranked)
    print(f"validation count calibration: score={threshold:g}, mean |pred-gt|={error:.2f}",
          flush=True)
    return threshold


def metric_value(metrics: dict, key: str, default: float = -1.0):
    # torchmetrics prefixes keys when bbox+segm are evaluated jointly.
    return float(metrics.get(key, metrics.get(f"segm_{key}", default)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    # CB55 keeps the flat paths its earlier runs already use; CS18/CS863 follow the
    # same layout one level down, mirroring the RT-DETR detector tree.
    parser.add_argument("--subset", choices=("CB55", "CS18", "CS863"), default="CB55")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--mask-roi-size", type=int, choices=(14, 28, 56), default=14,
                        help="ROI mask feature height (and width unless overridden)")
    parser.add_argument("--mask-roi-width", type=int, choices=(0, 28, 56, 112, 224), default=0,
                        help="optional wider ROI mask feature width for text lines")
    parser.add_argument("--roi-batch-size", type=int, choices=(64, 128, 256, 512), default=512,
                        help="sampled proposals per page during ROI-head training")
    parser.add_argument("--checkpoint-mask-head", action="store_true",
                        help="recompute mask-head activations during backward to save VRAM")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-shot", type=int, default=0,
                        help="number of labeled training pages; 0 uses all 20")
    parser.add_argument(
        "--k-shot-method",
        choices=("grayscale_variance", "random", "pca_max_distance", "pca_centroid",
                 "ica_max_distance", "ica_centroid"),
        default="grayscale_variance",
    )
    parser.add_argument("--k-shot-precomputed", type=Path, default=None)
    parser.add_argument("--mask-loss", choices=("bce", "tversky", "supervoxel"),
                        default="bce")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="probability cutoff used only for PAGE polygon export")
    parser.add_argument("--contour-smooth-passes", type=int, choices=range(0, 5), default=0,
                        help="Chaikin passes applied only to exported PAGE polygons")
    parser.add_argument("--contour-mode", choices=("approx", "two_stage"), default="approx",
                        help="PAGE contour conversion; two_stage skips approxPolyDP")
    parser.add_argument("--smoke", action="store_true", help="one train step and one-page evaluation")
    parser.add_argument("--evaluate-only", action="store_true", help="load best.pt and run final metrics")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="full Mask R-CNN state to start from (default for *_catmus arms: "
                             "the CATMuS pretrain best.pt)")
    parser.add_argument("--zero-shot", action="store_true",
                        help="skip training: score the init checkpoint as is")
    args = parser.parse_args()
    init_checkpoint = resolve_init_checkpoint(args.arm, args.init_checkpoint)
    if args.zero_shot and init_checkpoint is None:
        parser.error("--zero-shot needs an init checkpoint")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_mask_loss(args.mask_loss, device)
    subset = args.subset
    coco_dir = REPO / f"00_data/DIVA-HisDB/coco_task2_{subset}"
    image_root = REPO / f"00_data/DIVA-HisDB/yolo_dataset_{subset}/images"
    if args.k_shot < 0 or args.k_shot > 20:
        parser.error("--k-shot must be between 0 and 20")
    selected_pages: list[str] = []
    if args.k_shot:
        selection_image_dir = REPO / f"00_data/DIVA-HisDB/{subset}/img-{subset}/img/training"
        selected_pages = select_labeled_pages(
            img_dir=str(selection_image_dir),
            k=args.k_shot,
            method=args.k_shot_method,
            precomputed_path=(str(args.k_shot_precomputed)
                              if args.k_shot_precomputed is not None else None),
            seed=args.seed,
        )
    if os.environ.get("KSHOT_FIXED_PAGES"):   # same pages, other training seed (seed-variance runs)
        selected_pages = os.environ["KSHOT_FIXED_PAGES"].split(",")
    train_data = DivaCocoLines(
        coco_dir, image_root, "train", args.image_size, augment=True,
        include_stems=set(selected_pages) if selected_pages else None,
    )
    val_data = DivaCocoLines(coco_dir, image_root, "val", args.image_size, augment=False)
    test_data = DivaCocoLines(coco_dir, image_root, "test", args.image_size, augment=False)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2,
                              pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1,
                            pin_memory=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, collate_fn=collate)

    run_name = f"maskrcnn_{args.arm}_{args.image_size}"
    mask_roi_width = args.mask_roi_width or args.mask_roi_size
    if (args.mask_roi_size, mask_roi_width) != (14, 14):
        run_name += (f"_roi{args.mask_roi_size}" if mask_roi_width == args.mask_roi_size
                     else f"_roi{args.mask_roi_size}x{mask_roi_width}")
    if args.roi_batch_size != 512:
        run_name += f"_roib{args.roi_batch_size}"
    if args.checkpoint_mask_head:
        run_name += "_mhckpt"
    if args.mask_loss != "bce":
        run_name += f"_loss_{args.mask_loss}"
    if args.k_shot:
        run_name += f"_kshot_{args.k_shot_method}_k{args.k_shot}"
    if args.zero_shot:
        run_name += "_zeroshot"
    if args.smoke:
        run_name += "_smoke"
    if args.seed != 42:
        run_name += f"_s{args.seed}"
    run_dir = REPO / "80_models/instance_segmentation/mask_rcnn/diva-hisdb" / subset / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = (
        REPO / "99_evaluation/instance_segmentation/mask_rcnn/diva-hisdb" / subset / run_name
    )

    print(f"arm={args.arm} mask_loss={args.mask_loss} size={args.image_size} "
          f"mask_roi={args.mask_roi_size}x{mask_roi_width} "
          f"epochs={args.epochs} device={device}", flush=True)
    print(f"pages train/val/test={len(train_data)}/{len(val_data)}/{len(test_data)}", flush=True)
    if selected_pages:
        print(f"k-shot method={args.k_shot_method} pages={selected_pages}", flush=True)
    model = build_model(
        args.arm, args.image_size, args.mask_roi_size, args.mask_roi_width,
        args.roi_batch_size, args.checkpoint_mask_head,
    )
    if init_checkpoint is not None:
        init_epoch = load_init_checkpoint(model, init_checkpoint)
        if args.zero_shot:
            # The evaluation block below reloads run_dir/best.pt; make the init
            # weights that checkpoint so zero-shot goes through the same path.
            torch.save({"model": model.state_dict(), "args": serializable_args(args), "epoch": init_epoch},
                       run_dir / "best.pt")
            args.evaluate_only = True
    model = model.to(device)
    if args.arm == "stock":
        channels, reductions = "stock FPN", (4, 8, 16, 32, 64)
    else:
        channels = model.backbone.input_channels
        reductions = model.backbone.reductions
    total_params = sum(parameter.numel() for parameter in model.parameters())
    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_params}
    other_params = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    print(f"features channels={channels} reductions={reductions}", flush=True)
    print(f"params total={total_params / 1e6:.2f}M backbone+FPN="
          f"{sum(p.numel() for p in backbone_params) / 1e6:.2f}M", flush=True)
    started = time.time()
    epochs = 1 if args.smoke else args.epochs
    if args.evaluate_only:
        if not (run_dir / "best.pt").exists():
            raise FileNotFoundError(f"evaluation-only checkpoint missing: {run_dir / 'best.pt'}")
        history_path = run_dir / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        best_epoch = int(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["epoch"])
    else:
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params, "lr": args.lr},
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler("cuda", enabled=device.type == "cuda")
        history = []
        best_map = -1.0
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            loss, losses = train_one_epoch(
                model, train_loader, optimizer, scaler, device, max_steps=1 if args.smoke else 0
            )
            scheduler.step()
            peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
            row = {"epoch": epoch, "loss": loss, **losses, "peak_gpu_gb": peak_gb}
            should_eval = args.smoke or epoch % args.eval_every == 0 or epoch == epochs
            if should_eval:
                val = evaluate_coco(model, val_loader, device, max_pages=1 if args.smoke else 0)
                row.update({f"val_{key}": value for key, value in val.items()})
                segm_map = metric_value(val, "map")
                print(f"epoch={epoch:03d} loss={loss:.4f} val_segm_mAP={segm_map:.4f} "
                      f"peak={peak_gb:.2f}GB", flush=True)
                if segm_map > best_map:
                    best_map = segm_map
                    best_epoch = epoch
                    torch.save({"model": model.state_dict(), "args": serializable_args(args), "epoch": epoch},
                               run_dir / "best.pt")
            else:
                print(f"epoch={epoch:03d} loss={loss:.4f} peak={peak_gb:.2f}GB", flush=True)
            history.append(row)
            (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    val_metrics = evaluate_coco(model, val_loader, device, max_pages=1 if args.smoke else 0)
    test_metrics = evaluate_coco(model, test_loader, device, max_pages=1 if args.smoke else 0)
    diva_val = {}
    diva_test = {}
    if not args.smoke:
        val_predictions = collect_predictions(model, val_loader, device)
        score_threshold = pick_score_threshold(val_predictions)
        diva_val = run_diva(
            val_predictions, "val", evaluation_dir, score_threshold,
            mask_threshold=args.mask_threshold,
            contour_smooth_passes=args.contour_smooth_passes,
            contour_mode=args.contour_mode, subset=subset,
        )
        del val_predictions
        test_predictions = collect_predictions(model, test_loader, device)
        diva_test = run_diva(
            test_predictions, "test", evaluation_dir,
            score_threshold=diva_val["score_threshold"],
            mask_threshold=args.mask_threshold,
            contour_smooth_passes=args.contour_smooth_passes,
            contour_mode=args.contour_mode, subset=subset,
        )
        del test_predictions

    summary = {
        "model": "Mask R-CNN",
        "arm": args.arm,
        "mask_loss": args.mask_loss,
        "mask_threshold": args.mask_threshold,
        "dataset": f"DIVA-HisDB/{subset}/TASK-2",
        "k_shot": args.k_shot or len(train_data),
        "k_shot_method": args.k_shot_method if args.k_shot else "all",
        "selected_pages": selected_pages,
        "contour_smooth_passes": args.contour_smooth_passes,
        "contour_mode": args.contour_mode,
        "image_size": args.image_size,
        "mask_roi_size": args.mask_roi_size,
        "mask_roi_width": mask_roi_width,
        "mask_logit_size": [args.mask_roi_size * 2, mask_roi_width * 2],
        "roi_batch_size": args.roi_batch_size,
        "checkpoint_mask_head": args.checkpoint_mask_head,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "params_total": total_params,
        "params_backbone_fpn": sum(parameter.numel() for parameter in backbone_params),
        "runtime_seconds": time.time() - started,
        "validation": val_metrics,
        "test": test_metrics,
        "diva_validation": diva_val,
        "diva_test": diva_test,
    }
    if args.contour_mode == "two_stage":
        summary_name = "metrics_summary_two_stage_contour.json"
    elif args.contour_smooth_passes:
        summary_name = f"metrics_summary_smooth{args.contour_smooth_passes}.json"
    else:
        summary_name = "metrics_summary.json"
    (run_dir / summary_name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
