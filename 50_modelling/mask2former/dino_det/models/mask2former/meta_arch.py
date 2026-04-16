import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import BitMasks, Boxes, ImageList, Instances

from .criterion import SetCriterion
from .head import Mask2FormerHead
from .matcher import HungarianMatcher


@META_ARCH_REGISTRY.register()
class DinoMask2Former(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone,
        sem_seg_head,
        criterion,
        pixel_mean,
        pixel_std,
        mask_threshold,
        score_threshold,
        topk_per_image,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.mask_threshold = mask_threshold
        self.score_threshold = score_threshold
        self.topk_per_image = topk_per_image
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), persistent=False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = Mask2FormerHead(cfg, backbone.output_shape())
        matcher = HungarianMatcher(
            class_weight=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
            mask_weight=cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
            dice_weight=cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
        )
        criterion = SetCriterion(
            num_classes=cfg.MODEL.MASK_FORMER.NUM_CLASSES,
            matcher=matcher,
            weight_dict={
                "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
                "loss_mask": cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
                "loss_dice": cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
            },
            no_object_weight=cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT,
        )
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "mask_threshold": cfg.MODEL.MASK_FORMER.MASK_THRESHOLD,
            "score_threshold": cfg.MODEL.MASK_FORMER.SCORE_THRESHOLD,
            "topk_per_image": cfg.MODEL.MASK_FORMER.TOPK_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        images = [(sample["image"].to(self.device) - self.pixel_mean) / self.pixel_std for sample in batched_inputs]
        return ImageList.from_tensors(images, self.backbone.size_divisibility)

    def prepare_targets(self, batched_inputs):
        targets = []
        for sample in batched_inputs:
            instances = sample["instances"].to(self.device)
            gt_masks = instances.gt_masks.tensor if hasattr(instances.gt_masks, "tensor") else instances.gt_masks
            targets.append({
                "labels": instances.gt_classes,
                "masks": gt_masks.float(),
            })
        return targets

    @staticmethod
    def _mask_iou(mask_a, mask_b):
        intersection = (mask_a & mask_b).float().sum()
        union = (mask_a | mask_b).float().sum().clamp(min=1.0)
        return intersection / union

    def instance_inference(self, pred_logits, pred_masks, image_size):
        class_probabilities = F.softmax(pred_logits, dim=-1)[:, :-1]
        scores, labels = class_probabilities.max(dim=-1)
        mask_probabilities = pred_masks.sigmoid()
        binary_masks = mask_probabilities > self.mask_threshold

        kept = []
        ordered = torch.argsort(scores, descending=True)
        for index in ordered.tolist():
            if scores[index] < self.score_threshold:
                continue
            if binary_masks[index].sum() == 0:
                continue
            overlaps = False
            for kept_index in kept:
                if self._mask_iou(binary_masks[index], binary_masks[kept_index]) > 0.85:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(index)
            if len(kept) >= self.topk_per_image:
                break

        result = Instances(image_size)
        if not kept:
            result.pred_masks = torch.zeros((0, image_size[0], image_size[1]), dtype=torch.bool, device=pred_masks.device)
            result.scores = torch.zeros((0,), device=pred_masks.device)
            result.pred_classes = torch.zeros((0,), dtype=torch.int64, device=pred_masks.device)
            result.pred_boxes = Boxes(torch.zeros((0, 4), device=pred_masks.device))
            return result

        kept = torch.as_tensor(kept, dtype=torch.int64, device=pred_masks.device)
        selected_masks = binary_masks[kept]
        selected_scores = scores[kept]
        selected_labels = labels[kept]
        mask_quality = (
            (mask_probabilities[kept] * selected_masks.float()).flatten(1).sum(1)
            / selected_masks.float().flatten(1).sum(1).clamp(min=1.0)
        )
        selected_scores = selected_scores * mask_quality

        result.pred_masks = selected_masks
        result.scores = selected_scores
        result.pred_classes = selected_labels
        result.pred_boxes = BitMasks(selected_masks).get_bounding_boxes()
        return result

    def postprocess_instances(self, instances, output_height, output_width):
        processed = Instances((output_height, output_width))

        if not instances.has("pred_masks"):
            processed.scores = instances.scores
            processed.pred_classes = instances.pred_classes
            processed.pred_boxes = instances.pred_boxes
            return processed

        mask_tensor = instances.pred_masks.float().unsqueeze(1)
        if mask_tensor.shape[-2:] != (output_height, output_width):
            mask_tensor = F.interpolate(mask_tensor, size=(output_height, output_width), mode="nearest")

        processed_masks = mask_tensor[:, 0] > 0.5
        processed.pred_masks = processed_masks
        processed.scores = instances.scores
        processed.pred_classes = instances.pred_classes
        processed.pred_boxes = BitMasks(processed_masks).get_bounding_boxes()
        return processed

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            targets = self.prepare_targets(batched_inputs)
            return self.criterion(outputs, targets)

        results = []
        for sample, image_size, pred_logits, pred_masks in zip(
            batched_inputs,
            images.image_sizes,
            outputs["pred_logits"],
            outputs["pred_masks"],
        ):
            result = self.instance_inference(pred_logits, pred_masks, image_size)
            height = sample.get("height", image_size[0])
            width = sample.get("width", image_size[1])
            processed = self.postprocess_instances(result, height, width)
            results.append({"instances": processed})
        return results