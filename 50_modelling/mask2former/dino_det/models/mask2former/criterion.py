import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(inputs, targets):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, no_object_weight):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_indices = []
        src_indices = []
        for batch_index, (src, _) in enumerate(indices):
            batch_indices.append(torch.full_like(src, batch_index))
            src_indices.append(src)
        if not batch_indices:
            return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)
        return torch.cat(batch_indices), torch.cat(src_indices)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        if src_idx.numel() > 0:
            target_classes_o = torch.cat(
                [target["labels"][tgt_idx] for target, (_, tgt_idx) in zip(targets, indices)],
                dim=0,
            )
            target_classes[batch_idx, src_idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices):
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        if src_idx.numel() == 0:
            zero = outputs["pred_masks"].sum() * 0.0
            return {"loss_mask": zero, "loss_dice": zero}

        src_masks = outputs["pred_masks"][batch_idx, src_idx]
        target_masks = []
        for target, (_, tgt_idx) in zip(targets, indices):
            if tgt_idx.numel() == 0:
                continue
            resized = F.interpolate(
                target["masks"][tgt_idx].float().unsqueeze(1),
                size=src_masks.shape[-2:],
                mode="nearest",
            ).squeeze(1)
            target_masks.append(resized)

        target_masks = torch.cat(target_masks, dim=0)
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks)
        loss_dice = dice_loss(src_masks, target_masks).mean()
        return {"loss_mask": loss_mask, "loss_dice": loss_dice}

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_masks(outputs, targets, indices))
        for name, weight in self.weight_dict.items():
            losses[name] = losses[name] * weight
        return losses