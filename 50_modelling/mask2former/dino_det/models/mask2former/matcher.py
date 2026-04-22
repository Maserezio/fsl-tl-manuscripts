import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment


def batch_dice_cost(inputs, targets):
    inputs = inputs.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * torch.matmul(inputs, targets.T)
    denominator = inputs.sum(-1, keepdim=True) + targets.sum(-1).unsqueeze(0)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def batch_sigmoid_ce_cost(inputs, targets):
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    softplus = F.softplus(inputs).mean(-1, keepdim=True)
    projection = torch.matmul(inputs, targets.T) / inputs.shape[1]
    return softplus - projection


class HungarianMatcher(nn.Module):
    def __init__(self, class_weight=1.0, mask_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight

    @torch.no_grad()
    def forward(self, outputs, targets):
        # Compute matching in float32 to avoid AMP-induced inf/nan costs.
        pred_logits = outputs["pred_logits"].float()
        pred_masks = outputs["pred_masks"].float()
        indices = []

        for batch_index, target in enumerate(targets):
            target_labels = target["labels"]
            if target_labels.numel() == 0:
                empty = torch.empty(0, dtype=torch.int64, device=pred_logits.device)
                indices.append((empty, empty))
                continue

            out_prob = pred_logits[batch_index].softmax(-1)[:, :-1]
            cost_class = -out_prob[:, target_labels]

            target_masks = target["masks"].float()
            resized_target_masks = F.interpolate(
                target_masks.unsqueeze(1),
                size=pred_masks.shape[-2:],
                mode="nearest",
            ).squeeze(1)

            cost_mask = batch_sigmoid_ce_cost(pred_masks[batch_index], resized_target_masks)
            cost_dice = batch_dice_cost(pred_masks[batch_index], resized_target_masks)

            cost_matrix = (
                self.class_weight * cost_class
                + self.mask_weight * cost_mask
                + self.dice_weight * cost_dice
            )
            cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=-1e6)
            src_indices, tgt_indices = linear_sum_assignment(cost_matrix.cpu())
            indices.append(
                (
                    torch.as_tensor(src_indices, dtype=torch.int64, device=pred_logits.device),
                    torch.as_tensor(tgt_indices, dtype=torch.int64, device=pred_logits.device),
                )
            )
        return indices