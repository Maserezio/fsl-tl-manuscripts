"""
Supervoxel (SLIC superpixel) consistency loss.

Motivation
----------
In a few-shot setting, labeled data is scarce. Supervoxels partition the
image into small regions of visually similar, spatially adjacent pixels —
a compact prior from natural image structure. Forcing all pixels within
a supervoxel to agree on their prediction is a parameter-free regulariser
that leverages image appearance without requiring extra annotations.

Loss formulation
----------------
For each supervoxel S_k in image b:

    μ_k  = (1/|S_k|) Σ_{i ∈ S_k}  σ(logit_i)          (mean prediction)
    v_k  = (1/|S_k|) Σ_{i ∈ S_k}  (σ(logit_i) - μ_k)² (within-SP variance)

    L_sv = (1/B) Σ_b  mean_{k}(v_k)

Minimising L_sv pushes pixels in the same supervoxel toward a consensus
probability, effectively propagating confident predictions to uncertain
neighbours that share the same appearance.

This loss is applied to:
  - Labeled crops  (together with supervised BCE+Dice, as a regulariser)
  - Unlabeled crops (no GT needed — purely self-supervised signal)

Implementation
--------------
The inner loop is over batch elements (B is typically 2–4). Within each
element we use scatter_add_ on the flattened superpixel index to compute
per-supervoxel sums and counts in a single pass, then gather them back to
pixel positions to compute the variance efficiently without Python loops
over supervoxels.
"""

import torch
import torch.nn.functional as F


def supervoxel_variance_loss(
    logits: torch.Tensor,        # [B, 1, H, W]
    superpixels: torch.Tensor,   # [B, H, W] int64, contiguous IDs from 0
) -> torch.Tensor:
    """
    Intra-supervoxel prediction variance loss.

    Parameters
    ----------
    logits      : raw (pre-sigmoid) model output [B, 1, H, W]
    superpixels : SLIC segment map [B, H, W], IDs start at 0

    Returns
    -------
    Scalar tensor — mean intra-supervoxel variance over the batch.
    """
    probs = torch.sigmoid(logits[:, 0])   # [B, H, W]
    sp    = superpixels.to(probs.device)  # [B, H, W]

    B, H, W = probs.shape
    N = H * W

    p_flat  = probs.view(B, N)            # [B, N]
    sp_flat = sp.view(B, N)               # [B, N]

    total = probs.new_zeros(1)

    for b in range(B):
        p  = p_flat[b]    # [N]
        s  = sp_flat[b]   # [N]  values in [0, num_sp)

        num_sp = int(s.max().item()) + 1

        # Per-supervoxel pixel count
        count = p.new_zeros(num_sp)
        count.scatter_add_(0, s, torch.ones_like(p))

        # Per-supervoxel sum of probabilities
        sp_sum = p.new_zeros(num_sp)
        sp_sum.scatter_add_(0, s, p)

        # Per-supervoxel mean probability
        sp_mean = sp_sum / (count + 1e-6)   # [num_sp]

        # Per-pixel mean (gather from supervoxel means)
        p_mean = sp_mean[s]                  # [N]

        # Variance = mean squared deviation from supervoxel mean
        total = total + ((p - p_mean) ** 2).mean()

    return total / B


def supervoxel_boundary_affinity_loss(
    logits: torch.Tensor,        # [B, 1, H, W]
    superpixels: torch.Tensor,   # [B, H, W] int64
) -> torch.Tensor:
    """
    Complementary loss: penalise predictions that differ across supervoxel
    boundaries less than within a supervoxel.

    Concretely: for each pair of horizontally or vertically adjacent pixels
    that belong to *different* supervoxels, we push their predictions apart
    (high boundary contrast); for same-supervoxel neighbours, we push them
    together (low boundary contrast).

    This is a soft version of the SLIC boundary and is optional.
    It is NOT used by default in train.py but is available for ablation.

    Loss = mean_{cross-SP edges}  (p_i - p_j)²  would *maximise* diversity,
    which is the wrong direction.  Instead we define:

        L_ba = mean_{same-SP edges}   (p_i - p_j)²
             - mean_{cross-SP edges}  (p_i - p_j)²

    so that same-SP pairs are pulled together while cross-SP pairs may differ.
    In practice we clip at zero (hinge) to avoid penalising already-separated
    cross-SP predictions:

        L_ba = mean_{same-SP}(Δ²) - λ · max(0, τ - mean_{cross-SP}(Δ²))

    This function returns only the within-SP part (equivalent to
    supervoxel_variance_loss but computed via local pairwise differences,
    which gives a slightly different gradient flow).

    Parameters
    ----------
    logits      : [B, 1, H, W]
    superpixels : [B, H, W]

    Returns
    -------
    Scalar — mean within-supervoxel pairwise squared difference.
    """
    p  = torch.sigmoid(logits[:, 0])   # [B, H, W]
    sp = superpixels.to(p.device)

    # Horizontal neighbours: (b, h, w) vs (b, h, w+1)
    p_l  = p[:, :, :-1];   p_r  = p[:, :, 1:]
    sp_l = sp[:, :, :-1];  sp_r = sp[:, :, 1:]
    same_h = (sp_l == sp_r)

    # Vertical neighbours
    p_t  = p[:, :-1, :];   p_b  = p[:, 1:, :]
    sp_t = sp[:, :-1, :];  sp_b = sp[:, 1:, :]
    same_v = (sp_t == sp_b)

    diff_h = (p_l - p_r) ** 2
    diff_v = (p_t - p_b) ** 2

    loss = (diff_h[same_h].mean() + diff_v[same_v].mean()) / 2.0
    return loss
