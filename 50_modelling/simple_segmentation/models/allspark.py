"""
AllSpark module for semi-supervised semantic segmentation.

Reference:
  "AllSpark: Reborn Labeled Features from Unlabeled in Transformer for
   Semi-Supervised Semantic Segmentation"  CVPR 2024, arXiv 2403.01818
  Official code: https://github.com/xmed-lab/AllSpark

Overview
--------
AllSpark sits between the backbone and the decoder.  It receives a
*concatenated* batch [labeled | unlabeled] of feature maps and enriches the
labeled features by attending to the unlabeled features (or, after warmup, to
a persistent Semantic Memory of unlabeled prototypes).

Channel-wise attention
----------------------
Feature maps are flattened to [B, C, N] (N = H_f × W_f).
The *channels* act as the attention sequence; the *spatial positions* act as
the per-token feature vector.  One attention step:

    scores = (Q · K^T) / sqrt(N)          shape [B, C, C]
    out    = softmax(scores) · V           shape [B, C, N]

Each labeled channel (row of Q) attends to all unlabeled channels (rows of K)
whose spatial activation pattern is most similar to its own.

Semantic Memory (S-Mem)
-----------------------
A per-class FIFO circular buffer  kv_queue [num_class, C, N].
During training, unlabeled channels are assigned to classes via pseudo-label
similarity and enqueued.  After warmup, the labeled cross-attention keys/values
come from this buffer instead of the current mini-batch, giving access to a
much richer distribution of unlabeled features.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AllSparkModule(nn.Module):
    """
    Parameters
    ----------
    in_channels  : backbone output channels (C), same for labeled and unlabeled
    ec           : embedding_channels — hidden dimension inside the module
    num_heads    : attention heads  (ec must be divisible by num_heads;
                   head_dim = N // num_heads must be an integer too)
    num_class    : semantic classes (2 for text-line segmentation)
    patch_num    : spatial positions in the feature map = H_f × W_f
                   (must match the actual input at runtime)
    """

    def __init__(
        self,
        in_channels: int,
        ec: int,
        num_heads: int,
        num_class: int,
        patch_num: int,
    ):
        super().__init__()
        assert ec % num_heads == 0, "ec must be divisible by num_heads"
        assert patch_num % num_heads == 0, "patch_num must be divisible by num_heads"

        self.ec        = ec
        self.num_heads = num_heads
        self.num_class = num_class
        self.patch_num = patch_num
        # Scale for channel-wise attention: dot-products are over N-dim vectors
        self.scale = patch_num ** -0.5

        # --- channel dimension projections (N-agnostic) ---
        self.map_in = nn.Sequential(
            nn.Conv1d(in_channels, ec, 1, bias=False),
            nn.GELU(),
        )
        self.map_out = nn.Sequential(
            nn.Conv1d(ec, in_channels, 1, bias=False),
            nn.GELU(),
        )

        # instance-norm for attention stabilisation (ψ in the paper)
        self.attn_norm = nn.InstanceNorm1d(ec, affine=True)
        self.enc_norm  = nn.InstanceNorm1d(ec, affine=True)

        # Self-attention projections for unlabeled features
        self.q_sa  = nn.Conv1d(ec, ec, 1, bias=False)
        self.k_sa  = nn.Conv1d(ec, ec, 1, bias=False)
        self.v_sa  = nn.Conv1d(ec, ec, 1, bias=False)
        self.out_sa = nn.Conv1d(ec, ec, 1, bias=False)

        # Cross-attention projections for labeled → unlabeled/S-Mem
        self.q_ca  = nn.Conv1d(ec, ec, 1, bias=False)
        self.k_ca  = nn.Conv1d(ec, ec, 1, bias=False)
        self.v_ca  = nn.Conv1d(ec, ec, 1, bias=False)
        self.out_ca = nn.Conv1d(ec, ec, 1, bias=False)

        # --- Semantic Memory ---
        # kv_queue[c] : [ec, patch_num]  — circular buffer of unlabeled channel
        #               vectors grouped by class c.
        # queue_ptr[c]: next write position inside the ec dimension.
        self.register_buffer("kv_queue",
                             torch.randn(num_class, ec, patch_num) * 0.01)
        self.register_buffer("queue_ptr",
                             torch.zeros(num_class, dtype=torch.long))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _channel_attn(
        self,
        Q: torch.Tensor,   # [B, q_len, N]
        K: torch.Tensor,   # [B, k_len, N]
        V: torch.Tensor,   # [B, k_len, N]
    ) -> torch.Tensor:
        """
        Channel-wise attention.

        score[b, i, j] = Q[b,i,:] · K[b,j,:] / sqrt(N)
        → attention weights [B, q_len, k_len]
        → output [B, q_len, N]
        """
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # [B, q, k]
        # ψ: instance-normalise rows of the attention matrix before softmax
        scores = F.instance_norm(scores)
        attn   = torch.softmax(scores, dim=-1)                   # [B, q, k]
        return torch.bmm(attn, V)                                # [B, q, N]

    def _assign_channels_to_class(
        self,
        emb_u: torch.Tensor,        # [B, ec, N]  (after map_in)
        pseudo_prob: torch.Tensor,  # [B, num_class, H_f, W_f]
    ) -> torch.Tensor:
        """
        For each channel in emb_u determine which semantic class it belongs to,
        based on similarity with the pseudo-probability maps.

        Returns channel_cls [B, ec]  (integer class index per channel).
        """
        B, ec, N = emb_u.shape
        H_f = W_f = int(N ** 0.5)

        prob = F.interpolate(
            pseudo_prob.float(), size=(H_f, W_f),
            mode="bilinear", align_corners=False,
        ).reshape(B, self.num_class, N)           # [B, K, N]

        # L2-normalise along spatial dim
        prob_n  = F.normalize(prob,  p=2, dim=-1)  # [B, K, N]
        emb_n   = F.normalize(emb_u, p=2, dim=-1)  # [B, ec, N]

        # [B, K, N] × [B, N, ec] → [B, K, ec]
        sim = torch.bmm(prob_n, emb_n.transpose(1, 2))
        return sim.argmax(dim=1)   # [B, ec]

    @torch.no_grad()
    def _update_queue(
        self,
        emb_u: torch.Tensor,        # [B, ec, N]  (detached)
        channel_cls: torch.Tensor,  # [B, ec]
    ) -> None:
        """FIFO-enqueue unlabeled channel vectors into the Semantic Memory."""
        B, ec, N = emb_u.shape
        for c in range(self.num_class):
            parts = []
            for b in range(B):
                mask = channel_cls[b] == c       # [ec] bool
                if mask.any():
                    parts.append(emb_u[b, mask])  # [n_c, N]
            if not parts:
                continue
            new_feats = torch.cat(parts, dim=0)  # [total_c, N]

            # Clamp to queue capacity: if more than ec vectors arrived in one
            # step they would wrap multiple times; keep only the latest ec.
            if new_feats.shape[0] >= ec:
                self.kv_queue[c] = new_feats[-ec:].detach()
                self.queue_ptr[c] = 0
                continue

            n   = new_feats.shape[0]
            ptr = int(self.queue_ptr[c])
            end = ptr + n
            if end <= ec:
                self.kv_queue[c, ptr:end] = new_feats.detach()
            else:
                first = ec - ptr           # slots before wrap-around
                self.kv_queue[c, ptr:]     = new_feats[:first].detach()
                self.kv_queue[c, :n-first] = new_feats[first:].detach()
            self.queue_ptr[c] = int(end % ec)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,              # [2B, C, H_f, W_f]
        pseudo_prob: Optional[torch.Tensor], # [B, num_class, H_f, W_f]  or None
        using_smem: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features    : Concatenated [labeled (first half) | unlabeled (second half)]
                      feature maps from the backbone.
        pseudo_prob : Soft pseudo-labels for the *unlabeled* half.
                      Required only when using_smem=False and self.training=True.
                      Pass None during inference.
        using_smem  : After warmup, set True to draw K/V from the Semantic Memory
                      instead of the current batch.

        Returns
        -------
        [2B, C, H_f, W_f] — labeled half enriched by AllSpark;
                             unlabeled half processed by self-attention.
        """
        two_B, C, H_f, W_f = features.shape
        B = two_B // 2
        N = H_f * W_f

        feat_l = features[:B]
        feat_u = features[B:]

        # ── Flatten spatial ──────────────────────────────────────────
        emb_l = feat_l.reshape(B, C, N)   # [B, C, N]
        emb_u = feat_u.reshape(B, C, N)

        # ── Project to embedding dim ─────────────────────────────────
        emb_l = self.map_in(emb_l)        # [B, ec, N]
        emb_u = self.map_in(emb_u)
        res_l = emb_l
        res_u = emb_u

        norm_l = self.attn_norm(emb_l)
        norm_u = self.attn_norm(emb_u)

        # ── Unlabeled self-attention ─────────────────────────────────
        Q_u = self.q_sa(norm_u)
        K_u = self.k_sa(norm_u)
        V_u = self.v_sa(norm_u)
        sa_u = self.out_sa(self._channel_attn(Q_u, K_u, V_u)) + res_u

        # ── Semantic Memory update (training only) ───────────────────
        if self.training and pseudo_prob is not None:
            channel_cls = self._assign_channels_to_class(emb_u.detach(), pseudo_prob)
            self._update_queue(emb_u.detach(), channel_cls)

        # ── Labeled cross-attention ──────────────────────────────────
        Q_l = self.q_ca(norm_l)           # [B, ec, N]

        if using_smem:
            # K/V: all stored class prototypes  [num_class, ec, N]
            # Flatten to [1, num_class*ec, N] → expand to [B, num_class*ec, N]
            kv = self.kv_queue.reshape(1, self.num_class * self.ec, N)
            kv = kv.expand(B, -1, -1)            # [B, num_class*ec, N]
            # Score: [B, ec, N] @ [B, N, num_class*ec] → [B, ec, num_class*ec]
            scores = torch.bmm(Q_l, kv.transpose(1, 2)) * self.scale
            scores = F.instance_norm(scores)
            attn   = torch.softmax(scores, dim=-1)
            ca_l   = torch.bmm(attn, kv)         # [B, ec, N]
            ca_l   = self.out_ca(ca_l) + res_l
        else:
            K_ca = self.k_ca(norm_u)
            V_ca = self.v_ca(norm_u)
            ca_l = self.out_ca(self._channel_attn(Q_l, K_ca, V_ca)) + res_l

        # ── Project back to backbone channels ───────────────────────
        out_l = self.map_out(self.enc_norm(ca_l)).reshape(B, C, H_f, W_f)
        out_u = self.map_out(self.enc_norm(sa_u)).reshape(B, C, H_f, W_f)

        return torch.cat([out_l, out_u], dim=0)   # [2B, C, H_f, W_f]
