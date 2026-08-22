#!/usr/bin/env bash
# ============================================================================
# RQ1 / 01 -- U-DIADS-TL, DINO-SSL arm of the pretrain axis (bracket M only).
#
# Completes the third column for the two architectures that already have a
# random and an ImageNet arm, so each becomes a clean 3-way comparison:
#
#   architecture              random          ImageNet             DINO SSL
#   convnext_tiny  27.8M      _szrand (done)  _sz (done)           this script
#   vit_small/16   21.6M      _szrand (done)  _sz augreg (done)    this script
#
# SSL exists ONLY at bracket M. The smallest DINOv3 ConvNeXt is tiny (27.8M),
# the smallest DINOv3 ViT is small (21.6M), the smallest DINOv2 is ViT-S/14
# (21.7M) -- nothing was released below ~21M, and there are no SSL weights for
# hierarchical ViTs (PvtV2) at any size. XS and S therefore stay empty.
#
# No random counterpart is needed: random init does not depend on the pretraining
# source, so the existing _szrand runs already serve as the random baseline for
# both architectures.
#
# ---------------------------------------------------------------------------
# ONLY 3 OF THE 6 CELLS ACTUALLY TRAIN.
#
# unet_tu-convnext_tiny.dinov3_lvd1689m_udiads_<subset> already exists for all
# three subsets, and its recipe was checked against a _sz checkpoint field by
# field: training/data/supervoxel sections identical (100 epochs, k_shot=3,
# crop 448, batch 4, lr 1e-3 / 1e-4, wd 1e-4). The only diffs were
# lambda_boundary None-vs-0.0 -- train.py falls back to supervoxel.lambda_boundary
# (0.0) when the training key is absent, so both are 0 -- and freeze_backbone
# None-vs-false, which smp_unet.py reads only inside the hier-ViT branch and
# ignores for CNN encoders. Those runs are marked .done rather than retrained.
#
# RUN_SUFFIX is empty so the names match those existing folders.
# ---------------------------------------------------------------------------
#
# DINOv3 ViT NOTE: vit_small_patch16_dinov3 used to fail outright -- hier_encoder
# loaded it from Meta's gated torch.hub, which raised "silently fell back to a
# random-init backbone". It now resolves to the dinov3_vits16_timm spec, which
# pulls the same lvd1689m checkpoint from timm ungated (weights verified
# identical, 21.59M). The preflight below fails loudly if the stub ever returns.
#
# Checkpoints -> .../simple_segmentation/unet_<encoder>_udiads_<subset>/best.pth
# Runtime: ~3 ViT runs, budget ~1 h.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

CFG=configs/unet_sizeaxis_u_diads.yaml
OUT=80_models/01_simple_segmentation/u-diads-tl/segmentation/simple_segmentation
SUBSETS=(Latin14396 Latin2 Syr341)
EPOCHS=100
RUN_SUFFIX=""

ENCODERS=(
  "tu-convnext_tiny.dinov3_lvd1689m"   # M  CNN,      DINOv3   (pre-existing)
  "vit_small_patch16_dinov3"           # M  ViT flat, DINOv3   (trains here)
)

# Mark the pre-existing, recipe-verified ConvNeXt runs complete so the shared
# skip guard leaves them alone. Only done when best.pth is actually there.
for ms in "${SUBSETS[@]}"; do
  d="../../$OUT/unet_tu-convnext_tiny.dinov3_lvd1689m_udiads_${ms}"
  if [[ -f "$d/best.pth" && ! -f "$d/.done" ]]; then
    touch "$d/.done"
    echo "[reuse] marked existing run complete: $(basename "$d")"
  fi
done

source ./_size_axis_common.sh
