#!/usr/bin/env bash
# ============================================================================
# RQ1 / 01 -- size axis on U-DIADS-TL, RANDOM-INIT arm.
#
# Companion to run_train_udiads_size_axis.sh. Same 8 encoders, same 3 subsets,
# same recipe; only the config differs (unet_sizeaxis_random_u_diads.yaml).
# 24 runs, none exist yet.
#
# Covers two cells of the plan at once:
#   - the random column of the size axis (all 8 encoders)
#   - the Random column of the pretrain axis, bracket M
#     (tu-convnext_tiny 27.8M, tu-pvt_v2_b2 24.9M, vit_small 21.7M)
#
# Random init needs TWO keys, because the encoders take two different loaders:
#   encoder_weights: null -> smp/timm CNNs and hierarchical ViTs
#   pretrained: false     -> flat ViTs going through hier_encoder
#
# hier_encoder used to answer pretrained=False with FallbackViTBackbone, a stub
# with a different architecture, so a random flat-ViT arm would have trained the
# wrong model and reported it as ViT. load_backbone now builds the genuine timm
# architecture with random weights for timm_vit specs; the preflight asserts the
# stub is not in use, so this cannot regress silently.
#
# Run names carry a _rand tag. Without it they would collide with the ImageNet
# arm in the same out-dir and silently overwrite those checkpoints.
#
# ---------------------------------------------------------------------------
# LR_BACKBONE: read before trusting the numbers.
#
# The recipe puts lr=1e-3 on the decoder and 1e-4 on the encoder -- right for
# finetuning pretrained features, actively harmful when the encoder starts from
# noise. The two options measure different things:
#
#   0.0001 (default) -- identical recipe, so random-vs-ImageNet isolates the
#                       weights. Handicaps random: its encoder learns 10x slower
#                       than its own decoder, from scratch.
#   0.001            -- encoder at full LR, the usual from-scratch setting. Fair
#                       to random, but now two things differ between the arms.
#
# Default is the identical recipe: the plan's claim is about pretraining, and a
# confounded recipe cannot support it. If the random arm collapses, re-run with
# 0.001 before concluding anything -- that may be the LR, not the initialisation.
# ---------------------------------------------------------------------------
#
# Checkpoints -> .../simple_segmentation/unet_<encoder>_rand_udiads_<subset>/best.pth
# Runtime: budget ~5 h for all 24 at 100 epochs.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

CFG=configs/unet_sizeaxis_random_u_diads.yaml
OUT=80_models/01_simple_segmentation/u-diads-tl/segmentation/simple_segmentation
SUBSETS=(Latin14396 Latin2 Syr341)
EPOCHS=100
RUN_SUFFIX="_szrand"
LR=0.001
LR_BACKBONE=0.0001                 # see the note above before changing

ENCODERS=(
  "vit_tiny_patch16_224.augreg_in21k"   # XS  ViT flat (SFP)
  "tu-convnext_femto"                   # XS  CNN
  "tu-pvt_v2_b0"                        # XS  ViT hierarchical
  "tu-convnext_pico"                    # S   CNN
  "tu-pvt_v2_b1"                        # S   ViT hierarchical
  "tu-convnext_tiny"                    # M   CNN
  "vit_small_patch16_224.augreg_in21k"  # M   ViT flat (SFP)
  "tu-pvt_v2_b2"                        # M   ViT hierarchical
)

source ./_size_axis_common.sh
