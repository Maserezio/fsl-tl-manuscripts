#!/usr/bin/env bash
# ============================================================================
# RQ1 / 01 -- size axis on U-DIADS-TL, ImageNet arm.
#
# Three encoder families across three size brackets (backbone params, measured):
#
#   bracket   CNN                       ViT (hierarchical)     ViT (flat, SFP)
#   XS        tu-convnext_femto  4.8M   tu-pvt_v2_b0   3.4M    vit_tiny   5.5M
#   S         tu-convnext_pico   8.5M   tu-pvt_v2_b1  13.5M    --
#   M         tu-convnext_tiny  27.8M   tu-pvt_v2_b2  24.9M    vit_small 21.7M
#
# 8 encoders x 3 subsets = 24 cells, all trained fresh under ONE recipe.
#
# Nothing is reused. The existing U-DIADS runs for tu-convnext_tiny and tu-pvt_v2_b2
# are HPO-tuned: 60 epochs with per-encoder lr / weight_decay / lambda_boundary
# (e.g. pvt_v2_b2 got lr=0.00214, wd=0.00047, lambda_bnd=0.514). Reusing them would
# put individually-tuned models next to fixed-recipe ones in the same table. Runs
# here carry a _sz suffix, so those checkpoints are left untouched.
#
# The flat ViTs are NOT foundation models -- augreg_in21k is plain supervised
# ImageNet-21k. They still route through hier_encoder's SFP neck, because a flat
# ViT gives three feature maps all at stride 16 and smp.Unet's decoder needs a
# real pyramid.
#
# FROZEN-BACKBONE CONFOUND, fixed here. Everywhere else in this project the flat
# ViT path runs freeze_backbone=True: the ~30M ViT stays frozen and only the ~8M
# SFP neck trains, while CNN and PvtV2 encoders train end to end. In a size matrix
# that is not a size comparison at all. configs/unet_sizeaxis_u_diads.yaml sets
# freeze_backbone: false so every column trains the same way. Consequence: these
# ViT rows are NOT comparable with the existing frozen-ViT rows in the DIVA table.
#
# Recipe = the project's uniform matrix recipe, the one behind the DIVA
# "100-epoch matrix" and the two tu-convnext_tiny.* U-DIADS runs: 100 epochs,
# k_shot=3, crop 448, batch 4, lr 1e-3, lr_backbone 1e-4, weight_decay 1e-4,
# lambda_boundary 0. NOT run_train_all.sh's 200 -- no U-DIADS run in the repo
# actually uses 200 with this recipe.
#
# Checkpoints -> 80_models/01_simple_segmentation/u-diads-tl/segmentation/
#                simple_segmentation/unet_<encoder>_udiads_<subset>/best.pth
#
# Metrics are NOT produced here: U-DIADS scoring goes through the Zottin metric and
# predict_u_diads.py hardcodes a single checkpoint path, so it needs generalising
# first. Train now, score after.
#
# Runtime: ~8-10 min per CNN/PvtV2 run at 100 epochs; ViT rows slower. Budget ~5 h.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

CFG=configs/unet_sizeaxis_u_diads.yaml
OUT=80_models/01_simple_segmentation/u-diads-tl/segmentation/simple_segmentation
SUBSETS=(Latin14396 Latin2 Syr341)
EPOCHS=100
RUN_SUFFIX="_sz"

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
