#!/usr/bin/env bash
# ============================================================================
# Score the U-DIADS size-axis matrix (48 runs: 8 encoders x 3 subsets x 2 arms).
#
# Two stages, both already implemented -- this only drives them:
#   1. preprocessing/cache_predictions.py -- sliding-window inference, one .npy
#      probability map per page. The expensive part; cached, so re-running is free.
#   2. evaluate_lines.py -- ARU-Net baseline fusion, seam-carve disconnection +
#      small-object cleanup (per-subset params from postproc.PER_MS_PARAMS), then
#      the Zottin metric -> Pixel_IU / Line_IU / DR / RA / FM.
#
# No code changes were needed: both scripts build the run folder as
# unet_<encoder>_<family>_<subset>, so passing "tu-pvt_v2_b0_sz" as the encoder
# resolves to unet_tu-pvt_v2_b0_sz_udiads_<subset> -- exactly how the size-axis
# runs are named. The _sz / _szrand suffix rides along inside the encoder string.
#
# Prob cache -> 99_evaluation/01_simple_segmentation/u-diads-tl/prob_cache/<run>/
# Metrics    -> 99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_zottin.csv
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

SUBSETS=(Latin14396 Latin2 Syr341)
OUT=../../99_evaluation/01_simple_segmentation/u-diads-tl/size_axis_zottin.csv

ARCHS=(
  vit_tiny_patch16_224.augreg_in21k
  tu-convnext_femto
  tu-pvt_v2_b0
  tu-convnext_pico
  tu-pvt_v2_b1
  tu-convnext_tiny
  vit_small_patch16_224.augreg_in21k
  tu-pvt_v2_b2
)

ENCODERS=()
for a in "${ARCHS[@]}"; do ENCODERS+=("${a}_sz" "${a}_szrand"); done
echo "scoring ${#ENCODERS[@]} encoder-arms x ${#SUBSETS[@]} subsets = $(( ${#ENCODERS[@]} * ${#SUBSETS[@]} )) runs"

echo
echo "########## 1/2  caching probability maps ##########"
"$PY" preprocessing/cache_predictions.py \
  --family udiads --split test \
  --subsets "${SUBSETS[@]}" \
  --encoders "${ENCODERS[@]}"

echo
echo "########## 2/2  fusion + postproc + Zottin (parallel) ##########"
# eval_udiads_matrix.py rather than evaluate_lines.py: identical computation and
# verified to reproduce its numbers exactly, but it caches the ARU-Net baseline
# per page instead of per run and spreads the 23 s/page Zottin metric over a
# process pool. Single-threaded this stage is ~4.6 h; here it is ~20 min.
"$PY" eval_udiads_matrix.py

echo
echo "metrics -> $OUT"
