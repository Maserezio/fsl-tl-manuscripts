#!/usr/bin/env bash
# ============================================================================
# Skip-connection ablation, CB55, full-page semantic segmentation.
#
# Hypothesis under test: U-Net tolerates a *flat* ViT backbone precisely because
# the skip connections re-inject the high-frequency spatial detail the ViT's
# stride-16 tokens threw away. If so, removing the skips should cost the ViT arm
# much more than the CNN arm, whose coarse backbone features already carry more
# spatial structure of their own.
#
#   2 backbones x 3 skip levels = 6 runs
#     resnet34   CNN, native pyramid, supervised ImageNet
#     dinov2     ViT-S/14 DINOv2 SSL, flat tokens -> SFP neck (models/vit_hier.py)
#                (the SFP-ViT arm already present in diva_backbone_matrix.csv)
#
#     n_skips=4  all four skips        (baseline, identical to the matrix runs)
#     n_skips=2  two deepest only      (s16,s8 live; s4,s2 zeroed)
#     n_skips=0  none                  (all four zeroed; decoder sees s32 only)
#
# Skips are ABLATED BY ZEROING the encoder's skip tensors (a forward hook on the
# encoder, models/smp_unet.py). Architecture, layer shapes and parameter count
# are bit-for-bit identical across all three levels -- otherwise this would be a
# capacity ablation, not a skip ablation.
#
# Everything else is exactly the DIVA recipe of run_train_all.sh: same config
# (configs/unet_resnet_diva.yaml -> k_shot=20, crop 448, grayscale_variance page
# selection), same split, 50 epochs, batch 4, lr 1e-3 / lr_backbone 1e-4 /
# weight_decay 1e-4. The one addition is --seed: the earlier matrix runs were
# unseeded, and within an ablation the three levels of an arm must not differ by
# RNG. n_skips=4 is therefore re-trained here rather than reusing the matrix
# checkpoint, so all six cells share one code path and one seed.
#
# Checkpoints -> 80_models/.../simple_segmentation/skipabl_<enc>_diva_CB55_s<N>/
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

CFG=configs/unet_resnet_diva.yaml
OUT=80_models/01_simple_segmentation/diva-hisdb/segmentation/simple_segmentation
SUBSET=CB55
EPOCHS=50
SEED=0

BACKBONES=("resnet34" "dinov2")
SKIPS=(4 2 0)

for enc in "${BACKBONES[@]}"; do
  for n in "${SKIPS[@]}"; do
    run="skipabl_${enc}_diva_${SUBSET}_s${n}"
    if [[ -f "../../$OUT/$run/best.pth" ]]; then
      echo "[skip] $run already trained"
      continue
    fi
    echo "######## TRAIN  ${run}  (${EPOCHS} epochs, n_skips=${n}) ########"
    "$PY" train.py --config "$CFG" \
      --arch unet --encoder "$enc" --manuscript "$SUBSET" \
      --n-skips "$n" --seed "$SEED" \
      --batch-size 4 --epochs "$EPOCHS" \
      --lr 0.001 --lr-backbone 0.0001 --weight-decay 0.0001 \
      --run-name "$run" --out-dir "$OUT"
  done
done

echo "ALL TRAINING DONE (2 backbones x 3 skip levels = 6 checkpoints)"
