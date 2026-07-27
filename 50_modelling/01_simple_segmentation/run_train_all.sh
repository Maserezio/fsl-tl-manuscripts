#!/usr/bin/env bash
# ============================================================================
# Train the 6-backbone comparison (CNN vs ViT, supervised vs SSL) across all 6
# subsets: DIVA-HisDB (CB55, CS18, CS863) + U-DIADS-TL (Latin14396, Latin2,
# Syr341). All backbones go through smp.Unet uniformly -- the two ViT-S/16
# backbones use hier_encoder's SFP neck (models/vit_hier.py) instead of a
# different decoder architecture (see models/smp_unet.py).
#
#   resnet34               24M   CNN, supervised ImageNet
#   resnet50                32M   CNN, supervised ImageNet
#   tu-convnext_tiny.in12k_ft_in1k       28.6M  CNN, supervised ImageNet
#   tu-convnext_tiny.dinov3_lvd1689m     27.8M  CNN, DINOv3 SSL
#   vit_small_patch16_224.augreg_in21k   ~22M   ViT, supervised ImageNet
#   vit_small_patch16_dinov3             21.6M  ViT, DINOv3 SSL (Meta-gated weights --
#                                                fails loudly if unreachable, never
#                                                silently trains on random init)
#
# Per-family recipe kept as each family's own established best (not forced
# identical across families): DIVA = 50 epochs, k_shot=20 baked into
# configs/unet_resnet_diva.yaml; U-DIADS = 200 epochs, k_shot=3 baked into
# configs/unet_resnet_u_diads.yaml. batch=4 / lr=1e-3 / lr_backbone=1e-4 /
# weight_decay=1e-4 identical across both (same config values).
#
# Checkpoints -> 80_models/01_simple_segmentation/<diva-hisdb|u-diads-tl>/segmentation/
#                simple_segmentation/unet_<encoder>_<family>_<subset>/best.pth
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python

#   encoder
BACKBONES=(
  "resnet34"
  "resnet50"
  "tu-convnext_tiny.in12k_ft_in1k"
  "tu-convnext_tiny.dinov3_lvd1689m"
  "vit_small_patch16_224.augreg_in21k"
  "vit_small_patch16_dinov3"
)

train_one() {
  local family="$1" cfg="$2" subset="$3" epochs="$4" out="$5" enc="$6"
  local run="unet_${enc}_${family}_${subset}"
  if [[ -f "../../$out/$run/best.pth" ]]; then
    echo "[skip] $run already trained"
    return
  fi
  echo "######## TRAIN  ${run}  (${epochs} epochs) ########"
  "$PY" train.py --config "$cfg" \
    --arch unet --encoder "$enc" --manuscript "$subset" \
    --batch-size 4 --epochs "$epochs" \
    --lr 0.001 --lr-backbone 0.0001 --weight-decay 0.0001 \
    --run-name "$run" --out-dir "$out"
}

DIVA_CFG=configs/unet_resnet_diva.yaml
DIVA_OUT=80_models/01_simple_segmentation/diva-hisdb/segmentation/simple_segmentation
DIVA_SUBSETS=(CB55 CS18 CS863)
DIVA_EPOCHS=50

UDIADS_CFG=configs/unet_resnet_u_diads.yaml
UDIADS_OUT=80_models/01_simple_segmentation/u-diads-tl/segmentation/simple_segmentation
UDIADS_SUBSETS=(Latin14396 Latin2 Syr341)
UDIADS_EPOCHS=200

for ms in "${DIVA_SUBSETS[@]}"; do
  for enc in "${BACKBONES[@]}"; do
    train_one diva "$DIVA_CFG" "$ms" "$DIVA_EPOCHS" "$DIVA_OUT" "$enc"
  done
done

for ms in "${UDIADS_SUBSETS[@]}"; do
  for enc in "${BACKBONES[@]}"; do
    train_one udiads "$UDIADS_CFG" "$ms" "$UDIADS_EPOCHS" "$UDIADS_OUT" "$enc"
  done
done

echo "ALL TRAINING DONE (6 backbones x 6 subsets = 36 checkpoints)"
