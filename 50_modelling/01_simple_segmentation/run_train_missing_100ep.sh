#!/usr/bin/env bash
# Train the requested 1-class simple-segmentation runs that were not verified
# to have completed exactly 100 epochs in the checkpoint audit of 2026-07-25.
# Existing experiments are never overwritten: these use *_100ep run names.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-../../.venv/bin/python}
EPOCHS=100

train_one() {
  local family=$1 cfg=$2 subset=$3 out=$4 encoder=$5
  local safe_encoder=${encoder//\//-}
  local run="unet_${safe_encoder}_${family}_${subset}_100ep"
  local final="../../${out}/${run}/final.pth"

  if [[ -f "$final" ]] && "$PY" - "$final" <<'PY'
import sys, torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
cfg = checkpoint.get("cfg", {})
ok = (
    checkpoint.get("epoch") == 100
    and cfg.get("training", {}).get("epochs") == 100
    and cfg.get("model", {}).get("encoder_name") == sys.argv[1].split("/unet_", 1)[1].rsplit("_", 3)[0]
)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[skip] ${run}: verified 100-epoch checkpoint"
    return
  fi

  echo "######## TRAIN ${run} (100 epochs, 1 output class) ########"
  "$PY" train.py --config "$cfg" \
    --arch unet --encoder "$encoder" --manuscript "$subset" \
    --batch-size 4 --epochs "$EPOCHS" --patience 0 \
    --lr 0.001 --lr-backbone 0.0001 --weight-decay 0.0001 \
    --run-name "$run" --out-dir "$out"
}

DIVA_CFG=configs/unet_resnet_diva.yaml
DIVA_OUT=80_models/01_simple_segmentation/diva-hisdb/segmentation/simple_segmentation
UDIADS_CFG=configs/unet_resnet_u_diads.yaml
UDIADS_OUT=80_models/01_simple_segmentation/u-diads-tl/segmentation/simple_segmentation

# Six-backbone matrix: every DIVA run is missing at 100 epochs.
STANDARD_BACKBONES=(
  resnet34
  resnet50
  tu-convnext_tiny.in12k_ft_in1k
  tu-convnext_tiny.dinov3_lvd1689m
  vit_small_patch16_224.augreg_in21k
  vit_small_patch16_dinov3
)
for subset in CB55 CS18 CS863; do
  for encoder in "${STANDARD_BACKBONES[@]}"; do
    train_one diva "$DIVA_CFG" "$subset" "$DIVA_OUT" "$encoder"
  done
done

# U-DIADS missing/short/non-100 runs. The other nine standard runs completed 100.
for spec in \
  'Latin14396 resnet34' 'Latin14396 resnet50' \
  'Latin2 resnet34' 'Latin2 resnet50' \
  'Latin2 tu-convnext_tiny.in12k_ft_in1k' \
  'Latin2 tu-convnext_tiny.dinov3_lvd1689m' \
  'Syr341 resnet34' 'Syr341 resnet50' \
  'Syr341 vit_small_patch16_dinov3'
do
  read -r subset encoder <<<"$spec"
  train_one udiads "$UDIADS_CFG" "$subset" "$UDIADS_OUT" "$encoder"
done

# Foundation matrix (CS18/CS863 were explicitly marked not applicable).
FOUNDATION_BACKBONES=(dinov2 dinov2_reg dinov3 am-radio)
for subset in CB55; do
  for encoder in "${FOUNDATION_BACKBONES[@]}"; do
    train_one diva "$DIVA_CFG" "$subset" "$DIVA_OUT" "$encoder"
  done
done
for subset in Latin14396 Latin2 Syr341; do
  for encoder in "${FOUNDATION_BACKBONES[@]}"; do
    train_one udiads "$UDIADS_CFG" "$subset" "$UDIADS_OUT" "$encoder"
  done
done

echo "All 43 missing 100-epoch runs completed."
