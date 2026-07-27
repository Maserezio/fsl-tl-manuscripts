#!/usr/bin/env bash
# Colab worker: the 33 missing 100-epoch runs excluding every CB55 experiment.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python}
REPO_ROOT=$(cd ../.. && pwd)
LOCAL_OUT_ROOT=${LOCAL_OUT_ROOT:-$REPO_ROOT/colab_outputs}
RESULTS_DIR=${RESULTS_DIR:?Set RESULTS_DIR to a persistent Google Drive directory}

train_one() {
  local family=$1 cfg=$2 subset=$3 encoder=$4 family_dir=$5
  local run="unet_${encoder}_${family}_${subset}_100ep"
  local local_parent="$LOCAL_OUT_ROOT/$family_dir/segmentation/simple_segmentation"
  local drive_parent="$RESULTS_DIR/$family_dir/segmentation/simple_segmentation"
  local drive_final="$drive_parent/$run/final.pth"

  if [[ -f "$drive_final" ]] && "$PY" - "$drive_final" <<'PY'
import sys, torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)
raise SystemExit(0 if c.get("epoch") == 100 else 1)
PY
  then
    echo "[skip] $run already completed on Drive"
    return
  fi

  mkdir -p "$local_parent" "$drive_parent"
  echo "######## TRAIN $run ########"
  "$PY" train.py --config "$cfg" --arch unet --encoder "$encoder" \
    --manuscript "$subset" --batch-size 4 --epochs 100 --patience 0 \
    --lr 0.001 --lr-backbone 0.0001 --weight-decay 0.0001 \
    --run-name "$run" --out-dir "$local_parent"

  rm -rf "$drive_parent/$run"
  cp -a "$local_parent/$run" "$drive_parent/$run"
  echo "[saved] $drive_parent/$run"
}

DIVA_CFG=configs/unet_resnet_diva.yaml
UDIADS_CFG=configs/unet_resnet_u_diads.yaml

STANDARD_BACKBONES=(
  resnet34 resnet50
  tu-convnext_tiny.in12k_ft_in1k
  tu-convnext_tiny.dinov3_lvd1689m
  vit_small_patch16_224.augreg_in21k
  vit_small_patch16_dinov3
)
for subset in CS18 CS863; do
  for encoder in "${STANDARD_BACKBONES[@]}"; do
    train_one diva "$DIVA_CFG" "$subset" "$encoder" diva-hisdb
  done
done

# Only configurations not already verified at 100 epochs locally.
for spec in \
  'Latin14396 resnet34' 'Latin14396 resnet50' \
  'Latin2 resnet34' 'Latin2 resnet50' \
  'Latin2 tu-convnext_tiny.in12k_ft_in1k' \
  'Latin2 tu-convnext_tiny.dinov3_lvd1689m' \
  'Syr341 resnet34' 'Syr341 resnet50' \
  'Syr341 vit_small_patch16_dinov3'
do
  read -r subset encoder <<<"$spec"
  train_one udiads "$UDIADS_CFG" "$subset" "$encoder" u-diads-tl
done

FOUNDATION_BACKBONES=(dinov2 dinov2_reg dinov3 am-radio)
for subset in Latin14396 Latin2 Syr341; do
  for encoder in "${FOUNDATION_BACKBONES[@]}"; do
    train_one udiads "$UDIADS_CFG" "$subset" "$encoder" u-diads-tl
  done
done

echo "All 33 non-CB55 runs completed and copied to Drive."
