#!/usr/bin/env bash
# ============================================================================
# Train the 10-backbone 2-stage DETECTION comparison (CNN vs ViT, supervised vs
# SSL, plus native foundation models) across all 6 subsets: DIVA-HisDB (CB55,
# CS18, CS863) + U-DIADS-TL (Latin14396, Latin2, Syr341).
#
# The backbone set mirrors 01_simple_segmentation exactly (same names route to
# the same weights); transformer backbones go through hier_encoder's SFP neck
# (see hier_encoder_yolo.py), CNNs through stock YOLOv8 detection YAMLs
# (configs/detection/). All selection lives in train_detector.py's BACKBONES.
#
#   yolov8n                               stock YOLOv8 CSPDarknet baseline (COCO-pretrained)
#   resnet34                              CNN, supervised ImageNet
#   resnet50                              CNN, supervised ImageNet
#   convnext                              CNN, supervised (convnext_tiny.in12k_ft_in1k)
#   convnext_dinov3                       CNN, DINOv3 SSL (convnext_tiny.dinov3_lvd1689m)
#   vit_small_patch16_224.augreg_in21k    ViT-S/16, supervised ImageNet   (SFP)
#   vit_small_patch16_dinov3              ViT-S/16, DINOv3 SSL            (SFP)
#   dinov2                                native DINOv2 ViT-S/14 SSL      (SFP)
#   dinov2_reg                            native DINOv2 ViT-S/14+reg SSL  (SFP)
#   am-radio                              native AM-RADIO v2.5-B          (SFP)
#
# Native DINOv3 ViT-S/16 == vit_small_patch16_dinov3 (same weights), so it is not
# listed twice. DINOv2 (patch-14) needs imgsz divisible by 32 like every backbone;
# the SFP neck snaps its pyramid to a coherent [P3,P4,P5] internally.
#
# Checkpoints -> 80_models/02_2stage/<diva-hisdb|u-diads-tl>/detection/
#                yolov8_<encoder>_<family>_<subset>/weights/best.pt
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../.venv/bin/python}

BACKBONES=(
  "yolov8n"
  "resnet34"
  "resnet50"
  "convnext"
  "convnext_dinov3"
  "vit_small_patch16_224.augreg_in21k"
  "vit_small_patch16_dinov3"
  "dinov2"
  "dinov2_reg"
  "am-radio"
)

IMGSZ=${IMGSZ:-1280}
BATCH=${BATCH:-2}
LR0=${LR0:-0.001}
PATIENCE=${PATIENCE:-50}

train_one() {
  local family="$1" data="$2" subset="$3" epochs="$4" out="$5" enc="$6"
  local run="yolov8_${enc}_${family}_${subset}"
  if [[ -f "../../$out/$run/weights/best.pt" ]]; then
    echo "[skip] $run already trained"
    return
  fi
  echo "######## TRAIN  ${run}  (${epochs} epochs) ########"
  "$PY" train_detector.py --encoder "$enc" --data "$data" \
    --project "../../$out" --name "$run" \
    --epochs "$epochs" --imgsz "$IMGSZ" --batch "$BATCH" \
    --lr0 "$LR0" --patience "$PATIENCE"
}

DIVA_OUT=80_models/02_2stage/diva-hisdb/detection
DIVA_EPOCHS=${DIVA_EPOCHS:-100}
declare -A DIVA_DATA=(
  [CB55]=configs/data/diva_cb55_detect.yaml
  [CS18]=configs/data/diva_cs18_detect.yaml
  [CS863]=configs/data/diva_cs863_detect.yaml
)

UDIADS_OUT=80_models/02_2stage/u-diads-tl/detection
UDIADS_EPOCHS=${UDIADS_EPOCHS:-100}
declare -A UDIADS_DATA=(
  [Latin14396]=configs/data/udiads_latin14396_detect.yaml
  [Latin2]=configs/data/udiads_latin2_detect.yaml
  [Syr341]=configs/data/udiads_syr341_detect.yaml
)

for ms in CB55 CS18 CS863; do
  for enc in "${BACKBONES[@]}"; do
    train_one diva "${DIVA_DATA[$ms]}" "$ms" "$DIVA_EPOCHS" "$DIVA_OUT" "$enc"
  done
done

for ms in Latin14396 Latin2 Syr341; do
  for enc in "${BACKBONES[@]}"; do
    train_one udiads "${UDIADS_DATA[$ms]}" "$ms" "$UDIADS_EPOCHS" "$UDIADS_OUT" "$enc"
  done
done

echo "ALL TRAINING DONE (10 backbones x 6 subsets = 60 checkpoints)"
