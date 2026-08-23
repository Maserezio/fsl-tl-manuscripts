#!/usr/bin/env bash
# Line metrics for the RT-DETR backbone matrix on CS18 and CS863.
#
# CS18 and CS863 only: CB55 needs no width filter (its COCO already matches TASK-2
# inside the region), so its published numbers stand. Because the evaluation itself was corrected: detections are now
# also filtered by width (the training COCO annotates glosses as lines, TASK-2 does
# not), the confidence grid reaches below 0.1 where these detectors actually operate,
# and the mask is binarised at 0.5 to match how the segmenter was selected. CB55's
# earlier numbers predate all three, so it is rescored too rather than left mixed.
#
# ARM=bce matches how the CB55 rows in the same results.csv were produced, so the
# three subsets stay comparable. Metrics land in the `notes` column of each subset's
# results.csv, same as CB55.
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
REPO=$(cd .. && pwd)
ARM=bce
MARK="$REPO/80_models/.bbmatrix_markers"; mkdir -p "$MARK"

exec 9>/tmp/bbmatrix_eval.lock
flock -n 9 || { echo "ERROR: another copy is running" >&2; exit 1; }

ARMS_LIST="rtdetr_stock_random rtdetr_convnext_femto_random rtdetr_convnext_pico_random
rtdetr_convnext_tiny_random rtdetr_pvt_v2_b0_random rtdetr_pvt_v2_b1_random
rtdetr_convnext_tiny_imagenet rtdetr_pvt_v2_b0_imagenet rtdetr_pvt_v2_b1_imagenet
rtdetr_convnext_tiny_dinov3"

for SUB in CS18 CS863; do
  i=0
  for RUN in $ARMS_LIST; do
    i=$((i+1))
    if [ -f "$MARK/${SUB}_${RUN}" ]; then echo "[skip] $SUB/$RUN"; continue; fi
    echo "######## ($i/10) $SUB / $RUN ########"
    SUBSET="$SUB" RUN_NAME="$RUN" ARM="$ARM" \
      "$PY" 02_2stage/predict_and_eval_rtdetr_diva.py || { echo "[FAILED] $SUB/$RUN"; continue; }
    touch "$MARK/${SUB}_${RUN}"
  done
done
echo "BACKBONE_MATRIX_EVAL_FINISHED"
