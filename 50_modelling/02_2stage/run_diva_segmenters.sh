#!/usr/bin/env bash
# Stage 2 of the DIVA matrix, split out of run_diva_rtdetr_matrix.sh so the detectors
# (all 30 trained) do not get re-walked. CB55's three arms already have best.pth and
# their history, so they skip straight to the reporting step that crashed the first time.
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
for ds in CB55 CS863 CS18; do
  echo "######## segmenters: $ds ########"
  "$PY" train_crops_loss_ablation_2stage.py \
    --family diva --subset "$ds" --backbone resnet34 \
    --crop-width 1024 --crop-height 256
  echo "[ok] segmenters $ds"
done
echo "DIVA_SEGMENTERS_FINISHED"
