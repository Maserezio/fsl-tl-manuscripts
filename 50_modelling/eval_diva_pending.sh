#!/usr/bin/env bash
# Score the DIVA two-stage cells that are trained but never evaluated:
#   - CS863: all 10 backbone-matrix arms (the run was started and stopped)
#   - pvt_v2_b2 x {in1k, rand} on CB55/CS18/CS863 -- trained in Colab, size M in the
#     pvt branch, which no other arm covers
# Detectors and the full-data crop segmenters are already trained; this is evaluation
# only. Settings are the corrected ones: confidence grid from 0.01, mask threshold 0.5,
# per-subset width filter (0.0 / 0.8 / 0.5, calibrated on validation).
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
REPO=$(cd .. && pwd)
ARM=bce
MARK="$REPO/80_models/.diva_pending"; mkdir -p "$MARK"

exec 9>/tmp/diva_pending.lock
flock -n 9 || { echo "ERROR: another copy is running" >&2; exit 1; }

run () {                       # subset run_name
  local sub="$1" name="$2"
  if [ -f "$MARK/${sub}_${name}" ]; then echo "[skip] $sub/$name"; return; fi
  echo "######## $sub / $name ########"
  SUBSET="$sub" RUN_NAME="$name" ARM="$ARM" "$PY" 02_2stage/predict_and_eval_rtdetr_diva.py \
    || { echo "[FAILED] $sub/$name"; return; }
  touch "$MARK/${sub}_${name}"
}

echo "########## CS863: 10 арок матрицы ##########"
for name in rtdetr_stock_random rtdetr_convnext_femto_random rtdetr_convnext_pico_random \
            rtdetr_convnext_tiny_random rtdetr_pvt_v2_b0_random rtdetr_pvt_v2_b1_random \
            rtdetr_convnext_tiny_imagenet rtdetr_pvt_v2_b0_imagenet \
            rtdetr_pvt_v2_b1_imagenet rtdetr_convnext_tiny_dinov3; do
  run CS863 "$name"
done

echo "########## pvt_v2_b2 на трёх подмножествах ##########"
for sub in CB55 CS18 CS863; do
  for name in pvt_v2_b2_rand pvt_v2_b2_in1k; do
    run "$sub" "$name"
  done
done
echo "DIVA_PENDING_FINISHED"
