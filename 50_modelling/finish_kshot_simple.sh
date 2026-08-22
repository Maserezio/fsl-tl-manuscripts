#!/usr/bin/env bash
# Stop the sweep once the last one-stage run (k=15) is done, then score the five
# points with the official evaluator.
#
# The two-stage half of the sweep (5 detectors + 5 crop segmenters + their scoring)
# is deliberately NOT run -- the curve produced here covers the single-stage U-Net
# only. Its steps have no .done markers, so re-running kshot_cb55.sh later picks the
# two-stage arm up from a clean slate.
set -uo pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
MARK="$REPO/80_models/.kshot_markers"
LOG="$REPO/kshot_cb55.log"

echo "[finish] $(date '+%H:%M:%S') waiting for k=15 to finish"
while [ ! -f "$MARK/s1_k15" ]; do
  if ! pgrep -f "kshot_cb55.sh" >/dev/null; then
    echo "[finish] sweep died before k=15 completed; nothing to stop"
    break
  fi
  sleep 60
done

# Kill the sweep before it starts training detectors. Kill the children too: the
# script's own PID going away does not stop a python already on the GPU.
echo "[finish] $(date '+%H:%M:%S') stopping the sweep before stage 2"
for p in $(ps -eo pid,cmd --no-headers \
           | grep -E "kshot_cb55\.sh|train_rtdetr_hf\.py|train_crops_loss_ablation_2stage\.py" \
           | grep -v "bash -c" | awk '{print $1}'); do
  kill "$p" 2>/dev/null
done
sleep 20

echo "[finish] $(date '+%H:%M:%S') scoring the five one-stage runs"
RUNS=()
for K in 1 3 5 10 15; do RUNS+=("kshot_convnext_tiny_diva_CB55_k${K}"); done
( cd 01_simple_segmentation && ../../.venv/bin/python evaluate_lines.py \
    --family diva --subsets CB55 --runs "${RUNS[@]}" \
    --out "$REPO/99_evaluation/kshot_cb55_simple_raw.csv" )

K_VALUES="1 3 5 10 15" CURVE_CSV="99_evaluation/kshot_cb55.csv" SUBSET=CB55 \
  ../.venv/bin/python "$REPO/99_evaluation/merge_kshot_simple.py"
../.venv/bin/python "$REPO/99_evaluation/make_rq1_tables.py"
echo "KSHOT_SIMPLE_FINISHED"
