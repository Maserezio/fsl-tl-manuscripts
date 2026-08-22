#!/usr/bin/env bash
# Wait for the last CB55 segmenter, stop the sweep before it moves on to CS18, then
# score CB55's 25 cells.
#
# The sweep trains all 77 segmenters before evaluating anything (training is GPU-bound,
# the Java evaluator is not, so interleaving them wastes the card). Here we cut in early
# to get CB55's five metrics without waiting for the other two subsets. CS18 and CS863
# keep their .done markers, so re-running run_shot_selection.sh later resumes exactly
# where this left off.
set -uo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
REPO=$(cd .. && pwd)
MARK="$REPO/80_models/.shotsel_markers"
ARM=tversky
CURVE="99_evaluation/shot_selection_diva.csv"
LAST=CB55_k15_7e3c513c

echo "[cb55] $(date '+%H:%M:%S') waiting for segmenter $LAST"
while [ ! -f "$MARK/seg_$LAST" ]; do
  if ! pgrep -f "run_shot_selection.sh" >/dev/null; then
    echo "[cb55] sweep died before $LAST finished"; break
  fi
  sleep 30
done

echo "[cb55] $(date '+%H:%M:%S') stopping the sweep before the CS18 segmenters"
for p in $(ps -eo pid,cmd --no-headers \
           | grep -E "run_shot_selection\.sh|train_crops_loss_ablation_2stage\.py" \
           | grep -v "bash -c" | awk '{print $1}'); do kill "$p" 2>/dev/null; done
sleep 20

JOBS=$("$PY" - "$REPO/99_evaluation/shot_selection_manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
for j in m["jobs"]:
    if j["subset"] == "CB55":
        print(f"{j['job']}\t{j['k']}")
PYEOF
)
N=$(echo "$JOBS" | wc -l)
echo "[cb55] scoring $N cells"
i=0
while IFS=$'\t' read -r JOB K; do
  i=$((i+1))
  if [ -f "$MARK/eval_$JOB" ]; then echo "[skip] ($i/$N) $JOB"; continue; fi
  echo "######## ($i/$N) eval $JOB k=$K ########"
  SUBSET=CB55 RUN_NAME="sel_$JOB" K_SHOT="$K" \
    SEG_KSHOT_ROOT="80_models/02_2stage/diva-hisdb/crop_seg_kshot_1024x256/CB55/$JOB/k$K/$ARM" \
    CURVE_CSV="$CURVE" CURVE_APPROACH="two_stage" CURVE_METHOD="$JOB" \
    "$PY" 02_2stage/predict_and_eval_rtdetr_diva.py || echo "[FAILED] $JOB"
  touch "$MARK/eval_$JOB"
done <<< "$JOBS"

"$PY" "$REPO/99_evaluation/expand_shot_selection.py"
"$PY" "$REPO/99_evaluation/make_rq1_tables.py"
echo "CB55_SHOTSEL_EVAL_FINISHED"
