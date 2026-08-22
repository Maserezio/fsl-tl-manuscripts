#!/usr/bin/env bash
# ============================================================================
# RQ1 -- does it matter WHICH pages you label?
#
# One fixed model (RT-DETR convnext_tiny/imagenet at 1152, then a crop segmenter
# trained with Tversky), run across every DIVA subset x every selection method x
# k in {1,3,5,10,15}, scored with the five official metrics.
#
# 90 cells, but only 77 distinct page sets: methods often agree, and at k=15 of 20
# any two selections must overlap on at least 10 pages. shot_selection_plan.py does
# that dedup; cells sharing a page set share a trained model and are expanded back
# out at reporting time.
#
# k constrains BOTH stages. A crop segmenter that kept all 20 pages while only the
# detector was starved would flatter the two-stage numbers for a reason unrelated to
# the selection method under test.
#
# EVAL_EVERY_EPOCHS=5: the detector's validation pass runs on the full val split and
# costs the same regardless of k, so at k=1 it otherwise dominates the run. This
# leaves 20 checkpoint-selection points instead of 100. The default stays 1 elsewhere,
# so the backbone matrix remains reproducible.
#
# Selections come from make_shot_selection.py, NOT from the notebook: the notebook
# looks for a `public-train` directory that does not exist and silently falls back to
# selecting TEST pages.
#
# Outputs -> 99_evaluation/shot_selection_diva.csv
# Safe to re-run: every step is guarded by a .done marker.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
REPO=$(cd .. && pwd)

INIT=imagenet
BACKBONE=convnext_tiny
IMAGE_SIZE=1152
ARM=tversky
EVAL_EVERY=5        # detector: validate every Nth epoch
SEG_EVAL_EVERY=5    # crop segmenter: same lever, measured 3x faster (23 -> 7.5 min)
MANIFEST="$REPO/99_evaluation/shot_selection_manifest.json"
CURVE="99_evaluation/shot_selection_diva.csv"
MARK="$REPO/80_models/.shotsel_markers"
mkdir -p "$MARK"

exec 9>/tmp/shot_selection.lock
flock -n 9 || { echo "ERROR: another copy of this script is running" >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "ERROR: run shot_selection_plan.py first" >&2; exit 1; }

# job<TAB>subset<TAB>k<TAB>method  -- method is any one cell using that job; the page
# set is identical for all of them, so which one supplies the selection is irrelevant.
JOBS=$("$PY" - "$MANIFEST" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
first = {}
for c in m["cells"]:
    first.setdefault(c["job"], c)
for j in m["jobs"]:
    c = first[j["job"]]
    print(f"{j['job']}\t{j['subset']}\t{j['k']}\t{c['method']}")
PYEOF
)
TOTAL=$(echo "$JOBS" | wc -l)
echo "$TOTAL unique jobs"

sel_file() {  # subset k -> precomputed path ("" for random)
  [ "$1" = "random" ] && echo "" || \
    echo "00_data/DIVA-HisDB/shot_selection/diva_$(echo "$2" | tr 'A-Z' 'a-z')_$3_diverse_images.txt"
}

echo "########## STAGE 1/3 -- detectors ##########"
i=0
while IFS=$'\t' read -r JOB SUB K METHOD; do
  i=$((i+1))
  if [ -f "$MARK/det_$JOB" ]; then echo "[skip] ($i/$TOTAL) detector $JOB"; continue; fi
  echo "######## ($i/$TOTAL) detector $JOB  $SUB k=$K via $METHOD ########"
  DATASET="$SUB" BACKBONE="$BACKBONE" INIT="$INIT" IMAGE_SIZE="$IMAGE_SIZE" \
    K_SHOT="$K" K_SHOT_METHOD="$METHOD" \
    K_SHOT_PRECOMPUTED="$(SF=$(sel_file "$METHOD" "$SUB" "$K"); [ -n "$SF" ] && echo "$REPO/$SF")" \
    EVAL_EVERY_EPOCHS="$EVAL_EVERY" RUN_NAME="sel_$JOB" \
    "$PY" 02_2stage/train_rtdetr_hf.py
  touch "$MARK/det_$JOB"
done <<< "$JOBS"

echo
echo "########## STAGE 2/3 -- crop segmenters ($ARM) ##########"
i=0
while IFS=$'\t' read -r JOB SUB K METHOD; do
  i=$((i+1))
  if [ -f "$MARK/seg_$JOB" ]; then echo "[skip] ($i/$TOTAL) segmenter $JOB"; continue; fi
  echo "######## ($i/$TOTAL) segmenter $JOB ########"
  # random has no precomputed file; passing an empty --k-shot-precomputed would make
  # few_shot_sampler take the "file missing" branch silently instead of the random one.
  SF=$(sel_file "$METHOD" "$SUB" "$K")
  PRE=()
  [ -n "$SF" ] && PRE=(--k-shot-precomputed "$REPO/$SF")
  "$PY" 02_2stage/train_crops_loss_ablation_2stage.py \
    --family diva --subset "$SUB" --backbone resnet34 \
    --crop-width 1024 --crop-height 256 \
    --k-shot "$K" --k-shot-method "$METHOD" --arms "$ARM" --tag "$JOB" \
    --eval-every "$SEG_EVAL_EVERY" "${PRE[@]}"
  touch "$MARK/seg_$JOB"
done <<< "$JOBS"

echo
echo "########## STAGE 3/3 -- evaluation ##########"
i=0
while IFS=$'\t' read -r JOB SUB K METHOD; do
  i=$((i+1))
  if [ -f "$MARK/eval_$JOB" ]; then echo "[skip] ($i/$TOTAL) eval $JOB"; continue; fi
  echo "######## ($i/$TOTAL) eval $JOB ########"
  SUBSET="$SUB" RUN_NAME="sel_$JOB" K_SHOT="$K" \
    SEG_KSHOT_ROOT="80_models/02_2stage/diva-hisdb/crop_seg_kshot_1024x256/$SUB/$JOB/k$K/$ARM" \
    CURVE_CSV="$CURVE" CURVE_APPROACH="two_stage" CURVE_METHOD="$JOB" \
    "$PY" 02_2stage/predict_and_eval_rtdetr_diva.py
  touch "$MARK/eval_$JOB"
done <<< "$JOBS"

echo
echo "########## expanding jobs back to cells ##########"
"$PY" "$REPO/99_evaluation/expand_shot_selection.py"
"$PY" "$REPO/99_evaluation/make_rq1_tables.py"
echo "SHOT_SELECTION_FINISHED -> $REPO/$CURVE"
