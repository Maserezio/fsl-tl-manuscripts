#!/usr/bin/env bash
# ============================================================================
# RQ1 -- how much labeled data does each approach actually need?
#
# One encoder (convnext_tiny, ImageNet init) in BOTH pipelines, swept over
# k = 1, 3, 5, 10, 15 labeled CB55 pages, scored with the five official metrics.
# The result is two comparable curves: single-stage U-Net vs the two-stage
# detector + crop segmenter.
#
# The k pages come from few_shot_sampler's grayscale-variance selector in all three
# trainers, so every approach sees the IDENTICAL pages at a given k -- verified: the
# selection is nested (k=1 subset of k=3 subset of ... subset of k=15), so growing k
# adds pages rather than swapping them, which is what makes the curve a curve.
#
# In the two-stage arm k constrains BOTH stages. Letting the crop segmenter keep all
# 20 pages while only the detector is starved would make its curve look better than
# the single-stage one for a reason that has nothing to do with the architecture.
#
# CB55 train is 20 pages, so k=15 is the largest point that is still few-shot; the
# existing full-data runs (k=20) are the ceiling to compare against.
#
# Stage 3 evaluates. It is separated from training so the GPU is never idle waiting
# on the Java evaluator, which is CPU-bound.
#
# Outputs -> 99_evaluation/kshot_cb55.csv      (both approaches, one row per k)
#            99_evaluation/RQ1_ALL_RESULTS.{md,tex} via make_rq1_tables.py
#
# Safe to re-run: each step is guarded by a .done marker.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
REPO=$(cd .. && pwd)

K_VALUES=(${K_VALUES:-1 3 5 10 15})
ENCODER_01="tu-convnext_tiny.in12k_ft_in1k"   # smp/timm name for the one-stage U-Net
BACKBONE_02="convnext_tiny"                   # HF name for the RT-DETR backbone
INIT="imagenet"
SUBSET="CB55"
ARM="bce"                                     # loss arms differ within noise on CB55
IMAGE_SIZE=1152
CURVE_CSV="99_evaluation/kshot_cb55.csv"

MARK="$REPO/80_models/.kshot_markers"
mkdir -p "$MARK"
exec 9>/tmp/kshot_cb55.lock
flock -n 9 || { echo "ERROR: another copy of this script is running" >&2; exit 1; }

step() {  # step <marker> <description> -- skips if already done
  local m="$MARK/$1"; shift
  local desc="$1"; shift
  if [[ -f "$m" ]]; then echo "[skip] $desc"; return 1; fi
  echo "######## $desc ########"
  return 0
}

echo "########## STAGE 1/3 -- single-stage U-Net ##########"
for K in "${K_VALUES[@]}"; do
  if step "s1_k$K" "01 simple segmentation, k=$K"; then
    # cd: train.py resolves --config relative to the working directory (data_root is
    # resolved against the script's own location, so only the config path cares).
    # --out-dir: train.py's default checkpoint path has no dataset-family component,
    # but evaluate_lines.py looks under .../diva-hisdb/..., so without this the runs
    # land somewhere the scorer will never find them.
    ( cd 01_simple_segmentation && "../$PY" train.py \
      --config configs/unet_resnet_diva.yaml \
      --manuscript "$SUBSET" --encoder "$ENCODER_01" \
      --k-shot "$K" --k-shot-method grayscale_variance \
      --run-name "kshot_convnext_tiny_diva_${SUBSET}_k${K}" \
      --out-dir 80_models/01_simple_segmentation/diva-hisdb/segmentation/simple_segmentation )
    touch "$MARK/s1_k$K"
  fi
done

echo
echo "########## STAGE 2/3 -- two-stage: detector, then crop segmenter ##########"
for K in "${K_VALUES[@]}"; do
  if step "s2det_k$K" "02 detector, k=$K"; then
    DATASET="$SUBSET" BACKBONE="$BACKBONE_02" INIT="$INIT" IMAGE_SIZE="$IMAGE_SIZE" \
      K_SHOT="$K" RUN_NAME="kshot_${BACKBONE_02}_${INIT}_k${K}" \
      "$PY" 02_2stage/train_rtdetr_hf.py
    touch "$MARK/s2det_k$K"
  fi
  if step "s2seg_k$K" "02 crop segmenter ($ARM), k=$K"; then
    "$PY" 02_2stage/train_crops_loss_ablation_2stage.py \
      --family diva --subset "$SUBSET" --backbone resnet34 \
      --crop-width 1024 --crop-height 256 \
      --k-shot "$K" --arms "$ARM"
    touch "$MARK/s2seg_k$K"
  fi
done

echo
echo "########## STAGE 3/3 -- evaluation ##########"
# One-stage: evaluate_lines.py takes explicit run folder names and already emits the
# five-metric schema, so the whole sweep is a single call.
RUNS=()
for K in "${K_VALUES[@]}"; do RUNS+=("kshot_convnext_tiny_diva_${SUBSET}_k${K}"); done
if step "s3_simple" "01 evaluation (all k at once)"; then
  ( cd 01_simple_segmentation && "../$PY" evaluate_lines.py \
    --family diva --subsets "$SUBSET" --runs "${RUNS[@]}" \
    --out "$REPO/99_evaluation/kshot_cb55_simple_raw.csv" )
  touch "$MARK/s3_simple"
fi

for K in "${K_VALUES[@]}"; do
  if step "s3two_k$K" "02 evaluation, k=$K"; then
    SUBSET="$SUBSET" RUN_NAME="kshot_${BACKBONE_02}_${INIT}_k${K}" \
      SEG_KSHOT_ROOT="80_models/02_2stage/diva-hisdb/crop_seg_kshot_1024x256/${SUBSET}/k${K}/${ARM}" \
      CURVE_CSV="$CURVE_CSV" CURVE_APPROACH="two_stage" K_SHOT="$K" \
      "$PY" 02_2stage/predict_and_eval_rtdetr_diva.py
    touch "$MARK/s3two_k$K"
  fi
done

echo
echo "########## merging the one-stage rows into the curve ##########"
K_VALUES="${K_VALUES[*]}" CURVE_CSV="$CURVE_CSV" SUBSET="$SUBSET" \
  "$PY" "$REPO/99_evaluation/merge_kshot_simple.py"

"$PY" "$REPO/99_evaluation/make_rq1_tables.py"
echo
echo "KSHOT_CB55_FINISHED -> $REPO/$CURVE_CSV"
