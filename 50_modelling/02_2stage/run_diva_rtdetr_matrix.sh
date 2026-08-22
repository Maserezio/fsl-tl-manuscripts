#!/usr/bin/env bash
# ============================================================================
# RQ1 / 02 -- DIVA-HisDB two-stage: RT-DETR backbone matrix + crop segmenters.
#
# Same 10 arms and the same recipe as the U-DIADS matrix, so the two families are
# directly comparable. imgsz 1152 for all three subsets, as requested.
#
#   arms: stock R50-vd baseline, then {femto, pico, tiny, b0, b1} x random,
#         {tiny, b0, b1} x imagenet, tiny x dinov3
#
# RESOLUTION NOTE, worth reading before trusting CS18.
# 1152 rescales the page to a square canvas, so the line height each subset ends
# up with differs a lot:
#     CB55   152px native -> 27px    (75 lines/page)
#     CS863  119px native -> 27px    (56 lines/page)
#     CS18    58px native -> 13px   (170 lines/page)  <-- smallest in the project
# For reference, U-DIADS Syr341 had 19px at this resolution and reached FM 0.86
# only after the postprocessing sweep. CS18 is both finer and denser than that,
# so its numbers should be read as a lower bound rather than the subset's ceiling.
#
# pvt_v2_b2 is absent: it OOMs at 1152 on this 8GB card (measured -- femto 1.53 /
# pico 1.73 / tiny 2.68 / R50 2.71 / b0 3.98 / b1 4.69 GiB against a 6.1 GiB
# budget, b2 over). Run it on Colab if that cell is needed.
#
# Stage 2 trains the three losses per subset through the shared
# train_crops_loss_ablation_2stage.py (--family diva). DIVA supervision comes from
# PAGE TASK-2 polygons, which cover only the main text region -- fewer lines than
# the COCO file the detector is trained on, by design: the evaluation filters
# detections to that same region.
#
# Checkpoints -> 80_models/02_2stage/diva-hisdb/detection/rtdetr_hf/<subset>/<run>/
#                80_models/02_2stage/diva-hisdb/crop_seg_.../<subset>/<loss>/
# Metrics     -> 99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/<subset>/results.csv
#
# Runtime: 20 train pages -> ~1000 optimizer steps per run, ~30 min each.
# 30 detector runs is roughly 15 h; SUBSETS/ARMS below narrow it.
# Safe to re-run: finished runs are skipped via their .done marker.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
REPO=$(cd ../.. && pwd)

export IMAGE_SIZE=${IMAGE_SIZE:-1152}
read -r -a SUBSETS <<< "${SUBSETS:-CB55 CS863 CS18}"   # densest last

ARMS=(
  ":random"
  "convnext_femto:random"
  "convnext_pico:random"
  "convnext_tiny:random"
  "pvt_v2_b0:random"
  "pvt_v2_b1:random"
  "convnext_tiny:imagenet"
  "pvt_v2_b0:imagenet"
  "pvt_v2_b1:imagenet"
  "convnext_tiny:dinov3"
)

exec 9>/tmp/diva_rtdetr_matrix.lock
flock -n 9 || { echo "ERROR: another copy of this script is running" >&2; exit 1; }

echo "### preflight ###"
for spec in "${ARMS[@]}"; do
  bb="${spec%%:*}"; init="${spec##*:}"
  BACKBONE="$bb" INIT="$init" DATASET="${SUBSETS[0]}" "$PY" - <<'PYEOF' || exit 1
import importlib.util, sys
sys.argv = ["preflight"]
spec = importlib.util.spec_from_file_location("t", "train_rtdetr_hf.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
mdl = m.build_model({0: "TextLine"}, {"TextLine": 0})
n = sum(p.numel() for p in mdl.model.backbone.parameters()) / 1e6
print(f"  OK   {m.BACKBONE or 'stock R50-vd':16} {m.INIT:9} backbone={n:6.2f}M")
PYEOF
done
echo

echo "########## STAGE 1/2 -- RT-DETR detectors ##########"
for ds in "${SUBSETS[@]}"; do
  for spec in "${ARMS[@]}"; do
    bb="${spec%%:*}"; init="${spec##*:}"
    run="rtdetr_${bb:-stock}_${init}"
    if [ "$ds" = "CB55" ]; then
      dir="$REPO/80_models/02_2stage/diva-hisdb/detection/rtdetr_hf/$run"
    else
      dir="$REPO/80_models/02_2stage/diva-hisdb/detection/rtdetr_hf/$ds/$run"
    fi
    if [[ -f "$dir/.done" ]]; then echo "[skip] $ds/$run"; continue; fi
    [[ -f "$dir/metrics_summary.json" ]] && echo "[redo] $ds/$run incomplete"
    echo "######## $ds / $run ########"
    t0=$SECONDS
    DATASET="$ds" BACKBONE="$bb" INIT="$init" RUN_NAME="$run" "$PY" train_rtdetr_hf.py
    touch "$dir/.done"
    echo "[ok] $ds/$run in $(( (SECONDS - t0) / 60 )) min"
  done
done

echo
echo "########## STAGE 2/2 -- crop segmenters (1024x256, 3 losses) ##########"
for ds in "${SUBSETS[@]}"; do
  echo "######## segmenters: $ds ########"
  "$PY" train_crops_loss_ablation_2stage.py \
    --family diva --subset "$ds" --backbone resnet34 \
    --crop-width 1024 --crop-height 256
done

echo
echo "DIVA_MATRIX_FINISHED"
