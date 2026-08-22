#!/usr/bin/env bash
# ============================================================================
# RQ1 / 02 -- U-DIADS-TL two-stage: RT-DETR backbone matrix + crop segmenters.
#
# STAGE 1 -- detectors, RT-DETR only, imgsz 1152, 3 subsets.
#
#   bracket   CNN                      ViT hierarchical
#   XS        convnext_femto  4.83M    pvt_v2_b0   3.41M
#   S         convnext_pico   8.53M    pvt_v2_b1  13.50M
#   M         convnext_tiny  27.82M    pvt_v2_b2  24.85M  <- NOT RUN HERE, see below
#
#   arms: random / imagenet / dinov3(SSL, bracket M CNN only)
#   plus stock R50-vd 23.47M as the out-of-matrix baseline.
#
# WHAT IS DELIBERATELY MISSING, and why -- both measured, not assumed:
#
#   pvt_v2_b2 (all arms). OOMs at 1152 on this 8GB card; measured peaks at 1152
#     were femto 1.53 / pico 1.73 / tiny 2.68 / b0 3.98 / b1 4.69 / R50-vd 2.71 GiB
#     against a 6.1 GiB budget, and b2 does not fit. Run it on Colab.
#
#   convnext_femto + convnext_pico, imagenet arm. HF publishes no ConvNeXt
#     checkpoint below tiny -- facebook/convnext-{femto,pico}-224 do not exist
#     (checked). Those two are random-only here. timm has the weights, but moving
#     them into an HF ConvNextBackbone needs a key remap, which is its own task.
#
#   flat ViT column. HF rejects ViTConfig as an RT-DETR backbone outright
#     ("Unrecognized configuration class ... for this kind of AutoModel:
#     AutoBackbone") -- a flat ViT yields three maps all at stride 16 and RT-DETR
#     wants a pyramid. Needs an SFP adapter; excluded on request.
#
# Recipe = the one fixed for CB55: constant LR (cosine annealed to ~0 and stalled
# training), EMA on, grad_accum 2. Note U-DIADS has only 3 train pages per subset,
# so accum=16 never filled within an epoch -- 1 optimizer step per epoch, 100 per
# run. accum=2 roughly doubles that.
#
# STAGE 2 -- crop segmenters, 1024x256, resnet34, all three losses
# (bce / tversky / supervoxel; the trainer iterates them itself), per subset.
#
# Checkpoints -> 80_models/02_2stage/u-diads-tl/detection/rtdetr_hf/<subset>/<run>/
#                80_models/02_2stage/u-diads-tl/crop_seg_.../<subset>/<loss>/
# Metrics     -> 99_evaluation/02_2stage/u-diads-tl/rtdetr_hf/<subset>/results.csv
#
# Runtime: 30 detector runs, ~8 min each -> ~4 h. Segmenters add ~1-2 h.
# Safe to re-run: finished runs are skipped via their .done marker.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"
PY=../../.venv/bin/python
REPO=$(cd ../.. && pwd)

export IMAGE_SIZE=1152
SUBSETS=(Latin14396 Latin2 Syr341)

# "<backbone>:<init>" -- empty backbone = stock R50-vd baseline.
ARMS=(
  ":random"                 # stock R50-vd, out-of-matrix baseline
  "convnext_femto:random"
  "convnext_pico:random"
  "convnext_tiny:random"
  "pvt_v2_b0:random"
  "pvt_v2_b1:random"
  "convnext_tiny:imagenet"
  "pvt_v2_b0:imagenet"
  "pvt_v2_b1:imagenet"
  "convnext_tiny:dinov3"    # SSL arm, bracket M CNN
)

# Single instance: two copies would write the same run dirs and the loser would
# skip runs it never trained.
exec 9>/tmp/udiads_rtdetr_matrix.lock
flock -n 9 || { echo "ERROR: another copy of this script is running" >&2; exit 1; }

echo "### preflight: building every arm once ###"
for spec in "${ARMS[@]}"; do
  bb="${spec%%:*}"; init="${spec##*:}"
  BACKBONE="$bb" INIT="$init" DATASET=Latin14396 "$PY" - <<'PYEOF' || exit 1
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
for ms in "${SUBSETS[@]}"; do
  for spec in "${ARMS[@]}"; do
    bb="${spec%%:*}"; init="${spec##*:}"
    run="rtdetr_${bb:-stock}_${init}"
    dir="$REPO/80_models/02_2stage/u-diads-tl/detection/rtdetr_hf/$ms/$run"
    if [[ -f "$dir/.done" ]]; then
      echo "[skip] $ms/$run"
      continue
    fi
    [[ -f "$dir/metrics_summary.json" ]] && echo "[redo] $ms/$run has output but no .done -> incomplete"
    echo "######## $ms / $run ########"
    t0=$SECONDS
    DATASET="$ms" BACKBONE="$bb" INIT="$init" RUN_NAME="$run" "$PY" train_rtdetr_hf.py
    touch "$dir/.done"
    echo "[ok] $ms/$run in $(( (SECONDS - t0) / 60 )) min"
  done
done

echo
echo "########## STAGE 2/2 -- crop segmenters (1024x256, 3 losses) ##########"
for ms in "${SUBSETS[@]}"; do
  echo "######## segmenters: $ms ########"
  "$PY" train_crops_loss_ablation_2stage.py --family udiads \
    --subset "$ms" --backbone resnet34 \
    --crop-width 1024 --crop-height 256
done

echo
echo "ALL DONE -- detectors: 99_evaluation/02_2stage/u-diads-tl/rtdetr_hf/<subset>/results.csv"
echo "pvt_v2_b2 still owed (Colab): 3 subsets x {random, imagenet} = 6 runs"
