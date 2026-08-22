#!/usr/bin/env bash
# RT-DETR (HF) backbone ablation on CB55: stock R50-vd / ConvNeXt-tiny / PvtV2-b0.
#
# One resolution for all arms -- PvtV2 OOMs at 1408 on an 8GB card, so the matrix runs
# at 1152 and the stock arm is retrained there too, otherwise the rows are not
# comparable. Everything except the backbone is whatever train_rtdetr_hf.py already
# does (recipe, seed, epochs, schedule, EMA, augmentation).
#
# Checkpoints -> 80_models/02_2stage/diva-hisdb/detection/rtdetr_hf/<run>/
# Metrics     -> 99_evaluation/02_2stage/diva-hisdb/rtdetr_hf/results.csv
#
# Score a finished arm with:
#   RUN_NAME=rtdetr_bb_pvt_v2 python predict_and_eval_rtdetr_diva.py
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd ../.. && pwd)"
# shellcheck disable=SC1091
[ -f "$REPO_ROOT/.venv/bin/activate" ] && source "$REPO_ROOT/.venv/bin/activate"

export IMAGE_SIZE=1152
LOGDIR="$REPO_ROOT/99_evaluation/02_2stage/diva-hisdb/rtdetr_hf"
mkdir -p "$LOGDIR"

run () {  # $1 = BACKBONE ("" = stock r50vd), $2 = run name
  echo "=============== $2 (backbone='${1:-stock}') ==============="
  BACKBONE="$1" RUN_NAME="$2" python3 train_rtdetr_hf.py > "$LOGDIR/${2}_train.log" 2>&1
  python3 - "$2" <<'PY'
import json, sys, pathlib
p = pathlib.Path("../../80_models/02_2stage/diva-hisdb/detection/rtdetr_hf") / sys.argv[1] / "metrics_summary.json"
s = json.load(open(p))["summary"]
print({k: s[k] for k in ("backbone", "params_backbone", "validation_mAP@50",
                         "test_mAP@50", "test_mAP@50-95", "test_mAR@100")})
PY
}

run ""         rtdetr_bb_stock
run convnext   rtdetr_bb_convnext
run pvt_v2     rtdetr_bb_pvt_v2

echo "=============== ALL DONE ==============="
