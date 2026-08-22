#!/usr/bin/env bash
# Wait for the DIVA segmenter run to finish, then start the k-shot sweep.
#
# Two conditions, not one: the process has to be gone AND the card has to have given
# the memory back. Torch frees lazily, so starting the moment the PID disappears can
# still OOM the first detector.
set -uo pipefail
cd "$(dirname "$0")"
NEED_MIB=${NEED_MIB:-5500}

echo "[queue] $(date '+%H:%M:%S') waiting for the crop-segmenter run to finish"
while pgrep -f "train_crops_loss_ablation_2stage|run_diva_segmenters" >/dev/null; do
  sleep 60
done
echo "[queue] $(date '+%H:%M:%S') segmenters done; waiting for >=${NEED_MIB} MiB free"
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$(( total - used ))
  [ "$free" -ge "$NEED_MIB" ] && break
  echo "[queue] free=${free} MiB, waiting"
  sleep 60
done
echo "[queue] $(date '+%H:%M:%S') starting kshot_cb55.sh"
exec ./kshot_cb55.sh
