#!/usr/bin/env bash
# Restart the k-shot sweep if it dies.
#
# The segmenter runs died twice tonight when the owning session went away, losing
# hours of wall clock while the card sat idle. The sweep itself is resumable -- every
# step has a .done marker and kshot_cb55.sh holds an flock, so a restart picks up
# where it stopped and a double start is impossible. This just notices and restarts.
set -uo pipefail
cd "$(dirname "$0")"
REPO=$(cd .. && pwd)
LOG="$REPO/kshot_cb55.log"
MARK="$REPO/80_models/.kshot_markers"

while :; do
  if grep -q "KSHOT_CB55_FINISHED" "$LOG" 2>/dev/null; then
    echo "[watchdog] $(date '+%F %H:%M:%S') sweep finished; exiting"
    exit 0
  fi
  if ! pgrep -f "kshot_cb55.sh|queue_kshot.sh" >/dev/null; then
    n=$(ls "$MARK" 2>/dev/null | wc -l)
    echo "[watchdog] $(date '+%F %H:%M:%S') sweep not running at ${n}/21 steps -- restarting"
    setsid nohup ./kshot_cb55.sh >> "$LOG" 2>&1 < /dev/null &
    sleep 120
  fi
  sleep 300
done
