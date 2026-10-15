#!/usr/bin/env bash
set -euo pipefail
PIDS="$(pgrep -f 'final_work.gpu.qwen_hub.*--port 8010' || true)"
for PID in $PIDS; do kill -TERM "$PID"; done
for _ in $(seq 1 40); do
  ALIVE=""
  for PID in $PIDS; do kill -0 "$PID" 2>/dev/null && ALIVE="$ALIVE $PID"; done
  [[ -z "$ALIVE" ]] && break
  sleep .2
done
rm -f "${FINAL_WORK_TRACK_ROOT:-/home/sjtu/track}/final_work/qwen_hub.pid"
exec bash "${FINAL_WORK_TRACK_ROOT:-/home/sjtu/track}/final_work/run_qwen_hub_background.sh"
