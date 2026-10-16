#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/gpu/track}"
PID_FILE="$ROOT/final_work/qwen_hub.pid"
LOG_FILE="$ROOT/final_work/logs/qwen_hub.log"
mkdir -p "$ROOT/final_work/logs"

if pgrep -f 'final_work.gpu.qwen_hub.*--port 8010' >/dev/null; then
  echo "Shared Qwen hub already running"
  exit 0
fi

# Retire only the exact legacy 8010 landmark process. Other Qwen, pursuit and
# grasp processes are intentionally untouched.
LEGACY_PIDS="$(pgrep -f 'scripts/run_qwen_landmark_server.py.*--port 8010' || true)"
for PID in $LEGACY_PIDS; do kill -TERM "$PID"; done
for _ in $(seq 1 30); do
  [[ -z "$LEGACY_PIDS" ]] && break
  ALIVE=""
  for PID in $LEGACY_PIDS; do kill -0 "$PID" 2>/dev/null && ALIVE="$ALIVE $PID"; done
  [[ -z "$ALIVE" ]] && break
  sleep .2
done

cd "$ROOT"
nohup bash final_work/start_qwen_hub.sh >>"$LOG_FILE" 2>&1 &
NEW_PID="$!"
echo "$NEW_PID" > "$PID_FILE"
for _ in $(seq 1 40); do
  if curl -fsS --max-time 1 http://127.0.0.1:8010/agent/health >/dev/null; then
    echo "Shared Qwen hub started: pid=$NEW_PID log=$LOG_FILE"
    exit 0
  fi
  kill -0 "$NEW_PID" 2>/dev/null || break
  sleep .25
done

echo "Shared hub failed; restoring the legacy 8010 service" >&2
kill -TERM "$NEW_PID" 2>/dev/null || true
rm -f "$PID_FILE"
cd /home/gpu/3090
nohup /home/gpu/3090/.venv/bin/python scripts/run_qwen_landmark_server.py \
  --model-path /home/gpu/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --host 192.0.2.20 --port 8010 --device cuda \
  --pi-url http://192.0.2.11:8766 >>"$ROOT/final_work/logs/qwen_rollback.log" 2>&1 &
exit 1

