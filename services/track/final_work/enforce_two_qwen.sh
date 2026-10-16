#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/gpu/track}"
PYTHON="${FINAL_WORK_GPU_PYTHON:-/home/gpu/3090/.venv/bin/python}"
LOG_DIR="$ROOT/final_work/logs"
mkdir -p "$LOG_DIR"

# 8011 duplicates Qwen + YOLO capabilities that final_work obtains from the
# shared 8010 hub and 8021 pursuit service. Stop only that exact server.
STAGE1_PIDS="$(pgrep -f 'scripts/run_stage1_perception_server.py.*--port 8011' || true)"
for PID in $STAGE1_PIDS; do
  kill -TERM "$PID"
  echo "stopped duplicate stage1 Qwen process: pid=$PID"
done

# Restart 8021 so its old in-process Qwen allocation is released. The new
# server keeps YOLO/CSRT locally but sends grounding requests to 8010.
PURSUIT_PIDS="$(pgrep -f '/home/gpu/track/pursuit_server.py.*--port 8021' || true)"
for PID in $PURSUIT_PIDS; do kill -TERM "$PID"; done
for _ in $(seq 1 40); do
  ALIVE=""
  for PID in $PURSUIT_PIDS; do kill -0 "$PID" 2>/dev/null && ALIVE="$ALIVE $PID"; done
  [[ -z "$ALIVE" ]] && break
  sleep .2
done

cd "$ROOT"
nohup "$PYTHON" "$ROOT/pursuit_server.py" \
  --port 8021 --device cuda --no-depth \
  --qwen-hub-url http://127.0.0.1:8010 >>"$LOG_DIR/pursuit_shared_qwen.log" 2>&1 &
PURSUIT_PID="$!"
for _ in $(seq 1 60); do
  if curl -fsS --max-time 1 http://127.0.0.1:8021/health >/dev/null; then
    echo "pursuit restarted without local Qwen: pid=$PURSUIT_PID"
    echo "Qwen allocation policy: 8010 shared hub + 8020 grasp = at most two"
    exit 0
  fi
  kill -0 "$PURSUIT_PID" 2>/dev/null || break
  sleep .25
done
echo "8021 pursuit restart failed; see $LOG_DIR/pursuit_shared_qwen.log" >&2
exit 1

