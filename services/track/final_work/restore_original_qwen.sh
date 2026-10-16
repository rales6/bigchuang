#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/gpu/track}"
PIDS="$(pgrep -f 'final_work.gpu.qwen_hub.*--port 8010' || true)"
for PID in $PIDS; do kill -TERM "$PID"; done
sleep 1
cd /home/gpu/3090
mkdir -p "$ROOT/final_work/logs"
nohup /home/gpu/3090/.venv/bin/python scripts/run_qwen_landmark_server.py \
  --model-path /home/gpu/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --host 192.0.2.20 --port 8010 --device cuda \
  --pi-url http://192.0.2.11:8766 >>"$ROOT/final_work/logs/qwen_original.log" 2>&1 &
echo "Original Qwen landmark service restored on port 8010: pid=$!"

