#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/gpu/track}"
PYTHON="${FINAL_WORK_GPU_PYTHON:-/home/gpu/3090/.venv/bin/python}"
cd "$ROOT"
exec "$PYTHON" -m final_work.gpu.qwen_hub \
  --existing-script /home/gpu/3090/scripts/run_qwen_landmark_server.py \
  --source-root /home/gpu/3090/src \
  --model-path /home/gpu/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --device cuda --pi-url http://192.0.2.11:8766 --host 0.0.0.0 --port 8010
