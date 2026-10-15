#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/sjtu/track}"
PYTHON="${FINAL_WORK_GPU_PYTHON:-/home/sjtu/3090/.venv/bin/python}"
cd "$ROOT"
exec "$PYTHON" -m final_work.gpu.qwen_hub \
  --existing-script /home/sjtu/3090/scripts/run_qwen_landmark_server.py \
  --source-root /home/sjtu/3090/src \
  --model-path /home/sjtu/3090/models/qwen/Qwen3-VL-2B-Instruct \
  --device cuda --pi-url http://172.18.14.74:8766 --host 0.0.0.0 --port 8010
