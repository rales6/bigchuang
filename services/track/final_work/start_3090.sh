#!/usr/bin/env bash
set -euo pipefail

TRACK_ROOT="${FINAL_WORK_TRACK_ROOT:-/home/sjtu/track}"
PYTHON="${FINAL_WORK_GPU_PYTHON:-/home/sjtu/3090/.venv/bin/python}"
PORT="${FINAL_WORK_WEB_PORT:-8030}"
DATA_DIR="${FINAL_WORK_DATA_DIR:-${TRACK_ROOT}/final_work/data}"
mkdir -p "$DATA_DIR" "${TRACK_ROOT}/final_work/logs"
cd "$TRACK_ROOT"

export FINAL_WORK_DATA_DIR="$DATA_DIR"
export FINAL_WORK_PI_URL="${FINAL_WORK_PI_URL:-http://127.0.0.1:18780}"
export FINAL_WORK_PLANNER_BACKEND="${FINAL_WORK_PLANNER_BACKEND:-shared_http}"
export FINAL_WORK_QWEN_HUB_URL="${FINAL_WORK_QWEN_HUB_URL:-http://127.0.0.1:8010}"
export FINAL_WORK_QWEN_MODEL="${FINAL_WORK_QWEN_MODEL:-/home/sjtu/3090/models/qwen/Qwen3-VL-2B-Instruct}"
export FINAL_WORK_3090_SRC="${FINAL_WORK_3090_SRC:-/home/sjtu/3090/src}"
if [[ -z "${FINAL_WORK_HARDWARE_TOKEN:-}" && -r "$DATA_DIR/hardware_token" ]]; then
  export FINAL_WORK_HARDWARE_TOKEN="$(tr -d '\r\n' < "$DATA_DIR/hardware_token")"
fi

echo "final_work 3090 Agent"
echo "Web: http://0.0.0.0:${PORT}"
echo "Pi: ${FINAL_WORK_PI_URL}"
echo "history: ${FINAL_WORK_DATA_DIR}/history.sqlite3"
echo "planner: ${FINAL_WORK_PLANNER_BACKEND} (${FINAL_WORK_QWEN_MODEL})"

exec "$PYTHON" -m final_work.gpu.app --host 0.0.0.0 --port "$PORT"
