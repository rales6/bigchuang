#!/usr/bin/env bash
set -euo pipefail

TRACK_ROOT="${FINAL_WORK_TRACK_ROOT:-/home/robot/dachuang/track}"
PYTHON="${FINAL_WORK_PI_PYTHON:-/home/robot/dachuang/.venv/bin/python}"
PORT="${FINAL_WORK_PI_PORT:-8780}"
LOG_DIR="${FINAL_WORK_PI_LOG_DIR:-${TRACK_ROOT}/final_work/logs}"
DATA_DIR="${FINAL_WORK_DATA_DIR:-${TRACK_ROOT}/final_work/data}"
if [[ -z "${FINAL_WORK_ENABLE_HARDWARE+x}" && -r "$DATA_DIR/hardware_enabled" ]]; then
  FINAL_WORK_ENABLE_HARDWARE="$(tr -cd '01' < "$DATA_DIR/hardware_enabled" | head -c 1)"
  export FINAL_WORK_ENABLE_HARDWARE
fi
if [[ -z "${FINAL_WORK_HARDWARE_TOKEN:-}" && -r "$DATA_DIR/hardware_token" ]]; then
  FINAL_WORK_HARDWARE_TOKEN="$(tr -d '\r\n' < "$DATA_DIR/hardware_token")"
  export FINAL_WORK_HARDWARE_TOKEN
fi
mkdir -p "$LOG_DIR"
cd "$TRACK_ROOT"

echo "final_work Pi bridge"
echo "URL: http://0.0.0.0:${PORT}"
echo "hardware: ${FINAL_WORK_ENABLE_HARDWARE:-0} (0=safe, 1=real)"
echo "logs: ${LOG_DIR}"

exec "$PYTHON" -m final_work.pi.robot_bridge --host 0.0.0.0 --port "$PORT"
