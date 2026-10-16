#!/usr/bin/env bash
# Physical-left/right and the audited taught model use USB video2/video0.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
TASK_VENV="${HOME}/dachuang/.venv"
if [ ! -x "$TASK_VENV/bin/python" ]; then
    echo "Existing virtual environment unavailable: $TASK_VENV"
    exit 1
fi
export GRAB_LOG_DIR="${GRAB_LOG_DIR:-${SCRIPT_DIR}/logs}"
# Operator observed the old -20 us correction raised the tip: reverse it.
# Pin these values so inherited old service environment cannot undo this fix.
# S3 mirrors S2 in firmware. No lateral correction is applied anymore.
export ARM_GRASP_S2_NUDGE_US="20"
export ARM_GRASP_S0_NUDGE_US="0"
mkdir -p "$GRAB_LOG_DIR"
TASK_MODEL_PAIR="$("$TASK_VENV/bin/python" -c 'from camera_contract import model_devices; print(*model_devices(require_ports=True))')"
read -r TASK_LEFT_CAM TASK_RIGHT_CAM <<< "$TASK_MODEL_PAIR"
TASK_WEB_LAUNCH=("$TASK_VENV/bin/python")
TASK_TRACK_ROOT="${FINAL_WORK_TRACK_ROOT:-/home/robot/dachuang/track}"
if [ -f "$TASK_TRACK_ROOT/final_work/pi/preview_launcher.py" ]; then
    export FINAL_WORK_TRACK_ROOT="$TASK_TRACK_ROOT"
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${TASK_TRACK_ROOT}"
    TASK_WEB_LAUNCH+=(-m final_work.pi.preview_launcher)
fi
exec "${TASK_WEB_LAUNCH[@]}" "$SCRIPT_DIR/web_app.py" \
    --qwen-url "${QWEN_URL:-http://192.0.2.20:8010}" \
    --left-camera "$TASK_LEFT_CAM" --right-camera "$TASK_RIGHT_CAM" --host 0.0.0.0 --port 8080
