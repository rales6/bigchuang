#!/usr/bin/env bash
# Preserve the frozen grab_visual startup script while restoring the camera
# order used for the 30-sample arm-teach dataset.
set -euo pipefail

ARM_WEB_ROOT="${ARM_WEB_ROOT:-/home/dachuang/grab_visual}"
export LEFT_CAM="${LEFT_CAM:-2}"
export RIGHT_CAM="${RIGHT_CAM:-0}"

echo "arm-teach camera contract: LEFT_CAM=${LEFT_CAM}, RIGHT_CAM=${RIGHT_CAM}"
exec bash "${ARM_WEB_ROOT}/start_web.sh"
