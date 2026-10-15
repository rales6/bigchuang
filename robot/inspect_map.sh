#!/usr/bin/env bash
set -euo pipefail

cd ~/dachuang/car1.0
source .venv-pi/bin/activate
python3 -m compileall -q raspberry_pi

# One inspection process owns BLE, lidar and both cameras. Refuse a second
# launch instead of letting it fail later with misleading device errors.
exec 9>/tmp/car1_map_camera_inspection.lock
if ! flock -n 9; then
  echo "巡检程序已在运行；请先停止旧任务，不要同时占用蓝牙、雷达和摄像头。" >&2
  pgrep -af 'raspberry_pi.map_camera_inspection' >&2 || true
  exit 2
fi

export PYTHONUNBUFFERED=1
mkdir -p inspection_logs
INSPECTION_LOG="inspection_logs/inspect_map_$(date +%Y%m%d_%H%M%S).log"
echo "本次完整运行日志：$INSPECTION_LOG"

python3 -u -m raspberry_pi.map_camera_inspection \
  --enable-motion \
  --map maps/ble_scene_17.pgm \
  --map-yaml maps/ble_scene_17.yaml \
  --semantic-map maps/ble_scene_17_semantics_qwen.json \
  --ignore-verified-semantic-views \
  --skip-classified-semantic-components \
  --online-qwen-url http://192.168.55.33:8010/landmarks \
  --online-qwen-timeout-seconds 90 \
  --online-qwen-min-confidence 0.80 \
  --online-qwen-confirmations 2 \
  --online-qwen-max-component-attempts 3 \
  --defer-online-qwen \
  --open-space-panorama \
  --panorama-min-station-spacing 1.00 \
  --panorama-ideal-station-spacing 1.60 \
  --panorama-max-station-spacing 2.50 \
  --panorama-min-stations 3 \
  --panorama-max-stations 16 \
  --panorama-angle-step-deg 60 \
  --panorama-camera-fov-deg 82 \
  --panorama-camera-range 4.0 \
  --panorama-coverage-target 0.90 \
  --panorama-open-min-clearance 0.48 \
  --panorama-open-direction-clearance 0.72 \
  --panorama-open-required-directions 3 \
  --panorama-map-correction-distance 1.20 \
  --panorama-map-correction-gain 0.20 \
  --panorama-map-correction-confirmations 2 \
  --panorama-map-correction-radius 0.65 \
  --panorama-drive-speed 380 \
  --panorama-slow-speed 240 \
  --panorama-turn-speed 2600 \
  --panorama-turn-max-speed 3000 \
  --panorama-status-interval-seconds 0.20 \
  --panorama-startup-window-seconds "${PANORAMA_STARTUP_WINDOW_S:-0.70}" \
  --panorama-progress-window-seconds "${PANORAMA_PROGRESS_WINDOW_S:-1.80}" \
  --panorama-max-scan-age-seconds "${PANORAMA_MAX_SCAN_AGE_S:-0.50}" \
  --panorama-self-return-radius "${PANORAMA_SELF_RETURN_RADIUS_M:-0.19}" \
  --panorama-self-return-half-width "${PANORAMA_SELF_RETURN_HALF_WIDTH_M:-0.125}" \
  --panorama-body-length "${PANORAMA_BODY_LENGTH_M:-0.24}" \
  --panorama-body-width "${PANORAMA_BODY_WIDTH_M:-0.22}" \
  --panorama-footprint-margin "${PANORAMA_FOOTPRINT_MARGIN_M:-0.05}" \
  --information-view-distance 2.20 \
  --output inspections/ble_scene_17 \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --ble-address 64:B7:08:69:9A:E6 \
  --lidar-port /dev/ttyACM0 \
  --lidar-offset-x 0.14 \
  --lidar-offset-y 0.0 \
  --clearance "${PANORAMA_MAP_CLEARANCE_M:-0.22}" \
  --front-left-gain 1.15 \
  --front-right-gain 1.00 \
  --rear-left-gain 0.90 \
  --rear-right-gain 1.00 \
  --speed 330 \
  --max-drive-speed 450 \
  --drive-speed-step 30 \
  --start-kick-speed 420 \
  --slow-speed 240 \
  --navigation-fine-control-distance 0.60 \
  --navigation-fine-turn-speed 3000 \
  --navigation-fine-turn-min-speed 3000 \
  --navigation-precision-turn-ttl-ms 180 \
  --navigation-heading-seed-frames 5 \
  --turn-speed 2400 \
  --command-min-interval-ms 450 \
  --stop-distance 0.42 \
  --emergency-front-distance 0.34 \
  --collision-reverse-distance 0.06 \
  --collision-reverse-speed 240 \
  --collision-reverse-rear-clearance 0.25 \
  --collision-reverse-max-steps 2 \
  --collision-reverse-map-fallback \
  --collision-reverse-map-fallback-max-distance 0.08 \
  --waypoint-tolerance 0.18 \
  --heading-tolerance-deg 7 \
  --localization-timeout-seconds 120 \
  --localization-window 5 \
  --localization-min-accepted 4 \
  --localization-acceptance-ratio 0.80 \
  --max-localization-failures 5 \
  --relaxed-tracking-rmse 0.28 \
  --relaxed-tracking-inlier-ratio 0.28 \
  --localization-min-margin 0.060 \
  --localization-max-rmse 0.12 \
  --localization-min-inlier-ratio 0.60 \
  --localization-min-range-agreement 0.55 \
  --localization-max-range-error 0.24 \
  --localization-min-range-coverage 0.40 \
  --localization-fast-accept-margin 0.15 \
  --localization-fast-min-inlier-ratio 0.70 \
  --localization-fast-min-range-agreement 0.75 \
  --localization-fast-max-range-error 0.20 \
  --localization-disambiguation-distance 0.26 \
  --relocalization-distance 0.12 \
  --relocalization-speed 220 \
  --relocalization-max-speed 380 \
  --relocalization-speed-step 40 \
  --relocalization-turn-speed 1300 \
  --relocalization-max-turn-speed 3000 \
  --relocalization-turn-step 200 \
  --relocalization-turn-stall-at-max-frames 2 \
  --relocalization-turn-escape-distance 0.05 \
  --relocalization-turn-escape-max-steps 4 \
  --relocalization-turn-escape-max-attempts 2 \
  --relocalization-min-scan-change 0.025 \
  --localization-required-moves 1 \
  --localization-motion-min-distance 0.07 \
  --localization-motion-position-tolerance 0.24 \
  --localization-motion-yaw-tolerance-deg 16 \
  --ignore-trajectory-pose \
  --temporary-obstacle-radius 0.12 \
  --temporary-obstacle-confirm-distance 0.18 \
  --temporary-obstacle-min-hits 2 \
  --standoff 1.00 \
  --max-targets 24 \
  --views-per-wall 3 \
  --views-per-region 1 \
  --min-target-spacing 0.40 \
  --target-turn-cost-m-per-rad 0.35 \
  --target-retry-min-displacement 0.25 \
  --max-target-navigation-attempts 4 \
  --nearby-capture-distance 0.35 \
  --navigation-turn-stall-at-max-frames 4 \
  --navigation-linear-stall-at-max-frames 5 \
  --navigation-pure-turn-timeout-seconds 8 \
  --navigation-timeout-capture-max-heading-error-deg 45 \
  --navigation-turn-refresh-window-seconds 1.0 \
  --navigation-turn-escape-max-attempts 1 \
  --navigation-pose-stationary-yaw-limit-deg 18 \
  --navigation-pose-stationary-translation-limit 0.15 \
  --navigation-pose-direction-slack-deg 12 \
  --navigation-pose-max-correction-holds 3 \
  --navigation-map-correction-interval-seconds 6 \
  --navigation-map-correction-distance 0.50 \
  --navigation-map-correction-gain 0.25 \
  --navigation-recovery-min-inlier-ratio 0.42 \
  --navigation-recovery-max-rmse 0.12 \
  --navigation-recovery-confirmation-frames 2 \
  --alternate-view-max-shift 0.85 \
  --settle-seconds 1.0 \
  --camera-backend opencv \
  --dual-usb-cameras \
  --free-cameras \
  --camera-device 0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-yaw-offset-deg 0 \
  --max-runtime-min 60 2>&1 | tee "$INSPECTION_LOG"
