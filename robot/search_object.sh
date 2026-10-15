#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source .venv-pi/bin/activate
python3 -m compileall -q raspberry_pi

decision="${1:-maps/search_decision_cabinet_red_bottle.json}"
mode="${2:-simulate}"

if [[ "$mode" == "simulate" ]]; then
  exec python3 -u -m raspberry_pi.object_search_mission \
    --simulate \
    --decision "$decision" \
    --map maps/ble_scene_17.pgm \
    --map-yaml maps/ble_scene_17.yaml \
    --output search_runs/ble_scene_17
fi

if [[ "$mode" != "real" ]]; then
  echo "用法：./search_object.sh [decision.json] simulate|real" >&2
  exit 2
fi

exec 9>/tmp/car1_object_search.lock
if ! flock -n 9; then
  echo "另一个寻物任务正在占用底盘、雷达或摄像头。" >&2
  exit 2
fi

exec python3 -u -m raspberry_pi.object_search_mission \
  --enable-motion \
  --decision "$decision" \
  --map maps/ble_scene_17.pgm \
  --map-yaml maps/ble_scene_17.yaml \
  --output search_runs/ble_scene_17 \
  --3090-url "${SEARCH_3090_URL:-http://192.168.55.33:8011}" \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --ble-address 64:B7:08:69:9A:E6 \
  --lidar-port /dev/ttyACM0 \
  --speed 240 \
  --slow-speed 230 \
  --max-drive-speed 460 \
  --drive-speed-step 60 \
  --start-kick-speed 400 \
  --turn-speed 1250 \
  --max-turn-speed 2600 \
  --relocalization-speed 300 \
  --relocalization-max-speed 460 \
  --relocalization-turn-speed 1500 \
  --relocalization-max-turn-speed 2600 \
  --relocalization-turn-step 300 \
  --localization-required-moves 1 \
  --stop-distance 0.45 \
  --clearance 0.30 \
  --camera-backend opencv \
  --dual-usb-cameras \
  --camera-width 1280 \
  --camera-height 720 \
  --real-test-max-distance "${SEARCH_TEST_MAX_DISTANCE:-0}" \
  --max-runtime-min 15
