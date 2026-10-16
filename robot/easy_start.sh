#!/usr/bin/env bash
set -euo pipefail

cd ~/dachuang/car1.0
source .venv-pi/bin/activate
python3 -m compileall -q raspberry_pi

exec python3 -m raspberry_pi.autonomous_mapping \
  --enable-motion \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --ble-address AA:BB:CC:DD:EE:01 \
  --lidar-port /dev/ttyACM0 \
  --usable-fov 160 \
  --mapping-max-distance 4.0 \
  --wall-gap-max 0 \
  --stationary-map-distance 0.35 \
  --stationary-map-angle 20 \
  --stationary-settle-scans 2 \
  --stationary-fusion-scans 5 \
  --stationary-min-support 3 \
  --stationary-range-tolerance 0.10 \
  --min-obstacle-area 0.01 \
  --front-left-gain 1.15 \
  --front-right-gain 1.00 \
  --rear-left-gain 0.90 \
  --rear-right-gain 1.00 \
  --speed 300 \
  --max-drive-speed 420 \
  --slow-speed 220 \
  --measured-speed-min 0.075 \
  --measured-speed-max 0.22 \
  --measured-turn-min 0.12 \
  --measured-turn-max 0.70 \
  --turn-speed 1500 \
  --max-turn-speed 2600 \
  --turn-arc-speed 230 \
  --narrow-passage-width 0.95 \
  --narrow-speed 240 \
  --narrow-turn-speed 1700 \
  --link-reconnect-rounds 0 \
  --max-rejected-scans 8 \
  --stop-distance 0.40 \
  --slow-distance 0.75 \
  --clearance 0.30 \
  --map-size 20 \
  --max-runtime-min 15 \
  --save-every 0 \
  --live-map \
  --output maps/ble_scene_17
