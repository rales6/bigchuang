#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv-pi/bin/activate
# Share the inspection lock: do not launch another camera/motion session.
exec 9>/tmp/car1_map_camera_inspection.lock
flock -n 9 || { echo "已有巡检/联合建图任务运行，请先结束它。"; exit 1; }
mkdir -p inspection_logs
exec > >(tee -a "inspection_logs/joint_mapping_$(date +%Y%m%d_%H%M%S).log") 2>&1
exec python3 -u -m raspberry_pi.joint_mapping \
  --enable-motion --link ble --ble-address AA:BB:CC:DD:EE:01 \
  --lidar-port /dev/ttyACM0 --usable-fov 160 --mapping-max-distance 4 \
  --coarse-map maps/ble_scene_17.pgm \
  --stationary-map-distance 0.35 --stationary-map-angle 20 \
  --stationary-settle-scans 2 --stationary-fusion-scans 5 --stationary-min-support 3 \
  --front-left-gain 1.15 --front-right-gain 1.0 --rear-left-gain 0.90 --rear-right-gain 1.0 \
  --speed 250 --max-drive-speed 420 --slow-speed 220 \
  --turn-speed 1500 --max-turn-speed 2600 --turn-arc-speed 230 \
  --narrow-speed 240 --narrow-turn-speed 1700 \
  --stop-distance 0.40 --slow-distance 0.75 --clearance 0.30 \
  --max-runtime-min 15 --save-every 0 --live-map "$@"
