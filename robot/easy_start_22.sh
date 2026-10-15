#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f .venv-pi/bin/activate ]]; then
  source .venv-pi/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "No Python venv found. Expected .venv-pi or .venv." >&2
  exit 2
fi

OUTPUT="${OUTPUT:-maps/original_map}"
LIDAR_PORT="${LIDAR_PORT:-/dev/ttyACM0}"
BLE_NAME="${BLE_NAME:-ESP32-Robot-Car}"
BLE_ADDRESS="${BLE_ADDRESS:-64:B7:08:69:9A:E6}"

echo "Autonomous mapping with guarded global localization"
echo "Map output: $OUTPUT"
echo "Lidar: $LIDAR_PORT"
echo "BLE: $BLE_NAME ($BLE_ADDRESS)"
echo "The car will move. Keep an operator next to the emergency stop."

exec python3 -m raspberry_pi.autonomous_mapping \
  --enable-motion \
  --link ble \
  --ble-name "$BLE_NAME" \
  --ble-address "$BLE_ADDRESS" \
  --lidar-port "$LIDAR_PORT" \
  --usable-fov "${USABLE_FOV:-160}" \
  --mapping-max-distance "${MAPPING_MAX_DISTANCE:-6.0}" \
  --wall-gap-max "${WALL_GAP_MAX:-0}" \
  --stationary-map-distance "${STATIONARY_MAP_DISTANCE:-0.40}" \
  --stationary-map-angle "${STATIONARY_MAP_ANGLE:-30}" \
  --stationary-settle-scans "${STATIONARY_SETTLE_SCANS:-2}" \
  --stationary-fusion-scans "${STATIONARY_FUSION_SCANS:-5}" \
  --stationary-min-support "${STATIONARY_MIN_SUPPORT:-3}" \
  --stationary-range-tolerance "${STATIONARY_RANGE_TOLERANCE:-0.10}" \
  --min-obstacle-area "${MIN_OBSTACLE_AREA:-0.01}" \
  --resolution "${MAP_RESOLUTION:-0.05}" \
  --map-size "${MAP_SIZE:-20}" \
  --front-left-gain "${FRONT_LEFT_GAIN:-1.15}" \
  --front-right-gain "${FRONT_RIGHT_GAIN:-1.00}" \
  --rear-left-gain "${REAR_LEFT_GAIN:-0.90}" \
  --rear-right-gain "${REAR_RIGHT_GAIN:-1.00}" \
  --speed "${DRIVE_SPEED:-300}" \
  --max-drive-speed "${MAX_DRIVE_SPEED:-360}" \
  --slow-speed "${SLOW_SPEED:-220}" \
  --measured-speed-min "${MEASURED_SPEED_MIN:-0.075}" \
  --measured-speed-max "${MEASURED_SPEED_MAX:-0.22}" \
  --turn-speed "${TURN_SPEED:-1500}" \
  --max-turn-speed "${MAX_TURN_SPEED:-3000}" \
  --turn-arc-speed "${TURN_ARC_SPEED:-260}" \
  --narrow-passage-width "${NARROW_PASSAGE_WIDTH:-0.95}" \
  --narrow-speed "${NARROW_SPEED:-240}" \
  --narrow-turn-speed "${NARROW_TURN_SPEED:-1700}" \
  --stop-distance "${STOP_DISTANCE:-0.40}" \
  --slow-distance "${SLOW_DISTANCE:-0.75}" \
  --clearance "${CLEARANCE:-0.30}" \
  --link-reconnect-rounds "${LINK_RECONNECT_ROUNDS:-0}" \
  --command-ttl-ms "${COMMAND_TTL_MS:-700}" \
  --max-runtime-min "${MAX_RUNTIME_MIN:-15}" \
  --save-every "${SAVE_EVERY:-0}" \
  --live-map \
  --output "$OUTPUT" \
  "$@"
