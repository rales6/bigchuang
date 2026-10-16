#!/usr/bin/env bash
set -euo pipefail

release_dir="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
car_root="${CAR_ROOT:-/home/robot/dachuang/car1.0}"
cd "$car_root"
source "$car_root/.venv-pi/bin/activate"

instruction="${1:?请提供自然语言指令}"
decision="$car_root/search_runs/latest_decision.json"
gpu_url="${SEARCH_3090_URL:-http://192.0.2.20:8011}"

python3 -u -m raspberry_pi.submit_search_command \
  "$instruction" \
  --3090-url "$gpu_url" \
  --output "$decision"

python3 - "$decision" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
target = decision.get("target") or {}
route = decision.get("route_request") or {}
if decision.get("status") != "executable":
    raise SystemExit("任务不是 executable，拒绝启动实车")
if not target.get("class_name"):
    raise SystemExit("任务目标类别为空，拒绝启动实车")
if route.get("mode") not in {"landmark", "region", "semantic_search_region"}:
    raise SystemExit("不支持的路线模式：{}".format(route.get("mode")))
if route.get("mode") == "semantic_search_region":
    center = route.get("center_xy_m")
    if not isinstance(center, list) or len(center) != 2:
        raise SystemExit("语义搜索区域缺少 center_xy_m，拒绝启动实车")
    if route.get("require_live_visual_recheck") is not True:
        raise SystemExit("语义搜索区域未要求现场视觉复核，拒绝启动实车")
print(
    "实车任务确认：class={} mode={} anchor={} visual_recheck={}".format(
        target.get("class_name"),
        route.get("mode"),
        (decision.get("anchor") or {}).get("name"),
        route.get("require_live_visual_recheck"),
    ),
    flush=True,
)
PY

exec 9>/tmp/car1_relationship_object_search.lock
if ! flock -n 9; then
  echo "另一个新版寻物任务正在占用底盘、雷达或摄像头。" >&2
  exit 2
fi

exec python3 -u "$release_dir/relationship_object_search_entry.py" \
  --enable-motion \
  --decision "$decision" \
  --map "$release_dir/ble_scene_17.pgm" \
  --map-yaml "$release_dir/ble_scene_17.yaml" \
  --output "$car_root/search_runs/ble_scene_17" \
  --3090-url "$gpu_url" \
  --link ble \
  --ble-name ESP32-Robot-Car \
  --ble-address AA:BB:CC:DD:EE:01 \
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
  --visual-localization-semantic-map "$release_dir/semantic_localization_landmarks.json" \
  --max-runtime-min 15
