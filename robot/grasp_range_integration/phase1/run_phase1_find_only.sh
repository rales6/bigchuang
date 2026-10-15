#!/usr/bin/env bash
set -euo pipefail

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$package_root/search_object.sh" && -d "$package_root/.venv-pi" ]]; then
  car_root="$package_root"
  release_dir="$car_root/maps/semantic_release_20260919"
  python_root="$car_root"
else
  release_dir="$package_root"
  car_root="${CAR_ROOT:-$(dirname "$release_dir")}" 
  python_root="$release_dir"
fi
cd "$car_root"
source "$car_root/.venv-pi/bin/activate"
export PYTHONPATH="$python_root:$car_root${PYTHONPATH:+:$PYTHONPATH}"
# Audited 2026-09-20/25 staging area. An explicitly empty value restores the
# generic global/active localization path when the car is placed elsewhere.
export SEARCH_START_POSE_HINT="${SEARCH_START_POSE_HINT--1.55,-2.60,100,0.70,40}"
export PHASE1_START_SEARCH_BUDGET_SECONDS="${PHASE1_START_SEARCH_BUDGET_SECONDS:-20}"
export PHASE1_START_VERIFY_BUDGET_SECONDS="${PHASE1_START_VERIFY_BUDGET_SECONDS:-2}"

decision="${1:-search_runs/phase1/latest_deepseek_decision.json}"
confirmation="${2:-}"

if [[ "$confirmation" != "--execute-find-only" ]]; then
  echo "未启动：实车第一阶段必须显式添加 --execute-find-only。" >&2
  echo "用法：bash grasp_range_integration/phase1/run_phase1_find_only.sh DECISION.json --execute-find-only" >&2
  exit 2
fi

python3 -u -m grasp_range_integration.phase1.validate_decision "$decision"

# Collision incident: no hardware/previews are touched until the continuous
# glass boundary has been verified against this exact PGM + coordinate YAML.
export PHASE1_GLASS_BOUNDARIES="${PHASE1_GLASS_BOUNDARIES:-$release_dir/grasp_range_integration/phase1/glass_boundaries.json}"
python3 -u -m grasp_range_integration.phase1.glass_boundary_guard \
  --config "$PHASE1_GLASS_BOUNDARIES" --map "$release_dir/ble_scene_17.pgm" \
  --map-yaml "$release_dir/ble_scene_17.yaml"

# The release-local entry installs only the audited relationship route planner,
# localization guard and DeepSeek visual-localization hook.  The protected
# mission remains the sole chassis executor; no arm module is imported here.
exec 9>/tmp/car1_phase1_find_only.lock
if ! flock -n 9; then
  echo "另一个第一阶段寻物任务正在占用底盘、雷达或摄像头。" >&2
  exit 2
fi

# These USB port identities preserve the trained LEFT/RIGHT order even when
# /dev/videoN is renumbered. The two cameras share a serial, so by-id is unsafe.
left_usb=/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0
right_usb=/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0
if [[ ! -e "$left_usb" ]]; then left_usb=/dev/video2; fi
if [[ ! -e "$right_usb" ]]; then right_usb=/dev/video0; fi
export PHASE1_LEFT_CAMERA="${PHASE1_LEFT_CAMERA:-$left_usb}"
export PHASE1_RIGHT_CAMERA="${PHASE1_RIGHT_CAMERA:-$right_usb}"
for camera_node in "$PHASE1_LEFT_CAMERA" "$PHASE1_RIGHT_CAMERA"; do
  if [[ ! -c "$camera_node" || ! -r "$camera_node" || ! -w "$camera_node" ]]; then
    echo "未启动：摄像头设备不存在或无读写权限：$camera_node；请检查USB连接及video组权限。" >&2
    exit 2
  fi
done
if [[ "$(readlink -f "$PHASE1_LEFT_CAMERA")" == "$(readlink -f "$PHASE1_RIGHT_CAMERA")" ]]; then
  echo "未启动：左右摄像头解析到同一设备；不得交换训练左右顺序。" >&2
  exit 2
fi

track_root="${FINAL_WORK_TRACK_ROOT:-/home/dachuang/dachuang/track}"
if [[ -f "$track_root/final_work/data/camera_broker.json" ]]; then
  export FINAL_WORK_CAMERA_BROKER_URL=http://127.0.0.1:18881
  # Share cameras without replacing car1.0 navigation with track/raspberry_pi.
  export PYTHONPATH="$track_root/final_work/pi/camera_bootstrap:$car_root:$python_root:$track_root:$PYTHONPATH"
fi
if [[ -n "${FINAL_WORK_CAMERA_BROKER_URL:-}" ]]; then
  python3 -c 'from final_work.pi.camera_proxy import require_broker_ready; require_broker_ready()'
  echo "相机共享采集检查通过：复用任务录像采集，不重开USB设备。"
else
# Release ONLY this user's known grab_visual preview and its own ffmpeg
# children. No fuser -k, global pkill ffmpeg, sudo, or mission termination.
command -v fuser >/dev/null || { echo "未启动：缺少fuser（psmisc），无法检查摄像头占用。" >&2; exit 2; }
task_uid="$(id -u)"
mapfile -t preview_pids < <(ps -eo pid=,uid=,comm=,args= | awk -v owner="$task_uid" \
  '$2 == owner && $3 ~ /^python([0-9.]*)?$/ && $0 ~ /[[:space:]]\/home\/dachuang\/grab_visual\/web_app\.py([[:space:]]|$)/ {print $1}')
for preview_pid in "${preview_pids[@]}"; do
  mapfile -t preview_children < <(ps -eo pid=,ppid=,uid=,comm= | awk \
    -v parent="$preview_pid" -v owner="$task_uid" '$2 == parent && $3 == owner && $4 == "ffmpeg" {print $1}')
  echo "释放已知摄像头预览：grab_visual PID=$preview_pid（8080预览将关闭）"
  kill -TERM -- "$preview_pid" 2>/dev/null || true
  for child_pid in "${preview_children[@]}"; do
    if ps -p "$child_pid" -o uid=,comm= | awk -v owner="$task_uid" \
      '$1 == owner && $2 == "ffmpeg" {found=1} END {exit !found}'; then
      kill -TERM -- "$child_pid" 2>/dev/null || true
    fi
  done
done
for camera_wait in {1..15}; do
  if ! fuser "$PHASE1_LEFT_CAMERA" "$PHASE1_RIGHT_CAMERA" >/dev/null 2>&1; then break; fi
  sleep 0.2
done
if fuser "$PHASE1_LEFT_CAMERA" "$PHASE1_RIGHT_CAMERA" >/dev/null 2>&1; then
  echo "未启动：摄像头仍被其他进程占用；不会自动强杀：" >&2
  fuser -v "$PHASE1_LEFT_CAMERA" "$PHASE1_RIGHT_CAMERA" >&2 || true
  exit 2
fi
fi
echo "相机启动检查通过：LEFT=$PHASE1_LEFT_CAMERA RIGHT=$PHASE1_RIGHT_CAMERA；双摄试拍由任务入口完成。"

exec python3 -u "$release_dir/grasp_range_integration/phase1/relationship_find_only_entry.py" \
  --enable-motion \
  --decision "$decision" \
  --map "$release_dir/ble_scene_17.pgm" \
  --map-yaml "$release_dir/ble_scene_17.yaml" \
  --output "$car_root/search_runs/ble_scene_17" \
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
  --turn-speed 2600 \
  --max-turn-speed 2600 \
  --relocalization-speed 300 \
  --relocalization-max-speed 460 \
  --relocalization-turn-speed 1500 \
  --relocalization-max-turn-speed 2600 \
  --relocalization-turn-step 300 \
  --localization-timeout-seconds "${PHASE1_LOCALIZATION_TIMEOUT_SECONDS:-240}" \
  --localization-required-moves 1 \
  --stop-distance 0.45 \
  --clearance 0.30 \
  --camera-backend opencv \
  --camera-device 2 \
  --dual-usb-cameras \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-stream-base-url "" \
  --visual-localization-semantic-map "$release_dir/semantic_localization_landmarks.json" \
  --max-runtime-min 15
