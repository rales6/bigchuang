#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source .venv-pi/bin/activate

decision="${1:-maps/search_decision_cabinet_red_bottle.json}"
start_x="${2:?请传入当前雷达定位得到的 x，例如：./plan_search_route.sh "$decision" 0.0 0.0}"
start_y="${3:?请传入当前雷达定位得到的 y，例如：./plan_search_route.sh "$decision" 0.0 0.0}"
stem="$(basename "${decision%.*}")"

python3 -u -m raspberry_pi.search_route_planner \
  --map maps/ble_scene_17.pgm \
  --map-yaml maps/ble_scene_17.yaml \
  --decision "$decision" \
  --start-x "$start_x" \
  --start-y "$start_y" \
  --clearance 0.30 \
  --output "maps/${stem}_route.json" \
  --render "maps/${stem}_route.png"

echo "只生成了路径，未发送小车运动命令：maps/${stem}_route.json"
