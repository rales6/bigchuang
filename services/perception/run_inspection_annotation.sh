#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source .venv/bin/activate

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <巡检批次或巡检根目录> [额外参数]" >&2
  echo "示例: $0 ~/car_inspections/ble_scene_17 --semantic-map ~/car_inspections/ble_scene_17_semantics.json" >&2
  exit 2
fi

inspection_path="$1"
shift

exec python -u scripts/run_inspection_annotation.py \
  --inspection "$inspection_path" \
  --model-path models/qwen/Qwen3-VL-2B-Instruct \
  --device cuda \
  "$@"
