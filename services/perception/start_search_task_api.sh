#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source .venv/bin/activate

semantic_map="${SEMANTIC_MAP:-$HOME/car_runtime/semantic_release_current/semantic_runtime.json}"
model_path="${QWEN_MODEL:-$PWD/models/qwen/Qwen3-VL-2B-Instruct}"
yolo_model="${YOLO_MODEL:-$PWD/models/yolo/yolo11n-seg.pt}"

exec python -u scripts/run_search_task_api.py \
  --semantic-map "$semantic_map" \
  --model-path "$model_path" \
  --yolo-model "$yolo_model" \
  --device cuda \
  --backend qwen \
  --host 0.0.0.0 \
  --port "${SEARCH_TASK_PORT:-8030}"
