#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

MODEL_PATH="${MODEL_PATH:-$HOME/3090/models/qwen/Qwen3-VL-2B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
DEVICE="${DEVICE:-cuda}"
PI_URL="${PI_URL:-http://192.0.2.15:8766}"

echo "[3090] Qwen semantic map hub"
echo "[3090] web: http://$(hostname -I | awk '{print $1}'):${PORT}"
echo "[3090] pi : ${PI_URL}"

python scripts/run_qwen_landmark_server.py \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --pi-url "${PI_URL}"
