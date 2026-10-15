#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source .venv-pi/bin/activate

mkdir -p search_runs
exec python3 -u -m raspberry_pi.search_dashboard \
  --bind 0.0.0.0 \
  --port 8766 \
  --status search_runs/live_status.json \
  --image search_runs/live_map.png
