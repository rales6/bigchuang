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

instruction="${1:-去柜子边上夹取一个红色的小物块}"
output="${2:-search_runs/phase1/latest_deepseek_decision.json}"

# This script intentionally has no execution flag and cannot start the car.
exec python3 -u -m grasp_range_integration.phase1.prepare_phase1 \
  "$instruction" \
  --semantic-map "$release_dir/semantic_runtime.json" \
  --deepseek-config "${DEEPSEEK_CONFIG:-$car_root/secrets/deepseek_vision.json}" \
  --output "$output"
