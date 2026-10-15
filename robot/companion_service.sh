#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source .venv-pi/bin/activate
exec python3 -u -m raspberry_pi.companion_service --host 0.0.0.0 --port "${COMPANION_PORT:-8770}"
