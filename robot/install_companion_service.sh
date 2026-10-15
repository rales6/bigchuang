#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
chmod +x companion_service.sh
sudo install -m 0644 robot-companion.service /etc/systemd/system/robot-companion.service
sudo systemctl daemon-reload
sudo systemctl enable --now robot-companion.service
sudo systemctl --no-pager --full status robot-companion.service
