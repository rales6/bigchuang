#!/usr/bin/env bash
# 机械臂示教夹取模型服务 (3090, 端口 8022)
set -euo pipefail
cd "$(dirname "$0")"
exec /home/sjtu/3090/.venv/bin/python -u arm_teach_model.py \
  --host 0.0.0.0 --port 8022
