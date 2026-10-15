#!/usr/bin/env bash
# 3090 侧: 启动视觉伺服抓取服务器
#
# 用法:
#   bash grab_target/start_grasp_server.sh
#
# 依赖:
#   - 虚拟环境 ~/3090/.venv
#   - Qwen 模型 ~/3090/models/qwen/Qwen3-VL-2B-Instruct
#   - YOLO 模型 ~/3090/models/yolo/yolo11n-seg.pt (可选)
#
# 端口: 8020 (WebSocket /ws/grasp + HTTP /health)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

VENV="${HOME}/3090/.venv"
if [ ! -d "$VENV" ]; then
    echo "虚拟环境不存在: $VENV"
    exit 1
fi

DEFAULT_CONFIG="${PROJECT_DIR}/configs/linux_3090_remote.yaml"
if [ ! -f "$DEFAULT_CONFIG" ]; then
    DEFAULT_CONFIG="${HOME}/3090_new_method/configs/linux_3090_remote.yaml"
fi
CONFIG="${GRASP_CONFIG:-$DEFAULT_CONFIG}"
CALIBRATION="${GRASP_CALIBRATION:-${SCRIPT_DIR}/arm_visual_calibration.json}"
export GRAB_LOG_DIR="${GRAB_LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "$GRAB_LOG_DIR"

echo "=== 启动视觉伺服抓取服务器 ==="
echo "程序目录: ${SCRIPT_DIR}"
echo "配置文件: ${CONFIG}"
echo "标定文件: ${CALIBRATION}"
echo "端口: 8020"
echo

echo "JSONL logs: ${GRAB_LOG_DIR}"

exec "$VENV/bin/python" "$SCRIPT_DIR/arm_grasp_server.py" \
    --config "$CONFIG" \
    --calibration "$CALIBRATION" \
    --host 0.0.0.0 \
    --port 8020
