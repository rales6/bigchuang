#!/bin/bash
cd /home/robot/grab_visual
exec /home/robot/dachuang/.venv/bin/python arm_grasp_client.py --server "${GRASP_SERVER_URL:-ws://192.0.2.20:8020/ws/grasp}" --target "${1:-}" --camera-id 0
