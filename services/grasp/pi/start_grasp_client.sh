#!/bin/bash
cd /home/dachuang/grab_visual
exec /home/dachuang/dachuang/.venv/bin/python arm_grasp_client.py --server "${GRASP_SERVER_URL:-ws://192.168.55.33:8020/ws/grasp}" --target "${1:-}" --camera-id 0
