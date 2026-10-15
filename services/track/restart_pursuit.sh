#!/bin/bash
pkill -f "pursuit_server.py" 2>/dev/null
sleep 2
cd ~/track || exit 1
nohup ~/3090/.venv/bin/python pursuit_server.py --port 8021 --device cuda --no-depth > ~/track/server.log 2>&1 &
sleep 8
curl -s http://127.0.0.1:8021/health
echo
