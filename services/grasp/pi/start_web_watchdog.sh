#!/bin/bash
cd /home/dachuang/grab_visual
i=0
while true; do
  i=$((i+1))
  bash start_web.sh > logs/web_watchdog_${i}.log 2>&1
  echo "[watchdog] web_app exited, restarting in 3s..." >> logs/web_watchdog.log
  sleep 3
done
