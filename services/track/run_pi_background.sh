#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/dachuang/dachuang/track}"
PID_FILE="$ROOT/final_work/pi_bridge.pid"
LOG_FILE="$ROOT/final_work/logs/pi_bridge.log"
mkdir -p "$ROOT/final_work/logs"
if [[ -f "$PID_FILE" ]]; then
  PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$PID" && -r "/proc/$PID/cmdline" ]] && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q 'final_work.pi.robot_bridge'; then
    echo "Pi bridge already running: pid=$PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi
cd "$ROOT"
nohup setsid bash final_work/start_pi.sh </dev/null >>"$LOG_FILE" 2>&1 &
echo "$!" > "$PID_FILE"
sleep 1
echo "Pi bridge started: pid=$(cat "$PID_FILE") log=$LOG_FILE"
bash "$ROOT/final_work/run_pi_reverse_tunnel_background.sh"
