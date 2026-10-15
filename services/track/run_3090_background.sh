#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/sjtu/track}"
PID_FILE="$ROOT/final_work/agent_web.pid"
LOG_FILE="$ROOT/final_work/logs/agent_web.log"
mkdir -p "$ROOT/final_work/logs"
if [[ -f "$PID_FILE" ]]; then
  PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$PID" && -r "/proc/$PID/cmdline" ]] && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q 'final_work.gpu.app'; then
    echo "Agent web already running: pid=$PID"
    exit 0
  fi
  rm -f "$PID_FILE"
fi
cd "$ROOT"
nohup bash final_work/start_3090.sh >>"$LOG_FILE" 2>&1 &
echo "$!" > "$PID_FILE"
sleep 1
echo "Agent web started: pid=$(cat "$PID_FILE") log=$LOG_FILE"

