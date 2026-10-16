#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/gpu/track}"
PID_FILE="$ROOT/final_work/agent_web.pid"
if [[ ! -f "$PID_FILE" ]]; then echo "Agent web is not running"; exit 0; fi
PID="$(tr -cd '0-9' < "$PID_FILE")"
if [[ -n "$PID" && -r "/proc/$PID/cmdline" ]] && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q 'final_work.gpu.app'; then
  kill -TERM "$PID"
  echo "Agent web stopped: pid=$PID"
else
  echo "Stale PID file ignored: $PID_FILE"
fi
rm -f "$PID_FILE"
