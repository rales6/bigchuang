#!/usr/bin/env bash
set -euo pipefail
ROOT="${FINAL_WORK_TRACK_ROOT:-/home/robot/dachuang/track}"
PID_FILE="$ROOT/final_work/pi_bridge.pid"
if [[ -f "$PID_FILE" ]]; then
  PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$PID" && -r "/proc/$PID/cmdline" ]] && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q 'final_work.pi.robot_bridge'; then
    kill -TERM "$PID"
    echo "Pi bridge stopped: pid=$PID"
  else
    echo "Stale PID file ignored: $PID_FILE"
  fi
  rm -f "$PID_FILE"
else
  echo "Pi bridge is not running"
fi

TUNNEL_PID_FILE="$ROOT/final_work/pi_reverse_tunnel.pid"
if [[ -f "$TUNNEL_PID_FILE" ]]; then
  TUNNEL_PID="$(tr -cd '0-9' < "$TUNNEL_PID_FILE")"
  if [[ -n "$TUNNEL_PID" && -r "/proc/$TUNNEL_PID/cmdline" ]] && tr '\0' ' ' < "/proc/$TUNNEL_PID/cmdline" | grep -q 'start_pi_reverse_tunnel.sh'; then
    kill -TERM "$TUNNEL_PID" 2>/dev/null || true
    echo "Pi reverse tunnel stopped: pid=$TUNNEL_PID"
  fi
  rm -f "$TUNNEL_PID_FILE"
fi
