#!/usr/bin/env bash
set -euo pipefail

KEY="${FINAL_WORK_TUNNEL_KEY:-/home/robot/.ssh/final_work_3090_ed25519}"
GPU_HOST="${FINAL_WORK_GPU_HOST:-192.0.2.20}"
GPU_USER="${FINAL_WORK_GPU_USER:-sjtu}"
REMOTE_PORT="${FINAL_WORK_TUNNEL_PORT:-18780}"
SSH_PID=""

stop_tunnel() {
  if [[ -n "$SSH_PID" ]]; then
    kill -TERM "$SSH_PID" 2>/dev/null || true
    wait "$SSH_PID" 2>/dev/null || true
  fi
  exit 0
}
trap stop_tunnel TERM INT

if [[ ! -r "$KEY" ]]; then
  echo "reverse tunnel key missing: $KEY" >&2
  exit 2
fi

while true; do
  echo "[$(date '+%F %T')] forwarding 3090 127.0.0.1:${REMOTE_PORT} -> Pi 127.0.0.1:8780"
  ssh -N -T \
    -i "$KEY" \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -R "127.0.0.1:${REMOTE_PORT}:127.0.0.1:8780" \
    "${GPU_USER}@${GPU_HOST}" &
  SSH_PID="$!"
  wait "$SSH_PID" || true
  SSH_PID=""
  echo "[$(date '+%F %T')] reverse tunnel disconnected; retrying in 2s" >&2
  sleep 2
done
