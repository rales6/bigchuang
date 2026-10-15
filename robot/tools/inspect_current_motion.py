"""Read-only snapshot of the Pi motion stack for field diagnosis."""

from __future__ import annotations

import argparse
import getpass
import os
import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "dachuang.local", username="dachuang",
        password=os.environ.get("PI_SSH_PASSWORD") or getpass.getpass("Pi SSH password: "),
        timeout=30,
    )
    command = r"""
cd ~/dachuang/car1.0
echo '=== PROCESSES ==='
pgrep -af 'companion_service|autonomous_mapping|map_camera_inspection|object_search_mission|natural_search' || true
echo '=== CAMERA HOLDERS ==='
pgrep -af '[f]fmpeg|[r]picam|[l]ibcamera' || true
fuser -v /dev/video0 /dev/video2 2>/dev/null || true
echo '=== SERVICE ==='
systemctl is-active robot-companion.service || true
echo '=== BLUEZ ==='
bluetoothctl info 64:B7:08:69:9A:E6 2>/dev/null | grep -E 'Name:|Connected:|ServicesResolved:' || true
echo '=== LATEST SERVICE LOG ==='
latest=$(ls -1t service_runs/*.log 2>/dev/null | head -1)
echo "$latest"
test -n "$latest" && tail -n 100 "$latest" || true
echo '=== LATEST SEARCH LOG ==='
latest_search=$(find search_runs -type f \( -name '*.log' -o -name 'mission.json' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
echo "$latest_search"
test -n "$latest_search" && tail -n 80 "$latest_search" || true
echo '=== LATEST INSPECTION LOG ==='
latest_inspection_log=$(ls -1t inspection_logs/inspect_map_*.log 2>/dev/null | head -1)
echo "$latest_inspection_log"
test -n "$latest_inspection_log" && tail -n 160 "$latest_inspection_log" || true
echo '=== LATEST INSPECTION ==='
latest_inspection=$(find inspections/ble_scene_17 -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
echo "$latest_inspection"
test -n "$latest_inspection" && find "$latest_inspection" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %f %s bytes\n' | sort || true
echo '=== DEPLOYED LOCALIZATION LOGIC ==='
grep -n '无需强制换位\|拒绝跨区域全局跳点\|localization_fast_accept_margin' raspberry_pi/map_camera_inspection.py || true
grep -n 'localization-fast-accept\|ignore-trajectory' inspect_map.sh || true
stat -c '%y %n' raspberry_pi/map_camera_inspection.py inspect_map.sh 2>/dev/null || true
"""
    if args.compact:
        command = r"""
cd ~/dachuang/car1.0
echo '=== PROCESSES ==='
ps -eo pid,etime,stat,cmd | grep -E 'map_camera_inspection|object_search_mission|companion_service' | grep -v grep || true
echo '=== DEVICE HOLDERS ==='
fuser -v /dev/ttyACM0 /dev/video0 /dev/video2 2>/dev/null || true
echo '=== BLUEZ ==='
bluetoothctl info 64:B7:08:69:9A:E6 2>/dev/null | grep -E 'Name:|Connected:|ServicesResolved:' || true
echo '=== LATEST INSPECTION ==='
latest=$(find inspections/ble_scene_17 -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
echo "$latest"
test -n "$latest" && find "$latest" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %f %s bytes\n' | sort || true
echo '=== LATEST INSPECTION LOG ==='
latest_log=$(ls -1t inspection_logs/inspect_map_*.log 2>/dev/null | head -1)
echo "$latest_log"
test -n "$latest_log" && tail -n 120 "$latest_log" || true
echo '=== LOCALIZATION TAIL ==='
test -f "$latest/localization_diagnostics.jsonl" && tail -n 8 "$latest/localization_diagnostics.jsonl" || true
echo '=== INSPECTION SCRIPT SPEEDS ==='
grep -E -- '--speed |--max-drive-speed|--drive-speed-step|--start-kick-speed|--slow-speed|--turn-speed|--command-min-interval-ms|--command-ttl-ms' inspect_map.sh || true
grep -n '_start_motion_keepalive(client, owns_client)\|linear_start_kick_mm_s\|navigation_diagnostics.jsonl' raspberry_pi/map_camera_inspection.py || true
"""
    _stdin, stdout, stderr = client.exec_command(command, timeout=60)
    code = stdout.channel.recv_exit_status()
    print(stdout.read().decode("utf-8", errors="replace"))
    error = stderr.read().decode("utf-8", errors="replace")
    if error:
        print(error)
    client.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
