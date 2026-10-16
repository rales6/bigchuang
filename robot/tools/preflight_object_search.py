from __future__ import annotations

import os
import paramiko


def connect(host, user, password, port=22):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=45, auth_timeout=45, banner_timeout=45)
    return client


def read(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=40)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode("utf-8", errors="replace"), stderr.read().decode("utf-8", errors="replace")


def main():
    pi = connect("pi.example.invalid", "dachuang", os.environ["PI_SSH_PASSWORD"])
    _, pi_out, pi_err = read(
        pi,
        "echo PI_PROCESSES; pgrep -af 'autonomous_mapping|map_camera_inspection|object_search_mission|ffmpeg' || true; "
        "echo PI_CAMERA_PARENTS; ps -o pid,ppid,stat,lstart,cmd -p $(pgrep -d, -f '[f]fmpeg.*video[02]' || echo 0) 2>/dev/null || true; "
        "pgrep -af 'grab_visual|web_app|camera' || true; "
        "echo PI_DEVICES; fuser /dev/ttyACM0 /dev/video0 /dev/video2 2>/dev/null || true; "
        "echo PI_BLE_SCAN; timeout 8s sudo btmgmt --index 0 find -l 2>/dev/null | grep -A3 -B1 'ESP32-Robot-Car' || true",
    )
    pi.close()
    rtx = connect("198.51.100.20", "sjtu", os.environ["RTX_SSH_PASSWORD"], port=2020)
    _, rtx_out, rtx_err = read(
        rtx,
        "echo RTX_PROCESSES; pgrep -af 'run_stage1_perception_server|8011' || true; "
        "echo RTX_LISTEN; ss -ltnp 2>/dev/null | grep ':8011' || true; "
        "echo RTX_FILES; readlink -f /proc/$(pgrep -f 'run_stage1_perception_server' | head -1)/cwd 2>/dev/null || true; "
        "tr '\\0' ' ' </proc/$(pgrep -f 'run_stage1_perception_server' | head -1)/cmdline 2>/dev/null || true; echo",
    )
    rtx.close()
    print(pi_out)
    print(pi_err)
    print(rtx_out)
    print(rtx_err)


if __name__ == "__main__":
    main()
