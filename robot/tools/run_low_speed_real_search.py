from __future__ import annotations

import os
import shlex
import sys
import time

import paramiko


WEB_PATTERN = "[g]rab_visual/web_app.py"
WEB_COMMAND = (
    "/home/dachuang/dachuang/.venv/bin/python /home/dachuang/grab_visual/web_app.py "
    "--qwen-url http://192.168.55.33:8010 --left-camera 2 --right-camera 0 "
    "--host 0.0.0.0 --port 8080"
)


def remote(client, command, timeout=45):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if code:
        raise RuntimeError(error or output or f"remote command failed with {code}")
    return output


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "dachuang.local",
        username="dachuang",
        password=os.environ["PI_SSH_PASSWORD"],
        timeout=45,
    )
    web_was_running = bool(remote(client, f"pgrep -f '{WEB_PATTERN}' || true").strip())
    try:
        if web_was_running:
            print("STOPPING_EXISTING_CAMERA_WEB_SERVICE", flush=True)
            remote(
                client,
                f"parents=$(pgrep -f '{WEB_PATTERN}' || true); "
                "children=''; for p in $parents; do children=\"$children $(pgrep -P $p 2>/dev/null || true)\"; done; "
                "if [ -n \"$children\" ]; then kill -TERM $children 2>/dev/null || true; fi; "
                "if [ -n \"$parents\" ]; then kill -TERM $parents 2>/dev/null || true; fi; "
                "for i in $(seq 1 20); do "
                f"pgrep -f '{WEB_PATTERN}' >/dev/null || exit 0; sleep 0.5; done; "
                "exit 1",
                timeout=20,
            )
            # The web service normally owns these exact ffmpeg commands. Kill
            # only leftovers with the former parent gone, never arbitrary
            # camera processes.
            remote(
                client,
                "orphans=$(pgrep -f '[f]fmpeg.*-i /dev/video[02].*pipe:1' || true); "
                "if [ -n \"$orphans\" ]; then kill -TERM $orphans; sleep 2; fi",
                timeout=10,
            )
        busy = remote(
            client,
            "fuser /dev/ttyACM0 /dev/video0 /dev/video2 2>/dev/null || true",
        ).strip()
        if busy:
            raise RuntimeError(f"devices still busy after web stop: {busy}")
        print("DEVICES_FREE_STARTING_LOW_SPEED_TEST", flush=True)
        command = (
            "cd ~/dachuang/car1.0 && source .venv-pi/bin/activate && "
            "export PYTHONUNBUFFERED=1 SEARCH_TEST_MAX_DISTANCE=0.80 && "
            "timeout --signal=INT --kill-after=15s 300s "
            "./natural_search.sh '去柜子边上找一个红色瓶子' real"
        )
        _, stdout, stderr = client.exec_command(command, timeout=330)
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                text = stdout.channel.recv(4096).decode("utf-8", errors="replace")
                if text:
                    print(text, end="", flush=True)
            if stdout.channel.recv_stderr_ready():
                text = stdout.channel.recv_stderr(4096).decode("utf-8", errors="replace")
                if text:
                    print(text, end="", file=sys.stderr, flush=True)
            time.sleep(0.15)
        while stdout.channel.recv_ready():
            print(stdout.channel.recv(4096).decode("utf-8", errors="replace"), end="", flush=True)
        while stdout.channel.recv_stderr_ready():
            print(stdout.channel.recv_stderr(4096).decode("utf-8", errors="replace"), end="", file=sys.stderr, flush=True)
        code = stdout.channel.recv_exit_status()
        print(f"LOW_SPEED_SEARCH_EXIT={code}", flush=True)
    finally:
        # Belt-and-suspenders stop in case the mission process ended abnormally.
        try:
            remote(
                client,
                "pids=$(pgrep -f '[o]bject_search_mission' || true); "
                "if [ -n \"$pids\" ]; then kill -INT $pids; sleep 2; fi",
                timeout=10,
            )
        except Exception as exc:
            print(f"MISSION_CLEANUP_WARNING={exc}", flush=True)
        if web_was_running:
            print("RESTORING_CAMERA_WEB_SERVICE", flush=True)
            launch = (
                "cd /home/dachuang/grab_visual && "
                f"nohup {WEB_COMMAND} >/home/dachuang/grab_visual/web_app.log 2>&1 </dev/null &"
            )
            try:
                remote(client, launch, timeout=10)
                time.sleep(3)
                restored = remote(client, f"pgrep -af '{WEB_PATTERN}' || true").strip()
                print("CAMERA_WEB_RESTORED=" + (restored or "NO"), flush=True)
            except Exception as exc:
                print(f"CAMERA_WEB_RESTORE_WARNING={exc}", flush=True)
        client.close()


if __name__ == "__main__":
    main()
