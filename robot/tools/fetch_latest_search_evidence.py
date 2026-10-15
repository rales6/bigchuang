from __future__ import annotations

import json
import os
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "search_runs" / "latest_real_evidence"


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "dachuang.local", username="dachuang",
        password=os.environ["PI_SSH_PASSWORD"], timeout=45,
    )
    try:
        _stdin, stdout, stderr = client.exec_command(
            "cd ~/dachuang/car1.0 && "
            "find search_runs/ble_scene_17 -mindepth 2 -maxdepth 2 -name mission.json "
            "-printf '%T@ %h\\n' | sort -n | tail -1 | cut -d' ' -f2-",
            timeout=30,
        )
        if stdout.channel.recv_exit_status():
            raise RuntimeError(stderr.read().decode("utf-8", errors="replace"))
        directory = stdout.read().decode("utf-8", errors="replace").strip()
        if not directory:
            raise RuntimeError("latest mission directory not found")
        remote_root = "/home/dachuang/dachuang/car1.0"
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            for remote, name in (
                (f"{remote_root}/{directory}/mission.json", "mission.json"),
                (f"{remote_root}/{directory}/view_01.jpg", "view_01.jpg"),
                (f"{remote_root}/{directory}/view_01_camera_2.jpg", "view_01_camera_2.jpg"),
                (f"{remote_root}/search_runs/live_map.png", "live_map.png"),
                (f"{remote_root}/search_runs/live_status.json", "live_status.json"),
            ):
                try:
                    sftp.get(remote, str(OUTPUT / name))
                except FileNotFoundError:
                    pass
        mission = json.loads((OUTPUT / "mission.json").read_text(encoding="utf-8"))
        print(json.dumps({
            "remote_directory": directory,
            "local_directory": str(OUTPUT),
            "result": mission.get("result"),
            "summary": mission.get("summary"),
        }, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
