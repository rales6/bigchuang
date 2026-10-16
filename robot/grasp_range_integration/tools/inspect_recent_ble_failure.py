"""Read the latest Pi web/mission logs without starting robot hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath

from deploy_track_web_pick import HOSTS, _connect


def _read_tail(sftp, path: str, limit: int = 18000) -> str:
    with sftp.open(path, "rb") as stream:
        size = stream.stat().st_size
        stream.seek(max(0, size - limit))
        return stream.read().decode("utf-8", "replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concise", action="store_true")
    parser.add_argument("--ble-state", action="store_true")
    parser.add_argument("--helper", action="store_true")
    parser.add_argument("--rfkill-package", action="store_true")
    args = parser.parse_args()
    spec = HOSTS["pi"]
    client = _connect(spec)
    try:
        if args.rfkill_package:
            for command in ("apt-cache policy rfkill", "dpkg-query -W rfkill", "dpkg -L rfkill",
                            "ls -l /usr/sbin/rfkill", "dpkg -V rfkill"):
                _stdin, stdout, stderr = client.exec_command(command, timeout=10)
                print(command, "exit=", stdout.channel.recv_exit_status())
                print((stdout.read().decode("utf-8", "replace") +
                       stderr.read().decode("utf-8", "replace"))[-1500:])
            return
        if args.helper:
            with client.open_sftp() as sftp:
                path = "/usr/local/sbin/robot-bluetooth-reset"
                try:
                    with sftp.open(path, "rb") as stream:
                        print(stream.read().decode("utf-8", "replace"))
                except FileNotFoundError:
                    print("helper_missing")
            return
        if args.ble_state:
            for command in (
                "bluetoothctl show",
                "bluetoothctl info AA:BB:CC:DD:EE:01",
                "rfkill list bluetooth",
                "/usr/sbin/rfkill list bluetooth",
                "systemctl is-active bluetooth",
                "sudo -n true",
                "ps -eo pid=,args=",
                "curl --fail --silent --show-error http://127.0.0.1:8770/state",
            ):
                _stdin, stdout, stderr = client.exec_command(command, timeout=12)
                output = stdout.read().decode("utf-8", "replace")
                errors = stderr.read().decode("utf-8", "replace")
                print(command, "exit=", stdout.channel.recv_exit_status())
                if command.startswith("ps "):
                    print("\n".join(line for line in output.splitlines() if
                                    any(word in line for word in (
                                        "companion_service", "grab_visual", "pursuit",
                                        "object_search_mission", "relationship_find_only",
                                        "bluetoothd", "robot_bridge"))))
                else:
                    print((output + errors)[-1800:])
            return
        with client.open_sftp() as sftp:
            logs = spec["root"] + "/final_work/logs"
            entries = sorted(
                (item for item in sftp.listdir_attr(logs)
                 if item.filename.endswith(".log") and
                 ("map_navigation_search" in item.filename or
                  (not args.concise and item.filename == "pi_bridge.log"))),
                key=lambda item: item.st_mtime, reverse=True,
            )[:2 if args.concise else 5]
            for item in entries:
                path = str(PurePosixPath(logs) / item.filename)
                print("\n### " + path)
                print(_read_tail(sftp, path, 10000 if args.concise else 18000))
            car = "/home/robot/dachuang/car1.0/search_runs/ble_scene_17"
            runs = sorted(
                (item for item in sftp.listdir_attr(car)
                 if item.filename[:8].isdigit()),
                key=lambda item: item.st_mtime, reverse=True,
            )[:1 if args.concise else 4]
            for item in runs:
                root = str(PurePosixPath(car) / item.filename)
                print("\n### " + root)
                for name in ("mission.json", "phase1_verified_handoff.json"):
                    path = str(PurePosixPath(root) / name)
                    try:
                        with sftp.open(path, "rb") as stream:
                            value = json.load(stream)
                    except (FileNotFoundError, json.JSONDecodeError):
                        continue
                    if args.concise and name == "mission.json":
                        value = {key: value.get(key) for key in (
                            "instruction", "started_at", "finished_at", "result",
                            "summary", "hardware_motion_enabled")}
                        print(name, json.dumps(value, ensure_ascii=False))
                    else:
                        print(name, json.dumps(value, ensure_ascii=False)[-5000:])
    finally:
        client.close()


if __name__ == "__main__":
    main()
