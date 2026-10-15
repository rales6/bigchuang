"""Read-only audit of the existing GPU web step dispatch; no model or hardware calls."""

from __future__ import annotations

import argparse
import json

from deploy_track_web_pick import HOSTS, _connect, _read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()
    spec = HOSTS["gpu"]
    client = _connect(spec)
    try:
        _stdin, stdout, stderr = client.exec_command(
            "curl --fail --silent --show-error http://127.0.0.1:8030/api/health", timeout=12)
        raw = stdout.read().decode("utf-8", "replace")
        if stdout.channel.recv_exit_status():
            raise RuntimeError(stderr.read().decode("utf-8", "replace"))
        print(json.dumps(json.loads(raw), ensure_ascii=False))
        if args.health_only:
            return
        with client.open_sftp() as sftp:
            for filename in ("app.py", "robot_client.py", "planner.py"):
                path = spec["root"] + "/final_work/gpu/" + filename
                lines = _read(sftp, path).decode("utf-8", "replace").splitlines()
                hits = [index for index, line in enumerate(lines)
                        if any(word in line for word in (
                            "observation-plan", "patrol_search", "requested_intent",
                            "target_query", "def execute_step", "def _execute_step"))]
                selected = set()
                for hit in hits:
                    selected.update(range(max(0, hit - 5), min(len(lines), hit + 9)))
                print("\n### " + path)
                for index in sorted(selected):
                    print(f"{index + 1:4}: {lines[index]}")
                if filename == "robot_client.py":
                    for index in range(85, min(len(lines), 151)):
                        print(f"{index + 1:4}: {lines[index]}")
                if filename == "app.py":
                    for index in range(610, min(len(lines), 665)):
                        print(f"{index + 1:4}: {lines[index]}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
