"""Persistent port-8766 dashboard for known-map object-search missions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

from raspberry_pi.realtime_map import RealtimeMapServer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve persistent object-search map snapshots")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--status", default="search_runs/live_status.json")
    parser.add_argument("--image", default="search_runs/live_map.png")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser


def run(args: argparse.Namespace) -> None:
    server = RealtimeMapServer(args.bind, args.port, refresh_hz=2.0)
    print(f"寻物地图监控已启动：{server.start()}", flush=True)
    status_path = Path(args.status)
    image_path = Path(args.image)
    stopping = False

    def stop(_signal=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signature = None
    try:
        while not stopping:
            if status_path.exists() and image_path.exists():
                current = (status_path.stat().st_mtime_ns, image_path.stat().st_mtime_ns)
                if current != signature:
                    try:
                        status = json.loads(status_path.read_text(encoding="utf-8"))
                        server.publish_image(image_path.read_bytes(), status)
                        signature = current
                    except (OSError, json.JSONDecodeError):
                        # The mission uses atomic replace, but a failed read
                        # must never take down the persistent dashboard.
                        pass
            time.sleep(max(0.05, float(args.poll_seconds)))
    finally:
        server.stop()


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
