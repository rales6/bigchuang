"""Start one bridge-managed pursuit from the Pi without exposing its token.

Example: python -m final_work.pi.dispatch_pursuit --target 红色小物块 --real
The bridge retains ownership of the run and writes its normal task log.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--grasp", action="store_true", help="对齐通过后夹取（默认只追踪对齐）")
    parser.add_argument("--camera-order", choices=("normal", "swapped"),
                        help="相对USB 0/2；默认读取网页保存的左右配置")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    headers = {"Content-Type": "application/json; charset=utf-8"}
    if args.real:
        token_path = Path(__file__).resolve().parents[1] / "data" / "hardware_token"
        headers["X-Robot-Safety-Token"] = token_path.read_text(
            encoding="utf-8"
        ).strip()
    if args.camera_order is None:
        web_url = os.environ.get('FINAL_WORK_WEB_URL','http://192.0.2.20:8030').rstrip('/')
        with request.urlopen(web_url+'/api/robot/camera-order',timeout=5) as response:
            order=json.load(response)['camera_order']
        if not isinstance(order.get('swapped'),bool):
            raise RuntimeError('网页左右配置无效；请明确使用 --camera-order')
    else:
        order={'swapped':args.camera_order=='swapped','source':'explicit_cli'}
    payload = json.dumps({"target": args.target, "dry_run": not args.real,
                          "grasp":args.grasp,"camera_order":order}, ensure_ascii=False).encode("utf-8")
    action = request.Request(
        "http://127.0.0.1:8780/actions/pursuit/start",
        data=payload,
        headers=headers,
        method="POST",
    )
    with request.urlopen(action, timeout=8) as response:
        result = json.load(response)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
