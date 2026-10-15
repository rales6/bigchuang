from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp_range_integration.adapters.arm_teach_model import (  # noqa: E402
    ArmTeachModelClient,
    GraspModelError,
)
from grasp_range_integration.perception.dual_camera import (  # noqa: E402
    CameraFrameError,
    load_pair,
    stitch_labeled_pair,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="双摄夹取模型只预测测试：不连接 BLE，不发送任何运动指令。"
    )
    parser.add_argument("--model-url", default="http://192.168.55.33:8022")
    parser.add_argument("--target", default="红色小物块")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera-stream", metavar="URL")
    source.add_argument("--left", metavar="JPEG")
    parser.add_argument("--right", metavar="JPEG")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--save-stitched", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.left and not args.right:
        print("ERROR: 使用 --left 时必须同时提供 --right", file=sys.stderr)
        return 2
    report: dict[str, object] = {
        "schema": "grasp_model_dry_run_report.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "hardware_commands_sent": False,
        "model_url": args.model_url,
        "target": args.target,
    }
    try:
        left, right = load_pair(
            stream_base_url=args.camera_stream,
            left_path=args.left,
            right_path=args.right,
            timeout_s=min(args.timeout, 15.0),
        )
        stitched, encoded = stitch_labeled_pair(left, right)
        report["input"] = {
            "left_sha256": hashlib.sha256(left).hexdigest(),
            "right_sha256": hashlib.sha256(right).hexdigest(),
            "stitched_sha256": hashlib.sha256(stitched).hexdigest(),
            "stitched_size": [1280, 480],
        }
        if args.save_stitched:
            args.save_stitched.parent.mkdir(parents=True, exist_ok=True)
            args.save_stitched.write_bytes(stitched)
            report["stitched_path"] = str(args.save_stitched.resolve())
        client = ArmTeachModelClient(args.model_url, timeout_s=args.timeout)
        report["prediction"] = client.predict(encoded, args.target)
        report["ok"] = True
        report["gate"] = "PREDICTION_ONLY_PASSED"
    except (CameraFrameError, GraspModelError) as exc:
        report["ok"] = False
        report["gate"] = "PREDICTION_REJECTED"
        report["error"] = str(exc)

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

