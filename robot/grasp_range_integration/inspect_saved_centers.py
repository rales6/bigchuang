from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grasp_range_integration.adapters.arm_teach_model import (  # noqa: E402
    ArmTeachModelClient,
    GraspModelError,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="只检测已保存双摄拼接图的目标中心。")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-url", default="http://192.168.55.33:8022")
    parser.add_argument("--target", default="红色小物块")
    args = parser.parse_args()
    try:
        encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
        result = ArmTeachModelClient(args.model_url).detect_centers(encoded, args.target)
        result.update({"ok": True, "hardware_commands_sent": False})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, GraspModelError) as exc:
        print(json.dumps({
            "ok": False,
            "hardware_commands_sent": False,
            "error": str(exc),
        }, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
