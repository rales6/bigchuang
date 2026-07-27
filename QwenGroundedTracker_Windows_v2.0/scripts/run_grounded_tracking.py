from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.app import GroundedTrackingApp
from qwen_grounded_tracker.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL grounded target selection + CSRT continuous tracking demo"
    )
    parser.add_argument(
        "--config",
        default="configs/windows_cpu.yaml",
        help="YAML configuration path",
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Natural-language target instruction, e.g. 追踪右侧的蓝色水杯",
    )
    args = parser.parse_args()

    config = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    app = GroundedTrackingApp(config, args.command)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
