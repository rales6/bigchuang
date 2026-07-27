"""用网页虚拟硬件运行现有 Python 脚本。

示例：
    python -m car_sim.run_with_simulator raspberry_pi/mapping_app.py \
        --output car_sim/output/my_map --max-scans 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

from .virtual_hardware import SimulatedEsp32Client, SimulatedLidarDriver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将目标脚本的 ESP32 与 N10 雷达临时替换为网页仿真器"
    )
    parser.add_argument("script", help="要运行的 Python 脚本路径")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="原样传递给目标脚本的参数",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    script = Path(args.script).resolve()
    if not script.is_file():
        raise SystemExit(f"目标脚本不存在：{script}")

    import raspberry_pi.esp32 as esp32_package
    import raspberry_pi.esp32.client as esp32_module
    import raspberry_pi.lidar as lidar_package
    import raspberry_pi.lidar.n10_driver as lidar_module

    esp32_package.Esp32Client = SimulatedEsp32Client
    esp32_module.Esp32Client = SimulatedEsp32Client
    lidar_package.N10LidarDriver = SimulatedLidarDriver
    lidar_module.N10LidarDriver = SimulatedLidarDriver

    sys.argv = [str(script), *args.script_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
