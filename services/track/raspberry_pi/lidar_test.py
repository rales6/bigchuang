"""镭神 N10 串口和点云帧测试，不需要连接 ESP32。"""

import argparse
import math

import numpy as np

from raspberry_pi.config import LidarConfig
from raspberry_pi.lidar import N10LidarDriver


def build_argument_parser():
    parser = argparse.ArgumentParser(description="镭神 N10 串口点云测试")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--count", type=int, default=5, help="读取完整扫描圈数")
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--no-motor-control", action="store_true")
    return parser


def run(args):
    config = LidarConfig(
        port=args.port,
        angle_offset_deg=args.angle_offset,
        motor_control=not args.no_motor_control,
    )
    lidar = N10LidarDriver(config)
    try:
        for index, scan in enumerate(lidar.scans(), 1):
            valid = scan.distances_m[np.isfinite(scan.distances_m)]
            print(
                "扫描 {}/{}：点数={} 转速={:.2f}Hz 距离={:.2f}..{:.2f}m "
                "校验错误={} 格式错误={}".format(
                    index,
                    args.count,
                    len(scan.distances_m),
                    lidar.last_scan_frequency_hz,
                    float(valid.min()) if len(valid) else math.nan,
                    float(valid.max()) if len(valid) else math.nan,
                    lidar.parser.checksum_errors,
                    lidar.parser.format_errors,
                )
            )
            if index >= args.count:
                break
    finally:
        lidar.close()


def main():
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()

