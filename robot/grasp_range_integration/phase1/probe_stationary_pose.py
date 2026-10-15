"""Read-only N10 lidar pose check; this module never opens the motor link."""

from __future__ import annotations

import argparse
import json
import math

from raspberry_pi.config import LidarConfig
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.mapping import Pose2D


def _row(result):
    if result is None:
        return None
    return {
        "x_m": round(result.pose.x_m, 3),
        "y_m": round(result.pose.y_m, 3),
        "yaw_deg": round(math.degrees(result.pose.yaw_rad), 1),
        "rmse_m": round(result.rmse_m, 3),
        "inlier_ratio": round(result.inlier_ratio, 3),
        "reliable": bool(result.reliable),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, required=True)
    args = parser.parse_args()
    known_map = KnownOccupancyMap.load(args.map, args.map_yaml, clearance_m=0.30)
    localizer = ScanMapLocalizer(known_map, lidar_offset_x_m=0.14, lidar_offset_y_m=0.0)
    expected = Pose2D(args.x, args.y, math.radians(args.yaw_deg))
    lidar = N10LidarDriver(LidarConfig(port="/dev/ttyACM0", baudrate=230400))
    try:
        scans = iter(lidar.scans())
        first = next(scans)
        second = next(scans)
        third = next(scans)
        rows = []
        for scan in (first, second, third):
            rows.append({
                "near_last_pose": _row(localizer.localize_near_position(
                    scan, expected, radius_m=0.35
                )),
                "global": _row(localizer.global_localize(scan, hint=None)),
            })
        motion = localizer.relative_motion(first, third)
        print(json.dumps({
            "motor_link_created": False,
            "last_reported_pose": [args.x, args.y, args.yaw_deg],
            "scan_results": rows,
            "first_to_third_motion": {
                "translation_m": round(math.hypot(
                    motion.transform.x_m, motion.transform.y_m
                ), 4),
                "yaw_deg": round(math.degrees(motion.transform.yaw_rad), 2),
                "inlier_ratio": round(motion.inlier_ratio, 3),
            },
        }, ensure_ascii=False, indent=2))
    finally:
        lidar.close()


if __name__ == "__main__":
    main()
