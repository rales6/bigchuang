"""One bounded in-place right-turn pulse with lidar feedback; no route motion."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from raspberry_pi.config import LidarConfig, SerialConfig
from raspberry_pi.esp32 import Esp32Client
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar import N10LidarDriver


def _turn_clearance_mask(scan):
    distances = scan.distances_m
    finite = np.isfinite(distances) & (distances >= 0.10)
    # N10's clipped 0.104-0.105 m return appears on dozens of widely separated
    # rear/side rays from the chassis in this installation. Treat only that
    # exact repeated code as self-return; retain every other close distance.
    clipped = finite & (distances <= 0.110)
    angles = np.degrees(scan.angles_rad[clipped])
    self_return = len(angles) >= 15 and float(np.ptp(angles)) >= 60.0
    mask = finite & (~clipped if self_return else True)
    if self_return:
        # The lone ~0.19 m return amid the rear chassis clipping cluster is
        # also inside the measured robot footprint. Do not apply this mask to
        # side/front returns, or without the characteristic clipping cluster.
        bearings = (np.degrees(scan.angles_rad) + 180.0) % 360.0 - 180.0
        rear_chassis = (np.abs(bearings) >= 150.0) & (distances < 0.25)
        # A single 0.118-0.135 m return also wanders around the fixed
        # left-hand chassis edge at -92 degrees. This is inside the lidar-to-
        # body footprint, not the exterior turn clearance. Keep the sector
        # narrow and never suppress a return >=0.20 m there.
        side_chassis = (bearings >= -100.0) & (bearings <= -85.0) & (distances < 0.20)
        mask &= ~(rear_chassis | side_chassis)
    return mask, int(np.count_nonzero(clipped))


def _turn_clearance(scan) -> tuple[float, int]:
    mask, clipped_count = _turn_clearance_mask(scan)
    usable = scan.distances_m[mask]
    return (float(np.min(usable)) if len(usable) else 0.0, clipped_count)


def main() -> int:
    parser = argparse.ArgumentParser(description="单次有限时原地右转测量")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--inspect-clearance", action="store_true")
    parser.add_argument("--refresh-seconds", type=float, default=0.45)
    parser.add_argument("--map", required=True)
    parser.add_argument("--map-yaml", required=True)
    args = parser.parse_args()
    if not 0.35 <= args.refresh_seconds <= 1.20:
        parser.error("--refresh-seconds must be between 0.35 and 1.20")
    if not args.enable_motion and not args.inspect_clearance:
        print("未给出 --enable-motion；不会连接底盘或雷达。")
        return 0

    known_map = KnownOccupancyMap.load(args.map, args.map_yaml, clearance_m=0.30)
    localizer = ScanMapLocalizer(known_map, lidar_offset_x_m=0.14, lidar_offset_y_m=0.0)
    lidar = N10LidarDriver(LidarConfig(port="/dev/ttyACM0", baudrate=230400))
    if args.inspect_clearance:
        try:
            before = next(iter(lidar.scans()))
            valid = np.isfinite(before.distances_m) & (before.distances_m >= 0.10)
            distances = before.distances_m[valid]
            angles = np.degrees(before.angles_rad[valid])
            close = np.flatnonzero(distances < 0.45)
            print(
                "雷达净空分布：样本={} <0.45m={} p01={:.3f}m "
                "p05={:.3f}m p10={:.3f}m".format(
                    len(distances), len(close),
                    float(np.percentile(distances, 1)),
                    float(np.percentile(distances, 5)),
                    float(np.percentile(distances, 10)),
                )
            )
            print("最近回波(角度°,距离m)：" + str([
                (round(float(angles[index]), 1), round(float(distances[index]), 3))
                for index in np.argsort(distances)[:20]
            ]))
            clearance, clipped_count = _turn_clearance(before)
            print(f"排除重复车体截断回波后：净空={clearance:.3f}m 截断数={clipped_count}")
            remaining = np.flatnonzero(valid & (before.distances_m > 0.110))
            remaining = remaining[np.argsort(before.distances_m[remaining])[:15]]
            print("有效最近回波(角度°,距离m)：" + str([
                (round(float(np.degrees(before.angles_rad[index])), 1),
                 round(float(before.distances_m[index]), 3))
                for index in remaining
            ]))
            return 0
        finally:
            lidar.close()
    client = Esp32Client(SerialConfig(
        link_mode="ble", ble_device_name="ESP32-Robot-Car",
        ble_address="AA:BB:CC:DD:EE:01",
    ))
    try:
        client.heartbeat()
        client.stop()
        client.start()
        scans = iter(lidar.scans())
        before = next(scans)
        clearance, clipped_count = _turn_clearance(before)
        second = next(scans)
        second_clearance, _ = _turn_clearance(second)
        clearance = min(clearance, second_clearance)
        before = second
        print(
            f"360度有效最小雷达间距：{clearance:.3f}m"
            f"（重复车体截断回波{clipped_count}个）",
            flush=True,
        )
        if clearance < 0.45:
            raise RuntimeError("原地转向净空不足 0.45m，未发送运动命令")
        issued_at = time.monotonic()
        deadline = issued_at + args.refresh_seconds
        try:
            client.set_bounded_twist(
                0, -1500, ttl_ms=350, refresh_deadline_s=deadline
            )
            ack_seconds = time.monotonic() - issued_at
            print(f"转向命令 ACK 耗时：{ack_seconds:.3f}s，刷新硬截止 {args.refresh_seconds:.2f}s", flush=True)
            time.sleep(max(0.0, deadline + 0.40 - time.monotonic()))
        finally:
            client.stop()
        after = next(scans)
        match = localizer.relative_motion(before, after)
        print(
            "短右转结果：converged={} yaw={:+.2f}° translation={:.3f}m "
            "rmse={:.3f}m inliers={:.1%}".format(
                match.converged, math.degrees(match.transform.yaw_rad),
                math.hypot(match.transform.x_m, match.transform.y_m),
                match.rmse_m, match.inlier_ratio,
            ),
            flush=True,
        )
        return 0
    finally:
        try:
            client.stop()
        finally:
            lidar.close()
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
