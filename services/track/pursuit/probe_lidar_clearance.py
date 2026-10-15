"""Read fresh observable lidar sectors; never issue a chassis command."""

from __future__ import annotations

import json
import math
import time

from pursuit.hull_micro_adjust import ScanGuard
from pursuit.pursuit_config import PursuitConfig
from raspberry_pi.mapping.scan_filter import sector_min_distance


def main() -> int:
    guard = ScanGuard()
    try:
        guard.wait_ready(timeout_s=5.0)
        scan = guard.latest
        if scan is None or time.monotonic() - scan.timestamp_s > 0.5:
            raise RuntimeError("雷达扫描过期")
        cfg = PursuitConfig()
        front = sector_min_distance(
            scan, center_deg=0.0, width_deg=30.0,
            min_distance_m=0.02, max_distance_m=cfg.lidar_max_range_m)
        rear = sector_min_distance(
            scan, center_deg=180.0,
            width_deg=2.0 * cfg.recovery_rear_half_width_deg,
            min_distance_m=0.02, max_distance_m=cfg.lidar_max_range_m
        ) if cfg.recovery_rear_sensor_usable else float("nan")
        front_ready = math.isfinite(front)
        print(json.dumps({
            "front_m": round(front, 3) if math.isfinite(front) else None,
            "rear_m": round(rear, 3) if math.isfinite(rear) else None,
            "rear_sensor_usable": cfg.recovery_rear_sensor_usable,
            "reverse_required_clear_m": cfg.recovery_rear_clear_m,
            "reverse_allowed": math.isfinite(rear) and rear >= cfg.recovery_rear_clear_m,
            "reverse_policy": "rear_clearance" if cfg.recovery_rear_sensor_usable
                              else "budgeted_recent_path_only",
            "scan_age_s": round(time.monotonic() - scan.timestamp_s, 3),
        }, ensure_ascii=False))
        return 0 if front_ready else 2
    finally:
        guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
