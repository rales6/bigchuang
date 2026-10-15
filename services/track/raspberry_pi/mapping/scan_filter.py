"""按车体坐标裁剪雷达视场。

角度约定：0 指向车头，正方向指向车体左侧。
"""

import math

import numpy as np

from raspberry_pi.lidar.base import LaserScan


def normalize_angles(angles_rad):
    angles = np.asarray(angles_rad, dtype=np.float64)
    return (angles + math.pi) % (2.0 * math.pi) - math.pi


def filter_scan_sector(scan, center_deg=0.0, field_of_view_deg=180.0):
    """只保留指定扇区，跨越 ±180° 边界时也能正确过滤。"""
    if not 0.0 < field_of_view_deg <= 360.0:
        raise ValueError("field_of_view_deg must be in range (0, 360]")
    if field_of_view_deg >= 360.0:
        return scan
    center = math.radians(center_deg)
    half_width = math.radians(field_of_view_deg) / 2.0
    relative = normalize_angles(scan.angles_rad - center)
    selected = np.abs(relative) <= half_width
    return LaserScan(
        scan.angles_rad[selected],
        scan.distances_m[selected],
        scan.timestamp_s,
    )


def sector_min_distance(scan, center_deg=0.0, width_deg=30.0,
                        min_distance_m=0.05, max_distance_m=20.0):
    """返回指定方向扇区内的最近有效距离；没有有效点时返回无穷大。"""
    sector = filter_scan_sector(scan, center_deg, width_deg)
    valid = (
        np.isfinite(sector.distances_m)
        & (sector.distances_m >= min_distance_m)
        & (sector.distances_m <= max_distance_m)
    )
    if not np.any(valid):
        return math.inf
    return float(np.min(sector.distances_m[valid]))
