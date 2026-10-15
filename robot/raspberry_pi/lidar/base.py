"""与具体雷达品牌无关的数据结构。"""

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Protocol

import numpy as np


@dataclass(frozen=True)
class LaserScan:
    """一圈二维扫描，角度为弧度、距离为米。"""

    angles_rad: np.ndarray
    distances_m: np.ndarray
    timestamp_s: float
    # The web simulator may attach pose truth for offline evaluation. Mapping
    # and localization never consume this field.
    ground_truth_pose: Any = None

    def __post_init__(self):
        if self.angles_rad.ndim != 1 or self.distances_m.ndim != 1:
            raise ValueError("scan arrays must be one-dimensional")
        if self.angles_rad.shape != self.distances_m.shape:
            raise ValueError("angles and distances must have equal lengths")

    def points(self, min_distance_m=0.12, max_distance_m=8.0):
        valid = (
            np.isfinite(self.angles_rad)
            & np.isfinite(self.distances_m)
            & (self.distances_m >= min_distance_m)
            & (self.distances_m <= max_distance_m)
        )
        angles = self.angles_rad[valid]
        distances = self.distances_m[valid]
        return np.column_stack((distances * np.cos(angles), distances * np.sin(angles)))


class Lidar(Protocol):
    def scans(self) -> Iterator[LaserScan]:
        ...

    def close(self) -> None:
        ...


def make_scan(samples: Iterable[tuple], timestamp_s: float) -> LaserScan:
    """由 ``(angle_rad, distance_m)`` 样本构造扫描，方便自定义驱动。"""
    values = np.asarray(list(samples), dtype=np.float64)
    if values.size == 0:
        values = np.empty((0, 2), dtype=np.float64)
    return LaserScan(values[:, 0], values[:, 1], timestamp_s)
