from __future__ import annotations

from qwen_grounded_tracker.domain import LidarObservation


class NullLidarProvider:
    """Blank 2D LiDAR placeholder.

    Replace this class later with a serial/SDK implementation that returns real
    scan points and minimum obstacle distance. In the Windows validation config,
    lidar.require_ready is false so this placeholder does not block the demo.
    """

    def open(self) -> None:
        print("[LiDAR] null placeholder active; no scan data is produced")

    def read(self) -> LidarObservation:
        return LidarObservation(
            ready=False,
            obstacle=False,
            min_distance_m=None,
            status="placeholder: no 2D LiDAR data",
            points_xy=(),
        )

    def close(self) -> None:
        pass
