from __future__ import annotations

from qwen_grounded_tracker.domain import (
    LidarObservation,
    MotionGuidance,
    ObstacleObservation,
    SafetyDecision,
)


class SafetyArbiter:
    def __init__(
        self,
        require_lidar_ready: bool = False,
        stop_on_tracking_loss: bool = True,
        stop_on_yolo_obstacle: bool = True,
    ) -> None:
        self.require_lidar_ready = require_lidar_ready
        self.stop_on_tracking_loss = stop_on_tracking_loss
        self.stop_on_yolo_obstacle = stop_on_yolo_obstacle

    @staticmethod
    def _stop(reason: str) -> MotionGuidance:
        return MotionGuidance("STOP", 0.0, 0.0, reason)

    def decide(
        self,
        requested: MotionGuidance,
        tracking_visible: bool,
        yolo_obstacles: ObstacleObservation,
        lidar: LidarObservation,
        emergency_stop: bool,
    ) -> SafetyDecision:
        reasons: list[str] = []
        if emergency_stop:
            reasons.append("manual emergency stop")
        if self.stop_on_tracking_loss and not tracking_visible:
            reasons.append("target not safely tracked")
        if self.stop_on_yolo_obstacle and yolo_obstacles.danger:
            reasons.append("YOLO semantic obstacle in danger zone")
        if self.require_lidar_ready and not lidar.ready:
            reasons.append("2D LiDAR not ready")
        if lidar.obstacle:
            distance = "unknown" if lidar.min_distance_m is None else f"{lidar.min_distance_m:.2f}m"
            reasons.append(f"2D LiDAR obstacle at {distance}")

        if reasons:
            return SafetyDecision(
                guidance=self._stop("; ".join(reasons)),
                blocked=True,
                reasons=reasons,
            )
        return SafetyDecision(guidance=requested, blocked=False, reasons=[])
