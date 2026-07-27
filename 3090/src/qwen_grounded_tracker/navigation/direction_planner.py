from __future__ import annotations

from qwen_grounded_tracker.domain import BBox, MotionGuidance


class DirectionPlanner:
    def __init__(
        self,
        center_x: float = 0.5,
        center_tolerance: float = 0.07,
        stop_area_ratio: float = 0.16,
        too_close_area_ratio: float = 0.28,
        linear_speed: float = 0.08,
        backward_speed: float = 0.04,
        angular_speed: float = 0.22,
        enabled: bool = True,
    ) -> None:
        self.center_x = center_x
        self.center_tolerance = center_tolerance
        self.stop_area_ratio = stop_area_ratio
        self.too_close_area_ratio = too_close_area_ratio
        self.linear_speed = linear_speed
        self.backward_speed = backward_speed
        self.angular_speed = angular_speed
        self.enabled = enabled

    def plan(
        self,
        bbox: BBox | None,
        frame_width: int,
        frame_height: int,
    ) -> MotionGuidance:
        if not self.enabled:
            return MotionGuidance("STOP", 0.0, 0.0, "navigation disabled")
        if bbox is None:
            return MotionGuidance("STOP", 0.0, 0.0, "target not visible")

        center_x, _ = bbox.normalized_center(frame_width, frame_height)
        error_x = center_x - self.center_x
        area_ratio = bbox.area_ratio(frame_width, frame_height)

        if area_ratio >= self.too_close_area_ratio:
            return MotionGuidance(
                "BACKWARD",
                -self.backward_speed,
                0.0,
                f"target too close area={area_ratio:.3f}",
            )
        if abs(error_x) > self.center_tolerance:
            direction = "TURN_RIGHT" if error_x > 0 else "TURN_LEFT"
            angular = -self.angular_speed if error_x > 0 else self.angular_speed
            return MotionGuidance(
                direction,
                0.0,
                angular,
                f"horizontal error={error_x:.3f}",
            )
        if area_ratio >= self.stop_area_ratio:
            return MotionGuidance(
                "STOP",
                0.0,
                0.0,
                f"target reached area={area_ratio:.3f}",
            )
        return MotionGuidance(
            "FORWARD",
            self.linear_speed,
            0.0,
            f"target centered area={area_ratio:.3f}",
        )
