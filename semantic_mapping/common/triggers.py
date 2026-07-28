from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass
class LandmarkTriggerConfig:
    periodic_s: float = 5.0
    min_landmarks: int = 2
    rejected_frames: int = 3
    poor_rmse_m: float = 0.12
    yaw_change_deg: float = 35.0
    min_interval_s: float = 1.5


class LandmarkTriggerState:
    """决定何时低频调用 Qwen 选择自然参考点。"""

    def __init__(self, config: LandmarkTriggerConfig | None = None) -> None:
        self.config = config or LandmarkTriggerConfig()
        self._started = False
        self._last_trigger_s = 0.0
        self._last_trigger_yaw = None
        self._consecutive_rejected = 0

    def should_trigger(self, update, pose, landmark_count: int) -> tuple[bool, str]:
        now = time.monotonic()
        if now - self._last_trigger_s < self.config.min_interval_s:
            return False, ""

        if not self._started:
            self._started = True
            self._mark(now, pose.yaw_rad)
            return True, "startup"

        accepted = bool(getattr(update, "accepted", False))
        rmse = float(getattr(update, "rmse_m", math.inf))
        if accepted:
            self._consecutive_rejected = 0
        else:
            self._consecutive_rejected += 1

        if self._consecutive_rejected >= self.config.rejected_frames:
            self._mark(now, pose.yaw_rad)
            return True, "lidar_rejected"

        if math.isfinite(rmse) and rmse >= self.config.poor_rmse_m:
            self._mark(now, pose.yaw_rad)
            return True, "lidar_rmse_poor"

        if self._last_trigger_yaw is not None:
            yaw_delta = abs(_normalize_angle(pose.yaw_rad - self._last_trigger_yaw))
            if math.degrees(yaw_delta) >= self.config.yaw_change_deg:
                self._mark(now, pose.yaw_rad)
                return True, "after_turn"

        if landmark_count < self.config.min_landmarks:
            self._mark(now, pose.yaw_rad)
            return True, "too_few_landmarks"

        if now - self._last_trigger_s >= self.config.periodic_s:
            self._mark(now, pose.yaw_rad)
            return True, "periodic"

        return False, ""

    def _mark(self, now: float, yaw_rad: float) -> None:
        self._last_trigger_s = now
        self._last_trigger_yaw = yaw_rad


def _normalize_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
