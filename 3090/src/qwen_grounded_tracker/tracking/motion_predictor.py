from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from qwen_grounded_tracker.domain import BBox, TrackObservation


@dataclass
class _Sample:
    timestamp: float
    bbox: BBox
    contour: np.ndarray | None
    identity_score: float
    logical_target_id: str | None


class MotionPredictor:
    """Short-horizon bbox/contour prediction for smoother rendering and control."""

    def __init__(
        self,
        enabled: bool = True,
        history_size: int = 5,
        max_prediction_seconds: float = 0.35,
        velocity_damping: float = 0.75,
        max_velocity_px_per_second: float = 900.0,
        max_size_change_px_per_second: float = 500.0,
    ) -> None:
        self.enabled = enabled
        self.history: Deque[_Sample] = deque(maxlen=max(2, int(history_size)))
        self.max_prediction_seconds = max(0.0, float(max_prediction_seconds))
        self.velocity_damping = max(0.0, min(float(velocity_damping), 1.0))
        self.max_velocity_px_per_second = max(1.0, float(max_velocity_px_per_second))
        self.max_size_change_px_per_second = max(1.0, float(max_size_change_px_per_second))
        self.last_target_id: str | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_target_id = None

    def observe(self, track: TrackObservation, timestamp: float) -> None:
        if not self.enabled:
            return
        if not track.visible or track.bbox is None:
            return
        if track.logical_target_id != self.last_target_id:
            self.history.clear()
            self.last_target_id = track.logical_target_id
        contour = None if track.contour is None else track.contour.copy()
        self.history.append(
            _Sample(
                timestamp=timestamp,
                bbox=track.bbox,
                contour=contour,
                identity_score=track.identity_score,
                logical_target_id=track.logical_target_id,
            )
        )

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _remap_contour(contour: np.ndarray, old_box: BBox, new_box: BBox) -> np.ndarray:
        points = contour.reshape(-1, 2).astype(np.float32)
        nx = (points[:, 0] - old_box.x1) / max(old_box.width, 1.0)
        ny = (points[:, 1] - old_box.y1) / max(old_box.height, 1.0)
        points[:, 0] = new_box.x1 + nx * new_box.width
        points[:, 1] = new_box.y1 + ny * new_box.height
        return np.rint(points).astype(np.int32).reshape(-1, 1, 2)

    def _velocity(self) -> tuple[float, float, float, float]:
        if len(self.history) < 2:
            return 0.0, 0.0, 0.0, 0.0
        first = self.history[0]
        last = self.history[-1]
        elapsed = max(last.timestamp - first.timestamp, 1e-3)
        fx, fy = first.bbox.center
        lx, ly = last.bbox.center
        vx = self._clip((lx - fx) / elapsed, self.max_velocity_px_per_second)
        vy = self._clip((ly - fy) / elapsed, self.max_velocity_px_per_second)
        vw = self._clip(
            (last.bbox.width - first.bbox.width) / elapsed,
            self.max_size_change_px_per_second,
        )
        vh = self._clip(
            (last.bbox.height - first.bbox.height) / elapsed,
            self.max_size_change_px_per_second,
        )
        return (
            vx * self.velocity_damping,
            vy * self.velocity_damping,
            vw * self.velocity_damping,
            vh * self.velocity_damping,
        )

    def predict(
        self,
        timestamp: float,
        frame_width: int,
        frame_height: int,
        fallback: TrackObservation,
    ) -> tuple[TrackObservation, dict[str, float | bool | str]]:
        if not self.enabled or not self.history:
            return fallback, {"enabled": self.enabled, "active": False, "age_seconds": 0.0}

        last = self.history[-1]
        age = max(0.0, timestamp - last.timestamp)
        if age > self.max_prediction_seconds:
            return fallback, {
                "enabled": self.enabled,
                "active": False,
                "age_seconds": age,
                "reason": "prediction expired",
            }

        vx, vy, vw, vh = self._velocity()
        cx, cy = last.bbox.center
        width = max(3.0, last.bbox.width + vw * age)
        height = max(3.0, last.bbox.height + vh * age)
        predicted_box = BBox(
            cx + vx * age - width / 2.0,
            cy + vy * age - height / 2.0,
            cx + vx * age + width / 2.0,
            cy + vy * age + height / 2.0,
        ).clamp(frame_width, frame_height)

        contour = None
        if last.contour is not None:
            contour = self._remap_contour(last.contour, last.bbox, predicted_box)

        active = age > 1e-3
        status = "predicted motion" if active else fallback.status
        predicted = TrackObservation(
            visible=True,
            logical_target_id=last.logical_target_id,
            bbox=predicted_box,
            contour=contour,
            identity_score=last.identity_score,
            status=status,
            lost_seconds=0.0,
        )
        return predicted, {
            "enabled": self.enabled,
            "active": active,
            "age_seconds": age,
            "vx": vx,
            "vy": vy,
            "vw": vw,
            "vh": vh,
        }
