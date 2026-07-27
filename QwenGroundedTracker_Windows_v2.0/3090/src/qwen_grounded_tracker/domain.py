from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clamp(self, frame_width: int, frame_height: int) -> "BBox":
        x1 = min(max(self.x1, 0.0), max(frame_width - 1, 0))
        y1 = min(max(self.y1, 0.0), max(frame_height - 1, 0))
        x2 = min(max(self.x2, x1 + 1.0), float(frame_width))
        y2 = min(max(self.y2, y1 + 1.0), float(frame_height))
        return BBox(x1, y1, x2, y2)

    def to_xywh(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            max(1, int(round(self.width))),
            max(1, int(round(self.height))),
        )

    def normalized_center(self, frame_width: int, frame_height: int) -> tuple[float, float]:
        cx, cy = self.center
        return (
            cx / max(frame_width, 1),
            cy / max(frame_height, 1),
        )

    def area_ratio(self, frame_width: int, frame_height: int) -> float:
        return self.area / max(frame_width * frame_height, 1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        box = self.clamp(width, height)
        x, y, w, h = box.to_xywh()
        return frame[y : y + h, x : x + w].copy()

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


@dataclass
class GroundingResult:
    found: bool
    bbox: BBox | None
    target_name: str
    confidence: float
    raw_text: str = ""
    message: str = ""


@dataclass
class TrackObservation:
    visible: bool
    logical_target_id: str | None
    bbox: BBox | None
    contour: np.ndarray | None
    identity_score: float
    status: str
    lost_seconds: float = 0.0


@dataclass
class ObstacleDetection:
    label: str
    confidence: float
    bbox: BBox
    contour: np.ndarray | None = None
    in_danger_zone: bool = False


@dataclass
class ObstacleObservation:
    detections: list[ObstacleDetection] = field(default_factory=list)
    danger: bool = False
    status: str = "disabled"


@dataclass
class LidarObservation:
    ready: bool
    obstacle: bool
    min_distance_m: float | None
    status: str
    points_xy: Sequence[tuple[float, float]] = field(default_factory=tuple)


@dataclass
class MotionGuidance:
    direction: str
    linear: float
    angular: float
    reason: str


@dataclass
class SafetyDecision:
    guidance: MotionGuidance
    blocked: bool
    reasons: list[str] = field(default_factory=list)
