from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qwen_grounded_tracker.domain import BBox


@dataclass
class IdentityEvaluation:
    score: float
    accepted: bool
    bad_frames: int
    status: str


class IdentityGuard:
    """Detect obvious tracker drift with color, area, and aspect continuity."""

    def __init__(
        self,
        minimum_score: float = 0.32,
        maximum_bad_frames: int = 4,
        reference_update_score: float = 0.78,
    ) -> None:
        self.minimum_score = minimum_score
        self.maximum_bad_frames = maximum_bad_frames
        self.reference_update_score = reference_update_score
        self.reference_hist: np.ndarray | None = None
        self.reference_area: float | None = None
        self.reference_aspect: float | None = None
        self.bad_frames = 0

    @staticmethod
    def _histogram(frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        crop = bbox.crop(frame)
        if crop.size == 0 or min(crop.shape[:2]) < 3:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mask = np.where(saturation > 25, 255, 0).astype(np.uint8)
        if cv2.countNonZero(mask) < mask.size * 0.05:
            mask = None

        hist = cv2.calcHist([hsv], [0, 1], mask, [30, 16], [0, 180, 0, 256])
        if hist is None or float(hist.sum()) <= 0.0:
            return None
        cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
        return hist

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        self.reference_hist = self._histogram(frame, bbox)
        self.reference_area = max(bbox.area, 1.0)
        self.reference_aspect = bbox.width / max(bbox.height, 1.0)
        self.bad_frames = 0

    def evaluate(self, frame: np.ndarray, bbox: BBox) -> IdentityEvaluation:
        if self.reference_area is None or self.reference_aspect is None:
            self.initialize(frame, bbox)
            return IdentityEvaluation(1.0, True, 0, "identity initialized")

        current_hist = self._histogram(frame, bbox)
        if self.reference_hist is None or current_hist is None:
            histogram_score = 0.5
        else:
            distance = cv2.compareHist(
                self.reference_hist,
                current_hist,
                cv2.HISTCMP_BHATTACHARYYA,
            )
            histogram_score = max(0.0, min(1.0, 1.0 - float(distance)))

        area_ratio = bbox.area / max(self.reference_area, 1.0)
        area_score = min(area_ratio, 1.0 / max(area_ratio, 1e-6))

        aspect = bbox.width / max(bbox.height, 1.0)
        aspect_ratio = aspect / max(self.reference_aspect, 1e-6)
        aspect_score = min(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6))

        score = 0.65 * histogram_score + 0.20 * area_score + 0.15 * aspect_score
        accepted_this_frame = score >= self.minimum_score
        self.bad_frames = 0 if accepted_this_frame else self.bad_frames + 1
        accepted = self.bad_frames < self.maximum_bad_frames

        if score >= self.reference_update_score and current_hist is not None:
            if self.reference_hist is None:
                self.reference_hist = current_hist
            else:
                self.reference_hist = cv2.addWeighted(
                    self.reference_hist,
                    0.97,
                    current_hist,
                    0.03,
                    0.0,
                )
            self.reference_area = 0.98 * self.reference_area + 0.02 * bbox.area
            self.reference_aspect = 0.98 * self.reference_aspect + 0.02 * aspect

        status = (
            f"identity score={score:.2f}"
            if accepted_this_frame
            else f"identity uncertain {self.bad_frames}/{self.maximum_bad_frames} score={score:.2f}"
        )
        return IdentityEvaluation(score, accepted, self.bad_frames, status)

    def reset(self) -> None:
        self.reference_hist = None
        self.reference_area = None
        self.reference_aspect = None
        self.bad_frames = 0
