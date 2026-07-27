from __future__ import annotations

import cv2
import numpy as np

from qwen_grounded_tracker.domain import BBox


def _create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    raise RuntimeError(
        "OpenCV CSRT tracker is unavailable. Install opencv-contrib-python, "
        "then verify cv2.TrackerCSRT_create exists."
    )


class CSRTTargetTracker:
    def __init__(self) -> None:
        self.tracker = None
        self.initialized = False

    def initialize(self, frame: np.ndarray, bbox: BBox) -> None:
        height, width = frame.shape[:2]
        safe_box = bbox.clamp(width, height)
        self.tracker = _create_csrt_tracker()
        result = self.tracker.init(frame, safe_box.to_xywh())
        if result is False:
            raise RuntimeError("CSRT tracker initialization failed")
        self.initialized = True

    def update(self, frame: np.ndarray) -> tuple[bool, BBox | None]:
        if not self.initialized or self.tracker is None:
            return False, None

        ok, xywh = self.tracker.update(frame)
        if not ok:
            return False, None

        x, y, w, h = [float(value) for value in xywh]
        height, width = frame.shape[:2]
        bbox = BBox(x, y, x + w, y + h).clamp(width, height)
        if bbox.width < 3 or bbox.height < 3:
            return False, None
        return True, bbox

    def reset(self) -> None:
        self.tracker = None
        self.initialized = False
