from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "any": cv2.CAP_ANY,
}


@dataclass
class OpenCVCamera:
    source: int | str
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "dshow"
    side_by_side: str = "auto"

    def __post_init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None
        self.raw_size: tuple[int, int] | None = None

    def open(self) -> None:
        backend_id = _BACKENDS.get(self.backend.lower(), cv2.CAP_ANY)
        self.capture = cv2.VideoCapture(self.source, backend_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> np.ndarray | None:
        if self.capture is None:
            raise RuntimeError("Camera is not open")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None

        height, width = frame.shape[:2]
        if self.raw_size is None:
            self.raw_size = (width, height)
            print(f"[Camera] raw frame size: {width}x{height}")

        mode = self.side_by_side.lower()
        looks_side_by_side = width >= height * 2.2
        if mode == "left" or (mode == "auto" and looks_side_by_side):
            frame = frame[:, : width // 2]
        elif mode == "right":
            frame = frame[:, width // 2 :]

        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
