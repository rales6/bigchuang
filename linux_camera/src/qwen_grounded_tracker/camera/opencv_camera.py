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


def _fourcc_to_string(value: float) -> str:
    code = int(value)
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(ch if ch.isprintable() else "?" for ch in chars)


@dataclass
class OpenCVCamera:
    source: int | str
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "dshow"
    side_by_side: str = "auto"
    buffer_size: int = 1
    fourcc: str = ""

    def __post_init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None
        self.raw_size: tuple[int, int] | None = None

    def open(self) -> None:
        backend_id = _BACKENDS.get(self.backend.lower(), cv2.CAP_ANY)
        self.capture = cv2.VideoCapture(self.source, backend_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")

        # V4L2 有些摄像头要求先切换压缩格式，再设置分辨率/FPS，
        # 否则会静默回退到较慢的默认格式或默认帧率。
        if self.fourcc:
            code = self.fourcc.upper()[:4].ljust(4)
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)
        if self.buffer_size > 0:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        self._print_effective_settings()

    def _print_effective_settings(self) -> None:
        if self.capture is None:
            return
        width = int(round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        fourcc = _fourcc_to_string(self.capture.get(cv2.CAP_PROP_FOURCC))
        requested_fourcc = self.fourcc.upper() if self.fourcc else "default"
        print(
            "[Camera] requested "
            f"{self.width}x{self.height}@{self.fps} fourcc={requested_fourcc}; "
            f"effective {width}x{height}@{fps:.1f} fourcc={fourcc}"
        )

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

    def read_latest(self, drop_frames: int = 0) -> np.ndarray | None:
        if self.capture is None:
            raise RuntimeError("Camera is not open")

        # Some V4L2 cameras buffer old frames while the 3090 is processing.
        # Grab and discard a few queued frames so control uses the freshest image.
        for _ in range(max(0, int(drop_frames))):
            if not self.capture.grab():
                break
        return self.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
