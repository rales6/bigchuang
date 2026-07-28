from __future__ import annotations

import cv2


class CameraSampler:
    """轻量摄像头采样器，只在需要 Qwen 判断时抓一帧。"""

    def __init__(
        self,
        source: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        backend: str = "v4l2",
        fourcc: str = "MJPG",
        jpeg_quality: int = 65,
    ) -> None:
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.backend = backend
        self.fourcc = fourcc
        self.jpeg_quality = int(jpeg_quality)
        self.cap = None

    def open(self) -> None:
        api = cv2.CAP_V4L2 if self.backend.lower() == "v4l2" else 0
        try:
            source = int(self.source)
        except ValueError:
            source = self.source
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc[:4]))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def snapshot_jpeg(self) -> bytes:
        if self.cap is None:
            self.open()
        assert self.cap is not None
        # 丢弃旧帧，减少视觉语义和雷达状态之间的时间差。
        for _ in range(3):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera read failed")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return bytes(encoded)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
