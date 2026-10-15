from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image, ImageDraw


class CameraFrameError(RuntimeError):
    pass


def read_mjpeg_frame(url: str, timeout_s: float = 12.0) -> bytes:
    """Read exactly one complete JPEG from an existing MJPEG stream."""
    try:
        with urllib_request.urlopen(url, timeout=timeout_s) as response:
            buffer = b""
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                chunk = response.read(4096)
                if not chunk:
                    break
                buffer += chunk
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    return buffer[start : end + 2]
                if len(buffer) > 4 * 1024 * 1024:
                    buffer = buffer[-1024 * 1024 :]
    except (OSError, urllib_error.URLError) as exc:
        raise CameraFrameError(f"相机流不可用 {url}: {exc}") from exc
    raise CameraFrameError(f"相机流未在 {timeout_s:.1f}s 内返回完整 JPEG: {url}")


def load_pair(
    stream_base_url: str | None = None,
    left_path: str | None = None,
    right_path: str | None = None,
    timeout_s: float = 12.0,
) -> tuple[bytes, bytes]:
    if stream_base_url:
        base = stream_base_url.rstrip("/")
        return (
            read_mjpeg_frame(base + "/video/left", timeout_s),
            read_mjpeg_frame(base + "/video/right", timeout_s),
        )
    if not left_path or not right_path:
        raise CameraFrameError("必须提供 stream_base_url，或同时提供 left_path/right_path")
    try:
        return Path(left_path).read_bytes(), Path(right_path).read_bytes()
    except OSError as exc:
        raise CameraFrameError(f"读取双摄图片失败: {exc}") from exc


def stitch_labeled_pair(left_jpeg: bytes, right_jpeg: bytes) -> tuple[bytes, str]:
    """Build the exact 1280x480 LEFT|RIGHT input expected by the frozen 8022 service."""
    frames: list[Image.Image] = []
    for data, label in ((left_jpeg, "LEFT"), (right_jpeg, "RIGHT")):
        if len(data) <= 1000:
            raise CameraFrameError(f"{label} 相机 JPEG 过小或为空")
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB").resize((640, 480))
        except (OSError, ValueError) as exc:
            raise CameraFrameError(f"{label} 相机 JPEG 解码失败: {exc}") from exc
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 125, 38), fill=(0, 0, 0))
        draw.text((10, 10), label, fill=(255, 255, 0))
        frames.append(image)
    stitched = Image.new("RGB", (1280, 480))
    stitched.paste(frames[0], (0, 0))
    stitched.paste(frames[1], (640, 0))
    output = io.BytesIO()
    stitched.save(output, format="JPEG", quality=85)
    jpeg = output.getvalue()
    return jpeg, base64.b64encode(jpeg).decode("ascii")

