from __future__ import annotations

import json
from typing import Any


class QwenLandmarkClient:
    """树莓派侧调用 3090 Qwen 语义参考点服务。"""

    def __init__(self, url: str, timeout_s: float = 45.0) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)

    def detect(
        self,
        *,
        camera_jpeg: bytes,
        lidar_png: bytes,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        import requests

        files = {
            "camera_image": ("camera.jpg", camera_jpeg, "image/jpeg"),
            "lidar_image": ("lidar.png", lidar_png, "image/png"),
        }
        data = {"context_json": json.dumps(context, ensure_ascii=False)}
        response = requests.post(
            self.url,
            files=files,
            data=data,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()
