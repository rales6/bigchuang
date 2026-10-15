"""Qwen grounding client used by the lightweight 8021 pursuit service."""

from __future__ import annotations

import json
import uuid
from urllib import request as urlrequest

import cv2
import numpy as np

from qwen_grounded_tracker.domain import BBox, GroundingResult
try:
    from qwen_grounded_tracker.domain import GroundedCandidate
except ImportError:  # older/stub domain used by the pursuit web-page unit test
    GroundedCandidate = None  # type: ignore[assignment,misc]


class RemoteQwenGrounder:
    def __init__(self, hub_url: str, timeout_s: float = 90.0) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.timeout_s = timeout_s

    def ground(
        self,
        frame: np.ndarray,
        instruction: str,
        reference_image: np.ndarray | None = None,
    ) -> GroundingResult:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Qwen Hub 请求图像编码失败")
        parts: list[tuple[str, str, bytes, str]] = [
            ("image", "scene.jpg", encoded.tobytes(), "image/jpeg")
        ]
        if reference_image is not None and reference_image.size:
            ref_ok, ref_encoded = cv2.imencode(
                ".jpg", reference_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if ref_ok:
                parts.append(("reference", "reference.jpg", ref_encoded.tobytes(), "image/jpeg"))
        boundary = uuid.uuid4().hex
        body = bytearray()
        for name, filename, data, content_type in parts:
            body += (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
            body += data
            body += b"\r\n"
        body += (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="instruction"\r\n\r\n'
            f"{instruction}\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        request = urlrequest.Request(
            f"{self.hub_url}/agent/ground",
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urlrequest.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise RuntimeError(str(payload.get("error") if isinstance(payload, dict) else payload))
        bbox_value = payload.get("bbox")
        bbox = BBox(*[float(value) for value in bbox_value]) if isinstance(bbox_value, list) and len(bbox_value) == 4 else None
        candidates = []
        for item in payload.get("candidates", []):
            item_bbox = item.get("bbox") if isinstance(item, dict) else None
            if not isinstance(item_bbox, list) or len(item_bbox) != 4:
                continue
            if GroundedCandidate is None:
                continue
            candidates.append(GroundedCandidate(
                target_name=str(item.get("target_name") or ""),
                bbox=BBox(*[float(value) for value in item_bbox]),
                confidence=float(item.get("confidence") or 0.0),
                camera_view=str(item.get("camera_view") or ""),
            ))
        return GroundingResult(
            found=bool(payload.get("found")),
            bbox=bbox,
            target_name=str(payload.get("target_name") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            raw_text=str(payload.get("raw_text") or ""),
            message=str(payload.get("message") or ""),
            candidates=candidates,
        )
