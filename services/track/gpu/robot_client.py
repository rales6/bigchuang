"""HTTP client and guarded task-step dispatcher for the Pi bridge."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator
from urllib import request as urlrequest


class RobotClient:
    def __init__(self) -> None:
        # The 3090 subnet cannot route to the Pi subnet in the deployed lab.
        # The Pi therefore maintains a loopback-only reverse SSH tunnel.
        self.base_url = os.environ.get("FINAL_WORK_PI_URL", "http://127.0.0.1:18780").rstrip("/")
        self.safety_token = os.environ.get("FINAL_WORK_HARDWARE_TOKEN", "")

    def request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.safety_token:
            headers["X-Robot-Safety-Token"] = self.safety_token
        req = urlrequest.Request(
            f"{self.base_url}{path}",
            data=data,
            method="GET" if payload is None else "POST",
            headers=headers,
        )
        with urlrequest.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        return value if isinstance(value, dict) else {"value": value}

    def stream(self, path: str) -> tuple[str, Iterator[bytes]]:
        response = urlrequest.urlopen(f"{self.base_url}{path}", timeout=15)
        content_type = response.headers.get("Content-Type", "application/octet-stream")

        def chunks() -> Iterator[bytes]:
            try:
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        return
                    yield block
            finally:
                response.close()

        return content_type, chunks()

    def capture_frame(self, camera: str, *, timeout: float = 12.0) -> bytes:
        if camera not in {"left", "right"}:
            raise ValueError("摄像头名称无效")
        response = urlrequest.urlopen(f"{self.base_url}/video/{camera}", timeout=timeout)
        buffer = bytearray()
        try:
            while len(buffer) < 8 * 1024 * 1024:
                block = response.read(64 * 1024)
                if not block:
                    break
                buffer.extend(block)
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    continue
                end = buffer.find(b"\xff\xd9", start + 2)
                if end >= 0:
                    frame = bytes(buffer[start:end + 2])
                    if len(frame) < 1000:
                        raise RuntimeError("摄像头返回的 JPEG 帧过小")
                    return frame
        finally:
            response.close()
        raise RuntimeError(f"未能从 {camera} 摄像头取得完整 JPEG 帧")

    def _release_preview_for_exclusive_action(self) -> bool:
        """Stop only the passive preview before starting an exclusive mission."""
        try:
            state = self.request("/state", timeout=3.0)
            process = None
            if isinstance(state, dict):
                # The Pi bridge exposes its exclusive subprocess as ``managed``.
                # Keep the legacy ``process`` key for compatibility with older
                # bridge builds and tests.
                process = state.get("managed") or state.get("process")
            if (
                isinstance(process, dict)
                and process.get("running")
                and process.get("kind") == "camera_preview"
            ):
                self.request("/actions/preview/stop", {}, timeout=5.0)
                return True
        except Exception:
            # The mission endpoint remains the authority; it will safely reject
            # execution if another non-preview task owns the process slot.
            pass
        return False

    def dispatch_step(self, step: dict[str, Any], instruction: str) -> dict[str, Any]:
        action = str(step.get("action") or "")
        target = dict(step.get("target") or {})
        query = str(target.get("query") or target.get("name") or instruction)
        if action == "inspect_semantic_map":
            semantic = self.request("/semantic-map", timeout=3.0)
            entities = semantic.get("entities")
            count = len(entities) if isinstance(entities, dict) else len(semantic.get("landmarks", []) or semantic.get("objects", []))
            return {"status": "completed", "detail": {
                "entities": count,
                "semantic_status": semantic.get("_status"),
                "source_path": semantic.get("_source_path"),
                "warning": semantic.get("_warning", ""),
            }}
        if action == "patrol_search":
            preview_released = self._release_preview_for_exclusive_action()
            detail = self.request(
                "/interfaces/observation-plan",
                {"query": query, "instruction": instruction, "target": target},
                timeout=105.0,
            )
            detail["preview_released"] = preview_released
            return {"status": "running" if detail.get("ok") else "blocked", "detail": detail}
        if action in {"navigate_to", "return_home", "locate_destination"}:
            preview_released = self._release_preview_for_exclusive_action()
            detail = self.request(
                "/interfaces/navigation-plan",
                {"target": target, "query": query},
                timeout=105.0,
            )
            detail["preview_released"] = preview_released
            return {"status": "running" if detail.get("ok") else "blocked", "detail": detail}
        if action == "drive_relative":
            preview_released = self._release_preview_for_exclusive_action()
            detail = self.request("/actions/drive-relative", target, timeout=12.0)
            detail["preview_released"] = preview_released
            return {"status": "running" if detail.get("ok") else "blocked", "detail": detail}
        if action in {"locate_object", "align_for_grasp"}:
            detail = self.request("/actions/pursuit/start", {"target": query})
            return {"status": "running", "detail": detail}
        if action == "grasp":
            detail = self.request("/actions/grasp/start", {"target": query}, timeout=15.0)
            return {"status": "running", "detail": detail}
        if action == "release":
            detail = self.request("/actions/grasp/release", {}, timeout=12.0)
            return {"status": "running", "detail": detail}
        if action in {"verify_grasp", "verify_completion"}:
            detail = self.request("/state", timeout=3.0)
            return {"status": "awaiting_verification", "detail": detail.get("grasp", detail)}
        if action == "report_result":
            return {"status": "completed", "detail": {"message": "任务状态已写入会话历史"}}
        raise ValueError(f"尚未接入动作：{action}")
