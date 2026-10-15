"""HTTP client and guarded task-step dispatcher for the Pi bridge."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Iterator
from urllib import error as urlerror
from urllib import request as urlrequest


class RobotClient:
    def __init__(self) -> None:
        # The 3090 subnet cannot route to the Pi subnet in the deployed lab.
        # The Pi therefore maintains a loopback-only reverse SSH tunnel.
        self.base_url = os.environ.get("FINAL_WORK_PI_URL", "http://127.0.0.1:18780").rstrip("/")
        self.safety_token = os.environ.get("FINAL_WORK_HARDWARE_TOKEN", "")
        self.camera_order_provider = None

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
        try:
            with urlrequest.urlopen(req, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                body = json.loads(raw)
                detail = body.get("error") or body.get("detail") or raw
            except (json.JSONDecodeError, AttributeError):
                detail = raw or str(exc)
            raise RuntimeError(f"机器人桥接服务 {exc.code}: {detail}") from exc
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

    def probe_current_target(self, target: str) -> dict[str, Any]:
        """Ask the Pi for a binocular target check without moving hardware."""
        return self.request("/perception/current-view", {"target": target}, timeout=32.0)

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
            if target.get("integration_workflow") == "robot_skills_v1":
                if action == "locate_destination":
                    detail = self.request("/skills/resolve-destination", target, timeout=12)
                    return {"status": "completed" if detail.get("ok") else "blocked", "detail": detail}
                self._release_preview_for_exclusive_action()
                detail = self.request("/skills/navigate-region", target, timeout=12)
                return {"status": "running" if detail.get("ok") else "blocked", "detail": detail}
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
        if action == "prepare_stationary_grasp":
            state = self.request("/state", timeout=3.0)
            managed = state.get("managed") or {}
            if managed.get("running") and managed.get("kind") not in {"idle", "camera_preview"}:
                return {"status": "blocked", "detail": {
                    "message": f"底盘相关任务仍在运行：{managed.get('kind')}；不能开始原地抓取",
                    "managed": managed,
                }}
            grasp = state.get("grasp") or {}
            if not grasp.get("connected"):
                self.request("/actions/preview/start", {}, timeout=5.0)
            deadline = time.monotonic() + 12.0
            last_state = state
            while time.monotonic() < deadline:
                last_state = self.request("/state", timeout=3.0)
                grasp = last_state.get("grasp") or {}
                data = grasp.get("data") or {}
                cameras = data.get("cameras") or {}
                fresh = [
                    camera for camera in ("left", "right")
                    if isinstance(cameras.get(camera), dict)
                    and cameras[camera].get("frames", 0) > 0
                    and 0 <= float(cameras[camera].get("age_s", 99)) < 2.0
                ]
                if grasp.get("connected") and data.get("ble") is True and len(fresh) == 2:
                    return {"status": "completed", "detail": {
                        "message": "底盘保持停止，左右相机均已就绪",
                        "camera_ready": fresh,
                        "motion_allowed": False,
                    }}
                time.sleep(0.25)
            return {"status": "blocked", "detail": {
                "message": "底盘未移动，但双目画面或机械臂 BLE 未能在 12 秒内就绪",
                "state": last_state.get("grasp"),
            }}
        if action in {"locate_object", "align_for_grasp"}:
            payload = {"target": query}
            if self.camera_order_provider is not None:
                payload["camera_order"] = self.camera_order_provider()
            detail = self.request("/actions/pursuit/start", payload)
            return {"status": "running", "detail": detail}
        if action == "model_grasp":
            # Includes bounded pre-grasp hull micro-adjustment, a fresh
            # prediction, and the full queued arm sequence.  The previous
            # 215 s client deadline could expire while the Pi was still
            # moving the chassis or arm.
            detail = self.request(
                "/actions/grasp/model",
                {"target": query,
                 "auto_align": target.get("allow_chassis_motion") is True},
                timeout=390.0,
            )
            # Pi returns only after the queued servo sequence reaches done.
            # Visual verification remains a separate physical-outcome step.
            return {"status": "completed" if detail.get("ok") else "failed", "detail": detail}
        if action == "grasp":
            detail = self.request("/actions/grasp/start", {"target": query}, timeout=15.0)
            return {"status": "completed" if detail.get("ok") else "failed", "detail": detail}
        if action == "reset_arm":
            detail = self.request("/actions/grasp/home", {}, timeout=12.0)
            return {"status": "completed" if detail.get("ok") else "failed", "detail": detail}
        if action == "release":
            detail = self.request("/actions/grasp/release", {}, timeout=12.0)
            return {"status": "completed" if detail.get("ok") else "failed", "detail": detail}
        if action == "place_on_floor":
            # Destination navigation closed 8080 to take the cameras/BLE.
            # Restore the passive owner before accessing its arm queue.
            self.request("/actions/preview/start", {}, timeout=12)
            detail = self.request("/skills/place-on-floor", target, timeout=110)
            return {"status": "completed" if detail.get("ok") else "blocked", "detail": detail}
        if action in {"verify_grasp", "verify_completion"}:
            if action == "verify_grasp" and target.get("carry_after_grasp"):
                detail = self.request("/skills/check-carry", {}, timeout=8)
                return {"status": "completed" if detail.get("ok") else "blocked", "detail": detail}
            if action == "verify_completion" and target.get("integration_workflow") == "robot_skills_v1":
                state = self.request("/state", timeout=4)
                skills = state.get("robot_skills") or {}
                return {"status": "completed" if skills.get("status") == "placed" else "blocked",
                        "detail": {"skills": skills, "physical_place_verified": False,
                                   "message": "动作执行与照片已记录；没有独立实物成功证据时不声称视觉核验成功"}}
            detail = self.request("/state", timeout=3.0)
            return {"status": "awaiting_verification", "detail": detail.get("grasp", detail)}
        if action == "report_result":
            return {"status": "completed", "detail": {"message": "任务状态已写入会话历史"}}
        raise ValueError(f"尚未接入动作：{action}")

