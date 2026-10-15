"""把网页仿真器包装成项目现有 ESP32 与雷达接口。

推荐测试程序直接依赖本模块的两个类；也可以使用 run_with_simulator.py
在不改原脚本的情况下临时替换硬件类。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Iterator
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from raspberry_pi.lidar.base import LaserScan


class SimulatorUnavailable(ConnectionError):
    """网页仿真服务未启动或浏览器仿真页面未在线。"""


class _HttpClient:
    def __init__(self, base_url: str, timeout_s: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request(
        self,
        path: str,
        payload: dict | None = None,
        timeout_s: float | None = None,
    ) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(
                request,
                timeout=self.timeout_s if timeout_s is None else timeout_s,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise SimulatorUnavailable(
                f"无法连接网页仿真器 {self.base_url}：{exc}"
            ) from exc


class SimulatedEsp32Client:
    """与 ``raspberry_pi.esp32.Esp32Client`` 常用方法兼容。"""

    def __init__(
        self,
        config=None,
        base_url: str = "http://127.0.0.1:8765",
        **_kwargs,
    ) -> None:
        self.config = config
        self.http = _HttpClient(base_url)
        self.last_link_error = None
        self.last_status = None
        self._active_twist: tuple[int, int, int] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._keepalive_period_s = float(
            getattr(config, "keepalive_period_s", 0.25)
        )

    def _command(self, command: dict) -> dict:
        command.setdefault("source", "virtual-esp32")
        try:
            result = self.http.request("/api/commands", command)
            self.last_link_error = None
            return result
        except SimulatorUnavailable as exc:
            self.last_link_error = exc
            raise

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._maintain_link,
            name="car-sim-keepalive",
            daemon=True,
        )
        self._thread.start()

    def _maintain_link(self) -> None:
        while self._running:
            with self._lock:
                active = self._active_twist
            if active is not None:
                linear, angular, ttl = active
                try:
                    self._command(
                        {
                            "type": "set_twist",
                            "linear_mm_s": linear,
                            "angular_mrad_s": angular,
                            "ttl_ms": ttl,
                        }
                    )
                except SimulatorUnavailable:
                    pass
            time.sleep(max(0.05, self._keepalive_period_s))

    def heartbeat(self) -> dict:
        health = self.http.request("/api/health")
        state = self.http.request("/api/state")
        if not health.get("ok") or not state.get("online"):
            raise SimulatorUnavailable(
                "服务已启动，但网页仿真页面不在线；请打开 http://127.0.0.1:8765/"
            )
        return {"ok": True}

    def query_status(self) -> dict:
        state = self.http.request("/api/state")
        pose = state.get("pose", {})
        moving = bool(
            abs(state.get("linear_mm_s", 0)) > 1
            or abs(state.get("angular_mrad_s", 0)) > 1
        )
        self.last_status = {
            "uptime_ms": int(time.monotonic() * 1000) & 0xFFFFFFFF,
            "flags": (0x0001 if moving else 0) | (0x0008 if state.get("online") else 0),
            "linear_mm_s": int(state.get("linear_mm_s", 0)),
            "angular_mrad_s": int(state.get("angular_mrad_s", 0)),
            "left_output": 0,
            "right_output": 0,
            "wheel_feedback": (0, 0, 0, 0),
            "joint_positions": tuple(
                int(value)
                for value in state.get(
                    "joint_positions",
                    (1500, 1700, 2000, 1100, 1500, 1200),
                )
            ),
            "battery_mv": 12000,
            "bus_errors": 0,
            "ground_truth_pose": pose,
            "collision": bool(state.get("collision")),
            "task": state.get("task"),
            "arm_moving": bool(state.get("arm_moving")),
            "grasped_item_id": state.get("grasped_item_id"),
        }
        return self.last_status

    def set_twist(
        self,
        linear_mm_s: int,
        angular_mrad_s: int,
        ttl_ms: int = 600,
    ) -> dict:
        values = (int(linear_mm_s), int(angular_mrad_s), int(ttl_ms))
        result = self._command(
            {
                "type": "set_twist",
                "linear_mm_s": values[0],
                "angular_mrad_s": values[1],
                "ttl_ms": values[2],
            }
        )
        with self._lock:
            self._active_twist = values
        return result

    def stop(self) -> dict:
        with self._lock:
            self._active_twist = None
        return self._command({"type": "stop"})

    def cancel_all(self) -> dict:
        with self._lock:
            self._active_twist = None
        return self._command({"type": "cancel_all"})

    def set_pose(self, x_m: float, y_m: float, yaw_rad: float = 0.0) -> dict:
        """测试专用：重置仿真真值位姿，真实 ESP32 不提供该方法。"""
        with self._lock:
            self._active_twist = None
        return self._command(
            {
                "type": "set_pose",
                "x_m": float(x_m),
                "y_m": float(y_m),
                "yaw_rad": float(yaw_rad),
            }
        )

    def reset_map(self) -> dict:
        """测试专用：清空网页侧实时地图。"""
        return self._command({"type": "reset_map"})

    def select_task(self, task: str) -> dict:
        """测试专用：选择 ``mapping`` 或 ``pickup`` 仿真任务。"""
        return self._command({"type": "select_task", "task": str(task)})

    def goto(self, x_m: float, y_m: float) -> dict:
        """测试专用：让网页规划器导航到指定位置。"""
        return self._command(
            {"type": "goto", "x_m": float(x_m), "y_m": float(y_m)}
        )

    def set_led(self, mode, period_ms=0) -> dict:
        return {"ok": True, "simulated": True, "mode": mode, "period_ms": period_ms}

    def beep(self, repeat=1, on_ms=100, off_ms=100) -> dict:
        return {
            "ok": True,
            "simulated": True,
            "repeat": repeat,
            "on_ms": on_ms,
            "off_ms": off_ms,
        }

    def set_arm_joints(self, joints, duration_ms=800) -> dict:
        return self._command(
            {
                "type": "set_arm_joints",
                "joints": [list(joint) for joint in joints],
                "duration_ms": int(duration_ms),
            }
        )

    def arm_stop(self) -> dict:
        return self._command({"type": "arm_stop"})

    @property
    def active_transport(self) -> str:
        return "web-simulator"

    def close(self, send_cancel=True) -> None:
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if send_cancel:
            try:
                self.cancel_all()
            except SimulatorUnavailable:
                pass


class SimulatedCameraClient:
    """读取网页前置摄像头画面和结构化检测结果。

    ``read()`` 模仿 OpenCV ``VideoCapture.read()`` 返回 ``(True, BGR图像)``；
    ``snapshot()`` 返回物品 id、像素框、距离与水平偏角等规划信息。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        **_kwargs,
    ) -> None:
        self.http = _HttpClient(base_url)

    def snapshot(self) -> dict:
        state = self.http.request("/api/state")
        if not state.get("online"):
            raise SimulatorUnavailable(
                "网页摄像头不在线；请打开 http://127.0.0.1:8765/"
            )
        camera = state.get("camera")
        if not isinstance(camera, dict):
            return {"width": 640, "height": 360, "detections": []}
        return camera

    def read(self) -> tuple[bool, np.ndarray]:
        camera = self.snapshot()
        width = int(camera.get("width", 640))
        height = int(camera.get("height", 360))
        frame = np.empty((height, width, 3), dtype=np.uint8)
        horizon = int(height * 0.42)
        frame[:horizon] = (118, 105, 82)
        frame[horizon:] = (57, 62, 57)
        for detection in camera.get("detections", []):
            x, y, box_width, box_height = (
                int(value) for value in detection.get("bbox", (0, 0, 0, 0))
            )
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            x2 = max(x1 + 1, min(width, x + box_width))
            y2 = max(y1 + 1, min(height, y + box_height))
            frame[y1:y2, x1:x2] = (63, 163, 231)
            thickness = 3
            frame[y1:min(y2, y1 + thickness), x1:x2] = (198, 227, 98)
            frame[max(y1, y2 - thickness):y2, x1:x2] = (198, 227, 98)
            frame[y1:y2, x1:min(x2, x1 + thickness)] = (198, 227, 98)
            frame[y1:y2, max(x1, x2 - thickness):x2] = (198, 227, 98)
        return True, frame

    def detections(self) -> list[dict]:
        return list(self.snapshot().get("detections", []))

    def release(self) -> None:
        return None


class SimulatedLidarDriver:
    """通过长轮询读取网页生成的雷达帧。"""

    def __init__(
        self,
        config=None,
        base_url: str = "http://127.0.0.1:8765",
        poll_timeout_s: float = 2.0,
        **_kwargs,
    ) -> None:
        self.config = config
        self.http = _HttpClient(base_url, timeout_s=poll_timeout_s + 1.0)
        self.poll_timeout_s = poll_timeout_s
        self._sequence = 0
        self._closed = False
        self.last_ground_truth_pose: dict | None = None

    def scans(self) -> Iterator[LaserScan]:
        while not self._closed:
            query = urlencode(
                {
                    "after": self._sequence,
                    "timeout_ms": int(self.poll_timeout_s * 1000),
                }
            )
            result = self.http.request(
                f"/api/lidar?{query}",
                timeout_s=self.poll_timeout_s + 1.0,
            )
            scan = result.get("scan")
            if scan is None:
                continue
            self._sequence = int(scan["sequence"])
            self.last_ground_truth_pose = scan.get("ground_truth_pose")
            yield LaserScan(
                np.asarray(scan["angles_rad"], dtype=np.float64),
                np.asarray(scan["distances_m"], dtype=np.float64),
                float(scan["timestamp_s"]),
                ground_truth_pose=self.last_ground_truth_pose,
            )

    def close(self) -> None:
        self._closed = True

    def scene(self) -> dict:
        return self.http.request("/api/scene").get("scene", {})
