"""从其他 Python 文件向 car_sim 发送指令的轻量客户端。"""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


class SimulatorClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765") -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urlopen(request, timeout=2.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def set_twist(
        self,
        linear_mm_s: int,
        angular_mrad_s: int,
        ttl_ms: int = 600,
    ) -> dict:
        return self._request(
            "/api/commands",
            {
                "type": "set_twist",
                "linear_mm_s": linear_mm_s,
                "angular_mrad_s": angular_mrad_s,
                "ttl_ms": ttl_ms,
                "source": "python-client",
            },
        )

    def stop(self) -> dict:
        return self._request(
            "/api/commands", {"type": "stop", "source": "python-client"}
        )

    def cancel_all(self) -> dict:
        return self._request(
            "/api/commands", {"type": "cancel_all", "source": "python-client"}
        )

    def goto(self, x_m: float, y_m: float) -> dict:
        return self._request(
            "/api/commands",
            {"type": "goto", "x_m": x_m, "y_m": y_m, "source": "python-client"},
        )

    def set_pose(self, x_m: float, y_m: float, yaw_rad: float = 0.0) -> dict:
        return self._request(
            "/api/commands",
            {
                "type": "set_pose",
                "x_m": x_m,
                "y_m": y_m,
                "yaw_rad": yaw_rad,
                "source": "python-client",
            },
        )

    def select_task(self, task: str) -> dict:
        return self._request(
            "/api/commands",
            {"type": "select_task", "task": task, "source": "python-client"},
        )

    def set_arm_joints(self, joints, duration_ms: int = 800) -> dict:
        return self._request(
            "/api/commands",
            {
                "type": "set_arm_joints",
                "joints": [list(joint) for joint in joints],
                "duration_ms": duration_ms,
                "source": "python-client",
            },
        )

    def arm_stop(self) -> dict:
        return self._request(
            "/api/commands", {"type": "arm_stop", "source": "python-client"}
        )

    def state(self) -> dict:
        return self._request("/api/state")


if __name__ == "__main__":
    client = SimulatorClient()
    print(client.goto(6.4, 3.8))
