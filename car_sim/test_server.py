"""car_sim 服务端的无浏览器接口测试。"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import SimulationHandler


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), SimulationHandler)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path: str, payload: dict | None = None) -> tuple[int, dict | str]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urlopen(request, timeout=2) as response:
                content = response.read().decode("utf-8")
                if response.headers.get_content_type() == "application/json":
                    return response.status, json.loads(content)
                return response.status, content
        except HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_health_and_page(self) -> None:
        status, health = self.request("/api/health")
        self.assertEqual(200, status)
        self.assertTrue(health["ok"])
        status, page = self.request("/")
        self.assertEqual(200, status)
        self.assertIn("CAR·LAB", page)

    def test_command_round_trip(self) -> None:
        status, response = self.request(
            "/api/commands",
            {
                "type": "set_twist",
                "linear_mm_s": 250,
                "angular_mrad_s": -300,
                "ttl_ms": 600,
            },
        )
        self.assertEqual(202, status)
        command_id = response["command"]["id"]
        status, queue = self.request(f"/api/commands?after={command_id - 1}")
        self.assertEqual(200, status)
        self.assertEqual(command_id, queue["commands"][-1]["id"])

    def test_rejects_unsafe_command(self) -> None:
        status, response = self.request(
            "/api/commands",
            {
                "type": "set_twist",
                "linear_mm_s": 9000,
                "angular_mrad_s": 0,
                "ttl_ms": 600,
            },
        )
        self.assertEqual(400, status)
        self.assertFalse(response["ok"])

    def test_task_and_arm_commands(self) -> None:
        status, task = self.request(
            "/api/commands",
            {"type": "select_task", "task": "pickup"},
        )
        self.assertEqual(202, status)
        self.assertEqual("pickup", task["command"]["task"])

        status, arm = self.request(
            "/api/commands",
            {
                "type": "set_arm_joints",
                "joints": [[0, 1800], [5, 1500]],
                "duration_ms": 700,
            },
        )
        self.assertEqual(202, status)
        self.assertEqual([[0, 1800], [5, 1500]], arm["command"]["joints"])

        status, rejected = self.request(
            "/api/commands",
            {
                "type": "set_arm_joints",
                "joints": [[5, 2200]],
                "duration_ms": 700,
            },
        )
        self.assertEqual(400, status)
        self.assertFalse(rejected["ok"])

    def test_lidar_and_scene_round_trip(self) -> None:
        status, response = self.request(
            "/api/lidar",
            {
                "timestamp_s": 123.5,
                "angles_rad": [-0.5, 0.0, 0.5],
                "distances_m": [1.2, 2.0, 1.4],
                "ground_truth_pose": {
                    "x_m": 1.0,
                    "y_m": 2.0,
                    "yaw_rad": 0.2,
                },
            },
        )
        self.assertEqual(202, status)
        sequence = response["sequence"]
        status, lidar = self.request(f"/api/lidar?after={sequence - 1}")
        self.assertEqual(200, status)
        self.assertEqual([1.2, 2.0, 1.4], lidar["scan"]["distances_m"])

        scene = {
            "room": {"width": 8, "height": 5},
            "obstacles": [{"type": "circle", "x": 2, "y": 2, "r": 0.5}],
        }
        status, _ = self.request("/api/scene", scene)
        self.assertEqual(200, status)
        status, result = self.request("/api/scene")
        self.assertEqual(200, status)
        self.assertEqual(8.0, result["scene"]["room"]["width"])

    def test_z_only_claimed_simulator_can_publish_lidar(self) -> None:
        owner = "test-owner-session"
        status, response = self.request(
            "/api/session",
            {"simulator_session_id": owner},
        )
        self.assertEqual(200, status)
        self.assertEqual(owner, response["simulator_session_id"])

        scan = {
            "timestamp_s": 200.0,
            "angles_rad": [0.0],
            "distances_m": [1.0],
            "ground_truth_pose": {
                "x_m": 0.0,
                "y_m": 0.0,
                "yaw_rad": 0.0,
            },
        }
        status, _ = self.request("/api/lidar", scan)
        self.assertEqual(409, status)
        scan["simulator_session_id"] = owner
        status, _ = self.request("/api/lidar", scan)
        self.assertEqual(202, status)


if __name__ == "__main__":
    unittest.main()
