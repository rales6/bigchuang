from __future__ import annotations

import unittest

from final_work.gpu.robot_client import RobotClient


class RobotClientTests(unittest.TestCase):
    def test_relative_drive_uses_guarded_bridge_action(self) -> None:
        client = RobotClient()
        calls = []

        def request(path, payload=None, *, timeout=8.0):
            calls.append((path, payload))
            if path == "/state":
                return {"managed": {"kind": "idle", "running": False}}
            return {"ok": True, "mode": "simulate"}

        client.request = request  # type: ignore[method-assign]
        result = client.dispatch_step({
            "action": "drive_relative",
            "target": {"direction": "forward", "distance_m": 1.0, "angle_deg": 0.0},
        }, "向前行动1m")
        self.assertEqual(result["status"], "running")
        self.assertEqual(calls[1][0], "/actions/drive-relative")
        self.assertEqual(calls[1][1]["distance_m"], 1.0)

    def test_release_preview_stops_only_passive_camera_preview(self) -> None:
        client = RobotClient()
        calls = []

        def request(path, payload=None, *, timeout=8.0):
            calls.append((path, payload))
            if path == "/state":
                return {"managed": {"kind": "camera_preview", "running": True}}
            return {"ok": True}

        client.request = request  # type: ignore[method-assign]
        self.assertTrue(client._release_preview_for_exclusive_action())
        self.assertEqual(calls, [("/state", None), ("/actions/preview/stop", {})])

    def test_release_preview_never_stops_navigation(self) -> None:
        client = RobotClient()
        calls = []

        def request(path, payload=None, *, timeout=8.0):
            calls.append((path, payload))
            return {"managed": {"kind": "map_navigation_search", "running": True}}

        client.request = request  # type: ignore[method-assign]
        self.assertFalse(client._release_preview_for_exclusive_action())
        self.assertEqual(calls, [("/state", None)])

    def test_stationary_grasp_preparation_only_checks_fresh_cameras(self) -> None:
        client = RobotClient()
        calls = []

        def request(path, payload=None, *, timeout=8.0):
            calls.append((path, payload))
            return {
                "managed": {"kind": "idle", "running": False},
                "grasp": {"connected": True, "data": {"cameras": {
                    "left": {"frames": 10, "age_s": 0.1},
                    "right": {"frames": 11, "age_s": 0.1},
                }}},
            }

        client.request = request  # type: ignore[method-assign]
        result = client.dispatch_step({
            "action": "prepare_stationary_grasp",
            "target": {"query": "绿色方块"},
        }, "不移动，直接夹取视野中的绿色方块")
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["detail"]["motion_allowed"])
        self.assertEqual(calls, [("/state", None), ("/state", None)])

    def test_model_grasp_uses_arm_teach_route(self) -> None:
        client = RobotClient()
        calls = []

        def request(path, payload=None, *, timeout=8.0):
            calls.append((path, payload, timeout))
            return {"ok": True, "message": "queued"}

        client.request = request  # type: ignore[method-assign]
        result = client.dispatch_step({
            "action": "model_grasp", "target": {"query": "绿色方块"},
        }, "夹取绿色方块")
        self.assertEqual(result["status"], "running")
        self.assertEqual(calls[0][0], "/actions/grasp/model")
        self.assertEqual(calls[0][1], {"target": "绿色方块"})

    def test_arm_reset_uses_home_route(self) -> None:
        client = RobotClient()
        calls = []

        def request(path, payload=None, *, timeout=8.0):
            calls.append((path, payload))
            return {"ok": True}

        client.request = request  # type: ignore[method-assign]
        result = client.dispatch_step({"action": "reset_arm", "target": {}}, "机械臂复位")
        self.assertEqual(result["status"], "running")
        self.assertEqual(calls, [("/actions/grasp/home", {})])


if __name__ == "__main__":
    unittest.main()
