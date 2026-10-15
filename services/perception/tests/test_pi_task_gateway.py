from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.semantic_task_planner import SemanticTaskPlanner
from qwen_grounded_tracker.task_execution import command_payload


spec = importlib.util.spec_from_file_location(
    "run_pi_task_gateway", ROOT / "scripts/run_pi_task_gateway.py"
)
assert spec is not None and spec.loader is not None
gateway_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway_module)
PiTaskGateway = gateway_module.PiTaskGateway


def load_scene() -> dict:
    import json

    return json.loads((ROOT / "examples/demo_cleanup_scene.json").read_text(encoding="utf-8"))


class FakeHardwareGateway(PiTaskGateway):
    def __init__(self) -> None:
        super().__init__("http://map", True, "token")
        self.requests: list[tuple[str, str, dict | None]] = []
        self.semantic = {"landmarks": []}
        self.navigation_state = "running"
        self.rechecked_suspicious_ids: list[str] = []

    def _map_request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout_s: float = 2.0,
    ) -> dict:
        del timeout_s
        self.requests.append((method, path, payload))
        if path == "/api/status":
            return {
                "scan_count": 20,
                "lidar": {"ready": True},
                "control": {
                    "navigation": {"state": self.navigation_state, "reason": "test"},
                    "patrol": {
                        "rechecked_suspicious_location_ids": self.rechecked_suspicious_ids
                    },
                },
            }
        if path == "/api/semantic":
            return self.semantic
        return {"ok": True}


class FakeArmHardwareGateway(FakeHardwareGateway):
    def __init__(self) -> None:
        super().__init__()
        self.semantic = {
            "landmarks": [
                {
                    "id": "trash_01",
                    "name": "地面垃圾",
                    "type": "trash",
                    "map_xy": [1.25, -0.50],
                    "stability": 0.92,
                },
                {
                    "id": "dustpan_01",
                    "name": "绿色簸箕",
                    "type": "dustpan",
                    "map_xy": [2.0, 0.75],
                    "stability": 0.95,
                },
            ]
        }
        self.arm_poses: list[str] = []
        self.observation_overrides: dict[str, bool] = {}

    def _map_request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout_s: float = 2.0,
    ) -> dict:
        del timeout_s
        self.requests.append((method, path, payload))
        if path == "/api/status":
            return {
                "scan_count": 20,
                "lidar": {"ready": True},
                "control": {"navigation": {"state": self.navigation_state, "reason": "test"}},
            }
        if path == "/api/semantic":
            return self.semantic
        if path == "/api/arm/status":
            return {
                "ok": True,
                "ready": True,
                "moving": False,
                "fault": "",
                "profile_id": "simulated-calibrated-profile",
            }
        if path == "/api/arm/pose":
            pose = str((payload or {}).get("pose") or "")
            self.arm_poses.append(pose)
            return {
                "ok": True,
                "ready": True,
                "moving": False,
                "requested_pose": pose,
                "hardware_command_sent": True,
            }
        if path == "/api/manipulation/observation":
            target = dict((payload or {}).get("target") or {})
            is_destination = "簸箕" in str(target.get("query") or "")
            result = {
                "ok": True,
                "visible": True,
                "camera_fresh": True,
                "centered": True,
                "reachable": True,
                "lidar_clearance": True,
                "chassis_stopped": True,
                "object_in_gripper": not is_destination,
                "target_in_destination": is_destination,
            }
            result.update(self.observation_overrides)
            return result
        return {"ok": True}


class PiTaskGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = SemanticTaskPlanner().plan("帮我处理地上的垃圾", load_scene())

    def test_dry_run_gateway_never_sends_hardware(self) -> None:
        gateway = PiTaskGateway("http://127.0.0.1:8766", False, "")
        for step in self.plan.steps:
            result = gateway.command(command_payload(self.plan.task_id, step, False), "")
            self.assertEqual(result["state"], "succeeded")
            self.assertFalse(result["evidence"]["hardware_command_sent"])

    def test_hardware_gateway_blocks_unimplemented_arm_alignment_and_stops(self) -> None:
        gateway = FakeHardwareGateway()
        step = next(step for step in self.plan.steps if step.action == "align_for_grasp")
        result = gateway.command(command_payload(self.plan.task_id, step, True), "token")
        self.assertEqual(result["state"], "blocked")
        self.assertTrue(result["evidence"]["safe_stop_requested"])
        self.assertIn(("POST", "/api/patrol/stop", None), gateway.requests)
        self.assertIn(("POST", "/api/drive/stop", None), gateway.requests)

    def test_hardware_patrol_completes_only_after_target_is_in_semantic_map(self) -> None:
        gateway = FakeHardwareGateway()
        step = next(step for step in self.plan.steps if step.action == "patrol_search")
        started = gateway.command(command_payload(self.plan.task_id, step, True), "token")
        self.assertEqual(started["state"], "running")
        running = gateway.state(self.plan.task_id, step.id)
        self.assertEqual(running["state"], "running")
        gateway.semantic = {
            "landmarks": [{"id": "trash_01", "name": "地面垃圾", "type": "trash"}]
        }
        completed = gateway.state(self.plan.task_id, step.id)
        self.assertEqual(completed["state"], "succeeded")
        self.assertTrue(completed["evidence"]["target_found"])
        self.assertIn(("POST", "/api/patrol/stop", None), gateway.requests)

    def test_hardware_patrol_can_find_dynamic_bottle(self) -> None:
        gateway = FakeHardwareGateway()
        bottle_plan = SemanticTaskPlanner().plan("把蓝色水瓶放到工作桌上", load_scene())
        step = next(step for step in bottle_plan.steps if step.action == "patrol_search")
        started = gateway.command(command_payload(bottle_plan.task_id, step, True), "token")
        self.assertEqual(started["state"], "running")
        gateway.semantic = {
            "landmarks": [
                {
                    "id": "bottle_live_01",
                    "name": "蓝色水瓶",
                    "type": "bottle",
                    "stability": 0.88,
                }
            ]
        }
        completed = gateway.state(bottle_plan.task_id, step.id)
        self.assertEqual(completed["state"], "succeeded")
        self.assertTrue(completed["evidence"]["target_found"])

    def test_suspicious_location_must_be_rechecked_before_patrol_completes(self) -> None:
        scene = load_scene()
        scene["objects"].append({
            "id": "bottle_01", "name": "蓝色水瓶", "category": "bottle",
            "grid_xy": [12, 9], "approach_xy": [12, 10], "confidence": 0.9,
        })
        scene["suspicious_locations"] = [{
            "id": "suspicious_0001",
            "expected_item_id": "bottle_01",
            "expected_query": "蓝色水瓶",
            "grid_xy": [12, 10],
            "last_change_type": "moved",
            "requires_recheck_on_every_visit": True,
        }]
        plan = SemanticTaskPlanner().plan("去蓝色水瓶旁边看看它还在不在", scene)
        step = next(step for step in plan.steps if step.action == "patrol_search")
        gateway = FakeHardwareGateway()
        gateway.semantic = {
            "landmarks": [{"id": "bottle_01", "name": "蓝色水瓶", "type": "bottle"}]
        }
        started = gateway.command(command_payload(plan.task_id, step, True), "token")
        self.assertEqual(started["state"], "running")
        self.assertEqual(gateway.state(plan.task_id, step.id)["state"], "running")
        gateway.rechecked_suspicious_ids = ["suspicious_0001"]
        completed = gateway.state(plan.task_id, step.id)
        self.assertEqual(completed["state"], "succeeded")
        self.assertEqual(
            completed["evidence"]["rechecked_suspicious_location_ids"],
            ["suspicious_0001"],
        )

    def test_wrong_existing_id_is_rejected_then_matching_bottle_is_used(self) -> None:
        semantic = {
            "landmarks": [
                {"id": "cabinet_01", "name": "灰色柜子", "type": "cabinet", "stability": 0.99},
                {"id": "bottle_02", "name": "蓝色水瓶", "type": "bottle", "stability": 0.85},
            ]
        }
        resolved = PiTaskGateway._resolve_target(
            semantic,
            {"landmark_id": "cabinet_01", "query": "蓝色水瓶"},
        )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["id"], "bottle_02")

    def test_short_english_bin_alias_does_not_match_cabinet(self) -> None:
        semantic = {
            "landmarks": [
                {"id": "cabinet_01", "name": "cabinet", "type": "cabinet", "stability": 0.99}
            ]
        }
        self.assertIsNone(PiTaskGateway._resolve_target(semantic, {"query": "trash bin"}))

    def test_wrong_token_is_blocked_and_stopped(self) -> None:
        gateway = FakeHardwareGateway()
        step = self.plan.steps[0]
        result = gateway.command(command_payload(self.plan.task_id, step, True), "wrong")
        self.assertEqual(result["state"], "blocked")
        self.assertIn(("POST", "/api/drive/stop", None), gateway.requests)

    def test_hardware_navigation_resolves_landmark_and_waits_for_local_arrival(self) -> None:
        gateway = FakeHardwareGateway()
        gateway.semantic = {
            "landmarks": [
                {
                    "id": "trash_01",
                    "name": "地面垃圾",
                    "type": "trash",
                    "map_xy": [1.25, -0.50],
                    "stability": 0.9,
                }
            ]
        }
        step = next(step for step in self.plan.steps if step.action == "navigate_to")
        started = gateway.command(command_payload(self.plan.task_id, step, True), "token")
        self.assertEqual(started["state"], "running")
        self.assertIn(
            ("POST", "/api/navigation/goal", {"x_m": 1.25, "y_m": -0.5, "target_id": "trash_01"}),
            gateway.requests,
        )
        self.assertEqual(gateway.state(self.plan.task_id, step.id)["state"], "running")
        gateway.navigation_state = "succeeded"
        completed = gateway.state(self.plan.task_id, step.id)
        self.assertEqual(completed["state"], "succeeded")

    def test_calibrated_simulated_arm_executes_full_cleanup_action_chain(self) -> None:
        gateway = FakeArmHardwareGateway()
        completed_actions: list[str] = []
        for step in self.plan.steps:
            result = gateway.command(command_payload(self.plan.task_id, step, True), "token")
            polls = 0
            while result["state"] == "running":
                if step.action == "navigate_to":
                    gateway.navigation_state = "succeeded"
                result = gateway.state(self.plan.task_id, step.id)
                polls += 1
                self.assertLess(polls, 10)
            self.assertEqual(result["state"], "succeeded", (step.action, result))
            completed_actions.append(step.action)

        self.assertEqual(completed_actions, [step.action for step in self.plan.steps])
        self.assertEqual(
            gateway.arm_poses,
            ["pregrasp", "grasp", "lift", "carry", "preplace", "place", "open", "stow"],
        )

    def test_missing_camera_grasp_evidence_blocks_and_stops_everything(self) -> None:
        gateway = FakeArmHardwareGateway()
        gateway.observation_overrides["object_in_gripper"] = False
        step = next(step for step in self.plan.steps if step.action == "verify_grasp")
        result = gateway.command(command_payload(self.plan.task_id, step, True), "token")
        self.assertEqual(result["state"], "blocked")
        self.assertIn("object_in_gripper", result["evidence"]["failed_gates"])
        self.assertIn(("POST", "/api/arm/stop", None), gateway.requests)

    def test_health_exposes_arm_actions_only_when_profile_ready(self) -> None:
        ready = FakeArmHardwareGateway().health()
        self.assertIn("grasp", ready["connected_hardware_actions"])
        unready = FakeHardwareGateway().health()
        self.assertIn("grasp", unready["missing_hardware_actions"])


if __name__ == "__main__":
    unittest.main()
