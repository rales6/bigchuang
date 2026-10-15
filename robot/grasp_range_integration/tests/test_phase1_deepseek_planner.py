from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from grasp_range_integration.phase1.deepseek_planner import (
    compile_phase1_decision,
    parse_deepseek_contract,
    validate_phase1_decision,
)
from grasp_range_integration.phase1.relationship_find_only_entry import (
    _install_waypoint_obstacle_guard,
    _phase1_directive,
)


ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_PATH = next(
    path for path in (
        ROOT / "maps/semantic_release_20260919/semantic_runtime.json",
        ROOT / "semantic_runtime.json",
    )
    if path.is_file()
)
SEMANTIC_MAP = json.loads(SEMANTIC_PATH.read_text(encoding="utf-8"))
INSTRUCTION = "去柜子边上夹取一个红色的小物块"


def contract() -> dict:
    return {
        "intent": "find_object_near_anchor",
        "anchor_query": "柜子",
        "target": {"query": "红色小物块", "color": "red", "class_name": "block", "count": 1},
        "current_phase_action": "find_and_stop",
        "requested_final_action": "grasp",
        "confidence": 0.96,
        "needs_clarification": False,
        "clarification_question": "",
        "reason": "到柜子附近寻找红色小物块",
    }


class Phase1DeepSeekPlannerTests(unittest.TestCase):
    def test_exact_instruction_compiles_to_cabinet_find_only(self):
        decision = compile_phase1_decision(INSTRUCTION, contract(), SEMANTIC_MAP)
        self.assertEqual(decision["status"], "executable")
        self.assertEqual(decision["source"], "deepseek_validated")
        self.assertEqual(decision["anchor"]["search_region_id"], "layout_466a3c114803")
        self.assertEqual(decision["route_request"]["mode"], "semantic_search_region")
        self.assertEqual(
            decision["route_request"]["planning_policy"],
            "semantic_visibility_regions.v1",
        )
        self.assertTrue(decision["route_request"]["inspect_current_view_first"])
        self.assertTrue(decision["route_request"]["stop_on_visibility_entry"])
        self.assertEqual(decision["target"]["color"], "red")
        self.assertEqual(decision["target"]["class_name"], "block")
        self.assertEqual(decision["target"]["detection_mode"], "qwen_fallback")
        self.assertFalse(decision["capabilities"]["grasp_enabled"])
        self.assertFalse(decision["capabilities"]["arm_commands_allowed"])
        self.assertEqual(
            decision["phase_control"]["terminal_condition"],
            "target_found_and_vehicle_stopped",
        )
        self.assertEqual(decision["phase_control"]["deferred_actions"], ["grasp"])
        self.assertNotIn("grasp", {step["action"] for step in decision["steps"]})

    def test_model_coordinates_and_commands_are_ignored(self):
        malicious = contract()
        malicious.update({
            "coordinates": [99, 99],
            "path": [[1, 2], [3, 4]],
            "motor_command": "forward",
            "arm_command": "grasp_now",
        })
        decision = compile_phase1_decision(INSTRUCTION, malicious, SEMANTIC_MAP)
        self.assertEqual(
            decision["route_request"]["center_xy_m"],
            [0.8751325645044502, 1.0443147835391624],
        )
        audit = decision["model_interpretation"]
        self.assertNotIn("coordinates", audit)
        self.assertNotIn("path", audit)
        self.assertNotIn("motor_command", audit)
        self.assertNotIn("arm_command", audit)

    def test_explicit_words_override_model_misread(self):
        wrong = contract()
        wrong["anchor_query"] = "桌子"
        wrong["target"] = {"query": "蓝色瓶子", "color": "blue", "class_name": "bottle", "count": 1}
        decision = compile_phase1_decision(INSTRUCTION, wrong, SEMANTIC_MAP)
        self.assertEqual(decision["anchor"]["search_region_id"], "layout_466a3c114803")
        self.assertEqual(decision["target"]["color"], "red")
        self.assertEqual(decision["target"]["class_name"], "block")

    def test_low_confidence_never_becomes_executable(self):
        low = contract()
        low["confidence"] = 0.3
        decision = compile_phase1_decision(INSTRUCTION, low, SEMANTIC_MAP)
        self.assertEqual(decision["status"], "needs_clarification")
        with self.assertRaises(ValueError):
            validate_phase1_decision(decision, require_executable=True)

    def test_validator_rejects_added_grasp_step(self):
        decision = compile_phase1_decision(INSTRUCTION, contract(), SEMANTIC_MAP)
        changed = copy.deepcopy(decision)
        changed["steps"].append({"id": "step_08", "action": "grasp", "executor": "arm"})
        with self.assertRaises(ValueError):
            validate_phase1_decision(changed, require_executable=True)

    def test_validator_rejects_old_point_planning_policy(self):
        decision = compile_phase1_decision(INSTRUCTION, contract(), SEMANTIC_MAP)
        changed = copy.deepcopy(decision)
        changed["route_request"].pop("planning_policy")
        with self.assertRaises(ValueError):
            validate_phase1_decision(changed, require_executable=True)

    def test_parse_contract_discards_unsupported_current_action(self):
        parsed = parse_deepseek_contract(json.dumps({
            "intent": "find_object_near_anchor",
            "anchor_query": "柜子",
            "target": {"query": "红色小物块", "color": "red", "class_name": "block"},
            "current_phase_action": "grasp_now",
            "requested_final_action": "grasp",
            "confidence": 0.9,
        }))
        self.assertEqual(parsed["current_phase_action"], "find_and_stop")

    def test_waypoint_guard_selects_narrow_corridor_without_changing_distance(self):
        calls = []

        class FakeMission:
            build_search_route = staticmethod(lambda *args, **kwargs: {})
            _fuse_tracking_pose = staticmethod(lambda predicted, result: predicted)

            @staticmethod
            def _obstacle_safe(scan, linear, stop_distance, **kwargs):
                calls.append((scan, linear, stop_distance, kwargs))
                return True

        _install_waypoint_obstacle_guard(FakeMission)
        self.assertTrue(FakeMission._obstacle_safe("scan", 400, 0.45))
        self.assertEqual(calls, [("scan", 400, 0.45, {"angular_mrad_s": 0.0})])

    def test_view_order_and_low_confidence_gate(self):
        route = {"views": [
            {"visibility_zone": {"band": "high"}},
            {"visibility_zone": {"band": "medium"}},
            {"visibility_zone": {"band": "low"}},
        ]}
        state = {"latest_event": {"type": "observation_complete"}, "remaining_view_indices": [1, 2]}
        self.assertEqual(_phase1_directive(state, route, allow_low=False)["next_view_index"], 1)
        state["remaining_view_indices"] = [2]
        self.assertEqual(_phase1_directive(state, route, allow_low=False)["action"], "report_failure")
        self.assertEqual(_phase1_directive(state, route, allow_low=True)["next_view_index"], 2)

    def test_stereo_match_stops_before_another_view(self):
        route = {"views": [{"visibility_zone": {"band": "high"}}]}
        state = {
            "latest_event": {"type": "observation_complete"},
            "remaining_view_indices": [0],
            "detections": [
                {"confirmed_match": True, "dual_camera_confirmed": True, "stereo_pair_id": "p1", "camera_id": "left"},
                {"confirmed_match": True, "dual_camera_confirmed": True, "stereo_pair_id": "p1", "camera_id": "right"},
            ],
        }
        self.assertEqual(_phase1_directive(state, route, allow_low=False)["action"], "report_success")


if __name__ == "__main__":
    unittest.main()
