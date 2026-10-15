"""Regression checks for live-stereo plan shortening."""

from __future__ import annotations

import unittest

from .plan_adaptation import adapt_plan_to_live_view, normalize_web_plan


def sample_plan(intent: str = "pick") -> dict:
    actions = ["inspect_semantic_map", "patrol_search", "align_for_grasp",
               "model_grasp", "verify_grasp", "report_result"]
    if intent == "pick_and_place":
        actions = ["inspect_semantic_map", "patrol_search", "navigate_to",
                   "align_for_grasp", "model_grasp", "verify_grasp",
                   "locate_destination", "navigate_to", "release",
                   "verify_completion", "report_result"]
    if intent == "search":
        actions = ["inspect_semantic_map", "patrol_search", "align_for_grasp", "report_result"]
    steps = []
    for index, action in enumerate(actions, 1):
        steps.append({"id": f"step_{index:02d}", "action": action,
                      "description": action,
                      "target": {"query": "红色方块", "allow_chassis_motion": True},
                      "depends_on": [] if index == 1 else [f"step_{index - 1:02d}"]})
    return {"intent": intent, "summary": "原始计划", "source": "qwen_compiled", "steps": steps}


class PlanAdaptationTests(unittest.TestCase):
    def test_visible_and_in_region_skips_chassis(self) -> None:
        original = sample_plan()
        plan = adapt_plan_to_live_view(original, {"status": "visible_both",
                                                  "inside_grasp_region": True,
                                                  "grasp_margin_px": 25})
        self.assertEqual([s["action"] for s in plan["steps"]],
                         ["prepare_stationary_grasp", "model_grasp", "verify_grasp", "report_result"])
        self.assertFalse(plan["steps"][1]["target"]["allow_chassis_motion"])
        self.assertTrue(original["steps"][3]["target"]["allow_chassis_motion"])

    def test_visible_outside_region_goes_directly_to_pursuit(self) -> None:
        plan = adapt_plan_to_live_view(sample_plan(), {"status": "visible_both",
                                                       "inside_grasp_region": False})
        self.assertEqual([s["action"] for s in plan["steps"]],
                         ["align_for_grasp", "model_grasp", "verify_grasp", "report_result"])
        self.assertEqual(plan["steps"][1]["depends_on"], ["step_01"])

    def test_transport_keeps_destination_steps_and_verification(self) -> None:
        plan = adapt_plan_to_live_view(sample_plan("pick_and_place"),
                                       {"status": "visible_both", "inside_grasp_region": True})
        actions = [s["action"] for s in plan["steps"]]
        self.assertEqual(actions[:3], ["prepare_stationary_grasp", "model_grasp", "verify_grasp"])
        self.assertEqual(actions[3:6], ["locate_destination", "navigate_to", "release"])
        self.assertEqual(plan["steps"][3]["depends_on"], ["step_03"])

    def test_detector_miss_or_fault_does_not_rewrite_plan(self) -> None:
        plan = sample_plan()
        for status in ("not_visible_both", "unavailable"):
            self.assertIs(adapt_plan_to_live_view(plan, {"status": status}), plan)

    def test_search_already_seen_does_not_patrol(self) -> None:
        plan = adapt_plan_to_live_view(sample_plan("search"),
                                       {"status": "visible_both", "inside_grasp_region": False})
        self.assertEqual([s["action"] for s in plan["steps"]],
                         ["align_for_grasp", "report_result"])

    def test_old_model_plan_does_not_launch_pursuit_twice(self) -> None:
        plan = sample_plan()
        plan["steps"].insert(3, {"id": "old", "action": "locate_object",
                                  "target": {"query": "红色方块"}, "description": "定位"})
        plan["steps"][4]["description"] = "核对四维示教凸包"
        normalized = normalize_web_plan(plan)
        self.assertNotIn("locate_object", [s["action"] for s in normalized["steps"]])
        self.assertEqual(normalized["steps"][3]["depends_on"], ["step_03"])
        self.assertIn("左右各自二维", normalized["steps"][3]["description"])

    def test_turn_step_describes_forward_arc(self) -> None:
        plan = {"intent": "drive", "steps": [{"id": "step_01", "action": "drive_relative",
                 "description": "雷达保护下左转 180°",
                 "target": {"direction": "left", "angle_deg": 180}, "depends_on": []}]}
        normalized = normalize_web_plan(plan)
        self.assertIn("分段前进画弧左转 180°", normalized["steps"][0]["description"])

    def test_backward_step_does_not_claim_rear_lidar_guard(self) -> None:
        plan = {"intent": "drive", "steps": [{"id": "step_01", "action": "drive_relative",
                 "description": "雷达保护下后退 0.10m",
                 "target": {"direction": "backward", "distance_m": 0.10}, "depends_on": []}]}
        normalized = normalize_web_plan(plan)
        self.assertIn("后向雷达会看到车身", normalized["steps"][0]["description"])


if __name__ == "__main__":
    unittest.main()
