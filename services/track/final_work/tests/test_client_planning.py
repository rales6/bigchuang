from __future__ import annotations

import json
import unittest

from final_work.gpu.planner import AgentPlanner


class ClientPlanningTests(unittest.TestCase):
    def test_metric_drive_instruction_bypasses_object_search(self) -> None:
        plan = AgentPlanner().plan("向前行动1m", {}, {"provider": "local_qwen"})
        self.assertEqual(plan["intent"], "drive")
        self.assertEqual(plan["source"], "deterministic_motion_parser")
        self.assertEqual(plan["steps"][0]["action"], "drive_relative")
        self.assertEqual(plan["steps"][0]["target"]["direction"], "forward")
        self.assertEqual(plan["steps"][0]["target"]["distance_m"], 1.0)

    def test_centimetres_and_turn_angle_are_normalized(self) -> None:
        planner = AgentPlanner()
        backward = planner.plan("后退50厘米", {}, {"provider": "local_qwen"})
        turn = planner.plan("右转45度", {}, {"provider": "local_qwen"})
        self.assertEqual(backward["steps"][0]["target"]["distance_m"], 0.5)
        self.assertEqual(turn["steps"][0]["target"]["angle_deg"], 45.0)

    def test_current_view_pick_never_adds_map_or_chassis_motion(self) -> None:
        plan = AgentPlanner().plan(
            "不移动，直接夹取视野中的绿色方块", {}, {"provider": "local_qwen"}
        )
        actions = [step["action"] for step in plan["steps"]]
        self.assertEqual(plan["source"], "deterministic_stationary_pick")
        self.assertEqual(plan["intent"], "pick")
        self.assertEqual(
            actions,
            ["prepare_stationary_grasp", "model_grasp", "verify_grasp", "report_result"],
        )
        self.assertEqual(plan["steps"][0]["target"]["query"], "绿色方块")
        self.assertFalse({"inspect_semantic_map", "patrol_search", "navigate_to", "locate_object", "align_for_grasp"} & set(actions))

    def test_current_view_wording_alone_implies_stationary_scope(self) -> None:
        plan = AgentPlanner().plan(
            "夹取视野中的绿色方块", {}, {"provider": "local_qwen"}
        )
        self.assertEqual(plan["steps"][0]["action"], "prepare_stationary_grasp")

    def test_explicit_small_adjustment_overrides_current_view_stationary_default(self) -> None:
        plan = AgentPlanner().plan(
            "小步调整小车位置，抓取视野中的红色方块",
            {},
            {"provider": "local_qwen"},
        )
        actions = [step["action"] for step in plan["steps"]]
        self.assertEqual(plan["source"], "deterministic_current_view_align_pick")
        self.assertEqual(plan["steps"][0]["target"]["query"], "红色方块")
        self.assertEqual(
            actions,
            ["align_for_grasp", "model_grasp", "verify_grasp", "report_result"],
        )
        self.assertNotIn("prepare_stationary_grasp", actions)
        self.assertNotIn("patrol_search", actions)
        self.assertNotIn("inspect_semantic_map", actions)

    def test_client_model_cannot_override_explicit_no_motion_constraint(self) -> None:
        raw = json.dumps({
            "intent": "pick", "target_query": "绿色方块",
            "motion_policy": "allow", "perception_scope": "map",
        }, ensure_ascii=False)
        plan = AgentPlanner.compile_client_output(
            "不移动，直接夹取视野中的绿色方块", raw, "deepseek_flash_api"
        )
        actions = {step["action"] for step in plan["steps"]}
        self.assertIn("prepare_stationary_grasp", actions)
        self.assertNotIn("patrol_search", actions)
        self.assertNotIn("align_for_grasp", actions)

    def test_arm_reset_is_direct_and_never_starts_patrol(self) -> None:
        plan = AgentPlanner().plan("机械臂复位", {}, {"provider": "local_qwen"})
        actions = [step["action"] for step in plan["steps"]]
        self.assertEqual(plan["intent"], "arm_reset")
        self.assertEqual(plan["source"], "deterministic_arm_reset")
        self.assertEqual(actions, ["reset_arm", "report_result"])
        self.assertNotIn("patrol_search", actions)

    def test_prompt_contains_only_compact_semantic_context(self) -> None:
        prompt = AgentPlanner.client_prompt(
            "把红色方块放到中央桌子",
            {
                "objects": [{"id": "desk", "name": "中央桌子", "map_xy": [0.0, -1.3]}],
                "search_regions": [{"id": "desk", "query": "中央桌子"}],
                "private_debug": "must_not_be_forwarded",
            },
        )
        self.assertIn("中央桌子", prompt)
        self.assertNotIn("private_debug", prompt)
        self.assertNotIn("must_not_be_forwarded", prompt)

    def test_client_output_is_compiled_through_action_allowlist(self) -> None:
        raw = json.dumps({
            "intent": "pick_and_place",
            "target_query": "红色方块",
            "destination_query": "中央桌子",
            "target_landmark_id": None,
            "destination_landmark_id": "desk",
            "confidence": 0.91,
            "shell": "rm -rf /",
            "speed": 9999,
        }, ensure_ascii=False)
        plan = AgentPlanner.compile_client_output("把红色方块放到中央桌子", raw, "deepseek_flash_api")
        actions = {step["action"] for step in plan["steps"]}
        self.assertEqual(plan["source"], "deepseek_flash_api_client_compiled")
        self.assertNotIn("shell", plan)
        self.assertNotIn("speed", plan)
        self.assertTrue(actions <= {
            "inspect_semantic_map", "patrol_search", "navigate_to", "locate_object",
            "align_for_grasp", "grasp", "model_grasp", "verify_grasp", "locate_destination",
            "release", "verify_completion", "report_result",
        })

    def test_pick_near_anchor_keeps_target_and_search_area_separate(self) -> None:
        raw = json.dumps({
            "intent": "pick",
            "target_query": "绿色小木块",
            "search_near_query": "黄色柜子",
            "destination_query": "",
            "target_landmark_id": None,
            "search_near_landmark_id": "cabinet",
            "destination_landmark_id": None,
            "confidence": 0.95,
        }, ensure_ascii=False)
        plan = AgentPlanner.compile_client_output(
            "在黄色柜子旁边找到并夹取一个绿色小木块", raw, "deepseek_flash_api"
        )
        self.assertEqual(plan["intent"], "pick")
        patrol = next(step for step in plan["steps"] if step["action"] == "patrol_search")
        grasp = next(step for step in plan["steps"] if step["action"] == "model_grasp")
        self.assertEqual(patrol["target"]["query"], "黄色柜子")
        self.assertEqual(patrol["target"]["target_query"], "绿色小木块")
        self.assertEqual(grasp["target"]["query"], "绿色小木块")


if __name__ == "__main__":
    unittest.main()
