from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.semantic_task_planner import (
    SemanticTaskPlanner,
    build_qwen_intent_prompt,
    build_qwen_planning_prompt,
    parse_qwen_plan,
)
from qwen_grounded_tracker.task_dry_run import SymbolicDryRunExecutor
from qwen_grounded_tracker.task_execution import (
    HttpRobotBridge,
    PlanExecutionEngine,
    SimulationRobotBridge,
)


def load_scene() -> dict:
    return json.loads((ROOT / "examples/demo_cleanup_scene.json").read_text(encoding="utf-8"))


class SemanticTaskPlannerTests(unittest.TestCase):
    def test_cleanup_plan_contains_full_safe_loop(self) -> None:
        plan = SemanticTaskPlanner().plan("帮我处理地上的垃圾", load_scene())
        actions = [step.action for step in plan.steps]
        self.assertEqual(
            actions,
            [
                "inspect_semantic_map",
                "patrol_search",
                "locate_object",
                "navigate_to",
                "align_for_grasp",
                "grasp",
                "verify_grasp",
                "locate_destination",
                "navigate_to",
                "release",
                "verify_completion",
                "report_result",
            ],
        )
        self.assertEqual(plan.steps[7].target["landmark_id"], "dustpan_01")
        for step in plan.steps:
            if step.action in {"patrol_search", "navigate_to", "return_home"}:
                self.assertTrue({"lidar", "camera", "semantic_map"}.issubset(set(step.sensors)))

    def test_cleanup_dry_run_finds_dynamic_trash_and_reaches_dustpan(self) -> None:
        scene = load_scene()
        plan = SemanticTaskPlanner().plan("帮我处理地上的垃圾", scene)
        result = SymbolicDryRunExecutor().execute(plan, scene)
        self.assertTrue(result.completed)
        self.assertIsNone(result.carrying)
        self.assertEqual(result.final_robot_xy, [19, 12])
        self.assertGreater(len(result.path_cells), 0)
        self.assertTrue(all(item["hardware_command_sent"] is False for item in result.trace))

    def test_prompt_forbids_raw_control_and_contains_map_landmark(self) -> None:
        prompt = build_qwen_planning_prompt("帮我处理地上的垃圾", load_scene())
        self.assertIn("不直接输出电机转速", prompt)
        self.assertIn("dustpan_01", prompt)
        self.assertIn("lidar", prompt)
        self.assertIn("camera", prompt)

    def test_rejects_qwen_navigation_without_dual_safety(self) -> None:
        raw = json.dumps(
            {
                "schema": "robot_task_plan.v1",
                "task_id": "bad",
                "instruction": "前进",
                "intent": "navigate",
                "summary": "bad",
                "assumptions": [],
                "global_constraints": [],
                "steps": [
                    {
                        "id": "step_01",
                        "action": "navigate_to",
                        "description": "move",
                        "depends_on": [],
                        "target": {"query": "前方"},
                        "sensors": ["camera"],
                        "success_condition": "arrived",
                        "on_failure": "safe_stop",
                    }
                ],
            },
            ensure_ascii=False,
        )
        with self.assertRaisesRegex(ValueError, "lacks lidar/camera/semantic_map"):
            parse_qwen_plan(raw, "前进")

    def test_qwen_semantic_error_uses_safe_rule_fallback(self) -> None:
        class WeakBackend:
            def generate(self, prompt: str) -> str:
                del prompt
                return '{"intent":"cleanup","target_query":""}'

        plan = SemanticTaskPlanner(WeakBackend()).plan("帮我处理地上的垃圾", load_scene())
        self.assertEqual(plan.source, "rules_fallback")
        self.assertIn("missing target_query", plan.warnings[0])
        result = SymbolicDryRunExecutor().execute(plan, load_scene())
        self.assertTrue(result.completed)

    def test_valid_qwen_contract_is_compiled_to_safe_steps(self) -> None:
        class ContractBackend:
            def generate(self, prompt: str) -> str:
                self.prompt = prompt
                return json.dumps(
                    {
                        "intent": "cleanup",
                        "target_query": "地面纸团",
                        "target_landmark_id": None,
                        "destination_query": "绿色簸箕",
                        "destination_landmark_id": "dustpan_01",
                        "requested_count": 1,
                        "needs_scene_patrol": True,
                        "return_home": False,
                        "confidence": 0.93,
                    },
                    ensure_ascii=False,
                )

        backend = ContractBackend()
        plan = SemanticTaskPlanner(backend).plan("帮我处理地上的垃圾", load_scene())
        self.assertEqual(plan.source, "qwen_compiled")
        self.assertEqual(plan.model_interpretation["destination_landmark_id"], "dustpan_01")
        self.assertEqual(len(plan.steps), 12)
        self.assertIn("不要规划动作步骤", backend.prompt)
        self.assertTrue(SymbolicDryRunExecutor().execute(plan, load_scene()).completed)

    def test_generic_pick_and_place_uses_model_target_and_destination(self) -> None:
        class BottleBackend:
            def generate(self, prompt: str) -> str:
                del prompt
                return json.dumps(
                    {
                        "intent": "pick_and_place",
                        "target_query": "蓝色水瓶",
                        "target_landmark_id": None,
                        "destination_query": "工作桌",
                        "destination_landmark_id": "table_01",
                        "requested_count": 1,
                        "needs_scene_patrol": True,
                        "return_home": False,
                        "confidence": 0.91,
                    },
                    ensure_ascii=False,
                )

        scene = load_scene()
        plan = SemanticTaskPlanner(BottleBackend()).plan("把蓝色水瓶放到工作桌上", scene)
        self.assertEqual(plan.task_id, "semantic_pick_and_place")
        self.assertEqual(plan.steps[2].target["query"], "蓝色水瓶")
        self.assertEqual(plan.steps[7].target["landmark_id"], "table_01")
        self.assertIn("蓝色水瓶", plan.summary)
        result = SymbolicDryRunExecutor().execute(plan, scene)
        self.assertTrue(result.completed)
        self.assertEqual(result.final_robot_xy, [10, 7])

    def test_existing_but_semantically_wrong_landmark_id_is_ignored(self) -> None:
        class WrongIdBackend:
            def generate(self, prompt: str) -> str:
                del prompt
                return json.dumps(
                    {
                        "intent": "pick_and_place",
                        "target_query": "蓝色水瓶",
                        "target_landmark_id": "cabinet_01",
                        "destination_query": "工作桌",
                        "destination_landmark_id": "table_01",
                        "requested_count": 1,
                        "needs_scene_patrol": True,
                        "return_home": False,
                        "confidence": 0.96,
                    },
                    ensure_ascii=False,
                )

        plan = SemanticTaskPlanner(WrongIdBackend()).plan("把蓝色水瓶放到工作桌上", load_scene())
        self.assertIsNone(plan.steps[2].target["landmark_id"])
        self.assertEqual(plan.steps[7].target["landmark_id"], "table_01")
        self.assertIn("语义不一致", plan.warnings[0])

    def test_navigation_contract_resolves_known_table(self) -> None:
        class NavigationBackend:
            def generate(self, prompt: str) -> str:
                del prompt
                return json.dumps(
                    {
                        "intent": "navigate",
                        "target_query": "工作桌",
                        "target_landmark_id": "table_01",
                        "destination_query": "",
                        "destination_landmark_id": None,
                        "requested_count": 1,
                        "needs_scene_patrol": False,
                        "return_home": False,
                        "confidence": 0.95,
                    },
                    ensure_ascii=False,
                )

        plan = SemanticTaskPlanner(NavigationBackend()).plan("去工作桌旁边", load_scene())
        self.assertEqual(plan.task_id, "generic_semantic_navigation")
        self.assertEqual(plan.steps[0].target["landmark_id"], "table_01")
        self.assertEqual([step.action for step in plan.steps], [
            "inspect_semantic_map", "patrol_search", "navigate_to", "report_result"
        ])

    def test_explicit_navigation_corrects_model_example_copy(self) -> None:
        class CopyBiasedBackend:
            def generate(self, prompt: str) -> str:
                del prompt
                return json.dumps(
                    {
                        "intent": "cleanup",
                        "target_query": "地面垃圾",
                        "target_landmark_id": None,
                        "destination_query": "簸箕",
                        "destination_landmark_id": "dustpan_01",
                        "requested_count": 1,
                        "needs_scene_patrol": True,
                        "return_home": False,
                        "confidence": 0.0,
                    },
                    ensure_ascii=False,
                )

        plan = SemanticTaskPlanner(CopyBiasedBackend()).plan("去工作桌旁边", load_scene())
        self.assertEqual(plan.intent, "navigate")
        self.assertEqual(plan.steps[0].target["landmark_id"], "table_01")
        self.assertNotIn("grasp", [step.action for step in plan.steps])
        self.assertTrue(plan.model_interpretation["corrected_by_rules"])
        self.assertIn("已纠正", plan.warnings[-1])

    def test_unsupported_contract_never_generates_motion(self) -> None:
        class UnsupportedBackend:
            def generate(self, prompt: str) -> str:
                del prompt
                return json.dumps(
                    {
                        "intent": "unsupported",
                        "target_query": "唱歌",
                        "target_landmark_id": None,
                        "destination_query": "",
                        "destination_landmark_id": None,
                        "requested_count": 1,
                        "needs_scene_patrol": False,
                        "return_home": False,
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                )

        plan = SemanticTaskPlanner(UnsupportedBackend()).plan("给我唱一首歌", load_scene())
        self.assertEqual(plan.intent, "unsupported")
        self.assertEqual([step.action for step in plan.steps], ["report_result"])
        self.assertEqual(plan.source, "qwen_compiled")

    def test_intent_prompt_is_compact_contract_not_action_plan(self) -> None:
        prompt = build_qwen_intent_prompt("帮我处理地上的垃圾", load_scene())
        self.assertIn("intent、target_query", prompt)
        self.assertIn("intent=cleanup", prompt)
        self.assertIn("不得照抄", prompt)
        self.assertIn("不要规划动作步骤", prompt)

    def test_execution_engine_runs_all_compiled_steps_without_hardware(self) -> None:
        plan = SemanticTaskPlanner().plan("帮我处理地上的垃圾", load_scene())
        bridge = SimulationRobotBridge()
        engine = PlanExecutionEngine(bridge)
        engine.load(plan)
        engine.start()
        state = engine.run_until_terminal()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(bridge.commands), 12)
        self.assertFalse(state["hardware_enabled"])
        self.assertTrue(all(command["hardware_command_sent"] is False for command in bridge.commands))
        navigation = [command for command in bridge.commands if command["action"] in {"patrol_search", "navigate_to"}]
        self.assertTrue(all(command["safety_contract"]["require_lidar_clearance"] for command in navigation))
        self.assertTrue(all(command["safety_contract"]["require_camera_clearance"] for command in navigation))
        self.assertFalse(any("motor" in json.dumps(command).lower() for command in bridge.commands))
        self.assertFalse(any("servo" in json.dumps(command).lower() for command in bridge.commands))

    def test_execution_failure_requests_safe_stop(self) -> None:
        plan = SemanticTaskPlanner().plan("帮我处理地上的垃圾", load_scene())
        bridge = SimulationRobotBridge(blocked_action="navigate_to")
        engine = PlanExecutionEngine(bridge)
        engine.load(plan)
        engine.start()
        state = engine.run_until_terminal()
        self.assertEqual(state["status"], "blocked")
        self.assertEqual(len(bridge.stop_reasons), 1)
        self.assertIn("navigate_to", bridge.stop_reasons[0])

    def test_http_hardware_bridge_requires_safety_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a non-empty safety token"):
            HttpRobotBridge("http://127.0.0.1:8767", hardware_enabled=True)


if __name__ == "__main__":
    unittest.main()
