from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.scene_change_memory import SceneChangeMemory
from qwen_grounded_tracker.semantic_task_planner import SemanticTaskPlanner
from qwen_grounded_tracker.task_dry_run import SymbolicDryRunExecutor


def load_scene() -> dict:
    return json.loads((ROOT / "examples/demo_cleanup_scene.json").read_text(encoding="utf-8"))


class SceneChangeMemoryTests(unittest.TestCase):
    def test_overlay_persists_without_mutating_baseline(self) -> None:
        baseline = load_scene()
        original = copy.deepcopy(baseline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changes.json"
            memory = SceneChangeMemory(path, map_id=baseline["map_id"])
            recorded = memory.record(
                change_type="moved",
                expected_item_id="bottle_01",
                expected_query="蓝色水瓶",
                grid_xy=[12, 10],
                observed_item_id="bottle_01",
                confidence=0.91,
            )
            overlay = memory.overlay(baseline)
            reloaded = SceneChangeMemory(path)
        self.assertEqual(baseline, original)
        self.assertFalse(overlay["baseline_map_mutated"])
        self.assertEqual(recorded["id"], reloaded.locations[0]["id"])
        self.assertTrue(reloaded.locations[0]["requires_recheck_on_every_visit"])

    def test_future_visit_rechecks_moved_item_and_reaches_new_location(self) -> None:
        scene = load_scene()
        scene["objects"].append({
            "id": "bottle_01",
            "name": "蓝色水瓶",
            "category": "bottle",
            "grid_xy": [12, 9],
            "approach_xy": [12, 10],
            "confidence": 0.90,
        })
        memory = SceneChangeMemory(map_id=scene["map_id"])
        memory.record(
            change_type="moved",
            expected_item_id="bottle_01",
            expected_query="蓝色水瓶",
            grid_xy=[12, 10],
            observed_item_id="bottle_01",
            confidence=0.90,
        )
        planned_scene = memory.overlay(scene)
        plan = SemanticTaskPlanner().plan("去蓝色水瓶旁边看看它还在不在", planned_scene)
        patrol = next(step for step in plan.steps if step.action == "patrol_search")
        self.assertTrue(patrol.target["require_scene_recheck"])
        self.assertEqual(patrol.condition, "if_target_missing_stale_or_suspicious")
        result = SymbolicDryRunExecutor(memory).execute(plan, planned_scene)
        self.assertTrue(result.completed)
        self.assertGreaterEqual(len(result.suspicious_rechecks), 1)
        bottle = next(item for item in scene["simulated_detections"] if item["id"] == "bottle_01")
        self.assertEqual(result.final_robot_xy, bottle["approach_xy"])

    def test_missing_local_grasp_model_uses_requested_simulation_success(self) -> None:
        scene = load_scene()
        scene["grasp_model"] = {"available": False}
        plan = SemanticTaskPlanner().plan("把蓝色水瓶放到工作桌上", scene)
        result = SymbolicDryRunExecutor().execute(plan, scene)
        self.assertTrue(result.completed)
        self.assertEqual(result.grasp_attempts, result.grasp_successes)
        self.assertEqual(result.grasp_model_probe["mode"], "skipped_not_installed")


if __name__ == "__main__":
    unittest.main()
