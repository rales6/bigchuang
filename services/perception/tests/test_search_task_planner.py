from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.search_task_planner import InstructionCoordinator, SearchTaskPlanner


def scene() -> dict:
    return {
        "entities": {
            "object_10_1": {
                "target_id": "object_10_1",
                "target_x_m": 0.025,
                "target_y_m": 1.725,
                "semantic_label": "cabinet",
                "qwen_annotation": {
                    "semantic_label": "cabinet",
                    "name_zh": "柜子",
                    "category": "furniture",
                    "confidence": 0.82,
                },
            }
        }
    }


YOLO_CLASSES = ["person", "bottle", "cup", "chair", "book", "sports ball"]


class SearchPlannerTests(unittest.TestCase):
    def test_known_cabinet_and_supported_red_bottle_is_executable(self):
        decision = SearchTaskPlanner(YOLO_CLASSES).plan("去柜子边上找一个红色瓶子", scene())
        self.assertEqual(decision.status, "executable")
        self.assertEqual(decision.anchor["landmark_id"], "object_10_1")
        self.assertEqual(decision.target["class_name"], "bottle")
        self.assertEqual(decision.target["color"], "red")
        self.assertEqual(decision.steps[0]["action"], "localize")

    def test_road_search_is_rejected(self):
        decision = SearchTaskPlanner(YOLO_CLASSES).plan("去马路上找一个红色物体", scene())
        self.assertEqual(decision.status, "rejected")
        self.assertIn("地图之外", decision.reason)

    def test_missing_dustpan_requests_location_then_context_makes_executable(self):
        coordinator = InstructionCoordinator(SearchTaskPlanner(YOLO_CLASSES), scene())
        first = coordinator.submit("去簸箕旁找一个红色物品")
        self.assertEqual(first["status"], "needs_clarification")
        self.assertIn("右下角", first["clarification_question"])
        second = coordinator.submit("簸箕在地图的右下角")
        self.assertEqual(second["status"], "executable")
        self.assertEqual(second["route_request"]["mode"], "coarse_region")
        self.assertEqual(second["route_request"]["region_hint"], "bottom_right")

    def test_location_followup_bypasses_a_model_that_would_replace_the_target(self):
        class WrongFollowupBackend:
            def generate(self, _prompt):
                return json.dumps(
                    {
                        "intent": "search_object",
                        "target_query": "dustpan",
                        "target_class": "dustpan",
                        "target_color": "",
                        "anchor_query": "",
                        "region_hint": "bottom_right",
                        "environment": "mapped_indoor",
                    }
                )

        coordinator = InstructionCoordinator(
            SearchTaskPlanner(YOLO_CLASSES, WrongFollowupBackend()), scene()
        )
        first = coordinator.submit("去簸箕旁找一个红色物品")
        second = coordinator.submit("簸箕在地图的右下角")
        self.assertEqual(second["status"], "executable")
        self.assertEqual(second["target"]["color"], "red")
        self.assertEqual(second["target"]["class_name"], "any yolo object")
        self.assertEqual(second["source"], "deterministic_context_merge")

    def test_small_block_falls_back_to_qwen_when_yolo_has_no_block_class(self):
        decision = SearchTaskPlanner(YOLO_CLASSES).plan("去柜子边找红色小物块", scene())
        self.assertEqual(decision.status, "executable")
        self.assertEqual(decision.target["detection_mode"], "qwen_fallback")
        self.assertTrue(decision.capabilities["qwen_visual_fallback"])
        self.assertFalse(decision.capabilities["class_supported"])
        self.assertEqual(decision.steps[2]["action"], "navigate_to_search_area")
        self.assertEqual(decision.steps[4]["action"], "visual_detect_and_filter")

    def test_unverified_release_region_is_searchable_but_not_a_landmark(self):
        memory = {
            "schema": "semantic_object_memory.v1",
            "objects": [{
                "id": "cabinet_candidate",
                "name": "柜子",
                "category": "cabinet",
                "map_xy": [1.0, -2.0],
                "position_status": "estimated",
                "navigation_allowed": False,
                "position_verified": False,
                "confidence": 0.93,
            }],
            "search_regions": [{
                "id": "cabinet_candidate",
                "query": "柜子",
                "map_xy": [1.0, -2.0],
                "radius_m": 1.2,
                "requires_visual_recheck": True,
            }],
        }
        decision = SearchTaskPlanner(YOLO_CLASSES).plan(
            "去柜子旁找红色瓶子", memory
        )
        self.assertEqual(decision.status, "executable")
        self.assertIsNone(decision.anchor["landmark_id"])
        self.assertFalse(decision.anchor["position_verified"])
        self.assertTrue(decision.anchor["requires_visual_recheck"])
        self.assertEqual(decision.route_request["mode"], "semantic_search_region")
        self.assertTrue(decision.route_request["require_live_visual_recheck"])

    def test_new_command_preempts_idle_annotation_and_deduplicates(self):
        coordinator = InstructionCoordinator(SearchTaskPlanner(YOLO_CLASSES), scene())
        idle = coordinator.request_idle_annotation("map_sha:object_1:view_1")
        self.assertTrue(idle["accepted"])
        command = coordinator.submit("去柜子旁找红色瓶子")
        self.assertEqual(command["preempted_task_id"], idle["task_id"])
        self.assertIsNone(coordinator.snapshot()["idle_annotation"])
        coordinator.complete(command["task_id"], "not_found")
        idle2 = coordinator.request_idle_annotation("same")
        coordinator.finish_idle_annotation(idle2["task_id"], True)
        duplicate = coordinator.request_idle_annotation("same")
        self.assertEqual(duplicate["reason"], "duplicate_annotation")


if __name__ == "__main__":
    unittest.main()
