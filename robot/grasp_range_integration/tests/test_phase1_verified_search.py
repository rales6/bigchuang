"""Offline contract tests; these tests never start cameras, BLE, or an arm."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from grasp_range_integration.phase1 import verified_search
from grasp_range_integration.phase1.verified_search import verify_handoff


def decision():
    return {
        "schema": "robot_object_search_plan.v1", "status": "executable",
        "instruction": "去柜子边上夹取一个红色的小物块", "task_id": "test_pick",
        "target": {"query": "红色小物块", "color": "red", "class_name": "block",
                   "requested_count": 1},
        "anchor": {"search_region_id": "cabinet", "position_verified": False},
        "route_request": {"mode": "semantic_search_region",
                          "planning_policy": "semantic_visibility_regions.v1",
                          "inspect_current_view_first": True,
                          "stop_on_visibility_entry": True,
                          "center_xy_m": [1.0, 1.0]},
        "steps": [{"action": "localize"}],
        "capabilities": {"grasp_enabled": False, "arm_commands_allowed": False},
        "phase_control": {"phase": "phase1_find_only",
                          "terminal_condition": "target_found_and_vehicle_stopped",
                          "requested_final_action": "grasp"},
    }


def sample_mission():
    def item(camera):
        return {"camera_id": camera, "class_name": "block",
                "bbox_1000": [300, 200, 340, 260] if camera == "left"
                else [350, 210, 390, 270],
                "stereo_pair_id": "target_1", "task_region_eligible": True,
                "dual_camera_confirmed": True, "confirmed_match": True,
                "search_evidence": {"source_ok": True, "class_ok": True,
                                    "colour_ok": True, "bbox_complete": True}}

    return {
        "instruction": decision()["instruction"], "result": "success",
        "hardware_motion_enabled": True, "finished_at": "2026-09-29 12:00:00",
        "events": [
            {"type": "observation_complete", "dual_camera_confirmed": True,
             "complete_cameras": ["left", "right"], "observation_id": "obs_0002",
             "detections": [item("left"), item("right")],
             "archived_images": {"left": "observations/left.jpg",
                                 "right": "observations/right.jpg"}},
            {"type": "supervisor_directive",
             "directive": {"action": "report_success", "terminal": True}},
        ],
    }


class VerifiedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mission_dir = self.root / "20260929_120000"
        photos = self.mission_dir / "observations"
        photos.mkdir(parents=True)
        Image.new("RGB", (1280, 720), "gray").save(photos / "left.jpg")
        Image.new("RGB", (1280, 720), "gray").save(photos / "right.jpg")
        self.mission = sample_mission()

    def verify(self):
        (self.mission_dir / "mission.json").write_text(
            json.dumps(self.mission, ensure_ascii=False), encoding="utf-8")
        return verify_handoff(decision(), self.mission_dir, output_root=self.root)

    def test_confirmed_stereo_handoff_is_not_arm_authority(self):
        result = self.verify()
        self.assertEqual(result["observation_id"], "obs_0002")
        self.assertFalse(result["object_range_established"])
        self.assertFalse(result["arm_grasp_authorized"])
        self.assertTrue(result["pursuit_entry"]["ready"])

    def test_green_bottle_handoff_preserves_target_for_fresh_model_center_check(self):
        other = decision()
        other["instruction"] = "去柜子边上夹取绿色瓶子"
        other["target"] = {"query": "绿色瓶子", "color": "green",
                           "class_name": "bottle", "requested_count": 1}
        self.mission["instruction"] = other["instruction"]
        for item in self.mission["events"][0]["detections"]:
            item["class_name"] = "bottle"
            item["attributes"] = {"color": "green"}
        (self.mission_dir / "mission.json").write_text(
            json.dumps(self.mission, ensure_ascii=False), encoding="utf-8")
        handoff = verify_handoff(other, self.mission_dir, output_root=self.root)
        self.assertEqual(handoff["target"]["query"], "绿色瓶子")
        self.assertFalse(handoff["arm_grasp_authorized"])

    def test_stereo_box_at_tracker_left_edge_does_not_handoff(self):
        self.mission["events"][0]["detections"][0]["bbox_1000"] = [100, 95, 125, 155]
        with self.assertRaisesRegex(ValueError, "left_outside_tracker_entry_roi"):
            self.verify()

    def test_zero_exit_failure_result_does_not_handoff(self):
        self.mission["result"] = "failure"
        with self.assertRaisesRegex(ValueError, "未确认成功"):
            self.verify()

    def test_region_outside_detection_does_not_handoff(self):
        self.mission["events"][0]["detections"][0]["task_region_eligible"] = False
        with self.assertRaisesRegex(ValueError, "左右双摄"):
            self.verify()

    def test_one_camera_or_copied_pair_does_not_handoff(self):
        self.mission["events"][0]["detections"][1]["stereo_pair_id"] = "different"
        with self.assertRaisesRegex(ValueError, "左右双摄"):
            self.verify()

    def test_old_or_escaped_result_directory_rejected(self):
        outside = self.root.parent / "outside_test_mission"
        with self.assertRaises((ValueError, FileNotFoundError)):
            verify_handoff(decision(), outside, output_root=self.root)

    def test_wrapper_uses_only_a_new_successful_run(self):
        decision_path = self.root / "decision.json"
        decision_path.write_text(json.dumps(decision(), ensure_ascii=False), encoding="utf-8")
        old = self.root / "old_success"
        old.mkdir()

        class FakeProcess:
            def __init__(inner, command, **_kwargs):
                self.assertEqual(command[-1], "--execute-find-only")
                (self.mission_dir / "mission.json").write_text(
                    json.dumps(self.mission, ensure_ascii=False), encoding="utf-8")
                inner.stdout = StringIO("寻物任务结果目录：" + str(self.mission_dir))

            def wait(inner):
                return 0

        # The mission directory was prepared before run(), so it must not be
        # accepted as a new task even though its JSON would pass pure checks.
        with self.assertRaisesRegex(RuntimeError, "恰好一个"):
            verified_search.run(decision_path, runner=FakeProcess, output_root=self.root)

        fresh = self.root / "20260929_120500"

        class FreshProcess:
            def __init__(inner, command, **_kwargs):
                self.assertEqual(command[-1], "--execute-find-only")
                (fresh / "observations").mkdir(parents=True)
                Image.new("RGB", (1280, 720), "gray").save(fresh / "observations/left.jpg")
                Image.new("RGB", (1280, 720), "gray").save(fresh / "observations/right.jpg")
                (fresh / "mission.json").write_text(
                    json.dumps(self.mission, ensure_ascii=False), encoding="utf-8")
                inner.stdout = StringIO("")

            def wait(inner):
                return 0

        result = verified_search.run(decision_path, runner=FreshProcess,
                                     output_root=self.root)
        self.assertEqual(result["mission_dir"], str(fresh))
        self.assertTrue((fresh / "phase1_verified_handoff.json").is_file())


if __name__ == "__main__":
    unittest.main()
