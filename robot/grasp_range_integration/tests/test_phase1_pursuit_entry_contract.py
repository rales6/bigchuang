"""Offline handoff tests; no cameras, BLE, model calls, or movement."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from grasp_range_integration.phase1.observation_recovery import install_observation_adapter
from grasp_range_integration.phase1.pursuit_entry_contract import assess_pursuit_entry


def pair(left_box, right_box):
    return [{
        "camera_id": camera, "class_name": "block", "source": "qwen",
        "confidence": .95, "bbox_1000": box, "stereo_pair_id": "target_1",
        "confirmed_match": True, "dual_camera_confirmed": True,
        "complete_in_frame": True,
        "search_evidence": {"source_ok": True, "class_ok": True,
                            "colour_ok": True, "bbox_valid": True,
                            "bbox_complete": True},
    } for camera, box in (("left", left_box), ("right", right_box))]


class PursuitEntryTests(unittest.TestCase):
    sizes = {"left": (1280, 720), "right": (1280, 720)}

    def test_actual_162902_edge_pair_is_not_handoff_ready(self):
        result = assess_pursuit_entry(
            pair([100, 95, 125, 155], [325, 185, 350, 235]), self.sizes)
        self.assertEqual(result["reason"], "left_outside_tracker_entry_roi")
        self.assertEqual(result["steering_camera"], "left")

    def test_centered_pair_is_ready(self):
        result = assess_pursuit_entry(
            pair([300, 195, 340, 255], [350, 205, 390, 265]), self.sizes)
        self.assertTrue(result["ready"])

    def test_second_valid_stereo_pair_is_selected_over_first_edge_pair(self):
        first = pair([100, 95, 125, 155], [325, 185, 350, 235])
        second = pair([300, 195, 340, 255], [350, 205, 390, 265])
        for item in second:
            item["stereo_pair_id"] = "target_2"
        result = assess_pursuit_entry(first + second, self.sizes)
        self.assertTrue(result["ready"])
        self.assertEqual(result["stereo_pair_id"], "target_2")

    def test_edge_pair_requeues_stopped_view_instead_of_reporting_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            images = {}
            for camera in ("left", "right"):
                path = Path(temporary) / (camera + ".jpg")
                Image.new("RGB", (1280, 720), "gray").save(path)
                images[camera] = path

            class Camera:
                def capture(self, *_args, **_kwargs):
                    raise AssertionError("hardware capture not expected")

            class Gpu:
                def perceive(self, *_args, **_kwargs):
                    raise AssertionError("model call not expected")

            class Log:
                def event(self, *_args, **_kwargs):
                    return None

            mission = SimpleNamespace(
                GpuMissionClient=Gpu, MissionLog=Log,
                confirm_detections=lambda items, *_args: items,
                _same_stereo_target=lambda *_args: True,
                _fuse_tracking_pose=lambda *_args: None)
            recovery = install_observation_adapter(mission, Camera)
            recovery.context = {"task_region_view": {"eligible": True},
                                "robot_pose": [0., 0., 0.]}
            checked = mission.confirm_detections(
                pair([100, 95, 125, 155], [325, 185, 350, 235]),
                {"color": "red", "class_name": "block"}, images)
            self.assertFalse(any(item["dual_camera_confirmed"] for item in checked))
            self.assertEqual(checked[0]["handoff_alignment_camera"], "left")
            route = {"views": [{"view_xy_m": [0., 0.]}],
                     "maximum_total_views": 8}
            state = {"latest_event": {"type": "observation_complete"},
                     "current_view_index": 0, "remaining_view_indices": [],
                     "attempts": 1, "detections": checked}
            first = recovery.next(state, route)
            self.assertEqual(first["recovery"]["step_deg"], 0.)
            state["remaining_view_indices"].clear()
            second = recovery.next(state, route)
            self.assertEqual(second["recovery"]["step_deg"], 6.)
            self.assertFalse(second["recovery"]["linear_motion_allowed"])


if __name__ == "__main__":
    unittest.main()
