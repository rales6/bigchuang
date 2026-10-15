from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.mission_supervisor import MissionSupervisor


class Backend:
    def __init__(self, value):
        self.value = value

    def generate(self, _prompt):
        return json.dumps(self.value, ensure_ascii=False)


class MissionSupervisorTests(unittest.TestCase):
    def state(self, **updates):
        value = {
            "remaining_view_indices": [1, 2],
            "latest_event": {"type": "observation_complete"},
            "detections": [],
        }
        value.update(updates)
        return value

    def test_qwen_cannot_claim_success_without_confirmed_detection(self):
        result = MissionSupervisor(Backend({"action": "report_success"})).next(self.state())
        self.assertEqual(result.action, "navigate_next_view")

    def test_paired_complete_dual_camera_match_stops_successfully(self):
        result = MissionSupervisor(Backend({"action": "navigate_next_view"})).next(
            self.state(detections=[
                {
                    "camera_id": "left", "confirmed_match": True,
                    "dual_camera_confirmed": True, "stereo_pair_id": "target-1",
                },
                {
                    "camera_id": "right", "confirmed_match": True,
                    "dual_camera_confirmed": True, "stereo_pair_id": "target-1",
                },
            ])
        )
        self.assertEqual(result.action, "report_success")
        self.assertTrue(result.terminal)

    def test_single_camera_confirmed_match_cannot_finish(self):
        result = MissionSupervisor(Backend({"action": "report_success"})).next(
            self.state(detections=[{
                "camera_id": "left", "confirmed_match": True,
                "dual_camera_confirmed": True, "stereo_pair_id": "target-1",
            }])
        )
        self.assertEqual(result.action, "navigate_next_view")

    def test_temporary_obstacle_forces_replan(self):
        result = MissionSupervisor().next(
            self.state(latest_event={"type": "temporary_obstacle"})
        )
        self.assertEqual(result.action, "replan")

    def test_navigation_no_progress_forces_replan(self):
        result = MissionSupervisor().next(
            self.state(latest_event={"type": "navigation_no_progress"})
        )
        self.assertEqual(result.action, "replan")

    def test_exhaustion_is_failure(self):
        result = MissionSupervisor().next(self.state(remaining_view_indices=[]))
        self.assertEqual(result.action, "report_failure")


if __name__ == "__main__":
    unittest.main()
