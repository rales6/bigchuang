from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grasp_range_integration.phase1.relationship_find_only_entry import (
    _install_stationary_start_localization,
)


class FakeClient:
    def __init__(self):
        self.stops = 0

    def stop(self):
        self.stops += 1


class StationaryStartTests(unittest.TestCase):
    def setUp(self):
        self.prior = SimpleNamespace(
            x_m=-1.55, y_m=-2.60, yaw_deg=100.0,
            radius_m=0.70, yaw_tolerance_deg=40.0,
        )
        self.pose = SimpleNamespace(
            x_m=-1.575, y_m=-2.795, yaw_rad=math.radians(105.0),
        )
        self.candidate = SimpleNamespace(
            pose=self.pose, reliable=True, rmse_m=0.05, inlier_ratio=0.85,
        )
        self.args = SimpleNamespace(
            localization_window=5,
            localization_min_accepted=3,
            localization_timeout_seconds=10.0,
        )

    def _run(self, visual_status: str, visual_pose=None, diagnostics=None, second_status=None):
        candidate = self.candidate
        winner = SimpleNamespace(pose=visual_pose or self.pose, reliable=True)
        if diagnostics is None:
            diagnostics = {"engine": "fake"}
        self.visual_attempts = []
        def check(*args):
            self.visual_attempts.append(args[-1])
            if args[-1] == 2 and second_status is not None:
                return second_status, candidate, object(), diagnostics
            return visual_status, winner, object(), diagnostics
        mission = SimpleNamespace(
            _localization_acceptance=lambda rows, _args: (
                len(rows) >= 4, candidate, 1.0, len(rows), len(rows)
            ),
            _visual_localization_check=check,
        )
        _install_stationary_start_localization(mission, self.prior)
        localizer = SimpleNamespace(global_localize=lambda _scan, hint=None: candidate)
        camera = object()
        recognizer = SimpleNamespace(available=True)
        client = FakeClient()
        events = []
        result = mission._localize(
            self.args, client, iter([object() for _ in range(7)]),
            localizer, object(), lambda: False,
            camera=camera, visual_recognizer=recognizer,
            visual_directory=Path("search_runs"),
            event_callback=lambda event, **data: events.append((event, data)),
        )
        return result, client, events

    def test_conflict_recaptures_once_after_two_fresh_stationary_scans(self):
        result, client, events = self._run('conflict', second_status='confirmed')
        self.assertEqual(self.visual_attempts, [1, 2])
        self.assertEqual(result[0], self.pose)
        self.assertGreaterEqual(client.stops, 3)
        self.assertEqual(sum(event == 'startup_scan_scored' and data.get('retry', False)
                             for event, data in events), 2)
        final = next(data for event, data in events if event == 'stationary_start_checked')
        self.assertEqual(final['visual_attempts'], 2)

    def test_persistent_conflict_has_only_two_capture_attempts(self):
        with self.assertRaisesRegex(RuntimeError, '双摄复核未通过'):
            self._run('conflict')
        self.assertEqual(self.visual_attempts, [1, 2])

    def test_stationary_lidar_and_stereo_avoid_startup_motion(self):
        result, client, events = self._run("confirmed")
        self.assertEqual(result[0], self.pose)
        self.assertGreaterEqual(client.stops, 2)
        self.assertTrue(any(
            event == "stationary_start_checked" and data["verified"]
            for event, data in events
        ))

    def test_visual_pose_outside_start_envelope_fails_closed(self):
        far_pose = SimpleNamespace(
            x_m=0.8, y_m=0.0, yaw_rad=math.radians(-90.0)
        )
        with self.assertRaisesRegex(RuntimeError, "双摄复核未通过"):
            self._run("corrected", far_pose)

    def test_generic_uncertainty_does_not_approve_motion(self):
        with self.assertRaisesRegex(RuntimeError, "双摄复核未通过"):
            self._run("uncertain")

    def test_two_camera_deepseek_abstention_uses_strong_lidar_start(self):
        result, _client, events = self._run(
            "uncertain", diagnostics={
                "engine": "deepseek_vision_candidate_ranker",
                "camera_count": 2,
                "deepseek_result": {
                    "selected_candidate_id": "uncertain",
                    "consistent_repeats": 2,
                },
            },
        )
        self.assertEqual(result[0], self.pose)
        check = next(data for event, data in events if event == "stationary_start_checked")
        self.assertTrue(check["approved"])
        self.assertFalse(check["verified"])
        self.assertEqual(check["basis"], "lidar_start_prior_visual_abstention")

    def test_visual_conflict_still_stops(self):
        with self.assertRaisesRegex(RuntimeError, "双摄复核未通过"):
            self._run("conflict")

    def test_api_error_or_single_camera_still_stops(self):
        for extra in ({"deepseek_error": "timeout"}, {"camera_count": 1}):
            diagnostics = {
                "engine": "deepseek_vision_candidate_ranker",
                "camera_count": 2,
                "deepseek_result": {
                    "selected_candidate_id": "uncertain",
                    "consistent_repeats": 2,
                },
                **extra,
            }
            with self.subTest(extra=extra), self.assertRaisesRegex(
                RuntimeError, "双摄复核未通过"
            ):
                self._run("uncertain", diagnostics=diagnostics)

    def test_weak_lidar_still_stops_on_visual_abstention(self):
        self.candidate.inlier_ratio = 0.65
        diagnostics = {
            "engine": "deepseek_vision_candidate_ranker",
            "camera_count": 2,
            "deepseek_result": {
                "selected_candidate_id": "uncertain",
                "consistent_repeats": 2,
            },
        }
        with self.assertRaisesRegex(RuntimeError, "双摄复核未通过"):
            self._run("uncertain", diagnostics=diagnostics)

    def test_production_client_routes_failed_static_review_into_active_recovery(self):
        self.candidate.inlier_ratio = .5072
        with patch.object(FakeClient, 'startup_recovery_pulse', create=True), \
             patch('grasp_range_integration.phase1.startup_active_recovery.recover_startup',
                   return_value=(self.pose, object())) as active:
            result, client, events = self._run('uncertain')
        active.assert_called_once()
        self.assertEqual(result[0], self.pose)
        self.assertEqual(self.visual_attempts, [1,2])
        self.assertTrue(any(name == 'startup_static_fallback' for name,_ in events))


if __name__ == "__main__":
    unittest.main()
