from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from raspberry_pi.mapping import Pose2D

from grasp_range_integration.phase1.relationship_find_only_entry import (
    _install_midrun_visual_recheck,
)


class MidrunVisualRecheckTests(unittest.TestCase):
    def setUp(self):
        self.pose = Pose2D(x_m=-1.4, y_m=-1.5, yaw_rad=2.0)
        self.good = SimpleNamespace(pose=self.pose, usable=True)
        self.bad = SimpleNamespace(pose=self.pose, usable=False)
        self.client = SimpleNamespace(stops=0)
        self.client.stop = lambda: setattr(self.client, "stops", self.client.stops + 1)
        self.controller = SimpleNamespace(stops=0)
        self.controller.stopped = lambda: setattr(
            self.controller, "stops", self.controller.stops + 1
        )
        self.events = []
        self.calls = []

    def _install(self, visual_status="confirmed", visual_result=None, track_rows=None):
        rows = iter(track_rows or [self.bad] * 4)
        localizer = SimpleNamespace(
            track=lambda *_: next(rows),
            localize_near_position=lambda *_args, **_kwargs: self.bad,
            global_localization_candidates=lambda *_args, **_kwargs: [self.good],
        )

        def visual(*_args):
            self.calls.append("visual")
            return visual_status, visual_result or self.good, object(), {"engine": "fake"}

        mission = SimpleNamespace(
            _tracking_usable_for_photo_mission=lambda result, _args: bool(
                result is not None and result.usable
            ),
            _visual_localization_check=visual,
        )
        _install_midrun_visual_recheck(
            mission, localizer, object(), self.client, self.controller,
            object(), object(), object(), Path("search_runs"),
            lambda name, **data: self.events.append((name, data)),
        )
        return localizer

    def test_second_bad_scan_stops_and_accepts_near_visual_lidar_pose(self):
        localizer = self._install()
        self.assertIsNone(localizer.track(object(), self.pose))
        self.assertEqual(self.client.stops, 0)
        self.assertIs(localizer.track(object(), self.pose), self.good)
        self.assertEqual(self.client.stops, 1)
        self.assertEqual(self.controller.stops, 1)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.events[-1][1]["accepted"])

    def test_visual_conflict_does_not_reset_predicted_pose(self):
        far = SimpleNamespace(
            pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=-1.0), usable=True
        )
        localizer = self._install(visual_result=far)
        localizer.track(object(), self.pose)
        self.assertIsNone(localizer.track(object(), self.pose))
        self.assertFalse(self.events[-1][1]["accepted"])
        localizer.track(object(), self.pose)
        self.assertEqual(len(self.calls), 1)

    def test_good_track_resets_failure_streak(self):
        localizer = self._install(track_rows=[self.bad, self.good, self.bad])
        localizer.track(object(), self.pose)
        localizer.track(object(), self.pose)
        localizer.track(object(), self.pose)
        self.assertEqual(self.client.stops, 0)
        self.assertEqual(self.calls, [])

    def test_stationary_clearance_recheck_works_with_old_localizer_interface(self):
        localizer = self._install()
        self.good.rmse_m, self.good.inlier_ratio, self.good.score = .04, .8, 1.
        localizer.localize_near_position = lambda *a, **k: self.good
        self.assertIs(localizer._phase1_stationary_map_recheck(object(), self.pose), self.good)
        self.assertEqual(self.client.stops, 1)
        self.assertEqual(self.calls, [])
        self.assertTrue(self.events[-1][1]['accepted'])

    def test_stationary_clearance_recheck_uses_vision_after_weak_lidar(self):
        localizer = self._install()
        self.good.rmse_m, self.good.inlier_ratio, self.good.score = .04, .8, 1.
        self.bad.rmse_m, self.bad.inlier_ratio = .2, .3
        self.assertIs(localizer._phase1_stationary_map_recheck(object(), self.pose), self.good)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.client.stops, 1)

    def test_stationary_clearance_recheck_rejects_far_visual_pose(self):
        far = SimpleNamespace(pose=SimpleNamespace(x_m=self.pose.x_m+.25,
                y_m=self.pose.y_m, yaw_rad=self.pose.yaw_rad), usable=True,
                rmse_m=.04, inlier_ratio=.8, score=1.)
        localizer = self._install(visual_result=far)
        self.bad.rmse_m, self.bad.inlier_ratio = .2, .3
        self.assertIsNone(localizer._phase1_stationary_map_recheck(object(), self.pose))
        self.assertFalse(self.events[-1][1]['accepted'])

    def test_stationary_clearance_recheck_records_lidar_exception(self):
        localizer = self._install()
        def failed(*a, **k): raise RuntimeError('test scan fault')
        localizer.localize_near_position = failed
        self.assertIsNone(localizer._phase1_stationary_map_recheck(object(), self.pose))
        self.assertEqual(self.events[-1][1]['status'], 'lidar_error')
        self.assertEqual(self.client.stops, 1)

    def test_high_score_but_far_map_basin_is_not_fused(self):
        far = SimpleNamespace(
            pose=SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=-1.0), usable=True
        )
        localizer = self._install(visual_status="uncertain", track_rows=[far, far])
        self.assertIsNone(localizer.track(object(), self.pose))
        self.assertIsNone(localizer.track(object(), self.pose))
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.events[-1][1]["accepted"])

    def test_good_scan_odometry_bridges_short_map_dropout_without_stop(self):
        rows = iter([self.bad, self.good])
        localizer = SimpleNamespace(
            track=lambda *_: next(rows),
            localize_near_position=lambda *_args, **_kwargs: self.bad,
            map=SimpleNamespace(
                world_to_grid=lambda _points: [(1, 1)],
                in_bounds=lambda _column, _row: True,
                traversable=np.ones((2, 2), dtype=bool),
            ),
            _last_track_scan=object(),
        )
        motion = SimpleNamespace(
            converged=True, rmse_m=0.02, inlier_ratio=0.9,
            transform=SimpleNamespace(x_m=0.08, y_m=0.0, yaw_rad=0.01),
        )
        mission = SimpleNamespace(
            _tracking_usable_for_photo_mission=lambda result, _args: bool(
                result is not None and result.usable
            ),
            _visual_localization_check=lambda *_args: self.fail("unexpected visual call"),
            _scan_motion_prediction=lambda pose, _motion: (pose, True),
            LocalizationResult=lambda **values: SimpleNamespace(**values, usable=True),
        )
        _install_midrun_visual_recheck(
            mission, localizer, object(), self.client, self.controller,
            object(), object(), object(), Path("search_runs"),
            lambda name, **data: self.events.append((name, data)),
            initial_scan=object(),
        )
        mission._scan_motion_prediction(self.pose, motion)
        bridged = localizer.track(object(), self.pose)
        self.assertEqual(bridged.score, -1.0)
        self.assertEqual(self.client.stops, 0)
        self.assertIsNone(localizer._last_track_scan)
        mission._scan_motion_prediction(self.pose, motion)
        self.assertIs(localizer.track(object(), self.pose), self.good)
        self.assertEqual([data["status"] for name, data in self.events
                          if name == "midrun_odometry_bridge"],
                         ["started", "map_recovered"])

    def test_odometry_budget_cannot_rearm_without_map_recovery(self):
        localizer = SimpleNamespace(
            track=lambda *_: self.bad,
            localize_near_position=lambda *_args, **_kwargs: self.bad,
            global_localization_candidates=lambda *_args, **_kwargs: [self.good],
            map=SimpleNamespace(
                world_to_grid=lambda _points: [(1, 1)],
                in_bounds=lambda _column, _row: True,
                traversable=np.ones((2, 2), dtype=bool),
            ),
        )
        motion = SimpleNamespace(
            converged=True, rmse_m=0.02, inlier_ratio=0.9,
            transform=SimpleNamespace(x_m=0.16, y_m=0.0, yaw_rad=0.0),
        )
        mission = SimpleNamespace(
            _tracking_usable_for_photo_mission=lambda result, _args: bool(
                result is not None and result.usable
            ),
            _visual_localization_check=lambda *_args: (
                "uncertain", self.good, None, {}
            ),
            _scan_motion_prediction=lambda pose, _motion: (pose, True),
            LocalizationResult=lambda **values: SimpleNamespace(**values, usable=True),
        )
        _install_midrun_visual_recheck(
            mission, localizer, object(), self.client, self.controller,
            object(), object(), object(), Path("search_runs"),
            lambda name, **data: self.events.append((name, data)),
            initial_scan=object(),
        )
        outputs = []
        for _ in range(9):
            mission._scan_motion_prediction(self.pose, motion)
            outputs.append(localizer.track(object(), self.pose))
        self.assertEqual([item.score if item else None for item in outputs],
                         [-1.0] * 6 + [None] * 3)
        statuses = [data["status"] for name, data in self.events
                    if name == "midrun_odometry_bridge"]
        self.assertEqual(statuses, ["started", "budget_exhausted"])


if __name__ == "__main__":
    unittest.main()
