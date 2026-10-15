"""Bounded moving localization dropout tests. No hardware connections."""

import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.object_search_mission import _scan_motion_prediction
from grasp_range_integration.phase1.relationship_find_only_entry import (
    _install_midrun_visual_recheck,
)
from grasp_range_integration.phase1.motion_interlock import ESTIMATE_MAX_AGE_S


class BoundedNavigationTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.clock = patch('time.monotonic', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.pose = Pose2D(0., 0., 0.)
        self.events, self.stops, self.ds, self.near = [], [], [], []
        self.weak = NS(pose=self.pose, rmse_m=.07, inlier_ratio=.50, score=1.)
        self.strong = NS(pose=self.pose, rmse_m=.04, inlier_ratio=.85, score=1.)
        self.tracked = self.weak
        self.known = NS(world_to_grid=lambda pts: [(1, 1)],
                        in_bounds=lambda *a: True,
                        traversable=np.ones((3, 3), dtype=bool))
        def bounded(*a, **kw):
            self.near.append(True)
            return self.weak
        def vision(*a):
            self.ds.append(True)
            return 'uncertain', self.weak, None, {}
        self.mission = NS(_phase1_verified_glass_required=True,
            _tracking_usable_for_photo_mission=lambda r, a: bool(r and r.inlier_ratio >= .4),
            _visual_localization_check=vision,
            _scan_motion_prediction=_scan_motion_prediction,
            Pose2D=Pose2D, LocalizationResult=lambda **kw: NS(**kw))
        self.localizer = NS(track=lambda *a: self.tracked,
            localize_near_position=bounded, localize_bounded=bounded, map=self.known)
        _install_midrun_visual_recheck(self.mission, self.localizer, NS(),
            NS(stop=lambda: self.stops.append(True)), NS(stopped=lambda: None),
            None, None, None, Path('unused'),
            lambda name, **kw: self.events.append((name, kw)),
            initial_scan=NS(received_s=self.now))

    def step(self, distance=.08, yaw=0., *, age=0., quality=.9):
        match = MatchResult(Pose2D(distance, 0., yaw), .02, 100, True, quality)
        predicted, usable = self.mission._scan_motion_prediction(self.pose, match)
        return self.localizer.track(NS(received_s=self.now-age), predicted)

    def test_glass_mode_bridge_is_usable_without_stop_near_retry_or_ds(self):
        result = self.step()
        self.assertEqual(result.score, -1.)
        self.assertTrue(self.mission._tracking_usable_for_photo_mission(result, NS()))
        self.assertEqual((self.stops, self.near, self.ds), ([], [], []))
        estimate = self.mission._phase1_navigation_estimate
        self.assertAlmostEqual(estimate['uncertainty_m'], .062)

    def test_only_registered_estimate_can_pass_executor_quality(self):
        result = self.step()
        foreign = NS(**vars(result))
        self.assertFalse(self.mission._tracking_usable_for_photo_mission(foreign, NS()))

    def test_bridge_cannot_rearm_itself_as_map_recovery(self):
        self.tracked = self.step()
        self.step()
        self.assertAlmostEqual(self.mission._phase1_navigation_estimate['distance_m'], .16)

    def test_map_recovers_without_stopping_and_clears_estimated_command_limits(self):
        self.step()
        self.tracked = self.strong
        self.assertIs(self.step(), self.strong)
        self.assertIsNone(self.mission._phase1_navigation_estimate)
        self.assertEqual(self.stops, [])

    def test_distance_budget_exhaustion_cannot_rearm_without_map_recovery(self):
        self.assertIsNotNone(self.step(.26))
        self.assertIsNotNone(self.step(.26))
        self.assertIsNotNone(self.step(.26))
        self.assertIsNone(self.step(.26))
        self.assertIsNone(self.step(.01))
        self.assertIsNone(self.step(.01))
        self.assertEqual(len(self.ds), 1)
        self.assertIsNone(self.mission._phase1_navigation_estimate)

    def test_stationary_frames_do_not_reset_time_budget(self):
        self.step(0.)
        self.now += ESTIMATE_MAX_AGE_S + .1
        self.assertIsNone(self.step(0.))
        self.assertTrue(self.stops)

    def test_old_twelve_second_boundary_does_not_interrupt_credible_odometry(self):
        result = self.step(.08)
        self.now += 12.1
        self.assertTrue(self.mission._tracking_usable_for_photo_mission(result, NS()))
        self.assertIsNotNone(self.step(.08))
        self.assertEqual(self.stops, [])

    def test_output_budget_exhaustion_enters_recovery_not_zero_motion_rearming(self):
        self.step()
        self.mission._phase1_navigation_estimate['command_budget_exhausted'] = True
        self.assertIsNone(self.step(0.))
        self.assertIsNone(self.step(0.))
        self.assertEqual(len(self.ds),1)

    def test_expired_estimate_is_not_usable_even_before_next_track(self):
        result = self.step()
        self.now += ESTIMATE_MAX_AGE_S + .1
        self.assertFalse(self.mission._tracking_usable_for_photo_mission(result, NS()))

    def test_stale_scan_or_missing_free_map_support_cannot_start_bridge(self):
        self.assertIsNone(self.step(age=.6))
        self.known.traversable[:] = False
        self.assertIsNone(self.step())
        self.assertIsNone(getattr(self.mission, '_phase1_navigation_estimate', None))

    def test_unreliable_odometry_cannot_start_bridge(self):
        self.assertIsNone(self.step(quality=.45))
        self.assertTrue(self.stops)

    def test_cumulative_thirty_degrees_does_not_interrupt_credible_navigation(self):
        self.assertIsNotNone(self.step(0., math.radians(20.55)))
        self.assertIsNotNone(self.step(0., math.radians(12.)))
        estimate = self.mission._phase1_navigation_estimate
        self.assertGreater(estimate['rotation_rad'], math.radians(30.))
        self.assertGreater(estimate['uncertainty_m'], .05)
        self.assertEqual(self.stops, [])

    def test_turn_uncertainty_still_accumulates_without_fixed_angle_exit(self):
        for _ in range(10):
            self.assertIsNotNone(self.step(0., math.radians(20.)))
        self.assertIsNone(self.step(0., math.radians(20.)))
        self.assertTrue(self.stops)

    def test_ack_validated_pivot_translation_is_applied_in_full(self):
        raw = MatchResult(Pose2D(.1998, 0., math.radians(-20.55)), .02, 100, True, .9)
        self.mission._phase1_post_turn_match = raw
        predicted, usable = self.mission._scan_motion_prediction(self.pose, raw)
        self.assertTrue(usable)
        self.assertAlmostEqual(predicted.x_m, .1998)
        self.assertAlmostEqual(predicted.yaw_rad, raw.transform.yaw_rad)
        result = self.localizer.track(NS(received_s=self.now), predicted)
        self.assertEqual(result.score, -1.)
        self.assertLess(self.mission._phase1_navigation_estimate['uncertainty_m'], .12)
        self.assertEqual(self.stops, [])
        self.assertEqual(raw.transform.x_m, .1998)  # Raw evidence unchanged.
        self.assertTrue(any(name == 'turn_translation_applied' for name, _ in self.events))

    def test_unpaired_translation_is_not_silently_clamped(self):
        raw = MatchResult(Pose2D(.1998, 0., -.1), .02, 100, True, .9)
        predicted, _ = self.mission._scan_motion_prediction(self.pose, raw)
        self.assertAlmostEqual(predicted.x_m, .1998)
        self.assertEqual(self.localizer.track(NS(received_s=self.now), predicted).score, -1.)

    def test_23cm_slip_updates_pose_and_can_recover_map_without_stop(self):
        raw = MatchResult(Pose2D(.23, 0., 0.), .02, 100, True, .9)
        self.mission._phase1_post_turn_match = raw
        predicted, usable = self.mission._scan_motion_prediction(self.pose, raw)
        self.assertTrue(usable)
        self.assertAlmostEqual(predicted.x_m, .23)
        self.tracked = NS(pose=predicted, rmse_m=.02, inlier_ratio=.9, score=1.)
        self.assertIs(self.localizer.track(NS(received_s=self.now), predicted), self.tracked)
        self.assertEqual(self.stops, [])
        self.assertIsNone(getattr(self.mission, '_phase1_motion_fault', None))

    def test_repeated_slips_accumulate_in_global_pose_instead_of_reusing_old_pose(self):
        self.pose = Pose2D(1., 2., math.pi/2)
        raw = MatchResult(Pose2D(.3, -.1, -.1), .02, 100, True, .9)
        expected = self.pose
        for _ in range(3):
            self.mission._phase1_post_turn_match = raw
            self.pose, usable = self.mission._scan_motion_prediction(self.pose, raw)
            expected = expected.compose(raw.transform)
            self.assertTrue(usable)
            self.assertEqual(self.pose, expected)
        self.assertEqual(self.stops, [])


if __name__ == '__main__':
    unittest.main()
