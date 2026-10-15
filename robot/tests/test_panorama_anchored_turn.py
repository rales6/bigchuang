"""Turning regressions: angular shape matters even when ICP says 99%."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.open_space_panorama import _bounded_motion, _turn_increment, PanoramaConfig


class AnchoredTurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')

    def setUp(self):
        self.localizer = ScanMapLocalizer(self.known, lidar_offset_x_m=.14)
        self.origin = Pose2D(-.935, .625, math.radians(-128))
        self.before = self.scan(0., 1.)

    def scan(self, degrees, stamp):
        pose = self.origin.compose(Pose2D(yaw_rad=math.radians(degrees)))
        return LaserScan(*self.localizer.predicted_range_profile(pose, 360), stamp)

    def enable_anchor(self):
        self.localizer._panorama_rotation_only = True
        self.localizer._panorama_turn_anchor = (self.before, self.origin)

    def test_changed_shape_rejects_zero_yaw_despite_excellent_icp_score(self):
        self.enable_anchor()
        fake = MatchResult(Pose2D(), .005, 300, True, .99)
        # Both competing endpoint-only fits falsely report no movement.
        with patch.object(self.localizer, 'relative_turn_motion', return_value=fake), patch.object(
                self.localizer.motion_matcher, 'match_rotation', return_value=fake):
            self.localizer._panorama_turn_model_diagnostic = {'model': 'chassis_pivot'}
            pose, _, accepted = _bounded_motion(self.localizer, self.before, self.scan(40., 2.), self.origin)
        self.assertFalse(accepted)
        self.assertEqual(pose, self.origin)
        self.assertEqual(self.localizer._panorama_motion_diagnostic['rejection_reason'], 'turn_range_disagrees')
        self.assertEqual(self.localizer._panorama_pending_motion, (self.before, self.origin))

    def test_whole_turn_is_measured_from_same_anchor_without_double_counting(self):
        self.enable_anchor()
        pose, first, accepted = _bounded_motion(self.localizer, self.before, self.scan(20., 2.), self.origin)
        self.assertTrue(accepted)
        pose, second, accepted = _bounded_motion(self.localizer, self.scan(20., 2.), self.scan(40., 3.), pose)
        self.assertTrue(accepted)
        self.assertAlmostEqual(math.degrees(first.transform.yaw_rad + second.transform.yaw_rad), 40., delta=.6)
        self.assertAlmostEqual(math.degrees(pose.yaw_rad - self.origin.yaw_rad), 40., delta=.6)
        self.assertEqual(self.localizer._panorama_motion_diagnostic['turn_anchor_timestamp_s'], 1.)

    def test_true_stationary_scan_is_allowed_not_treated_as_required_rotation(self):
        self.enable_anchor()
        pose, _, accepted = _bounded_motion(self.localizer, self.before, self.scan(0., 2.), self.origin)
        self.assertTrue(accepted)
        self.assertAlmostEqual(pose.yaw_rad, self.origin.yaw_rad)

    def test_duplicate_scan_cannot_prove_no_rotation(self):
        self.enable_anchor()
        pose, _, accepted = _bounded_motion(self.localizer, self.before, self.scan(20., 2.), self.origin)
        self.assertTrue(accepted)
        returned, _, accepted = _bounded_motion(self.localizer, self.before, self.scan(20., 2.), pose)
        self.assertFalse(accepted)
        self.assertEqual(returned, pose)
        self.assertEqual(self.localizer._panorama_motion_diagnostic['rejection_reason'], 'turn_scan_not_new')

    def test_noise_and_partial_glass_outliers_do_not_force_a_stop(self):
        self.enable_anchor()
        current = self.scan(40., 2.)
        rng = np.random.default_rng(5)
        distances = current.distances_m + rng.normal(0., .008, len(current.distances_m))
        distances[(current.angles_rad > 1.) & (current.angles_rad < 1.18)] += .8
        distances[::17] = np.nan
        pose, _, accepted = _bounded_motion(self.localizer, self.before,
            LaserScan(current.angles_rad, distances, 2.), self.origin)
        self.assertTrue(accepted)
        self.assertAlmostEqual(math.degrees(pose.yaw_rad - self.origin.yaw_rad), 40., delta=1.)

    def test_one_direction_cannot_hide_bad_ranges_in_other_sectors(self):
        distances = self.before.distances_m.copy()
        distances[np.abs(self.before.angles_rad) > .5] *= .55
        changed = LaserScan(self.before.angles_rad, distances, 2.)
        self.assertFalse(self.localizer.turn_range_evidence(self.before, changed, Pose2D())['accepted'])

    def test_missing_rays_do_not_confirm_heading(self):
        distances = self.before.distances_m.copy()
        distances[np.abs(self.before.angles_rad) > .2] = np.nan
        changed = LaserScan(self.before.angles_rad, distances, 2.)
        self.assertFalse(self.localizer.turn_range_evidence(self.before, changed, Pose2D())['accepted'])

    def test_large_anchor_yaw_is_not_mistaken_for_free_translation(self):
        fit = self.localizer.relative_turn_motion_from_anchor(self.before, self.scan(100., 2.))
        self.assertTrue(self.localizer._panorama_turn_model_diagnostic['range_evidence']['accepted'])
        self.assertAlmostEqual(math.degrees(fit.transform.yaw_rad), 100., delta=.6)
        self.assertEqual((fit.transform.x_m, fit.transform.y_m), (0., 0.))

    def test_rejected_anchor_keeps_last_displayed_pose_and_original_reference(self):
        self.enable_anchor()
        pose, _, accepted = _bounded_motion(self.localizer, self.before, self.scan(20., 2.), self.origin)
        self.assertTrue(accepted)
        invalid = LaserScan(np.array([]), np.array([]), 3.)
        returned, _, accepted = _bounded_motion(self.localizer, self.scan(20., 2.), invalid, pose)
        self.assertFalse(accepted)
        self.assertEqual(returned, pose)
        self.assertEqual(self.localizer._panorama_pending_motion, (self.before, self.origin))
        # Once the wrapper has returned, pending motion still uses ray checks.
        self.localizer._panorama_turn_anchor = None
        self.localizer._panorama_rotation_only = False
        returned, increment, accepted = _bounded_motion(self.localizer, invalid, self.scan(40., 4.), pose)
        self.assertTrue(accepted)
        self.assertAlmostEqual(math.degrees(increment.transform.yaw_rad), 20., delta=.6)
        self.assertAlmostEqual(math.degrees(returned.yaw_rad - self.origin.yaw_rad), 40., delta=.6)

    def test_turn_context_restored_on_exception(self):
        localizer = SimpleNamespace(_panorama_rotation_only=False)
        with patch('raspberry_pi.open_space_panorama._turn_increment_impl', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                _turn_increment(None, None, localizer, self.before, self.origin, .5, PanoramaConfig(), None)
        self.assertFalse(localizer._panorama_rotation_only)
        self.assertIsNone(localizer._panorama_turn_anchor)

    def test_reciprocal_18cm_turn_alias_cannot_move_the_anchor(self):
        self.enable_anchor()
        fake = MatchResult(Pose2D(.186, 0., -.20), .058, 174, True, .956)
        self.localizer._panorama_turn_model_diagnostic = {'translation_verified': True,
                                                         'range_evidence': {'accepted': True}}
        with patch.object(self.localizer, 'relative_turn_motion_from_anchor', return_value=fake):
            pose, _, accepted = _bounded_motion(self.localizer, self.before, self.scan(20., 2.), self.origin)
        self.assertFalse(accepted)
        self.assertEqual(pose, self.origin)
        self.assertEqual(self.localizer._panorama_motion_diagnostic['rejection_reason'],
                         'turn_anchor_translation_exceeds')

    def test_nonfinite_transform_is_rejected_before_pose_composition(self):
        for yaw in [math.inf, math.nan]:
            localizer = SimpleNamespace(relative_motion=lambda *_: MatchResult(
                Pose2D(yaw_rad=yaw), .01, 100, True, .95))
            pose, _, accepted = _bounded_motion(localizer, self.before, self.scan(0., 2.), self.origin)
            self.assertFalse(accepted)
            self.assertEqual(pose, self.origin)

    def synthetic_controller(self, increments):
        values = iter(increments)
        def relative(*_):
            return MatchResult(Pose2D(yaw_rad=math.radians(next(values))), .01, 100, True, .95)
        localizer = SimpleNamespace(relative_motion=relative)
        baseline = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.), 0.)
        client = SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: None)
        return _turn_increment(client, iter([baseline] * 20), localizer, baseline, Pose2D(),
                               math.radians(60), PanoramaConfig(), lambda *_: None,
                               motion_is_safe=lambda *_: True)

    def test_negative_heading_adjustment_reduces_accumulated_angle(self):
        _, pose, measured, success = self.synthetic_controller([20., -10., 35., 10., 0.])
        self.assertTrue(success)
        self.assertAlmostEqual(math.degrees(measured), 55.)
        self.assertAlmostEqual(math.degrees(pose.yaw_rad), 55.)

    def test_large_braking_tail_is_not_reported_as_successful_60_degree_turn(self):
        _, _, measured, success = self.synthetic_controller([60., 40.])
        self.assertFalse(success)
        self.assertAlmostEqual(math.degrees(measured), 100.)


if __name__ == '__main__':
    unittest.main()
