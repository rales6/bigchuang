"""Regression of moving confirmation windows and bounded post-turn checks."""
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer, LocalizationResult
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.geometry import normalize_angle
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.open_space_panorama import (
    PanoramaConfig, _CorrectionObservation, _correction_consistent,
    _correction_residual, _correction_rejection_reason, _transport_candidate,
    _stopped_heading_calibration,
)


def result(pose, rmse=.04, inliers=.9, margin=.1):
    return LocalizationResult(pose, rmse, inliers, .85, margin)


class PanoramaMapCorrectionTests(unittest.TestCase):
    def test_compare_candidates_after_real_translation_and_turn(self):
        old = Pose2D(.2, -.3, math.radians(25))
        new = old.compose(Pose2D(.8, .1, math.radians(65)))
        map_from_odometry = Pose2D(.1, -.2, math.radians(18))
        first = _CorrectionObservation(result(map_from_odometry.compose(old)), old)
        second = _CorrectionObservation(result(map_from_odometry.compose(new)), new)
        self.assertFalse(_correction_consistent(first.result, second.result))
        self.assertTrue(_correction_consistent(first, second))
        self.assertLess(_correction_residual(first, second)[0], 1e-10)

    def test_actual_candidate_jump_is_not_hidden_by_motion_compensation(self):
        first = _CorrectionObservation(result(Pose2D()), Pose2D())
        odometry = Pose2D(.7, .2, math.radians(40))
        for jump in (Pose2D(.21, 0, 0), Pose2D(0, 0, math.radians(20))):
            second = _CorrectionObservation(result(odometry.compose(jump)), odometry)
            self.assertFalse(_correction_consistent(first, second))

    def test_stationary_confirmation_and_yaw_wrap(self):
        a = _CorrectionObservation(result(Pose2D(0, 0, math.radians(179))), Pose2D())
        b = _CorrectionObservation(result(Pose2D(.01, 0, math.radians(-179))), Pose2D())
        self.assertTrue(_correction_consistent(a, b))

    def test_transport_candidate_includes_stopped_tail_once(self):
        origin = Pose2D(.1, .2, math.radians(70))
        movement = Pose2D(.02, -.01, math.radians(6))
        candidate = result(Pose2D(.15, .3, math.radians(80)))
        actual = _transport_candidate(candidate, origin, origin.compose(movement))
        expected = candidate.pose.compose(movement)
        self.assertAlmostEqual(actual.pose.x_m, expected.x_m)
        self.assertAlmostEqual(actual.pose.yaw_rad, expected.yaw_rad)
        self.assertEqual(actual.ambiguity_margin, candidate.ambiguity_margin)

    def test_explicit_rejection_reasons(self):
        pose = Pose2D()
        cases = [
            (None, 'no_candidate'),
            (result(pose, rmse=.3), 'candidate_quality_low'),
            (result(Pose2D(math.nan, 0, 0)), 'candidate_quality_low'),
            (result(Pose2D(.9, 0, 0)), 'candidate_translation_outside_window'),
            (result(Pose2D(0, 0, math.pi)), 'candidate_yaw_outside_window'),
        ]
        for candidate, reason in cases:
            self.assertEqual(_correction_rejection_reason(candidate, pose, .65, 35), reason)

    def calibrate(self, candidates, duplicate=False, tail=Pose2D(), slow=False):
        sent, stopped, options = [], [], []
        clock = [10.]
        client = SimpleNamespace(stop=lambda: stopped.append(True),
                                 set_twist=lambda *a, **k: sent.append(a))
        def bounded(scan, pose, **kwargs):
            options.append(kwargs)
            if slow:
                clock[0] += 2.1
            return next(candidates)
        calls = [0]
        def relative(*_):
            calls[0] += 1
            delta = tail if calls[0] == 3 else Pose2D()
            return MatchResult(delta, .01, 180, True, .95)
        localizer = SimpleNamespace(relative_motion=relative, localize_bounded=bounded)
        def scan(t):
            return LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.), t)
        def frames():
            for i in range(3):
                clock[0] = 10. + (i + 1) * .1
                yield scan(10.1 if duplicate and i == 1 else clock[0])
        with patch('raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: clock[0]):
            _, pose, accepted = _stopped_heading_calibration(
                client, frames(), localizer, scan(10.), Pose2D(), PanoramaConfig())
        self.assertFalse(sent)
        self.assertTrue(stopped)
        self.assertTrue(all(x['yaw_radius_deg'] == 45. and x['radius_m'] == .18 for x in options))
        return pose, accepted, localizer._panorama_heading_diagnostic

    def test_two_stopped_frames_blend_heading_without_map_translation(self):
        candidate = result(Pose2D(.1, .05, math.radians(30)))
        pose, accepted, diagnostic = self.calibrate(iter([candidate, candidate]))
        self.assertTrue(accepted)
        self.assertEqual((pose.x_m, pose.y_m), (0., 0.))
        self.assertAlmostEqual(math.degrees(pose.yaw_rad), 10.5)
        self.assertEqual(diagnostic['result'], 'yaw_blended')

    def test_heading_fusion_is_capped_and_not_observed_as_real_motion(self):
        candidate = result(Pose2D(0, 0, math.radians(40)))
        pose, accepted, _ = self.calibrate(iter([candidate, candidate]))
        self.assertTrue(accepted)
        self.assertAlmostEqual(math.degrees(pose.yaw_rad), 12.)

    def test_tail_motion_is_preserved_without_double_fusing_it(self):
        candidate = result(Pose2D(0, 0, math.radians(30)))
        pose, accepted, _ = self.calibrate(iter([candidate, candidate]), tail=Pose2D(.02, 0, math.radians(5)))
        self.assertTrue(accepted)
        self.assertAlmostEqual(pose.x_m, .02)
        self.assertAlmostEqual(math.degrees(pose.yaw_rad), 15.5)

    def test_duplicate_scan_cannot_confirm(self):
        candidate = result(Pose2D(0, 0, math.radians(30)))
        pose, accepted, diagnostic = self.calibrate(iter([candidate]), duplicate=True)
        self.assertFalse(accepted)
        self.assertEqual(pose, Pose2D())
        self.assertEqual(diagnostic['result'], 'duplicate_scan')

    def test_low_quality_ambiguity_and_opposite_heading_do_not_correct(self):
        for candidate in (result(Pose2D(), inliers=.4), result(Pose2D(), margin=0.),
                          result(Pose2D(0, 0, math.pi)), result(Pose2D(1., 0, 0))):
            pose, accepted, diagnostic = self.calibrate(iter([candidate]))
            self.assertFalse(accepted)
            self.assertEqual(pose, Pose2D())
            self.assertNotEqual(diagnostic['result'], 'yaw_blended')

    def test_inconsistent_stopped_frames_keep_odometry(self):
        _, accepted, diagnostic = self.calibrate(iter([
            result(Pose2D(0, 0, math.radians(30))), result(Pose2D(0, 0, math.radians(-30)))]))
        self.assertFalse(accepted)
        self.assertEqual(diagnostic['result'], 'motion_compensated_candidates_inconsistent')

    def test_heading_check_has_budget_and_never_commands_a_search_turn(self):
        _, accepted, diagnostic = self.calibrate(iter([result(Pose2D())]), slow=True)
        self.assertFalse(accepted)
        self.assertEqual(diagnostic['result'], 'heading_check_cancelled_or_budget')

    def test_real_map_expanded_heading_search_recovers_30_degree_bias(self):
        paths = [Path('maps/ble_scene_17.pgm'), Path('maps/ble_scene_17.yaml')]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        localizer = ScanMapLocalizer(KnownOccupancyMap.load(*paths))
        truth = Pose2D(-.935, .625, math.radians(-128))
        angles, ranges = localizer.predicted_range_profile(truth, 180)
        scan = LaserScan(angles, ranges, 0.)
        hint = Pose2D(truth.x_m, truth.y_m, truth.yaw_rad + math.radians(30))
        candidate = localizer.localize_bounded(scan, hint, radius_m=.18, yaw_radius_deg=45.)
        self.assertIsNotNone(candidate)
        self.assertTrue(math.isfinite(candidate.range_agreement))
        self.assertTrue(math.isfinite(candidate.range_mean_error_m))
        self.assertGreaterEqual(candidate.range_coverage, .60)
        self.assertLess(abs(normalize_angle(candidate.pose.yaw_rad - truth.yaw_rad)), math.radians(4))
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])


if __name__ == '__main__':
    unittest.main()
