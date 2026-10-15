import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult, IcpScanMatcher
from raspberry_pi.open_space_panorama import (
    PanoramaConfig, _recover_pending_turn_heading, _turn_pulse_ttl_ms,
)
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer


class HeadingRecoveryTests(unittest.TestCase):
    def test_production_unresolved_turn_cannot_reanchor_at_old_yaw_and_drive(self):
        from tests import test_panorama_obstruction_recheck as fixture
        summary, _, commands, navigation = fixture.RuntimeCourseEvidenceTests().runtime(pending_turn=True)
        self.assertEqual(summary['finish_reason'], 'turn_heading_unresolved')
        self.assertFalse(commands)
        self.assertFalse(any(row['state'] == 'ODOMETRY_REANCHORED_UNCERTAIN' for row in navigation))

    def recovery(self, yaws, *, quality=.95, duplicates=False):
        anchor = Pose2D(-.8, .4, -.5)
        def scan(stamp):
            return LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.), stamp)
        original = scan(9.)
        localizer = SimpleNamespace(_panorama_pending_motion=(original, anchor),
            _panorama_pending_rotation_only=True)
        calls, outputs, clock = [], [], [10.]
        yaw_iterator = iter(yaws)
        def recover(reference, _current, pose):
            calls.append((reference, pose))
            return MatchResult(Pose2D(yaw_rad=math.radians(next(yaw_iterator))), .02, 100, True, quality)
        localizer.recover_turn_heading = recover
        def scans():
            for i in range(6):
                clock[0] += .1
                yield scan(10. if duplicates else clock[0])
        with patch('raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: clock[0]):
            result = _recover_pending_turn_heading(SimpleNamespace(stop=lambda: outputs.append('stop')),
                iter(scans()), localizer, scan(10.), anchor, PanoramaConfig())
        self.assertEqual(outputs, ['stop'])
        self.assertTrue(all(ref is original and pose == anchor for ref, pose in calls))
        return result, localizer, anchor

    def test_lost_yaw_is_restored_from_original_anchor_without_xy_jump(self):
        result, localizer, anchor = self.recovery([40., 41., 41.5])
        self.assertTrue(result[2])
        self.assertEqual((result[1].x_m, result[1].y_m), (anchor.x_m, anchor.y_m))
        self.assertAlmostEqual(result[1].yaw_rad, anchor.yaw_rad + math.radians(41.5))
        self.assertIsNone(localizer._panorama_pending_motion)

    def test_inconsistent_yaw_never_discards_original_reference(self):
        result, localizer, anchor = self.recovery([0., 30., 0., 30., 0., 30.])
        self.assertFalse(result[2])
        self.assertEqual(result[1], anchor)
        self.assertIsNotNone(localizer._panorama_pending_motion)

    def test_low_quality_recovery_is_not_applied(self):
        result, localizer, anchor = self.recovery([40.] * 6, quality=.33)
        self.assertFalse(result[2])
        self.assertEqual(result[1], anchor)
        self.assertTrue(localizer._panorama_pending_rotation_only)

    def test_duplicate_scan_is_not_a_confirmation(self):
        result, _, _ = self.recovery([], duplicates=True)
        self.assertFalse(result[2])

    def test_small_requested_angle_no_long_startup_output(self):
        self.assertEqual(_turn_pulse_ttl_ms(math.radians(20), 3000, 0., True), 120)
        self.assertLessEqual(_turn_pulse_ttl_ms(math.radians(60), 3000, 0., True), 230)
        self.assertLessEqual(_turn_pulse_ttl_ms(math.radians(60), 3000, 5., True), 120)

    def test_loaded_no_response_duration_boost_is_bounded_and_stops_once_moving(self):
        short = _turn_pulse_ttl_ms(math.radians(20), 3000, 0., True)
        loaded = _turn_pulse_ttl_ms(math.radians(20), 3000, 0., True, stalled=4)
        self.assertGreater(loaded, short)
        self.assertLessEqual(loaded, 220)
        self.assertEqual(_turn_pulse_ttl_ms(math.radians(20), 3000, 1., False, stalled=4), short)

    def test_explicit_large_yaw_rotation_recovery(self):
        rng = np.random.default_rng(44)
        reference = rng.normal(size=(100, 2)) * np.array([1.3, .7])
        current = Pose2D(yaw_rad=math.radians(-100)).transform_points(reference)
        fit = IcpScanMatcher().match_rotation(reference, current, yaw_range_deg=180)
        self.assertLess(abs(math.degrees(fit.transform.yaw_rad) - 100), .2)
        self.assertLess(fit.rmse_m, .01)

    def test_real_old_map_rays_recover_heading_without_xy_search(self):
        localizer = ScanMapLocalizer(KnownOccupancyMap.load(
            'maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml'), lidar_offset_x_m=.14)
        anchor = Pose2D(-.935, .625, math.radians(-128))
        angles, ranges = localizer.predicted_range_profile(anchor, 180)
        current_angles, current_ranges = localizer.predicted_range_profile(
            anchor.compose(Pose2D(yaw_rad=math.radians(40))), 180)
        fit = localizer.recover_turn_heading(LaserScan(angles, ranges, 1.),
            LaserScan(current_angles, current_ranges, 2.), anchor)
        self.assertIsNotNone(fit)
        self.assertLess(abs(math.degrees(fit.transform.yaw_rad) - 40), .5)
        self.assertEqual((fit.transform.x_m, fit.transform.y_m), (0., 0.))

    def test_map_disagreement_or_rotation_alias_prevents_recovery(self):
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        localizer = ScanMapLocalizer(known)
        scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.), 10.)
        forward = MatchResult(Pose2D(yaw_rad=.4), .02, 100, True, .95)
        reverse = MatchResult(Pose2D(yaw_rad=-.4), .02, 100, True, .95)
        with patch.object(localizer.motion_matcher, 'match_rotation', side_effect=[forward, reverse]), patch.object(
            localizer.motion_matcher, 'evaluate', return_value=(.03, 100, .95)):
            self.assertIsNone(localizer.recover_turn_heading(scan, scan, Pose2D()))
            self.assertEqual(localizer._panorama_turn_recovery_diagnostic['result'], 'separated_yaw_ambiguous')
        with patch.object(localizer.motion_matcher, 'match_rotation', side_effect=[forward, reverse]), patch.object(
            localizer.motion_matcher, 'evaluate', return_value=(.30, 100, .95)), patch.object(
            localizer, '_evaluate', return_value=(.5, .03, .85)), patch.object(
            localizer, '_range_consistency', return_value=(.1, 1., .9)):
            self.assertIsNone(localizer.recover_turn_heading(scan, scan, Pose2D()))
            self.assertEqual(localizer._panorama_turn_recovery_diagnostic['result'], 'original_anchor_map_disagrees')


if __name__ == '__main__':
    unittest.main()
