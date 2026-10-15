"""Local anti-revisit selection and chassis-centered pivot model regressions."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import IcpScanMatcher, MatchResult
from raspberry_pi.open_space_panorama import (
    PanoramaConfig, OpenSpaceDirectionSelector, _turn_increment, _observe_photo_coverage,
)


class RepeatMotionTests(unittest.TestCase):
    def test_uncertain_photos_only_update_exploration_coverage(self):
        verified, provisional = Mock(), Mock()
        provisional.observe.return_value = 23
        pose, config = Pose2D(), PanoramaConfig()
        self.assertEqual(_observe_photo_coverage(
            verified, provisional, pose, .7, config, True), (0, 23))
        verified.observe.assert_not_called()
        provisional.observe.assert_called_once_with(pose, .7, config)

    def test_trusted_photos_update_both_independent_coverage_layers(self):
        verified, provisional = Mock(), Mock()
        verified.observe.return_value = 19
        provisional.observe.return_value = 7
        pose, config = Pose2D(), PanoramaConfig()
        self.assertEqual(_observe_photo_coverage(
            verified, provisional, pose, .7, config, False), (19, 7))
        verified.observe.assert_called_once_with(pose, .7, config)
        provisional.observe.assert_called_once_with(pose, .7, config)

    def choose(self, selector):
        pose = Pose2D()
        assessment = {'can_advance': True, 'lidar_m': 2.2, 'map_m': 2.2, 'support': 36}
        coverage = SimpleNamespace(points=np.empty((0, 2)), covered=np.array([], dtype=bool),
                                   expected_panorama_gain=lambda *_: 0)
        with patch('raspberry_pi.open_space_panorama._nearest_map_guard_pose', return_value=pose), patch(
            'raspberry_pi.open_space_panorama._advance_assessment', return_value=assessment):
            return selector.choose(SimpleNamespace(), pose,
                                   SimpleNamespace(line_is_clear=lambda *_: True), coverage, PanoramaConfig())

    def test_new_front_corridor_still_has_straight_priority(self):
        choice = self.choose(OpenSpaceDirectionSelector())
        self.assertEqual(choice['selection_policy'], 'safe_straight')
        self.assertEqual(choice['relative_heading_deg'], 0.)

    def test_previously_traversed_front_allows_a_novel_safe_side_exit(self):
        selector = OpenSpaceDirectionSelector()
        for x in np.arange(.55, 1.76, .15):
            selector.record_translation(Pose2D(float(x), 0, 0))
        choice = self.choose(selector)
        self.assertEqual(choice['selection_policy'], 'leave_visited_area')
        self.assertGreater(abs(choice['relative_heading_deg']), 0.)
        self.assertLessEqual(abs(choice['relative_heading_deg']), 70.)
        self.assertEqual(choice['revisit_cost'], 0)

    def test_small_noise_does_not_create_traversal_credit(self):
        selector = OpenSpaceDirectionSelector()
        for _ in range(30):
            selector.record_translation(Pose2D(.001, -.001, 0))
        self.assertEqual(len(selector.travel_history), 1)

    def test_recent_segment_is_not_treated_as_a_completed_loop(self):
        selector = OpenSpaceDirectionSelector()
        for x in (.6, .75, .9, 1.05):
            selector.record_translation(Pose2D(x, 0, 0))
        self.assertEqual(self.choose(selector)['selection_policy'], 'safe_straight')

    def test_rotation_fit_uses_zero_chassis_translation(self):
        rng = np.random.default_rng(30)
        reference = rng.uniform(-2., 2., (240, 2))
        truth = Pose2D(0, 0, math.radians(23))
        current = truth.inverse().transform_points(reference) + rng.normal(0, .002, reference.shape)
        match = IcpScanMatcher().match_rotation(reference, current)
        self.assertTrue(match.converged)
        self.assertEqual((match.transform.x_m, match.transform.y_m), (0., 0.))
        self.assertLess(abs(match.transform.yaw_rad - truth.yaw_rad), math.radians(.5))

    def test_real_map_turn_model_accounts_for_offset_lidar(self):
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        localizer = ScanMapLocalizer(known, lidar_offset_x_m=.14)
        pose = Pose2D(-.935, .625, math.radians(-128))
        rotated = pose.compose(Pose2D(0, 0, math.radians(20)))
        a, r = localizer.predicted_range_profile(pose, 360)
        b, s = localizer.predicted_range_profile(rotated, 360)
        match = localizer.relative_turn_motion(LaserScan(a, r, 0), LaserScan(b, s, 0))
        self.assertEqual(localizer._panorama_turn_model_diagnostic['model'], 'chassis_pivot')
        self.assertEqual((match.transform.x_m, match.transform.y_m), (0., 0.))
        self.assertLess(abs(math.degrees(match.transform.yaw_rad) - 20.), 2.)

    def test_real_translation_is_not_forced_into_a_bad_pivot_model(self):
        localizer = ScanMapLocalizer(SimpleNamespace())
        free = MatchResult(Pose2D(.2, .1, .2), .01, 100, True, .99)
        pivot = MatchResult(Pose2D(0, 0, .2), .1, 80, True, .8)
        with patch.object(localizer, 'points', return_value=np.ones((100, 2))), patch.object(
            localizer.motion_matcher, 'match', return_value=free), patch.object(
            localizer.motion_matcher, 'match_rotation', return_value=pivot):
            self.assertIs(localizer.relative_turn_motion(None, None), free)

    def test_large_translation_is_preserved_even_with_an_ambiguous_good_pivot_cost(self):
        localizer = ScanMapLocalizer(SimpleNamespace())
        free = MatchResult(Pose2D(.2, .1, .2), .01, 100, True, .99)
        pivot = MatchResult(Pose2D(0, 0, .2), .02, 100, True, .99)
        with patch.object(localizer, 'points', return_value=np.ones((100, 2))), patch.object(
            localizer.motion_matcher, 'match', return_value=free), patch.object(
            localizer.motion_matcher, 'match_rotation', return_value=pivot):
            self.assertIs(localizer.relative_turn_motion(None, None), free)

    def test_turn_context_is_restored_even_when_the_turn_fails(self):
        localizer = SimpleNamespace(_panorama_rotation_only=False)
        def fail(*_):
            self.assertTrue(localizer._panorama_rotation_only)
            raise RuntimeError('test turn failure')
        with patch('raspberry_pi.open_space_panorama._turn_increment_impl', side_effect=fail):
            with self.assertRaises(RuntimeError):
                _turn_increment(None, None, localizer, None, Pose2D(), .3, PanoramaConfig(), None)
        self.assertFalse(localizer._panorama_rotation_only)


if __name__ == '__main__':
    unittest.main()
