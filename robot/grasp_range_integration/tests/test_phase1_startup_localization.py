"""Real stationary scan replay, CPU only, with explicit offline timestamps."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
import numpy as np

from grasp_range_integration.phase1.startup_localization import StartupMatcher
from grasp_range_integration.phase1.rotation_guard import assess_rotation, installed_body_mask
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'maps/semantic_release_20260919' if (ROOT / 'maps/semantic_release_20260919/ble_scene_17.pgm').exists() else ROOT
FIXTURE = Path(__file__).parent / 'fixtures/startup_stationary_20260926.json'


class StartupReplayTests(unittest.TestCase):
    @classmethod
    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def setUpClass(cls):
        data = json.loads(FIXTURE.read_text(encoding='utf-8'))
        assert data['hardware_motion_commands'] == 0 and data['offline_replay_only']
        cls.scans = [LaserScan(np.array(frame['raw_angles_rad'], dtype=float),
                              np.array(frame['raw_distances_m'], dtype=float), 0.) for frame in data['frames']]
        cls.known = KnownOccupancyMap.load(RELEASE / 'ble_scene_17.pgm')
        cls.prior = SimpleNamespace(x_m=-1.55, y_m=-2.60, yaw_deg=100., radius_m=.70, yaw_tolerance_deg=40.)

    def test_one_bounded_search_then_three_local_verifications_keep_original_result(self):
        localizer = ScanMapLocalizer(self.known, lidar_offset_x_m=.14, lidar_offset_y_m=0.)
        matcher = StartupMatcher(localizer, self.prior)
        diagnostics, results = [], []
        for scan in self.scans:
            result = matcher.match(scan, time.monotonic() + 60)
            self.assertIsNotNone(result, matcher.diagnostic)
            self.assertTrue(result.reliable)
            diagnostics.append(dict(matcher.diagnostic))
            results.append(result)
        self.assertEqual(matcher.searches, 1)
        self.assertEqual([d['mode'] for d in diagnostics], ['bounded_start_search'] + ['stationary_local_verify'] * 3)
        # Original constrained algorithm on this exact captured first frame
        # returned (-1.59,-2.36,128 deg). It is a baseline, NOT ground truth.
        self.assertAlmostEqual(results[0].pose.x_m, -1.59, places=2)
        self.assertAlmostEqual(results[0].pose.y_m, -2.36, places=2)
        self.assertAlmostEqual(math.degrees(results[0].pose.yaw_rad), 128., places=1)
        for result in results[1:]:
            self.assertLess(math.hypot(result.pose.x_m-results[0].pose.x_m, result.pose.y_m-results[0].pose.y_m), .12)

    def test_real_rear_body_arc_requires_map_and_reliable_pose(self):
        scan = self.scans[0]
        pose = SimpleNamespace(x_m=-1.3663, y_m=-2.5653, yaw_rad=2.2683)
        value = assess_rotation(scan, known_map=self.known, pose=pose, map_pose_reliable=True)
        self.assertGreater(value['max_unobserved_gap_deg'], 150)
        self.assertTrue(value['installed_body_blind_arc'])
        self.assertTrue(value['safe'], value)
        self.assertEqual(value['reason'], 'installed_body_occlusion_with_local_map_support')
        self.assertFalse(assess_rotation(scan, known_map=self.known, pose=pose)['safe'])
        self.assertFalse(assess_rotation(scan, pose=pose, map_pose_reliable=True)['safe'])

    def test_cropping_or_removing_body_evidence_cannot_fake_known_occlusion(self):
        scan = self.scans[0]
        front = np.abs((scan.angles_rad + math.pi) % (2*math.pi) - math.pi) < math.radians(42)
        cropped = LaserScan(scan.angles_rad[front], scan.distances_m[front], 0.)
        pose = SimpleNamespace(x_m=-1.3663, y_m=-2.5653, yaw_rad=2.2683)
        self.assertFalse(assess_rotation(cropped, known_map=self.known, pose=pose, map_pose_reliable=True)['safe'])
        distances = scan.distances_m.copy()
        distances[installed_body_mask(scan.angles_rad, distances)] = math.nan
        removed = LaserScan(scan.angles_rad, distances, 0.)
        self.assertFalse(assess_rotation(removed, known_map=self.known, pose=pose, map_pose_reliable=True)['safe'])

    def test_photo_reviewed_real_boundary_point_is_audited_and_map_gated(self):
        pose = SimpleNamespace(x_m=-1.3663, y_m=-2.5653, yaw_rad=2.2683)
        result = assess_rotation(self.scans[2], known_map=self.known, pose=pose, map_pose_reliable=True)
        self.assertTrue(result['safe'])
        self.assertEqual(result['installation_patch_count'],1)
        self.assertAlmostEqual(result['installation_patch_points'][0]['distance_m'],.142)
        self.assertAlmostEqual(result['installation_patch_points'][0]['bearing_deg'],117.)
        self.assertEqual(result['body_mask_profile'],'front_n10_20260929_wider_opposite_edge_v4')
        self.assertFalse(assess_rotation(self.scans[2],known_map=self.known,pose=pose)['safe'])
        self.assertFalse(assess_rotation(self.scans[2],pose=pose,map_pose_reliable=True)['safe'])


class StartupBudgetTests(unittest.TestCase):
    def setUp(self):
        self.prior = SimpleNamespace(x_m=0., y_m=0., yaw_deg=0., radius_m=.7, yaw_tolerance_deg=40.)
        self.result = SimpleNamespace(pose=SimpleNamespace(x_m=0., y_m=0., yaw_rad=0.), reliable=True)

    def test_global_api_called_once_and_followup_scans_use_track(self):
        calls = []
        localizer = SimpleNamespace(global_localize=lambda *a, **k: calls.append('search') or self.result,
                                    track=lambda *a, **k: calls.append('verify') or self.result)
        matcher = StartupMatcher(localizer, self.prior)
        for _ in range(4):
            self.assertIsNotNone(matcher.match(object(), time.monotonic()+10))
        self.assertEqual(calls, ['search','verify','verify','verify'])

    def test_partial_results_after_deadline_are_discarded(self):
        localizer = SimpleNamespace(global_localize=lambda *a, **k: self.result)
        matcher = StartupMatcher(localizer, self.prior)
        with patch('grasp_range_integration.phase1.startup_localization.time.monotonic', side_effect=[0.,21.,21.]):
            self.assertIsNone(matcher.match(object(), 60.))
        self.assertEqual(matcher.diagnostic['result'], 'budget_exhausted')
        self.assertIsNone(matcher.hint)

    def test_failed_initial_search_is_not_repeated_forever(self):
        calls = []
        matcher = StartupMatcher(SimpleNamespace(global_localize=lambda *a, **k: calls.append(True)), self.prior)
        for _ in range(5): matcher.match(object(), time.monotonic()+10)
        self.assertEqual(len(calls), 2)
        self.assertEqual(matcher.diagnostic['result'], 'initial_search_limit')

    def test_local_verify_cannot_jump_pose_or_leave_confirmed_start(self):
        localizer = SimpleNamespace(global_localize=lambda *a, **k: self.result)
        matcher = StartupMatcher(localizer, self.prior)
        matcher.match(object(), time.monotonic()+10)
        localizer.track = lambda *a: SimpleNamespace(pose=SimpleNamespace(x_m=.5,y_m=0.,yaw_rad=0.),reliable=True)
        self.assertIsNone(matcher.match(object(), time.monotonic()+10))
        self.assertEqual(matcher.diagnostic['result'], 'stationary_pose_changed')
        self.assertEqual(matcher.hint.x_m, 0.)


if __name__ == '__main__': unittest.main()
