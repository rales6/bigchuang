"""Replay the 19:52 region run's 30-second estimate expiry without hardware."""
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from raspberry_pi.mapping import Pose2D
from raspberry_pi.known_map_navigation import LocalizationResult
from grasp_range_integration.phase1.relationship_find_only_entry import _install_midrun_visual_recheck
from grasp_range_integration.phase1.motion_interlock import estimate_within_budget, estimate_command_horizon
from grasp_range_integration.phase1 import startup_active_recovery as recovery


class RegionOdometryTests(unittest.TestCase):
    def install(self):
        self.now = 1000.
        self.pose = Pose2D()
        bad = LocalizationResult(self.pose, .20, .4, .1)
        self.localizer = NS(track=lambda *a: bad, localize_near_position=lambda *a, **k: bad,
            map=NS(world_to_grid=lambda xy: [(1, 1)], in_bounds=lambda *a: True,
                   traversable=np.ones((3, 3), dtype=bool)))
        self.client = NS(stop=Mock())
        self.mission = NS(_phase1_verified_glass_required=True, _phase1_region_continuous_odometry=True,
            _tracking_usable_for_photo_mission=lambda result, args: bool(result and result.reliable),
            _visual_localization_check=Mock(side_effect=AssertionError('unnecessary visual recheck')),
            _scan_motion_prediction=lambda pose, motion: (pose, True), LocalizationResult=LocalizationResult)
        self.events = []
        _install_midrun_visual_recheck(self.mission, self.localizer, NS(), self.client,
            NS(stopped=Mock()), object(), object(), object(), Path('.'),
            lambda name, **data: self.events.append((name, data)))

    def frame(self, sequence, *, rmse=.01, stale=False):
        self.now += 1.
        motion = NS(converged=True, rmse_m=rmse, inlier_ratio=.99, transform=Pose2D(.02, 0, 0))
        self.pose, _ = self.mission._scan_motion_prediction(self.pose, motion)
        scan = NS(sequence=sequence, received_s=self.now-(1. if stale else 0.))
        return self.localizer.track(scan, self.pose)

    def test_good_evidence_continues_past_30_seconds_and_1m_without_fake_map_anchor(self):
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', lambda: self.now):
            self.install()
            for i in range(65):
                result = self.frame(i)
                self.assertIsNotNone(result)
                self.assertEqual(result.score, -1.)
            estimate = self.mission._phase1_navigation_estimate
            self.assertGreater(self.now-estimate['started_s'], 30.)
            self.assertGreater(estimate['distance_m'], 1.)
            self.assertTrue(estimate_within_budget(estimate, self.now))
            self.assertGreaterEqual(estimate_command_horizon(estimate, self.now, .4), .25)
            self.assertGreater(estimate['uncertainty_m'], .05)
        self.client.stop.assert_not_called()
        self.mission._visual_localization_check.assert_not_called()

    def test_continuity_never_resets_accumulated_uncertainty_or_accepts_stale_evidence(self):
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', lambda: self.now):
            self.install()
            values = []
            for i in range(200):
                if self.frame(i) is None:
                    break
                values.append(self.mission._phase1_navigation_estimate['uncertainty_m'])
            self.assertGreater(len(values), 100)
            self.assertLess(len(values), 200)
            self.assertTrue(all(b >= a for a, b in zip(values, values[1:])))
            self.assertLessEqual(max(values), .35)
            self.assertIsNone(self.mission._phase1_navigation_estimate)

    def test_weak_feedback_cannot_inherit_continuous_policy(self):
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', lambda: self.now):
            self.install()
            for i in range(35): self.assertIsNotNone(self.frame(i))
            self.assertIsNone(self.frame(36, rmse=.10))

    def test_freshness_and_uncertainty_still_limit_actual_motor_lease(self):
        estimate = {'policy': 'continuous_lidar_odometry', 'started_s': 0.,
                    'last_evidence_s': 100., 'uncertainty_m': .29, 'distance_m': 2.}
        self.assertTrue(estimate_within_budget(estimate, 100.))
        self.assertFalse(estimate_within_budget(estimate, 100.51))
        self.assertEqual(estimate_command_horizon(estimate, 100.51, .4), 0.)
        estimate['uncertainty_m'] = .34
        self.assertEqual(estimate_command_horizon(estimate, 100., .4), 0.)
        estimate['uncertainty_m'] = .1; estimate.pop('policy')
        self.assertFalse(estimate_within_budget(estimate, 100.))

    def test_forward_recovery_does_not_compute_unused_rear_disk_or_duplicate_glass_sweep(self):
        scan = NS(received_s=0., angles_rad=np.linspace(-math.pi, math.pi, 360, endpoint=False),
                  distances_m=np.full(360, 2.))
        prior = NS(x_m=0., y_m=0., yaw_deg=0., radius_m=.2, yaw_tolerance_deg=30.)
        with patch.object(recovery, 'map_rotation_evidence', side_effect=AssertionError('unused rear disk')), \
             patch.object(recovery, 'swept_glass_clearance', side_effect=AssertionError('duplicate glass disk')), \
             patch.object(recovery, 'bounded_recovery_glass_sweep', return_value={'safe': True}) as glass:
            result = recovery.assess_recovery_action(scan, object(), prior, 400, 0, 200,
                directional_translation=True, clock=lambda: .1)
        self.assertTrue(result['safe'])
        glass.assert_called_once()
        self.assertFalse(result['rear_turn_authorized'])


if __name__ == '__main__':
    unittest.main()
