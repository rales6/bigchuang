"""Replay actual 210504 stop; all IO is software-only, no API or motors."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.glass_boundary_guard import DEFAULT_CONFIG, attach_boundaries
from grasp_range_integration.phase1.navigation_rotation_support import navigation_rotation_support
from grasp_range_integration.phase1.rotation_guard import assess_rotation
from grasp_range_integration.phase1.relationship_find_only_entry import _install_waypoint_obstacle_guard
from grasp_range_integration.phase1 import turn_policy, visibility_route_planner as planner


class NavigationRotationSupportTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        f = json.loads((Path(__file__).parent / 'fixtures/navigation_body_blind_20261002_210504.json').read_text(encoding='utf-8'))
        pgm = root / 'maps/semantic_release_20260919/ble_scene_17.pgm'
        if not pgm.exists():
            pgm = root / 'ble_scene_17.pgm'
        self.map = KnownOccupancyMap.load(pgm)
        attach_boundaries(self.map, json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8')))
        p = f['assessment']['map_support']['pose']
        self.pose = NS(x_m=p[0], y_m=p[1], yaw_rad=p[2])
        self.scan = NS(angles_rad=np.asarray(f['angles_rad']), distances_m=np.asarray(f['distances_m']), received_s=10.)
        self.context = dict(map=self.map, pose=self.pose, reliable=True, updated=10.,
            quality=copy.deepcopy(f['assessment']['pose_quality']),
            bridge_uncertainty_m=f['assessment']['pose_quality']['fused_delta_m'])

    def assess(self):
        original = assess_rotation(self.scan, known_map=self.map, pose=self.pose,
            map_pose_reliable=self.context['reliable'],
            map_pose_uncertainty_m=self.context['bridge_uncertainty_m'])
        return navigation_rotation_support(original, self.scan, self.context, now=10.1)

    def test_recorded_body_blind_shell_does_not_request_static_relocalization(self):
        result = self.assess()
        self.assertTrue(result['safe'], result)
        self.assertEqual(result['reason'], 'installed_body_occlusion_with_navigation_support')
        self.assertFalse(result['map_precision_required'])
        self.assertFalse(result['map_support']['clear'])  # Original evidence retained.
        self.assertTrue(result['navigation_support']['physical']['map_support']['clear'])

    def test_registered_estimate_is_navigation_support_not_absolute_map_confirmation(self):
        self.context['quality'].update(map_match=False, estimated_navigation=True, reason='bounded_scan_odometry')
        result = self.assess()
        self.assertTrue(result['safe'], result)
        self.assertTrue(result['navigation_support']['estimated_navigation'])
        self.assertAlmostEqual(result['navigation_support']['position_uncertainty_m'], .08623039027185586)

    def test_real_near_return_remains_blocking(self):
        i = np.argmin(abs(self.scan.angles_rad))
        self.scan.distances_m[i] = .2
        result = self.assess()
        self.assertFalse(result['safe'])
        self.assertEqual(result['reason'], 'exterior_near_obstacle')

    def test_unknown_occupied_or_temporary_cell_in_physical_disk_remains_blocking(self):
        c, r = self.map.world_to_grid(((self.pose.x_m, self.pose.y_m),))[0]
        for name in ('occupied', 'unknown', 'temporary_occupied'):
            layer = getattr(self.map, name)
            with self.subTest(layer=name):
                layer[r, c] = True
                self.assertFalse(self.assess()['safe'])
                layer[r, c] = False

    def test_glass_uncertainty_still_blocks_even_if_physical_disk_is_clear(self):
        # Controlled continuous-boundary fixture: .70m clear distance vs
        # .64 physical requirement, but .726 bounded-estimate requirement.
        self.map._phase1_glass_halfplane = (np.array([self.pose.x_m-.7, 0.]), np.array([1., 0.]))
        result = self.assess()
        self.assertFalse(result['safe'])
        self.assertEqual(result['navigation_support']['glass']['reason'], 'glass_sweep_blocked')

    def test_missing_glass_boundary_does_not_authorize_blind_turn(self):
        del self.map._phase1_glass_segments
        self.assertFalse(self.assess()['safe'])

    def test_unexplained_blind_arc_does_not_authorize_turn(self):
        # Remove measured rear body signature; don't declare unseen space free.
        self.scan.distances_m[:] = np.nan
        self.scan.distances_m[:70] = 2.
        self.assertFalse(self.assess()['safe'])

    def test_stale_scan_stale_pose_foreign_quality_and_excess_drift_are_rejected(self):
        original = assess_rotation(self.scan, known_map=self.map, pose=self.pose,
            map_pose_reliable=True, map_pose_uncertainty_m=self.context['bridge_uncertainty_m'])
        self.assertFalse(original['safe'])
        for key, value in [('updated', 9.), ('reliable', False),
                           ('quality', {'reason': 'map_quality_insufficient'}),
                           ('bridge_uncertainty_m', .36), ('bridge_uncertainty_m', math.nan)]:
            old = self.context[key]
            with self.subTest(key=key):
                self.context[key] = value
                self.assertFalse(navigation_rotation_support(original, self.scan, self.context, now=10.1)['safe'])
                self.context[key] = old
        self.scan.received_s = 9.
        self.assertFalse(self.assess()['safe'])

    def test_no_map_or_scan_rewrite(self):
        layers = [getattr(self.map, n).copy() for n in ('occupied', 'unknown', 'temporary_occupied')]
        ranges = self.scan.distances_m.copy()
        self.assertTrue(self.assess()['safe'])
        for before, name in zip(layers, ('occupied', 'unknown', 'temporary_occupied')):
            np.testing.assert_array_equal(before, getattr(self.map, name))
        np.testing.assert_array_equal(ranges, self.scan.distances_m)

    def test_entry_and_turn_start_callback_pass_without_eight_zero_frames(self):
        self._assert_entry_and_turn_start_callback_pass()

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def test_three_run_body_edges_preserve_turn_command_without_forward_recovery(self):
        frames=json.loads((Path(__file__).parent/'fixtures/body_occlusion_20261003.json').read_text(encoding='utf-8'))['frames']
        for frame in frames:
            a=frame['assessment']
            if a['reason']!='unobserved_sector_without_safe_map_support':continue
            with self.subTest(run=frame['source_run'],file=frame['source_file']):
                p=a['map_support']['pose']
                self.pose=NS(x_m=p[0],y_m=p[1],yaw_rad=p[2])
                self.scan=NS(angles_rad=np.asarray(frame['angles_rad'],dtype=float),
                    distances_m=np.asarray(frame['distances_m'],dtype=float),received_s=10.)
                self._assert_entry_and_turn_start_callback_pass()

    def _assert_entry_and_turn_start_callback_pass(self):
        previous_route = planner.LATEST_ROUTE
        previous_callback = turn_policy.TELEMETRY_CALLBACK
        events, rechecks = [], []
        try:
            planner.LATEST_ROUTE = {'navigation_stage': 'approach', 'target_region': {'center_xy_m': [1., 1.], 'radius_m': .9}}
            turn_policy.TELEMETRY_CALLBACK = lambda kind, **fields: events.append(dict(event=kind, **fields))
            class Log:
                def event(self, *a, **k):
                    return k
            class Localizer:
                def track(inner, scan, pose):
                    return result
            result = NS(pose=self.pose, score=-1., rmse_m=.02, inlier_ratio=.95)
            mission = NS(ScanMapLocalizer=Localizer, MissionLog=Log,
                _phase1_verified_glass_required=True,
                build_search_route=lambda *a, **k: planner.LATEST_ROUTE,
                _strict_command_for_pose=lambda *a, **k: (0, 1200, False),
                _fuse_tracking_pose=lambda *a: self.pose,
                _obstacle_safe=lambda *a, **k: True, _remember_temporary_obstacle=lambda *a: 0,
                _phase1_navigation_estimate={'result': result, 'started_s': 10., 'uncertainty_m': .08623039})
            _install_waypoint_obstacle_guard(mission)
            mission.build_search_route(self.map, (self.pose.x_m, self.pose.y_m), {})
            localizer = Localizer()
            localizer._phase1_stationary_map_recheck = lambda *a: rechecks.append(True)
            with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=10.1):
                localizer.track(self.scan, self.pose)
                mission._fuse_tracking_pose(self.pose, result)
                for _ in range(10):
                    self.assertEqual(mission._strict_command_for_pose(), (0, 1200, False))
                    self.assertTrue(mission._obstacle_safe(self.scan, 0, .45))
                    self.assertTrue(mission._phase1_rotation_assess(self.scan)['safe'])
                self.assertTrue(mission._phase1_authorize_motion(0, 3500, 650))
            self.assertEqual(rechecks, [])
            self.assertFalse(any(e.get('event') == 'rotation_clearance_wait' for e in events))
        finally:
            planner.LATEST_ROUTE = previous_route
            turn_policy.TELEMETRY_CALLBACK = previous_callback
            turn_policy.LAST_REQUEST = (0, 0)


if __name__ == '__main__':
    unittest.main()
