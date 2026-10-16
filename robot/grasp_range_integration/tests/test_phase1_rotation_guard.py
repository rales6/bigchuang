"""Software-only guard regression. No camera, lidar or motor IO."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from grasp_range_integration.phase1.rotation_guard import (
    assess_rotation, installed_body_mask, body_mask_components, map_rotation_clear, map_rotation_evidence, RotationClearanceError,
)
from grasp_range_integration.phase1.relationship_find_only_entry import _install_waypoint_obstacle_guard
from grasp_range_integration.phase1 import turn_policy
from raspberry_pi.known_map_navigation import KnownOccupancyMap


def scan():
    return SimpleNamespace(angles_rad=np.deg2rad(np.arange(-180, 180, 2)), distances_m=np.ones(180))


def free_map():
    return KnownOccupancyMap(np.full((80, 80), 255, dtype=np.uint8), .05, -2, -2)


POSE = SimpleNamespace(x_m=0., y_m=0., yaw_rad=0.)


class RotationGuardTests(unittest.TestCase):
    def tearDown(self):
        turn_policy.LAST_REQUEST = (0, 0)
        turn_policy.TELEMETRY_CALLBACK = None

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def installation_scan(self):
        current = scan()
        # Existing actual rear returns, plus all five rejected edge points
        # from 20260926_135005. This composes recorded rays, NOT a saved full
        # live scan. Full-circle exterior rays are synthetic safety fixtures.
        angles = [-102, -103, -108, -105, -104, -186, -214, -187,
                  111,114,113,110,110]
        distances = [.103,.111,.111,.127,.127,.158,.158,.237,
                     .140,.155,.155,.1475,.141]
        current.angles_rad=np.r_[current.angles_rad,np.deg2rad(angles)]
        current.distances_m=np.r_[current.distances_m,distances]
        return current

    def test_all_five_logged_edge_points_are_filtered_separately(self):
        current=self.installation_scan()
        legacy,patch=body_mask_components(current.angles_rad,current.distances_m)
        self.assertEqual(np.count_nonzero(legacy),8)
        self.assertEqual(np.count_nonzero(patch),5)
        self.assertTrue(patch[-5:].all())
        decision=assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True)
        self.assertTrue(decision['safe'])
        self.assertEqual(decision['installation_patch_count'],5)
        self.assertAlmostEqual(decision['map_checked_radius_m'],.44)
        self.assertEqual(len(decision['installation_patch_points']),5)

    def test_edge_patch_cannot_authorize_motion_without_map_even_with_full_scan(self):
        current=self.installation_scan()
        for known,reliable in [(None,True),(free_map(),False)]:
            result=assess_rotation(current,known_map=known,pose=POSE,map_pose_reliable=reliable)
            self.assertFalse(result['safe'])
            self.assertEqual(result['reason'],'installation_edge_occlusion_without_safe_map_support')
        known=free_map()
        cell=known.world_to_grid(((.30,0),))[0]
        for mask in (known.occupied,known.unknown,known.temporary_occupied):
            mask[cell[1],cell[0]]=True
            self.assertFalse(assess_rotation(current,known_map=known,pose=POSE,map_pose_reliable=True)['safe'])
            mask[cell[1],cell[0]]=False

    def test_isolated_edge_point_without_body_signature_still_blocks(self):
        current=scan()
        current.angles_rad=np.r_[current.angles_rad,math.radians(110)]
        current.distances_m=np.r_[current.distances_m,.141]
        result=assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True)
        self.assertFalse(result['safe'])
        self.assertEqual(result['installation_patch_count'],0)

    def test_new_patch_does_not_swallow_front_opposite_side_or_larger_returns(self):
        for angle,distance in [(0,.14),(-110,.14),(90,.14),(105,.15),(120,.15),
                               (110,.18),(110,.10),(110,.25),(180,.30)]:
            current=self.installation_scan()
            current.angles_rad=np.r_[current.angles_rad,math.radians(angle)]
            current.distances_m=np.r_[current.distances_m,distance]
            legacy,patch=body_mask_components(current.angles_rad,current.distances_m)
            self.assertFalse(patch[-1],(angle,distance))
            # Some shorter points are still in the UNCHANGED legacy body
            # mask, not expanded by this installation-edge patch.
            if not legacy[-1] and distance<.30:
                self.assertFalse(assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True)['safe'])

    def test_audit_preserves_raw_arrays_and_marks_patch_not_free_space(self):
        from grasp_range_integration.phase1.inspect_body_returns import describe
        current=self.installation_scan();current.timestamp_s=1.
        raw=current.distances_m.copy()
        report=describe(current)
        np.testing.assert_array_equal(current.distances_m,raw)
        self.assertEqual(len(report['angles_rad']),len(raw))
        self.assertEqual(len(report['installation_patch_indices']),5)
        self.assertFalse(report['rotation_without_map']['safe'])

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def test_real_30_frame_opposite_edge_audit_is_filtered_but_requires_map(self):
        import json
        from pathlib import Path
        from raspberry_pi.lidar.base import LaserScan
        data=json.loads((Path(__file__).parent/'fixtures/body_mask_stationary_20260926_150426.json').read_text(encoding='utf-8'))
        self.assertFalse(data['base_connected']);self.assertFalse(data['motion_sent'])
        self.assertEqual(len(data['frames']),30)
        reviewed=0
        for frame in data['frames']:
            current=LaserScan(np.asarray(frame['angles_rad']),np.asarray(frame['distances_m'],dtype=float),0.)
            legacy,patch=body_mask_components(current.angles_rad,current.distances_m)
            for point in frame['near_points']:
                if point['classification']=='unmasked_near':
                    reviewed+=1
                    self.assertTrue(patch[point['index']],point)
            if patch.any():
                self.assertFalse(assess_rotation(current)['safe'])
                self.assertTrue(assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True)['safe'])
        self.assertGreaterEqual(reviewed,8)

    def test_opposite_edge_patch_does_not_hide_larger_or_other_direction_obstacles(self):
        for angle,distance in [(-104,.15),(-110,.138),(-100,.138),(0,.138),(-104,.18)]:
            current=self.installation_scan()
            current.angles_rad=np.r_[current.angles_rad,math.radians(angle)]
            current.distances_m=np.r_[current.distances_m,distance]
            legacy,patch=body_mask_components(current.angles_rad,current.distances_m)
            self.assertFalse(patch[-1],(angle,distance))
            if not legacy[-1]:
                self.assertFalse(assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True)['safe'])

    def test_20260929_chassis_edge_returns_are_masked(self):
        current=self.installation_scan()
        current.angles_rad=np.r_[current.angles_rad,np.deg2rad([-109,-105])]
        current.distances_m=np.r_[current.distances_m,[.133,.1405]]
        _,patch=body_mask_components(current.angles_rad,current.distances_m)
        self.assertTrue(patch[-2:].all())

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def test_20261001_opposite_edge_has_no_cartesian_gap(self):
        import json
        from pathlib import Path
        recorded = json.loads((Path(__file__).parent /
            'fixtures/rotation_edge_20261001_173725.json').read_text(encoding='utf-8'))
        current = self.installation_scan()
        current.angles_rad = np.r_[current.angles_rad, np.deg2rad(recorded['bearing_deg'])]
        current.distances_m = np.r_[current.distances_m, recorded['distance_m']]
        _, patched = body_mask_components(current.angles_rad, current.distances_m)
        self.assertTrue(patched[-2:].all())
        # Cover the bounded geometry continuously, not just two exact values.
        for distance in np.linspace(.140, .145, 30):
            a = np.r_[current.angles_rad, math.radians(-105)]
            d = np.r_[current.distances_m, distance]
            self.assertTrue(body_mask_components(a, d)[1][-1], distance)
        self.assertFalse(assess_rotation(current)['safe'])
        self.assertTrue(assess_rotation(current, known_map=free_map(), pose=POSE,
                                        map_pose_reliable=True)['safe'])

    def test_sparse_logged_near_returns_use_installed_geometry_not_cluster_count(self):
        # These are the diagnostic's observed near returns, NOT a full live scan.
        angles = np.deg2rad([-102, -103, -108, -105, -104, -186, -214, -187])
        distances = np.array([.103, .111, .111, .127, .127, .158, .158, .237])
        self.assertTrue(installed_body_mask(angles, distances).all())
        current = scan()
        current.angles_rad = np.r_[current.angles_rad, angles]
        current.distances_m = np.r_[current.distances_m, distances]
        decision = assess_rotation(current)
        self.assertTrue(decision['safe'])
        self.assertEqual(decision['self_return_count'], len(angles))
        self.assertEqual(decision['min_distance_m'], 1.)

    def test_mask_matches_previously_deployed_panorama_contract(self):
        from raspberry_pi.open_space_panorama import _filter_body_returns
        current = scan()
        current.distances_m = np.linspace(.03, .32, len(current.distances_m))
        filtered = _filter_body_returns(current, .19, .14, 0., .125)
        expected = np.isnan(filtered.distances_m)
        np.testing.assert_array_equal(installed_body_mask(current.angles_rad, current.distances_m), expected)

    def test_actual_near_obstacle_and_outside_mask_are_kept(self):
        for angle, distance in [(0, .103), (-105, .14), (90, .13), (180, .34), (0, .05)]:
            current = scan()
            current.angles_rad = np.r_[current.angles_rad, math.radians(angle)]
            current.distances_m = np.r_[current.distances_m, distance]
            decision = assess_rotation(current, known_map=free_map(), pose=POSE, map_pose_reliable=True)
            if distance < .30:
                self.assertFalse(decision['safe'], (angle, distance, decision))
                self.assertEqual(decision['reason'], 'exterior_near_obstacle')
                self.assertAlmostEqual(decision['min_distance_m'], distance)
            else:
                self.assertEqual(decision['self_return_count'], 0)

    def test_blind_sector_requires_fresh_reliable_local_map_support(self):
        current = scan()
        current.distances_m[(np.abs(current.angles_rad) > math.radians(149))] = math.inf
        self.assertFalse(assess_rotation(current)['safe'])
        known = free_map()
        self.assertFalse(assess_rotation(current, known_map=known, pose=POSE)['safe'])
        decision = assess_rotation(current, known_map=known, pose=POSE, map_pose_reliable=True)
        self.assertTrue(decision['safe'])
        self.assertEqual(decision['reason'], 'lidar_with_local_map_support')
        for mask in (known.occupied, known.unknown, known.temporary_occupied):
            cell = known.world_to_grid(((.30, 0),))[0]
            mask[cell[1], cell[0]] = True
            self.assertFalse(assess_rotation(current, known_map=known, pose=POSE, map_pose_reliable=True)['safe'])
            mask[cell[1], cell[0]] = False

    def test_missing_half_scan_never_authorized_by_map(self):
        current = scan()
        current.distances_m[current.angles_rad < 0] = math.inf
        self.assertFalse(assess_rotation(current, known_map=free_map(), pose=POSE, map_pose_reliable=True)['safe'])

    def test_all_body_or_invalid_scan_is_not_free(self):
        current = SimpleNamespace(angles_rad=np.array([math.pi] * 40), distances_m=np.array([.14] * 40))
        self.assertFalse(assess_rotation(current, known_map=free_map(), pose=POSE, map_pose_reliable=True)['safe'])
        current.distances_m = np.array([])
        self.assertEqual(assess_rotation(current)['reason'], 'invalid_scan')
        self.assertFalse(map_rotation_clear(free_map(), SimpleNamespace(x_m=1.9, y_m=0, yaw_rad=0), .44))

    def test_octant_boundary_hole_does_not_mean_45_degree_blindness(self):
        current = scan()
        # A 44-degree hole around the old 180/225-degree bin boundary remains
        # within the existing 45-degree evidence tolerance.
        current.distances_m[(current.angles_rad > math.radians(-178))
                            & (current.angles_rad < math.radians(-138))] = math.inf
        self.assertTrue(assess_rotation(current)['safe'])

    def adapted_mission(self):
        calls = []
        mission = SimpleNamespace(
            _obstacle_safe=lambda *args, **kwargs: calls.append((args, kwargs)) or False,
            build_search_route=lambda *args, **kwargs: 'route',
            _fuse_tracking_pose=lambda predicted, result: predicted,
        )
        _install_waypoint_obstacle_guard(mission)
        return mission, calls

    def test_turn_veto_raises_before_frontal_obstacle_branch(self):
        mission, calls = self.adapted_mission()
        events = []
        turn_policy.TELEMETRY_CALLBACK = lambda kind, **data: events.append((kind, data))
        turn_policy.LAST_REQUEST = (0, 1200)
        current = scan()
        current.distances_m[90] = .1  # front, never a body return
        with self.assertRaisesRegex(RotationClearanceError, '未添加前方障碍'):
            mission._obstacle_safe(current, 0, .45)
        self.assertEqual(calls, [])
        self.assertEqual(events[-1][0], 'rotation_clearance_rejected')
        self.assertEqual(events[-1][1]['reason'], 'exterior_near_obstacle')
        self.assertEqual(events[-1][1]['min_bearing_deg'], 0.)
        # Translation still uses precisely the old forward guard arguments.
        mission._obstacle_safe(current, 240, .45)
        self.assertEqual(calls[-1], ((current, 240, .45), {'angular_mrad_s': 0.0}))

    def test_adapter_only_uses_recent_strong_consistent_map_pose(self):
        mission, _ = self.adapted_mission()
        turn_policy.LAST_REQUEST = (0, 1200)
        mission.build_search_route(free_map(), (0, 0), {})
        current = scan()
        current.distances_m[np.abs(current.angles_rad) > math.radians(149)] = math.inf
        result = SimpleNamespace(pose=POSE, rmse_m=.04, inlier_ratio=.8)
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=10):
            mission._fuse_tracking_pose(POSE, result)
            self.assertTrue(mission._obstacle_safe(current, 0, .45))
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=12):
            with self.assertRaises(RotationClearanceError):
                mission._obstacle_safe(current, 0, .45)
        for bad in (SimpleNamespace(pose=POSE, rmse_m=.12, inlier_ratio=.8),
                    SimpleNamespace(pose=SimpleNamespace(x_m=.5, y_m=0, yaw_rad=0), rmse_m=.04, inlier_ratio=.8)):
            mission._fuse_tracking_pose(POSE, bad)
            with self.assertRaises(RotationClearanceError):
                mission._obstacle_safe(current, 0, .45)

    def test_one_near_frame_holds_zero_then_fresh_clear_frame_authorizes_turn(self):
        result = SimpleNamespace(pose=POSE, rmse_m=.04, inlier_ratio=.8)
        class Localizer:
            def track(self, scan, pose): return result
        def requested(*args):
            turn_policy.LAST_REQUEST = (0, 1200)
            return 0, 1200, False
        mission = SimpleNamespace(ScanMapLocalizer=Localizer,
                                  _strict_command_for_pose=requested,
                                  _obstacle_safe=lambda *args, **kwargs: True,
                                  build_search_route=lambda *args: {},
                                  _fuse_tracking_pose=lambda pose, result: pose)
        _install_waypoint_obstacle_guard(mission)
        mission.build_search_route(free_map(), (0, 0), {})
        localizer = Localizer()
        bad = scan(); bad.distances_m[90] = .1
        localizer.track(bad, POSE); mission._fuse_tracking_pose(POSE, result)
        self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
        self.assertEqual(turn_policy.LAST_REQUEST, (0, 0))
        clear = scan()
        localizer.track(clear, POSE); mission._fuse_tracking_pose(POSE, result)
        self.assertEqual(mission._strict_command_for_pose(), (0, 1200, False))
        self.assertTrue(mission._obstacle_safe(clear, 0, .45))

    def test_three_persistent_near_frames_never_authorize_rotation(self):
        result = SimpleNamespace(pose=POSE, rmse_m=.04, inlier_ratio=.8)
        class Localizer:
            def track(self, scan, pose): return result
        mission = SimpleNamespace(ScanMapLocalizer=Localizer,
                                  _strict_command_for_pose=lambda *args: (0, 1200, False),
                                  _obstacle_safe=lambda *args, **kwargs: True,
                                  build_search_route=lambda *args: {},
                                  _fuse_tracking_pose=lambda pose, result: pose)
        _install_waypoint_obstacle_guard(mission)
        localizer = Localizer()
        bad = scan(); bad.distances_m[90] = .1
        for _ in range(2):
            localizer.track(bad, POSE); mission._fuse_tracking_pose(POSE, result)
            self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
        localizer.track(bad, POSE); mission._fuse_tracking_pose(POSE, result)
        with self.assertRaises(RotationClearanceError): mission._strict_command_for_pose()

    def recovery_mission(self):
        result = SimpleNamespace(pose=POSE, rmse_m=.04, inlier_ratio=.8, score=1.)
        class Localizer:
            def track(self, current, pose): return result
        class Log:
            def event(self, name, **fields): return dict(type=name, **fields)
        remembered = []
        self.route_requests = []
        def build(_known, _start, request):
            self.route_requests.append(dict(request))
            return {}
        def requested(*args):
            turn_policy.LAST_REQUEST = (0, 1200)
            return 0, 1200, False
        mission = SimpleNamespace(ScanMapLocalizer=Localizer, MissionLog=Log,
            _strict_command_for_pose=requested, _obstacle_safe=lambda *a, **k: True,
            _remember_temporary_obstacle=lambda *a: remembered.append(a) or 99,
            build_search_route=build, _fuse_tracking_pose=lambda pose, value: pose)
        _install_waypoint_obstacle_guard(mission)
        known = free_map()
        mission.build_search_route(known, (0, 0), {})
        return mission, Localizer(), known, result, remembered

    def test_map_evidence_identifies_each_blocking_layer_without_mutation(self):
        known = free_map()
        cell = known.world_to_grid(((.30, 0),))[0]
        for name in ('occupied', 'unknown', 'temporary_occupied'):
            layer = getattr(known, name)
            layer[cell[1], cell[0]] = True
            value = map_rotation_evidence(known, POSE, .44)
            self.assertFalse(value['clear'])
            self.assertEqual(value['reason'], 'blocked_cells')
            self.assertEqual(value['layers'][name]['count'], 1)
            self.assertLess(value['layers'][name]['nearest_distance_m'], .44)
            self.assertTrue(layer[cell[1], cell[0]])
            layer[cell[1], cell[0]] = False
        self.assertTrue(map_rotation_evidence(known, POSE, .44)['clear'])

    def test_rotation_map_support_replan_offers_forward_view_only_with_strong_pose(self):
        mission, localizer, known, result, _ = self.recovery_mission()
        cell = known.world_to_grid(((.30,0),))[0]
        known.temporary_occupied[cell[1],cell[0]] = True
        current = self.installation_scan()
        for _ in range(8):
            localizer.track(current,POSE)
            mission._fuse_tracking_pose(POSE,result)
            mission._strict_command_for_pose()
            mission._obstacle_safe(current,0,.45)
        mission._remember_temporary_obstacle(current)
        event=mission.MissionLog().event('temporary_obstacle')
        self.assertEqual(event['recovery_cause'],'rotation_map_support_recovery')
        mission.build_search_route(known,(0,0),{'mode':'semantic_search_region'})
        self.assertEqual(self.route_requests[-1]['_phase1_forward_view_yaw_rad'],0.)
        # The trigger is consumed, never silently reused in later replans.
        mission.build_search_route(known,(0,0),{'mode':'semantic_search_region'})
        self.assertNotIn('_phase1_forward_view_yaw_rad',self.route_requests[-1])

    def test_map_deficit_refresh_then_replan_not_three_frame_exit_or_ghost_wall(self):
        mission, localizer, known, result, remembered = self.recovery_mission()
        cell = known.world_to_grid(((.30, 0),))[0]
        known.temporary_occupied[cell[1], cell[0]] = True
        before = known.temporary_occupied.copy()
        rechecks = []
        localizer._phase1_stationary_map_recheck = lambda *a: rechecks.append(a) or result
        current = self.installation_scan()
        for i in range(8):
            localizer.track(current, POSE)
            mission._fuse_tracking_pose(POSE, result)
            self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
            self.assertEqual(mission._obstacle_safe(current, 0, .45), i < 7)
        self.assertEqual(len(rechecks), 1)
        self.assertEqual(mission._remember_temporary_obstacle(current), 0)
        self.assertEqual(remembered, [])
        event = mission.MissionLog().event('temporary_obstacle', added_cells=0)
        self.assertEqual(event['type'], 'navigation_no_progress')
        self.assertEqual(event['recovery_cause'], 'rotation_map_support_recovery')
        self.assertEqual(event['assessment']['map_support']['layers']['temporary_occupied']['count'], 1)
        np.testing.assert_array_equal(known.temporary_occupied, before)
        from grasp_range_integration.phase1.relationship_find_only_entry import _phase1_directive
        self.assertEqual(_phase1_directive({'latest_event': event}, {}, allow_low=False)['action'], 'replan')
        # After that one synthetic replan event, genuine frontal obstacles
        # keep the old writer and their own event type.
        self.assertEqual(mission._remember_temporary_obstacle(current), 99)
        self.assertEqual(mission.MissionLog().event('temporary_obstacle')['type'], 'temporary_obstacle')

    def test_near_obstacle_blocks_motion_and_enters_dedicated_replan(self):
        mission, localizer, known, result, remembered = self.recovery_mission()
        current = self.installation_scan(); current.distances_m[90] = .1
        for _ in range(2):
            localizer.track(current, POSE); mission._fuse_tracking_pose(POSE, result)
            self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
        localizer.track(current, POSE); mission._fuse_tracking_pose(POSE, result)
        self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
        self.assertFalse(mission._obstacle_safe(current, 0, .45))
        self.assertEqual(mission._remember_temporary_obstacle(current), 0)
        event = mission.MissionLog().event('temporary_obstacle')
        self.assertEqual(event['type'], 'navigation_no_progress')
        self.assertEqual(event['recovery_cause'], 'rotation_obstacle_recovery')
        self.assertEqual(remembered, [])

    def test_combined_turn_policy_zero_hold_reaches_replan_despite_pose_drift(self):
        # Reproduce the 17:30 run with BOTH wrappers installed in production
        # order. Testing the guard alone missed the earlier convergence error.
        class Controller:
            def __init__(self): self.last_command = None
            def observe(self, update): pass
            def stopped(self): self.last_command = None
            def ttl_for(self, command, ttl): return ttl
        class Localizer:
            def track(self, current, pose): return result
        class Log:
            def event(self, name, **fields): return dict(type=name, **fields)
        result = SimpleNamespace(pose=POSE, rmse_m=.04, inlier_ratio=.8, score=1.)
        mission = SimpleNamespace(AdaptiveSpeedController=Controller,
            ScanMapLocalizer=Localizer, MissionLog=Log,
            _strict_command_for_pose=lambda *a: (0, 1200, False),
            _obstacle_safe=lambda *a, **k: True, _remember_temporary_obstacle=lambda *a: 99,
            build_search_route=lambda *a: {}, _fuse_tracking_pose=lambda pose, value: pose)
        turn_policy.install_turn_policy(mission)
        _install_waypoint_obstacle_guard(mission)
        known = free_map()
        cell = known.world_to_grid(((.30, 0),))[0]
        known.temporary_occupied[cell[1], cell[0]] = True
        mission.build_search_route(known, (0, 0), {})
        controller, localizer = Controller(), Localizer()
        current = self.installation_scan()
        rechecks = []
        localizer._phase1_stationary_map_recheck = lambda *a: rechecks.append(a) or result
        feedback = SimpleNamespace(accepted=True, rmse_m=.02, inlier_ratio=.9,
                                   rotation_rad=0., translation_m=0.)
        for index, degrees in enumerate((81.76, 86.98, 92., 96.25, 100., 104., 108., 112.)):
            pose = SimpleNamespace(x_m=0., y_m=0., yaw_rad=math.radians(-degrees))
            result.pose = pose
            localizer.track(current, pose)
            mission._fuse_tracking_pose(pose, result)
            controller.observe(feedback)
            self.assertEqual(mission._strict_command_for_pose(object(), pose, (1., 0.)), (0, 0, False))
            controller.last_command = SimpleNamespace(linear_mm_s=0, angular_mrad_s=0)
            self.assertEqual(mission._obstacle_safe(current, 0, .45), index < 7)
        self.assertEqual(len(rechecks), 1)
        self.assertEqual(mission._remember_temporary_obstacle(current), 0)
        event = mission.MissionLog().event('temporary_obstacle')
        self.assertEqual(event['recovery_cause'], 'rotation_map_support_recovery')

    def test_odometry_bridge_is_not_reliable_rotation_map_support(self):
        mission, localizer, known, result, _ = self.recovery_mission()
        result.score = -1.
        current = self.installation_scan()
        localizer.track(current, POSE); mission._fuse_tracking_pose(POSE, result)
        value = mission._phase1_rotation_assess(current)
        self.assertFalse(value['safe'])
        self.assertFalse(value['pose_quality']['map_match'])
        self.assertFalse(value['map_pose_reliable'])

    def test_near_replan_writes_only_distinct_fresh_stopped_verified_scans(self):
        mission, localizer, known, result, remembered = self.recovery_mission()
        counter = []
        def fresh(reference):
            current = self.installation_scan(); current.distances_m[90] = .24
            current.received_s = 10.99
            current.sequence = len(counter) + 1
            counter.append(current)
            return current, True
        mission._phase1_refresh_rotation_evidence = fresh
        old = self.installation_scan(); old.received_s = 10.
        args = SimpleNamespace(temporary_obstacle_radius=.12, temporary_obstacle_min_hits=2)
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=11.):
            for _ in range(3):
                localizer.track(old, POSE); mission._fuse_tracking_pose(POSE, result)
                self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
            self.assertFalse(mission._obstacle_safe(old, 0, .45))
            self.assertGreater(mission._remember_temporary_obstacle(old, None, POSE, known, args), 0)
        self.assertEqual(remembered, [])
        event = mission.MissionLog().event('temporary_obstacle')
        self.assertEqual(event['recovery_cause'], 'rotation_obstacle_recovery')
        self.assertGreater(event['added_cells'], 0)

    def test_weak_near_pose_gets_recheck_then_bounded_replan_without_map_paint(self):
        mission, localizer, known, result, remembered = self.recovery_mission()
        result.rmse_m = .12
        current = self.installation_scan(); current.distances_m[90] = .24
        rechecks = []
        localizer._phase1_stationary_map_recheck = lambda *a: rechecks.append(a)
        before = known.temporary_occupied.copy()
        for index in range(8):
            localizer.track(current, POSE); mission._fuse_tracking_pose(POSE, result)
            self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
            self.assertEqual(mission._obstacle_safe(current, 0, .45), index < 7)
        self.assertEqual(len(rechecks), 1)
        args = SimpleNamespace(temporary_obstacle_radius=.12, temporary_obstacle_min_hits=2)
        self.assertEqual(mission._remember_temporary_obstacle(current, None, POSE, known, args), 0)
        np.testing.assert_array_equal(before, known.temporary_occupied)
        self.assertEqual(remembered, [])

    def test_forward_reserve_uses_fresh_stopped_confirmation_before_writing(self):
        mission, localizer, known, result, remembered = self.recovery_mission()
        current = scan(); current.distances_m[90] = .80; current.received_s = 10.
        frames = []
        def refresh(reference):
            fresh = scan(); fresh.distances_m[90] = .80
            fresh.received_s = 10.99; fresh.sequence = len(frames)+1
            frames.append(fresh)
            return fresh, True
        mission._phase1_refresh_rotation_evidence = refresh
        args = SimpleNamespace(temporary_obstacle_radius=.12, temporary_obstacle_min_hits=2)
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=11.):
            localizer.track(current, POSE); mission._fuse_tracking_pose(POSE, result)
            turn_policy.LAST_REQUEST = (400, 0)
            self.assertFalse(mission._obstacle_safe(current, 400, .42))
            self.assertFalse(known.temporary_occupied.any())
            self.assertGreater(mission._remember_temporary_obstacle(current, None, POSE, known, args), 0)
        self.assertEqual(len(frames), 2)
        event = mission.MissionLog().event('temporary_obstacle')
        self.assertEqual(event['recovery_cause'], 'navigation_turn_space_recovery')

    def test_stale_frame_after_recheck_cannot_authorize_turn(self):
        mission, localizer, known, result, _ = self.recovery_mission()
        current = self.installation_scan(); current.received_s = 10.
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=11.):
            localizer.track(current, POSE); mission._fuse_tracking_pose(POSE, result)
            value = mission._phase1_rotation_assess(current)
            self.assertFalse(value['safe'])
            self.assertEqual(value['reason'], 'stale_scan_after_stationary_recheck')
            self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))

    def test_fresh_strong_map_support_resumes_same_task_before_replan(self):
        mission, localizer, known, result, _ = self.recovery_mission()
        current = self.installation_scan()
        for _ in range(3):
            localizer.track(current, POSE)
            mission._fuse_tracking_pose(POSE, SimpleNamespace(pose=POSE, rmse_m=.12, inlier_ratio=.8))
            self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
        called = []
        localizer._phase1_stationary_map_recheck = lambda *a: called.append(a) or result
        value = localizer.track(current, POSE); mission._fuse_tracking_pose(POSE, value)
        self.assertEqual(len(called), 1)
        self.assertEqual(mission._strict_command_for_pose(), (0, 1200, False))

    def test_one_102_point_after_stale_frames_does_not_inherit_stale_count(self):
        mission, localizer, known, result, _ = self.recovery_mission()
        old = self.installation_scan(); old.received_s = 10.
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=11.):
            for _ in range(3):
                localizer.track(old, POSE); mission._fuse_tracking_pose(POSE, result)
                self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
        near = self.installation_scan()
        near.angles_rad = np.r_[near.angles_rad, math.radians(102)]
        near.distances_m = np.r_[near.distances_m, .158]
        localizer.track(near, POSE); mission._fuse_tracking_pose(POSE, result)
        self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
        clear = self.installation_scan()
        localizer.track(clear, POSE); mission._fuse_tracking_pose(POSE, result)
        self.assertEqual(mission._strict_command_for_pose(), (0, 1200, False))

    def test_slow_map_fit_refreshes_stopped_clearance_not_threshold_widening(self):
        mission, localizer, known, result, _ = self.recovery_mission()
        old = self.installation_scan(); old.received_s = 10.
        fresh = self.installation_scan(); fresh.received_s = 10.95
        calls = []
        mission._phase1_refresh_rotation_evidence = lambda s: calls.append(s) or (fresh, True)
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=11.):
            localizer.track(old, POSE); mission._fuse_tracking_pose(POSE, result)
            self.assertEqual(mission._strict_command_for_pose(), (0, 1200, False))
            self.assertTrue(mission._obstacle_safe(old, 0, .45))
        self.assertEqual(calls, [old])

    def test_new_clearance_with_large_motion_cannot_reuse_old_map_pose(self):
        mission, localizer, known, result, _ = self.recovery_mission()
        old = self.installation_scan(); old.received_s = 10.
        fresh = self.installation_scan(); fresh.received_s = 10.95
        mission._phase1_refresh_rotation_evidence = lambda s: (fresh, False)
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic', return_value=11.):
            localizer.track(old, POSE); mission._fuse_tracking_pose(POSE, result)
            self.assertEqual(mission._strict_command_for_pose(), (0, 0, False))
            self.assertFalse(mission._phase1_rotation_assess(old)['map_pose_reliable'])


if __name__ == '__main__':
    unittest.main()
