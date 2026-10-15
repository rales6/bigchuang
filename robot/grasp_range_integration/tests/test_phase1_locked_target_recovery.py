"""Station/yaw feedback regression; all IO mocked, never move hardware."""
from copy import deepcopy
import math
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import numpy as np

from grasp_range_integration.phase1.observation_recovery import (
    CandidateRecovery, colour_visual_cues, candidate,
)
from grasp_range_integration.phase1 import visibility_route_planner as planner, turn_policy
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.mapping import Pose2D


def target(box=(700, 170, 740, 215), camera="right"):
    return dict(camera_id=camera, class_name="block", source="phase1_deepseek_green_block",
                confidence=.95, bbox_1000=list(box), attributes={"color": "green"},
                single_camera_match=True, complete_in_frame=True, dual_camera_confirmed=False,
                search_evidence=dict(source_ok=True, class_ok=True, colour_ok=True, bbox_valid=True))


def fragment(top=True, colour="green"):
    return dict(camera_id="left", class_name=colour+"_region",
                source="phase1_"+colour+"_visual_cue", attributes={"color": colour},
                confidence=.55, bbox_1000=[140, 0 if top else 996, 165, 4 if top else 1000],
                single_camera_match=False, complete_in_frame=False, dual_camera_confirmed=False,
                search_evidence=dict(visual_cue_only=True, colour_ok=True, bbox_valid=True,
                                     vertical_edge_fragment=True, mission_success_allowed=False))


class LockedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.recovery = CandidateRecovery()
        self.recovery.context = dict(robot_pose=[0., 0., 0.], task_region_view={"eligible": True})
        self.route = dict(mode="semantic_visibility_regions", maximum_total_views=8,
                          views=[dict(view_xy_m=[0., 0.], target_yaw_rad=0.,
                                      visibility_zone=dict(sweep_count=1))],
                          target_region=dict(center_xy_m=[2.2, 0.], radius_m=.95,
                                             minimum_standoff_from_region_m=.8))
        self.state = dict(latest_event={"type": "observation_complete"}, current_view_index=0,
                          remaining_view_indices=[], attempts=1, detections=[target()])
        self.old_map = planner.LATEST_KNOWN_MAP
        planner.LATEST_KNOWN_MAP = KnownOccupancyMap(np.full((200, 200), 255, dtype=np.uint8), .05, -5, -5)

    def tearDown(self):
        planner.LATEST_KNOWN_MAP = self.old_map
        planner.ACTIVE_VIEW = planner.LATEST_ROUTE = None

    def observe(self, detections=None, pose=None):
        if detections is not None:
            self.state["detections"] = detections
        if pose is not None:
            self.recovery.context["robot_pose"] = pose
        self.state["remaining_view_indices"].clear()
        directive = self.recovery.next(self.state, self.route)
        self.state["attempts"] += 1
        return directive

    def lock(self, box=(700, 170, 740, 215)):
        self.observe([target(box)])
        return self.observe()

    def exhaust_yaw(self):
        """Isolate map-shift checks AFTER the local angular candidates were used."""
        lock = self.recovery.target_lock
        lock['pending'] = None
        lock['yaw_goals'] = [lock['origin_yaw'] + math.radians(d) for d in (3, -3, 6, -6, 9, -9)]

    def test_verified_lock_has_one_extra_observation_reserve(self):
        self.lock()
        self.assertEqual(self.route["maximum_total_views"], 20)
        self.observe()
        self.assertEqual(self.route["maximum_total_views"], 20)

    def test_unverified_colour_cue_does_not_authorize_station_move(self):
        cue = target((380, 170, 415, 205))
        cue.update(single_camera_match=False, source="phase1_green_visual_cue", class_name="green_region")
        cue["search_evidence"].update(source_ok=False, class_ok=False, visual_cue_only=True)
        self.observe([cue, fragment()])
        directive = self.observe()
        self.assertIsNone(self.recovery.target_lock)
        self.assertFalse(directive["recovery"]["linear_motion_allowed"])

    def test_worsening_yaw_returns_to_best_without_chasing_turn_translation(self):
        command = self.lock()
        self.assertLess(command["recovery"]["step_deg"], 0)
        # Turn moved the object farther RIGHT, opposite the intended centering.
        undo = self.observe([target((750, 170, 790, 215))], [.01, -.02, -.06])
        self.assertEqual(undo["reason"], "single_camera_return_best_heading")
        self.assertEqual(self.route["views"][0]["target_yaw_rad"], 0.)
        self.assertEqual(self.route["views"][0]["view_xy_m"], [.01, -.02])
        self.assertFalse(undo["recovery"]["linear_motion_allowed"])

    def test_target_lost_after_turn_rolls_back_not_sweeps_blindly(self):
        self.lock()
        undo = self.observe([], [0., 0., -.08])
        self.assertEqual(undo["reason"], "single_camera_return_best_heading")
        self.assertTrue(undo["recovery"]["target_lost"])
        later = self.observe([], [0., 0., 0.])
        self.assertEqual(later["action"], "navigate_next_view")
        self.assertFalse(later["recovery"]["linear_motion_allowed"])
        self.assertFalse(later["terminal"])

    def test_upper_unverified_peer_does_not_override_confirmed_eye_yaw(self):
        self.observe([target((380, 170, 415, 205)), fragment()])
        directive = self.observe()
        self.assertEqual(directive['reason'], 'single_camera_locked_yaw_alignment')
        self.assertEqual(directive['recovery']['step_deg'], 3.)
        self.assertFalse(directive['recovery']['linear_motion_allowed'])

    def test_station_move_remains_available_after_local_yaw_probes(self):
        self.lock((380, 170, 415, 205)); self.exhaust_yaw()
        directive = self.observe([target((380, 170, 415, 205)), fragment()])
        self.assertEqual(directive["reason"], "locked_target_station_shift")
        self.assertEqual(directive["recovery"]["station_shift_m"], .12)
        self.assertTrue(directive["recovery"]["linear_motion_allowed"])
        self.assertEqual(directive["recovery"]["step_deg"], 0.)
        self.assertTrue(directive["recovery"]["vertical_edge"])
        self.assertFalse(any(d.get("dual_camera_confirmed") for d in self.state["detections"]))

    def test_rejected_12cm_proposal_tries_10cm_without_relaxing_path_gate(self):
        self.lock((380, 170, 415, 205))
        self.exhaust_yaw()
        def paths(_known, start, goal, *_args):
            return [] if math.dist(start, goal) > .11 else [start, goal]
        with patch.object(planner, "work_area_path", side_effect=paths):
            directive = self.observe([target((380, 170, 415, 205)), fragment()])
        self.assertEqual(directive["recovery"]["station_shift_m"], .1)
        self.assertEqual(directive["recovery"]["rejected_forward_steps_m"], [.12])

    def test_wrong_colour_peer_strip_not_treated_as_target_vertical_hint(self):
        self.observe([target(), fragment(colour="red")])
        directive = self.observe()
        self.assertEqual(directive["reason"], "single_camera_locked_yaw_alignment")

    def test_blocked_forward_move_replans_instead_of_stopping_or_blind_driving(self):
        self.lock((380, 170, 415, 205))
        self.exhaust_yaw()
        with patch.object(planner, "work_area_path", return_value=[]):
            directive = self.observe([target((380, 170, 415, 205)), fragment()])
        # Only after the local probes were explicitly exhausted may it hold
        # for no safe shift. Do not leak the global planner's exception.
        self.assertEqual(directive["action"], "report_failure")
        self.assertFalse(directive["recovery"]["linear_motion_allowed"])
        self.assertEqual(self.recovery.last_status, "locked_target_local_recovery_exhausted")

    def test_bottom_edge_does_not_reverse_into_unobserved_rear(self):
        self.lock((380, 880, 415, 950))
        directive = self.observe()
        self.assertEqual(directive["action"], "navigate_next_view")
        self.assertFalse(directive["recovery"]["linear_motion_allowed"])
        self.assertFalse(directive["terminal"])

    def test_work_area_buffer_still_rejects_forward_station_inside_region(self):
        self.route["target_region"]["center_xy_m"] = [1.80, 0.]
        self.lock((380, 170, 415, 205))
        self.exhaust_yaw()
        directive = self.observe([target((380, 170, 415, 205)), fragment()])
        self.assertEqual(directive["action"], "replan")

    def test_new_route_does_not_inherit_old_lock_into_start_navigation(self):
        self.lock()
        self.route["views"] = deepcopy(self.route["views"])
        self.recovery.context["task_region_view"] = {"eligible": False}
        self.assertIsNone(self.observe())
        self.assertIsNone(self.recovery.target_lock)

    def test_replan_drops_old_borderline_view_snapshot(self):
        self.lock((380, 170, 415, 205))
        self.exhaust_yaw()
        self.recovery.borderline_original = (0, deepcopy(self.route["views"][0]))
        with patch.object(planner, "short_forward_view_path", return_value=[]), patch.object(
                planner, "build_visibility_route", return_value={'views': [{'view_xy_m': [1., 0.]}]}):
            self.observe([target((380, 170, 415, 205)), fragment()])
        self.assertIsNone(self.recovery.borderline_original)
        self.assertIsNone(self.recovery.previous)

    def test_side_detour_is_not_a_short_forward_viewpoint_move(self):
        self.lock((380, 170, 415, 205))
        self.exhaust_yaw()
        with patch.object(planner, "work_area_path", return_value=[(0, 0), (.12, .10), (.12, 0)]):
            directive = self.observe([target((380, 170, 415, 205)), fragment()])
        self.assertEqual(directive["action"], "replan")

    def test_same_station_does_not_replenish_reserve_when_recovery_stalls(self):
        self.lock()
        for _ in range(16):
            directive = self.observe()
            if directive["action"] == "replan":
                break
        self.assertEqual(directive["action"], "replan")
        self.assertFalse(directive["terminal"])
        self.assertEqual(self.route["maximum_total_views"], 20)

    def test_target_reserve_survives_next_station_planning_at_eighth_photo(self):
        self.lock()
        previous_count = planner.OBSERVATIONS_COMPLETED
        planner.LATEST_ROUTE = self.route
        planner.OBSERVATIONS_COMPLETED = 8
        try:
            route = planner.build_visibility_route(planner.LATEST_KNOWN_MAP, (0., 0.), dict(
                center_xy_m=[2.2, 0.], radius_m=.95, maximum_views=8))
            self.assertEqual(route["maximum_total_views"], 20)
            self.assertTrue(route["locked_target_reserve_granted"])
            self.assertTrue(route["views"])
        finally:
            planner.OBSERVATIONS_COMPLETED = previous_count

    def test_true_stereo_clears_lock_without_extra_movement(self):
        self.lock()
        pair = [target(camera="left"), target()]
        for d in pair:
            d.update(confirmed_match=True, dual_camera_confirmed=True, stereo_pair_id="real")
        self.assertIsNone(self.observe(pair))
        self.assertIsNone(self.recovery.target_lock)

    def test_thin_top_strip_is_only_recovery_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Image.new("RGB", (1280, 720), (100, 100, 100))
            ImageDraw.Draw(image).rectangle((176, 0, 201, 3), fill=(40, 150, 50))
            path = Path(temporary)/"left.png"; image.save(path)
            cues = colour_visual_cues({"left": path}, "green")
        self.assertEqual(len(cues), 1)
        self.assertTrue(cues[0]["search_evidence"]["vertical_edge_fragment"])
        self.assertFalse(cues[0]["complete_in_frame"])
        self.assertFalse(cues[0]["single_camera_match"])
        self.assertIsNone(candidate(cues))

    def test_shift_exceeds_normal_route_tolerance_so_actually_requests_drive(self):
        import raspberry_pi.object_search_mission as core
        mission = SimpleNamespace(_strict_command_for_pose=core._strict_command_for_pose,
                                  AdaptiveSpeedController=type("Controller", (core.AdaptiveSpeedController,), {}))
        turn_policy.install_turn_policy(mission)
        planner.ACTIVE_VIEW = dict(view_xy_m=[.12, 0.], target_yaw_rad=0.,
                                   visibility_zone={"stop_rule": "locked_target_station_shift"})
        planner.LATEST_ROUTE = {"mode": "semantic_visibility_regions"}
        args = core.build_argument_parser().parse_args(["--decision", "unused", "--output", "unused"])
        command = mission._strict_command_for_pose(args, Pose2D(0, 0, 0), (.12, 0))
        self.assertGreater(command[0], 0)
        self.assertFalse(command[2])
        self.assertAlmostEqual(mission._phase1_arrival_distance_m, .12)


class RealCabinetMapReplayTests(unittest.TestCase):
    """Replay the actual failed geometry; never connect cameras/motors/API."""
    def setUp(self):
        self.saved = (planner.LATEST_ROUTE, planner.LATEST_KNOWN_MAP,
                      planner.ACTIVE_VIEW, list(planner.VISITED_POINTS), planner.OBSERVATIONS_COMPLETED)
        self.fixture = json.loads((Path(__file__).parent/'fixtures/locked_view_20261001_194340.json').read_text(encoding='utf-8'))
        root = Path(__file__).resolve().parents[2]
        pgm = root/'ble_scene_17.pgm'
        if not pgm.exists():
            pgm = root/'maps/semantic_release_20260919/ble_scene_17.pgm'
        self.assertTrue(pgm.is_file(), 'published map required for actual-run regression')
        self.assertEqual(hashlib.sha256(pgm.read_bytes()).hexdigest(), self.fixture['map_pgm_sha256'])
        self.assertEqual(hashlib.sha256(pgm.with_suffix('.yaml').read_bytes()).hexdigest(), self.fixture['map_yaml_sha256'])
        self.known = KnownOccupancyMap.load(pgm, pgm.with_suffix('.yaml'))
        from grasp_range_integration.phase1.glass_boundary_guard import attach_boundaries
        data = json.loads((Path(__file__).parents[1]/'phase1/glass_boundaries.json').read_text(encoding='utf-8'))
        attach_boundaries(self.known, data)
        self.pose = self.fixture['robot_pose']
        self.center = self.fixture['region_center']
        original = self.known.path
        from grasp_range_integration.phase1.navigation_avoidance import reserve_path
        # After slow model review, do NOT fabricate a fresh map yaw. Small
        # proposals still obey the full inflated-mask geometry; motor output
        # must be validated later against fresh scans by the real executor.
        self.known.path = lambda start, goal: reserve_path(self.known, start, goal, original,
            allow_short_forward=getattr(self.known, '_phase1_short_view_probe', False)
                or ((planner.ACTIVE_VIEW or {}).get('visibility_zone') or {}).get('stop_rule') ==
                    'locked_target_station_shift')
        self.known._phase1_original_path = self.known.path
        planner.LATEST_ROUTE = planner.ACTIVE_VIEW = None
        planner.LATEST_KNOWN_MAP = self.known
        planner.VISITED_POINTS[:] = self.fixture['visited_station_xy_m']
        planner.OBSERVATIONS_COMPLETED = 4

    def tearDown(self):
        (planner.LATEST_ROUTE, planner.LATEST_KNOWN_MAP, planner.ACTIVE_VIEW,
         visited, planner.OBSERVATIONS_COMPLETED) = self.saved
        planner.VISITED_POINTS[:] = visited

    def route(self):
        return dict(mode='semantic_visibility_regions', maximum_total_views=14,
            locked_target_reserve_granted=True, views=[dict(view_xy_m=self.pose[:2],
                target_yaw_rad=self.pose[2], visibility_zone={'sweep_count': 1})],
            target_region=dict(center_xy_m=self.center, radius_m=.95,
                               minimum_standoff_from_region_m=.8))

    def test_actual_run_shortens_to_10cm_exact_straight_world_path(self):
        route = self.route();recovery = CandidateRecovery()
        recovery.context = dict(robot_pose=self.pose, task_region_view={'eligible': True})
        peer = fragment();peer['bbox_1000'] = self.fixture['peer_fragment_bbox_1000']
        state = dict(latest_event={'type':'observation_complete'}, current_view_index=0,
                     remaining_view_indices=[], attempts=2,
                     detections=[target(self.fixture['primary_bbox_1000']), peer])
        planner.LATEST_ROUTE = route;planner.ACTIVE_VIEW = route['views'][0]
        for _ in range(2):
            state['remaining_view_indices'].clear()
            directive = recovery.next(state, route)
            state['attempts'] += 1
        # The new priority is angular first. Replay station geometry only
        # after those local headings have already been inspected.
        self.assertEqual(directive['reason'], 'single_camera_locked_yaw_alignment')
        recovery.target_lock['pending'] = None
        recovery.target_lock['yaw_goals'] = [self.pose[2]+math.radians(d) for d in (3,-3,6,-6,9,-9)]
        directive = recovery.next(state, route)
        self.assertEqual(directive['reason'], 'locked_target_station_shift')
        self.assertEqual(directive['recovery']['station_shift_m'], .1)
        self.assertEqual(directive['recovery']['rejected_forward_steps_m'], [.12])
        goal = route['views'][0]['view_xy_m']
        # The executor rebuilds the route too: ensure it also gets exact
        # endpoints, not a grid zigzag demanding a pivot before a tiny move.
        path = planner.work_area_path(self.known, tuple(self.pose[:2]), tuple(goal), tuple(self.center), 1.75)
        self.assertEqual(path, [tuple(self.pose[:2]), tuple(goal)])
        self.assertGreaterEqual(math.dist(goal, self.center), 1.75)
        self.assertFalse(getattr(self.known, '_phase1_short_view_probe', False))

    def test_actual_replan_can_select_distinct_nearby_station_after_locked_target(self):
        request = dict(center_xy_m=self.center, radius_m=.95, maximum_views=8)
        planner.LATEST_ROUTE = self.route()
        result = planner.build_visibility_route(self.known, tuple(self.pose[:2]), request)
        self.assertEqual(result['minimum_station_spacing_m'], .18)
        self.assertEqual(result['maximum_total_views'], 14)
        for view in result['views']:
            self.assertGreaterEqual(min(math.dist(view['view_xy_m'], old)
                for old in self.fixture['visited_station_xy_m']), .18)
            self.assertGreaterEqual(math.dist(view['view_xy_m'], self.center), 1.75)
        self.assertTrue(result['views'])

    def test_without_locked_target_the_original_station_spacing_is_unchanged(self):
        planner.LATEST_ROUTE = dict(self.route(), locked_target_reserve_granted=False)
        with self.assertRaisesRegex(RuntimeError, '没有可达'):
            planner.build_visibility_route(self.known, tuple(self.pose[:2]),
                dict(center_xy_m=self.center, radius_m=.95, maximum_views=8))

    def test_actual_runtime_reserve_wrapper_imports_and_handles_short_probe(self):
        import raspberry_pi.object_search_mission as core
        from grasp_range_integration.phase1.relationship_find_only_entry import _install_waypoint_obstacle_guard
        known = KnownOccupancyMap(np.full((100, 100), 255, dtype=np.uint8), .05, -2.5, -2.5)
        mission = SimpleNamespace(_obstacle_safe=lambda *_args: True,
            build_search_route=lambda *_args: {}, _fuse_tracking_pose=core._fuse_tracking_pose)
        _install_waypoint_obstacle_guard(mission)
        mission.build_search_route(known, (0., 0.), {})
        path = planner.short_forward_view_path(known, (0., 0.), (.1, 0.), (2.2, 0.), 1.75)
        self.assertEqual(path, [(0., 0.), (.1, 0.)])
        self.assertFalse(known._phase1_short_view_probe)


class SingleEyePriorityReplayTests(unittest.TestCase):
    """The 210758 trace must issue yaw, not 0deg/0m then global replan."""

    def setUp(self):
        LockedRecoveryTests.setUp(self)
        self.fixture = json.loads((Path(__file__).parent /
            'fixtures/locked_view_20261001_210758.json').read_text(encoding='utf-8'))

    tearDown = LockedRecoveryTests.tearDown
    observe = LockedRecoveryTests.observe
    lock = LockedRecoveryTests.lock

    def actual_detection(self, observation):
        primary = target(observation['primary_bbox_1000'], observation['primary_camera'])
        primary['search_evidence']['deepseek_review'] = self.fixture['independent_model_review']
        peer = fragment(); peer['bbox_1000'] = observation['peer_fragment_bbox_1000']
        return [peer, primary]

    def test_each_actual_failed_frame_prioritizes_measurable_yaw(self):
        for observation in self.fixture['observations']:
            with self.subTest(observation=observation['id']):
                self.recovery = CandidateRecovery()
                self.route['target_region']['center_xy_m'] = self.fixture['region_center']
                pose = observation['robot_pose']
                self.route['views'][0].update(view_xy_m=pose[:2], target_yaw_rad=pose[2])
                with patch.object(planner, 'build_visibility_route', side_effect=
                        AssertionError('visible target must not enter global planning')), patch.object(
                        planner, 'short_forward_view_path', side_effect=
                        AssertionError('yaw must precede translation')):
                    directive = self.observe(self.actual_detection(observation), pose)
                self.assertEqual(directive['reason'], 'single_camera_locked_yaw_alignment')
                self.assertEqual(directive['recovery']['step_deg'], 3.)
                self.assertFalse(directive['terminal'])
                self.assertFalse(directive['recovery']['linear_motion_allowed'])
                self.assertEqual(self.route['views'][0]['view_xy_m'], pose[:2])
                self.assertFalse(any(d.get('dual_camera_confirmed') for d in self.state['detections']))

    def test_centered_single_eye_probes_instead_of_returning_to_global_region(self):
        primary = target((485, 170, 515, 215))
        primary['search_evidence']['deepseek_review'] = self.fixture['independent_model_review']
        with patch.object(planner, 'build_visibility_route', side_effect=AssertionError('not yet')):
            directive = self.observe([primary])
        self.assertEqual(directive['reason'], 'single_camera_locked_local_probe')
        self.assertGreater(abs(directive['recovery']['step_deg']), 2.)
        self.assertFalse(directive['recovery']['linear_motion_allowed'])

    def test_actual_yaw_is_not_swallowed_by_navigation_arrival_window(self):
        import raspberry_pi.object_search_mission as core
        observation = self.fixture['observations'][0]; pose = observation['robot_pose']
        directive = self.observe(self.actual_detection(observation), pose)
        mission = SimpleNamespace(_strict_command_for_pose=core._strict_command_for_pose,
            AdaptiveSpeedController=type('Controller', (core.AdaptiveSpeedController,), {}))
        turn_policy.install_turn_policy(mission)
        planner.ACTIVE_VIEW = self.route['views'][0]; planner.LATEST_ROUTE = self.route
        args = core.build_argument_parser().parse_args(['--decision', 'unused', '--output', 'unused'])
        linear, angular, reached = mission._strict_command_for_pose(args, Pose2D(*pose), tuple(pose[:2]))
        self.assertEqual(linear, 0); self.assertGreater(angular, 0); self.assertFalse(reached)
        self.assertEqual(directive['recovery']['step_deg'], 3.)

    def test_supervisor_prioritizes_actual_target_over_region_fallback(self):
        from grasp_range_integration.phase1.relationship_find_only_entry import _install_phase1_supervisor
        observation = self.fixture['observations'][0];pose = observation['robot_pose']
        self.recovery.context.update(robot_pose=pose)
        self.state['detections'] = self.actual_detection(observation)
        self.route['views'][0].update(view_xy_m=pose[:2], target_yaw_rad=pose[2])
        planner.LATEST_ROUTE = self.route
        previous_visited = list(planner.VISITED_POINTS)
        previous_count = planner.OBSERVATIONS_COMPLETED
        mission = SimpleNamespace(GpuMissionClient=type('Gpu', (), {}))
        try:
            _install_phase1_supervisor(mission, self.recovery)
            with patch('grasp_range_integration.phase1.relationship_find_only_entry._phase1_directive',
                       side_effect=AssertionError('generic region fallback must not run')):
                directive = mission.GpuMissionClient().next(self.state)
            self.assertEqual(directive['reason'], 'single_camera_locked_yaw_alignment')
            self.assertIs(planner.ACTIVE_VIEW, self.route['views'][0])
            self.assertEqual(self.state['remaining_view_indices'], [0])
        finally:
            planner.VISITED_POINTS[:] = previous_visited
            planner.OBSERVATIONS_COMPLETED = previous_count

    def test_no_global_station_does_not_skip_remaining_local_angle(self):
        self.lock((485, 170, 515, 215))
        with patch.object(planner, 'short_forward_view_path', return_value=[]), patch.object(
                planner, 'build_visibility_route', side_effect=planner.NoReachableVisibilityError('none')) as build:
            directive = self.observe(pose=[0., 0., math.radians(3)])
        self.assertEqual(directive['action'], 'navigate_next_view')
        self.assertFalse(directive['recovery']['linear_motion_allowed'])
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
