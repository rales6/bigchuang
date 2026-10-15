import json
import math
import inspect
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from raspberry_pi.known_map_navigation import (
    InspectionTarget,
    KnownOccupancyMap,
    LocalizationResult,
    ScanMapLocalizer,
    consistent_localization,
    order_targets,
)
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping.geometry import Pose2D, normalize_angle
from raspberry_pi.mapping import (
    AdaptiveSpeedConfig,
    AdaptiveSpeedController,
    MotionCommand,
    SlamUpdate,
)
from raspberry_pi.map_camera_inspection import (
    CameraCapture,
    InspectionReport,
    OnlineQwenComponentLabeler,
    PermanentSemanticMap,
    run as run_inspection,
    _panorama_result_message,
)
from raspberry_pi.open_space_panorama import (
    OpenSpaceDirectionSelector,
    PanoramaConfig,
    SceneCoverageTracker,
    directional_clearances,
    open_space_assessment,
    _turn_increment,
    forward_swept_clearance,
    _candidate_in_envelope,
    _bounded_motion,
    _short_reverse,
    _filter_body_returns,
    _LocalHeadingLock,
    _local_drive_angular,
    _local_motion_action,
    run_open_space_panorama,
)
from raspberry_pi.map_camera_inspection import (
    build_argument_parser,
    _build_speed_controller,
    _blocked_path_heading_command,
    _command_for_pose,
    _start_motion_keepalive,
    _first_trajectory_pose,
    _fast_unique_radar_localization,
    _execute_active_localization_turn,
    _execute_localization_turn_escape,
    _inspection_standoff_reached,
    _information_view_for_target,
    _inspection_navigation_command_ttl,
    _inspection_path_for_target,
    _inspection_route_cost,
    _final_heading_timeout_seconds,
    _skip_classified_interior_targets,
    _limit_navigation_linear_request,
    _inspection_target_visible_from,
    _is_turn_motion_stalled,
    _localization_acceptance,
    _localization_motion_consistent,
    _navigation_turn_stall_observed,
    _navigation_linear_stall_observed,
    _nearby_turn_timeout_capture_allowed,
    _navigation_turn_timeout_action,
    _navigation_motion_pose,
    _periodic_map_correction_due,
    _prefer_displaced_retry_choices,
    _blend_navigation_pose,
    _precision_heading_progress,
    _circular_heading_seed,
    _apply_verified_semantic_views,
    _within_capture_reference,
    _navigation_pose_jump_reason,
    _navigation_recovery_candidate_usable,
    _navigation_recovery_candidate_bounded,
    _navigation_recovery_candidates_consistent,
    _best_bounded_recovery_candidate,
    _alternate_inspection_target,
    _collision_reverse_is_safe,
    _execute_collision_reverse_escape,
    _obstacle_safe,
    _navigation_command_ttl,
    _pose_matches_motion_prediction,
    _radar_window_can_finish_without_motion,
    _scan_change_m,
    _stabilize_navigation_turn_pose,
    _strict_radar_localization,
    _spaced_targets,
    _tracking_usable_for_photo_mission,
    _tracking_pose_available,
    _tracking_update,
    _turn_escape_is_safe,
    _turn_is_at_adaptive_max,
    run as run_map_camera_inspection,
)
from raspberry_pi.mapping.scan_matcher import MatchResult, IcpScanMatcher
from raspberry_pi.object_search_mission import (
    ResilientCameraCapture,
    _create_client_with_retry,
    _fuse_tracking_pose,
    _scan_motion_prediction,
    _strict_command_for_pose,
    build_argument_parser as build_search_argument_parser,
    confirm_detections,
)


def _room_image():
    image = np.full((100, 120), 205, dtype=np.uint8)
    image[10:90, 10:110] = 254
    image[10:13, 10:110] = 0
    image[87:90, 10:110] = 0
    image[10:90, 10:13] = 0
    image[10:90, 107:110] = 0
    image[40:58, 50:67] = 0
    return image


class KnownMapTests(unittest.TestCase):
    def _select_local_direction_for_test(self, ranges, support=8, reject_front=False, map_blocked=False):
        selector = OpenSpaceDirectionSelector()
        pose = Pose2D()
        if reject_front:
            selector.reject_direction(pose, 0)
        coverage = SimpleNamespace(points=np.array([[-2., 0.]]), covered=np.array([False]),
            expected_panorama_gain=lambda endpoint, _config: 10000 if abs(endpoint.yaw_rad) > math.radians(35) else 1)
        with patch('raspberry_pi.open_space_panorama._nearest_map_guard_pose', return_value=pose), patch(
            'raspberry_pi.open_space_panorama._robust_sector_clearance',
            side_effect=lambda _scan, angle, **_kw: (ranges(angle), support)
        ), patch('raspberry_pi.open_space_panorama._map_ray_clearance', return_value=2.2):
            scan = LaserScan(np.linspace(-math.pi, math.pi, 360, endpoint=False), np.full(360, 2.2), 0.)
            return selector.choose(scan, pose, SimpleNamespace(line_is_clear=lambda *_a: not map_blocked),
                coverage, PanoramaConfig())

    def test_safe_straight_beats_arbitrarily_large_side_coverage_gain(self):
        choice = self._select_local_direction_for_test(lambda angle: 1.4 if angle == 0 else 2.2)
        self.assertEqual(choice['relative_heading_deg'], 0)
        self.assertEqual(choice['selection_policy'], 'safe_straight')

    def test_blocked_straight_prefers_near_forward_before_large_turn(self):
        choice = self._select_local_direction_for_test(lambda angle: .3 if angle == 0 else 2.2)
        self.assertEqual(abs(choice['relative_heading_deg']), 15)
        self.assertEqual(choice['selection_policy'], 'near_forward')

    def test_wide_escape_only_when_forward_candidates_unavailable(self):
        choice = self._select_local_direction_for_test(lambda angle: .3 if abs(angle) <= 35 else 2.2)
        self.assertGreater(abs(choice['relative_heading_deg']), 35)
        self.assertEqual(choice['selection_policy'], 'wide_escape')

    def test_recently_rejected_direction_not_forced_as_safe_straight(self):
        choice = self._select_local_direction_for_test(lambda _angle: 2.2, reject_front=True)
        self.assertEqual(abs(choice['relative_heading_deg']), 35)
        self.assertEqual(choice['selection_policy'], 'forward_escape')

    def test_missing_lidar_support_does_not_enable_safe_straight_priority(self):
        choice = self._select_local_direction_for_test(lambda _angle: 2.2, support=0)
        self.assertEqual(choice['selection_policy'], 'wide_escape')

    def test_map_obstruction_still_rejects_all_directions(self):
        self.assertIsNone(self._select_local_direction_for_test(lambda _angle: 2.2, map_blocked=True))

    def test_forward_pulse_cannot_exceed_obstacle_prediction_horizon(self):
        for speed in (240, 330, 420, 450):
            for requested_ttl in (200, 350, 500, 700):
                actual = _navigation_command_ttl(requested_ttl, speed)
                self.assertLessEqual(actual, requested_ttl)
                self.assertLessEqual(speed / 1000 * actual / 1000,
                                     speed / 1000 * 0.35 + 1e-9)
        self.assertEqual(180, _navigation_command_ttl(180, 0))

    def test_forward_ttl_uses_lidar_clearance_instead_of_fixed_cap(self):
        self.assertEqual(
            700,
            _navigation_command_ttl(700, 450, 1.00, 0.42),
        )
        near_ttl = _navigation_command_ttl(700, 450, 0.80, 0.42)
        self.assertGreaterEqual(near_ttl, 350)
        self.assertLess(near_ttl, 700)
        self.assertEqual(
            200,
            _navigation_command_ttl(700, 450, 0.65, 0.42),
        )

    def test_mixed_steering_does_not_truncate_forward_start_ttl(self):
        args = build_argument_parser().parse_args([])
        controller = _build_speed_controller(args, navigation=True)
        mixed = MotionCommand(420, 220, "advancing", "path correction")
        pure_turn = MotionCommand(0, 2400, "turning", "align")
        self.assertEqual(
            700,
            _inspection_navigation_command_ttl(
                controller, mixed, 700, 1.50, 0.42
            ),
        )
        self.assertEqual(
            220,
            _inspection_navigation_command_ttl(
                controller, pure_turn, 700, 1.50, 0.42
            ),
        )

    def test_dynamic_route_cost_penalizes_wrong_side_arrival(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        path = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
        aligned = InspectionTarget(
            "aligned", "perimeter_wall", (2.0, 0.0),
            (1.0, 0.0), 0.0, 10,
        )
        reversed_view = InspectionTarget(
            "reversed", "perimeter_wall", (0.0, 0.0),
            (1.0, 0.0), math.pi, 10,
        )
        aligned_cost = _inspection_route_cost(pose, aligned, path, 0.35)
        reversed_cost = _inspection_route_cost(
            pose, reversed_view, path, 0.35
        )
        self.assertLess(
            math.degrees(aligned_cost["arrival_turn_rad"]), 1.0
        )
        self.assertGreater(
            math.degrees(reversed_cost["arrival_turn_rad"]), 179.0
        )
        self.assertGreater(reversed_cost["score"], aligned_cost["score"] + 2.0)

    def test_dynamic_path_adds_camera_aligned_staging_approach(self):
        image = np.full((100, 100), 254, dtype=np.uint8)
        image[[0, -1], :] = 0
        image[:, [0, -1]] = 0
        known = KnownOccupancyMap(
            image, 0.05, -2.5, -2.5, clearance_m=0.20
        )
        pose = Pose2D(-0.5, 0.0, 0.0)
        target = InspectionTarget(
            "wall_test", "perimeter_wall", (-2.0, 0.0),
            (0.0, 0.0), math.pi, 10,
        )
        proposal = _inspection_path_for_target(
            known, pose, target, 0.35
        )
        self.assertIsNotNone(proposal)
        self.assertTrue(proposal["staged_camera_approach"])
        self.assertLess(
            math.degrees(proposal["metrics"]["arrival_turn_rad"]), 45.0
        )

    def test_final_heading_timeout_scales_with_turn_angle(self):
        self.assertEqual(8.0, _final_heading_timeout_seconds(8.0, 0.0))
        self.assertEqual(
            8.0,
            _final_heading_timeout_seconds(8.0, math.radians(45.0)),
        )
        self.assertGreaterEqual(
            _final_heading_timeout_seconds(8.0, math.pi), 18.0
        )

    def test_deferred_qwen_mode_is_explicit(self):
        args = build_argument_parser().parse_args(["--defer-online-qwen"])
        self.assertTrue(args.defer_online_qwen)

    def test_open_space_panorama_mode_defaults_to_low_rate_correction(self):
        args = build_argument_parser().parse_args(["--open-space-panorama"])
        config = PanoramaConfig.from_args(args)
        config.validate()
        self.assertTrue(args.open_space_panorama)
        self.assertGreaterEqual(config.map_correction_distance_m, 1.0)
        self.assertEqual(config.map_correction_confirmations, 2)

    def test_panorama_camera_test_generates_no_fixed_targets_or_semantic_entities(self):
        camera = SimpleNamespace(capture=lambda _path: {
            "backend": "simulated",
            "brightness": 100.0,
            "contrast": 30.0,
        })
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--open-space-panorama",
                "--camera-test-only",
                "--output", directory,
                "--map", "maps/ble_scene_17.pgm",
                "--map-yaml", "maps/ble_scene_17.yaml",
            ])
            with patch.object(
                KnownOccupancyMap,
                "inspection_targets",
                side_effect=AssertionError("fixed targets must not be generated"),
            ), patch(
                "raspberry_pi.map_camera_inspection.PermanentSemanticMap",
                side_effect=AssertionError("permanent semantic map must not be written"),
            ):
                result = run_inspection(args, camera=camera)
                self.assertTrue(result.is_dir())

    def test_panorama_chassis_corridor_detects_side_front_table_leg(self):
        angles = np.radians((-80.0, -40.0, 0.0, 40.0, 80.0))
        scan = LaserScan(angles, np.array((2.0, 2.0, 3.0, 2.0, 0.22)), 0.0)
        self.assertLess(forward_swept_clearance(scan), 0.10)

    def test_panorama_sensor_offset_includes_return_behind_sensor_ahead_of_car(self):
        scan = LaserScan(np.radians((110.0, 0.0)), np.array((0.20, 3.0)), 0.0)
        self.assertLess(forward_swept_clearance(scan, offset_x_m=0.14), 0.10)

    def test_panorama_missing_scan_is_not_free_space(self):
        scan = LaserScan(np.array((0.0,)), np.array((np.nan,)), 0.0)
        self.assertEqual(forward_swept_clearance(scan), 0.0)

    def test_panorama_body_returns_are_removed_only_inside_rear_inner_disk(self):
        original = LaserScan(np.radians((180, -104, -90, 80, -90)),
                             np.array((0.082, 0.121, 0.121, 0.10, 0.25)), 3.0)
        filtered = _filter_body_returns(original, 0.19, offset_x_m=0.14)
        self.assertEqual(filtered.body_return_count, 3)
        self.assertTrue(np.isnan(filtered.distances_m[:3]).all())
        np.testing.assert_array_equal(filtered.distances_m[3:], original.distances_m[3:])
        self.assertTrue(np.isfinite(original.distances_m).all())
        self.assertEqual(filtered.timestamp_s, 3.0)
        self.assertEqual(len(filtered.points()), 1)

    def test_panorama_masked_scan_is_unknown_not_free_and_filter_can_be_disabled(self):
        original = LaserScan(np.radians((180, -104)), np.array((0.082, 0.121)), 0.0)
        self.assertIs(_filter_body_returns(original, 0), original)
        filtered = _filter_body_returns(original, 0.19, offset_x_m=0.14)
        self.assertEqual(forward_swept_clearance(filtered, offset_x_m=0.14), 0.0)
        self.assertEqual(forward_swept_clearance(filtered, offset_x_m=0.14, reverse=True), 0.0)

    def test_panorama_external_thin_leg_and_glass_guard_survive_body_filter(self):
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        distances = np.full(180, 3.0)
        distances[175] = 0.22  # +170 degrees, not ahead; add a side-front leg below.
        distances[130] = 0.22  # +80 degrees ahead of the sensor: must never be masked.
        scan = _filter_body_returns(LaserScan(angles, distances, 0.0), 0.19, offset_x_m=0.14)
        self.assertLess(forward_swept_clearance(scan, offset_x_m=0.14), 0.20)
        known = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        leaked = _filter_body_returns(LaserScan(angles, np.full(180, 8.0), 0.0), 0.19, offset_x_m=0.14)
        values = directional_clearances(leaked, known, Pose2D(-0.935, 0.625, math.radians(-126)))
        self.assertTrue(any(v["effective_m"] < v["lidar_m"] for v in values.values()))
        # Outside the real lateral body strip must survive even if inside
        # the circular safety envelope (a rectangular chassis is not a disk).
        sensor_point = np.array((-0.14, 0.18))
        lateral = LaserScan(np.array((math.atan2(sensor_point[1], sensor_point[0]),)),
                            np.array((float(np.linalg.norm(sensor_point)),)), 0.0)
        kept = _filter_body_returns(lateral, 0.19, offset_x_m=0.14)
        self.assertTrue(np.isfinite(kept.distances_m).all())

    def test_panorama_guard_rejects_high_quality_symmetric_yaw(self):
        candidate = SimpleNamespace(rmse_m=0.05, inlier_ratio=0.95, pose=Pose2D(0.1, 0.0, math.pi))
        self.assertFalse(_candidate_in_envelope(candidate, Pose2D(), PanoramaConfig()))

    def test_panorama_accepts_real_match_result_without_translation_field(self):
        match = MatchResult(Pose2D(0.03, 0.04, math.radians(12)), 0.01, 100, True, 0.99)
        self.assertFalse(hasattr(match, "translation_m"))
        localizer = SimpleNamespace(relative_motion=lambda *_a: match)
        pose, returned, accepted = _bounded_motion(localizer, object(), object(), Pose2D())
        self.assertTrue(accepted)
        self.assertIs(returned, match)
        self.assertAlmostEqual(pose.x_m, 0.03)
        self.assertAlmostEqual(pose.y_m, 0.04)
        self.assertAlmostEqual(localizer._panorama_motion_diagnostic["translation_m"], 0.05)

    def test_panorama_real_icp_output_is_accepted(self):
        rng = np.random.default_rng(21)
        points = rng.uniform(-2, 2, (160, 2))
        motion = Pose2D(0.03, -0.02, math.radians(3))
        current = motion.inverse().transform_points(points)
        matcher = IcpScanMatcher()
        localizer = SimpleNamespace(relative_motion=matcher.match)
        pose, _match, accepted = _bounded_motion(localizer, points, current, Pose2D())
        self.assertTrue(accepted)
        self.assertAlmostEqual(pose.x_m, motion.x_m, places=3)
        self.assertAlmostEqual(pose.yaw_rad, motion.yaw_rad, places=3)

    def test_panorama_real_scan_localizer_pipeline_tracks_stationary_and_moving_scans(self):
        localizer = ScanMapLocalizer(None, lidar_offset_x_m=0.14)
        rng = np.random.default_rng(22)
        points = rng.uniform(-2, 2, (180, 2))

        def scan(sensor_points):
            return LaserScan(np.arctan2(sensor_points[:, 1], sensor_points[:, 0]),
                             np.linalg.norm(sensor_points, axis=1), 0.0)

        baseline = scan(points)
        stationary_pose, match, accepted = _bounded_motion(localizer, baseline, baseline, Pose2D())
        self.assertIsInstance(match, MatchResult)
        self.assertTrue(accepted)
        self.assertLess(math.hypot(stationary_pose.x_m, stationary_pose.y_m), 1e-6)
        motion = Pose2D(0.04, 0, math.radians(3))
        current = scan(motion.inverse().transform_points(points + localizer.lidar_offset) - localizer.lidar_offset)
        pose, _match, accepted = _bounded_motion(localizer, baseline, current, stationary_pose)
        self.assertTrue(accepted)
        self.assertAlmostEqual(pose.x_m, motion.x_m, places=3)
        self.assertAlmostEqual(pose.yaw_rad, motion.yaw_rad, places=3)

    def test_local_heading_uses_only_accepted_relative_yaw_not_map_correction(self):
        lock = _LocalHeadingLock(math.radians(70))
        match = MatchResult(Pose2D(.1, 0, math.radians(20)), .01, 100, True, inlier_ratio=.95)
        localizer = SimpleNamespace(relative_motion=lambda *_a: match, _panorama_local_heading_lock=lock)
        _bounded_motion(localizer, object(), object(), Pose2D(0, 0, math.radians(-130)))
        self.assertAlmostEqual(math.degrees(lock.error_rad), 50)
        self.assertAlmostEqual(lock.distance_m, .1)
        localizer.relative_motion = lambda *_a: MatchResult(Pose2D(0, 0, math.radians(90)), .01, 100, True, inlier_ratio=.95)
        _bounded_motion(localizer, object(), object(), Pose2D())
        self.assertAlmostEqual(math.degrees(lock.error_rad), 50)

    def test_local_alignment_hysteresis_allows_safe_drive_with_small_error(self):
        lock = _LocalHeadingLock(math.radians(30))
        self.assertFalse(lock.needs_alignment())
        self.assertTrue(lock.needs_alignment(front_blocked=True))
        lock.error_rad = math.radians(40)
        self.assertTrue(lock.needs_alignment())
        lock.error_rad = math.radians(30)
        self.assertTrue(lock.needs_alignment())
        lock.error_rad = math.radians(24)
        self.assertFalse(lock.needs_alignment())

    def test_local_heading_does_not_expire_while_stationary_turning(self):
        lock = _LocalHeadingLock(1)
        self.assertFalse(lock.expired(1000))
        lock.mark_drive(1)
        self.assertFalse(lock.expired(1000))
        lock.observe(Pose2D(.16, 0, 0))
        self.assertTrue(lock.expired(5))
        lock = _LocalHeadingLock(0)
        lock.observe(Pose2D(.66, 0, 0))
        self.assertTrue(lock.expired(1))

    def test_local_drive_steering_is_small_and_follows_lidar_clearance(self):
        self.assertGreater(_local_drive_angular(0, 2, .5), 0)
        self.assertLess(_local_drive_angular(0, .5, 2), 0)
        self.assertLessEqual(abs(_local_drive_angular(math.pi, 3, .5)), 450)
        self.assertEqual(_local_drive_angular(math.radians(25), 2, 2),
                         _local_drive_angular(math.radians(90), 2, 2))

    def test_clear_front_never_requires_in_place_heading_alignment(self):
        for yaw_deg in (-180, -90, -40, 15, 40, 90, 180):
            self.assertEqual(_local_motion_action(1.2, .72, .5, math.radians(yaw_deg), 6), 'drive')

    def test_blocked_front_allows_only_bounded_lidar_safe_turn(self):
        self.assertEqual(_local_motion_action(.6, .72, .5, math.radians(70), 0), 'turn')
        self.assertEqual(_local_motion_action(.6, .72, .5, math.radians(70), 6), 'blocked')
        self.assertEqual(_local_motion_action(.6, .72, .2, math.radians(70), 0), 'blocked')
        self.assertEqual(_local_motion_action(.72, .72, .5, 0, 0), 'blocked')

    def test_local_turn_uses_short_pulses_and_integrates_stopped_tail(self):
        baseline, rotated, settled = object(), object(), object()
        sent = []
        lock = _LocalHeadingLock(math.radians(40))
        def relative(_before, current):
            return MatchResult(Pose2D(0, 0, math.radians(16 if current is rotated else 4)), .01, 100, True, inlier_ratio=.95)
        client = SimpleNamespace(stop=lambda: sent.append('stop'),
            set_twist=lambda _linear, _angular, ttl_ms: sent.append(ttl_ms))
        _scan, pose, measured, ok = _turn_increment(client, iter((rotated, settled)),
            SimpleNamespace(relative_motion=relative, _panorama_local_heading_lock=lock),
            baseline, Pose2D(), math.radians(20), PanoramaConfig(), lambda *_a: None,
            lambda *_a: True, 'LOCAL_HEADING_TURN')
        self.assertTrue(ok)
        self.assertGreaterEqual(next(v for v in sent if isinstance(v, int)), 120)
        self.assertLessEqual(next(v for v in sent if isinstance(v, int)), 350)
        self.assertAlmostEqual(math.degrees(measured), 20)
        self.assertAlmostEqual(math.degrees(pose.yaw_rad), 20)
        self.assertAlmostEqual(math.degrees(lock.error_rad), 20)
        self.assertIs(_scan, settled)

    def test_local_drive_ignores_absolute_yaw_mismatch_with_safe_front(self):
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), 0)
        sent = []
        values = {name: {'effective_m': 2.5, 'lidar_m': 2.5, 'map_m': 2.5, 'support': 8}
            for name in ('front', 'left', 'rear', 'right', 'front_left', 'front_right')}
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args(['--open-space-panorama'])
            with patch('raspberry_pi.open_space_panorama.directional_clearances', return_value=values), patch(
                'raspberry_pi.open_space_panorama.open_space_assessment',
                return_value={'is_open': False, 'minimum_clearance_m': 2.5, 'open_direction_count': 4}
            ), patch('raspberry_pi.open_space_panorama.forward_swept_clearance', return_value=2.5), patch(
                'raspberry_pi.open_space_panorama._turn_increment', side_effect=AssertionError('must drive, not align map yaw')
            ), patch.object(OpenSpaceDirectionSelector, 'choose', return_value={
                'world_heading_rad': math.pi, 'relative_heading_deg': 30., 'clearance_m': 2.5, 'expected_new_samples': 0}
            ), patch.object(KnownOccupancyMap, 'line_is_clear', return_value=True), patch(
                'raspberry_pi.open_space_panorama._render_live_map'
            ):
                run_open_space_panorama(args=args, known_map=known,
                    localizer=SimpleNamespace(relative_motion=lambda *_a: MatchResult(Pose2D(.02, 0, 0), .01, 100, True, inlier_ratio=.95)),
                    initial_pose=Pose2D(), initial_scan=scan, scan_iterator=iter((scan,) * 10),
                    client=SimpleNamespace(stop=lambda: None, set_twist=lambda linear, angular, ttl_ms: sent.append((linear, angular, ttl_ms))),
                    camera=SimpleNamespace(), report=InspectionReport(directory, args.map),
                    online_qwen=SimpleNamespace(url=''), should_stop=lambda: bool(sent))
        self.assertEqual(len(sent), 1)
        self.assertGreater(sent[0][0], 0)
        self.assertLessEqual(abs(sent[0][1]), 450)
        self.assertLessEqual(sent[0][2], 500)

    def test_panorama_relaxed_yaw_limit_accepts_field_turn_and_boundary(self):
        for yaw_deg in (-48.0946, 48.0946, -60.0, 60.0):
            with self.subTest(yaw_deg=yaw_deg):
                match = MatchResult(Pose2D(.215085, 0, math.radians(yaw_deg)), .01, 94, True, .99)
                localizer = SimpleNamespace(relative_motion=lambda *_a: match)
                pose, _match, accepted = _bounded_motion(localizer, object(), object(), Pose2D())
                self.assertTrue(accepted)
                self.assertAlmostEqual(pose.yaw_rad, math.radians(yaw_deg))
                self.assertIsNone(localizer._panorama_pending_motion)

    def test_panorama_real_transform_bounds_remain_enforced(self):
        for transform, reason in (
            (Pose2D(0.4, 0, 0), "translation_exceeds_envelope"),
            (Pose2D(0, 0, math.radians(60.1)), "yaw_exceeds_envelope"),
            (Pose2D(0, 0, math.radians(-60.1)), "yaw_exceeds_envelope"),
            (Pose2D(math.nan, 0, 0), "non_finite_transform"),
            (Pose2D(0, 0, math.nan), "non_finite_transform"),
        ):
            with self.subTest(reason=reason):
                match = MatchResult(transform, 0.01, 100, True, 0.99)
                localizer = SimpleNamespace(relative_motion=lambda *_a: match)
                pose, _match, accepted = _bounded_motion(localizer, object(), object(), Pose2D())
                self.assertFalse(accepted)
                self.assertEqual(pose, Pose2D())
                self.assertEqual(localizer._panorama_motion_diagnostic["rejection_reason"], reason)

    def test_panorama_zero_photos_failure_is_not_reported_as_complete(self):
        result = {"station_count": 0, "photo_count": 0, "coverage": {"coverage_ratio": 0.0},
                  "motion_complete": False, "finish_reason": "odometry_unavailable_after_stopped_recovery"}
        message = _panorama_result_message(result)
        self.assertIn("任务未完成", message)
        self.assertIn("雷达帧间里程计停车恢复失败", message)
        self.assertNotIn("巡检完成：", message)
        result["motion_complete"] = True
        self.assertIn("巡检完成：", _panorama_result_message(result))

    def test_panorama_first_photo_uses_real_scan_matching_not_mock_distance(self):
        known_map = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), 0.0)
        localizer = ScanMapLocalizer(known_map)
        captured = []
        values = {name: {"effective_m": 2.5, "lidar_m": 2.5, "map_m": 2.5, "support": 8}
                  for name in ("front", "left", "rear", "right", "front_left", "front_right")}
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args(["--open-space-panorama"])
            with patch("raspberry_pi.open_space_panorama.directional_clearances", return_value=values), patch(
                "raspberry_pi.open_space_panorama._render_live_map"
            ), patch("raspberry_pi.open_space_panorama.time.sleep"):
                result = run_open_space_panorama(
                    args=args, known_map=known_map, localizer=localizer,
                    initial_pose=Pose2D(-0.935, 0.625, 0), initial_scan=scan,
                    scan_iterator=iter([scan] * 50),
                    client=SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: self.fail("stop requested after first photo")),
                    camera=SimpleNamespace(capture=lambda path: (captured.append(path.name), {"backend": "fake"})[1]),
                    report=InspectionReport(directory, args.map), online_qwen=SimpleNamespace(url=""),
                    should_stop=lambda: bool(captured),
                )
        self.assertEqual(result["photo_count"], 1)
        self.assertEqual(result["finish_reason"], "requested_stop")
        self.assertTrue(localizer._panorama_motion_diagnostic["accepted"])

    def test_panorama_body_occlusion_status_is_strict_json_and_capture_still_starts(self):
        known_map = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        distances = np.where(np.cos(angles) <= 1e-9, 0.082, 2.5)
        scan = LaserScan(angles, distances, 0)
        photos = []
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args(["--open-space-panorama", "--panorama-self-return-radius", "0.19"])
            report = InspectionReport(directory, args.map)
            with patch("raspberry_pi.open_space_panorama._render_live_map"), patch("raspberry_pi.open_space_panorama.time.sleep"):
                summary = run_open_space_panorama(
                    args=args, known_map=known_map, localizer=ScanMapLocalizer(known_map),
                    initial_pose=Pose2D(-1.1668, 0.3546, math.radians(-64.02)), initial_scan=scan,
                    scan_iterator=iter([scan] * 50),
                    client=SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: self.fail("stop after first photo")),
                    camera=SimpleNamespace(capture=lambda p: (photos.append(p.name), {"backend": "fake"})[1]),
                    report=report, online_qwen=SimpleNamespace(url=""), should_stop=lambda: bool(photos),
                )
            status = json.loads((report.directory / "live_pose.json").read_text(encoding="utf-8"),
                                parse_constant=lambda value: self.fail("invalid JSON constant: " + value))
            self.assertIsNone(status["directional_clearance_details"]["rear"]["lidar_m"])
            self.assertGreater(status["body_return_filter"]["excluded_point_count"], 0)
            self.assertEqual(summary["photo_count"], 1)

    def test_panorama_double_dropout_preserves_anchor_across_later_control_frame(self):
        baseline, other, recovered = object(), object(), object()
        refs = []
        matches = iter((MatchResult(Pose2D(), math.inf, 0, False), MatchResult(Pose2D(), math.inf, 0, False), MatchResult(Pose2D(0.05, 0, math.radians(18)), 0.01, 100, True, inlier_ratio=.95)))
        localizer = SimpleNamespace(relative_motion=lambda ref, _s: (refs.append(ref), next(matches))[1])
        pose = Pose2D()
        for reference, current in ((baseline, other), (baseline, other), (other, recovered)):
            pose, _match, _ok = _bounded_motion(localizer, reference, current, pose)
        self.assertEqual(refs, [baseline] * 3)
        self.assertAlmostEqual(pose.yaw_rad, math.radians(18))
        self.assertAlmostEqual(pose.x_m, 0.05)
        self.assertIsNone(localizer._panorama_pending_motion)

    def test_panorama_short_reverse_waits_through_command_latency(self):
        elapsed, events, frames = [0.0], [], []
        scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), 0.0)
        known_map = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        args = build_argument_parser().parse_args(["--open-space-panorama", "--lidar-offset-x", "0"])

        def scans():
            for index in range(8):
                elapsed[0] += 0.18
                frames.append(index)
                yield scan

        def relative(*_args):
            distance = 0.0 if len(frames) == 1 else 0.04
            return MatchResult(Pose2D(-distance, 0, 0), 0.01, 100, True, inlier_ratio=.95)

        with patch("raspberry_pi.open_space_panorama.time.monotonic", side_effect=lambda: elapsed[0]), patch.object(KnownOccupancyMap, "line_is_clear", return_value=True):
            _scan, _pose, moved = _short_reverse(
                SimpleNamespace(stop=lambda: events.append("stop"), set_twist=lambda *a, **k: events.append((a, k))),
                scans(), SimpleNamespace(relative_motion=relative), scan, Pose2D(-0.935, 0.625, 0), PanoramaConfig(), known_map, args, lambda *_a: None, lambda: False,
            )
        self.assertGreaterEqual(len(frames), 3)
        self.assertGreaterEqual(moved, 0.08)
        self.assertEqual(len([e for e in events if isinstance(e, tuple)]), 1)
        self.assertEqual(events[-1], "stop")

    def test_panorama_occluded_reverse_refreshes_scan_without_driving(self):
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        distances = np.full(180, 2.5)
        distances[np.cos(angles) < 0] = np.nan
        baseline = LaserScan(angles, distances, 0)
        fresh = LaserScan(angles, distances, 1)
        known = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        args = build_argument_parser().parse_args(["--open-space-panorama", "--lidar-offset-x", "0"])
        returned, _pose, moved = _short_reverse(
            SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: self.fail("rear occluded")),
            iter((fresh,)), SimpleNamespace(relative_motion=lambda *_a: MatchResult(Pose2D(), 0.01, 100, True, inlier_ratio=.95)),
            baseline, Pose2D(-1.1668, 0.3546, math.radians(-64)), PanoramaConfig(), known, args,
            lambda *_a: None, lambda: False,
        )
        self.assertIs(returned, fresh)
        self.assertEqual(moved, 0)

    def test_panorama_partial_attempts_keep_images_and_infer_only_after_stop(self):
        known_map = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), 0.0)
        sent, captured, inferred = [], [], []
        client = SimpleNamespace(stop=lambda: sent.append("stop"), set_twist=lambda *_a, **_k: sent.append("drive"))

        def capture(path):
            captured.append(path.name)
            Image.new("RGB", (8, 8)).save(path)
            second = path.stem + "_camera_2.jpg"
            Image.new("RGB", (8, 8)).save(path.with_name(second))
            return {"backend": "fake", "camera_2_image": second}

        def infer(record, *_paths):
            self.assertEqual(sent[-1], "stop")
            self.assertEqual(len(captured), 3)
            inferred.append(record["target_id"])
            return {"enabled": True, "promoted": False, "landmarks": []}

        def bounded(_localizer, _reference, _current, pose):
            return pose.compose(Pose2D(0.06, 0.0, 0.0)), MatchResult(Pose2D(0.06, 0.0, 0.0), 0.01, 100, True, inlier_ratio=.95), True

        values = {name: {"effective_m": 2.5, "lidar_m": 2.5, "map_m": 2.5, "support": 8} for name in ("front", "left", "rear", "right", "front_left", "front_right")}
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args(["--open-space-panorama", "--max-runtime-min", "1"])
            report = InspectionReport(directory, args.map)
            with patch("raspberry_pi.open_space_panorama._bounded_motion", side_effect=bounded), patch(
                'raspberry_pi.open_space_panorama._advance_assessment', return_value={
                    'can_advance': True, 'effective_m': 2.5, 'stop_m': .72, 'horizon_m': .2,
                    'lidar_m': 2.5, 'map_m': 2.5, 'swept_m': 2.5}
            ), patch(
                "raspberry_pi.open_space_panorama._turn_increment", side_effect=lambda _c, _i, _l, s, p, *_rest: (s, p, 0.0, False)
            ), patch("raspberry_pi.open_space_panorama.directional_clearances", return_value=values), patch(
                "raspberry_pi.open_space_panorama.forward_swept_clearance", return_value=3.0
            ), patch.object(OpenSpaceDirectionSelector, "choose", return_value={"world_heading_rad": 0.0, "relative_heading_deg": 0.0, "clearance_m": 2.5, "expected_new_samples": 0}
            ), patch.object(KnownOccupancyMap, "line_is_clear", return_value=True), patch(
                "raspberry_pi.open_space_panorama._render_live_map"
            ), patch("raspberry_pi.open_space_panorama.time.sleep"):
                result = run_open_space_panorama(
                    args=args, known_map=known_map, localizer=SimpleNamespace(localize_near_position=lambda *_a, **_k: None),
                    initial_pose=Pose2D(-0.935, 0.625, 0.0), initial_scan=scan,
                    scan_iterator=iter([scan] * 300), client=client, camera=SimpleNamespace(capture=capture), report=report,
                    online_qwen=SimpleNamespace(url="http://example.test", analyze_panorama=infer), should_stop=lambda: len(captured) >= 3,
                )
            self.assertEqual(result["station_count"], 0)
            self.assertEqual(result["station_attempt_count"], 3)
            self.assertEqual(len(set(captured)), 3)
            self.assertEqual(len(inferred), 3)
            self.assertTrue(all((report.directory / name).exists() for name in captured))

    def test_panorama_front_drive_stop_does_not_discard_safe_escape_turn(self):
        known_map = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), time.monotonic())
        finished = []

        def fresh_scans():
            for _ in range(50):
                yield LaserScan(scan.angles_rad, scan.distances_m, time.monotonic())

        def turn(_client, _iterator, _localizer, current, pose, *_rest):
            finished.append(True)
            return current, pose, math.radians(45), True

        values = {name: {"effective_m": (0.60 if name == "front" else 2.5), "lidar_m": 2.5, "map_m": 2.5, "support": 8} for name in ("front", "left", "rear", "right", "front_left", "front_right")}
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args(["--open-space-panorama"])
            with patch("raspberry_pi.open_space_panorama.directional_clearances", return_value=values), patch(
                "raspberry_pi.open_space_panorama.open_space_assessment", return_value={"is_open": False, "minimum_clearance_m": 0.6, "open_direction_count": 3}
            ), patch("raspberry_pi.open_space_panorama._swept_clearance_details",
                     return_value={'clearance_m': .6, 'nearest_point': None}), patch(
                "raspberry_pi.open_space_panorama._reverse_escape_assessment",
                return_value={'can_reverse': False, 'reason': 'rear_support_missing'}
            ), patch(
                "raspberry_pi.open_space_panorama._turn_increment", side_effect=turn
            ), patch.object(OpenSpaceDirectionSelector, "choose", return_value={"world_heading_rad": math.radians(45), "relative_heading_deg": 45, "clearance_m": 2.5, "expected_new_samples": 0}), patch(
                "raspberry_pi.open_space_panorama._render_live_map"
            ):
                run_open_space_panorama(
                    args=args, known_map=known_map, localizer=SimpleNamespace(relative_motion=lambda *_:
                        MatchResult(Pose2D(), .01, 100, True, inlier_ratio=.95)), initial_pose=Pose2D(), initial_scan=scan,
                    scan_iterator=fresh_scans(), camera=SimpleNamespace(), report=InspectionReport(directory, args.map),
                    client=SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: self.fail("front blockage must prohibit translation")),
                    online_qwen=SimpleNamespace(url=""), should_stop=lambda: bool(finished),
                )
        self.assertEqual(len(finished), 1)

    def test_panorama_repeated_heading_changes_do_not_hide_translation_stall(self):
        elapsed = [0.0]
        known = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
        scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), 0)

        def scans():
            for _ in range(100):
                elapsed[0] += 1.0
                yield scan

        def heading(_s, pose, *_a):
            return {"world_heading_rad": normalize_angle(pose.yaw_rad + math.radians(45)),
                    "relative_heading_deg": 45, "clearance_m": 2.5, "expected_new_samples": 0}

        values = {name: {"effective_m": 2.5, "lidar_m": 2.5, "map_m": 2.5, "support": 8}
                  for name in ("front", "left", "rear", "right", "front_left", "front_right")}
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args(["--open-space-panorama"])
            with patch("raspberry_pi.open_space_panorama.time.monotonic", side_effect=lambda: elapsed[0]), patch(
                "raspberry_pi.open_space_panorama.directional_clearances", return_value=values
            ), patch("raspberry_pi.open_space_panorama.open_space_assessment", return_value={
                "is_open": False, "minimum_clearance_m": 2.5, "open_direction_count": 3
            }), patch.object(OpenSpaceDirectionSelector, "choose", side_effect=heading), patch(
                "raspberry_pi.open_space_panorama._render_live_map"
            ):
                result = run_open_space_panorama(
                    args=args, known_map=known,
                    localizer=SimpleNamespace(localize_near_position=lambda *_a, **_kw: None,
                        relative_motion=lambda *_a: MatchResult(
                        Pose2D(0, 0, math.radians(10)), 0.01, 100, True, inlier_ratio=.95)),
                    initial_pose=Pose2D(), initial_scan=scan, scan_iterator=scans(),
                    client=SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: None),
                    camera=SimpleNamespace(), report=InspectionReport(directory, args.map),
                    online_qwen=SimpleNamespace(url=""), should_stop=lambda: False,
                )
        self.assertIn(result["finish_reason"], ('no_translation_progress', 'actuator_no_response'))
        self.assertFalse(result["motion_complete"])
        # Intentional stopped pose checks are excluded from the driving
        # watchdog; repeated turning itself must still terminate.
        self.assertLess(elapsed[0], 90)

    def test_open_space_direction_selection_never_calls_global_path_planner(self):
        known_map = KnownOccupancyMap.load(
            "maps/ble_scene_17.pgm",
            "maps/ble_scene_17.yaml",
            clearance_m=0.30,
        )
        coverage = SceneCoverageTracker(known_map)
        pose = Pose2D(-0.935, 0.625, math.radians(-126.0))
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        scan = LaserScan(angles, np.full(len(angles), 2.5), 0.0)
        known_map.path = lambda *_args, **_kwargs: self.fail(
            "open-space mode must not call A* path planning"
        )
        selected = OpenSpaceDirectionSelector().choose(
            scan, pose, known_map, coverage, PanoramaConfig()
        )
        self.assertIsNotNone(selected)

    def test_glass_leak_cannot_override_permanent_map_clearance(self):
        known_map = KnownOccupancyMap.load(
            "maps/ble_scene_17.pgm",
            "maps/ble_scene_17.yaml",
            clearance_m=0.30,
        )
        pose = Pose2D(-0.935, 0.625, math.radians(-126.0))
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        scan = LaserScan(angles, np.full(len(angles), 8.0), 0.0)
        clearances = directional_clearances(scan, known_map, pose)
        self.assertTrue(any(
            item["effective_m"] < item["lidar_m"]
            for item in clearances.values()
        ))

    def test_open_space_assessment_requires_multiple_clear_directions(self):
        config = PanoramaConfig()
        clearances = {
            name: {"effective_m": value}
            for name, value in {
                "front": 1.0,
                "left": 1.0,
                "rear": 0.50,
                "right": 0.60,
            }.items()
        }
        result = open_space_assessment(clearances, config)
        self.assertFalse(result["is_open"])

    def test_panorama_invalid_turn_increment_stops_and_recovers_same_reference(self):
        sent = []
        client = SimpleNamespace(
            stop=lambda: sent.append("stop"),
            set_twist=lambda linear, angular, ttl_ms: sent.append(
                (linear, angular, ttl_ms)
            ),
            allow_twist_to_expire=lambda: None,
        )
        matches = iter((
            MatchResult(Pose2D(), math.inf, 0, False),
            MatchResult(Pose2D(0.0, 0.0, math.radians(30.0)), 0.01, 100, True, inlier_ratio=.95),
            MatchResult(Pose2D(), .01, 100, True, inlier_ratio=.95),
        ))
        references = []

        def relative_motion(reference, current):
            references.append(reference)
            return next(matches)

        baseline = object()
        _scan, pose, measured, success = _turn_increment(
            client,
            iter((object(), object(), object())),
            SimpleNamespace(relative_motion=relative_motion),
            baseline,
            Pose2D(),
            math.radians(30.0),
            PanoramaConfig(),
            lambda *_args: None,
        )
        self.assertTrue(success)
        self.assertAlmostEqual(measured, math.radians(30.0))
        self.assertEqual(references[:2], [baseline, baseline])
        self.assertEqual(len([item for item in sent if isinstance(item, tuple)]), 1)
        self.assertAlmostEqual(pose.yaw_rad, math.radians(30.0))

    def test_panorama_transient_close_return_waits_stopped_then_resumes_turn(self):
        blocked, still_blocked, clear, rotated = object(), object(), object(), object()
        events = []
        def relative(_reference, current):
            return MatchResult(Pose2D(0, 0, math.radians(30) if current is rotated else 0), 0.01, 100, True, inlier_ratio=.95)
        def safe(scan, _pose):
            if scan in (blocked, still_blocked):
                self.assertNotIn("turn", events)
                return False
            return True
        _scan, _pose, measured, success = _turn_increment(
            SimpleNamespace(stop=lambda: events.append("stop"), set_twist=lambda *_a, **_k: events.append("turn")),
            iter((still_blocked, clear, rotated, clear)), SimpleNamespace(relative_motion=relative),
            blocked, Pose2D(), math.radians(30), PanoramaConfig(), lambda *_a: None, safe,
        )
        self.assertTrue(success)
        self.assertAlmostEqual(measured, math.radians(30))
        self.assertEqual(events.count("turn"), 1)

    def test_panorama_persistent_close_return_never_energizes_turn(self):
        scan = object()
        _scan, _pose, _yaw, success = _turn_increment(
            SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: self.fail("persistent obstacle")),
            iter((scan,) * 3), SimpleNamespace(relative_motion=lambda *_a: MatchResult(Pose2D(), 0.01, 100, True, inlier_ratio=.95)),
            scan, Pose2D(), math.radians(30), PanoramaConfig(), lambda *_a: None, lambda *_a: False,
        )
        self.assertFalse(success)

    def test_panorama_qwen_preserves_multiple_observations_without_map_promotion(self):
        payload = {
            "scene_summary": "table and cabinet",
            "landmarks": [
                {"name": "table", "confidence": 0.99},
                {"name": "cabinet", "confidence": 0.98},
            ],
            "objects": [{"name": "bottle", "confidence": 0.90}],
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        labeler = OnlineQwenComponentLabeler("http://example.test/landmarks")
        with patch.object(labeler, "_jpeg_pair", return_value=b"image"), patch.object(
            labeler, "_multipart", return_value=(b"body", "multipart/form-data")
        ) as multipart, patch(
            "raspberry_pi.map_camera_inspection.urllib.request.urlopen",
            return_value=Response(),
        ):
            result = labeler.analyze_panorama(
                {"panorama_station_id": 1, "yaw_deg": 60.0},
                Path("left.jpg"), Path("right.jpg"),
            )
        self.assertFalse(result["promoted"])
        self.assertEqual(len(result["landmarks"]), 2)
        self.assertEqual(len(result["objects"]), 1)
        context = multipart.call_args.args[1]
        self.assertEqual(context["inspection_mode"], "panorama_sector")
        self.assertFalse(context["assign_entire_black_component"])

    def test_blocked_target_retry_waits_until_vehicle_has_moved(self):
        blocked = (0.1, 0.1, "blocked", 0, [(0.0, 0.0)])
        other = (0.4, 0.4, "other", 1, [(0.0, 0.0)])
        selected = _prefer_displaced_retry_choices(
            [blocked, other], Pose2D(0.05, 0.0, 0.0),
            {"blocked": (0.0, 0.0)}, 0.25,
        )
        self.assertEqual([other], selected)
        selected = _prefer_displaced_retry_choices(
            [blocked, other], Pose2D(0.30, 0.0, 0.0),
            {"blocked": (0.0, 0.0)}, 0.25,
        )
        self.assertEqual([blocked, other], selected)

    def test_retry_cooldown_cannot_deadlock_last_target(self):
        blocked = (0.1, 0.1, "blocked", 0, [(0.0, 0.0)])
        selected = _prefer_displaced_retry_choices(
            [blocked], Pose2D(0.0, 0.0, 0.0),
            {"blocked": (0.0, 0.0)}, 0.25,
        )
        self.assertEqual([blocked], selected)

    def test_blocked_arc_recovery_turns_in_place_before_advancing(self):
        args = build_argument_parser().parse_args([])
        command = _blocked_path_heading_command(
            args, Pose2D(0.0, 0.0, math.radians(35.0)), (1.0, 0.0)
        )
        self.assertEqual(0, command[0])
        self.assertLess(command[1], 0)
        self.assertFalse(command[2])
        self.assertIsNone(_blocked_path_heading_command(
            args, Pose2D(0.0, 0.0, math.radians(8.0)), (1.0, 0.0)
        ))

    def test_live_mission_never_counts_remote_proxy_as_target_capture(self):
        source = inspect.getsource(run_map_camera_inspection)
        self.assertNotIn("_remember_temporary_obstacle(", source)
        self.assertNotIn("turn_stalled_coverage_proxy", source)
        self.assertNotIn("blocked_local_coverage_proxy", source)
        self.assertNotIn("coverage_proxy_no_strict_path", source)
        self.assertNotIn("heading_limited_coverage_proxy", source)
        self.assertIn("report.add_skipped_target(", source)
        self.assertIn("no remote proxy photo was taken", source)

    def test_periodic_map_correction_runs_on_time_or_distance(self):
        self.assertFalse(_periodic_map_correction_due(
            104.0, 100.0, 0.40, 8.0, 0.80
        ))
        self.assertTrue(_periodic_map_correction_due(
            108.0, 100.0, 0.40, 8.0, 0.80
        ))
        self.assertTrue(_periodic_map_correction_due(
            104.0, 100.0, 0.80, 8.0, 0.80
        ))
        self.assertTrue(_periodic_map_correction_due(
            101.0, 100.0, 0.01, 8.0, 0.80, force=True
        ))

    def test_scan_motion_drives_turn_between_map_corrections(self):
        start = Pose2D(1.0, 2.0, math.radians(20.0))
        motion = MatchResult(
            transform=Pose2D(0.0, 0.0, math.radians(12.0)),
            rmse_m=0.02,
            correspondences=120,
            inlier_ratio=0.9,
            converged=True,
        )
        predicted = _navigation_motion_pose(start, motion)
        self.assertAlmostEqual(1.0, predicted.x_m)
        self.assertAlmostEqual(2.0, predicted.y_m)
        self.assertAlmostEqual(32.0, math.degrees(predicted.yaw_rad))

    def test_periodic_map_correction_is_blended_not_snapped(self):
        predicted = Pose2D(0.0, 0.0, math.radians(10.0))
        mapped = Pose2D(0.12, -0.08, math.radians(20.0))
        corrected = _blend_navigation_pose(predicted, mapped, 0.35)
        self.assertAlmostEqual(0.042, corrected.x_m, places=3)
        self.assertAlmostEqual(-0.028, corrected.y_m, places=3)
        self.assertAlmostEqual(13.5, math.degrees(corrected.yaw_rad))

    def test_precision_turn_does_not_inherit_adaptive_turn_floor(self):
        controller = AdaptiveSpeedController(
            2600,
            420,
            AdaptiveSpeedConfig(
                initial_turn_floor_mrad_s=2600,
                initial_turn_limit_mrad_s=2600,
            ),
        )
        applied = controller.apply(MotionCommand(
            0, 900, "precision_turning", "camera heading"
        ))
        self.assertEqual(900, applied.angular_mrad_s)

    def test_precision_heading_uses_relative_scan_yaw(self):
        motion = MatchResult(
            transform=Pose2D(0.01, 0.0, math.radians(12.0)),
            rmse_m=0.03,
            correspondences=100,
            converged=True,
            inlier_ratio=0.85,
        )
        progress, source = _precision_heading_progress(
            1100, motion, 240, 0.8
        )
        self.assertEqual("scan_motion", source)
        self.assertAlmostEqual(12.0, math.degrees(progress))

        stopped = MatchResult(
            transform=Pose2D(0.0, 0.0, 0.0),
            rmse_m=0.02,
            correspondences=100,
            converged=True,
            inlier_ratio=0.90,
        )
        progress, source = _precision_heading_progress(
            1100, stopped, 240, 0.8
        )
        self.assertEqual("scan_motion", source)
        self.assertEqual(0.0, progress)

    def test_precision_heading_invalid_jump_does_not_invent_motion(self):
        opposite_jump = MatchResult(
            transform=Pose2D(0.0, 0.0, math.radians(-55.0)),
            rmse_m=0.03,
            correspondences=100,
            converged=True,
            inlier_ratio=0.85,
        )
        progress, source = _precision_heading_progress(
            1100, opposite_jump, 240, 0.8
        )
        self.assertEqual("unverified_no_progress", source)
        self.assertEqual(0.0, progress)

    def test_verified_cabinet_pose_wins_over_duplicate_wall_target(self):
        cabinet = InspectionTarget(
            "object_10_1", "interior_black_region",
            (0.025, 1.725), (-0.575, 1.475), math.radians(22.6), 20,
        )
        duplicate_wall = InspectionTarget(
            "wall_6_1", "perimeter_wall",
            (-0.025, 1.625), (-0.675, 1.625), 0.0, 65,
        )
        semantic = type("Semantic", (), {"data": {"entities": {
            "object_10_1": {
                "qwen_annotation": {
                    "promoted_to_semantic_label": True,
                    "views": [{"target_visible": True}],
                    "source_images": ["verified_cabinet.jpg"],
                },
                "observations": [{
                    "image": "/old/verified_cabinet.jpg",
                    "robot_x_m": -0.613,
                    "robot_y_m": 1.4667,
                    "robot_yaw_deg": 33.0,
                }],
            },
        }}})()
        adjusted, verified = _apply_verified_semantic_views(
            [duplicate_wall, cabinet], semantic
        )
        self.assertEqual(["object_10_1"], [item.target_id for item in adjusted])
        self.assertAlmostEqual(-0.613, adjusted[0].view_xy_m[0])
        self.assertAlmostEqual(33.0, math.degrees(adjusted[0].yaw_rad))
        self.assertIn("object_10_1", verified)

    def test_target_spacing_keeps_object_view_over_clustered_wall_view(self):
        wall = InspectionTarget(
            "wall_1_1", "perimeter_wall",
            (0.0, 0.0), (0.0, 0.0), 0.0, 100,
        )
        object_view = InspectionTarget(
            "object_2_1", "interior_black_region",
            (0.2, 0.0), (0.1, 0.0), 0.0, 20,
        )
        distant_wall = InspectionTarget(
            "wall_3_1", "perimeter_wall",
            (1.0, 0.0), (1.0, 0.0), 0.0, 100,
        )
        spaced = _spaced_targets(
            [wall, object_view, distant_wall], min_spacing_m=0.40
        )
        self.assertEqual(
            ["object_2_1", "wall_3_1"],
            [target.target_id for target in spaced],
        )

    def test_information_view_moves_outward_but_keeps_same_component(self):
        known = KnownOccupancyMap(
            _room_image(), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        target = known.inspection_targets(
            standoff_m=0.65,
            maximum_targets=12,
            views_per_wall=4,
            views_per_region=2,
        )[0]
        widened, changed = _information_view_for_target(
            known, target, maximum_distance_m=1.60
        )
        self.assertTrue(changed)
        self.assertEqual(target.target_id, widened.target_id)
        self.assertEqual(target.target_xy_m, widened.target_xy_m)
        self.assertGreater(
            math.dist(widened.view_xy_m, target.target_xy_m),
            math.dist(target.view_xy_m, target.target_xy_m) + 0.30,
        )

    def test_online_qwen_rejects_generic_boxes_and_accepts_bounded_stereo(self):
        labeler = OnlineQwenComponentLabeler(
            "http://example.invalid/landmarks",
            min_confidence=0.80,
            confirmations=2,
        )
        generic = {
            "landmarks": [{
                "type": "fixed_cabinet",
                "confidence": 0.95,
                "camera_bbox": [0, 0, 1000, 1000],
                "stereo_consistency": "both",
                "ground_contact": True,
                "map_memory_allowed": True,
                "behind_glass": False,
            }]
        }
        self.assertIsNone(labeler._candidate(generic))
        bounded = {
            "landmarks": [{
                "type": "fixed_table",
                "confidence": 0.88,
                "camera_bbox": [150, 250, 350, 850],
                "stereo_consistency": "both",
                "ground_contact": True,
                "map_memory_allowed": True,
                "behind_glass": False,
            }]
        }
        confidence, semantic_label, _item = labeler._candidate(bounded)
        self.assertAlmostEqual(0.88, confidence)
        self.assertEqual("table", semantic_label)

        interior_target = InspectionTarget(
            "object_10_1", "interior_black_region", (0.0, 0.0),
            (0.5, 0.0), math.pi, 100,
        )
        specific_left_half = {
            "landmarks": [{
                "type": "fixed_cabinet",
                "confidence": 0.95,
                "camera_bbox": [0, 0, 500, 1000],
                "stereo_consistency": "both",
                "ground_contact": True,
                "map_memory_allowed": True,
                "behind_glass": False,
            }]
        }
        confidence, semantic_label, item = labeler._candidate(
            specific_left_half, interior_target
        )
        self.assertAlmostEqual(0.95, confidence)
        self.assertEqual("cabinet", semantic_label)
        self.assertEqual(
            "specific_stereo_classification_without_tight_bbox",
            item["evidence_mode"],
        )

        generic_structure = {
            "landmarks": [{
                "type": "fixed_structure",
                "confidence": 0.99,
                "camera_bbox": [0, 0, 500, 1000],
                "stereo_consistency": "both",
                "ground_contact": True,
                "map_memory_allowed": True,
                "behind_glass": False,
            }]
        }
        self.assertIsNone(labeler._candidate(
            generic_structure, interior_target
        ))

    def test_verified_target_rejects_latest_far_table_view(self):
        verified = {"object_10_1": Pose2D(
            -0.613, 1.4667, math.radians(33.0)
        )}
        references = {"object_10_1": (-0.613, 1.4667)}
        allowed, distance, limit_m = _within_capture_reference(
            Pose2D(-0.5235, 0.6807, math.radians(68.3)),
            "object_10_1", references, verified, 0.50,
        )
        self.assertFalse(allowed)
        self.assertGreater(distance, 0.78)
        self.assertEqual(0.32, limit_m)

        allowed, distance, _limit = _within_capture_reference(
            Pose2D(-0.60, 1.45, math.radians(31.0)),
            "object_10_1", references, verified, 0.50,
        )
        self.assertTrue(allowed)
        self.assertLess(distance, 0.03)

    def test_precision_heading_seed_averages_across_wraparound(self):
        seed = _circular_heading_seed([
            math.radians(178.0),
            math.radians(-179.0),
            math.radians(177.0),
        ])
        self.assertIsNotNone(seed)
        self.assertLess(
            abs(normalize_angle(seed - math.radians(178.67))),
            math.radians(0.2),
        )

    def test_nearby_turn_timeout_captures_only_at_final_view(self):
        pose = Pose2D(-0.3793, 1.0825, math.radians(-170.7))
        allowed, distance = _nearby_turn_timeout_capture_allowed(
            pose,
            (-0.675, 1.625),
            waypoint_index=3,
            waypoint_count=4,
            final_translation_complete=True,
            nearby_capture_distance_m=0.70,
        )
        self.assertTrue(allowed)
        self.assertAlmostEqual(0.6178, distance, places=3)

        allowed, _distance = _nearby_turn_timeout_capture_allowed(
            pose,
            (-0.675, 1.625),
            waypoint_index=0,
            waypoint_count=4,
            final_translation_complete=False,
            nearby_capture_distance_m=0.70,
        )
        self.assertFalse(allowed)
        self.assertEqual(
            "capture",
            _navigation_turn_timeout_action(8.0, 8.0, True),
        )
        self.assertEqual(
            "continue",
            _navigation_turn_timeout_action(80.0, 8.0, False),
        )
        self.assertEqual(
            "replan",
            _navigation_turn_timeout_action(
                8.1,
                8.0,
                True,
                heading_error_rad=math.radians(69.0),
                maximum_capture_error_deg=25.0,
            ),
        )
        self.assertEqual(
            "capture",
            _navigation_turn_timeout_action(
                8.1,
                8.0,
                True,
                heading_error_rad=math.radians(18.0),
                maximum_capture_error_deg=25.0,
            ),
        )

    def test_pulsed_navigation_uses_bounded_front_envelope(self):
        scan = LaserScan(
            np.linspace(-math.pi, math.pi, 360, endpoint=False),
            np.full(360, 0.763),
            time.monotonic(),
        )
        # The lidar selects the longest pulse that can still stop inside the
        # measured corridor. 0.763m is therefore a valid route clearance,
        # while the same requested 700ms pulse is shortened automatically.
        safe_ttl_ms = _navigation_command_ttl(700, 360, 0.763, 0.42)
        self.assertGreaterEqual(safe_ttl_ms, 350)
        self.assertLess(safe_ttl_ms, 700)
        self.assertTrue(_obstacle_safe(
            scan,
            360,
            0.42,
            0,
            full_scan_guard=True,
            command_ttl_ms=safe_ttl_ms,
        ))
        close_scan = LaserScan(
            np.linspace(-math.pi, math.pi, 360, endpoint=False),
            np.full(360, 0.60),
            time.monotonic(),
        )
        close_ttl_ms = _navigation_command_ttl(700, 360, 0.60, 0.42)
        self.assertFalse(_obstacle_safe(
            close_scan,
            360,
            0.42,
            0,
            full_scan_guard=True,
            command_ttl_ms=close_ttl_ms,
        ))
        # Initial active localization keeps the shorter, stopped-action
        # envelope used by the previously successful field run.
        self.assertTrue(_obstacle_safe(scan, 360, 0.42, 0))

    def test_collision_reverse_requires_lidar_and_map_corridor(self):
        args = build_argument_parser().parse_args([])
        image = np.full((80, 80), 254, dtype=np.uint8)
        image[[0, -1], :] = 0
        image[:, [0, -1]] = 0
        known = KnownOccupancyMap(
            image,
            resolution_m=0.05,
            origin_x_m=0.0,
            origin_y_m=0.0,
            clearance_m=0.10,
        )
        pose = Pose2D(2.0, 2.0, 0.0)
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        clear_scan = LaserScan(
            angles, np.full(360, 1.0), time.monotonic()
        )
        safe, rear, required, endpoint, reason = (
            _collision_reverse_is_safe(
                clear_scan, pose, known, args
            )
        )
        self.assertTrue(safe)
        self.assertGreater(rear, required)
        self.assertEqual("", reason)

        unsupported_rear = LaserScan(
            np.asarray((-math.pi / 2.0, math.pi / 2.0)),
            np.asarray((1.0, 1.0)),
            time.monotonic(),
        )
        safe, rear, _required, _endpoint, reason = (
            _collision_reverse_is_safe(
                unsupported_rear, pose, known, args
            )
        )
        self.assertFalse(safe)
        self.assertTrue(math.isinf(rear))
        self.assertEqual("rear_lidar_support_low", reason)

        fallback_args = build_argument_parser().parse_args([
            "--collision-reverse-distance", "0.06",
            "--collision-reverse-map-fallback",
            "--collision-reverse-map-fallback-max-distance", "0.08",
        ])
        safe, rear, _required, fallback_endpoint, reason = (
            _collision_reverse_is_safe(
                unsupported_rear, pose, known, fallback_args
            )
        )
        self.assertTrue(safe)
        self.assertTrue(math.isinf(rear))
        self.assertEqual("map_only_rear_lidar_fallback", reason)
        self.assertAlmostEqual(1.94, fallback_endpoint[0], places=6)

        sparse_close_rear = LaserScan(
            np.asarray((math.pi - 0.02, math.pi + 0.02)),
            np.asarray((0.18, 0.18)),
            time.monotonic(),
        )
        safe, _rear, _required, _endpoint, reason = (
            _collision_reverse_is_safe(
                sparse_close_rear, pose, known, fallback_args
            )
        )
        self.assertFalse(safe)
        self.assertEqual("rear_sparse_close_return", reason)

        rear_blocked_ranges = np.full(360, 1.0)
        rear_angles = np.abs(
            (angles - math.pi + math.pi) % (2.0 * math.pi) - math.pi
        ) <= math.radians(25.0)
        rear_blocked_ranges[rear_angles] = 0.30
        rear_blocked = LaserScan(
            angles, rear_blocked_ranges, time.monotonic()
        )
        safe, _rear, _required, _endpoint, reason = (
            _collision_reverse_is_safe(
                rear_blocked, pose, known, args
            )
        )
        self.assertFalse(safe)
        self.assertEqual("rear_clearance_low", reason)

        probe_args = build_argument_parser().parse_args([
            "--collision-reverse-distance", "0.06",
            "--collision-reverse-rear-clearance", "0.28",
        ])
        isolated_speckle_ranges = np.full(360, 1.0)
        rear_indices = np.flatnonzero(rear_angles)
        isolated_speckle_ranges[rear_indices[len(rear_indices) // 2]] = 0.08
        isolated_speckle = LaserScan(
            angles, isolated_speckle_ranges, time.monotonic()
        )
        safe, rear, required, _endpoint, reason = (
            _collision_reverse_is_safe(
                isolated_speckle, pose, known, probe_args
            )
        )
        self.assertTrue(safe)
        self.assertGreater(rear, required)
        self.assertEqual("", reason)

        # The installed lidar sees the chassis/cable bundle at 8--10cm over
        # several adjacent rear rays.  Treat that band as self-shadow.
        clustered_ranges = np.full(360, 1.0)
        center = len(rear_indices) // 2
        clustered_ranges[rear_indices[center - 1:center + 2]] = 0.08
        clustered_scan = LaserScan(
            angles, clustered_ranges, time.monotonic()
        )
        safe, _rear, _required, _endpoint, reason = (
            _collision_reverse_is_safe(
                clustered_scan, pose, known, probe_args
            )
        )
        self.assertTrue(safe)
        self.assertEqual("", reason)

        real_cluster_ranges = np.full(360, 1.0)
        real_cluster_ranges[rear_indices[center - 1:center + 2]] = 0.18
        real_cluster_scan = LaserScan(
            angles, real_cluster_ranges, time.monotonic()
        )
        safe, _rear, _required, _endpoint, reason = (
            _collision_reverse_is_safe(
                real_cluster_scan, pose, known, probe_args
            )
        )
        self.assertFalse(safe)
        self.assertEqual("rear_hard_obstacle_cluster", reason)

        endpoint_cell = known.world_to_grid(
            np.asarray((endpoint,))
        )[0]
        column, row = (int(value) for value in endpoint_cell)
        known.traversable[row, column] = False
        safe, _rear, _required, _endpoint, reason = (
            _collision_reverse_is_safe(
                clear_scan, pose, known, args
            )
        )
        self.assertFalse(safe)
        self.assertEqual("reverse_map_corridor_blocked", reason)

    def test_navigation_pose_guard_holds_only_physically_inconsistent_jumps(self):
        args = build_argument_parser().parse_args([])
        trusted = Pose2D(0.0, 0.0, 0.0)

        stopped_jump = LocalizationResult(
            Pose2D(0.02, 0.01, math.radians(66.0)),
            0.04,
            0.96,
            1.0,
        )
        self.assertEqual(
            "stationary_yaw_jump",
            _navigation_pose_jump_reason(
                trusted, stopped_jump, (0, 0), 0.5, 450, args
            ),
        )

        wrong_direction = LocalizationResult(
            Pose2D(0.0, 0.0, math.radians(-30.0)),
            0.05,
            0.80,
            1.0,
        )
        self.assertEqual(
            "turn_direction_contradiction",
            _navigation_pose_jump_reason(
                trusted, wrong_direction, (0, 1500), 0.5, 450, args
            ),
        )

        normal_turn = LocalizationResult(
            Pose2D(0.02, -0.01, math.radians(35.0)),
            0.06,
            0.76,
            1.0,
        )
        self.assertEqual(
            "",
            _navigation_pose_jump_reason(
                trusted, normal_turn, (0, 1500), 0.5, 450, args
            ),
        )

        field_turn_alias = LocalizationResult(
            Pose2D(0.0, 0.0, math.radians(61.0)),
            0.035,
            0.95,
            1.0,
        )
        self.assertEqual(
            "yaw_exceeds_command_envelope",
            _navigation_pose_jump_reason(
                trusted, field_turn_alias, (0, 2600), 0.72, 700, args
            ),
        )

        field_forward_alias = LocalizationResult(
            Pose2D(0.32, 0.02, math.radians(-49.0)),
            0.04,
            0.92,
            1.0,
        )
        self.assertEqual(
            "yaw_exceeds_command_envelope",
            _navigation_pose_jump_reason(
                trusted, field_forward_alias, (400, 18), 0.84, 500, args
            ),
        )

        bounded_recovery = LocalizationResult(
            Pose2D(0.10, 0.09, math.radians(8.0)),
            0.09,
            0.30,
            1.0,
        )
        self.assertEqual(
            "",
            _navigation_pose_jump_reason(
                trusted, bounded_recovery, (0, 0), 0.7, 450, args
            ),
        )

    def test_blocked_view_gets_nearby_view_of_same_target(self):
        known = KnownOccupancyMap(
            _room_image(), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        target = known.inspection_targets(
            standoff_m=0.65,
            maximum_targets=12,
            views_per_wall=4,
            views_per_region=2,
        )[0]
        known.mark_temporary_obstacles(
            (target.view_xy_m,), radius_m=0.18, minimum_hits=1
        )
        pose = Pose2D(-0.5, 0.0, 0.0)
        replacement = _alternate_inspection_target(
            known, pose, target, max_shift_m=1.0
        )
        self.assertIsNotNone(replacement)
        self.assertEqual(target.target_id, replacement.target_id)
        self.assertEqual(target.target_xy_m, replacement.target_xy_m)
        self.assertLessEqual(
            math.hypot(
                replacement.view_xy_m[0] - target.view_xy_m[0],
                replacement.view_xy_m[1] - target.view_xy_m[1],
            ),
            1.0,
        )
        self.assertTrue(_inspection_target_visible_from(
            known, replacement.view_xy_m, target.target_xy_m
        ))
        second_replacement = _alternate_inspection_target(
            known,
            pose,
            replacement,
            max_shift_m=1.0,
            excluded_views=(target.view_xy_m, replacement.view_xy_m),
        )
        self.assertIsNotNone(second_replacement)
        self.assertGreaterEqual(
            math.hypot(
                second_replacement.view_xy_m[0] - replacement.view_xy_m[0],
                second_replacement.view_xy_m[1] - replacement.view_xy_m[1],
            ),
            0.16,
        )

    def test_recovery_requires_quality_and_two_consistent_reanchored_frames(self):
        args = build_argument_parser().parse_args([])
        trusted = Pose2D(0.0, 0.0, 0.0)
        weak = LocalizationResult(
            Pose2D(0.06, 0.04, math.radians(4.0)), 0.08, 0.20, 1.0
        )
        first = LocalizationResult(
            Pose2D(0.05, 0.03, math.radians(3.0)), 0.07, 0.60, 1.0
        )
        second = LocalizationResult(
            Pose2D(0.07, 0.04, math.radians(4.0)), 0.08, 0.58, 1.0
        )
        disagreement = LocalizationResult(
            Pose2D(-0.05, -0.04, math.radians(-5.0)), 0.07, 0.62, 1.0
        )

        self.assertFalse(_navigation_recovery_candidate_usable(
            trusted, weak, args
        ))
        self.assertTrue(_navigation_recovery_candidate_bounded(
            trusted, weak, args
        ))
        self.assertTrue(_navigation_recovery_candidate_usable(
            trusted, first, args
        ))
        self.assertFalse(_navigation_recovery_candidates_consistent(
            [first], required_frames=2
        ))
        self.assertTrue(_navigation_recovery_candidates_consistent(
            [first, second], required_frames=2
        ))
        self.assertFalse(_navigation_recovery_candidates_consistent(
            [first, disagreement], required_frames=2
        ))
        self.assertTrue(_navigation_recovery_candidates_consistent(
            [first, disagreement],
            required_frames=2,
            position_tolerance_m=0.15,
            yaw_tolerance_deg=10.0,
        ))
        self.assertIs(first, _best_bounded_recovery_candidate(
            trusted, [weak, first, second]
        ))

    def test_annotation_owned_client_starts_independent_motion_keepalive(self):
        class Client:
            maintenance_running = False

            def start(self):
                self.maintenance_running = True

        client = Client()
        self.assertTrue(_start_motion_keepalive(client, owns_client=True))
        self.assertTrue(client.maintenance_running)

        injected = Client()
        self.assertFalse(_start_motion_keepalive(injected, owns_client=False))
        self.assertFalse(injected.maintenance_running)

    def test_load_preserves_ros_world_coordinates_and_never_mutates_file(self):
        with tempfile.TemporaryDirectory() as directory:
            pgm = Path(directory) / "map.pgm"
            yaml = Path(directory) / "map.yaml"
            Image.fromarray(_room_image()).save(pgm)
            yaml.write_text(
                "image: map.pgm\nresolution: 0.05\n"
                "origin: [-3.0, -2.5, 0.0]\n",
                encoding="utf-8",
            )
            before = pgm.read_bytes()
            known = KnownOccupancyMap.load(pgm, yaml, clearance_m=0.20)
            self.assertEqual((100, 120), known.image.shape)
            cell = known.world_to_grid(((-2.475, -1.975),))[0]
            self.assertEqual((10, 10), tuple(cell))
            self.assertEqual(before, pgm.read_bytes())

    def test_path_avoids_inner_black_region(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        start = tuple(known.grid_to_world(((25, 50),))[0])
        goal = tuple(known.grid_to_world(((90, 50),))[0])
        path = known.path(start, goal)
        self.assertGreater(len(path), 2)
        self.assertTrue(all(known.line_is_clear(a, b) for a, b in zip(path, path[1:])))

    def test_generates_wall_and_inner_object_views(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        targets = known.inspection_targets(standoff_m=0.60, maximum_targets=12)
        kinds = {item.kind for item in targets}
        self.assertIn("perimeter_wall", kinds)
        self.assertIn("interior_black_region", kinds)
        for target in targets:
            column, row = known.world_to_grid((target.view_xy_m,))[0]
            self.assertTrue(known.traversable[row, column])

    def test_orders_only_reachable_targets(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        targets = known.inspection_targets(standoff_m=0.60, maximum_targets=8)
        start = tuple(known.grid_to_world(((25, 50),))[0])
        ordered = order_targets(known, start, targets)
        self.assertTrue(ordered)
        self.assertLessEqual(len(ordered), len(targets))

    def test_temporary_obstacle_changes_only_session_planning_layer(self):
        permanent = np.flipud(_room_image())
        known = KnownOccupancyMap(
            permanent, 0.05, -3.0, -2.5, clearance_m=0.20
        )
        permanent_before = known.image.copy()
        occupied_before = known.occupied.copy()
        start = tuple(known.grid_to_world(((25, 50),))[0])
        goal = tuple(known.grid_to_world(((42, 50),))[0])
        direct_path = known.path(start, goal)
        self.assertTrue(direct_path)
        obstacle = tuple(known.grid_to_world(((34, 50),))[0])
        self.assertGreater(
            known.mark_temporary_obstacles(
                (obstacle,), radius_m=0.10, minimum_hits=1
            ),
            0,
        )
        detour = known.path(start, goal)
        self.assertTrue(detour)
        self.assertGreaterEqual(len(detour), len(direct_path))
        np.testing.assert_array_equal(permanent_before, known.image)
        np.testing.assert_array_equal(occupied_before, known.occupied)
        known.clear_temporary_obstacles()
        self.assertFalse(np.any(known.temporary_occupied))

    def test_new_map_instance_forgets_previous_temporary_obstacles(self):
        first = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        point = tuple(first.grid_to_world(((30, 30),))[0])
        first.mark_temporary_obstacles(
            (point,), radius_m=0.10, minimum_hits=1
        )
        self.assertTrue(np.any(first.temporary_occupied))
        second = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        self.assertFalse(np.any(second.temporary_occupied))

    def test_temporary_obstacle_requires_repeated_scan_when_configured(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        first = tuple(known.grid_to_world(((30, 30),))[0])
        jittered = tuple(known.grid_to_world(((31, 30),))[0])
        self.assertEqual(
            0,
            known.mark_temporary_obstacles(
                (first,), radius_m=0.10, minimum_hits=2
            ),
        )
        self.assertFalse(np.any(known.temporary_occupied))
        self.assertGreater(
            known.mark_temporary_obstacles(
                (jittered,), radius_m=0.10, minimum_hits=2
            ),
            0,
        )


class LocalizerTests(unittest.TestCase):
    def test_tracking_motion_prior_can_be_reanchored_after_guarded_jump(self):
        known = KnownOccupancyMap(
            _room_image(), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        localizer = ScanMapLocalizer(known)
        localizer._last_track_scan = object()
        localizer.reset_tracking_motion_prior()
        self.assertIsNone(localizer._last_track_scan)
        self.assertTrue(localizer._wide_track_once)

    def test_navigation_uses_short_turn_pulses(self):
        args = build_argument_parser().parse_args([])
        controller = _build_speed_controller(args, navigation=True)
        diagnostics = controller.diagnostics()
        self.assertEqual(220, diagnostics["turn_ttl_ms"])
        self.assertEqual(200, controller.config.minimum_turn_ttl_ms)
        self.assertEqual(260, controller.config.maximum_turn_ttl_ms)

    def test_camera_stream_failure_falls_back_and_sticks_to_direct_camera(self):
        class Backend:
            def __init__(self, name, fail=False):
                self.name = name
                self.fail = fail
                self.calls = 0

            def capture(self, path):
                self.calls += 1
                if self.fail:
                    raise RuntimeError(f"{self.name} unavailable")
                Path(path).write_bytes(b"jpeg")
                return {"backend": self.name, "camera_count": 2}

        stream = Backend("mjpeg", fail=True)
        direct = Backend("opencv")
        camera = ResilientCameraCapture(
            stream, direct, attempts=2, retry_delay_s=0.0
        )
        with tempfile.TemporaryDirectory() as directory:
            first = camera.capture(Path(directory) / "first.jpg")
            second = camera.capture(Path(directory) / "second.jpg")
        self.assertTrue(first["camera_failover_used"])
        self.assertTrue(second["camera_failover_used"])
        self.assertEqual(1, stream.calls)
        self.assertEqual(2, direct.calls)

    def test_camera_failure_reports_both_backends_after_bounded_retries(self):
        class BrokenBackend:
            def __init__(self, name):
                self.name = name
                self.calls = 0

            def capture(self, _path):
                self.calls += 1
                raise RuntimeError(self.name)

        stream = BrokenBackend("stream down")
        direct = BrokenBackend("v4l2 down")
        camera = ResilientCameraCapture(
            stream, direct, attempts=2, retry_delay_s=0.0
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "均不可用"):
                camera.capture(Path(directory) / "failed.jpg")
        self.assertEqual(2, stream.calls)
        self.assertEqual(2, direct.calls)

    def test_search_client_creation_retries_before_motion(self):
        class Client:
            def heartbeat(self):
                return True

            def stop(self):
                return None

        attempts = []

        def factory():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("temporary link failure")
            return Client()

        result = _create_client_with_retry(factory, attempts=3, delay_s=0.0)
        self.assertIsInstance(result, Client)
        self.assertEqual(len(attempts), 3)

    def test_search_navigation_predicts_motion_then_bounds_map_correction(self):
        pose = Pose2D(1.0, 2.0, math.radians(10.0))
        motion = MatchResult(Pose2D(0.10, 0.0, math.radians(3.0)), 0.03, 120, True, 0.8)
        predicted, accepted = _scan_motion_prediction(pose, motion)
        self.assertTrue(accepted)
        self.assertGreater(math.hypot(predicted.x_m - pose.x_m, predicted.y_m - pose.y_m), 0.09)

        from raspberry_pi.known_map_navigation import LocalizationResult
        map_result = LocalizationResult(
            Pose2D(predicted.x_m + 0.20, predicted.y_m - 0.10, predicted.yaw_rad + 0.25),
            0.04, 0.85, 0.75, 0.2,
        )
        fused = _fuse_tracking_pose(predicted, map_result)
        self.assertLess(math.hypot(fused.x_m - predicted.x_m, fused.y_m - predicted.y_m), 0.04)
        self.assertLess(abs(normalize_angle(fused.yaw_rad - predicted.yaw_rad)), math.radians(2.1))

    def test_search_command_uses_arc_for_moderate_heading_error(self):
        args = build_search_argument_parser().parse_args(["--decision", "unused.json"])
        linear, angular, reached = _strict_command_for_pose(
            args, Pose2D(0.0, 0.0, 0.0), (1.0, math.tan(math.radians(12.0)))
        )
        self.assertFalse(reached)
        self.assertGreater(linear, 0)
        self.assertGreater(angular, 0)

    def test_search_accepts_close_waypoint_instead_of_replanning_in_place(self):
        args = build_search_argument_parser().parse_args(["--decision", "unused.json"])
        linear, angular, reached = _strict_command_for_pose(
            args, Pose2D(0.0, 0.0, 0.0), (0.158, 0.0)
        )
        self.assertTrue(reached)
        self.assertEqual((linear, angular), (0, 0))

    def test_search_accepts_configured_eighteen_centimetre_waypoint_tolerance(self):
        args = build_search_argument_parser().parse_args(["--decision", "unused.json"])
        linear, angular, reached = _strict_command_for_pose(
            args, Pose2D(0.0, 0.0, math.pi), (0.174, 0.0)
        )
        self.assertTrue(reached)
        self.assertEqual((linear, angular), (0, 0))

    def test_object_search_start_kick_crosses_static_friction_floor(self):
        controller = AdaptiveSpeedController(
            2600,
            460,
            AdaptiveSpeedConfig(
                minimum_applied_linear_speed_mm_s=230,
                linear_start_kick_mm_s=400,
            ),
        )
        applied = controller.apply(MotionCommand(230, 0, "advancing", "test"))
        self.assertEqual(applied.linear_mm_s, 400)

    def test_object_search_requires_complete_target_in_both_camera_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.jpg"
            right = Path(directory) / "right.jpg"
            Image.new("RGB", (320, 240), "white").save(left)
            Image.new("RGB", (320, 240), "white").save(right)
            detections = [
                {
                    "camera_id": "left", "class_name": "bottle", "source": "yolo",
                    "confidence": 0.91, "bbox_1000": [220, 180, 480, 760],
                    "attributes": {},
                },
                {
                    "camera_id": "right", "class_name": "bottle", "source": "yolo",
                    "confidence": 0.88, "bbox_1000": [410, 190, 675, 770],
                    "attributes": {},
                },
            ]
            result = confirm_detections(
                detections,
                {"class_name": "bottle", "detection_mode": "yolo primary"},
                {"left": left, "right": right},
            )
            self.assertEqual(2, sum(bool(item["confirmed_match"]) for item in result))
            self.assertTrue(all(item["dual_camera_confirmed"] for item in result))

    def test_object_search_does_not_finish_from_one_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.jpg"
            right = Path(directory) / "right.jpg"
            Image.new("RGB", (320, 240), "white").save(left)
            Image.new("RGB", (320, 240), "white").save(right)
            result = confirm_detections(
                [{
                    "camera_id": "left", "class_name": "bottle", "source": "yolo",
                    "confidence": 0.91, "bbox_1000": [220, 180, 480, 760],
                    "attributes": {},
                }],
                {"class_name": "bottle", "detection_mode": "yolo primary"},
                {"left": left, "right": right},
            )
            self.assertTrue(result[0]["single_camera_match"])
            self.assertFalse(result[0]["confirmed_match"])

    def test_object_search_rejects_target_cropped_in_either_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.jpg"
            right = Path(directory) / "right.jpg"
            Image.new("RGB", (320, 240), "white").save(left)
            Image.new("RGB", (320, 240), "white").save(right)
            detections = [
                {
                    "camera_id": "left", "class_name": "bottle", "source": "yolo",
                    "confidence": 0.91, "bbox_1000": [220, 180, 480, 760],
                    "attributes": {},
                },
                {
                    "camera_id": "right", "class_name": "bottle", "source": "yolo",
                    "confidence": 0.90, "bbox_1000": [0, 190, 260, 770],
                    "attributes": {},
                },
            ]
            result = confirm_detections(
                detections,
                {"class_name": "bottle", "detection_mode": "yolo primary"},
                {"left": left, "right": right},
            )
            self.assertFalse(any(item["confirmed_match"] for item in result))
            self.assertFalse(result[1]["complete_in_frame"])

    def test_tracking_matches_synthetic_scan_to_map(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.15
        )
        true_pose = Pose2D(-1.4, -0.3, math.radians(18.0))
        world_points = known.occupied_points(maximum=700)
        local = true_pose.inverse().transform_points(world_points)
        ranges = np.linalg.norm(local, axis=1)
        local = local[(ranges > 0.12) & (ranges < 4.0)]
        angles = np.arctan2(local[:, 1], local[:, 0])
        distances = np.linalg.norm(local, axis=1)
        scan = LaserScan(angles, distances, 1.0)
        localizer = ScanMapLocalizer(known, lidar_offset_x_m=0.0)
        estimate = localizer.track(
            scan, Pose2D(-1.34, -0.34, math.radians(21.0))
        )
        self.assertIsNotNone(estimate)
        self.assertLessEqual(math.hypot(
            estimate.pose.x_m - true_pose.x_m,
            estimate.pose.y_m - true_pose.y_m,
        ), 0.11)
        self.assertLess(abs(estimate.pose.yaw_rad - true_pose.yaw_rad), math.radians(5.0))

    def test_tracking_limits_expensive_metric_raycast_shortlist(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.15
        )
        localizer = ScanMapLocalizer(known, lidar_offset_x_m=0.0)
        candidates = [
            Pose2D(-1.8 + index * 0.01, -0.8, 0.0)
            for index in range(100)
        ]
        raycast_scores = []
        localizer._evaluate = lambda _points, pose: (
            pose.x_m,
            0.03,
            0.9,
        )

        def range_score(pose, _profile):
            raycast_scores.append(pose)
            return 1.0, 0.0, 1.0

        localizer._range_consistency = range_score
        ranked = localizer._rank(
            np.empty((0, 2)),
            candidates,
            keep=4,
            range_profile=np.asarray(((0.0, 1.0),)),
            range_shortlist_min=12,
        )
        self.assertGreaterEqual(len(ranked), 1)
        self.assertEqual(12, len(raycast_scores))

    def test_metric_beam_ranges_reject_a_shape_similar_wrong_corner(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.15
        )
        localizer = ScanMapLocalizer(known, lidar_offset_x_m=0.0)
        true_pose = Pose2D(-2.15, -1.55, math.radians(12.0))
        wrong_pose = Pose2D(1.65, 1.05, math.radians(-168.0))
        profile = []
        for angle in np.linspace(-math.pi, math.pi, 40, endpoint=False):
            distance = localizer._raycast_range(true_pose, angle)
            if distance is not None:
                profile.append((angle, distance))
        self.assertGreaterEqual(len(profile), 20)
        true_score = localizer._range_consistency(true_pose, np.asarray(profile))
        wrong_score = localizer._range_consistency(wrong_pose, np.asarray(profile))
        self.assertIsNotNone(true_score)
        self.assertIsNotNone(wrong_score)
        self.assertGreater(true_score[0], wrong_score[0] + 0.30)
        self.assertLess(true_score[1], wrong_score[1])

    def test_global_candidates_keep_separated_pose_basins(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.15
        )
        true_pose = Pose2D(-1.4, -0.3, math.radians(18.0))
        local = true_pose.inverse().transform_points(known.occupied_points(maximum=900))
        ranges = np.linalg.norm(local, axis=1)
        local = local[(ranges > 0.12) & (ranges < 4.0)]
        scan = LaserScan(np.arctan2(local[:, 1], local[:, 0]), np.linalg.norm(local, axis=1), 1.0)
        candidates = ScanMapLocalizer(known, lidar_offset_x_m=0.0).global_localization_candidates(
            scan, maximum=3
        )
        self.assertGreaterEqual(len(candidates), 1)
        for first, second in zip(candidates, candidates[1:]):
            separated = (
                math.hypot(first.pose.x_m - second.pose.x_m, first.pose.y_m - second.pose.y_m) >= 0.45
                or abs(normalize_angle(first.pose.yaw_rad - second.pose.yaw_rad)) >= math.radians(25)
            )
            self.assertTrue(separated)

    def test_disambiguating_motion_maximizes_future_metric_range_difference(self):
        image = np.full((140, 180), 254, dtype=np.uint8)
        image[[0, -1], :] = 0
        image[:, [0, -1]] = 0
        image[55:85, 78:102] = 0
        image[20:45, 130:136] = 0
        image[95:102, 25:60] = 0
        known = KnownOccupancyMap(
            image, 0.05, -4.5, -3.5, clearance_m=0.20
        )
        localizer = ScanMapLocalizer(known, lidar_offset_x_m=0.0)
        from raspberry_pi.known_map_navigation import LocalizationResult
        candidates = [
            LocalizationResult(Pose2D(-3.0, -2.0, 0.0), 0.05, 0.8, 0.80, 0.02),
            LocalizationResult(Pose2D(2.5, -2.0, math.pi), 0.05, 0.8, 0.77, 0.02),
        ]
        action = localizer.disambiguating_motion(candidates, maximum_distance_m=0.26)
        self.assertIsNotNone(action)
        self.assertGreater(action["separation_score_m"], 0.10)
        self.assertIn(action["turn_deg"], {-70.0, -35.0, 0.0, 35.0, 70.0})

    def test_annotation_startup_requires_shape_and_metric_range_agreement(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.15
        )
        localizer = ScanMapLocalizer(known, lidar_offset_x_m=0.0)
        true_pose = Pose2D(-2.15, -1.55, math.radians(12.0))
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        ranges = np.asarray([
            localizer._raycast_range(true_pose, angle) or 8.0
            for angle in angles
        ])
        scan = LaserScan(angles, ranges, 1.0)
        from raspberry_pi.known_map_navigation import LocalizationResult
        result = LocalizationResult(true_pose, 0.05, 0.82, 0.8, 0.10)
        args = build_search_argument_parser().parse_args(["--decision", "unused.json"])
        # The object-search parser does not own annotation thresholds; attach
        # the strict annotation values explicitly to this compact fixture.
        args.localization_max_rmse = 0.12
        args.localization_min_inlier_ratio = 0.60
        args.localization_min_range_agreement = 0.55
        args.localization_max_range_error = 0.24
        args.localization_min_range_coverage = 0.40
        accepted, evidence = _strict_radar_localization(
            result, scan, localizer, args
        )
        self.assertTrue(accepted, evidence)
        wrong = LocalizationResult(
            Pose2D(1.65, 1.05, math.radians(-168.0)), 0.05, 0.82, 0.8, 0.10
        )
        rejected, wrong_evidence = _strict_radar_localization(
            wrong, scan, localizer, args
        )
        self.assertFalse(rejected)
        self.assertGreater(
            wrong_evidence["range_mean_error_m"],
            evidence["range_mean_error_m"],
        )

    def test_relative_motion_matches_two_synthetic_viewpoints(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.15
        )
        first_pose = Pose2D(-1.4, -0.3, math.radians(8.0))
        delta = Pose2D(0.18, 0.02, math.radians(5.0))
        second_pose = first_pose.compose(delta)
        world = known.occupied_points(maximum=900)

        def scan_at(pose, timestamp):
            points = pose.inverse().transform_points(world)
            ranges = np.linalg.norm(points, axis=1)
            points = points[(ranges > 0.12) & (ranges < 4.0)]
            return LaserScan(
                np.arctan2(points[:, 1], points[:, 0]),
                np.linalg.norm(points, axis=1),
                timestamp,
            )

        localizer = ScanMapLocalizer(known, lidar_offset_x_m=0.0)
        match = localizer.relative_motion(
            scan_at(first_pose, 1.0), scan_at(second_pose, 2.0)
        )
        self.assertTrue(match.converged)
        self.assertLess(
            math.hypot(
                match.transform.x_m - delta.x_m,
                match.transform.y_m - delta.y_m,
            ),
            0.08,
        )
        self.assertLess(
            abs(normalize_angle(match.transform.yaw_rad - delta.yaw_rad)),
            math.radians(4.0),
        )

    def test_consistency_requires_three_agreeing_results(self):
        from raspberry_pi.known_map_navigation import LocalizationResult
        values = [
            LocalizationResult(Pose2D(1.0, 2.0, 0.1), 0.05, 0.8, 0.7, 0.1),
            LocalizationResult(Pose2D(1.02, 1.98, 0.11), 0.06, 0.78, 0.68, 0.1),
            LocalizationResult(Pose2D(0.99, 2.01, 0.09), 0.04, 0.82, 0.72, 0.1),
        ]
        self.assertIsNotNone(consistent_localization(values))
        self.assertIsNone(consistent_localization(values[:2]))


class SemanticMapTests(unittest.TestCase):
    def test_inspection_report_recovers_if_run_directory_disappears(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        targets = known.inspection_targets(standoff_m=0.60, maximum_targets=2)
        self.assertTrue(targets)
        with tempfile.TemporaryDirectory() as directory:
            report = InspectionReport(directory, "map.pgm")
            shutil.rmtree(report.directory)
            report.save_plan(
                known,
                Pose2D(targets[0].view_xy_m[0], targets[0].view_xy_m[1], 0.0),
                [(targets[0], [targets[0].view_xy_m])],
            )
            report.save()

            self.assertTrue((report.directory / "inspection_plan.png").is_file())
            manifest = json.loads(
                (report.directory / "inspection_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                "inspection_output_directory_disappeared",
                manifest["storage_recovery_events"][0]["reason"],
            )

    def test_semantics_persist_for_same_map_and_reset_for_rebuilt_map(self):
        known = KnownOccupancyMap(
            np.flipud(_room_image()), 0.05, -3.0, -2.5, clearance_m=0.20
        )
        targets = known.inspection_targets(standoff_m=0.60, maximum_targets=4)
        self.assertTrue(targets)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_semantics.json"
            first = PermanentSemanticMap(path, "map.pgm", "hash-a", targets)
            entity = first.data["entities"][targets[0].target_id]
            entity["semantic_label"] = "table"
            first.save()
            same = PermanentSemanticMap(path, "map.pgm", "hash-a", targets)
            self.assertEqual(
                "table",
                same.data["entities"][targets[0].target_id]["semantic_label"],
            )
            rebuilt = PermanentSemanticMap(path, "map.pgm", "hash-b", targets)
            self.assertNotEqual(
                "table",
                rebuilt.data["entities"][targets[0].target_id]["semantic_label"],
            )

    def test_online_label_propagates_only_within_same_black_component(self):
        targets = [
            InspectionTarget(
                "wall_3_1", "perimeter_wall", (0.0, 0.0),
                (0.5, 0.0), math.pi, 120,
            ),
            InspectionTarget(
                "wall_3_2", "perimeter_wall", (0.0, 0.5),
                (0.5, 0.5), math.pi, 120,
            ),
            InspectionTarget(
                "object_4_1", "interior_black_region", (1.0, 1.0),
                (1.5, 1.0), math.pi, 40,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            semantic = PermanentSemanticMap(
                Path(directory) / "map_semantics.json",
                "map.pgm",
                "hash-a",
                targets,
            )
            semantic.record_online_qwen(
                targets[0],
                {
                    "promoted": True,
                    "candidate_label": "table",
                    "candidate_confidence": 0.91,
                    "required_confirmations": 2,
                },
                Path(directory) / "evidence.jpg",
            )
            self.assertEqual(
                "table", semantic.data["entities"]["wall_3_1"]["semantic_label"]
            )
            self.assertEqual(
                "table", semantic.data["entities"]["wall_3_2"]["semantic_label"]
            )
            self.assertEqual(
                "entire_connected_black_component",
                semantic.data["entities"]["wall_3_2"]["semantic_scope"],
            )
            self.assertEqual(
                "unclassified_permanent_obstacle",
                semantic.data["entities"]["object_4_1"]["semantic_label"],
            )

    def test_classified_interior_component_is_skipped_on_next_mission(self):
        targets = [
            InspectionTarget(
                "object_4_1", "interior_black_region", (1.0, 1.0),
                (1.5, 1.0), math.pi, 40,
            ),
            InspectionTarget(
                "object_5_1", "interior_black_region", (2.0, 1.0),
                (2.5, 1.0), math.pi, 50,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            semantic = PermanentSemanticMap(
                Path(directory) / "map_semantics.json",
                "map.pgm", "hash-a", targets,
            )
            semantic.data["entities"]["object_4_1"][
                "semantic_label"
            ] = "cabinet"
            semantic.data["entities"]["object_4_1"]["online_qwen"] = {
                "promoted": True,
            }
            # A legacy/offline label is not strong enough to skip a mission
            # target; it must still receive strict online confirmation.
            semantic.data["entities"]["object_5_1"][
                "semantic_label"
            ] = "chair"
            remaining, skipped = _skip_classified_interior_targets(
                targets, semantic
            )
            self.assertEqual({"object_4"}, skipped)
            self.assertEqual(
                ["object_5_1"],
                [target.target_id for target in remaining],
            )


class ActiveLocalizationTests(unittest.TestCase):
    class _Args:
        localization_window = 5
        localization_min_accepted = 3
        localization_acceptance_ratio = 0.60
        localization_min_margin = 0.025
        localization_motion_min_distance = 0.07
        localization_motion_min_yaw_deg = 8.0
        localization_motion_position_tolerance = 0.24
        localization_motion_yaw_tolerance_deg = 16.0
        localization_fast_accept_margin = 0.15
        localization_fast_min_inlier_ratio = 0.70
        localization_fast_min_range_agreement = 0.75
        localization_fast_max_range_error = 0.20
        localization_min_range_coverage = 0.40
        relaxed_tracking_rmse = 0.28
        relaxed_tracking_inlier_ratio = 0.28

    @staticmethod
    def _accepted(x_m=1.0):
        from raspberry_pi.known_map_navigation import LocalizationResult
        return LocalizationResult(
            Pose2D(x_m, 2.0, 0.1),
            0.05,
            0.75,
            0.70,
            0.08,
        )

    def test_acceptance_ratio_starts_as_soon_as_three_good_frames_agree(self):
        ready, localization, ratio, accepted, total = _localization_acceptance(
            [self._accepted(), self._accepted(1.01), self._accepted(0.99)],
            self._Args(),
        )
        self.assertTrue(ready)
        self.assertIsNotNone(localization)
        self.assertEqual((3, 3), (accepted, total))
        self.assertEqual(1.0, ratio)

    def test_unique_strong_radar_solution_can_skip_forced_motion(self):
        result = self._accepted()
        result = type(result)(
            result.pose, 0.04, 0.77, 0.8, math.inf
        )
        evidence = {
            "range_agreement": 0.91,
            "range_mean_error_m": 0.104,
            "range_coverage": 0.99,
        }
        self.assertTrue(
            _fast_unique_radar_localization(result, evidence, self._Args())
        )
        ambiguous = type(result)(
            result.pose, 0.04, 0.77, 0.8, 0.08
        )
        self.assertFalse(
            _fast_unique_radar_localization(ambiguous, evidence, self._Args())
        )

    def test_replays_field_log_as_fast_localization_success(self):
        from raspberry_pi.known_map_navigation import LocalizationResult

        class StrictArgs(self._Args):
            localization_min_accepted = 4
            localization_acceptance_ratio = 0.80
            localization_min_margin = 0.060

        values = [
            LocalizationResult(Pose2D(-1.49, -0.25, math.radians(-96)), 0.046, 0.75, 0.8, math.inf),
            LocalizationResult(Pose2D(-1.41, -0.25, math.radians(-98)), 0.044, 0.74, 0.8, math.inf),
            LocalizationResult(Pose2D(-1.44, -0.19, math.radians(-97)), 0.052, 0.76, 0.8, math.inf),
            LocalizationResult(Pose2D(-1.44, -0.23, math.radians(-97)), 0.047, 0.77, 0.8, math.inf),
        ]
        ready, candidate, ratio, accepted, total = _localization_acceptance(
            values, StrictArgs()
        )
        self.assertTrue(ready)
        self.assertEqual((accepted, total, ratio), (4, 4, 1.0))
        self.assertTrue(_fast_unique_radar_localization(
            candidate,
            {
                "range_agreement": 0.91,
                "range_mean_error_m": 0.104,
                "range_coverage": 0.99,
            },
            StrictArgs(),
        ))

    def test_strict_ready_window_does_not_need_second_fast_threshold(self):
        result = self._accepted()
        result = type(result)(result.pose, 0.08, 0.65, 0.7, 0.07)
        evidence = {
            "range_agreement": 0.60,
            "range_mean_error_m": 0.22,
            "range_coverage": 0.80,
        }

        class AdaptiveArgs(self._Args):
            localization_required_moves = 0

        self.assertFalse(
            _fast_unique_radar_localization(result, evidence, AdaptiveArgs())
        )
        self.assertTrue(
            _radar_window_can_finish_without_motion(
                result, evidence, AdaptiveArgs()
            )
        )

    def test_motion_prediction_rejects_cross_room_pose_jump(self):
        predicted = Pose2D(-1.44, -0.40, math.radians(-100.6))
        nearby = self._accepted()
        nearby = type(nearby)(
            Pose2D(-1.40, -0.42, math.radians(-99.0)),
            0.04, 0.75, 0.8, 0.2,
        )
        accepted, position_error, yaw_error = _pose_matches_motion_prediction(
            nearby, predicted, self._Args()
        )
        self.assertTrue(accepted)
        self.assertLess(position_error, 0.05)
        jump = type(nearby)(
            Pose2D(-0.71, -3.22, math.radians(-6.0)),
            0.06, 0.79, 0.8, 0.3,
        )
        rejected, position_error, yaw_error = _pose_matches_motion_prediction(
            jump, predicted, self._Args()
        )
        self.assertFalse(rejected)
        self.assertGreater(position_error, 2.0)
        self.assertGreater(yaw_error, math.radians(80.0))

    def test_low_acceptance_ratio_keeps_active_localization_running(self):
        ready, localization, ratio, accepted, total = _localization_acceptance(
            [self._accepted(), None, None, self._accepted(1.01), None],
            self._Args(),
        )
        self.assertFalse(ready)
        self.assertIsNone(localization)
        self.assertEqual((2, 5), (accepted, total))
        self.assertAlmostEqual(0.4, ratio)

    def test_navigation_uses_every_finite_pose_without_quality_veto(self):
        from raspberry_pi.known_map_navigation import LocalizationResult

        result = LocalizationResult(
            Pose2D(1.2, -0.4, math.radians(35.0)),
            0.45,
            0.10,
            0.10,
            0.0,
        )
        self.assertTrue(_tracking_pose_available(result))
        update = _tracking_update(Pose2D(), result, 0.5)
        self.assertTrue(update.accepted)
        self.assertEqual("", update.rejection_reason)

        invalid = LocalizationResult(
            Pose2D(math.nan, 0.0, 0.0),
            0.01,
            1.0,
            1.0,
            1.0,
        )
        self.assertFalse(_tracking_pose_available(invalid))

    def test_navigation_controller_uses_mission_tracking_envelope(self):
        args = build_argument_parser().parse_args([
            "--relaxed-tracking-rmse", "0.28",
            "--relaxed-tracking-inlier-ratio", "0.28",
        ])
        startup = _build_speed_controller(args)
        navigation = _build_speed_controller(args, navigation=True)

        self.assertAlmostEqual(0.09, startup.config.poor_rmse_m)
        self.assertAlmostEqual(0.65, startup.config.poor_inlier_ratio)
        self.assertAlmostEqual(0.28, navigation.config.poor_rmse_m)
        self.assertAlmostEqual(
            0.28, navigation.config.poor_inlier_ratio
        )

    def test_navigation_drives_a_forward_arc_for_moderate_heading_error(self):
        args = build_argument_parser().parse_args([
            "--speed", "320", "--slow-speed", "220"
        ])
        angle = math.radians(35.0)
        linear, angular, reached = _command_for_pose(
            args,
            Pose2D(0.0, 0.0, 0.0),
            (2.0 * math.cos(angle), 2.0 * math.sin(angle)),
        )
        self.assertFalse(reached)
        self.assertEqual(320, linear)
        self.assertGreater(angular, 0)

    def test_stalled_wide_turn_can_be_retried_as_forward_arc(self):
        args = build_argument_parser().parse_args([
            "--speed", "320", "--slow-speed", "220"
        ])
        angle = math.radians(110.0)
        waypoint = (2.0 * math.cos(angle), 2.0 * math.sin(angle))
        linear, angular, reached = _command_for_pose(
            args, Pose2D(), waypoint
        )
        self.assertEqual(0, linear)
        self.assertGreater(angular, 0)
        self.assertFalse(reached)

        linear, angular, reached = _command_for_pose(
            args, Pose2D(), waypoint, force_forward_arc=True
        )
        self.assertEqual(220, linear)
        self.assertEqual(700, angular)
        self.assertFalse(reached)

    def test_navigation_forward_guard_ignores_safe_corridor_side_walls(self):
        scan = LaserScan(
            np.radians(np.asarray((-25.0, 0.0, 25.0))),
            np.asarray((0.45, 2.0, 0.45)),
            time.monotonic(),
        )
        self.assertFalse(_obstacle_safe(scan, 300, 0.42))
        self.assertTrue(_obstacle_safe(scan, 300, 0.42, 0))

    def test_near_target_limit_precedes_start_kick_and_then_cruises_slowly(self):
        args = build_argument_parser().parse_args([
            "--speed", "260",
            "--max-drive-speed", "360",
            "--start-kick-speed", "360",
            "--slow-speed", "170",
        ])
        controller = _build_speed_controller(args, navigation=True)
        requested_speed = _limit_navigation_linear_request(
            args.speed,
            0.95,
            0.40,
            True,
            args.slow_speed,
            args.navigation_fine_control_distance,
            controller.diagnostics()["linear_motion_confirmed"],
        )
        requested = MotionCommand(
            requested_speed, 0, "advancing", "near-target test"
        )
        applied = controller.apply(requested)

        # The feedback controller records exactly the command the motor gets:
        # a slow request plus one bounded static-friction kick.
        self.assertEqual(170, controller.last_requested_command.linear_mm_s)
        self.assertEqual(360, applied.linear_mm_s)
        self.assertEqual(applied, controller.last_command)
        self.assertEqual(
            1, controller.config.linear_progress_confirmation_frames
        )

        from raspberry_pi.map_camera_inspection import _motion_update
        controller.observe(_motion_update(
            MatchResult(
                Pose2D(0.10, 0.0, 0.0), 0.02, 120, True, 0.90
            ),
            1.0,
        ))
        self.assertTrue(
            controller.diagnostics()["linear_motion_confirmed"]
        )

        requested_speed = _limit_navigation_linear_request(
            args.speed,
            0.95,
            0.30,
            True,
            args.slow_speed,
            args.navigation_fine_control_distance,
            True,
        )
        applied = controller.apply(MotionCommand(
            requested_speed, 0, "advancing", "near-target test"
        ))
        self.assertEqual(170, applied.linear_mm_s)
        self.assertEqual(applied, controller.last_command)

    def test_navigation_start_kick_is_not_repeated_when_pose_is_held(self):
        args = build_argument_parser().parse_args([
            "--speed", "330",
            "--max-drive-speed", "450",
            "--start-kick-speed", "420",
            "--slow-speed", "240",
        ])
        controller = _build_speed_controller(args, navigation=True)
        requested = MotionCommand(240, -300, "advancing", "test")

        first = controller.apply(requested)
        controller.linear_adjustment_mm_s = 150
        second = controller.apply(requested)

        self.assertEqual(420, first.linear_mm_s)
        self.assertEqual(240, second.linear_mm_s)
        self.assertEqual(
            1, controller.diagnostics()["linear_start_kick_frames"]
        )

    def test_navigation_confirmed_stall_rearms_one_stronger_kick(self):
        args = build_argument_parser().parse_args([
            "--speed", "330",
            "--max-drive-speed", "450",
            "--start-kick-speed", "420",
            "--slow-speed", "240",
            "--drive-speed-step", "30",
        ])
        controller = _build_speed_controller(args, navigation=True)
        requested = MotionCommand(240, -300, "advancing", "test")

        outputs = []
        moved = False
        for _frame in range(14):
            applied = controller.apply(requested)
            outputs.append(applied.linear_mm_s)
            # Reproduce the 20260905 field dead zone: 420 and below do not
            # move this loaded chassis, while a bounded 450 pulse does.
            translation_m = 0.050 if applied.linear_mm_s >= 450 else 0.001
            controller.observe(_tracking_update(
                Pose2D(),
                LocalizationResult(
                    Pose2D(translation_m, 0.0, 0.0),
                    0.03,
                    0.90,
                    math.inf,
                ),
                0.45,
            ))
            if translation_m >= 0.02:
                moved = True
                break

        self.assertTrue(moved, outputs)
        self.assertEqual(450, outputs[-1])
        self.assertLessEqual(outputs.count(450), 1)
        following = controller.apply(requested)
        self.assertEqual(240, following.linear_mm_s)

    def test_planned_stop_does_not_reuse_stall_boost_without_new_evidence(self):
        args = build_argument_parser().parse_args([
            "--max-drive-speed", "450",
            "--start-kick-speed", "420",
            "--slow-speed", "240",
        ])
        controller = _build_speed_controller(args, navigation=True)
        controller.linear_adjustment_mm_s = 210
        controller._linear_stall_events = 4
        controller.stopped()

        first = controller.apply(MotionCommand(
            240, 0, "advancing", "new waypoint"
        ))
        second = controller.apply(MotionCommand(
            240, 0, "advancing", "new waypoint"
        ))
        self.assertEqual(420, first.linear_mm_s)
        self.assertEqual(240, second.linear_mm_s)

    def test_inspection_subject_can_finish_translation_near_final_view(self):
        pose = Pose2D(-0.73, 1.472, math.radians(151.5))
        waypoint = (-0.925, 1.525)
        self.assertTrue(_inspection_standoff_reached(
            pose, waypoint, 0.432, 0.18, 0.42
        ))
        self.assertFalse(_inspection_standoff_reached(
            Pose2D(-0.465, 1.461, 0.0),
            waypoint,
            0.652,
            0.18,
            0.42,
        ))
        self.assertFalse(_inspection_standoff_reached(
            pose, waypoint, 0.35, 0.18, 0.42
        ))
        self.assertTrue(_inspection_standoff_reached(
            Pose2D(-0.53, 1.472, 0.0),
            waypoint,
            0.60,
            0.18,
            0.42,
            0.45,
        ))

    def test_navigation_turn_holds_position_but_keeps_measured_yaw(self):
        from raspberry_pi.known_map_navigation import LocalizationResult

        previous = Pose2D(-0.94, 1.28, math.radians(8.9))
        result = LocalizationResult(
            Pose2D(-0.76, 1.02, math.radians(-52.9)),
            0.057,
            0.61,
            0.5,
        )
        stabilized, anchor, suppressed = _stabilize_navigation_turn_pose(
            previous, result, (0, 1500), None
        )
        self.assertEqual(previous.x_m, stabilized.pose.x_m)
        self.assertEqual(previous.y_m, stabilized.pose.y_m)
        self.assertEqual(result.pose.yaw_rad, stabilized.pose.yaw_rad)
        self.assertEqual(previous, anchor)
        self.assertGreater(suppressed, 0.30)

        advancing, anchor, suppressed = _stabilize_navigation_turn_pose(
            previous, result, (400, 0), previous
        )
        self.assertIs(result, advancing)
        self.assertIsNone(anchor)
        self.assertEqual(0.0, suppressed)

    def test_all_map_targets_are_kept_by_default(self):
        args = build_argument_parser().parse_args([])
        self.assertEqual(0.0, args.min_target_spacing)
        self.assertEqual(1, args.max_target_navigation_attempts)
        self.assertEqual(0.45, args.nearby_capture_distance)
        self.assertEqual(8, args.navigation_turn_stall_at_max_frames)
        self.assertEqual(1, args.navigation_turn_escape_max_attempts)

    @staticmethod
    def _motion_scan(x_m=0.0, yaw_rad=0.0, clearance_m=1.0):
        scan = LaserScan(
            np.linspace(-math.pi, math.pi, 180, endpoint=False),
            np.full(180, clearance_m),
            time.monotonic(),
        )
        # Test-only physical coordinates used by the fake relative matcher.
        object.__setattr__(scan, "test_x_m", float(x_m))
        object.__setattr__(scan, "test_yaw_rad", float(yaw_rad))
        return scan

    @staticmethod
    def _motion_match(before, after):
        return MatchResult(
            Pose2D(
                after.test_x_m - before.test_x_m,
                0.0,
                normalize_angle(
                    after.test_yaw_rad - before.test_yaw_rad
                ),
            ),
            0.02,
            120,
            True,
            0.90,
        )

    def test_active_turn_adapts_to_max_then_declares_stall(self):
        controller = AdaptiveSpeedController(
            2600,
            460,
            AdaptiveSpeedConfig(
                initial_turn_limit_mrad_s=2600,
                turn_adjust_step_mrad_s=300,
            ),
        )
        requested = MotionCommand(
            0, 1500, "active_localization_turn", "test"
        )
        still = MatchResult(Pose2D(), 0.02, 120, True, 0.90)
        sequence = []
        for _index in range(8):
            applied = controller.apply(requested)
            sequence.append(applied.angular_mrad_s)
            from raspberry_pi.map_camera_inspection import _motion_update
            controller.observe(_motion_update(still, 1.0))

        self.assertEqual(
            [1500, 1500, 1500, 1800, 1800, 2400, 2600, 2600],
            sequence,
        )
        diagnostics = controller.diagnostics()
        self.assertTrue(_is_turn_motion_stalled(
            applied, still, controller.config
        ))
        self.assertTrue(_turn_is_at_adaptive_max(
            applied, diagnostics
        ))
        self.assertGreaterEqual(diagnostics["turn_stall_events"], 3)

    def test_turn_escape_requires_explicit_forward_clearance(self):
        args = build_argument_parser().parse_args([])
        safe, front, required = _turn_escape_is_safe(
            self._motion_scan(clearance_m=0.90), args
        )
        self.assertTrue(safe)
        self.assertGreater(front, required)

        safe, front, required = _turn_escape_is_safe(
            self._motion_scan(clearance_m=0.37), args
        )
        self.assertFalse(safe)
        self.assertLess(front, required)

        unsupported = LaserScan(
            np.asarray((math.pi / 2.0, -math.pi / 2.0)),
            np.asarray((1.0, 1.0)),
            time.monotonic(),
        )
        safe, front, _required = _turn_escape_is_safe(
            unsupported, args
        )
        self.assertFalse(safe)
        self.assertTrue(math.isinf(front))

    def test_navigation_turn_stall_requires_max_output_and_no_yaw(self):
        args = build_argument_parser().parse_args([])
        controller = _build_speed_controller(args, navigation=True)
        maximum = controller.diagnostics()["maximum_turn_speed_mrad_s"]
        controller.turn_floor_mrad_s = maximum
        controller.turn_limit_mrad_s = maximum
        command = MotionCommand(0, maximum, "turning", "test")
        diagnostics = controller.diagnostics()

        self.assertTrue(_navigation_turn_stall_observed(
            (0, maximum),
            math.radians(0.2),
            command,
            diagnostics,
            controller.config,
        ))
        self.assertFalse(_navigation_turn_stall_observed(
            (0, maximum),
            math.radians(4.0),
            command,
            diagnostics,
            controller.config,
        ))
        self.assertFalse(_navigation_turn_stall_observed(
            (400, 0),
            0.0,
            command,
            diagnostics,
            controller.config,
        ))

    def test_navigation_linear_stall_requires_max_output_and_scan_no_motion(self):
        args = build_argument_parser().parse_args([])
        controller = _build_speed_controller(args, navigation=True)
        stopped_feedback = SlamUpdate(
            Pose2D(), True, 0.03, 100, 100,
            inlier_ratio=0.9, translation_m=0.002,
        )
        moving_feedback = SlamUpdate(
            Pose2D(), True, 0.03, 100, 100,
            inlier_ratio=0.9, translation_m=0.03,
        )
        maximum = controller.maximum_linear_speed_mm_s
        command = MotionCommand(maximum, 600, "advancing", "test")

        self.assertTrue(_navigation_linear_stall_observed(
            (maximum, 600), stopped_feedback, command,
            controller.diagnostics(), maximum,
            controller.config,
        ))
        self.assertFalse(_navigation_linear_stall_observed(
            (maximum, 600), moving_feedback, command,
            controller.diagnostics(), maximum,
            controller.config,
        ))
        self.assertFalse(_navigation_linear_stall_observed(
            (maximum - 80, 600), stopped_feedback,
            MotionCommand(maximum - 80, 600, "advancing", "test"),
            controller.diagnostics(), maximum, controller.config,
        ))
        controller.linear_adjustment_mm_s = (
            controller.config.maximum_linear_boost_mm_s
        )
        self.assertTrue(_navigation_linear_stall_observed(
            (240, 600), stopped_feedback,
            MotionCommand(240, 600, "advancing", "test"),
            controller.diagnostics(), maximum, controller.config,
        ))
        self.assertFalse(_navigation_linear_stall_observed(
            (0, 3000), stopped_feedback, command,
            controller.diagnostics(), maximum,
            controller.config,
        ))

    def test_collision_reverse_requires_measured_body_motion(self):
        args = build_argument_parser().parse_args([
            "--collision-reverse-distance", "0.06",
            "--collision-reverse-speed", "240",
            "--collision-reverse-rear-clearance", "0.25",
            "--collision-reverse-max-steps", "2",
        ])

        class Client:
            def __init__(self):
                self.stop_count = 0

            def stop(self):
                self.stop_count += 1

        class Map:
            @staticmethod
            def line_is_clear(_start, _end):
                return True

        class Localizer:
            relative_motion = staticmethod(
                ActiveLocalizationTests._motion_match
            )

        initial = self._motion_scan(x_m=0.0)
        with patch(
            "raspberry_pi.map_camera_inspection._timed_twist"
        ):
            _scan, moved, distance, reason = (
                _execute_collision_reverse_escape(
                    Client(), iter((self._motion_scan(x_m=0.0),)),
                    initial, Pose2D(), Map(), args, Localizer(),
                )
            )
        self.assertFalse(moved)
        self.assertEqual(0.0, distance)
        self.assertEqual("reverse_no_measured_motion", reason)

        with patch(
            "raspberry_pi.map_camera_inspection._timed_twist"
        ):
            _scan, moved, distance, reason = (
                _execute_collision_reverse_escape(
                    Client(), iter((
                        self._motion_scan(x_m=-0.03),
                        self._motion_scan(x_m=-0.06),
                    )), initial, Pose2D(), Map(), args, Localizer(),
                )
            )
        self.assertTrue(moved)
        self.assertGreaterEqual(distance, 0.045)
        self.assertEqual("", reason)

    def test_turn_escape_uses_controller_and_stops_after_real_translation(self):
        args = build_argument_parser().parse_args([])
        controller = AdaptiveSpeedController(
            2600,
            460,
            AdaptiveSpeedConfig(linear_start_kick_mm_s=400),
        )

        class Client:
            def __init__(self):
                self.stop_count = 0

            def stop(self):
                self.stop_count += 1

        class Localizer:
            relative_motion = staticmethod(
                ActiveLocalizationTests._motion_match
            )

        client = Client()
        current_scan = self._motion_scan()
        applied_commands = []

        def measured_step(requested):
            nonlocal current_scan
            applied = controller.apply(requested)
            applied_commands.append(applied)
            next_scan = self._motion_scan(
                x_m=current_scan.test_x_m + 0.03
            )
            match = self._motion_match(current_scan, next_scan)
            current_scan = next_scan
            return applied, match, next_scan

        _scan, succeeded, distance, _reason = (
            _execute_localization_turn_escape(
                client,
                Localizer(),
                controller,
                current_scan,
                args,
                time.monotonic() + 5.0,
                measured_step,
            )
        )

        self.assertTrue(succeeded)
        self.assertGreaterEqual(distance, 0.05)
        self.assertEqual(
            {"turn_escape"}, {item.state for item in applied_commands}
        )
        self.assertTrue(all(
            item.linear_mm_s >= args.start_kick_speed
            for item in applied_commands
        ))
        self.assertGreaterEqual(client.stop_count, 1)

    def test_active_turn_escape_retries_turn_after_wheels_roll(self):
        args = build_argument_parser().parse_args([])
        controller = AdaptiveSpeedController(
            2600,
            460,
            AdaptiveSpeedConfig(
                initial_turn_limit_mrad_s=2600,
                turn_adjust_step_mrad_s=300,
                linear_start_kick_mm_s=400,
            ),
        )

        class Client:
            def __init__(self):
                self.stop_count = 0

            def stop(self):
                self.stop_count += 1

        class Localizer:
            relative_motion = staticmethod(
                ActiveLocalizationTests._motion_match
            )

        client = Client()
        current_scan = self._motion_scan()
        states = []
        escaped = False

        def measured_step(requested):
            nonlocal current_scan, escaped
            applied = controller.apply(requested)
            states.append(requested.state)
            if requested.state == "turn_escape":
                next_scan = self._motion_scan(
                    x_m=current_scan.test_x_m + 0.03,
                    yaw_rad=current_scan.test_yaw_rad,
                )
                if next_scan.test_x_m >= 0.05:
                    escaped = True
            elif escaped:
                next_scan = self._motion_scan(
                    x_m=current_scan.test_x_m,
                    yaw_rad=current_scan.test_yaw_rad
                    + math.radians(8.0),
                )
            else:
                next_scan = self._motion_scan(
                    x_m=current_scan.test_x_m,
                    yaw_rad=current_scan.test_yaw_rad,
                )
            match = self._motion_match(current_scan, next_scan)
            from raspberry_pi.map_camera_inspection import _motion_update
            controller.observe(_motion_update(match, 1.0))
            current_scan = next_scan
            return applied, match, next_scan

        final_scan = _execute_active_localization_turn(
            client,
            Localizer(),
            controller,
            current_scan,
            -args.relocalization_turn_speed,
            "test blocked turn",
            math.radians(10.0),
            args,
            time.monotonic() + 5.0,
            measured_step,
        )

        self.assertIn("turn_escape", states)
        self.assertEqual("active_localization_turn", states[-1])
        self.assertGreaterEqual(final_scan.test_x_m, 0.05)
        self.assertGreaterEqual(
            abs(final_scan.test_yaw_rad), math.radians(7.0)
        )
        self.assertGreaterEqual(client.stop_count, 2)

    def test_unsafe_turn_escape_is_bounded_and_always_stops(self):
        args = build_argument_parser().parse_args([
            "--relocalization-turn-escape-max-attempts", "1",
        ])
        controller = AdaptiveSpeedController(
            2600,
            460,
            AdaptiveSpeedConfig(
                initial_turn_limit_mrad_s=2600,
                turn_adjust_step_mrad_s=300,
            ),
        )

        class Client:
            def __init__(self):
                self.stop_count = 0

            def stop(self):
                self.stop_count += 1

        class Localizer:
            relative_motion = staticmethod(
                ActiveLocalizationTests._motion_match
            )

        client = Client()
        current_scan = self._motion_scan(clearance_m=0.37)
        turn_directions = []

        def measured_step(requested):
            nonlocal current_scan
            applied = controller.apply(requested)
            turn_directions.append(int(math.copysign(
                1, requested.angular_mrad_s
            )))
            next_scan = self._motion_scan(clearance_m=0.37)
            match = self._motion_match(current_scan, next_scan)
            from raspberry_pi.map_camera_inspection import _motion_update
            controller.observe(_motion_update(match, 1.0))
            current_scan = next_scan
            return applied, match, next_scan

        final_scan = _execute_active_localization_turn(
            client,
            Localizer(),
            controller,
            current_scan,
            args.relocalization_turn_speed,
            "test unsafe escape",
            math.radians(10.0),
            args,
            time.monotonic() + 5.0,
            measured_step,
        )

        self.assertIn(1, turn_directions)
        self.assertIn(-1, turn_directions)
        self.assertGreaterEqual(client.stop_count, 3)
        self.assertIs(final_scan, current_scan)

    def test_dominant_pose_cluster_counts_before_single_frame_margin(self):
        from raspberry_pi.known_map_navigation import LocalizationResult

        class StrictArgs(self._Args):
            localization_min_accepted = 4
            localization_acceptance_ratio = 0.80
            localization_window = 5
            localization_min_margin = 0.060

        values = [
            LocalizationResult(Pose2D(-2.255, -2.075, math.radians(-148)), 0.058, 0.80, 0.70, 0.011),
            LocalizationResult(Pose2D(-2.295, -2.055, math.radians(-148)), 0.047, 0.79, 0.72, 0.035),
            LocalizationResult(Pose2D(1.305, -0.515, math.radians(35)), 0.052, 0.81, 0.69, 0.005),
            LocalizationResult(Pose2D(-2.295, -2.035, math.radians(-148)), 0.054, 0.79, 0.75, 0.426),
            LocalizationResult(Pose2D(-2.255, -2.015, math.radians(-152)), 0.066, 0.82, 0.71, 0.056),
        ]
        ready, candidate, ratio, accepted, total = _localization_acceptance(
            values, StrictArgs()
        )
        self.assertTrue(ready)
        self.assertEqual((accepted, total, ratio), (4, 5, 0.8))
        self.assertAlmostEqual(candidate.pose.x_m, -2.275, places=2)
        self.assertLess(candidate.pose.y_m, -2.0)

    def test_two_competing_pose_clusters_do_not_pass_eighty_percent(self):
        from raspberry_pi.known_map_navigation import LocalizationResult

        class StrictArgs(self._Args):
            localization_min_accepted = 4
            localization_acceptance_ratio = 0.80
            localization_window = 5

        left = self._accepted(1.0)
        right = LocalizationResult(Pose2D(-1.5, -1.0, 2.0), 0.05, 0.8, 0.7, 0.01)
        ready, _candidate, ratio, accepted, total = _localization_acceptance(
            [left, left, right, right, left], StrictArgs()
        )
        self.assertFalse(ready)
        self.assertEqual((accepted, total), (3, 5))
        self.assertAlmostEqual(ratio, 0.6)

    def test_scan_profile_change_detects_motion(self):
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        before = LaserScan(angles, np.full(180, 2.0), 1.0)
        still = LaserScan(angles, np.full(180, 2.005), 1.1)
        moved = LaserScan(
            angles,
            2.0 + 0.12 * np.cos(angles),
            1.2,
        )
        self.assertLess(_scan_change_m(before, still), 0.025)
        self.assertGreater(_scan_change_m(before, moved), 0.025)

    def test_cross_position_check_rejects_stable_wrong_map_jump(self):
        previous = self._accepted(1.0)
        current = self._accepted(1.2)
        current = type(current)(
            Pose2D(1.2, 2.0, 0.1), 0.05, 0.75, 0.70, 0.08
        )
        measured = MatchResult(Pose2D(0.20, 0.0, 0.0), 0.03, 100, True, 0.8)
        valid, _position_error, _yaw_error, distance = (
            _localization_motion_consistent(
                previous, current, measured, self._Args()
            )
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(0.20, distance)
        wrong = type(current)(
            Pose2D(-1.0, 3.0, 1.5), 0.05, 0.75, 0.70, 0.08
        )
        self.assertFalse(_localization_motion_consistent(
            previous, wrong, measured, self._Args()
        )[0])

    def test_cross_position_check_accepts_consistent_rotation_without_translation(self):
        previous = self._accepted(1.0)
        yaw_step = math.radians(12.0)
        current = type(previous)(
            Pose2D(1.0, 2.0, previous.pose.yaw_rad + yaw_step),
            0.04,
            0.82,
            0.75,
            0.10,
        )
        measured = MatchResult(
            Pose2D(0.0, 0.0, yaw_step), 0.03, 120, True, 0.82
        )
        valid, position_error, yaw_error, distance = (
            _localization_motion_consistent(
                previous, current, measured, self._Args()
            )
        )
        self.assertTrue(valid)
        self.assertAlmostEqual(0.0, distance)
        self.assertLess(position_error, 0.01)
        self.assertLess(yaw_error, math.radians(0.5))

    def test_trajectory_hint_uses_first_not_last_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            map_path = Path(directory) / "scene.pgm"
            trajectory = Path(directory) / "scene_trajectory.csv"
            trajectory.write_text(
                "timestamp_s,x_m,y_m,yaw_rad\n"
                "1,0.0,0.0,0.0\n2,1.25,-2.14,0.65\n",
                encoding="utf-8",
            )
            pose = _first_trajectory_pose(map_path)
            self.assertEqual(Pose2D(0.0, 0.0, 0.0), pose)

    def test_photo_mission_accepts_coarse_but_bounded_tracking(self):
        from raspberry_pi.known_map_navigation import LocalizationResult
        coarse = LocalizationResult(
            Pose2D(1.0, 2.0, 0.1), 0.22, 0.32, 0.2, 0.0
        )
        self.assertFalse(coarse.reliable)
        self.assertTrue(
            _tracking_usable_for_photo_mission(coarse, self._Args())
        )
        lost = LocalizationResult(
            Pose2D(1.0, 2.0, 0.1), 0.40, 0.12, 0.1, 0.0
        )
        self.assertFalse(
            _tracking_usable_for_photo_mission(lost, self._Args())
        )


class CameraSelectionTests(unittest.TestCase):
    def test_usb_nodes_make_auto_mode_try_opencv_before_rpicam(self):
        camera = CameraCapture("auto", 0, 640, 480, 2.0)
        with patch(
            "raspberry_pi.map_camera_inspection.glob.glob",
            return_value=["/dev/video0", "/dev/video1"],
        ), patch(
            "raspberry_pi.map_camera_inspection.shutil.which",
            side_effect=lambda command: "/usr/bin/" + command,
        ):
            self.assertEqual(
                ["opencv", "rpicam", "libcamera"],
                camera._candidate_backends(),
            )

    def test_explicit_backend_does_not_fall_back_silently(self):
        camera = CameraCapture("opencv", 2, 640, 480, 2.0)
        self.assertEqual(["opencv"], camera._candidate_backends())

    def test_dual_camera_selection_prefers_stable_usb_port_paths(self):
        paths = [
            "/dev/v4l/by-path/platform-xhci-hcd.0-usb-video-index0",
            "/dev/v4l/by-path/platform-xhci-hcd.1-usb-video-index0",
        ]
        with patch(
            "raspberry_pi.map_camera_inspection.glob.glob",
            side_effect=lambda pattern: paths if "by-path" in pattern else [],
        ), patch(
            "raspberry_pi.map_camera_inspection.Path.resolve",
            side_effect=[Path("/dev/video0"), Path("/dev/video2")],
        ):
            self.assertEqual(paths, CameraCapture._stable_usb_capture_nodes())

    def test_stable_camera_link_opens_current_numeric_v4l_node(self):
        with patch(
            "raspberry_pi.map_camera_inspection.Path.resolve",
            return_value=Path("/dev/video2"),
        ):
            index, resolved = CameraCapture._opencv_device_index(
                "/dev/v4l/by-path/platform-xhci-hcd.1-usb-video-index0"
            )
        self.assertEqual(2, index)
        self.assertEqual("video2", Path(resolved).name)

    def test_dual_mode_is_opt_in_for_library_compatibility(self):
        single = CameraCapture("opencv", 0, 640, 480, 2.0)
        dual = CameraCapture(
            "opencv", 0, 640, 480, 2.0, dual_usb_cameras=True
        )
        self.assertFalse(single.dual_usb_cameras)
        self.assertTrue(dual.dual_usb_cameras)


if __name__ == "__main__":
    unittest.main()
