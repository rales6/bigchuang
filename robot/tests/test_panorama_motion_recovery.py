"""Field-failure branch regressions; not a whole-scene simulation."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.map_camera_inspection import InspectionReport, build_argument_parser
from raspberry_pi.open_space_panorama import (
    PanoramaConfig, OpenSpaceDirectionSelector, _ReceivedScan, _scan_age_s,
    _advance_assessment, _DriveProgressWindow, _MotionAuditClient,
    _filter_body_returns, run_open_space_panorama,
    _refresh_stopped_scan, _turn_increment,
    _front_blockage_kind, _reverse_escape_assessment, _short_reverse,
    _rotation_clearance, forward_swept_clearance, _local_stop_diagnostic,
)


class PanoramaRecoveryTests(unittest.TestCase):
    def test_field_46ms_ttl_is_deferred_then_fresh_scan_can_drive(self):
        sent, stopped = [], []
        with tempfile.TemporaryDirectory() as folder, patch(
            'raspberry_pi.open_space_panorama.time.monotonic', return_value=10.):
            fake = SimpleNamespace(stop=lambda: stopped.append(True),
                set_bounded_twist=lambda *a, **kw: sent.append((a, kw)))
            audit = _MotionAuditClient(fake, PanoramaConfig(), Path(folder))
            audit.scan = _ReceivedScan(self.scan(10. - .453872572))
            self.assertFalse(audit.set_twist(420, 342, 500))
            self.assertEqual(sent, [])
            self.assertTrue(stopped)
            self.assertEqual(audit.current_reason, 'scan_budget_too_short')
            event = json.loads((Path(folder) / 'panorama_motion.jsonl').read_text().splitlines()[0])
            self.assertEqual(event['available_ttl_ms'], 46)
            audit.scan = _ReceivedScan(self.scan(9.95))
            self.assertTrue(audit.set_twist(420, 342, 500))
            self.assertEqual(len(sent), 1)
            self.assertGreaterEqual(sent[0][1]['ttl_ms'], 100)
            self.assertLessEqual(sent[0][1]['ttl_ms'] / 1000. + .05, .5)

    def test_short_requested_ttl_is_not_padded_past_requested_lease(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = SimpleNamespace(stop=lambda: None,
                set_twist=lambda *_a, **_k: self.fail('short motion must not be sent'))
            audit = _MotionAuditClient(fake, PanoramaConfig(), Path(folder))
            audit.scan = _ReceivedScan(self.scan())
            self.assertFalse(audit.set_twist(0, 2600, 46))

    def test_measured_body_geometry_and_map_floor(self):
        config = PanoramaConfig.from_args(build_argument_parser().parse_args(['--open-space-panorama']))
        config.validate()
        self.assertAlmostEqual(config.swept_half_width_m, .16)
        self.assertAlmostEqual(config.rotation_clearance_m, math.hypot(.12, .11) + .05)
        self.assertAlmostEqual(config.map_clearance_m(.01), .22)
        larger = PanoramaConfig(body_length_m=.40, body_width_m=.30)
        self.assertAlmostEqual(larger.map_clearance_m(.22), .30)
        self.assertAlmostEqual(config.map_clearance_m(.35), .35)
        for config in [PanoramaConfig(body_width_m=.01), PanoramaConfig(footprint_margin_m=.01),
                       PanoramaConfig(body_length_m=float('nan'))]:
            with self.assertRaises(ValueError):
                config.validate()

    def test_narrower_sweep_ignores_only_points_outside_measured_corridor(self):
        config = PanoramaConfig(lidar_offset_x_m=0.)
        def point_scan(y):
            return LaserScan(np.array([0., math.atan2(y, .30)]),
                             np.array([2., math.hypot(.30, y)]), 0.)
        self.assertAlmostEqual(forward_swept_clearance(point_scan(.20)), .30)
        self.assertAlmostEqual(forward_swept_clearance(point_scan(.20), config.swept_half_width_m), 2.)
        self.assertAlmostEqual(forward_swept_clearance(point_scan(.15), config.swept_half_width_m), .30)

    def test_rotation_uses_body_center_and_includes_diagonal_map_obstacle(self):
        config = PanoramaConfig()
        known = KnownOccupancyMap(np.full((100, 100), 255, dtype=np.uint8), .05, -2.5, -2.5, .22)
        safe = self.scan()
        self.assertGreater(_rotation_clearance(safe, known, Pose2D(), config), config.rotation_clearance_m)
        self.assertGreater(_rotation_clearance(safe, known, Pose2D(), config), .40)
        # Rear range .34 is only .20 from the body center with a .14 offset.
        distances = np.full(360, 2.)
        distances[0] = .34
        self.assertLess(_rotation_clearance(self.scan(distances=distances), known, Pose2D(), config),
                        config.rotation_clearance_m)
        distances[0] = .38  # .24 m from center: safe with measured footprint.
        self.assertGreater(_rotation_clearance(self.scan(distances=distances), known, Pose2D(), config),
                           config.rotation_clearance_m)
        col, row = map(int, known.world_to_grid(((.15, .15),))[0])
        known.occupied[row, col] = True
        self.assertLess(_rotation_clearance(safe, known, Pose2D(), config), config.rotation_clearance_m)
        known.occupied[row, col] = False
        known.unknown[row, col] = True
        self.assertLess(_rotation_clearance(safe, known, Pose2D(), config), config.rotation_clearance_m)

    def test_confirmed_body_mask_is_independent_of_collision_dimensions(self):
        scan = LaserScan(np.radians([-103., -101., -100., 0., -90.]),
                         np.array([.115, .098, .098, .15, .25]), 0.)
        filtered = _filter_body_returns(scan, .19, .14, 0., .125)
        self.assertTrue(np.isnan(filtered.distances_m[:3]).all())
        self.assertTrue(np.isfinite(filtered.distances_m[3:]).all())
        # Neither changing collision margin nor dimensions enlarges this mask.
        for config in [PanoramaConfig(), PanoramaConfig(body_length_m=.4, footprint_margin_m=.15)]:
            repeated = _filter_body_returns(scan, .19, config.lidar_offset_x_m, 0., .125)
            np.testing.assert_equal(repeated.distances_m, filtered.distances_m)

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def field_scan(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' /
                              'panorama_body_return_20260917_154320.json').read_text())
        event = fixture['stop_event']
        scan = LaserScan(np.array(event['raw_angles_rad']), np.array(event['raw_distances_m']), 0.)
        pose = fixture['pose']
        return fixture, scan, Pose2D(pose['x_m'], pose['y_m'], math.radians(pose['yaw_deg']))

    def test_real_field_frame_reproduces_old_block_and_allows_fixed_motion(self):
        fixture, raw, pose = self.field_scan()
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml', .22)
        args = build_argument_parser().parse_args(['--open-space-panorama', '--panorama-self-return-radius', '.19'])
        config = PanoramaConfig.from_args(args)
        old = LaserScan(raw.angles_rad, np.array(fixture['stop_event']['filtered_distances_m'], dtype=float), 0.)
        blocked = _advance_assessment(old, pose, known, config)
        self.assertAlmostEqual(blocked['swept_m'], .11413062875045553)
        self.assertFalse(blocked['can_advance'])
        self.assertAlmostEqual(blocked['nearest_swept_point']['sensor_angle_deg'], -103.)
        fixed = _filter_body_returns(raw, .19, .14, 0., .125)
        assessment = _advance_assessment(fixed, pose, known, config, speed_mm_s=420)
        self.assertTrue(assessment['can_advance'], assessment)
        self.assertGreater(_rotation_clearance(fixed, known, pose, config), config.rotation_clearance_m)
        self.assertEqual(_reverse_escape_assessment(fixed, pose, config, known, args)['reason'], 'rear_support_missing')

    def test_real_field_frame_keeps_added_external_obstacle(self):
        _, raw, pose = self.field_scan()
        scan = LaserScan(np.append(raw.angles_rad, 0.), np.append(raw.distances_m, .15), 0.)
        fixed = _filter_body_returns(scan, .19, .14, 0., .125)
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml', .22)
        assessment = _advance_assessment(fixed, pose, known, PanoramaConfig())
        self.assertFalse(assessment['can_advance'])
        self.assertEqual(_local_stop_diagnostic(assessment)[0], 'LIDAR_NEAR_POINT_STOP')
        point = assessment['nearest_swept_point']
        self.assertEqual(point['scan_index'], len(raw.angles_rad))
        self.assertFalse(point['filtered_out'])
        self.assertAlmostEqual(point['body_x_m'], .29)

    def test_production_loop_sends_forward_motion_with_real_field_frame(self):
        _, raw, pose = self.field_scan()
        elapsed, sent = [10.], []
        args = build_argument_parser().parse_args(['--open-space-panorama',
            '--panorama-self-return-radius', '.19', '--clearance', '.22'])
        known = KnownOccupancyMap.load(args.map, args.map_yaml, .22)
        def scans():
            for _ in range(100):
                elapsed[0] += .1
                yield LaserScan(raw.angles_rad, raw.distances_m, elapsed[0])
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch(
                'raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: elapsed[0]), patch(
                'raspberry_pi.open_space_panorama._render_live_map'), patch(
                'raspberry_pi.open_space_panorama.open_space_assessment',
                return_value={'is_open': False, 'minimum_clearance_m': 1., 'open_direction_count': 4}):
            report = InspectionReport(folder, args.map)
            result = run_open_space_panorama(args=args, known_map=known,
                localizer=SimpleNamespace(relative_motion=lambda *_a: MatchResult(Pose2D(), .01, 100, True, inlier_ratio=.95),
                    localize_bounded=lambda *_a, **_k: None),
                initial_pose=pose, initial_scan=raw, scan_iterator=scans(),
                client=SimpleNamespace(stop=lambda: None,
                    set_twist=lambda linear, angular, ttl_ms: sent.append((linear, angular, ttl_ms))),
                camera=SimpleNamespace(), report=report,
                online_qwen=SimpleNamespace(url=''), should_stop=lambda: bool(sent))
            live = json.loads((report.directory / 'live_pose.json').read_text())
        self.assertEqual(result['finish_reason'], 'requested_stop')
        self.assertGreater(sent[0][0], 0)
        self.assertTrue(live['body_return_filter']['independent_of_collision_footprint'])
        self.assertIn('nearest_point', live['front_swept_details'])

    def test_stop_diagnostic_distinguishes_map_and_missing_lidar(self):
        base = {'can_advance': False, 'reason': 'map_short_segment_blocked',
                'lidar_m': 2., 'swept_m': 2., 'stop_m': .61}
        self.assertEqual(_local_stop_diagnostic(base)[0], 'MAP_GUARD_STOP')
        self.assertEqual(_local_stop_diagnostic({**base, 'reason': 'lidar_support_missing'})[0],
                         'WAIT_LIDAR_EVIDENCE')

    def test_front_blockage_distinguishes_live_obstacle_map_and_unknown(self):
        base = {'can_advance': False, 'reason': 'swept_or_distance_blocked',
                'lidar_m': 2., 'swept_m': 2., 'stop_m': .72}
        self.assertEqual(_front_blockage_kind(base), 'map_only')
        self.assertEqual(_front_blockage_kind({**base, 'swept_m': .13}), 'lidar_obstacle')
        self.assertEqual(_front_blockage_kind({**base, 'lidar_m': .49}), 'lidar_obstacle')
        self.assertEqual(_front_blockage_kind({**base, 'swept_m': 0}), 'sensor_unknown')
        self.assertEqual(_front_blockage_kind({**base, 'reason': 'lidar_support_missing'}), 'sensor_unknown')

    def test_reverse_requires_live_rear_and_preserves_mapped_glass(self):
        args = build_argument_parser().parse_args(['--open-space-panorama'])
        known = SimpleNamespace(line_is_clear=lambda *_a: True)
        with patch('raspberry_pi.open_space_panorama._nearest_map_guard_pose', return_value=Pose2D()), patch(
            'raspberry_pi.open_space_panorama._map_ray_clearance', return_value=2.):
            self.assertTrue(_reverse_escape_assessment(self.scan(), Pose2D(), PanoramaConfig(), known, args)['can_reverse'])
            distances = np.full(360, 2.)
            distances[np.cos(self.scan().angles_rad) < 0] = np.nan
            result = _reverse_escape_assessment(self.scan(distances=distances), Pose2D(), PanoramaConfig(), known, args)
            self.assertEqual(result['reason'], 'rear_support_missing')
            distances[:] = 2.
            distances[0] = .3
            self.assertFalse(_reverse_escape_assessment(self.scan(distances=distances), Pose2D(), PanoramaConfig(), known, args)['can_reverse'])
        with patch('raspberry_pi.open_space_panorama._nearest_map_guard_pose', return_value=Pose2D()), patch(
            'raspberry_pi.open_space_panorama._map_ray_clearance', return_value=.2):
            result = _reverse_escape_assessment(self.scan(), Pose2D(), PanoramaConfig(), known, args)
        self.assertEqual(result['reason'], 'rear_map_or_glass_guard')

    def test_short_reverse_does_not_credit_rotation_lateral_or_forward_motion(self):
        args = build_argument_parser().parse_args(['--open-space-panorama'])
        for transform in [Pose2D(0, 0, math.radians(20)), Pose2D(0, .04, 0), Pose2D(.04, 0, 0)]:
            elapsed = [10.]
            def scans():
                for _ in range(10):
                    elapsed[0] += .1
                    yield self.scan(elapsed[0])
            with patch('raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: elapsed[0]), patch(
                'raspberry_pi.open_space_panorama._reverse_escape_assessment', return_value={'can_reverse': True}):
                _, _, distance = _short_reverse(SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: None),
                    scans(), SimpleNamespace(relative_motion=lambda *_a: MatchResult(transform, .01, 100, True, inlier_ratio=.95)),
                    self.scan(10.), Pose2D(), PanoramaConfig(), SimpleNamespace(), args,
                    lambda *_a: None, lambda: False)
            self.assertEqual(distance, 0.)

    def recovery_order_case(self, map_only=False, exhausted_turns=False, reverse_progress=.08, persistent=False, no_direction=False, map_only_reselect=False):
        elapsed, turns, reverse, matches, sent = [10.], [], [], [], []
        args = build_argument_parser().parse_args(['--open-space-panorama'])
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        def scans():
            for _ in range(1200 if persistent else 200):
                elapsed[0] += .1
                yield self.scan(elapsed[0])
        def turn(_c, iterator, _l, _s, pose, *_a):
            turns.append(True)
            return next(iterator), pose, 0., False
        def retreat(_c, iterator, _l, _s, pose, *_a):
            reverse.append(True)
            return next(iterator), pose.compose(Pose2D(-reverse_progress, 0, 0)), reverse_progress
        def match(*_a, **_k):
            matches.append(True)
            return None
        advance = {'can_advance': False, 'reason': 'swept_or_distance_blocked', 'stop_m': .72,
                   'lidar_m': 2. if map_only else .49, 'swept_m': 2. if map_only else .49,
                   'effective_m': .49, 'map_m': .49 if map_only else 2., 'horizon_m': .2}
        values = {name: {'effective_m': 2.5, 'lidar_m': 2.5, 'map_m': 2.5, 'support': 36}
                  for name in ('front', 'rear', 'left', 'right', 'front_left', 'front_right')}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch(
            'raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: elapsed[0]
        ), patch('raspberry_pi.open_space_panorama._advance_assessment', return_value=advance), patch(
            'raspberry_pi.open_space_panorama.directional_clearances', return_value=values
        ), patch('raspberry_pi.open_space_panorama.open_space_assessment',
                 return_value={'is_open': False, 'minimum_clearance_m': .49, 'open_direction_count': 3}), patch(
            'raspberry_pi.open_space_panorama._render_live_map'
        ), patch.object(OpenSpaceDirectionSelector, 'choose', return_value=None if no_direction else {
            'world_heading_rad': math.radians(70), 'relative_heading_deg': 70., 'clearance_m': 2., 'expected_new_samples': 20}
        ), patch('raspberry_pi.open_space_panorama._reverse_escape_assessment',
                 side_effect=lambda *_a: {'can_reverse': not exhausted_turns or len(turns) >= 6}), patch(
            'raspberry_pi.open_space_panorama._short_reverse', side_effect=retreat
        ), patch('raspberry_pi.open_space_panorama._turn_increment', side_effect=turn):
            report = InspectionReport(folder, args.map)
            result = run_open_space_panorama(args=args, known_map=known,
                localizer=SimpleNamespace(relative_motion=lambda *_a: MatchResult(Pose2D(), .01, 100, True, inlier_ratio=.95),
                                         localize_bounded=match),
                initial_pose=Pose2D(-.935, .625, 0), initial_scan=self.scan(10.), scan_iterator=scans(),
                client=SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: sent.append(True)),
                camera=SimpleNamespace(), report=report, online_qwen=SimpleNamespace(url=''),
                should_stop=lambda: len(turns) >= 7 if map_only_reselect else False if persistent else bool(matches) if map_only else bool(reverse))
            events = [json.loads(x) for x in (report.directory / 'panorama_motion.jsonl').read_text(encoding='utf-8').splitlines()]
        return result, turns, reverse, matches, sent, events

    def test_production_front_obstacle_reverses_before_turn(self):
        _, turns, reverse, _, sent, events = self.recovery_order_case()
        self.assertTrue(reverse)
        self.assertEqual(turns, [])
        self.assertEqual(sent, [])  # injected short reverse only; no forward drive
        self.assertTrue(next(x for x in events if x.get('event') == 'reverse_escape_result')['turn_budget_reset'])

    def test_production_reverse_does_not_require_a_long_candidate_direction(self):
        _, turns, reverse, _, _, _ = self.recovery_order_case(no_direction=True)
        self.assertTrue(reverse)
        self.assertEqual(turns, [])

    def test_production_reverse_precedes_exhausted_turn_budget_stop(self):
        result, turns, reverse, _, _, events = self.recovery_order_case(exhausted_turns=True)
        self.assertEqual(len(turns), 6)
        self.assertTrue(reverse)
        self.assertEqual(result['finish_reason'], 'requested_stop')
        self.assertTrue(next(x for x in events if x.get('event') == 'reverse_escape_result')['turn_budget_reset'])

    def test_production_reverse_without_progress_does_not_reset_turn_budget(self):
        _, _, _, _, _, events = self.recovery_order_case(reverse_progress=0.)
        self.assertFalse(next(x for x in events if x.get('event') == 'reverse_escape_result')['turn_budget_reset'])

    def test_production_map_only_block_checks_pose_without_reverse_or_turn(self):
        _, turns, reverse, matches, sent, _ = self.recovery_order_case(map_only=True)
        self.assertTrue(matches)
        self.assertEqual(turns, [])
        self.assertEqual(reverse, [])
        self.assertEqual(sent, [])

    def test_production_persistent_obstacle_cannot_replenish_reverse_allowance(self):
        result, turns, reverse, _, _, events = self.recovery_order_case(persistent=True)
        self.assertEqual(len(reverse), 3)
        self.assertGreater(len(turns), 18)
        self.assertEqual(result['finish_reason'], 'no_translation_progress')
        attempts = [x['attempt'] for x in events if x.get('event') == 'reverse_escape_result']
        self.assertEqual(attempts, [1, 2, 3])

    def test_production_map_only_six_turns_allow_one_guarded_reselection(self):
        result, turns, reverse, _, sent, _ = self.recovery_order_case(map_only=True, map_only_reselect=True)
        self.assertEqual(len(turns), 7)
        self.assertEqual(result['finish_reason'], 'requested_stop')
        self.assertEqual(reverse, [])
        self.assertEqual(sent, [])

    def test_production_map_only_reselection_does_not_allow_unbounded_turning(self):
        result, turns, reverse, _, sent, _ = self.recovery_order_case(map_only=True, persistent=True)
        self.assertGreater(len(turns), 12)
        self.assertEqual(result['finish_reason'], 'no_translation_progress')
        self.assertEqual(reverse, [])
        self.assertEqual(sent, [])

    def test_stopped_refresh_skips_empty_and_stale_without_motion(self):
        pose = Pose2D(1, 2, .3)
        fresh = self.scan(10.)
        client = SimpleNamespace(stop=lambda: None, set_twist=lambda *_a: self.fail('stopped refresh'))
        localizer = SimpleNamespace(relative_motion=lambda *_a: MatchResult(Pose2D(), .01, 100, True, inlier_ratio=.95))
        with patch('raspberry_pi.open_space_panorama.time.monotonic', return_value=10.):
            result, actual, accepted = _refresh_stopped_scan(client,
                iter([LaserScan(np.empty(0), np.empty(0), 10.), self.scan(8.), fresh]),
                localizer, self.scan(7.), pose, PanoramaConfig())
        self.assertIs(result, fresh)
        self.assertTrue(accepted)
        self.assertAlmostEqual(actual.yaw_rad, pose.yaw_rad)
        self.assertEqual((actual.x_m, actual.y_m), (pose.x_m, pose.y_m))

    def test_stopped_refresh_never_applies_rejected_pose(self):
        pose = Pose2D(1, 2, .3)
        old = self.scan(8.)
        localizer = SimpleNamespace(relative_motion=lambda *_a: MatchResult(Pose2D(3, 0, 0), .01, 100, True, inlier_ratio=.95))
        with patch('raspberry_pi.open_space_panorama.time.monotonic', return_value=10.):
            result, actual, accepted = _refresh_stopped_scan(SimpleNamespace(stop=lambda: None),
                iter([self.scan(10.)] * 6), localizer, old, pose, PanoramaConfig())
        self.assertFalse(accepted)
        self.assertIs(result, old)
        self.assertEqual(actual, pose)

    def test_turn_refresh_counts_accepted_tail_before_sending_more_output(self):
        client = SimpleNamespace(stop=lambda: None,
            set_twist=lambda *_a, **_k: self.fail('already reached angle during stopped refresh'))
        localizer = SimpleNamespace(relative_motion=lambda *_a: MatchResult(
            Pose2D(0, 0, math.radians(59)), .01, 100, True, inlier_ratio=.95))
        with patch('raspberry_pi.open_space_panorama.time.monotonic', return_value=10.):
            _, pose, measured, success = _turn_increment(client, iter([self.scan(10.)]), localizer,
                self.scan(8.), Pose2D(), math.radians(60), PanoramaConfig(), lambda *_a: None)
        self.assertTrue(success)
        self.assertAlmostEqual(measured, math.radians(59))
        self.assertAlmostEqual(pose.yaw_rad, measured)

    def test_bounded_map_search_discards_partial_endpoint_ranking(self):
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        localizer = ScanMapLocalizer(known)
        elapsed, count = [10.], [0]
        def slow_evaluate(*_a):
            elapsed[0] += .1
            count[0] += 1
            return 1., .01, .99
        with patch('raspberry_pi.known_map_navigation.time.monotonic', side_effect=lambda: elapsed[0]), patch.object(
            localizer, '_evaluate', side_effect=slow_evaluate):
            candidate = localizer.localize_bounded(self.scan(), Pose2D(-.935, .625, 0), budget_s=.35)
        self.assertIsNone(candidate)
        self.assertLessEqual(count[0], 4)

    def test_bounded_ray_matches_original_including_map_exit(self):
        import time
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        localizer = ScanMapLocalizer(known)
        for pose in (Pose2D(-.935, .625, 0), Pose2D(-100., -100., 0)):
            for angle in np.linspace(-math.pi, math.pi, 36, endpoint=False):
                expected = localizer._raycast_range(pose, float(angle))
                actual = localizer._bounded_raycast_range(pose, float(angle), time.monotonic() + 1.)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertAlmostEqual(actual, expected, places=8)

    def test_bounded_map_search_discards_partial_ray_ranking(self):
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        localizer = ScanMapLocalizer(known)
        original_ray = localizer._bounded_raycast_range
        elapsed, rays = [10.], [0]
        def slow_ray(*args):
            elapsed[0] += .1
            rays[0] += 1
            return original_ray(*args)
        with patch('raspberry_pi.known_map_navigation.time.monotonic', side_effect=lambda: elapsed[0]), patch.object(
            localizer, '_evaluate', return_value=(1., .01, .99)), patch.object(
            localizer, '_bounded_raycast_range', side_effect=slow_ray):
            result = localizer.localize_bounded(self.scan(), Pose2D(-.935, .625, 0))
        self.assertIsNone(result)
        self.assertEqual(rays[0], 4)

    def test_bounded_map_search_recovers_ideal_old_map_pose(self):
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        localizer = ScanMapLocalizer(known)
        truth = Pose2D(-.935, .625, math.radians(-128))
        angles, ranges = localizer.predicted_range_profile(truth, 180)
        candidate = localizer.localize_bounded(LaserScan(angles, ranges, 0.), truth)
        self.assertIsNotNone(candidate)
        self.assertLess(math.hypot(candidate.pose.x_m - truth.x_m, candidate.pose.y_m - truth.y_m), .07)
        self.assertLess(abs(candidate.pose.yaw_rad - truth.yaw_rad), math.radians(3))

    def test_delayed_dual_capture_still_completes_six_real_turn_directions(self):
        elapsed, turning, photos, commands = [10.], [False], [], []
        args = build_argument_parser().parse_args(['--open-space-panorama'])
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        last_scan_s = [10.]
        def scans():
            for _ in range(200):
                elapsed[0] += .1
                if elapsed[0] - last_scan_s[0] > .5:
                    last_scan_s[0] = elapsed[0]
                    yield LaserScan(np.empty(0), np.empty(0), elapsed[0])
                else:
                    last_scan_s[0] = elapsed[0]
                    yield self.scan(elapsed[0])
        def send(linear, angular, ttl_ms):
            self.assertGreater(ttl_ms, 0)
            turning[0] = linear == 0 and angular != 0
            commands.append((linear, angular))
        def capture(path):
            turning[0] = False
            elapsed[0] += 2.1  # actual camera latency; not mocked away
            photos.append(path.name)
            return {'camera_count': 2}
        values = {name: {'effective_m': 2.5, 'lidar_m': 2.5, 'map_m': 2.5, 'support': 36}
                  for name in ('front', 'rear', 'left', 'right', 'front_left', 'front_right')}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch(
            'raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: elapsed[0]
        ), patch('raspberry_pi.open_space_panorama.time.sleep', side_effect=lambda delay: elapsed.__setitem__(0, elapsed[0] + delay)), patch(
            'raspberry_pi.open_space_panorama.directional_clearances', return_value=values
        ), patch('raspberry_pi.open_space_panorama._render_live_map'), patch(
            'raspberry_pi.open_space_panorama.open_space_assessment',
            return_value={'is_open': True, 'minimum_clearance_m': 2.5, 'open_direction_count': 4}
        ):
            report = InspectionReport(folder, args.map)
            result = run_open_space_panorama(args=args, known_map=known,
                localizer=SimpleNamespace(relative_motion=lambda *_a: MatchResult(
                    Pose2D(0, 0, math.radians(20) if turning[0] else 0), .01, 100, True, inlier_ratio=.95)),
                initial_pose=Pose2D(-.935, .625, 0), initial_scan=self.scan(10.), scan_iterator=scans(),
                client=SimpleNamespace(stop=lambda: turning.__setitem__(0, False), set_twist=send),
                camera=SimpleNamespace(capture=capture), report=report, online_qwen=SimpleNamespace(url=''),
                should_stop=lambda: len(photos) >= 6)
            manifest = json.loads((report.directory / 'inspection_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(result['station_count'], 1)
        self.assertEqual(result['photo_count'], 6)
        self.assertEqual(len(photos), 6)
        self.assertTrue(commands)
        self.assertEqual([x['relative_panorama_yaw_deg'] for x in manifest['captures']], [0, 60, 120, 180, 240, 300])

    def scan(self, timestamp=0., distances=None):
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        return LaserScan(angles, np.full(360, 2.124) if distances is None else distances, timestamp)

    def assessment(self, scan, speed=240, map_m=.78, map_clear=True):
        with patch('raspberry_pi.open_space_panorama._nearest_map_guard_pose', return_value=Pose2D()), patch(
            'raspberry_pi.open_space_panorama._map_ray_clearance', return_value=map_m
        ):
            return _advance_assessment(scan, Pose2D(), SimpleNamespace(line_is_clear=lambda *_: map_clear),
                PanoramaConfig(), speed_mm_s=speed)

    def test_field_078m_course_can_advance_and_is_selected_straight(self):
        scan = self.scan()
        self.assertTrue(self.assessment(scan)['can_advance'])
        coverage = SimpleNamespace(points=np.array([[-2., 0.]]), covered=np.array([False]),
            expected_panorama_gain=lambda *_: 10000)
        with patch('raspberry_pi.open_space_panorama._nearest_map_guard_pose', return_value=Pose2D()), patch(
            'raspberry_pi.open_space_panorama._map_ray_clearance', return_value=.78
        ):
            choice = OpenSpaceDirectionSelector().choose(scan, Pose2D(),
                SimpleNamespace(line_is_clear=lambda *_: True), coverage, PanoramaConfig())
        self.assertEqual(choice['selection_policy'], 'safe_straight')

    def test_speed_braking_guard_does_not_force_unsafe_kick(self):
        self.assertTrue(self.assessment(self.scan(), speed=240, map_m=.65)['can_advance'])
        self.assertFalse(self.assessment(self.scan(), speed=450, map_m=.65)['can_advance'])

    def test_single_near_external_point_stops_even_with_clear_map(self):
        distances = np.full(360, 2.124)
        distances[180] = .132  # forward; definitely not a rear-body return
        scan = _filter_body_returns(self.scan(distances=distances), .19, .14, 0., .125)
        self.assertFalse(self.assessment(scan, map_m=3.)['can_advance'])

    def test_map_protects_glass_when_lidar_sees_through(self):
        self.assertFalse(self.assessment(self.scan(), map_m=.3)['can_advance'])
        self.assertFalse(self.assessment(self.scan(), map_m=3., map_clear=False)['can_advance'])

    def test_unsupported_front_cannot_advance_from_map_alone(self):
        scan = LaserScan(np.radians([90, 100, 110]), np.full(3, 2.), 0.)
        assessment = self.assessment(scan, map_m=3.)
        self.assertFalse(assessment['can_advance'])
        self.assertEqual(assessment['reason'], 'lidar_support_missing')

    def test_receipt_does_not_make_old_acquisition_fresh(self):
        with patch('raspberry_pi.open_space_panorama.time.monotonic', return_value=10.):
            scan = _ReceivedScan(self.scan(8.))
            self.assertEqual(_scan_age_s(scan), 2.)
            self.assertEqual(self.assessment(scan)['reason'], 'scan_stale')

    def test_invalid_acquisition_clock_is_rejected(self):
        with patch('raspberry_pi.open_space_panorama.time.monotonic', return_value=10.):
            for stamp in (math.nan, math.inf, -1., 11.):
                self.assertTrue(math.isinf(_scan_age_s(self.scan(stamp))))

    def test_many_small_startup_frames_do_not_trigger_count_failure(self):
        window = _DriveProgressWindow(PanoramaConfig())
        window.begin(0.)
        for step in range(1, 36):
            state = window.observe(step * .02, 0., True)
            self.assertNotIn(state, ('stalled', 'unknown'))
        self.assertEqual(window.observe(.8, .03, True), 'progress')

    def test_pause_excludes_intentional_stopping_from_window(self):
        window = _DriveProgressWindow(PanoramaConfig())
        window.begin(0.)
        window.observe(.4, 0., True)
        window.pause()
        self.assertEqual(window.observe(100., 0., True), 'paused')
        window.begin(100.)
        self.assertEqual(window.observe(100.1, 0., True), 'starting')

    def test_unknown_and_expired_output_not_reported_as_mechanical_stall(self):
        window = _DriveProgressWindow(PanoramaConfig())
        window.begin(0.)
        self.assertEqual(window.observe(2., 0., False), 'unknown')
        window.pause()
        window.begin(0.)
        self.assertEqual(window.observe(4., 0., True, energized_s=.1), 'unknown')

    def test_continuously_energized_no_progress_is_bounded_stall(self):
        window = _DriveProgressWindow(PanoramaConfig())
        window.begin(0.)
        self.assertEqual(window.observe(1.9, 0., True), 'stalled')

    def test_audit_denies_stale_commands_and_retains_exact_stop_scan(self):
        sent = []
        with tempfile.TemporaryDirectory() as folder, patch(
            'raspberry_pi.open_space_panorama.time.monotonic', return_value=10.
        ):
            audit = _MotionAuditClient(SimpleNamespace(stop=lambda: sent.append('stop'),
                set_twist=lambda *_a, **_k: sent.append('drive')), PanoramaConfig(), Path(folder))
            audit.scan = _ReceivedScan(self.scan(8.))
            self.assertFalse(audit.set_twist(420, 0, 500))
            self.assertNotIn('drive', sent)
            audit.stop_event('external_obstacle', audit.scan, {'can_advance': False})
            event = json.loads((Path(folder) / 'panorama_stop_events.jsonl').read_text())
            self.assertEqual(event['source_timestamp_s'], 8.)
            self.assertEqual(event['raw_distances_m'], audit.scan.distances_m.tolist())

    def test_refresh_and_ttl_cannot_outlive_safety_frame(self):
        refreshed, expired = [], []
        with tempfile.TemporaryDirectory() as folder, patch(
            'raspberry_pi.open_space_panorama.time.monotonic', return_value=10.
        ):
            fake = SimpleNamespace(stop=lambda: None, set_twist=lambda *_a, **_k: None,
                allow_twist_to_refresh_for=refreshed.append, allow_twist_to_expire=lambda: expired.append(True))
            audit = _MotionAuditClient(fake, PanoramaConfig(), Path(folder))
            audit.scan = _ReceivedScan(self.scan(9.8))
            self.assertTrue(audit.set_twist(420, 0, 700))
            self.assertLessEqual(audit.last['ttl_ms'] / 1000 + audit.last['refresh_window_s'] + .2, .501)
            self.assertTrue(audit.set_twist(0, 2600, 220))
            self.assertTrue(expired)

    def test_link_fault_cancels_motion_instead_of_refreshing(self):
        sent = []
        with tempfile.TemporaryDirectory() as folder:
            fake = SimpleNamespace(last_link_error=RuntimeError('link'), stop=lambda: sent.append('stop'),
                set_twist=lambda *_a, **_k: sent.append('drive'))
            audit = _MotionAuditClient(fake, PanoramaConfig(), Path(folder))
            audit.scan = _ReceivedScan(self.scan())
            self.assertFalse(audit.set_twist(420, 0, 500))
        self.assertEqual(sent, ['stop'])

    def test_explicit_command_denial_is_not_treated_as_energized_motion(self):
        stopped = []
        with tempfile.TemporaryDirectory() as folder:
            audit = _MotionAuditClient(SimpleNamespace(stop=lambda: stopped.append(True),
                set_twist=lambda *_a, **_k: False), PanoramaConfig(), Path(folder))
            audit.scan = _ReceivedScan(self.scan())
            self.assertFalse(audit.set_twist(420, 0, 500))
            self.assertEqual(audit.last['error'], 'command_denied')
        self.assertEqual(stopped, [True])

    def runtime_case(self, recover, scenario='normal'):
        elapsed, sent = [10. if scenario == 'stale' else 0.], []
        config_args = build_argument_parser().parse_args(['--open-space-panorama', '--start-kick-speed', '420'])
        if scenario == 'slow_matching':
            config_args.panorama_map_correction_distance = .3
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        scan = self.scan(8. if scenario == 'stale' else 0.)
        def scans():
            for _ in range(500):
                elapsed[0] += .02 if recover else .1
                yield self.scan(elapsed[0] - 2. if scenario == 'stale' else 0.)
        def relative(before, _current):
            if scenario == 'reanchor' and before.received_monotonic_s < 1.:
                return MatchResult(Pose2D(0, 0, math.radians(80)), .01, 100, True, inlier_ratio=.95)
            distance = .006 if scenario == 'slow_matching' else (.05 if scenario == 'reanchor' or (recover and len(sent) > 35) else 0.)
            return MatchResult(Pose2D(distance, 0, 0), .01, 100, True, inlier_ratio=.95)
        def slow_matching(*_a, **_k):
            elapsed[0] += 50.  # a broken injected implementation, never accept its result
            return None
        safe = {'can_advance': True, 'effective_m': 2.5, 'stop_m': .72, 'horizon_m': .2,
                'lidar_m': 2.5, 'map_m': 2.5, 'swept_m': 2.5}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch(
            'raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: elapsed[0]
        ), patch('raspberry_pi.open_space_panorama._advance_assessment', return_value=safe), patch(
            'raspberry_pi.open_space_panorama.open_space_assessment',
            return_value={'is_open': False, 'minimum_clearance_m': 2.5, 'open_direction_count': 4}
        ), patch.object(OpenSpaceDirectionSelector, 'choose', return_value={
            'world_heading_rad': 0., 'relative_heading_deg': 0., 'clearance_m': 2.5, 'expected_new_samples': 0}
        ), patch.object(KnownOccupancyMap, 'line_is_clear', return_value=True), patch(
            'raspberry_pi.open_space_panorama._render_live_map'
        ):
            report = InspectionReport(folder, config_args.map)
            result = run_open_space_panorama(args=config_args, known_map=known,
                localizer=SimpleNamespace(relative_motion=relative, localize_near_position=lambda *_a, **_k: None,
                    localize_bounded=slow_matching if scenario == 'slow_matching' else lambda *_a, **_k: None),
                initial_pose=Pose2D(-.935, .625, 0), initial_scan=scan, scan_iterator=scans(),
                client=SimpleNamespace(stop=lambda: None,
                    set_twist=lambda linear, angular, ttl_ms: sent.append((linear, angular, ttl_ms))),
                camera=SimpleNamespace(), report=report, online_qwen=SimpleNamespace(url=''),
                should_stop=lambda: recover and len(sent) >= (5 if scenario == 'reanchor' else 100 if scenario == 'slow_matching' else 50))
            summary = json.loads((report.directory / 'panorama_motion_summary.json').read_text())
        return result, summary, sent, elapsed[0]

    def test_production_loop_survives_35_zero_frames_then_moves(self):
        result, summary, sent, elapsed = self.runtime_case(True)
        self.assertEqual(len(sent), 50)
        self.assertEqual(result['finish_reason'], 'requested_stop')
        self.assertEqual(summary['finish_reason'], result['finish_reason'])
        self.assertEqual(sent[0][0], 420)
        self.assertGreater(result['final_pose']['x_m'], -.935)

    def test_production_loop_bounds_persistent_no_translation_retries(self):
        result, summary, sent, elapsed = self.runtime_case(False)
        self.assertEqual(result['finish_reason'], 'actuator_no_response')
        self.assertFalse(summary['motion_complete'])
        # Includes the finite, rear-guarded reverse fallback after three
        # complete no-progress drive windows; it must still be bounded.
        self.assertLess(elapsed, 12.)
        self.assertGreater(len(sent), 30)
        self.assertTrue(any(linear < 0 for linear, _, _ in sent))
        self.assertLessEqual(sum(linear < 0 for linear, _, _ in sent), 3)

    def test_production_loop_reanchors_without_applying_rejected_yaw(self):
        result, summary, sent, elapsed = self.runtime_case(True, 'reanchor')
        self.assertEqual(result['finish_reason'], 'requested_stop')
        self.assertTrue(result['pose_uncertain'])
        self.assertEqual(result['final_pose']['yaw_deg'], 0.)
        self.assertGreater(result['final_pose']['x_m'], -.935)
        self.assertEqual(len(sent), 5)
        self.assertLess(elapsed, 3.)

    def test_production_loop_stale_frames_never_send_motion(self):
        result, summary, sent, elapsed = self.runtime_case(False, 'stale')
        self.assertEqual(result['finish_reason'], 'lidar_data_unavailable')
        self.assertEqual(sent, [])
        self.assertFalse(summary['motion_complete'])

    def test_production_loop_excludes_stopped_matcher_overrun_from_drive_watchdog(self):
        result, summary, sent, elapsed = self.runtime_case(True, 'slow_matching')
        self.assertEqual(result['finish_reason'], 'requested_stop')
        self.assertEqual(len(sent), 100)
        self.assertGreater(elapsed, 50.)
        self.assertFalse(summary['motion_complete'])


if __name__ == '__main__':
    unittest.main()
