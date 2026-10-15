"""Course alignment and fresh stopped obstacle evidence, including runtime."""
import contextlib
import io
import json
import math
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.map_camera_inspection import InspectionReport, build_argument_parser
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.open_space_panorama import (
    PanoramaConfig, OpenSpaceDirectionSelector, _CourseTurnBudget,
    _EscapeTurnBudget, _recheck_local_obstruction, run_open_space_panorama,
)


def scan(stamp, blocked=False):
    return LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False),
                     np.full(180, .135 if blocked else 2.), stamp)


def assessment(current, _pose, *_args, **_kwargs):
    blocked = current.distances_m[0] < .2
    distance = .107 if blocked else 2.
    return {'can_advance': not blocked,
            'reason': 'swept_or_distance_blocked' if blocked else '',
            'lidar_m': 1.2, 'swept_m': distance, 'effective_m': distance,
            'map_m': 2., 'stop_m': .61, 'horizon_m': .2}


class ObstructionRecheckTests(unittest.TestCase):
    def recheck(self, frames, *, quality=.95, stop=False):
        clock, commands = [10.], []
        def iterator():
            for frame in frames:
                clock[0] = max(clock[0], frame.timestamp_s)
                yield frame
        localizer = SimpleNamespace(relative_motion=lambda *_: MatchResult(
            Pose2D(.001, 0, 0), .01, 100, True, inlier_ratio=quality))
        with patch('raspberry_pi.open_space_panorama.time.monotonic',
                   side_effect=lambda: clock[0]):
            result = _recheck_local_obstruction(
                SimpleNamespace(stop=lambda: commands.append('stop')),
                iter(iterator()), localizer, scan(10., True), Pose2D(0, 0, 0),
                PanoramaConfig(), assessment, lambda: stop)
        self.assertEqual(commands, ['stop'])
        return result, localizer

    def test_transient_echo_requires_two_new_clear_acquisitions(self):
        result, _ = self.recheck([scan(10., False), scan(10.1), scan(10.2)])
        self.assertEqual(result[2], 'clear')
        self.assertEqual(result[0].timestamp_s, 10.2)
        self.assertAlmostEqual(result[1].x_m, .002)

    def test_persistent_real_obstacle_remains_blocked(self):
        result, _ = self.recheck([scan(10.1, True), scan(10.2, True)])
        self.assertEqual(result[2], 'lidar_obstacle')

    def test_duplicate_clear_frames_never_authorize_motion(self):
        result, _ = self.recheck([scan(10.)] * 6)
        self.assertEqual(result[2], 'sensor_unknown')
        self.assertEqual(result[1], Pose2D(0, 0, 0))

    def test_stale_clear_frames_never_authorize_motion(self):
        result, _ = self.recheck([scan(9.)] * 6)
        self.assertEqual(result[2], 'sensor_unknown')

    def test_alternating_noise_is_not_confirmed_clear_or_blocked(self):
        result, _ = self.recheck([scan(10.1 + i * .1, bool(i % 2)) for i in range(6)])
        self.assertEqual(result[2], 'sensor_unknown')

    def test_rejected_odometry_keeps_pending_anchor_and_no_clearance_claim(self):
        result, localizer = self.recheck([scan(10.1 + i * .1) for i in range(6)], quality=.33)
        self.assertEqual(result[2], 'sensor_unknown')
        self.assertEqual(result[1], Pose2D(0, 0, 0))
        self.assertIsNotNone(localizer._panorama_pending_motion)

    def test_cancel_keeps_vehicle_stopped(self):
        result, _ = self.recheck([], stop=True)
        self.assertEqual(result[2], 'sensor_unknown')

    def test_last_single_clear_frame_after_unknown_data_is_not_a_confirmation(self):
        invalid = [LaserScan(scan(10.1 + i * .1).angles_rad,
                            np.full(180, np.nan), 10.1 + i * .1) for i in range(5)]
        result, _ = self.recheck(invalid + [scan(10.6)])
        self.assertEqual(result[0].timestamp_s, 10.6)
        self.assertEqual(result[2], 'sensor_unknown')

    def test_optional_course_turns_have_their_own_finite_allowance(self):
        course, escape = _CourseTurnBudget(), _EscapeTurnBudget()
        course.begin(0.)
        for _ in range(4):
            course.observe(math.radians(2.))
        self.assertTrue(course.reason(2.))
        self.assertEqual(escape.snapshot(2.)['steps'], 0)


class RuntimeCourseEvidenceTests(unittest.TestCase):
    def runtime(self, *, exhausted=False, persistent=False, transient_after_turns=False,
                delayed_confirmation=False, pending_turn=False):
        clock, turns, commands = [10.], [], []
        args = build_argument_parser().parse_args(['--open-space-panorama'])
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        budget = _EscapeTurnBudget()
        if exhausted:
            budget.begin(10.)
            for _ in range(12):
                budget.observe(math.radians(2.))
        def scans():
            index = 0
            while True:
                clock[0] += .1
                current = scan(clock[0], persistent)
                if delayed_confirmation and index < 5:
                    current = LaserScan(current.angles_rad, np.full(180, np.nan), clock[0])
                index += 1
                yield current
        def turn(_client, iterator, localizer, _scan, pose, requested,
                 _config, _status, _guard, _state, translation_callback):
            self.assertIsNone(translation_callback)
            turns.append(requested)
            if pending_turn:
                localizer._panorama_pending_motion = (_scan, pose)
                localizer._panorama_pending_rotation_only = True
                return next(iterator), pose, 0., False
            measured = math.radians(2.)
            localizer._panorama_local_heading_lock.observe(Pose2D(0, 0, measured))
            current = next(iterator)
            if transient_after_turns and len(turns) == 4:
                current = scan(current.timestamp_s, True)
            return current, pose.compose(Pose2D(0, 0, measured)), measured, True
        values = {name: {'effective_m': 2.5, 'lidar_m': 2.5, 'map_m': 2.5, 'support': 36}
                  for name in ('front', 'rear', 'left', 'right', 'front_left', 'front_right')}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch(
            'raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: clock[0]
        ), patch('raspberry_pi.open_space_panorama._advance_assessment', side_effect=assessment), patch(
            'raspberry_pi.open_space_panorama.directional_clearances', return_value=values
        ), patch('raspberry_pi.open_space_panorama.open_space_assessment',
                 return_value={'is_open': False, 'minimum_clearance_m': .3, 'open_direction_count': 3}), patch(
            'raspberry_pi.open_space_panorama._rotation_assessment',
            return_value={'clearance_m': .1 if persistent else .5}
        ), patch('raspberry_pi.open_space_panorama._render_live_map'), patch.object(
            OpenSpaceDirectionSelector, 'choose', side_effect=lambda *_: {
                'world_heading_rad': math.radians(80.), 'relative_heading_deg': 80.,
                'clearance_m': 2., 'expected_new_samples': 20,
                'selection_policy': 'leave_visited_area'}
        ), patch('raspberry_pi.open_space_panorama._reverse_escape_assessment',
                 return_value={'can_reverse': False, 'reason': 'rear_support_missing'}), patch(
            'raspberry_pi.open_space_panorama._turn_increment', side_effect=turn
        ), patch('raspberry_pi.open_space_panorama._EscapeTurnBudget', return_value=budget):
            report = InspectionReport(folder, args.map)
            localizer = SimpleNamespace(relative_motion=lambda *_: MatchResult(
                Pose2D(0. if persistent else .001, 0, 0), .01, 100, True, inlier_ratio=.33 if pending_turn else .95),
                localize_bounded=lambda *_a, **_kw: None)
            summary = run_open_space_panorama(args=args, known_map=known,
                localizer=localizer,
                initial_pose=Pose2D(-.935, .625, 0), initial_scan=scan(10., exhausted),
                scan_iterator=scans(), client=SimpleNamespace(stop=lambda: None,
                    set_twist=lambda *a, **_k: commands.append(a)),
                camera=SimpleNamespace(), report=report, online_qwen=SimpleNamespace(url=''),
                should_stop=lambda: bool(commands))
            navigation = [json.loads(line) for line in
                          (report.directory / 'panorama_navigation.jsonl').read_text(encoding='utf-8').splitlines()]
        return summary, turns, commands, navigation

    def test_tiny_normal_turns_do_not_consume_escape_credit_or_end_route(self):
        summary, turns, commands, navigation = self.runtime(transient_after_turns=True)
        self.assertEqual(len(turns), 4)
        self.assertTrue(commands)
        self.assertEqual(commands[0][1], 0)
        self.assertEqual(summary['finish_reason'], 'requested_stop')
        self.assertTrue(all(item['local_escape_budget']['steps'] == 0 for item in navigation))
        self.assertTrue(any(item.get('obstruction_recheck_result') == 'clear' for item in navigation))

    def test_old_exhausted_quota_and_single_near_echo_do_not_end_route(self):
        summary, _, commands, navigation = self.runtime(exhausted=True)
        self.assertTrue(commands)
        self.assertEqual(summary['finish_reason'], 'requested_stop')
        self.assertTrue(any(item.get('obstruction_recheck_result') == 'clear' for item in navigation))

    def test_old_quota_does_not_end_route_but_real_persistent_no_progress_still_stops(self):
        summary, turns, commands, navigation = self.runtime(exhausted=True, persistent=True)
        self.assertFalse(commands)
        self.assertFalse(turns)
        self.assertEqual(summary['finish_reason'], 'no_translation_progress')
        self.assertTrue(any(item.get('obstruction_recheck_result') == 'lidar_obstacle' for item in navigation))

    def test_unknown_recheck_must_not_be_bypassed_by_its_last_single_clear_frame(self):
        summary, _, commands, navigation = self.runtime(exhausted=True, delayed_confirmation=True)
        self.assertTrue(commands)
        self.assertEqual(summary['finish_reason'], 'requested_stop')
        results = [item['obstruction_recheck_result'] for item in navigation
                   if 'obstruction_recheck_result' in item]
        self.assertEqual(results, ['sensor_unknown', 'clear'])


if __name__ == '__main__':
    unittest.main()
