"""A rearward exit must not lose a responsive turn at the sixth step."""
import contextlib
import io
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
from raspberry_pi.open_space_panorama import _EscapeTurnBudget, OpenSpaceDirectionSelector, run_open_space_panorama


class EscapeTurnBudgetTests(unittest.TestCase):
    def test_responsive_half_turn_is_not_limited_to_six_steps(self):
        budget = _EscapeTurnBudget()
        budget.begin(0.)
        for i in range(9):
            budget.observe(math.radians(20))
            self.assertEqual(budget.reason((i + 1) * .5), '')
        self.assertAlmostEqual(budget.snapshot(4.5)['actual_angle_deg'], 180.)

    def test_actual_turn_angle_prevents_repeated_circles(self):
        budget = _EscapeTurnBudget()
        budget.begin(0.)
        for _ in range(10):
            budget.observe(math.radians(20))
        self.assertEqual(budget.reason(5.), 'turn_angle_limit')

    def test_alternating_turns_consume_angle_without_net_heading_change(self):
        budget = _EscapeTurnBudget()
        budget.begin(0.)
        for i in range(10):
            budget.observe(math.radians(20 if i % 2 else -20))
        self.assertEqual(budget.reason(5.), 'turn_angle_limit')

    def test_six_nonresponsive_attempts_still_stop(self):
        budget = _EscapeTurnBudget()
        budget.begin(0.)
        for _ in range(6):
            budget.observe(0.)
        self.assertEqual(budget.reason(6.), 'turn_no_response')

    def test_small_apparent_motion_cannot_extend_step_budget_forever(self):
        budget = _EscapeTurnBudget()
        budget.begin(0.)
        for _ in range(12):
            budget.observe(math.radians(2.))
        self.assertEqual(budget.reason(6.), 'turn_step_limit')

    def test_processing_pause_does_not_consume_escape_time(self):
        budget = _EscapeTurnBudget()
        budget.begin(10.)
        budget.pause(2.)
        self.assertEqual(budget.reason(31.9), '')
        self.assertEqual(budget.reason(32.1), 'turn_time_limit')

    def test_reset_clears_episode_and_nonfinite_measurement_is_not_progress(self):
        budget = _EscapeTurnBudget()
        budget.begin(0.)
        for _ in range(6):
            budget.observe(math.nan)
        self.assertEqual(budget.reason(6.), 'turn_no_response')
        budget.reset()
        self.assertEqual(budget.reason(100.), '')
        self.assertEqual(budget.snapshot(100.)['steps'], 0)

    def production_exit(self, clear_angle_deg=159.):
        elapsed, turns, commands = [10.], [], []
        args = build_argument_parser().parse_args(['--open-space-panorama'])
        known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
        def scan():
            return LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.), elapsed[0])
        def scans():
            while True:
                elapsed[0] += .1
                yield scan()
        def turn(_client, iterator, localizer, _scan, pose, requested, *_):
            turns.append(requested)
            localizer._panorama_local_heading_lock.observe(Pose2D(0, 0, requested))
            return next(iterator), pose.compose(Pose2D(0, 0, requested)), requested, True
        def advance(_scan, pose, *_args, **_kwargs):
            clear = sum(abs(angle) for angle in turns) >= math.radians(clear_angle_deg)
            distance = 2. if clear else .3
            return {'can_advance': clear, 'reason': '' if clear else 'swept_or_distance_blocked',
                    'lidar_m': distance, 'swept_m': distance, 'effective_m': distance,
                    'map_m': 2., 'stop_m': .61, 'horizon_m': .2}
        values = {name: {'effective_m': 2.5, 'lidar_m': 2.5, 'map_m': 2.5, 'support': 36}
                  for name in ('front', 'rear', 'left', 'right', 'front_left', 'front_right')}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), patch(
            'raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: elapsed[0]
        ), patch('raspberry_pi.open_space_panorama._advance_assessment', side_effect=advance), patch(
            'raspberry_pi.open_space_panorama.directional_clearances', return_value=values
        ), patch('raspberry_pi.open_space_panorama.open_space_assessment',
                 return_value={'is_open': False, 'minimum_clearance_m': .3, 'open_direction_count': 3}), patch(
            'raspberry_pi.open_space_panorama._rotation_assessment', return_value={'clearance_m': .5}
        ), patch('raspberry_pi.open_space_panorama._render_live_map'), patch.object(
            OpenSpaceDirectionSelector, 'choose', return_value={
                'world_heading_rad': math.pi, 'relative_heading_deg': 180., 'clearance_m': 2., 'expected_new_samples': 20}
        ), patch('raspberry_pi.open_space_panorama._reverse_escape_assessment',
                 return_value={'can_reverse': False, 'reason': 'rear_support_missing'}), patch(
            'raspberry_pi.open_space_panorama._turn_increment', side_effect=turn):
            report = InspectionReport(folder, args.map)
            summary = run_open_space_panorama(args=args, known_map=known,
                localizer=SimpleNamespace(relative_motion=lambda *_: MatchResult(Pose2D(.06, 0, 0), .01, 100, True, inlier_ratio=.95),
                                         localize_bounded=lambda *_a, **_kw: None),
                initial_pose=Pose2D(-.935, .625, 0), initial_scan=scan(), scan_iterator=scans(),
                client=SimpleNamespace(stop=lambda: None, set_twist=lambda *a, **_k: commands.append(a)),
                camera=SimpleNamespace(), report=report, online_qwen=SimpleNamespace(url=''),
                should_stop=lambda: bool(commands))
        self.assertTrue(commands)
        self.assertGreater(commands[0][0], 0)
        self.assertEqual(summary['finish_reason'], 'requested_stop')
        return turns

    def test_production_reaches_visible_exit_after_more_than_six_turns(self):
        turns = self.production_exit()
        self.assertGreater(len(turns), 6)

    def test_production_can_take_safe_exit_after_old_angle_and_step_limits(self):
        turns = self.production_exit(260.)
        self.assertGreater(len(turns), 12)
        self.assertGreaterEqual(sum(abs(angle) for angle in turns), math.radians(260.))


if __name__ == '__main__':
    unittest.main()
