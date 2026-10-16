import math
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raspberry_pi.autonomous_mapping import (  # noqa: E402
    ActiveRuntimeBudget,
    MotionCommandScheduler,
    build_argument_parser,
    run,
)
from raspberry_pi.esp32 import LinkError  # noqa: E402
from raspberry_pi.lidar import LaserScan  # noqa: E402
from raspberry_pi.mapping import OccupancyGrid, Pose2D  # noqa: E402
from raspberry_pi.mapping.adaptive_speed import (  # noqa: E402
    AdaptiveSpeedConfig,
    AdaptiveSpeedController,
)
from raspberry_pi.mapping.explorer import (  # noqa: E402
    _component_outline_is_closed,
    _enclosed_room_bounds,
    _inspection_view_gain,
    _room_required_cells,
    ExplorationConfig,
    FrontierExplorer,
    MotionCommand,
    find_obstacle_inspection_path,
    find_frontier_path,
    find_observation_path,
    has_unresolved_obstacle_shadows,
)
from raspberry_pi.mapping.slam import SlamUpdate  # noqa: E402
from raspberry_pi.mapping.scan_filter import filter_scan_sector  # noqa: E402


def front_scan(distance=2.0, timestamp=1.0):
    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 181)
    return LaserScan(
        angles,
        np.full_like(angles, distance),
        timestamp,
    )


def corridor_scan(width=0.80, distance=3.0, timestamp=1.0):
    """Synthetic front lidar scan between two parallel corridor walls."""
    angles = np.radians(np.linspace(-80.0, 80.0, 161))
    distances = np.full_like(angles, distance)
    side_rays = np.abs(angles) >= math.radians(20.0)
    wall_ranges = (width / 2.0) / np.maximum(
        np.abs(np.sin(angles)),
        1e-6,
    )
    distances[side_rays] = np.minimum(
        distances[side_rays],
        wall_ranges[side_rays],
    )
    return LaserScan(angles, distances, timestamp)


class FakeClient:
    active_transport = "ble"

    def __init__(self):
        self.heartbeat_count = 0
        self.commands = []
        self.events = []
        self.stopped = 0
        self.closed = False

    def heartbeat(self):
        self.heartbeat_count += 1

    def set_twist(self, linear, angular, ttl_ms):
        self.commands.append((linear, angular, ttl_ms))
        self.events.append(("command", linear, angular))

    def stop(self):
        self.stopped += 1
        self.events.append(("stop",))

    def close(self):
        self.closed = True


class FakeLidar:
    def __init__(self, scans):
        self._scans = scans
        self.closed = False

    def scans(self):
        yield from self._scans

    def close(self):
        self.closed = True


class AutonomousMappingTest(unittest.TestCase):
    def test_reachable_gray_frontier_preempts_obstacle_inspection(self):
        explorer = FrontierExplorer(ExplorationConfig(initial_spin_rad=0.0))
        explorer.state = "exploring"
        frontier_path = [(0.4, 0.0), (1.2, 0.0)]
        inspection_path = [(0.0, 0.4), (0.0, 0.8)]
        with patch(
            "raspberry_pi.mapping.explorer.find_frontier_path",
            return_value=frontier_path,
        ) as frontier, patch(
            "raspberry_pi.mapping.explorer.find_observation_path",
        ) as observation, patch(
            "raspberry_pi.mapping.explorer.find_obstacle_inspection_path",
            return_value=(
                inspection_path,
                (0.0, 0.0),
                (0.1, 0.1),
                True,
            ),
        ) as inspection:
            selected = explorer._find_exploration_path(
                OccupancyGrid(40, 40, 0.1),
                Pose2D(),
            )

        self.assertEqual(frontier_path, selected)
        frontier.assert_called_once()
        observation.assert_not_called()
        inspection.assert_not_called()
        self.assertIsNone(explorer._inspection_look_at)

    def test_obstacle_outline_requires_a_closed_saved_contour(self):
        rendered = np.zeros((30, 30), dtype=bool)
        rendered[10, 10:21] = True
        rendered[10:21, 10] = True
        rendered[10:21, 20] = True
        component = list(map(tuple, np.argwhere(rendered)))

        self.assertFalse(
            _component_outline_is_closed(rendered, component)
        )

        rendered[20, 10:21] = True
        component = list(map(tuple, np.argwhere(rendered)))
        self.assertTrue(
            _component_outline_is_closed(rendered, component)
        )

    def test_runtime_budget_excludes_reconnect_pause(self):
        now = [10.0]
        budget = ActiveRuntimeBudget(3.0, clock=lambda: now[0])
        now[0] = 11.0
        budget.pause()
        now[0] = 31.0
        self.assertAlmostEqual(1.0, budget.elapsed_s)
        budget.resume()
        now[0] = 32.9
        self.assertFalse(budget.expired())
        now[0] = 33.1
        self.assertTrue(budget.expired())

    def test_command_scheduler_limits_repeated_commands_but_sends_stop(self):
        scheduler = MotionCommandScheduler(220)
        moving = MotionCommand(300, 0, "moving", "test")
        stopped = MotionCommand(0, 0, "stopped", "test")
        self.assertTrue(scheduler.should_send(moving, 600, now_s=1.0))
        scheduler.mark_sent(moving, 600, now_s=1.0)
        self.assertFalse(scheduler.should_send(moving, 600, now_s=1.1))
        self.assertTrue(scheduler.should_send(moving, 600, now_s=1.23))
        self.assertTrue(scheduler.should_send(stopped, 600, now_s=1.1))
        self.assertTrue(scheduler.is_motion_refresh(moving))
        self.assertFalse(scheduler.is_motion_refresh(stopped))

    def test_command_scheduler_treats_ttl_only_change_as_refresh(self):
        scheduler = MotionCommandScheduler(220)
        turning = MotionCommand(0, -1500, "turning", "test")
        scheduler.mark_sent(turning, 550, now_s=1.0)

        self.assertTrue(scheduler.is_motion_refresh(turning))
        self.assertTrue(scheduler.should_send(turning, 600, now_s=1.23))

    def test_defaults_use_ble_and_acm_lidar(self):
        args = build_argument_parser().parse_args([])
        self.assertEqual("ble", args.link)
        self.assertEqual("/dev/serial0", args.esp_port)
        self.assertEqual(1.5, args.usb_open_settle)
        self.assertEqual("/dev/ttyACM0", args.lidar_port)
        self.assertEqual(160.0, args.usable_fov)
        self.assertEqual(6.0, args.mapping_max_distance)
        self.assertEqual(0.0, args.wall_gap_max)
        self.assertEqual(250, args.speed)
        self.assertEqual(460, args.max_drive_speed)
        self.assertEqual(220, args.slow_speed)
        self.assertEqual(1500, args.turn_speed)
        self.assertEqual(3000, args.max_turn_speed)
        self.assertEqual(260, args.turn_arc_speed)
        self.assertEqual(0.95, args.narrow_passage_width)
        self.assertEqual(240, args.narrow_speed)
        self.assertEqual(1900, args.narrow_turn_speed)
        self.assertEqual(0.14, args.lidar_offset_x)
        self.assertEqual(0.01, args.min_obstacle_area)
        self.assertEqual(6.0, args.ble_connect_timeout)
        self.assertEqual(0.8, args.ble_operation_timeout)
        self.assertEqual(0.40, args.link_request_timeout)
        self.assertEqual(1, args.link_request_retries)
        self.assertEqual(0.40, args.stationary_map_distance)
        self.assertEqual(30.0, args.stationary_map_angle)
        self.assertEqual(2, args.stationary_settle_scans)
        self.assertEqual(5, args.stationary_fusion_scans)
        self.assertEqual(3, args.stationary_min_support)
        self.assertEqual(0.10, args.stationary_range_tolerance)
        self.assertEqual(0, args.save_every)
        self.assertFalse(args.manhattan)
        self.assertEqual(700, args.command_ttl_ms)
        self.assertEqual(450, args.command_min_interval_ms)
        self.assertEqual(8, args.link_reconnect_attempts)
        self.assertEqual(0.25, args.link_reconnect_delay)
        self.assertEqual(0, args.link_reconnect_rounds)
        self.assertFalse(args.live_map)
        self.assertEqual("0.0.0.0", args.live_map_bind)
        self.assertEqual(8766, args.live_map_port)
        self.assertEqual(2.0, args.live_map_refresh_hz)

    def test_explorer_defaults_prioritize_translation_over_long_spins(self):
        config = ExplorationConfig()
        self.assertLessEqual(
            config.initial_spin_rad,
            math.radians(90.0),
        )
        self.assertLessEqual(
            config.maximum_stationary_turn_rad,
            math.radians(90.0),
        )
        self.assertGreaterEqual(config.stationary_turn_radius_m, 0.20)
        self.assertLessEqual(config.progress_window_updates, 24)

    def test_filter_keeps_only_front_half(self):
        angles = np.radians(np.array((-179, -91, -90, 0, 90, 91, 179)))
        scan = LaserScan(angles, np.ones(len(angles)), 1.0)
        filtered = filter_scan_sector(scan, 0, 180)
        np.testing.assert_allclose(
            filtered.angles_rad, np.radians((-90, 0, 90))
        )

    def test_frontier_path_stays_in_observed_free_space(self):
        grid = OccupancyGrid(60, 60, 0.1)
        grid.observed[24:37, 24:37] = True
        grid.log_odds[24:37, 24:37] = -1.0
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.2,
            min_frontier_cluster_cells=3,
        )
        path = find_frontier_path(grid, Pose2D(), config)
        self.assertGreater(len(path), 1)
        probabilities = grid.probabilities()
        for point in path:
            column, row = grid.world_to_grid(np.asarray((point,)))[0]
            self.assertTrue(grid.observed[row, column])
            self.assertLess(probabilities[row, column], 0.45)

    def test_glass_leak_outside_supported_room_is_not_a_frontier(self):
        grid = OccupancyGrid(
            80, 80, 0.1, min_obstacle_area_m2=0.01
        )
        grid.observed[25:56, 25:56] = True
        grid.log_odds[25:56, 25:56] = -2.0
        grid.log_odds[25, 25:56] = 4.0
        grid.log_odds[55, 25:56] = 4.0
        grid.log_odds[25:56, 25] = 4.0
        grid.log_odds[25:56, 55] = 4.0
        # A glass gap creates observed free cells outside, followed by sparse
        # false occupied returns farther away.
        grid.log_odds[25, 39:42] = -2.0
        grid.observed[17:25, 38:43] = True
        grid.log_odds[17:25, 38:43] = -2.0
        grid.observed[16, 38:44] = True
        grid.log_odds[16, 38:44] = 4.0
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
        )

        self.assertFalse(find_frontier_path(grid, Pose2D(), config))
        self.assertFalse(find_observation_path(grid, Pose2D(), config))
        self.assertFalse(has_unresolved_obstacle_shadows(grid, config))

        # A real unknown pocket inside the supported walls remains eligible.
        grid.observed[38:43, 48:52] = False
        path = find_frontier_path(grid, Pose2D(), config)
        self.assertTrue(path)
        col, row = grid.world_to_grid(np.asarray((path[-1],)))[0]
        self.assertTrue(25 <= row <= 55)
        self.assertTrue(25 <= col <= 55)

    def test_table_shadow_selects_an_interior_inspection_viewpoint(self):
        grid = OccupancyGrid(
            80, 80, 0.1, min_obstacle_area_m2=0.01
        )
        grid.observed[20:61, 20:61] = True
        grid.log_odds[20:61, 20:61] = -2.0
        grid.log_odds[20, 20:61] = 4.0
        grid.log_odds[60, 20:61] = 4.0
        grid.log_odds[20:61, 20] = 4.0
        grid.log_odds[20:61, 60] = 4.0
        grid.log_odds[36:45, 44:46] = 4.0
        grid.observed[37:44, 46:51] = False
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            obstacle_inspection_min_observed_area_m2=1.0,
            obstacle_inspection_standoff_min_m=0.3,
            obstacle_inspection_standoff_max_m=1.2,
        )

        path, look_at = find_obstacle_inspection_path(
            grid,
            Pose2D(),
            config,
            return_look_at=True,
        )

        self.assertTrue(path)
        self.assertIsNotNone(look_at)
        self.assertLess(abs(look_at[0]), 1.0)
        self.assertLess(abs(look_at[1]), 1.0)

    def test_fused_inspection_viewpoint_is_consumed_once(self):
        explorer = FrontierExplorer(ExplorationConfig())
        explorer._inspection_look_at = (0.4, 0.0)
        explorer._inspection_component_anchor = (0.4, 0.0)
        explorer._path = []
        explorer._inspection_hold_count = 1

        self.assertTrue(explorer.note_stationary_map_integrated(Pose2D()))
        self.assertIsNone(explorer._inspection_look_at)
        self.assertEqual((0.4, 0.0), explorer._inspection_component_anchor)
        self.assertEqual([], explorer._path)
        self.assertFalse(explorer.note_stationary_map_integrated(Pose2D()))

    def test_frontier_uses_farthest_straight_line_boundary(self):
        grid = OccupancyGrid(80, 80, 0.1)
        grid.observed[15:66, 15:66] = True
        grid.log_odds[15:66, 15:66] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        near_col = int(center_col + 8)
        far_col = int(center_col - 22)
        row = int(center_row)
        grid.observed[row - 1:row + 2, near_col - 1:near_col + 2] = False
        grid.observed[row - 6:row + 7, far_col - 6:far_col + 7] = False
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
        )

        path = find_frontier_path(grid, Pose2D(), config)

        self.assertTrue(path)
        self.assertLess(path[-1][0], 0.0)

    def test_frontier_can_still_use_nearest_policy_when_requested(self):
        grid = OccupancyGrid(80, 80, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        row = int(center_row)
        grid.observed[
            row - 1:row + 2,
            int(center_col + 8) - 1:int(center_col + 8) + 2,
        ] = False
        grid.observed[
            row - 2:row + 3,
            int(center_col - 22) - 2:int(center_col - 22) + 3,
        ] = False
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
            prefer_distant_frontiers=False,
        )

        path = find_frontier_path(grid, Pose2D(), config)

        self.assertTrue(path)
        self.assertGreater(path[-1][0], 0.0)

    def test_blacklisted_near_frontier_selects_another_reachable_frontier(self):
        grid = OccupancyGrid(80, 80, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        row = int(center_row)
        near_col = int(center_col + 8)
        far_col = int(center_col - 18)
        grid.observed[row - 2:row + 3, near_col - 2:near_col + 3] = False
        grid.observed[row - 2:row + 3, far_col - 2:far_col + 3] = False
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
            frontier_blacklist_radius_m=0.6,
        )
        far_path = find_frontier_path(grid, Pose2D(), config)

        alternate_path = find_frontier_path(
            grid,
            Pose2D(),
            config,
            excluded_targets=[(*far_path[-1], 999)],
        )

        self.assertTrue(alternate_path)
        self.assertGreater(
            math.hypot(
                alternate_path[-1][0] - far_path[-1][0],
                alternate_path[-1][1] - far_path[-1][1],
            ),
            config.frontier_blacklist_radius_m,
        )

    def test_blacklist_does_not_hide_rest_of_same_white_boundary(self):
        grid = OccupancyGrid(80, 80, 0.1)
        grid.observed[18:63, 18:63] = True
        grid.log_odds[18:63, 18:63] = -2.0
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
            frontier_blacklist_radius_m=0.6,
        )
        first_path = find_frontier_path(grid, Pose2D(), config)

        alternate_path = find_frontier_path(
            grid,
            Pose2D(),
            config,
            excluded_targets=[(*first_path[-1], 999)],
        )

        self.assertTrue(alternate_path)
        self.assertGreater(
            math.hypot(
                alternate_path[-1][0] - first_path[-1][0],
                alternate_path[-1][1] - first_path[-1][1],
            ),
            config.frontier_blacklist_radius_m,
        )

    def test_inflated_table_island_triggers_breadcrumb_retrace(self):
        grid = OccupancyGrid(100, 100, 0.10)
        grid.observed[20:81, 20:81] = True
        grid.log_odds[20:81, 20:81] = -2.0
        pose = Pose2D(x_m=0.25, y_m=0.0)
        center_col, center_row = map(int, grid.world_to_grid(
            np.asarray(((pose.x_m, pose.y_m),))
        )[0])
        # A broken rectangular table trace has a real two-cell opening on the
        # left. Normal 0.30 m clearance inflation closes it and isolates the
        # current pose even though the chassis just drove through that gap.
        radius = 6
        grid.log_odds[
            center_row - radius:center_row + radius + 1,
            center_col - radius,
        ] = 3.0
        grid.log_odds[
            center_row - radius:center_row + radius + 1,
            center_col + radius,
        ] = 3.0
        grid.log_odds[
            center_row - radius,
            center_col - radius:center_col + radius + 1,
        ] = 3.0
        grid.log_odds[
            center_row + radius,
            center_col - radius:center_col + radius + 1,
        ] = 3.0
        grid.log_odds[
            center_row - 1:center_row + 2,
            center_col - radius,
        ] = -2.0
        grid.refresh_obstacle_filter()
        config = ExplorationConfig(
            robot_clearance_m=0.30,
            min_frontier_distance_m=0.10,
            min_frontier_cluster_cells=2,
            history_corridor_island_max_area_m2=1.20,
        )

        explorer = FrontierExplorer(config)
        explorer.state = "exploring"
        explorer._breadcrumbs.extend((
            (-1.40, 0.0),
            (-1.20, 0.0),
            (-1.00, 0.0),
            (-0.80, 0.0),
            (-0.60, 0.0),
            (-0.40, 0.0),
            (-0.20, 0.0),
            (0.0, 0.0),
            (0.20, 0.0),
        ))
        explorer._find_exploration_path = lambda *_args, **_kwargs: []

        command = explorer.update(pose, front_scan(), grid)

        self.assertTrue(explorer._backtracking_active)
        self.assertTrue(explorer._path)
        self.assertLess(explorer._path[-1][0], -0.20)
        self.assertNotEqual("completion_scan", command.state)

    def test_unproductive_disconnected_backtrack_is_not_repeated_forever(self):
        explorer = FrontierExplorer(ExplorationConfig(initial_spin_rad=0.0))
        explorer.state = "exploring"
        explorer._disconnected_backtrack_failures = 2
        explorer._breadcrumbs.extend(((-0.4, 0.0), (-0.2, 0.0)))
        explorer._find_exploration_path = lambda *_args, **_kwargs: []
        grid = OccupancyGrid(40, 40, 0.1)

        with patch(
            "raspberry_pi.mapping.explorer._has_global_frontier",
            return_value=True,
        ), patch(
            "raspberry_pi.mapping.explorer._reachable_clearance_area_m2",
            return_value=0.1,
        ):
            explorer.update(Pose2D(), front_scan(), grid)

        self.assertFalse(explorer._backtracking_active)

    def test_front_obstacle_turns_without_reverse(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            replan_every_updates=100,
            stop_distance_m=0.4,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        scan = front_scan()
        scan.distances_m[np.abs(scan.angles_rad) < math.radians(10)] = 0.2
        command = explorer.update(
            Pose2D(), scan, OccupancyGrid(40, 40, 0.1)
        )
        self.assertEqual("avoidance", command.state)
        self.assertEqual(0, command.linear_mm_s)
        self.assertNotEqual(0, command.angular_mrad_s)

    def test_committed_frontier_is_not_replaced_periodically(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            replan_every_updates=1,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        explorer._active_frontier_target = (1.0, 0.0)

        command = explorer.update(
            Pose2D(),
            front_scan(),
            OccupancyGrid(40, 40, 0.1),
        )

        self.assertEqual((1.0, 0.0), command.target_xy_m)
        self.assertEqual((1.0, 0.0), explorer._active_frontier_target)

    def test_front_wall_uses_known_free_rear_corridor_to_escape(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            replan_every_updates=100,
            stop_distance_m=0.4,
            wall_escape_backoff_m=0.18,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        grid = OccupancyGrid(60, 60, 0.05)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        blocked_scan = front_scan()
        blocked_scan.distances_m[
            np.abs(blocked_scan.angles_rad) < math.radians(10)
        ] = 0.2

        backoff = explorer.update(Pose2D(), blocked_scan, grid)
        self.assertEqual("wall_escape_backoff", backoff.state)
        self.assertLess(backoff.linear_mm_s, 0)

        turning = explorer.update(
            Pose2D(x_m=-0.20),
            front_scan(timestamp=2.0),
            grid,
        )
        self.assertEqual("wall_escape_turn", turning.state)
        self.assertNotEqual(0, turning.angular_mrad_s)

        leaving = explorer.update(
            Pose2D(x_m=-0.20, yaw_rad=math.radians(60.0)),
            front_scan(timestamp=3.0),
            grid,
        )
        self.assertEqual("wall_escape_forward", leaving.state)
        self.assertGreater(leaving.linear_mm_s, 0)

    def test_front_wall_never_reverses_into_unknown_space(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            replan_every_updates=100,
            stop_distance_m=0.4,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        scan = front_scan()
        scan.distances_m[
            np.abs(scan.angles_rad) < math.radians(10)
        ] = 0.2
        command = explorer.update(
            Pose2D(),
            scan,
            OccupancyGrid(60, 60, 0.05),
        )
        self.assertGreaterEqual(command.linear_mm_s, 0)
        self.assertNotEqual("wall_escape_backoff", command.state)

    def test_recent_approach_path_allows_retrace_over_unknown_grid(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            replan_every_updates=100,
            stop_distance_m=0.4,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        explorer._pose_history.extend(((-0.30, 0.0), (-0.16, 0.0)))
        scan = front_scan()
        scan.distances_m[
            np.abs(scan.angles_rad) < math.radians(10)
        ] = 0.2
        command = explorer.update(
            Pose2D(),
            scan,
            OccupancyGrid(60, 60, 0.05),
        )
        self.assertEqual("wall_escape_backoff", command.state)
        self.assertEqual(
            -explorer.config.wall_escape_reverse_speed_mm_s,
            command.linear_mm_s,
        )

    def test_explorer_does_not_complete_before_real_travel(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            completion_confirmations=1,
            completion_spin_rad=0.0,
            min_completion_travel_m=0.75,
        ))
        explorer.state = "exploring"
        command = explorer.update(
            Pose2D(),
            front_scan(),
            OccupancyGrid(40, 40, 0.1),
        )
        self.assertFalse(command.finished)
        self.assertNotEqual("complete", command.state)
        self.assertEqual("recovery_probe", command.state)
        self.assertGreater(command.linear_mm_s, 0)

    def test_blacklist_cannot_cause_false_exploration_completion(self):
        config = ExplorationConfig(
            initial_spin_rad=0.0,
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
            frontier_blacklist_radius_m=1.0,
        )
        explorer = FrontierExplorer(config)
        explorer.state = "exploring"
        grid = OccupancyGrid(60, 60, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        row = int(center_row)
        col = int(center_col + 10)
        grid.observed[row - 1:row + 2, col - 1:col + 2] = False
        frontier = find_frontier_path(grid, Pose2D(), config)
        self.assertTrue(frontier)
        explorer._frontier_blacklist.append((
            frontier[-1][0],
            frontier[-1][1],
            1000,
        ))

        command = explorer.update(Pose2D(), front_scan(), grid)

        self.assertFalse(command.finished)
        self.assertTrue(explorer._path)

    def test_observation_path_targets_unknown_missed_by_frontier_filter(self):
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=1000,
            occluded_unknown_radius_m=2.0,
            observation_candidate_spacing_m=0.1,
        )
        grid = OccupancyGrid(60, 60, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        row = int(center_row)
        col = int(center_col + 12)
        grid.observed[row - 2:row + 3, col - 2:col + 3] = False

        self.assertFalse(find_frontier_path(grid, Pose2D(), config))
        self.assertTrue(find_observation_path(grid, Pose2D(), config))

    def test_observation_path_ignores_isolated_unknown_raster_holes(self):
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=1000,
            occluded_unknown_radius_m=2.0,
            observation_candidate_spacing_m=0.1,
            observation_min_unknown_area_m2=0.02,
        )
        grid = OccupancyGrid(60, 60, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        grid.observed[int(center_row), int(center_col + 12)] = False

        self.assertFalse(find_observation_path(grid, Pose2D(), config))

    def test_obstacle_inspection_targets_unseen_back_side(self):
        config = ExplorationConfig(
            robot_clearance_m=0.10,
            robot_length_m=0.20,
            robot_width_m=0.16,
            min_frontier_distance_m=0.10,
            obstacle_inspection_min_observed_area_m2=0.0,
            obstacle_inspection_min_cells=4,
            obstacle_inspection_standoff_min_m=0.35,
            obstacle_inspection_standoff_max_m=1.20,
            observation_candidate_spacing_m=0.10,
        )
        grid = OccupancyGrid(
            100, 100, 0.10, min_obstacle_area_m2=0.0
        )
        grid.observed[20:81, 15:86] = True
        grid.log_odds[20:81, 15:86] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        center_col = int(center_col)
        center_row = int(center_row)
        # Only the west face has been mapped.  The east-side shadow is gray,
        # while a known-free route around the north and south remains.
        grid.log_odds[
            center_row - 5:center_row + 6,
            center_col - 4,
        ] = 4.0
        grid.observed[
            center_row - 4:center_row + 5,
            center_col - 3:center_col + 11,
        ] = False
        pose = Pose2D(x_m=-2.0, y_m=0.0)

        path = find_obstacle_inspection_path(grid, pose, config)

        self.assertTrue(path)
        self.assertGreater(path[-1][0], -0.25)
        self.assertGreater(abs(path[-1][1]), 0.20)

    def test_obstacle_inspection_keeps_preferred_component_until_complete(self):
        config = ExplorationConfig(
            robot_clearance_m=0.10,
            robot_length_m=0.20,
            robot_width_m=0.16,
            min_frontier_distance_m=0.10,
            obstacle_inspection_min_observed_area_m2=0.0,
            obstacle_inspection_min_cells=4,
            obstacle_inspection_standoff_min_m=0.35,
            obstacle_inspection_standoff_max_m=1.20,
            observation_candidate_spacing_m=0.10,
        )
        grid = OccupancyGrid(
            120, 100, 0.10, min_obstacle_area_m2=0.0
        )
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        for center_x in (-1.0, 2.0):
            center_col, center_row = grid.world_to_grid(
                np.asarray(((center_x, 0.0),))
            )[0]
            center_col = int(center_col)
            center_row = int(center_row)
            grid.log_odds[
                center_row - 5:center_row + 6,
                center_col - 3,
            ] = 4.0
            grid.observed[
                center_row - 4:center_row + 5,
                center_col - 2:center_col + 9,
            ] = False
            grid.observed[
                center_row - 5:center_row + 6,
                center_col - 3,
            ] = True

        path, look_at, component_anchor, preferred_pending = (
            find_obstacle_inspection_path(
                grid,
                Pose2D(x_m=-2.5),
                config,
                preferred_component_xy=(2.0, 0.0),
                return_details=True,
            )
        )

        self.assertTrue(path)
        self.assertTrue(preferred_pending)
        self.assertIsNotNone(look_at)
        self.assertGreater(component_anchor[0], 1.0)

    def test_closed_obstacle_without_external_shadow_needs_no_inspection(self):
        config = ExplorationConfig(
            robot_clearance_m=0.10,
            robot_length_m=0.20,
            robot_width_m=0.16,
            min_frontier_distance_m=0.10,
            obstacle_inspection_min_observed_area_m2=0.0,
            obstacle_inspection_min_cells=4,
        )
        grid = OccupancyGrid(
            80, 80, 0.10, min_obstacle_area_m2=0.0
        )
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        center_col = int(center_col)
        center_row = int(center_row)
        grid.log_odds[
            center_row - 4:center_row + 5,
            center_col - 4:center_col + 5,
        ] = 4.0

        path = find_obstacle_inspection_path(grid, Pose2D(-2.0), config)

        self.assertFalse(path)
        self.assertFalse(has_unresolved_obstacle_shadows(grid, config))

    def test_supported_room_requires_diverse_views_even_after_shadow_clears(self):
        config = ExplorationConfig(
            robot_clearance_m=0.10,
            robot_length_m=0.20,
            robot_width_m=0.16,
            min_frontier_distance_m=0.10,
            obstacle_inspection_min_observed_area_m2=0.0,
            obstacle_inspection_min_cells=4,
            obstacle_inspection_standoff_min_m=0.35,
            obstacle_inspection_standoff_max_m=1.20,
            enclosed_room_min_edge_cells=3,
        )
        grid = OccupancyGrid(
            80, 80, 0.10, min_obstacle_area_m2=0.0
        )
        grid.observed[15:66, 15:66] = True
        grid.log_odds[15:66, 15:66] = -2.0
        grid.log_odds[15, 15:66] = 4.0
        grid.log_odds[65, 15:66] = 4.0
        grid.log_odds[15:66, 15] = 4.0
        grid.log_odds[15:66, 65] = 4.0
        grid.log_odds[37:43, 37:43] = 4.0

        path, look_at, anchor, pending = find_obstacle_inspection_path(
            grid,
            Pose2D(x_m=-1.5),
            config,
            return_details=True,
        )

        self.assertTrue(path)
        self.assertIsNotNone(look_at)
        self.assertFalse(pending)
        self.assertIsNotNone(anchor)
        completed = find_obstacle_inspection_path(
            grid,
            Pose2D(x_m=-1.5),
            config,
            completed_components_xy=(anchor,),
        )
        self.assertFalse(completed)

    def test_stationary_inspection_records_distinct_view_sectors(self):
        explorer = FrontierExplorer(ExplorationConfig())
        explorer._inspection_component_anchor = (0.0, 0.0)
        explorer._inspection_look_at = (0.0, 0.0)
        explorer._path = []

        self.assertTrue(explorer.note_stationary_map_integrated(
            Pose2D(x_m=1.0, y_m=0.0)
        ))
        explorer._inspection_look_at = (0.0, 0.0)
        self.assertTrue(explorer.note_stationary_map_integrated(
            Pose2D(x_m=0.0, y_m=1.0)
        ))

        self.assertEqual({0, 3}, explorer._inspection_seen_sectors)

    def test_enroute_stationary_fusion_counts_without_consuming_viewpoint(self):
        explorer = FrontierExplorer(ExplorationConfig())
        explorer._inspection_component_anchor = (0.0, 0.0)
        explorer._inspection_look_at = (0.0, 0.0)
        explorer._path = [(2.0, 0.0)]

        self.assertTrue(explorer.note_stationary_map_integrated(
            Pose2D(x_m=1.0, y_m=0.0)
        ))

        self.assertEqual({0}, explorer._inspection_seen_sectors)
        self.assertEqual([(2.0, 0.0)], explorer._path)
        self.assertEqual((0.0, 0.0), explorer._inspection_look_at)

    def test_partial_obstacle_shadow_blocks_completion(self):
        config = ExplorationConfig(
            obstacle_inspection_min_observed_area_m2=0.0,
            obstacle_inspection_min_cells=4,
            obstacle_inspection_shadow_depth_m=0.8,
            obstacle_inspection_min_shadow_cells=2,
        )
        grid = OccupancyGrid(
            80, 80, 0.10, min_obstacle_area_m2=0.0
        )
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        center_col = int(center_col)
        center_row = int(center_row)
        grid.log_odds[
            center_row - 5:center_row + 6,
            center_col - 3,
        ] = 4.0
        grid.observed[
            center_row - 4:center_row + 5,
            center_col - 2:center_col + 10,
        ] = False

        self.assertTrue(has_unresolved_obstacle_shadows(grid, config))

    def test_inspection_viewpoint_turns_to_face_object_then_holds(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            obstacle_inspection_hold_updates=2,
            progress_window_updates=100,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        explorer._active_frontier_target = (1.0, 0.0)
        explorer._inspection_look_at = (0.0, 0.0)
        grid = OccupancyGrid(60, 60, 0.1)

        turning = explorer.update(
            Pose2D(x_m=1.0, y_m=0.0, yaw_rad=0.0),
            front_scan(),
            grid,
        )
        self.assertEqual("obstacle_inspection_turn", turning.state)
        self.assertEqual(0, turning.linear_mm_s)
        self.assertNotEqual(0, turning.angular_mrad_s)

        first_hold = explorer.update(
            Pose2D(x_m=1.0, y_m=0.0, yaw_rad=math.pi),
            front_scan(timestamp=2.0),
            grid,
        )
        second_hold = explorer.update(
            Pose2D(x_m=1.0, y_m=0.0, yaw_rad=math.pi),
            front_scan(timestamp=3.0),
            grid,
        )
        self.assertEqual("obstacle_inspection_observe", first_hold.state)
        self.assertEqual("obstacle_inspection_observe", second_hold.state)
        self.assertEqual((0, 0), (
            second_hold.linear_mm_s,
            second_hold.angular_mrad_s,
        ))

    def test_completion_requires_a_full_rescan(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            completion_confirmations=1,
            completion_spin_rad=math.tau,
            min_completion_travel_m=0.0,
            progress_window_updates=100,
        ))
        explorer.state = "exploring"
        grid = OccupancyGrid(40, 40, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        commands = []
        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.0):
            commands.append(explorer.update(
                Pose2D(yaw_rad=yaw),
                front_scan(),
                grid,
            ))

        self.assertTrue(all(not item.finished for item in commands[:-1]))
        self.assertTrue(commands[-1].finished)
        self.assertEqual("complete", commands[-1].state)

    def test_unreachable_gray_shadow_does_not_block_completion(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            completion_confirmations=1,
            completion_spin_rad=math.tau,
            min_completion_travel_m=0.0,
            progress_window_updates=100,
        ))
        explorer.state = "exploring"
        grid = OccupancyGrid(40, 40, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        commands = []
        with patch(
            "raspberry_pi.mapping.explorer.find_frontier_path",
            return_value=[],
        ), patch(
            "raspberry_pi.mapping.explorer.find_observation_path",
            return_value=[],
        ), patch(
            "raspberry_pi.mapping.explorer.find_obstacle_inspection_path",
            return_value=([], None, None, False),
        ), patch(
            "raspberry_pi.mapping.explorer.has_unresolved_obstacle_shadows",
            return_value=True,
        ):
            for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.0):
                commands.append(explorer.update(
                    Pose2D(yaw_rad=yaw),
                    front_scan(),
                    grid,
                ))

        self.assertTrue(commands[-1].finished)
        self.assertEqual("complete", commands[-1].state)

    def test_rebuilt_map_frontier_cancels_completion(self):
        config = ExplorationConfig(
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
        )
        explorer = FrontierExplorer(config)
        explorer.state = "complete"
        grid = OccupancyGrid(60, 60, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        row = int(center_row)
        col = int(center_col + 12)
        grid.observed[row - 2:row + 3, col - 2:col + 3] = False

        resumed = explorer.resume_after_map_rebuild(grid, Pose2D())

        self.assertTrue(resumed)
        self.assertEqual("exploring", explorer.state)
        self.assertTrue(explorer._path)

    def test_no_progress_forces_vehicle_to_leave_local_area(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            progress_window_updates=4,
            progress_min_span_m=0.20,
            progress_escape_updates=3,
        ))
        explorer.state = "exploring"
        commands = [
            explorer.update(
                Pose2D(),
                front_scan(timestamp=float(index + 1)),
                OccupancyGrid(40, 40, 0.1),
            )
            for index in range(4)
        ]
        self.assertEqual("progress_escape", commands[-1].state)
        self.assertGreater(commands[-1].linear_mm_s, 0)

    def test_pose_motion_toward_committed_frontier_is_progress(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            progress_window_updates=4,
            progress_min_span_m=0.20,
            progress_min_new_area_m2=0.02,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        explorer._active_frontier_target = (1.0, 0.0)
        grid = OccupancyGrid(40, 40, 0.1)

        commands = [
            explorer.update(
                Pose2D(x_m=index * 0.12),
                front_scan(timestamp=float(index + 1)),
                grid,
            )
            for index in range(4)
        ]

        self.assertNotIn(
            commands[-1].state,
            ("progress_escape", "progress_escape_turn"),
        )
        self.assertFalse(explorer._frontier_blacklist)

    def test_failed_frontier_immediately_switches_to_another_gray_boundary(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            progress_window_updates=3,
            progress_min_new_area_m2=0.02,
            robot_clearance_m=0.1,
            min_frontier_distance_m=0.1,
            min_frontier_cluster_cells=2,
            replan_every_updates=100,
        ))
        explorer.state = "exploring"
        grid = OccupancyGrid(80, 80, 0.1)
        grid.observed[:] = True
        grid.log_odds[:] = -2.0
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        row = int(center_row)
        near_col = int(center_col + 8)
        other_col = int(center_col - 18)
        grid.observed[row - 2:row + 3, near_col - 2:near_col + 3] = False
        grid.observed[row - 2:row + 3, other_col - 2:other_col + 3] = False
        explorer._path = find_frontier_path(grid, Pose2D(), explorer.config)
        explorer._active_frontier_target = explorer._path[-1]
        failed_target = explorer._active_frontier_target

        for index in range(3):
            command = explorer.update(
                Pose2D(),
                front_scan(timestamp=float(index + 1)),
                grid,
            )

        self.assertIsNotNone(explorer._active_frontier_target)
        self.assertGreater(
            math.hypot(
                explorer._active_frontier_target[0] - failed_target[0],
                explorer._active_frontier_target[1] - failed_target[1],
            ),
            explorer.config.frontier_blacklist_radius_m,
        )
        self.assertNotEqual("progress_escape", command.state)

    def test_newly_observed_map_area_counts_as_exploration_progress(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            progress_window_updates=4,
            progress_min_new_area_m2=0.02,
            replan_every_updates=100,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        grid = OccupancyGrid(40, 40, 0.1)
        commands = []
        for index in range(4):
            grid.observed[10, 10:10 + index + 1] = True
            grid.log_odds[10, 10:10 + index + 1] = -2.0
            commands.append(explorer.update(
                Pose2D(),
                front_scan(timestamp=float(index + 1)),
                grid,
            ))

        self.assertNotIn(
            commands[-1].state,
            ("progress_escape", "progress_escape_turn"),
        )

    def test_small_heading_error_uses_reduced_turn_speed(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            replan_every_updates=100,
            turn_speed_mrad_s=2500,
        ))
        explorer.state = "exploring"
        angle = math.radians(40.0)
        explorer._path = [(math.cos(angle), math.sin(angle))]
        command = explorer.update(
            Pose2D(),
            front_scan(),
            OccupancyGrid(40, 40, 0.1),
        )
        self.assertEqual("turning", command.state)
        self.assertEqual(1500, command.angular_mrad_s)
        self.assertGreater(command.linear_mm_s, 0)

    def test_small_heading_error_keeps_a_straight_command(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            replan_every_updates=100,
        ))
        explorer.state = "exploring"
        angle = math.radians(8.0)
        explorer._path = [(math.cos(angle), math.sin(angle))]

        command = explorer.update(
            Pose2D(),
            front_scan(),
            OccupancyGrid(40, 40, 0.1),
        )

        self.assertEqual("advancing", command.state)
        self.assertGreater(command.linear_mm_s, 0)
        self.assertEqual(0, command.angular_mrad_s)

    def test_narrow_passage_caps_learned_linear_and_turn_speed(self):
        explorer = FrontierExplorer(ExplorationConfig(
            cruise_speed_mm_s=300,
            narrow_passage_confirmations=2,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        grid = OccupancyGrid(40, 40, 0.1)
        explorer.update(Pose2D(), corridor_scan(timestamp=1.0), grid)
        command = explorer.update(
            Pose2D(), corridor_scan(timestamp=2.0), grid
        )
        self.assertEqual("narrow_passage_advancing", command.state)

        controller = AdaptiveSpeedController(3000, 460)
        controller.linear_adjustment_mm_s = 160
        applied = controller.apply(command)
        self.assertGreaterEqual(applied.linear_mm_s, 220)
        self.assertLessEqual(applied.linear_mm_s, 280)

        turning = MotionCommand(
            0,
            1900,
            "narrow_passage_turning",
            "synthetic corridor alignment",
        )
        controller.turn_floor_mrad_s = 3000
        controller.turn_limit_mrad_s = 3000
        self.assertEqual(2000, controller.apply(turning).angular_mrad_s)

    def test_one_close_side_does_not_trigger_narrow_passage(self):
        scan = corridor_scan(width=0.80)
        scan.distances_m[scan.angles_rad < 0.0] = 3.0
        explorer = FrontierExplorer(ExplorationConfig(
            cruise_speed_mm_s=300,
            narrow_passage_confirmations=2,
        ))
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        grid = OccupancyGrid(40, 40, 0.1)
        explorer.update(Pose2D(), scan, grid)
        command = explorer.update(Pose2D(), scan, grid)
        self.assertEqual("advancing", command.state)
        self.assertEqual(300, command.linear_mm_s)

    def test_open_space_turn_uses_forward_arc(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            slow_speed_mm_s=350,
            turn_arc_speed_mm_s=260,
        ))
        command = explorer.update(
            Pose2D(),
            front_scan(),
            OccupancyGrid(40, 40, 0.1),
        )
        self.assertEqual("initial_scan", command.state)
        self.assertEqual(260, command.linear_mm_s)
        self.assertNotEqual(0, command.angular_mrad_s)

    def test_stationary_turn_limit_forces_forward_escape(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            maximum_stationary_turn_rad=math.radians(10.0),
            turn_escape_updates=3,
        ))
        grid = OccupancyGrid(40, 40, 0.1)
        explorer.update(Pose2D(), front_scan(), grid)
        command = explorer.update(
            Pose2D(yaw_rad=math.radians(12.0)),
            front_scan(timestamp=2.0),
            grid,
        )
        self.assertEqual("turn_escape", command.state)
        self.assertGreater(command.linear_mm_s, 0)
        self.assertEqual(0, command.angular_mrad_s)

    def test_successful_waypoint_alignment_resets_turn_accumulator(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=0.0,
            maximum_stationary_turn_rad=math.radians(155.0),
        ))
        grid = OccupancyGrid(40, 40, 0.1)
        explorer.state = "exploring"
        explorer._path = [(1.0, 0.0)]
        explorer._active_frontier_target = (1.0, 0.0)
        explorer._turn_anchor_xy = (0.0, 0.0)
        explorer._last_turn_pose = (0.0, 0.0, 0.0)
        explorer._stationary_turn_accumulated = math.radians(140.0)

        command = explorer.update(Pose2D(), front_scan(), grid)

        self.assertEqual("advancing", command.state)
        self.assertEqual(0.0, explorer._stationary_turn_accumulated)
        self.assertIsNone(explorer._turn_anchor_xy)

    def test_small_circle_does_not_reset_stationary_turn_limit(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            maximum_stationary_turn_rad=math.radians(30.0),
            stationary_turn_radius_m=0.22,
        ))
        grid = OccupancyGrid(40, 40, 0.1)
        explorer.update(Pose2D(), front_scan(), grid)
        explorer.update(
            Pose2D(0.12, 0.0, math.radians(20.0)),
            front_scan(timestamp=2.0),
            grid,
        )
        command = explorer.update(
            Pose2D(0.0, 0.12, math.radians(40.0)),
            front_scan(timestamp=3.0),
            grid,
        )
        self.assertEqual("turn_escape", command.state)
        self.assertGreater(command.linear_mm_s, 0)

    def test_blocked_turn_limit_retraces_recent_approach(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            maximum_stationary_turn_rad=math.radians(10.0),
        ))
        explorer._pose_history.extend(((-0.30, 0.0), (-0.16, 0.0)))
        grid = OccupancyGrid(60, 60, 0.05)
        blocked_scan = front_scan()
        blocked_scan.distances_m[
            np.abs(blocked_scan.angles_rad) < math.radians(12)
        ] = 0.2
        explorer.update(Pose2D(), blocked_scan, grid)
        command = explorer.update(
            Pose2D(yaw_rad=math.radians(12.0)),
            blocked_scan,
            grid,
        )
        self.assertEqual("wall_escape_backoff", command.state)
        self.assertLess(command.linear_mm_s, 0)

    def test_repeated_turn_stall_does_not_spin_forever(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            turn_stall_updates=3,
        ))
        grid = OccupancyGrid(40, 40, 0.1)
        commands = [
            explorer.update(
                Pose2D(),
                front_scan(timestamp=float(index + 1)),
                grid,
            )
            for index in range(5)
        ]
        self.assertIn(
            "turn_escape",
            [command.state for command in commands],
        )

    def test_turn_stall_ignores_small_icp_translation_jitter(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            turn_stall_updates=3,
        ))
        grid = OccupancyGrid(40, 40, 0.1)
        poses = (
            Pose2D(0.000, 0.000, 0.000),
            Pose2D(0.009, -0.007, math.radians(0.2)),
            Pose2D(-0.008, 0.006, math.radians(-0.1)),
            Pose2D(0.007, 0.008, math.radians(0.1)),
            Pose2D(-0.006, -0.009, math.radians(0.0)),
        )
        commands = [
            explorer.update(
                pose,
                front_scan(timestamp=float(index + 1)),
                grid,
            )
            for index, pose in enumerate(poses)
        ]
        self.assertIn(
            "turn_escape",
            [command.state for command in commands],
        )

    def test_rotation_stops_when_vehicle_corner_would_hit(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            robot_length_m=0.35,
            robot_width_m=0.24,
            lidar_offset_x_m=0.15,
            safety_margin_m=0.04,
        ))
        scan = front_scan()
        index = np.argmin(np.abs(
            scan.angles_rad - math.radians(80.0)
        ))
        scan.distances_m[index - 1:index + 2] = 0.05
        scan.distances_m[
            np.abs(scan.angles_rad) < math.radians(3.0)
        ] = 0.2
        first = explorer.update(
            Pose2D(),
            scan,
            OccupancyGrid(40, 40, 0.1),
        )
        command = explorer.update(
            Pose2D(),
            scan,
            OccupancyGrid(40, 40, 0.1),
        )
        self.assertEqual("cautious_turn_probe", first.state)
        self.assertEqual("rotation_blocked", command.state)
        self.assertEqual((0, 0), (
            command.linear_mm_s,
            command.angular_mrad_s,
        ))

    def test_isolated_short_return_does_not_block_rotation(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
        ))
        scan = front_scan()
        index = np.argmin(np.abs(
            scan.angles_rad - math.radians(80.0)
        ))
        scan.distances_m[index] = 0.05
        command = explorer.update(
            Pose2D(),
            scan,
            OccupancyGrid(40, 40, 0.1),
        )
        self.assertEqual("initial_scan", command.state)
        self.assertNotEqual((0, 0), (
            command.linear_mm_s,
            command.angular_mrad_s,
        ))

    def test_stale_map_blocker_gets_periodic_cautious_probe(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            map_only_probe_every_updates=4,
        ))
        scan = front_scan(distance=0.45)
        grid = OccupancyGrid(40, 40, 0.1)
        center_col, center_row = grid.world_to_grid(
            np.asarray(((0.0, 0.0),))
        )[0]
        row = int(center_row)
        col = int(center_col)
        grid.observed[row, col - 1:col + 2] = True
        grid.log_odds[row, col - 1:col + 2] = 2.0
        commands = [
            explorer.update(Pose2D(), scan, grid)
            for _ in range(4)
        ]
        self.assertEqual("rotation_blocked", commands[1].state)
        self.assertEqual("cautious_turn_probe", commands[3].state)
        self.assertNotEqual(0, commands[3].angular_mrad_s)

    def test_adaptive_turn_speed_reacts_to_scan_rotation(self):
        controller = AdaptiveSpeedController(2500)
        requested = MotionCommand(0, 2500, "turning", "test")
        applied = controller.apply(requested)
        self.assertEqual(2200, applied.angular_mrad_s)
        self.assertEqual(500, controller.ttl_for(applied, 700))

        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.03,
            100,
            120,
            inlier_ratio=0.9,
            rotation_rad=math.radians(14.0),
        ))
        applied = controller.apply(requested)
        self.assertEqual(2100, applied.angular_mrad_s)
        self.assertEqual(450, controller.ttl_for(applied, 700))

        for _ in range(3):
            controller.observe(SlamUpdate(
                Pose2D(),
                True,
                0.01,
                110,
                120,
                inlier_ratio=0.95,
                rotation_rad=math.radians(0.2),
            ))
            controller.apply(requested)
        self.assertEqual(2200, controller.turn_limit_mrad_s)
        for _ in range(20):
            controller.observe(SlamUpdate(
                Pose2D(),
                True,
                0.01,
                110,
                120,
                inlier_ratio=0.95,
                rotation_rad=0.0,
            ))
            applied = controller.apply(requested)
        self.assertLessEqual(
            controller.ttl_for(applied, 700),
            controller.config.maximum_turn_ttl_ms,
        )

    def test_adaptive_controller_limits_arc_turn_and_uses_short_ttl(self):
        controller = AdaptiveSpeedController(2500, 500)
        controller.linear_adjustment_mm_s = 160
        command = controller.apply(
            MotionCommand(220, 2500, "turning", "arc")
        )
        self.assertEqual(220, command.linear_mm_s)
        self.assertEqual(2200, command.angular_mrad_s)
        self.assertEqual(500, controller.ttl_for(command, 700))

    def test_stalled_arc_learns_more_turning_torque(self):
        controller = AdaptiveSpeedController(2600, 460)
        requested = MotionCommand(
            220,
            1500,
            "initial_scan",
            "test startup arc",
        )
        stalled = SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.95,
            translation_m=0.002,
            rotation_rad=math.radians(0.2),
            linear_speed_m_s=0.02,
            angular_speed_rad_s=0.03,
        )

        for _ in range(3):
            controller.apply(requested)
            controller.observe(stalled)

        self.assertEqual(1600, controller.turn_floor_mrad_s)
        self.assertEqual(2300, controller.turn_limit_mrad_s)
        applied = controller.apply(requested)
        self.assertEqual(220, applied.linear_mm_s)
        self.assertEqual(1600, applied.angular_mrad_s)

    def test_narrow_passage_stall_boost_has_starting_headroom(self):
        controller = AdaptiveSpeedController(
            2600,
            460,
            AdaptiveSpeedConfig(
                maximum_narrow_passage_linear_speed_mm_s=400,
            ),
        )
        controller.linear_adjustment_mm_s = 160

        applied = controller.apply(MotionCommand(
            240,
            0,
            "narrow_passage_advancing",
            "test narrow starting torque",
        ))

        self.assertEqual(400, applied.linear_mm_s)

    def test_poor_arc_match_reduces_instead_of_learning_more_torque(self):
        controller = AdaptiveSpeedController(2600, 460)
        requested = MotionCommand(
            220,
            1900,
            "initial_scan",
            "test poor startup arc",
        )
        controller.apply(requested)
        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.14,
            60,
            120,
            inlier_ratio=0.45,
            translation_m=0.002,
            rotation_rad=math.radians(0.2),
            linear_speed_m_s=0.02,
            angular_speed_rad_s=0.03,
        ))

        self.assertEqual(1500, controller.turn_floor_mrad_s)
        self.assertEqual(2100, controller.turn_limit_mrad_s)

    def test_cautious_turn_is_not_boosted_but_straight_escape_is(self):
        controller = AdaptiveSpeedController(2500, 500)
        controller.turn_floor_mrad_s = 2500
        controller.linear_adjustment_mm_s = 160
        turn = controller.apply(MotionCommand(
            0,
            1500,
            "cautious_turn_probe",
            "test",
        ))
        self.assertEqual(1900, turn.angular_mrad_s)
        self.assertEqual(500, controller.ttl_for(turn, 700))
        escape = controller.apply(MotionCommand(
            260,
            0,
            "turn_escape",
            "test",
        ))
        self.assertEqual(420, escape.linear_mm_s)

    def test_turn_escape_continues_with_its_entry_clearance(self):
        explorer = FrontierExplorer(ExplorationConfig(
            initial_spin_rad=10.0,
            rotation_block_confirmations=1,
            stop_distance_m=0.40,
            turn_escape_front_margin_m=0.18,
        ))
        scan = front_scan(distance=0.52)
        grid = OccupancyGrid(80, 80, 0.05)
        explorer._rotation_block_sources = lambda *_args, **_kwargs: (
            True,
            False,
        )
        explorer._forward_corridor_is_safe = lambda *_args, **_kwargs: True

        first = explorer.update(Pose2D(), scan, grid)
        second = explorer.update(Pose2D(), scan, grid)

        self.assertEqual("turn_escape", first.state)
        self.assertEqual("turn_escape", second.state)
        self.assertGreater(second.linear_mm_s, 0)

    def test_adaptive_linear_speed_boosts_when_stalled(self):
        controller = AdaptiveSpeedController(2500, 500)
        requested = MotionCommand(350, 0, "advancing", "test")
        for _ in range(3):
            controller.apply(requested)
            controller.observe(SlamUpdate(
                Pose2D(),
                True,
                0.02,
                100,
                120,
                inlier_ratio=0.9,
                translation_m=0.002,
            ))
        applied = controller.apply(requested)
        self.assertEqual(390, applied.linear_mm_s)
        self.assertEqual(40, controller.linear_adjustment_mm_s)

    def test_repeated_linear_stall_accelerates_faster(self):
        controller = AdaptiveSpeedController(2500, 500)
        requested = MotionCommand(300, 0, "advancing", "test")
        stalled_update = SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.9,
            translation_m=0.002,
        )
        for _ in range(3):
            controller.apply(requested)
            controller.observe(stalled_update)
        self.assertEqual(40, controller.linear_adjustment_mm_s)

        for _ in range(2):
            controller.apply(requested)
            controller.observe(stalled_update)
        self.assertEqual(120, controller.linear_adjustment_mm_s)
        self.assertEqual(420, controller.apply(requested).linear_mm_s)

        controller.observe(stalled_update)
        self.assertEqual(220, controller.linear_adjustment_mm_s)
        self.assertEqual(500, controller.apply(requested).linear_mm_s)

    def test_stop_and_ble_reconnect_preserve_starting_boost(self):
        controller = AdaptiveSpeedController(2500, 500)
        controller.linear_adjustment_mm_s = 120
        requested = MotionCommand(300, 0, "advancing", "test")
        controller.apply(requested)

        controller.stopped()

        self.assertEqual(120, controller.linear_adjustment_mm_s)
        self.assertEqual(420, controller.apply(requested).linear_mm_s)

    def test_ble_interruption_uses_stronger_restart_immediately(self):
        controller = AdaptiveSpeedController(2500, 460)
        requested = MotionCommand(300, 0, "advancing", "test")
        controller.apply(requested)

        self.assertTrue(controller.note_link_interruption())
        self.assertEqual(120, controller.linear_adjustment_mm_s)
        self.assertEqual(420, controller.apply(requested).linear_mm_s)

        self.assertTrue(controller.note_link_interruption())
        self.assertEqual(160, controller.linear_adjustment_mm_s)
        self.assertEqual(460, controller.apply(requested).linear_mm_s)

    def test_ble_interruption_does_not_boost_stationary_turn(self):
        controller = AdaptiveSpeedController(2500, 460)
        controller.apply(MotionCommand(0, 1800, "turning", "test"))

        self.assertFalse(controller.note_link_interruption())
        self.assertEqual(0, controller.linear_adjustment_mm_s)

    def test_adaptive_linear_speed_reduces_after_large_motion(self):
        controller = AdaptiveSpeedController(2500, 500)
        controller.linear_adjustment_mm_s = 80
        requested = MotionCommand(350, 0, "advancing", "test")
        controller.apply(requested)
        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.03,
            100,
            120,
            inlier_ratio=0.9,
            translation_m=0.10,
        ))
        self.assertEqual(40, controller.linear_adjustment_mm_s)

    def test_adaptive_linear_speed_can_drop_below_requested_speed(self):
        controller = AdaptiveSpeedController(2500, 500)
        requested = MotionCommand(350, 0, "advancing", "test")
        controller.apply(requested)
        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.03,
            100,
            120,
            inlier_ratio=0.9,
            translation_m=0.10,
        ))
        applied = controller.apply(requested)
        self.assertEqual(310, applied.linear_mm_s)
        self.assertEqual(-40, controller.linear_adjustment_mm_s)

    def test_adaptive_speed_keeps_enough_output_to_move(self):
        controller = AdaptiveSpeedController(2500, 500)
        controller.linear_adjustment_mm_s = -80
        requested = MotionCommand(180, 0, "advancing", "test")

        applied = controller.apply(requested)

        self.assertEqual(220, applied.linear_mm_s)

    def test_advancing_steering_is_not_promoted_to_stationary_turn_floor(self):
        controller = AdaptiveSpeedController(2500, 500)
        requested = MotionCommand(300, 250, "advancing", "test")

        applied = controller.apply(requested)

        self.assertEqual(250, applied.angular_mrad_s)

    def test_adaptive_linear_speed_uses_lidar_measured_speed(self):
        controller = AdaptiveSpeedController(2500, 500)
        requested = MotionCommand(400, 0, "advancing", "test")
        controller.apply(requested)

        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.9,
            translation_m=0.01,
            linear_speed_m_s=0.28,
        ))

        self.assertEqual(-40, controller.linear_adjustment_mm_s)
        self.assertEqual(360, controller.apply(requested).linear_mm_s)

    def test_adaptive_turn_floor_rises_when_stalled(self):
        controller = AdaptiveSpeedController(2500)
        requested = MotionCommand(0, 1500, "turning", "test")
        for _ in range(3):
            controller.apply(requested)
            controller.observe(SlamUpdate(
                Pose2D(),
                True,
                0.02,
                100,
                120,
                inlier_ratio=0.9,
                rotation_rad=math.radians(0.2),
            ))
        applied = controller.apply(requested)
        self.assertEqual(1600, applied.angular_mrad_s)

    def test_nominal_turn_speed_does_not_cap_stall_boost(self):
        controller = AdaptiveSpeedController(3000)
        requested = MotionCommand(0, 1500, "turning", "test")
        self.assertEqual(1500, controller.apply(requested).angular_mrad_s)
        for _ in range(6):
            controller.apply(requested)
            controller.observe(SlamUpdate(
                Pose2D(),
                True,
                0.02,
                100,
                120,
                inlier_ratio=0.9,
                rotation_rad=math.radians(0.2),
            ))
        self.assertGreater(
            controller.apply(requested).angular_mrad_s,
            1500,
        )

    def test_narrow_avoidance_turn_escapes_its_normal_speed_cap(self):
        controller = AdaptiveSpeedController(
            2600,
            config=AdaptiveSpeedConfig(
                maximum_narrow_passage_turn_mrad_s=1700,
                maximum_narrow_passage_stall_turn_mrad_s=2600,
            ),
        )
        requested = MotionCommand(
            0,
            1700,
            "narrow_passage_turning",
            "front obstacle inside stop distance",
        )
        stalled = SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.90,
            rotation_rad=math.radians(0.2),
            angular_speed_rad_s=0.03,
        )

        self.assertEqual(1700, controller.apply(requested).angular_mrad_s)
        for _ in range(3):
            controller.apply(requested)
            controller.observe(stalled)
        self.assertEqual(1700, controller.apply(requested).angular_mrad_s)

        for _ in range(2):
            controller.apply(requested)
            controller.observe(stalled)
        self.assertEqual(1800, controller.apply(requested).angular_mrad_s)

        controller.observe(stalled)
        self.assertEqual(2000, controller.apply(requested).angular_mrad_s)
        self.assertLessEqual(
            controller.apply(requested).angular_mrad_s,
            controller.maximum_turn_speed_mrad_s,
        )

    def test_turn_start_boost_is_released_as_soon_as_yaw_moves(self):
        controller = AdaptiveSpeedController(
            2200,
            config=AdaptiveSpeedConfig(
                maximum_narrow_passage_turn_mrad_s=1700,
                maximum_narrow_passage_stall_turn_mrad_s=2200,
            ),
        )
        requested = MotionCommand(
            0,
            1700,
            "narrow_passage_turning",
            "front obstacle inside stop distance",
        )
        controller.turn_floor_mrad_s = 2200
        controller.turn_limit_mrad_s = 2200
        controller._narrow_turn_stall_boost_mrad_s = 500
        controller._turn_stall_events = 3
        controller.apply(requested)

        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.90,
            rotation_rad=math.radians(3.0),
            angular_speed_rad_s=0.35,
        ))

        self.assertEqual(1500, controller.turn_floor_mrad_s)
        self.assertEqual(2200, controller.turn_limit_mrad_s)
        self.assertEqual(0, controller._narrow_turn_stall_boost_mrad_s)
        self.assertEqual(1700, controller.apply(requested).angular_mrad_s)

    def test_normal_turn_uses_requested_speed_until_stall_is_confirmed(self):
        controller = AdaptiveSpeedController(2600)

        applied = controller.apply(MotionCommand(
            0,
            1500,
            "turning",
            "normal heading alignment",
        ))

        self.assertEqual(1500, applied.angular_mrad_s)

    def test_moving_arc_still_learns_when_only_turn_axis_is_stalled(self):
        controller = AdaptiveSpeedController(2600, 460)
        requested = MotionCommand(
            220,
            1500,
            "initial_scan",
            "translate while acquiring initial heading coverage",
        )
        update = SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.90,
            translation_m=0.010,
            rotation_rad=0.0,
            linear_speed_m_s=0.10,
            angular_speed_rad_s=0.0,
        )

        for _ in range(6):
            controller.apply(requested)
            controller.observe(update)

        applied = controller.apply(requested)
        self.assertGreater(applied.angular_mrad_s, 1500)
        self.assertEqual(220, applied.linear_mm_s)

    def test_fast_arc_does_not_reduce_later_straight_drive_torque(self):
        controller = AdaptiveSpeedController(2600, 460)
        controller.linear_adjustment_mm_s = 120
        controller.apply(MotionCommand(
            230,
            1700,
            "narrow_passage_turning",
            "clearance checked arc",
        ))

        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.90,
            translation_m=0.004,
            rotation_rad=math.radians(8.0),
            linear_speed_m_s=0.04,
            angular_speed_rad_s=0.90,
        ))

        self.assertEqual(120, controller.linear_adjustment_mm_s)

    def test_confirmed_motion_uses_boost_only_as_starting_kick(self):
        controller = AdaptiveSpeedController(
            2600,
            460,
            AdaptiveSpeedConfig(
                maximum_narrow_passage_linear_speed_mm_s=460,
            ),
        )
        requested = MotionCommand(
            240,
            0,
            "narrow_passage_advancing",
            "test transient start boost",
        )
        controller.linear_adjustment_mm_s = 220
        self.assertEqual(460, controller.apply(requested).linear_mm_s)

        controller.observe(SlamUpdate(
            Pose2D(),
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.90,
            translation_m=0.010,
            linear_speed_m_s=0.10,
        ))

        self.assertEqual(240, controller.apply(requested).linear_mm_s)

    def test_script_uses_ble_client_and_saves_map(self):
        client = FakeClient()
        lidar = FakeLidar(tuple(
            front_scan(timestamp=1.0 + index * 0.1)
            for index in range(7)
        ))
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
            ])
            paths = run(args, client=client, lidar=lidar)
            self.assertEqual(1, client.heartbeat_count)
            self.assertTrue(client.commands)
            self.assertTrue(all(command[0] >= 0 for command in client.commands))
            self.assertTrue(all(200 <= command[2] <= 700 for command in client.commands))
            self.assertTrue(client.stopped)
            self.assertTrue(client.closed)
            self.assertTrue(lidar.closed)
            self.assertTrue(all(path.exists() for path in paths))

    def test_live_map_receives_real_slam_updates_and_stops(self):
        class FakeLiveMap:
            def __init__(self):
                self.started = 0
                self.published = []
                self.stopped = 0

            def start(self):
                self.started += 1
                return "http://192.0.2.14:8766"

            def publish(self, slam, status, force=False):
                self.published.append((slam, status, force))
                return True

            def stop(self):
                self.stopped += 1

        client = FakeClient()
        lidar = FakeLidar(tuple(
            front_scan(timestamp=1.0 + index * 0.1)
            for index in range(7)
        ))
        live_map = FakeLiveMap()
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--live-map",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
            ])
            run(
                args,
                client=client,
                lidar=lidar,
                live_map_server=live_map,
            )

        self.assertEqual(1, live_map.started)
        self.assertGreaterEqual(len(live_map.published), 2)
        self.assertTrue(any(
            item[1]["mapping"]["integrated_count"] >= 1
            for item in live_map.published
        ))
        self.assertEqual(7, live_map.published[-1][1]["scan_count"])
        self.assertFalse(live_map.published[-1][1]["running"])
        self.assertTrue(live_map.published[-1][2])
        self.assertEqual(1, live_map.stopped)

    def test_first_rejected_scan_stops_before_motion_resumes(self):
        client = FakeClient()
        sparse = LaserScan(
            np.linspace(-0.2, 0.2, 8),
            np.full(8, 2.0),
            2.0,
        )
        stable_scans = tuple(
            front_scan(timestamp=1.0 + index * 0.1)
            for index in range(7)
        )
        lidar = FakeLidar(
            stable_scans
            + (
                sparse,
                front_scan(timestamp=3.0),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
            ])
            run(args, client=client, lidar=lidar)
        first_stop = client.events.index(("stop",))
        commands_after_stop = [
            event for event in client.events[first_stop + 1:]
            if event[0] == "command"
        ]
        self.assertTrue(commands_after_stop)

    def test_link_timeout_reconnects_and_reseeds_instead_of_exiting(self):
        class FailingClient(FakeClient):
            def set_twist(self, linear, angular, ttl_ms):
                raise LinkError("simulated BLE timeout")

        initial_client = FailingClient()
        recovered_clients = []

        def factory():
            recovered = FakeClient()
            recovered_clients.append(recovered)
            return recovered

        lidar = FakeLidar((
            front_scan(timestamp=1.0),
            front_scan(timestamp=2.0),
            front_scan(timestamp=3.0),
        ))
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
                "--link-reconnect-delay", "0.1",
                "--reconnect-discard-scans", "0",
            ])
            paths = run(
                args,
                client=initial_client,
                lidar=lidar,
                client_factory=factory,
            )
            self.assertTrue(all(path.exists() for path in paths))
        self.assertTrue(initial_client.closed)
        self.assertEqual(1, len(recovered_clients))
        self.assertGreaterEqual(recovered_clients[0].heartbeat_count, 1)
        self.assertTrue(recovered_clients[0].commands)

    def test_false_room_box_excluding_vehicle_is_not_accepted(self):
        grid = OccupancyGrid(
            100, 100, 0.10, min_obstacle_area_m2=0.0
        )
        occupied = np.zeros((100, 100), dtype=bool)
        occupied[40, 40:61] = True
        occupied[60, 40:61] = True
        occupied[40:61, 40] = True
        occupied[40:61, 60] = True
        pose = Pose2D(x_m=3.0, y_m=0.0)
        config = ExplorationConfig(
            enclosed_room_min_span_m=1.4,
            enclosed_room_min_edge_cells=3,
        )

        self.assertIsNotNone(
            _enclosed_room_bounds(occupied, config, grid.resolution_m)
        )
        self.assertIsNone(_enclosed_room_bounds(
            occupied,
            config,
            grid.resolution_m,
            required_cells=_room_required_cells(grid, pose),
        ))

    def test_furniture_face_cannot_hide_confirmed_free_space_as_room_wall(self):
        occupied = np.zeros((100, 100), dtype=bool)
        occupied[20, 20:81] = True
        occupied[80, 20:81] = True
        occupied[20:81, 20] = True
        # No outer right wall has been observed yet.  This furniture face is
        # the strongest remaining vertical projection but must not close the
        # room at column 55.
        occupied[35:66, 55] = True
        confirmed_free = np.zeros_like(occupied)
        confirmed_free[25:76, 25:76] = True
        config = ExplorationConfig(
            enclosed_room_min_span_m=1.4,
            enclosed_room_min_edge_cells=3,
        )

        provisional = _enclosed_room_bounds(
            occupied,
            config,
            0.10,
            required_cells=((50, 30),),
        )

        self.assertIsNotNone(provisional)
        self.assertIsNone(_enclosed_room_bounds(
            occupied,
            config,
            0.10,
            required_cells=((50, 30),),
            confirmed_free=confirmed_free,
        ))

    def test_obstacle_shadow_prefers_oblique_view_over_seen_front(self):
        shadows = [12] + [0] * 11

        tangent_gain, _ = _inspection_view_gain(shadows, 2)
        direct_gain, _ = _inspection_view_gain(shadows, 0)
        opposite_gain, _ = _inspection_view_gain(shadows, 6)

        self.assertGreater(tangent_gain, direct_gain)
        self.assertGreater(direct_gain, opposite_gain)

    def test_no_path_can_backtrack_along_driven_breadcrumbs(self):
        explorer = FrontierExplorer(ExplorationConfig(
            history_backtrack_distance_m=0.70,
            history_breadcrumb_spacing_m=0.15,
        ))
        explorer._breadcrumbs.extend(
            (float(index) * 0.20, 0.0) for index in range(10)
        )

        started = explorer._start_history_backtrack(
            Pose2D(x_m=1.8, y_m=0.0)
        )

        self.assertTrue(started)
        self.assertTrue(explorer._backtracking_active)
        self.assertGreaterEqual(len(explorer._path), 3)
        self.assertLess(explorer._path[-1][0], 1.3)

    def test_motion_reconnect_waits_for_a_later_successful_round(self):
        class FailingClient(FakeClient):
            def set_twist(self, linear, angular, ttl_ms):
                raise LinkError("simulated BLE timeout")

        initial_client = FailingClient()
        factory_calls = []
        recovered = FakeClient()

        def factory():
            factory_calls.append(1)
            if len(factory_calls) == 1:
                raise RuntimeError("BlueZ still releasing old connection")
            return recovered

        lidar = FakeLidar((
            front_scan(timestamp=1.0),
            front_scan(timestamp=2.0),
            front_scan(timestamp=3.0),
        ))
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
                "--link-reconnect-attempts", "1",
                "--link-reconnect-delay", "0.1",
                "--reconnect-discard-scans", "0",
            ])
            with patch("raspberry_pi.autonomous_mapping.time.sleep"):
                paths = run(
                    args,
                    client=initial_client,
                    lidar=lidar,
                    client_factory=factory,
                )
            self.assertTrue(all(path.exists() for path in paths))

        self.assertEqual(2, len(factory_calls))
        self.assertTrue(recovered.commands)

    def test_startup_heartbeat_timeout_reconnects_before_lidar_starts(self):
        class StartupFailingClient(FakeClient):
            def heartbeat(self):
                self.heartbeat_count += 1
                raise LinkError("simulated startup heartbeat timeout")

        initial_client = StartupFailingClient()
        recovered_clients = []

        def factory():
            recovered = FakeClient()
            recovered_clients.append(recovered)
            return recovered

        lidar = FakeLidar((front_scan(timestamp=1.0),))
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
                "--link-reconnect-delay", "0.1",
            ])
            paths = run(
                args,
                client=initial_client,
                lidar=lidar,
                client_factory=factory,
            )
            self.assertTrue(all(path.exists() for path in paths))
        self.assertTrue(initial_client.closed)
        self.assertEqual(1, len(recovered_clients))
        self.assertGreaterEqual(recovered_clients[0].heartbeat_count, 1)

    def test_initial_client_creation_uses_persistent_reconnect_policy(self):
        factory_calls = []
        recovered = FakeClient()

        def factory():
            factory_calls.append(1)
            if len(factory_calls) == 1:
                raise RuntimeError("simulated initial BlueZ timeout")
            return recovered

        lidar = FakeLidar((front_scan(timestamp=1.0),))
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
                "--link-reconnect-attempts", "1",
                "--link-reconnect-rounds", "0",
                "--link-reconnect-delay", "0.1",
            ])
            with patch("raspberry_pi.autonomous_mapping.time.sleep"):
                paths = run(
                    args,
                    lidar=lidar,
                    client_factory=factory,
                )
            self.assertTrue(all(path.exists() for path in paths))

        self.assertEqual(2, len(factory_calls))
        self.assertGreaterEqual(recovered.heartbeat_count, 2)
        self.assertTrue(recovered.closed)

    def test_exhausted_motion_reconnect_saves_without_raising(self):
        class FailingClient(FakeClient):
            def set_twist(self, linear, angular, ttl_ms):
                raise LinkError("simulated persistent BLE outage")

        initial_client = FailingClient()

        def factory():
            raise RuntimeError("simulated service discovery failure")

        lidar = FakeLidar((
            front_scan(timestamp=1.0),
            front_scan(timestamp=2.0),
        ))
        with tempfile.TemporaryDirectory() as directory:
            args = build_argument_parser().parse_args([
                "--enable-motion",
                "--map-size", "5",
                "--output", os.path.join(directory, "auto"),
                "--save-every", "0",
                "--link-reconnect-attempts", "1",
                "--link-reconnect-rounds", "1",
                "--link-reconnect-delay", "0.1",
                "--reconnect-discard-scans", "0",
            ])
            paths = run(
                args,
                client=initial_client,
                lidar=lidar,
                client_factory=factory,
            )
            self.assertTrue(all(path.exists() for path in paths))

        self.assertTrue(initial_client.closed)


if __name__ == "__main__":
    unittest.main()
