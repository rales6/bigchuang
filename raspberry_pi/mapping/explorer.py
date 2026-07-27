"""基于占据栅格前沿的保守型自主探索器。

探索器只输出前进和原地转向，不会倒车，因为这里只信任车头方向的雷达数据。
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .geometry import normalize_angle
from .scan_filter import sector_min_distance


@dataclass(frozen=True)
class ExplorationConfig:
    cruise_speed_mm_s: int = 250
    slow_speed_mm_s: int = 220
    turn_speed_mrad_s: int = 2500
    stop_distance_m: float = 0.40
    slow_distance_m: float = 0.75
    robot_clearance_m: float = 0.30
    waypoint_tolerance_m: float = 0.16
    heading_tolerance_deg: float = 32.0
    straight_heading_deadband_deg: float = 10.0
    straight_angular_gain: float = 700.0
    minimum_turn_speed_mrad_s: int = 1500
    turn_proportional_gain: float = 1800.0
    initial_spin_rad: float = math.radians(45.0)
    # A selected branch is committed until it is reached or proven blocked.
    # Kept for API compatibility; periodic target replacement is disabled.
    replan_every_updates: int = 0
    min_frontier_cluster_cells: int = 4
    min_frontier_distance_m: float = 0.35
    completion_confirmations: int = 12
    completion_spin_rad: float = math.radians(60.0)
    min_completion_travel_m: float = 3.0
    free_probability: float = 0.45
    occupied_probability: float = 0.65
    recovery_open_margin_m: float = 0.25
    robot_length_m: float = 0.35
    robot_width_m: float = 0.24
    lidar_offset_x_m: float = 0.14
    lidar_offset_y_m: float = 0.0
    safety_margin_m: float = 0.04
    turn_arc_speed_mm_s: int = 260
    arc_sweep_extra_margin_m: float = 0.08
    arc_front_margin_m: float = 0.12
    maximum_stationary_turn_rad: float = math.radians(55.0)
    stationary_turn_radius_m: float = 0.22
    turn_stall_yaw_deg: float = 1.0
    turn_stall_updates: int = 6
    turn_escape_updates: int = 10
    turn_escape_front_margin_m: float = 0.18
    rotation_obstacle_min_points: int = 3
    rotation_obstacle_max_angle_gap_deg: float = 6.0
    rotation_block_confirmations: int = 2
    map_only_probe_every_updates: int = 8
    progress_window_updates: int = 20
    progress_min_span_m: float = 0.35
    progress_min_new_area_m2: float = 0.04
    progress_escape_updates: int = 14
    progress_escape_front_margin_m: float = 0.12
    # Committing to the farthest reachable white/unknown boundary produces a
    # useful translation goal instead of repeatedly orbiting nearby frontiers.
    prefer_distant_frontiers: bool = True
    frontier_information_radius_m: float = 0.60
    frontier_information_weight: float = 1.8
    # Among frontiers whose direct distances are nearly tied, prefer the one
    # already closer to the vehicle heading. This prevents left/right target
    # flips while retaining the request to explore distant boundaries first.
    frontier_distance_slack_m: float = 0.35
    frontier_blacklist_radius_m: float = 0.45
    frontier_blacklist_updates: int = 160
    path_segment_max_m: float = 0.45
    wall_escape_backoff_m: float = 0.18
    wall_escape_reverse_speed_mm_s: int = 450
    wall_escape_turn_rad: float = math.radians(55.0)
    wall_escape_forward_updates: int = 10
    wall_escape_max_reverse_updates: int = 20
    wall_escape_min_known_ratio: float = 0.85
    wall_escape_corridor_margin_m: float = 0.04


@dataclass(frozen=True)
class MotionCommand:
    linear_mm_s: int
    angular_mrad_s: int
    state: str
    reason: str
    target_xy_m: object = None
    finished: bool = False


class FrontierExplorer:
    def __init__(self, config=None):
        self.config = config or ExplorationConfig()
        self.state = "initial_scan"
        self._last_yaw = None
        self._spin_accumulated = 0.0
        self._updates = 0
        self._path = []
        self._no_frontier_count = 0
        self._last_rescan_yaw = None
        self._rescan_accumulated = 0.0
        self._previous_pose_xy = None
        self._travelled_m = 0.0
        self._turn_anchor_xy = None
        self._last_turn_pose = None
        self._stationary_turn_accumulated = 0.0
        self._turn_stall_count = 0
        self._turn_escape_updates_left = 0
        self._rotation_blocked_updates = 0
        self._progress_history = deque(
            maxlen=max(2, self.config.progress_window_updates)
        )
        self._coverage_history = deque(
            maxlen=max(2, self.config.progress_window_updates)
        )
        self._progress_escape_updates_left = 0
        self._active_frontier_target = None
        self._frontier_blacklist = []
        self._pose_history = deque(maxlen=80)
        self._wall_escape_phase = None
        self._wall_escape_direction = 1
        self._wall_escape_anchor_xy = None
        self._wall_escape_last_yaw = None
        self._wall_escape_turn_accumulated = 0.0
        self._wall_escape_updates_left = 0

    def update(self, pose, scan, grid):
        """根据最新可靠位姿、前向扫描和地图生成一条短时速度命令。"""
        self._updates += 1
        current_xy = (pose.x_m, pose.y_m)
        if self._previous_pose_xy is not None:
            self._travelled_m += math.hypot(
                current_xy[0] - self._previous_pose_xy[0],
                current_xy[1] - self._previous_pose_xy[1],
            )
        self._previous_pose_xy = current_xy
        self._pose_history.append(current_xy)
        if len(scan.angles_rad) < 20:
            return MotionCommand(0, 0, "stopped", "usable scan has too few points")

        wall_escape = self._wall_escape_command(pose, scan, grid)
        if wall_escape is not None:
            return wall_escape

        if self.state != "initial_scan":
            self._progress_history.append(current_xy)
            self._coverage_history.append(
                int(np.count_nonzero(grid.observed))
            )
            progress_command = self._progress_escape_command(
                pose,
                scan,
                grid,
            )
            if progress_command is not None:
                return progress_command

        if self._turn_escape_updates_left > 0:
            front = sector_min_distance(scan, 0.0, 32.0)
            required = (
                self.config.stop_distance_m
                + self.config.turn_escape_front_margin_m
            )
            if front >= required:
                self._turn_escape_updates_left -= 1
                self._reset_turn_tracking()
                return MotionCommand(
                    min(
                        self.config.slow_speed_mm_s,
                        self.config.turn_arc_speed_mm_s,
                    ),
                    0,
                    "turn_escape",
                    "moving away from an unproductive turn",
                )
            self._turn_escape_updates_left = 0

        if self.state == "initial_scan":
            if self._last_yaw is not None:
                self._spin_accumulated += abs(
                    normalize_angle(pose.yaw_rad - self._last_yaw)
                )
            self._last_yaw = pose.yaw_rad
            if self._spin_accumulated < self.config.initial_spin_rad:
                return self._turn_command(
                    self.config.turn_speed_mrad_s,
                    "initial_scan",
                    "rotating to observe the surroundings",
                    pose,
                    scan,
                    grid,
                )
            self.state = "exploring"
            self._path = []
            self._reset_progress_tracking()
            self._reset_turn_tracking()

        # Depth-first-style commitment: do not periodically replace a valid
        # path merely because a newer/nearer frontier appeared elsewhere.
        should_replan = not self._path
        if should_replan:
            self._expire_frontier_blacklist()
            new_path = find_frontier_path(
                grid,
                pose,
                self.config,
                excluded_targets=self._frontier_blacklist,
            )
            new_target = new_path[-1] if new_path else None
            if self._frontier_target_changed(new_target):
                self._reset_progress_tracking()
            self._path = new_path
            self._active_frontier_target = new_target
            if not self._path:
                self._no_frontier_count += 1
                if self._last_rescan_yaw is not None:
                    self._rescan_accumulated += abs(normalize_angle(
                        pose.yaw_rad - self._last_rescan_yaw
                    ))
                self._last_rescan_yaw = pose.yaw_rad
                finished = (
                    self._no_frontier_count
                    >= self.config.completion_confirmations
                    and self._rescan_accumulated
                    >= self.config.completion_spin_rad
                    and self._travelled_m
                    >= self.config.min_completion_travel_m
                )
                if finished:
                    self.state = "complete"
                if not finished:
                    return self._recovery_command(scan, pose, grid)
                return MotionCommand(
                    0,
                    0,
                    self.state,
                    "no reachable frontier",
                    finished=finished,
                )
            self._no_frontier_count = 0
            self._last_rescan_yaw = None
            self._rescan_accumulated = 0.0

        target = self._select_waypoint(pose)
        if target is None:
            self._path = []
            self._reset_turn_tracking()
            return MotionCommand(0, 0, "exploring", "replanning")

        dx = target[0] - pose.x_m
        dy = target[1] - pose.y_m
        heading_error = normalize_angle(math.atan2(dy, dx) - pose.yaw_rad)
        heading_tolerance = math.radians(self.config.heading_tolerance_deg)
        if abs(heading_error) > heading_tolerance:
            magnitude = int(max(
                self.config.minimum_turn_speed_mrad_s,
                min(
                    self.config.turn_speed_mrad_s,
                    abs(heading_error) * self.config.turn_proportional_gain,
                ),
            ))
            angular = (
                magnitude if heading_error > 0 else -magnitude
            )
            return self._turn_command(
                angular,
                "turning",
                "aligning with frontier path",
                pose,
                scan,
                grid,
                target,
            )

        front = sector_min_distance(scan, 0.0, 36.0)
        left = sector_min_distance(scan, 55.0, 55.0)
        right = sector_min_distance(scan, -55.0, 55.0)
        if front <= self.config.stop_distance_m:
            # 后方雷达不可信，因此不倒车，只朝更开阔的一侧原地转向。
            angular = (
                self.config.turn_speed_mrad_s
                if left >= right
                else -self.config.turn_speed_mrad_s
            )
            self._blacklist_active_frontier()
            self._path = []
            self._active_frontier_target = None
            if self._begin_wall_escape(pose, grid, angular):
                return self._wall_escape_command(pose, scan, grid)
            return self._turn_command(
                angular,
                "avoidance",
                "front obstacle inside stop distance",
                pose,
                scan,
                grid,
                target,
            )

        speed = self.config.cruise_speed_mm_s
        if front < self.config.slow_distance_m:
            speed = self.config.slow_speed_mm_s
        deadband = math.radians(
            self.config.straight_heading_deadband_deg
        )
        if abs(heading_error) <= deadband:
            angular = 0
        else:
            correction = (
                abs(heading_error) - deadband
            ) * self.config.straight_angular_gain
            angular = int(math.copysign(
                min(self.config.turn_speed_mrad_s, correction),
                heading_error,
            ))
        self._reset_turn_tracking()
        return MotionCommand(
            speed, angular, "advancing", "following frontier path", target
        )

    def _begin_wall_escape(self, pose, grid, angular):
        if not self._reverse_corridor_is_safe(pose, grid):
            return False
        self._path = []
        self._reset_turn_tracking()
        self._wall_escape_phase = "backoff"
        self._wall_escape_direction = 1 if angular >= 0 else -1
        self._wall_escape_anchor_xy = (pose.x_m, pose.y_m)
        self._wall_escape_last_yaw = pose.yaw_rad
        self._wall_escape_turn_accumulated = 0.0
        self._wall_escape_updates_left = (
            self.config.wall_escape_max_reverse_updates
        )
        return True

    def _wall_escape_command(self, pose, scan, grid):
        if self._wall_escape_phase is None:
            return None

        if self._wall_escape_phase == "backoff":
            travelled = math.hypot(
                pose.x_m - self._wall_escape_anchor_xy[0],
                pose.y_m - self._wall_escape_anchor_xy[1],
            )
            if (
                travelled >= self.config.wall_escape_backoff_m
                or self._wall_escape_updates_left <= 0
            ):
                self._wall_escape_phase = "turn"
                self._wall_escape_last_yaw = pose.yaw_rad
                self._wall_escape_turn_accumulated = 0.0
            elif self._reverse_corridor_is_safe(pose, grid):
                self._wall_escape_updates_left -= 1
                return MotionCommand(
                    -abs(self.config.wall_escape_reverse_speed_mm_s),
                    0,
                    "wall_escape_backoff",
                    "backing through recently observed free space",
                )
            else:
                self._clear_wall_escape()
                return None

        if self._wall_escape_phase == "turn":
            delta_yaw = abs(normalize_angle(
                pose.yaw_rad - self._wall_escape_last_yaw
            ))
            self._wall_escape_last_yaw = pose.yaw_rad
            self._wall_escape_turn_accumulated += delta_yaw
            if (
                self._wall_escape_turn_accumulated
                >= self.config.wall_escape_turn_rad
            ):
                self._wall_escape_phase = "forward"
                self._wall_escape_updates_left = (
                    self.config.wall_escape_forward_updates
                )
                self._reset_turn_tracking()
            else:
                return self._turn_command(
                    self._wall_escape_direction
                    * self.config.minimum_turn_speed_mrad_s,
                    "wall_escape_turn",
                    "turning toward the clearer side after backing off",
                    pose,
                    scan,
                    grid,
                )

        if self._wall_escape_phase == "forward":
            front = sector_min_distance(scan, 0.0, 32.0)
            required = (
                self.config.stop_distance_m
                + self.config.progress_escape_front_margin_m
            )
            if front >= required and self._wall_escape_updates_left > 0:
                self._wall_escape_updates_left -= 1
                return MotionCommand(
                    self.config.cruise_speed_mm_s,
                    0,
                    "wall_escape_forward",
                    "leaving the wall before selecting a new frontier",
                )
            self._clear_wall_escape()
            self._path = []
            return None
        return None

    def _clear_wall_escape(self):
        self._wall_escape_phase = None
        self._wall_escape_anchor_xy = None
        self._wall_escape_last_yaw = None
        self._wall_escape_turn_accumulated = 0.0
        self._wall_escape_updates_left = 0

    def _reverse_corridor_is_safe(self, pose, grid):
        """Allow reverse only into map cells previously observed as free."""
        yaw_cos = math.cos(pose.yaw_rad)
        yaw_sin = math.sin(pose.yaw_rad)
        center_x = (
            pose.x_m
            - yaw_cos * self.config.lidar_offset_x_m
            + yaw_sin * self.config.lidar_offset_y_m
        )
        center_y = (
            pose.y_m
            - yaw_sin * self.config.lidar_offset_x_m
            - yaw_cos * self.config.lidar_offset_y_m
        )
        half_length = self.config.robot_length_m / 2.0
        half_width = (
            self.config.robot_width_m / 2.0
            + self.config.wall_escape_corridor_margin_m
        )
        rear_start = (
            -half_length
            - self.config.wall_escape_backoff_m
            - self.config.wall_escape_corridor_margin_m
        )
        rear_end = -half_length
        radius = math.hypot(rear_start, half_width)
        center_cell = grid.world_to_grid(np.asarray(((
            center_x, center_y
        ),)))[0]
        cell_radius = int(math.ceil(radius / grid.resolution_m))
        probabilities = grid.probabilities()
        occupied = grid.significant_obstacles(
            self.config.occupied_probability
        )
        checked = 0
        known_free = 0
        center_col, center_row = map(int, center_cell)
        for row in range(center_row - cell_radius, center_row + cell_radius + 1):
            for col in range(
                center_col - cell_radius,
                center_col + cell_radius + 1,
            ):
                if not grid.in_bounds(col, row):
                    continue
                world_x = (
                    grid.origin_x_m + (col + 0.5) * grid.resolution_m
                )
                world_y = (
                    grid.origin_y_m + (row + 0.5) * grid.resolution_m
                )
                dx = world_x - center_x
                dy = world_y - center_y
                local_x = yaw_cos * dx + yaw_sin * dy
                local_y = -yaw_sin * dx + yaw_cos * dy
                if not (
                    rear_start <= local_x <= rear_end
                    and abs(local_y) <= half_width
                ):
                    continue
                checked += 1
                if occupied[row, col]:
                    return False
                if (
                    grid.observed[row, col]
                    and probabilities[row, col]
                    <= self.config.free_probability
                ):
                    known_free += 1
        map_corridor_is_known_free = (
            checked > 0
            and known_free / checked
            >= self.config.wall_escape_min_known_ratio
        )
        return (
            map_corridor_is_known_free
            or self._recent_path_allows_reverse(pose)
        )

    def _recent_path_allows_reverse(self, pose):
        """Permit a short retrace when the car demonstrably arrived from behind."""
        if len(self._pose_history) < 2:
            return False
        yaw_cos = math.cos(pose.yaw_rad)
        yaw_sin = math.sin(pose.yaw_rad)
        required_distance = self.config.wall_escape_backoff_m * 0.80
        lateral_limit = (
            self.config.robot_width_m / 2.0
            + self.config.wall_escape_corridor_margin_m
            + 0.08
        )
        maximum_distance = max(
            self.config.wall_escape_backoff_m * 2.5,
            0.45,
        )
        for old_x, old_y in reversed(tuple(self._pose_history)[:-1]):
            dx = old_x - pose.x_m
            dy = old_y - pose.y_m
            local_x = yaw_cos * dx + yaw_sin * dy
            local_y = -yaw_sin * dx + yaw_cos * dy
            distance = math.hypot(dx, dy)
            if distance > maximum_distance:
                break
            if (
                local_x <= -required_distance
                and abs(local_y) <= lateral_limit
            ):
                return True
        return False

    def _progress_escape_command(self, pose, scan, grid):
        front = sector_min_distance(scan, 0.0, 32.0)
        required = (
            self.config.stop_distance_m
            + self.config.progress_escape_front_margin_m
        )
        if self._progress_escape_updates_left > 0:
            if front >= required:
                self._progress_escape_updates_left -= 1
                self._path = []
                self._reset_turn_tracking()
                return MotionCommand(
                    self.config.cruise_speed_mm_s,
                    0,
                    "progress_escape",
                    "leaving an area with no exploration progress",
                )
            self._progress_escape_updates_left = 0
            self._reset_progress_tracking()
            left = sector_min_distance(scan, 48.0, 48.0)
            right = sector_min_distance(scan, -48.0, 48.0)
            angular = (
                self.config.minimum_turn_speed_mrad_s
                if left >= right
                else -self.config.minimum_turn_speed_mrad_s
            )
            return self._turn_command(
                angular,
                "progress_escape_turn",
                "front blocked while escaping a local exploration loop",
                pose,
                scan,
                grid,
            )

        if (
            len(self._progress_history) < self._progress_history.maxlen
            or len(self._coverage_history) < self._coverage_history.maxlen
        ):
            return None
        anchor = self._progress_history[0]
        span = max(
            math.hypot(point[0] - anchor[0], point[1] - anchor[1])
            for point in self._progress_history
        )
        new_cells = max(
            0,
            self._coverage_history[-1] - self._coverage_history[0],
        )
        minimum_new_cells = max(
            1,
            int(math.ceil(
                self.config.progress_min_new_area_m2
                / (grid.resolution_m * grid.resolution_m)
            )),
        )
        if new_cells >= minimum_new_cells:
            return None

        self._blacklist_active_frontier()
        self._path = []
        self._active_frontier_target = None
        self._reset_progress_tracking()
        self._expire_frontier_blacklist()
        alternate_path = find_frontier_path(
            grid,
            pose,
            self.config,
            excluded_targets=self._frontier_blacklist,
        )
        if alternate_path:
            self._path = alternate_path
            self._active_frontier_target = alternate_path[-1]
            self._reset_turn_tracking()
            return None
        reason = (
            "map coverage stalled despite {:.2f}m pose span; "
            "current frontier temporarily blacklisted"
        ).format(span)
        if front >= required:
            self._progress_escape_updates_left = max(
                0, self.config.progress_escape_updates - 1
            )
            self._reset_turn_tracking()
            return MotionCommand(
                self.config.cruise_speed_mm_s,
                0,
                "progress_escape",
                reason,
            )
        left = sector_min_distance(scan, 48.0, 48.0)
        right = sector_min_distance(scan, -48.0, 48.0)
        angular = (
            self.config.minimum_turn_speed_mrad_s
            if left >= right
            else -self.config.minimum_turn_speed_mrad_s
        )
        return self._turn_command(
            angular,
            "progress_escape_turn",
            reason,
            pose,
            scan,
            grid,
        )

    def _reset_progress_tracking(self):
        self._progress_history.clear()
        self._coverage_history.clear()

    def _frontier_target_changed(self, target):
        previous = self._active_frontier_target
        if previous is None or target is None:
            return previous != target
        return math.hypot(
            target[0] - previous[0],
            target[1] - previous[1],
        ) > self.config.frontier_blacklist_radius_m

    def _blacklist_active_frontier(self):
        if self._active_frontier_target is None:
            return
        self._expire_frontier_blacklist()
        expires_at = self._updates + max(
            1, self.config.frontier_blacklist_updates
        )
        self._frontier_blacklist.append((
            self._active_frontier_target[0],
            self._active_frontier_target[1],
            expires_at,
        ))

    def _expire_frontier_blacklist(self):
        self._frontier_blacklist = [
            item for item in self._frontier_blacklist
            if item[2] > self._updates
        ]

    def _recovery_command(self, scan, pose, grid):
        """没有可达前沿时，沿当前开阔方向短程探测，避免一直原地打转。"""
        front = sector_min_distance(scan, 0.0, 32.0)
        left = sector_min_distance(scan, 48.0, 48.0)
        right = sector_min_distance(scan, -48.0, 48.0)
        open_distance = (
            self.config.stop_distance_m
            + self.config.recovery_open_margin_m
        )
        if front >= open_distance:
            self._reset_turn_tracking()
            return MotionCommand(
                self.config.slow_speed_mm_s,
                0,
                "recovery_probe",
                "probing open space to reveal hidden frontiers",
            )
        angular = (
            self.config.minimum_turn_speed_mrad_s
            if left >= right
            else -self.config.minimum_turn_speed_mrad_s
        )
        return self._turn_command(
            angular,
            "recovery_turn",
            "turning toward the clearer side",
            pose,
            scan,
            grid,
        )

    def _turn_command(
        self,
        angular,
        state,
        reason,
        pose,
        scan,
        grid,
        target=None,
    ):
        turn_limit_reached = (
            False
            if self._rotation_blocked_updates
            else self._observe_turn_progress(pose)
        )
        if turn_limit_reached:
            front = sector_min_distance(scan, 0.0, 32.0)
            required = (
                self.config.stop_distance_m
                + self.config.turn_escape_front_margin_m
            )
            self._path = []
            if front >= required:
                self._turn_escape_updates_left = max(
                    0, self.config.turn_escape_updates - 1
                )
                self._reset_turn_tracking()
                return MotionCommand(
                    min(
                        self.config.slow_speed_mm_s,
                        self.config.turn_arc_speed_mm_s,
                    ),
                    0,
                    "turn_escape",
                    "stationary turn made no useful progress",
                    target,
                )
            if self._begin_wall_escape(pose, grid, angular):
                return self._wall_escape_command(pose, scan, grid)
            return MotionCommand(
                0,
                0,
                "turn_limit_blocked",
                "stationary turn limit reached and front is blocked",
                target,
            )

        front = sector_min_distance(scan, 0.0, 32.0)
        arc_scan_blocked, arc_map_blocked = (
            self._rotation_block_sources(
                pose,
                scan,
                grid,
                extra_radius_m=self.config.arc_sweep_extra_margin_m,
            )
        )
        arc_is_safe = (
            front
            >= self.config.stop_distance_m
            + self.config.arc_front_margin_m
            and not arc_scan_blocked
            and not arc_map_blocked
        )
        if arc_is_safe:
            self._rotation_blocked_updates = 0
            return MotionCommand(
                min(
                    self.config.slow_speed_mm_s,
                    self.config.turn_arc_speed_mm_s,
                ),
                angular,
                state,
                reason + "; using a short forward arc",
                target,
            )
        scan_blocked, map_blocked = self._rotation_block_sources(
            pose,
            scan,
            grid,
        )
        if not scan_blocked and not map_blocked:
            self._rotation_blocked_updates = 0
            return MotionCommand(
                0, angular, state, reason, target
            )

        self._rotation_blocked_updates += 1
        direction = 1 if angular >= 0 else -1
        cautious_angular = (
            direction * self.config.minimum_turn_speed_mrad_s
        )
        if (
            self._rotation_blocked_updates
            < self.config.rotation_block_confirmations
        ):
            return MotionCommand(
                0,
                cautious_angular,
                "cautious_turn_probe",
                "confirming a possible rotation obstacle",
                target,
            )

        self._path = []
        escape_required = (
            self.config.stop_distance_m
            + min(self.config.turn_escape_front_margin_m, 0.10)
        )
        if front >= escape_required:
            self._turn_escape_updates_left = max(
                0, self.config.turn_escape_updates - 1
            )
            self._reset_turn_tracking()
            return MotionCommand(
                min(
                    self.config.slow_speed_mm_s,
                    self.config.turn_arc_speed_mm_s,
                ),
                0,
                "turn_escape",
                "moving forward to clear the rotation envelope",
                target,
            )
        if self._begin_wall_escape(pose, grid, cautious_angular):
            return self._wall_escape_command(pose, scan, grid)
        if (
            not scan_blocked
            and map_blocked
            and self.config.map_only_probe_every_updates > 0
            and self._rotation_blocked_updates
            % self.config.map_only_probe_every_updates
            == 0
        ):
            return MotionCommand(
                0,
                cautious_angular,
                "cautious_turn_probe",
                "current scan is clear; testing a stale map obstacle",
                target,
            )
        source = (
            "current lidar scan"
            if scan_blocked
            else "occupied map cells"
        )
        return MotionCommand(
            0,
            0,
            "rotation_blocked",
            "vehicle footprint is blocked by " + source,
            target,
        )

    def _observe_turn_progress(self, pose):
        current = (pose.x_m, pose.y_m, pose.yaw_rad)
        if self._turn_anchor_xy is None:
            self._turn_anchor_xy = current[:2]
            self._last_turn_pose = current
            return False

        distance_from_anchor = math.hypot(
            current[0] - self._turn_anchor_xy[0],
            current[1] - self._turn_anchor_xy[1],
        )
        if distance_from_anchor >= self.config.stationary_turn_radius_m:
            self._turn_anchor_xy = current[:2]
            self._last_turn_pose = current
            self._stationary_turn_accumulated = 0.0
            self._turn_stall_count = 0
            return False

        delta_translation = math.hypot(
            current[0] - self._last_turn_pose[0],
            current[1] - self._last_turn_pose[1],
        )
        delta_yaw = abs(normalize_angle(
            current[2] - self._last_turn_pose[2]
        ))
        self._last_turn_pose = current
        self._stationary_turn_accumulated += delta_yaw
        if (
            delta_translation < 0.005
            and delta_yaw < math.radians(self.config.turn_stall_yaw_deg)
        ):
            self._turn_stall_count += 1
        else:
            self._turn_stall_count = 0
        return (
            self._stationary_turn_accumulated
            >= self.config.maximum_stationary_turn_rad
            or self._turn_stall_count >= self.config.turn_stall_updates
        )

    def _reset_turn_tracking(self):
        self._turn_anchor_xy = None
        self._last_turn_pose = None
        self._stationary_turn_accumulated = 0.0
        self._turn_stall_count = 0
        self._rotation_blocked_updates = 0

    def _rotation_block_sources(
        self,
        pose,
        scan,
        grid,
        extra_radius_m=0.0,
    ):
        radius = (
            math.hypot(
                self.config.robot_length_m / 2.0,
                self.config.robot_width_m / 2.0,
            )
            + self.config.safety_margin_m
            + max(0.0, extra_radius_m)
        )

        distances = np.asarray(scan.distances_m, dtype=np.float64)
        angles = np.asarray(scan.angles_rad, dtype=np.float64)
        valid = np.isfinite(distances) & (distances > 0)
        scan_blocked = False
        if np.any(valid):
            local_x = (
                self.config.lidar_offset_x_m
                + distances[valid] * np.cos(angles[valid])
            )
            local_y = (
                self.config.lidar_offset_y_m
                + distances[valid] * np.sin(angles[valid])
            )
            close = np.hypot(local_x, local_y) <= radius
            scan_blocked = _has_angular_cluster(
                angles[valid],
                close,
                self.config.rotation_obstacle_min_points,
                math.radians(
                    self.config.rotation_obstacle_max_angle_gap_deg
                ),
            )

        occupied = grid.significant_obstacles(
            self.config.occupied_probability
        )
        center = grid.world_to_grid(np.asarray(((
            pose.x_m, pose.y_m
        ),)))[0]
        center_col, center_row = int(center[0]), int(center[1])
        cells = int(math.ceil(radius / grid.resolution_m))
        local_occupied = np.zeros(
            (2 * cells + 1, 2 * cells + 1),
            dtype=bool,
        )
        for row in range(center_row - cells, center_row + cells + 1):
            for col in range(center_col - cells, center_col + cells + 1):
                if not grid.in_bounds(col, row) or not occupied[row, col]:
                    continue
                world_x = (
                    grid.origin_x_m + (col + 0.5) * grid.resolution_m
                )
                world_y = (
                    grid.origin_y_m + (row + 0.5) * grid.resolution_m
                )
                if math.hypot(
                    world_x - pose.x_m,
                    world_y - pose.y_m,
                ) <= radius:
                    local_occupied[
                        row - center_row + cells,
                        col - center_col + cells,
                    ] = True
        map_blocked = _has_connected_cluster(
            local_occupied,
            self.config.rotation_obstacle_min_points,
        )
        return scan_blocked, map_blocked

    def _select_waypoint(self, pose):
        while self._path:
            first = self._path[0]
            distance = math.hypot(
                first[0] - pose.x_m, first[1] - pose.y_m
            )
            if distance > self.config.waypoint_tolerance_m:
                break
            self._path.pop(0)
        return self._path[0] if self._path else None


def _has_angular_cluster(angles, close_mask, minimum_points, max_gap):
    if minimum_points <= 1:
        return bool(np.any(close_mask))
    run = 0
    previous_angle = None
    for angle, is_close in zip(angles, close_mask):
        angle = float(angle)
        if (
            is_close
            and previous_angle is not None
            and abs(normalize_angle(angle - previous_angle)) <= max_gap
        ):
            run += 1
        elif is_close:
            run = 1
        else:
            run = 0
        previous_angle = angle
        if run >= minimum_points:
            return True
    return False


def _has_connected_cluster(mask, minimum_points):
    if minimum_points <= 1:
        return bool(np.any(mask))
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        start = remaining.pop()
        size = 1
        queue = deque((start,))
        while queue:
            row, col = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (row + dy, col + dx)
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    size += 1
                    if size >= minimum_points:
                        return True
    return False


def find_frontier_path(
    grid,
    pose,
    config=None,
    excluded_targets=None,
):
    """在已知空闲区内寻找成簇未知边界，并优先扩展探索范围。"""
    cfg = config or ExplorationConfig()
    probabilities = grid.probabilities()
    occupied = grid.significant_obstacles(
        cfg.occupied_probability
    )
    free = grid.observed & (probabilities <= cfg.free_probability)
    footprint_radius = (
        math.hypot(cfg.robot_length_m / 2.0, cfg.robot_width_m / 2.0)
        + cfg.safety_margin_m
    )
    clearance_cells = int(math.ceil(
        max(cfg.robot_clearance_m, footprint_radius)
        / grid.resolution_m
    ))
    traversable = free & ~_inflate(occupied, clearance_cells)

    start = grid.world_to_grid(
        np.asarray(((pose.x_m, pose.y_m),))
    )[0]
    start_col, start_row = int(start[0]), int(start[1])
    if not grid.in_bounds(start_col, start_row):
        return []
    traversable[start_row, start_col] = True

    frontier = traversable & _adjacent_to(~grid.observed)
    distances, parents = _reachable_bfs(
        traversable, start_col, start_row
    )
    clusters = _clusters(frontier & (distances >= 0))
    minimum_steps = int(
        math.ceil(cfg.min_frontier_distance_m / grid.resolution_m)
    )
    candidates = []
    for cluster in clusters:
        if len(cluster) < cfg.min_frontier_cluster_cells:
            continue
        usable = []
        for row, col in cluster:
            if distances[row, col] < minimum_steps:
                continue
            cell_xy = (
                grid.origin_x_m + (col + 0.5) * grid.resolution_m,
                grid.origin_y_m + (row + 0.5) * grid.resolution_m,
            )
            if _target_is_excluded(
                cell_xy,
                excluded_targets,
                cfg.frontier_blacklist_radius_m,
            ):
                continue
            usable.append((row, col))
        if not usable:
            continue
        if cfg.prefer_distant_frontiers:
            cell = max(
                usable,
                key=lambda item: (
                    math.hypot(
                        item[1] - start_col,
                        item[0] - start_row,
                    ),
                    distances[item[0], item[1]],
                ),
            )
        else:
            cell = min(
                usable, key=lambda item: distances[item[0], item[1]]
            )
        target_xy = (
            grid.origin_x_m + (cell[1] + 0.5) * grid.resolution_m,
            grid.origin_y_m + (cell[0] + 0.5) * grid.resolution_m,
        )
        distance_steps = int(distances[cell[0], cell[1]])
        straight_distance_m = math.hypot(
            target_xy[0] - pose.x_m,
            target_xy[1] - pose.y_m,
        )
        information_gain = _frontier_information_gain(
            cluster,
            ~grid.observed,
            int(math.ceil(
                cfg.frontier_information_radius_m
                / grid.resolution_m
            )),
        )
        information_extent_m = (
            math.sqrt(information_gain) * grid.resolution_m
        )
        information_score = (
            cfg.frontier_information_weight * information_extent_m
        )
        heading_error = abs(normalize_angle(
            math.atan2(
                target_xy[1] - pose.y_m,
                target_xy[0] - pose.x_m,
            )
            - pose.yaw_rad
        ))
        candidates.append((
            straight_distance_m,
            distance_steps,
            heading_error,
            -information_score,
            cell,
        ))
    if not candidates:
        return []

    if cfg.prefer_distant_frontiers:
        farthest_distance = max(item[0] for item in candidates)
        nearly_farthest = [
            item for item in candidates
            if item[0]
            >= farthest_distance - cfg.frontier_distance_slack_m
        ]
        selected = min(
            nearly_farthest,
            key=lambda item: (
                item[2],
                item[3],
                -item[0],
                item[1],
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda item: (item[1], item[2], item[3]),
        )
    (
        _straight_distance,
        _distance,
        _heading_error,
        _negative_information,
        (target_row, target_col),
    ) = selected
    cells = []
    row, col = target_row, target_col
    while (col, row) != (start_col, start_row):
        cells.append((col, row))
        parent_row, parent_col = parents[row, col]
        if parent_row < 0:
            return []
        row, col = int(parent_row), int(parent_col)
    cells.reverse()
    cells = _simplify_grid_path(
        [(start_col, start_row)] + cells,
        traversable,
        max(
            1,
            int(math.floor(
                cfg.path_segment_max_m / grid.resolution_m
            )),
        ),
    )[1:]
    return [
        (
            grid.origin_x_m + (col + 0.5) * grid.resolution_m,
            grid.origin_y_m + (row + 0.5) * grid.resolution_m,
        )
        for col, row in cells
    ]


def _target_is_excluded(target_xy, excluded_targets, radius_m):
    if not excluded_targets:
        return False
    for item in excluded_targets:
        x_m, y_m = item[:2]
        if math.hypot(target_xy[0] - x_m, target_xy[1] - y_m) <= radius_m:
            return True
    return False


def _frontier_information_gain(cluster, unknown, radius):
    """Count unknown cells that a frontier cluster can reveal."""
    if radius <= 0:
        return len(cluster)
    height, width = unknown.shape
    nearby = set()
    for row, col in cluster:
        row_start = max(0, row - radius)
        row_end = min(height, row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(width, col + radius + 1)
        locations = np.argwhere(
            unknown[row_start:row_end, col_start:col_end]
        )
        for local_row, local_col in locations:
            nearby.add((
                row_start + int(local_row),
                col_start + int(local_col),
            ))
    return len(nearby)


def _simplify_grid_path(cells, traversable, maximum_segment_cells):
    """Remove BFS stair steps while keeping every shortcut collision-free."""
    if len(cells) <= 2:
        return cells
    simplified = [cells[0]]
    index = 0
    while index < len(cells) - 1:
        next_index = index + 1
        furthest = min(
            len(cells) - 1,
            index + max(1, maximum_segment_cells),
        )
        for candidate in range(furthest, index, -1):
            if _grid_line_is_clear(
                cells[index],
                cells[candidate],
                traversable,
            ):
                next_index = candidate
                break
        simplified.append(cells[next_index])
        index = next_index
    return simplified


def _grid_line_is_clear(start, end, traversable):
    for col, row in _grid_line_cells(start, end):
        if (
            row < 0
            or row >= traversable.shape[0]
            or col < 0
            or col >= traversable.shape[1]
            or not traversable[row, col]
        ):
            return False
    return True


def _grid_line_cells(start, end):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy


def _inflate(mask, radius):
    if radius <= 0 or not np.any(mask):
        return mask.copy()
    inflated = np.zeros_like(mask)
    height, width = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            source_y0 = max(0, -dy)
            source_y1 = min(height, height - dy)
            source_x0 = max(0, -dx)
            source_x1 = min(width, width - dx)
            target_y0 = source_y0 + dy
            target_y1 = source_y1 + dy
            target_x0 = source_x0 + dx
            target_x1 = source_x1 + dx
            inflated[target_y0:target_y1, target_x0:target_x1] |= (
                mask[source_y0:source_y1, source_x0:source_x1]
            )
    return inflated


def _adjacent_to(mask):
    adjacent = np.zeros_like(mask)
    adjacent[1:, :] |= mask[:-1, :]
    adjacent[:-1, :] |= mask[1:, :]
    adjacent[:, 1:] |= mask[:, :-1]
    adjacent[:, :-1] |= mask[:, 1:]
    return adjacent


def _reachable_bfs(traversable, start_col, start_row):
    height, width = traversable.shape
    distances = np.full((height, width), -1, dtype=np.int32)
    parents = np.full((height, width, 2), -1, dtype=np.int32)
    distances[start_row, start_col] = 0
    queue = deque(((start_row, start_col),))
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        row, col = queue.popleft()
        for dy, dx in neighbors:
            next_row, next_col = row + dy, col + dx
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and traversable[next_row, next_col]
                and distances[next_row, next_col] < 0
            ):
                distances[next_row, next_col] = (
                    distances[row, col] + 1
                )
                parents[next_row, next_col] = (row, col)
                queue.append((next_row, next_col))
    return distances, parents


def _clusters(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    clusters = []
    while remaining:
        start = remaining.pop()
        cluster = [start]
        queue = deque((start,))
        while queue:
            row, col = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (row + dy, col + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        cluster.append(neighbor)
                        queue.append(neighbor)
        clusters.append(cluster)
    return clusters
