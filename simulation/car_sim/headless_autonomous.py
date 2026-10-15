"""Fast, deterministic closed-loop simulation for frontier exploration.

This deliberately has no browser or wall-clock dependency: one Python loop
advances the differential-drive chassis, produces lidar rays, updates an
occupancy grid and asks the production ``FrontierExplorer`` for its next
command.  It is intended for regression/parameter work, not photorealism.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path

import numpy as np

from raspberry_pi.lidar import LaserScan
from raspberry_pi.mapping import OccupancyGrid, Pose2D, SlamUpdate
from raspberry_pi.mapping.adaptive_speed import (
    AdaptiveSpeedConfig,
    AdaptiveSpeedController,
)
from raspberry_pi.mapping.explorer import (
    ExplorationConfig,
    FrontierExplorer,
    MotionCommand,
    find_frontier_path,
    find_observation_path,
    has_unresolved_obstacle_shadows,
)
from raspberry_pi.mapping.map_visualization import save_trajectory_png
from raspberry_pi.mapping.scan_filter import filter_scan_sector
from car_sim.adaptive_motion_stress import (
    DeadZoneChassis,
    DeadZoneChassisConfig,
)


@dataclass(frozen=True)
class Rectangle:
    x: float
    y: float
    width: float
    height: float


_SCENE_NAME = "full"


def configure_scene(name):
    """Select the obstacle layout used by helper metrics and simulations."""
    global _SCENE_NAME
    if name not in ("full", "single-rectangle"):
        raise ValueError("unknown headless scene: {}".format(name))
    _SCENE_NAME = name


def _truth_at_timestamp(truth, timestamp_s):
    """Return the nearest truth sample without assuming every SLAM frame passed."""
    if not truth:
        return None
    return min(
        truth,
        key=lambda item: abs(float(item["timestamp_s"]) - timestamp_s),
    )


def _obstacle_view_sectors(truth, sector_count=12):
    """Count distinct viewing sectors around every configured obstacle."""
    obstacles = (
        (Rectangle(3.0, 2.0, 0.0, 0.0),)
        if _SCENE_NAME == "single-rectangle"
        else (
            Rectangle(2.05, 1.25, 0.95, 0.70),
            Rectangle(3.65, 2.75, 0.60, 0.55),
        )
    )
    result = []
    for obstacle in obstacles:
        center_x = obstacle.x + obstacle.width * 0.5
        center_y = obstacle.y + obstacle.height * 0.5
        sectors = set()
        for pose in truth:
            angle = math.atan2(
                float(pose["y_m"]) - center_y,
                float(pose["x_m"]) - center_x,
            )
            sector = int(((angle + math.pi) % math.tau) / math.tau * sector_count)
            sectors.add(sector)
        result.append(len(sectors))
    return result


def _confined_motion_metrics(truth, progress_indices=(), radius_m=0.35):
    """Measure the longest local dwell and yaw accumulated without progress."""
    if len(truth) < 2:
        return 0.0, 0.0
    progress = set(int(value) for value in progress_indices)
    anchor_index = 0
    longest_s = 0.0
    longest_yaw = 0.0
    accumulated_yaw = 0.0
    for index in range(1, len(truth)):
        previous = truth[index - 1]
        current = truth[index]
        delta_yaw = abs((
            float(current["yaw_rad"]) - float(previous["yaw_rad"]) + math.pi
        ) % math.tau - math.pi)
        accumulated_yaw += delta_yaw
        distance = math.hypot(
            float(current["x_m"]) - float(truth[anchor_index]["x_m"]),
            float(current["y_m"]) - float(truth[anchor_index]["y_m"]),
        )
        if distance > radius_m or index in progress:
            anchor_index = index
            accumulated_yaw = 0.0
            continue
        duration = (
            float(current["timestamp_s"])
            - float(truth[anchor_index]["timestamp_s"])
        )
        if duration >= longest_s:
            longest_s = duration
            longest_yaw = math.degrees(accumulated_yaw)
    return longest_s, longest_yaw


def _ray_rectangle(x, y, dx, dy, rectangle):
    near = -math.inf
    far = math.inf
    for origin, direction, lower, upper in (
        (x, dx, rectangle.x, rectangle.x + rectangle.width),
        (y, dy, rectangle.y, rectangle.y + rectangle.height),
    ):
        if abs(direction) < 1e-12:
            if origin < lower or origin > upper:
                return math.inf
            continue
        first = (lower - origin) / direction
        second = (upper - origin) / direction
        near = max(near, min(first, second))
        far = min(far, max(first, second))
        if near > far:
            return math.inf
    if far < 0.0:
        return math.inf
    return near if near >= 0.0 else far


def _scan(pose, timestamp_s, room, obstacles, ray_count=240, noise_m=0.006,
          dropout=0.015, rng=None):
    rng = rng or np.random.default_rng(7)
    angles = np.linspace(-math.pi, math.pi, ray_count, endpoint=False)
    distances = []
    walls = (
        Rectangle(0.0, -0.02, room[0], 0.02),
        Rectangle(0.0, room[1], room[0], 0.02),
        Rectangle(-0.02, 0.0, 0.02, room[1]),
        Rectangle(room[0], 0.0, 0.02, room[1]),
    )
    for relative in angles:
        world = pose.yaw_rad + relative
        dx, dy = math.cos(world), math.sin(world)
        distance = min(
            _ray_rectangle(pose.x_m, pose.y_m, dx, dy, item)
            for item in (*walls, *obstacles)
        )
        distance = min(6.0, max(0.12, distance + rng.normal(0.0, noise_m)))
        distances.append(distance)
    distances = np.asarray(distances, dtype=np.float64)
    keep = rng.random(len(distances)) >= dropout
    return LaserScan(
        angles[keep],
        distances[keep],
        timestamp_s,
        ground_truth_pose={
            "x_m": pose.x_m,
            "y_m": pose.y_m,
            "yaw_rad": pose.yaw_rad,
        },
    )


def _point_blocked(x, y, room, obstacles, radius=0.25):
    if x < radius or y < radius or x > room[0] - radius or y > room[1] - radius:
        return True
    for obstacle in obstacles:
        closest_x = min(max(x, obstacle.x), obstacle.x + obstacle.width)
        closest_y = min(max(y, obstacle.y), obstacle.y + obstacle.height)
        if math.hypot(x - closest_x, y - closest_y) < radius:
            return True
    return False


def _integrate_truth_scan(grid, pose, scan, refresh_filter=True):
    points = scan.points(0.12, 6.0)
    grid.update(
        (pose.x_m, pose.y_m),
        pose.transform_points(points),
        refresh_filter=refresh_filter,
    )


def _truth_map_metrics(grid, room, obstacles, trajectory):
    """Return scene-ground-truth metrics; PNG appearance is not a pass test."""
    yy, xx = np.indices((grid.height, grid.width))
    world_x = grid.origin_x_m + (xx + 0.5) * grid.resolution_m
    world_y = grid.origin_y_m + (yy + 0.5) * grid.resolution_m
    inside = (
        (world_x >= 0.0) & (world_x <= room[0])
        & (world_y >= 0.0) & (world_y <= room[1])
    )
    for obstacle in obstacles:
        inside &= ~(
            (world_x >= obstacle.x)
            & (world_x <= obstacle.x + obstacle.width)
            & (world_y >= obstacle.y)
            & (world_y <= obstacle.y + obstacle.height)
        )
    free_coverage = float(np.count_nonzero(grid.observed & inside)) / max(
        1, int(np.count_nonzero(inside))
    )
    occupied = grid.significant_obstacles()

    def edge_recall(rectangle, tolerance_m=0.12):
        spacing = grid.resolution_m * 0.5
        samples = []
        for start, stop, fixed, horizontal in (
            (rectangle.x, rectangle.x + rectangle.width, rectangle.y, True),
            (rectangle.x, rectangle.x + rectangle.width,
             rectangle.y + rectangle.height, True),
            (rectangle.y, rectangle.y + rectangle.height, rectangle.x, False),
            (rectangle.y, rectangle.y + rectangle.height,
             rectangle.x + rectangle.width, False),
        ):
            count = max(2, int(math.ceil((stop - start) / spacing)) + 1)
            for value in np.linspace(start, stop, count):
                samples.append((value, fixed) if horizontal else (fixed, value))
        radius = max(1, int(math.ceil(tolerance_m / grid.resolution_m)))
        hits = 0
        for column, row in grid.world_to_grid(np.asarray(samples)):
            c0, c1 = max(0, column - radius), min(grid.width, column + radius + 1)
            r0, r1 = max(0, row - radius), min(grid.height, row + radius + 1)
            hits += bool(np.any(occupied[r0:r1, c0:c1]))
        return float(hits) / max(1, len(samples))

    wall_box = Rectangle(0.0, 0.0, room[0], room[1])
    obstacle_recalls = [edge_recall(item) for item in obstacles]
    views = []
    for obstacle in obstacles:
        sectors = set()
        cx = obstacle.x + obstacle.width * 0.5
        cy = obstacle.y + obstacle.height * 0.5
        for item in trajectory:
            angle = math.atan2(item.y_m - cy, item.x_m - cx)
            sectors.add(int(((angle + math.pi) % math.tau) / math.tau * 12))
        views.append(len(sectors))
    return {
        "truth_free_coverage": free_coverage,
        "truth_outer_wall_recall": edge_recall(wall_box),
        "truth_obstacle_outline_recall": obstacle_recalls,
        "truth_obstacle_view_sectors": views,
    }


def run_fast_simulation(steps=4500, dt_s=0.10, seed=7, output=None):
    room = (5.2, 4.2)
    obstacles = (
        Rectangle(2.05, 1.25, 0.95, 0.70),
        Rectangle(3.65, 2.75, 0.60, 0.55),
    )
    pose = Pose2D(1.0, 1.0, 0.15)
    chassis = DeadZoneChassis(DeadZoneChassisConfig(
        linear_start_mm_s=420,
        linear_hold_mm_s=220,
        angular_start_mrad_s=2300,
        angular_hold_mrad_s=1500,
    ))
    chassis.pose = pose
    config = ExplorationConfig(
        cruise_speed_mm_s=300,
        slow_speed_mm_s=220,
        turn_speed_mrad_s=1500,
        turn_arc_speed_mm_s=230,
        narrow_passage_speed_mm_s=240,
        narrow_passage_turn_speed_mrad_s=1700,
        initial_spin_rad=math.radians(45.0),
    )
    explorer = FrontierExplorer(config)
    speed_controller = AdaptiveSpeedController(
        maximum_turn_speed_mrad_s=2600,
        maximum_linear_speed_mm_s=420,
        config=AdaptiveSpeedConfig(
            target_linear_speed_min_m_s=0.075,
            target_linear_speed_max_m_s=0.22,
            target_angular_speed_min_rad_s=0.12,
            target_angular_speed_max_rad_s=0.70,
            maximum_narrow_passage_linear_speed_mm_s=420,
            maximum_narrow_passage_turn_mrad_s=1700,
            maximum_narrow_passage_stall_turn_mrad_s=2600,
        ),
    )
    grid = OccupancyGrid(
        width_cells=140,
        height_cells=120,
        resolution_m=0.05,
        origin_x_m=-0.9,
        origin_y_m=-0.9,
        occupied_inflation_cells=1,
        min_obstacle_area_m2=0.01,
    )
    rng = np.random.default_rng(seed)
    trajectory = [pose]
    states = {}
    total_yaw = 0.0
    travelled = 0.0
    max_confined_s = 0.0
    confined_anchor = pose
    confined_start = 0.0
    completed = False
    command = None
    commanded_forward_m = 0.0
    collision_cancelled_m = 0.0
    collision_cancelled_steps = 0
    reason_counts = {}
    pure_turn_steps = 0
    arc_turn_steps = 0
    straight_steps = 0
    target_switches = 0
    turn_direction_reversals = 0
    previous_target = None
    previous_turn_direction = 0
    map_anchor = None
    stationary_steps_left = 7  # 2 settle scans + 5 fusion scans
    stationary_fusions = 0
    map_poses = []

    for index in range(int(steps)):
        timestamp_s = index * dt_s
        scan = _scan(pose, timestamp_s, room, obstacles, rng=rng)
        mapping_scan = filter_scan_sector(scan, 0.0, 160.0)
        if stationary_steps_left > 0:
            speed_controller.stopped()
            command = MotionCommand(
                0,
                0,
                "stationary_mapping",
                "stopped lidar settling/fusion",
            )
            stationary_steps_left -= 1
            if stationary_steps_left == 0:
                # Production integrates one robustly fused scan after two
                # settling and five collection scans.  Noise is already small
                # and deterministic here, so the final stopped scan stands in
                # for that fused frame.
                _integrate_truth_scan(grid, pose, mapping_scan)
                stationary_fusions += 1
                map_anchor = pose
                map_poses.append(pose)
                explorer.note_stationary_map_integrated(pose, grid)
        else:
            distance_since_map = math.hypot(
                pose.x_m - map_anchor.x_m,
                pose.y_m - map_anchor.y_m,
            )
            angle_since_map = abs((
                pose.yaw_rad - map_anchor.yaw_rad + math.pi
            ) % math.tau - math.pi)
            if (
                distance_since_map >= 0.35
                or angle_since_map >= math.radians(20.0)
            ):
                stationary_steps_left = 6
                speed_controller.stopped()
                command = MotionCommand(
                    0,
                    0,
                    "stationary_mapping",
                    "stopped lidar settling/fusion",
                )
            else:
                command = speed_controller.apply(
                    explorer.update(pose, mapping_scan, grid)
                )
                if command.state == "obstacle_inspection_observe":
                    # Begin the same stopped fusion gate used by the real-car
                    # loop. The completed view is recorded only after the
                    # simulated fusion has actually finished.
                    stationary_steps_left = 7
        states[command.state] = states.get(command.state, 0) + 1
        reason_counts[command.reason] = reason_counts.get(command.reason, 0) + 1
        if command.angular_mrad_s != 0:
            if command.linear_mm_s == 0:
                pure_turn_steps += 1
            else:
                arc_turn_steps += 1
            turn_direction = 1 if command.angular_mrad_s > 0 else -1
            if (
                previous_turn_direction != 0
                and turn_direction != previous_turn_direction
            ):
                turn_direction_reversals += 1
            previous_turn_direction = turn_direction
        elif command.linear_mm_s != 0:
            straight_steps += 1
        target = command.target_xy_m
        if target is not None:
            target = (float(target[0]), float(target[1]))
            if (
                previous_target is not None
                and math.hypot(
                    target[0] - previous_target[0],
                    target[1] - previous_target[1],
                ) > 0.20
            ):
                target_switches += 1
            previous_target = target
        if command.finished:
            completed = True
            break

        previous_pose = pose
        update = chassis.step(command, dt_s)
        candidate_pose = chassis.pose
        requested_step_distance = (
            abs(command.linear_mm_s) * 0.00032 * dt_s
        )
        if command.linear_mm_s != 0:
            commanded_forward_m += requested_step_distance
        if _point_blocked(
            candidate_pose.x_m,
            candidate_pose.y_m,
            room,
            obstacles,
        ):
            collision_cancelled_m += requested_step_distance
            collision_cancelled_steps += 1
            chassis.pose = previous_pose
            chassis.linear_speed_m_s = 0.0
            chassis.angular_speed_rad_s = 0.0
            chassis._linear_running = False
            chassis._angular_running = False
            candidate_pose = previous_pose
        step_distance = math.hypot(
            candidate_pose.x_m - previous_pose.x_m,
            candidate_pose.y_m - previous_pose.y_m,
        )
        delta_yaw = (candidate_pose.yaw_rad - previous_pose.yaw_rad + math.pi) % math.tau - math.pi
        travelled += step_distance
        total_yaw += abs(delta_yaw)
        pose = candidate_pose
        trajectory.append(pose)
        # Feed the motion actually produced by the simulated chassis back to
        # the production adaptive controller.  Without this observation the
        # previous "closed-loop" simulation never exercised stall detection,
        # start-torque learning, or release of a transient turn kick.
        speed_controller.observe(SlamUpdate(
            pose,
            True,
            0.02,
            100,
            len(scan.distances_m),
            inlier_ratio=0.92,
            translation_m=step_distance,
            rotation_rad=delta_yaw,
            linear_speed_m_s=(
                abs(chassis.linear_speed_m_s)
                if step_distance > 0.0 else 0.0
            ),
            angular_speed_rad_s=(
                abs(chassis.angular_speed_rad_s)
                if abs(delta_yaw) > 0.0 else 0.0
            ),
        ))

        planned_stationary = command.state in (
            "stationary_mapping",
            "completion_scan",
            "obstacle_inspection_observe",
        )
        if planned_stationary:
            # Stable fusion and the required final full sweep are deliberate,
            # bounded stops, not local exploration failure.
            confined_anchor = pose
            confined_start = timestamp_s
        elif math.hypot(
            pose.x_m - confined_anchor.x_m,
            pose.y_m - confined_anchor.y_m,
        ) > 0.35:
            confined_anchor = pose
            confined_start = timestamp_s
        if not planned_stationary:
            max_confined_s = max(
                max_confined_s,
                timestamp_s - confined_start,
            )

    observed = int(np.count_nonzero(grid.observed))
    truth_metrics = _truth_map_metrics(
        grid,
        room,
        obstacles,
        map_poses,
    )
    acceptance_passed = (
        completed
        and truth_metrics["truth_free_coverage"] >= 0.95
        and truth_metrics["truth_outer_wall_recall"] >= 0.90
        and all(
            recall >= 0.80
            for recall in truth_metrics["truth_obstacle_outline_recall"]
        )
        and all(
            sectors >= 7
            for sectors in truth_metrics["truth_obstacle_view_sectors"]
        )
        and collision_cancelled_steps == 0
        and max_confined_s <= 30.0
    )
    result = {
        "steps": len(trajectory) - 1,
        "simulated_s": (len(trajectory) - 1) * dt_s,
        "completed": completed,
        "acceptance_passed": acceptance_passed,
        "travelled_m": travelled,
        "commanded_forward_m": commanded_forward_m,
        "collision_cancelled_m": collision_cancelled_m,
        "collision_cancelled_steps": collision_cancelled_steps,
        "absolute_yaw_turns": total_yaw / math.tau,
        "max_confined_s": max_confined_s,
        "observed_cells": observed,
        "stationary_fusions": stationary_fusions,
        **truth_metrics,
        "state_counts": states,
        "reason_counts": reason_counts,
        "pure_turn_steps": pure_turn_steps,
        "arc_turn_steps": arc_turn_steps,
        "straight_steps": straight_steps,
        "target_switches": target_switches,
        "turn_direction_reversals": turn_direction_reversals,
        "remaining_path_points": len(explorer._path),
        "active_frontier_target": explorer._active_frontier_target,
        "frontier_blacklist_count": len(explorer._frontier_blacklist),
        "completion_no_frontier_count": explorer._no_frontier_count,
        "completion_rescan_turns": explorer._rescan_accumulated / math.tau,
        "inspection_pending": explorer._inspection_task_pending,
        "inspection_completed_components": len(
            explorer._completed_inspection_components
        ),
        "unresolved_obstacle_shadows": has_unresolved_obstacle_shadows(
            grid, config, pose=pose
        ),
        "remaining_frontier_probe": find_frontier_path(
            grid, pose, config
        )[-1:] or [],
        "remaining_observation_probe": find_observation_path(
            grid, pose, config
        )[-1:] or [],
        "final_pose": [pose.x_m, pose.y_m, pose.yaw_rad],
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        png_path = path.with_suffix(".png")
        save_trajectory_png(
            grid,
            tuple(
                (index * dt_s, item.x_m, item.y_m, item.yaw_rad)
                for index, item in enumerate(trajectory)
            ),
            png_path,
        )
        result["map_png"] = str(png_path)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="加速自主探索闭环仿真")
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--dt", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="car_sim/output/headless_autonomous.json")
    return parser


def main():
    args = build_parser().parse_args()
    result = run_fast_simulation(args.steps, args.dt, args.seed, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
