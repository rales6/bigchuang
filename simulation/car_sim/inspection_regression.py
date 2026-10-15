"""Deterministic regression simulation for known-map photo inspection.

This is deliberately headless and hardware-free.  It replays the geometry
observed in the 2026-08-20 field logs against the production safety and photo
decision helpers using the real ``ble_scene_17`` occupancy map.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.map_camera_inspection import (
    _collision_reverse_is_safe,
    _inspection_standoff_reached,
    _nearby_turn_timeout_capture_allowed,
    _navigation_turn_timeout_action,
    _obstacle_safe,
    build_argument_parser,
)
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.geometry import normalize_angle


def _scan(front_m=2.0, rear_m=2.0, timestamp_s=0.0):
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    distances = np.full(360, 2.0, dtype=np.float64)
    front = np.abs((angles + math.pi) % math.tau - math.pi) <= math.radians(30.0)
    rear = np.abs((angles - math.pi + math.pi) % math.tau - math.pi) <= math.radians(30.0)
    distances[front] = float(front_m)
    distances[rear] = float(rear_m)
    return LaserScan(angles, distances, float(timestamp_s))


def run(output="car_sim/output/inspection_regression.json", seed=20):
    args = build_argument_parser().parse_args([
        "--map", "maps/ble_scene_17.pgm",
        "--map-yaml", "maps/ble_scene_17.yaml",
        "--nearby-capture-distance", "0.75",
        "--navigation-pure-turn-timeout-seconds", "8",
        "--navigation-timeout-capture-max-heading-error-deg", "35",
        "--collision-reverse-distance", "0.06",
        "--collision-reverse-speed", "140",
        "--collision-reverse-rear-clearance", "0.28",
        "--collision-reverse-max-steps", "4",
    ])
    known_map = KnownOccupancyMap.load(
        args.map, args.map_yaml, clearance_m=args.clearance
    )
    scenarios = []

    # Exact wall_3_2 state from the 21:30 field run.  The permanent map still
    # has a path, but the volatile obstacle layer made the planned view
    # unreachable.  At 0.618m from that view, the revised nearby rule should
    # stop translation and retain a useful photo instead of skipping it.
    corner_pose = Pose2D(-0.3793, 1.0825, math.radians(-170.71))
    corner_view = (-0.675, 1.625)
    permanent_path = known_map.path(
        (corner_pose.x_m, corner_pose.y_m), corner_view
    )
    standoff = _inspection_standoff_reached(
        corner_pose,
        corner_view,
        front_clearance_m=0.465,
        waypoint_tolerance_m=args.waypoint_tolerance,
        stop_distance_m=args.stop_distance,
        nearby_capture_distance_m=args.nearby_capture_distance,
    )
    timeout_capture, corner_distance = (
        _nearby_turn_timeout_capture_allowed(
            corner_pose,
            corner_view,
            waypoint_index=3,
            waypoint_count=4,
            final_translation_complete=True,
            nearby_capture_distance_m=args.nearby_capture_distance,
        )
    )
    scenarios.append({
        "name": "wall_3_2_nearby_photo",
        "passed": bool(
            permanent_path
            and standoff
            and timeout_capture
            and corner_distance <= args.nearby_capture_distance
        ),
        "distance_to_view_m": corner_distance,
        "front_clearance_m": 0.465,
        "permanent_path_points": len(permanent_path),
        "decision": (
            "accept_approximate_position_then_align_camera"
            if timeout_capture else "skip"
        ),
    })

    # Reproduce the later final-view stall.  A frozen chassis receives at
    # most timeout/dt pure-turn decisions.  It may capture only when the
    # remaining camera-heading error is small; a sideways/reversed view is
    # deferred while an intermediate waypoint remains unaffected.
    final_pose = Pose2D(-0.40, 1.06, math.radians(-150.8))
    final_view = (-0.42, 1.03)
    dt_s = 0.25
    turn_updates = int(math.ceil(
        args.navigation_pure_turn_timeout_seconds / dt_s
    ))
    final_allowed, final_distance = _nearby_turn_timeout_capture_allowed(
        final_pose,
        final_view,
        waypoint_index=0,
        waypoint_count=1,
        final_translation_complete=False,
        nearby_capture_distance_m=args.nearby_capture_distance,
    )
    transit_allowed, _ = _nearby_turn_timeout_capture_allowed(
        final_pose,
        final_view,
        waypoint_index=0,
        waypoint_count=4,
        final_translation_complete=False,
        nearby_capture_distance_m=args.nearby_capture_distance,
    )
    final_action = _navigation_turn_timeout_action(
        args.navigation_pure_turn_timeout_seconds,
        args.navigation_pure_turn_timeout_seconds,
        final_allowed,
        heading_error_rad=normalize_angle(
            math.atan2(
                final_view[1] - final_pose.y_m,
                final_view[0] - final_pose.x_m,
            ) - final_pose.yaw_rad
        ),
        maximum_capture_error_deg=(
            args.navigation_timeout_capture_max_heading_error_deg
        ),
    )
    transit_action = _navigation_turn_timeout_action(
        args.navigation_pure_turn_timeout_seconds,
        args.navigation_pure_turn_timeout_seconds,
        transit_allowed,
    )
    scenarios.append({
        "name": "frozen_final_turn_is_bounded",
        "passed": bool(
            final_allowed
            and not transit_allowed
            and final_action == "capture"
            and transit_action == "continue"
            and turn_updates == 32
        ),
        "simulated_turn_updates": turn_updates,
        "stop_time_s": args.navigation_pure_turn_timeout_seconds,
        "distance_to_view_m": final_distance,
        "final_action": final_action,
        "transit_action": transit_action,
    })

    # Exact first-photo geometry from 22:32: pose yaw 14.57deg while the
    # mapped wall required about 83.2deg.  The old timeout captured this 69deg
    # error; the revised decision must put the point at the back of the queue.
    latest_pose = Pose2D(-0.1312, 0.7348, math.radians(14.57))
    latest_target = (-0.025, 1.625)
    latest_error = normalize_angle(
        math.atan2(
            latest_target[1] - latest_pose.y_m,
            latest_target[0] - latest_pose.x_m,
        ) - latest_pose.yaw_rad
    )
    latest_action = _navigation_turn_timeout_action(
        8.4,
        args.navigation_pure_turn_timeout_seconds,
        True,
        heading_error_rad=latest_error,
        maximum_capture_error_deg=(
            args.navigation_timeout_capture_max_heading_error_deg
        ),
    )
    scenarios.append({
        "name": "wrong_first_photo_heading_is_not_captured",
        "passed": latest_action == "replan",
        "heading_error_deg": math.degrees(latest_error),
        "decision": latest_action,
    })

    # Reproduce the glass-wall range discontinuity.  The production full-scan
    # guard must stop on the 0.907m frame, before the following 0.083m frame.
    glass_ranges = (0.932, 0.907, 0.763, 0.083)
    safe_sequence = [
        _obstacle_safe(
            _scan(front_m=value, timestamp_s=index * 0.8),
            360,
            args.stop_distance,
            0,
            full_scan_guard=True,
        )
        for index, value in enumerate(glass_ranges)
    ]
    first_stop = next(
        (index for index, safe in enumerate(safe_sequence) if not safe),
        None,
    )
    scenarios.append({
        "name": "glass_wall_stops_one_scan_early",
        "passed": first_stop == 1,
        "ranges_m": list(glass_ranges),
        "safe_sequence": safe_sequence,
        "first_stop_index": first_stop,
    })

    # The reverse escape requires both a clear rear scan and a clear corridor
    # in the real map.  A broad 0.20m rear obstacle must veto the maneuver,
    # while one isolated 0.08m reflection is treated as N10 speckle.
    collision_pose = Pose2D(-0.8207, 0.2675, math.radians(-170.18))
    reverse_safe = _collision_reverse_is_safe(
        _scan(front_m=0.083, rear_m=1.0),
        collision_pose,
        known_map,
        args,
    )
    reverse_blocked = _collision_reverse_is_safe(
        _scan(front_m=0.083, rear_m=0.20),
        collision_pose,
        known_map,
        args,
    )
    speckle_scan = _scan(front_m=0.083, rear_m=1.0)
    speckle_distances = np.array(speckle_scan.distances_m, copy=True)
    rear_relative = (
        speckle_scan.angles_rad - math.pi + math.pi
    ) % math.tau - math.pi
    rear_indices = np.flatnonzero(
        np.abs(rear_relative) <= math.radians(25.0)
    )
    speckle_distances[rear_indices[len(rear_indices) // 2]] = 0.08
    reverse_speckle = _collision_reverse_is_safe(
        LaserScan(
            speckle_scan.angles_rad,
            speckle_distances,
            speckle_scan.timestamp_s,
        ),
        collision_pose,
        known_map,
        args,
    )
    scenarios.append({
        "name": "reverse_requires_dual_clearance_evidence",
        "passed": bool(
            reverse_safe[0]
            and reverse_speckle[0]
            and not reverse_blocked[0]
        ),
        "clear_rear_decision": reverse_safe[0],
        "isolated_8cm_speckle_decision": reverse_speckle[0],
        "blocked_rear_decision": reverse_blocked[0],
        "blocked_reason": reverse_blocked[-1],
    })

    # Add bounded pose noise around the skipped-corner record.  This checks
    # that ordinary localization jitter does not push the nearby decision back
    # into the old 0.45m skip boundary.
    rng = np.random.default_rng(int(seed))
    captures = 0
    trials = 500
    distances = []
    for _ in range(trials):
        noisy = Pose2D(
            corner_pose.x_m + rng.normal(0.0, 0.025),
            corner_pose.y_m + rng.normal(0.0, 0.025),
            corner_pose.yaw_rad + math.radians(rng.normal(0.0, 6.0)),
        )
        allowed, distance = _nearby_turn_timeout_capture_allowed(
            noisy,
            corner_view,
            waypoint_index=3,
            waypoint_count=4,
            final_translation_complete=True,
            nearby_capture_distance_m=args.nearby_capture_distance,
        )
        captures += int(allowed)
        distances.append(distance)
    capture_ratio = captures / trials
    scenarios.append({
        "name": "nearby_photo_pose_noise_monte_carlo",
        "passed": capture_ratio >= 0.98,
        "trials": trials,
        "capture_ratio": capture_ratio,
        "distance_p95_m": float(np.percentile(distances, 95)),
    })

    result = {
        "map": args.map,
        "seed": int(seed),
        "passed": all(item["passed"] for item in scenarios),
        "scenarios": scenarios,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="known-map photo-inspection regression simulation"
    )
    parser.add_argument(
        "--output",
        default="car_sim/output/inspection_regression.json",
    )
    parser.add_argument("--seed", type=int, default=20)
    return parser


def main():
    cli = build_parser().parse_args()
    result = run(output=cli.output, seed=cli.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
