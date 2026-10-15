"""Full closed-loop regression for every known-map inspection target.

Unlike ``inspection_regression``, this simulation does not stop at helper
decisions.  It loads the real occupancy map, dynamically plans to every one of
the production targets, approaches through the real A* path, and runs the
production precision-heading accumulator until a photo would be taken.

The stochastic model covers the field failures seen on 2026-08-20/21:

* approximate rather than exact final positions;
* left/right motor response variation and first-pulse static friction;
* scan-to-scan yaw noise;
* glass-induced ICP dropouts, opposite-direction jumps and excessive jumps;
* absolute map-yaw jumps during a final turn (deliberately ignored by the
  relative-heading controller).

Success is strict: every generated target must have a path, every target must
produce one capture, no target may be skipped, and physical camera-heading
error must remain within the configured timeout capture tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap, order_targets
from raspberry_pi.map_camera_inspection import (
    _precision_heading_progress,
    build_argument_parser,
)
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.geometry import normalize_angle
from raspberry_pi.mapping.scan_matcher import MatchResult


def _motion(delta_yaw_rad, rng, mode):
    """Create one scan matcher result, including glass failure modes."""
    if mode == "dropout":
        return MatchResult(
            Pose2D(0.0, 0.0, 0.0),
            math.inf,
            0,
            False,
            0.0,
        )
    if mode == "opposite_jump":
        measured = -math.copysign(
            math.radians(rng.uniform(35.0, 65.0)),
            delta_yaw_rad if delta_yaw_rad else 1.0,
        )
    elif mode == "excessive_jump":
        measured = math.copysign(
            math.radians(rng.uniform(55.0, 85.0)),
            delta_yaw_rad if delta_yaw_rad else 1.0,
        )
    else:
        measured = delta_yaw_rad + math.radians(rng.normal(0.0, 0.8))
    return MatchResult(
        Pose2D(
            float(rng.normal(0.0, 0.004)),
            float(rng.normal(0.0, 0.004)),
            normalize_angle(measured),
        ),
        0.035,
        150,
        True,
        0.82,
    )


def _approach_pose(path, target, incoming_yaw, rng, nearby_distance_m):
    """Stop before exact docking while remaining on the real final segment."""
    view = np.asarray(target.view_xy_m, dtype=np.float64)
    if len(path) >= 2:
        previous = np.asarray(path[-2], dtype=np.float64)
        vector = previous - view
        length = float(np.linalg.norm(vector))
        if length > 1e-6:
            vector /= length
            stand_early_m = min(
                float(nearby_distance_m) * 0.82,
                max(0.08, length * rng.uniform(0.25, 0.80)),
            )
            position = view + vector * stand_early_m
        else:
            position = view
    else:
        position = view
    position += rng.normal(0.0, 0.018, size=2)
    return Pose2D(
        float(position[0]),
        float(position[1]),
        normalize_angle(incoming_yaw),
    )


def _turn_and_capture(actual_pose, target, args, rng, glass_severity):
    """Run up to two production-style final-heading attempts."""
    camera_offset = math.radians(args.camera_yaw_offset_deg)
    desired_actual_yaw = normalize_angle(
        math.atan2(
            target.target_xy_m[1] - actual_pose.y_m,
            target.target_xy_m[0] - actual_pose.x_m,
        ) - camera_offset
    )
    pulse_ttl_s = args.navigation_precision_turn_ttl_ms / 1000.0
    total_pulses = 0
    fallback_frames = 0
    scan_frames = 0
    response_estimate = 0.75
    for attempt in range(2):
        # Position/yaw at the start of a final turn comes from the bounded map
        # tracker.  Rare glass frames have more yaw noise, but not a new room.
        yaw_sigma_deg = 4.0 + 3.0 * glass_severity
        seed_errors = []
        for _ in range(args.navigation_heading_seed_frames):
            estimated_yaw = normalize_angle(
                actual_pose.yaw_rad
                + math.radians(rng.normal(0.0, yaw_sigma_deg))
            )
            estimated_x = actual_pose.x_m + float(rng.normal(0.0, 0.025))
            estimated_y = actual_pose.y_m + float(rng.normal(0.0, 0.025))
            desired_estimated_yaw = normalize_angle(
                math.atan2(
                    target.target_xy_m[1] - estimated_y,
                    target.target_xy_m[0] - estimated_x,
                ) - camera_offset
            )
            seed_errors.append(normalize_angle(
                desired_estimated_yaw - estimated_yaw
            ))
        remaining = math.atan2(
            sum(math.sin(value) for value in seed_errors),
            sum(math.cos(value) for value in seed_errors),
        )
        elapsed_s = 0.0
        first_pulse = True
        while total_pulses < 48:
            physical_error = abs(normalize_angle(
                desired_actual_yaw - actual_pose.yaw_rad
            ))
            if abs(remaining) <= math.radians(args.heading_tolerance_deg):
                return {
                    "captured": physical_error <= math.radians(
                        args.navigation_timeout_capture_max_heading_error_deg
                    ),
                    "pose": actual_pose,
                    "physical_heading_error_deg": math.degrees(physical_error),
                    "estimated_heading_error_deg": math.degrees(abs(remaining)),
                    "attempts": attempt + 1,
                    "pulses": total_pulses,
                    "scan_frames": scan_frames,
                    "fallback_frames": fallback_frames,
                }
            if (
                elapsed_s >= args.navigation_pure_turn_timeout_seconds
                and abs(remaining) <= math.radians(
                    args.navigation_timeout_capture_max_heading_error_deg
                )
            ):
                return {
                    "captured": physical_error <= math.radians(
                        args.navigation_timeout_capture_max_heading_error_deg
                    ),
                    "pose": actual_pose,
                    "physical_heading_error_deg": math.degrees(physical_error),
                    "estimated_heading_error_deg": math.degrees(abs(remaining)),
                    "attempts": attempt + 1,
                    "pulses": total_pulses,
                    "scan_frames": scan_frames,
                    "fallback_frames": fallback_frames,
                }
            angular = int(math.copysign(
                max(
                    550,
                    min(
                        args.navigation_fine_turn_speed,
                        abs(remaining) * 900.0,
                    ),
                ),
                remaining,
            ))
            # Measured chassis variation.  A first low-output pulse may only
            # load the tyres/motors without visible body yaw.
            response_gain = float(rng.uniform(0.72, 1.16))
            stuck = bool(
                first_pulse
                and abs(angular) <= 700
                and rng.random() < 0.45
            )
            actual_delta = (
                0.0
                if stuck
                else angular / 1000.0 * pulse_ttl_s * response_gain
            )
            actual_pose = Pose2D(
                actual_pose.x_m,
                actual_pose.y_m,
                normalize_angle(actual_pose.yaw_rad + actual_delta),
            )
            first_pulse = False
            total_pulses += 1
            elapsed_s += 0.80
            failure_roll = rng.random()
            if failure_roll < 0.035 * glass_severity:
                mode = "opposite_jump"
            elif failure_roll < 0.070 * glass_severity:
                mode = "excessive_jump"
            elif failure_roll < 0.13 * glass_severity:
                mode = "dropout"
            else:
                mode = "normal"
            motion = _motion(actual_delta, rng, mode)
            progress, source = _precision_heading_progress(
                angular,
                motion,
                args.navigation_precision_turn_ttl_ms,
                0.80,
                response_estimate,
            )
            if source == "scan_motion" and abs(progress) >= math.radians(1.0):
                nominal = abs(angular) / 1000.0 * pulse_ttl_s
                observed = max(0.35, min(1.25, abs(progress) / nominal))
                response_estimate = 0.70 * response_estimate + 0.30 * observed
            remaining = normalize_angle(remaining - progress)
            if source == "scan_motion":
                scan_frames += 1
            else:
                fallback_frames += 1
            if elapsed_s >= args.navigation_pure_turn_timeout_seconds:
                break
        # Production puts a badly aligned target at the back of the queue and
        # tries once more.  The simulation retries immediately but refreshes
        # the bounded map heading, which is the relevant control effect.
    physical_error = abs(normalize_angle(
        desired_actual_yaw - actual_pose.yaw_rad
    ))
    return {
        "captured": False,
        "pose": actual_pose,
        "physical_heading_error_deg": math.degrees(physical_error),
        "estimated_heading_error_deg": math.degrees(abs(remaining)),
        "attempts": 2,
        "pulses": total_pulses,
        "scan_frames": scan_frames,
        "fallback_frames": fallback_frames,
    }


def simulate_seed(route, args, seed):
    rng = np.random.default_rng(int(seed))
    current_pose = Pose2D(0.13, 0.73, math.radians(-114.0))
    captures = []
    failures = []
    for target, path in route:
        if len(path) >= 2:
            incoming_yaw = math.atan2(
                path[-1][1] - path[-2][1],
                path[-1][0] - path[-2][0],
            )
        else:
            incoming_yaw = current_pose.yaw_rad
        current_pose = _approach_pose(
            path,
            target,
            incoming_yaw,
            rng,
            args.nearby_capture_distance,
        )
        distance_to_view = math.hypot(
            target.view_xy_m[0] - current_pose.x_m,
            target.view_xy_m[1] - current_pose.y_m,
        )
        # Glass is strongest for the left/top part of this specific scene.
        glass_severity = 1.0 + (
            0.8
            if current_pose.x_m < -0.35 and current_pose.y_m > 0.7
            else 0.0
        )
        outcome = _turn_and_capture(
            current_pose, target, args, rng, glass_severity
        )
        current_pose = outcome.pop("pose")
        outcome.update({
            "target_id": target.target_id,
            "distance_to_view_m": distance_to_view,
            "path_points": len(path),
        })
        captures.append(outcome)
        if not outcome["captured"]:
            failures.append(target.target_id)
    maximum_error = max(
        (item["physical_heading_error_deg"] for item in captures),
        default=math.inf,
    )
    return {
        "seed": int(seed),
        "target_count": len(route),
        "capture_count": sum(item["captured"] for item in captures),
        "failures": failures,
        "maximum_heading_error_deg": maximum_error,
        "captures": captures,
        "passed": (
            len(captures) == len(route)
            and not failures
            and all(
                item["distance_to_view_m"]
                <= args.nearby_capture_distance + 1e-6
                for item in captures
            )
            and maximum_error
            <= args.navigation_timeout_capture_max_heading_error_deg
        ),
    }


def _legacy_endpoint_only_run(
    seed_count=100,
    output="car_sim/output/full_inspection_mission.json",
):
    args = build_argument_parser().parse_args([
        "--map", "maps/ble_scene_17.pgm",
        "--map-yaml", "maps/ble_scene_17.yaml",
        "--clearance", "0.30",
        "--standoff", "0.65",
        "--max-targets", "24",
        "--views-per-wall", "3",
        "--views-per-region", "1",
        "--nearby-capture-distance", "0.75",
        "--heading-tolerance-deg", "7",
        "--navigation-timeout-capture-max-heading-error-deg", "35",
        "--navigation-fine-turn-speed", "1100",
        "--navigation-precision-turn-ttl-ms", "180",
        "--navigation-heading-seed-frames", "5",
        "--navigation-pure-turn-timeout-seconds", "8",
    ])
    known_map = KnownOccupancyMap.load(
        args.map,
        args.map_yaml,
        clearance_m=args.clearance,
    )
    targets = known_map.inspection_targets(
        standoff_m=args.standoff,
        maximum_targets=args.max_targets,
        views_per_wall=args.views_per_wall,
        views_per_region=args.views_per_region,
    )
    route = order_targets(known_map, (0.13, 0.73), targets)
    seeds = [
        simulate_seed(route, args, seed)
        for seed in range(int(seed_count))
    ]
    failed = [item for item in seeds if not item["passed"]]
    target_statistics = {}
    for target in targets:
        samples = [
            capture
            for seed in seeds
            for capture in seed["captures"]
            if capture["target_id"] == target.target_id
        ]
        heading_errors = [
            item["physical_heading_error_deg"] for item in samples
        ]
        target_statistics[target.target_id] = {
            "capture_count": sum(item["captured"] for item in samples),
            "expected_capture_count": len(seeds),
            "maximum_heading_error_deg": max(heading_errors, default=math.inf),
            "p95_heading_error_deg": (
                float(np.percentile(heading_errors, 95.0))
                if heading_errors else math.inf
            ),
            "maximum_distance_to_view_m": max(
                (item["distance_to_view_m"] for item in samples),
                default=math.inf,
            ),
            "maximum_turn_pulses": max(
                (item["pulses"] for item in samples), default=0
            ),
            "maximum_fallback_frames": max(
                (item["fallback_frames"] for item in samples), default=0
            ),
        }
    every_target_completed = all(
        stats["capture_count"] == stats["expected_capture_count"]
        for stats in target_statistics.values()
    )
    result = {
        "map": args.map,
        "target_count": len(targets),
        "route_target_count": len(route),
        "target_ids": [item.target_id for item in targets],
        "seed_count": len(seeds),
        "passed_seed_count": len(seeds) - len(failed),
        "failed_seed_count": len(failed),
        "maximum_heading_error_deg": max(
            item["maximum_heading_error_deg"] for item in seeds
        ),
        "total_captures": sum(item["capture_count"] for item in seeds),
        "expected_total_captures": len(targets) * len(seeds),
        "every_target_completed": every_target_completed,
        "target_statistics": target_statistics,
        "passed": (
            not failed
            and len(route) == len(targets)
            and every_target_completed
        ),
        "failed_seeds": failed[:10],
        "worst_seed": max(
            seeds,
            key=lambda item: item["maximum_heading_error_deg"],
        ),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def run(seed_count=100, output="car_sim/output/full_inspection_mission.json"):
    """Compatibility entry point for the real-PGM closed-loop simulator."""
    from car_sim.known_map_closed_loop import run as closed_loop_run

    return closed_loop_run(seed_count=seed_count, output=output)


def main():
    parser = argparse.ArgumentParser(
        description="simulate every known-map photo target"
    )
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument(
        "--output",
        default="car_sim/output/full_inspection_mission.json",
    )
    cli = parser.parse_args()
    result = run(cli.seed_count, cli.output)
    summary = dict(result)
    summary.pop("worst_seed", None)
    summary.pop("failed_seeds", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
