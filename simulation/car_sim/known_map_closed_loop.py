"""Closed-loop inspection simulation on the real saved occupancy map.

This model never teleports the chassis to a photo point.  It follows every
production A* waypoint from the latest field start pose, integrates a
differential-drive body, checks the inflated PGM traversability mask, carries
bounded localization drift, and aligns the camera using measured turn motion.
Verified semantic photo poses are loaded from the map's Qwen annotation file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap, order_targets
from raspberry_pi.map_camera_inspection import (
    _command_for_pose,
    _blocked_path_heading_command,
    _information_view_targets,
    _inspection_path_for_target,
    _prefer_displaced_retry_choices,
    _precision_heading_progress,
    _skip_classified_interior_targets,
    _spaced_targets,
    _within_capture_reference,
    build_argument_parser,
)
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.geometry import normalize_angle
from raspberry_pi.mapping.scan_matcher import MatchResult


FIELD_START_POSE = Pose2D(-0.935, 0.625, math.radians(-126.0))


class _SemanticSnapshot:
    """Read-only semantic-map facade used by the production target filters."""

    def __init__(self, data):
        self.data = data

    def target_online_confirmed(self, target_id):
        entity = self.data.get("entities", {}).get(target_id, {})
        return bool((entity.get("online_qwen") or {}).get("promoted", False))


def _load_semantic(path):
    return _SemanticSnapshot(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def _motion_result(delta, rng, glass_severity):
    failure = rng.random()
    if failure < 0.025 * glass_severity:
        return MatchResult(Pose2D(0.0, 0.0, 0.0), math.inf, 0, False, 0.0)
    if failure < 0.040 * glass_severity:
        measured = -math.copysign(math.radians(45.0), delta.yaw_rad or 1.0)
    elif failure < 0.055 * glass_severity:
        measured = math.copysign(math.radians(75.0), delta.yaw_rad or 1.0)
    else:
        measured = delta.yaw_rad + math.radians(rng.normal(0.0, 0.55))
    return MatchResult(
        Pose2D(
            delta.x_m + float(rng.normal(0.0, 0.003)),
            delta.y_m + float(rng.normal(0.0, 0.003)),
            normalize_angle(measured),
        ),
        0.035,
        150,
        True,
        0.82,
    )


def _traversable(known_map, pose):
    column, row = known_map.world_to_grid(((pose.x_m, pose.y_m),))[0]
    return bool(
        known_map.in_bounds(int(column), int(row))
        and known_map.traversable[int(row), int(column)]
    )


def _integrate_chassis(
    known_map,
    pose,
    linear_mm_s,
    angular_mrad_s,
    duration_s,
    rng,
    motor_gain,
    turn_floor_mrad_s,
    straight_bias_rad_m,
):
    linear_m_s = float(linear_mm_s) / 1000.0 * motor_gain
    travelled_m = linear_m_s * duration_s
    angular_output_gain = 1.8  # car/config.py on the deployed ESP32
    if int(linear_mm_s) == 0 and int(angular_mrad_s) != 0:
        magnitude_mrad_s = abs(float(angular_mrad_s))
        if magnitude_mrad_s < float(turn_floor_mrad_s):
            commanded_rotation_rad = 0.0
        else:
            # The ESP32 ramps angular velocity at 15000 mrad/s^2.  A short
            # pulse may spend most of its life crossing the four-wheel scrub
            # friction threshold; a one-second refreshed command, by contrast,
            # reaches full speed and reproduces the 60--124 degree field jumps.
            acceleration_rad_s2 = 15.0
            target_rad_s = magnitude_mrad_s / 1000.0
            floor_rad_s = float(turn_floor_mrad_s) / 1000.0
            threshold_s = floor_rad_s / acceleration_rad_s2
            target_s = target_rad_s / acceleration_rad_s2
            ramp_end_s = min(float(duration_s), target_s)
            ramp_area = 0.0
            if ramp_end_s > threshold_s:
                ramp_area = 0.5 * acceleration_rad_s2 * (
                    ramp_end_s ** 2 - threshold_s ** 2
                )
            cruise_area = target_rad_s * max(0.0, float(duration_s) - target_s)
            commanded_rotation_rad = ramp_area + cruise_area
        yaw_delta = (
            math.copysign(
                commanded_rotation_rad * motor_gain * angular_output_gain,
                angular_mrad_s,
            )
            + travelled_m * straight_bias_rad_m
        )
    else:
        angular_rad_s = (
            float(angular_mrad_s) / 1000.0
            * motor_gain * angular_output_gain
        )
        yaw_delta = (
            angular_rad_s * duration_s
            + travelled_m * straight_bias_rad_m
        )
    mid_yaw = pose.yaw_rad + yaw_delta * 0.5
    candidate = Pose2D(
        pose.x_m + travelled_m * math.cos(mid_yaw),
        pose.y_m + travelled_m * math.sin(mid_yaw),
        normalize_angle(pose.yaw_rad + yaw_delta),
    )
    if linear_mm_s and not _traversable(known_map, candidate):
        candidate = Pose2D(pose.x_m, pose.y_m, candidate.yaw_rad)
        return candidate, pose.inverse().compose(candidate), True
    return candidate, pose.inverse().compose(candidate), False


def _estimated_pose(truth, rng, glass_severity):
    # This is the bounded pose available after production jump rejection, not
    # an unbounded raw map candidate.  Glass increases noise but cannot move
    # the simulated car into another room.
    return Pose2D(
        truth.x_m + float(rng.normal(0.0, 0.018 * glass_severity)),
        truth.y_m + float(rng.normal(0.0, 0.018 * glass_severity)),
        normalize_angle(
            truth.yaw_rad
            + math.radians(rng.normal(0.0, 2.2 * glass_severity))
        ),
    )


def _path_length(path):
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(path, path[1:])
    )


def _align_camera(
    known_map,
    truth,
    target,
    desired_yaw,
    args,
    rng,
    motor_gain,
    turn_floor,
    straight_bias,
):
    remaining = normalize_angle(desired_yaw - truth.yaw_rad)
    scan_frames = 0
    rejected_frames = 0
    maximum_pulses = 52
    for pulse in range(maximum_pulses):
        if abs(remaining) <= math.radians(args.heading_tolerance_deg):
            break
        angular = int(math.copysign(
            max(
                args.navigation_fine_turn_min_speed,
                min(args.navigation_fine_turn_speed, abs(remaining) * 900.0),
            ),
            remaining,
        ))
        previous = truth
        truth, delta, _collision = _integrate_chassis(
            known_map,
            truth,
            0,
            angular,
            args.navigation_precision_turn_ttl_ms / 1000.0,
            rng,
            motor_gain,
            turn_floor,
            straight_bias,
        )
        severity = 1.8 if truth.x_m < -0.35 and truth.y_m > 0.7 else 1.0
        motion = _motion_result(delta, rng, severity)
        progress, source = _precision_heading_progress(
            angular,
            motion,
            args.navigation_precision_turn_ttl_ms,
            0.55,
        )
        if source == "scan_motion":
            scan_frames += 1
        else:
            rejected_frames += 1
        if source == "unverified_no_progress":
            # Production stops and averages stationary map headings.  Truth
            # is used only by the evaluator; five noisy samples model that
            # bounded re-seed without inventing command motion.
            severity = 1.8 if truth.x_m < -0.35 and truth.y_m > 0.7 else 1.0
            samples = [
                normalize_angle(
                    desired_yaw - truth.yaw_rad
                    + math.radians(rng.normal(0.0, 2.2 * severity))
                )
                for _ in range(args.navigation_heading_seed_frames)
            ]
            remaining = math.atan2(
                sum(math.sin(value) for value in samples),
                sum(math.cos(value) for value in samples),
            )
        else:
            remaining = normalize_angle(remaining - progress)
        # A stuck pulse is deliberately retained as zero truth motion.  The
        # no-fake-progress controller must wait rather than subtract command
        # time from the desired angle.
        if previous == truth and source != "scan_motion":
            continue
    physical_error = abs(normalize_angle(desired_yaw - truth.yaw_rad))
    return truth, {
        "heading_error_deg": math.degrees(physical_error),
        "estimated_remaining_deg": math.degrees(abs(remaining)),
        "turn_pulses": pulse + 1,
        "scan_motion_frames": scan_frames,
        "rejected_motion_frames": rejected_frames,
        # Match the production completion contract.  Per-target p95 and
        # maximum errors remain reported separately, so a rare fallback near
        # this bound cannot be hidden by the pass/fail threshold.
        "heading_ok": physical_error <= math.radians(
            args.navigation_timeout_capture_max_heading_error_deg
        ),
    }


def _drive_target(
    known_map,
    truth,
    target,
    path,
    reference_views,
    verified,
    args,
    rng,
):
    motor_gain = float(rng.uniform(0.78, 1.08))
    # Field run 20260821_130545 repeatedly showed no useful rotation at
    # 1300--1900 and breakaway only near 2200--2600.  The old 500--720 model
    # was the main reason the simulator incorrectly considered long turns
    # stable.
    # Include chassis samples whose pivot breakaway is above the configured
    # 2600 mrad/s ceiling. Production must still finish those routes by
    # switching to its forward-arc fallback after four measured stall frames.
    turn_floor = float(rng.uniform(1850.0, 2650.0))
    straight_bias = math.radians(float(rng.normal(0.0, 2.0)))
    waypoints = path[1:] or [target.view_xy_m]
    waypoint_index = 0
    collisions = 0
    consecutive_collisions = 0
    replans = 0
    turn_stalls = 0
    force_forward_arc = False
    blocked_heading_recovery = False
    steps = 0
    maximum_steps = max(450, len(waypoints) * 35)
    while steps < maximum_steps:
        steps += 1
        severity = 1.7 if truth.x_m < -0.35 and truth.y_m > 0.7 else 1.0
        estimate = _estimated_pose(truth, rng, severity)
        reference_ok, reference_distance, limit_m = _within_capture_reference(
            truth,
            target.target_id,
            reference_views,
            verified,
            args.nearby_capture_distance,
        )
        if reference_ok:
            break
        waypoint = waypoints[min(waypoint_index, len(waypoints) - 1)]
        lookahead_stop = min(len(waypoints), waypoint_index + 7)
        lookahead = [
            (
                math.hypot(
                    waypoints[index][0] - estimate.x_m,
                    waypoints[index][1] - estimate.y_m,
                ),
                index,
            )
            for index in range(waypoint_index, lookahead_stop)
        ]
        nearest_distance, nearest_index = min(lookahead)
        current_distance = math.hypot(
            waypoint[0] - estimate.x_m,
            waypoint[1] - estimate.y_m,
        )
        if (
            nearest_index > waypoint_index
            and nearest_distance + 0.06 < current_distance
        ):
            waypoint_index = nearest_index
            waypoint = waypoints[waypoint_index]
        if math.hypot(waypoint[0] - estimate.x_m, waypoint[1] - estimate.y_m) <= args.waypoint_tolerance:
            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                waypoint_index = len(waypoints) - 1
        waypoint = waypoints[waypoint_index]
        blocked_command = (
            _blocked_path_heading_command(args, estimate, waypoint)
            if blocked_heading_recovery else None
        )
        if blocked_command is not None:
            linear, angular, _reached = blocked_command
        else:
            blocked_heading_recovery = False
            linear, angular, _reached = _command_for_pose(
                args,
                estimate,
                waypoint,
                final_yaw=None,
                force_forward_arc=force_forward_arc,
            )
        if linear == 0 and angular:
            angular = int(math.copysign(
                min(
                    args.relocalization_max_turn_speed,
                    max(abs(angular), args.turn_speed)
                    + turn_stalls * 200,
                ),
                angular,
            ))
        duration_s = 0.35 if linear else 0.22
        truth, delta, collision = _integrate_chassis(
            known_map,
            truth,
            linear,
            angular,
            duration_s,
            rng,
            motor_gain,
            turn_floor,
            straight_bias,
        )
        if linear == 0 and angular:
            if abs(delta.yaw_rad) < math.radians(1.0):
                turn_stalls = min(4, turn_stalls + 1)
                if turn_stalls >= 4:
                    force_forward_arc = True
            else:
                turn_stalls = 0
        else:
            turn_stalls = 0
        collisions += int(collision)
        if collision:
            # A forward-arc escape is only a turn-stiction workaround.  Once
            # it intersects the inflated map, production must return to a
            # stopped heading correction before attempting that corner again.
            force_forward_arc = False
            blocked_heading_recovery = True
        consecutive_collisions = (
            consecutive_collisions + 1 if collision else 0
        )
        if consecutive_collisions >= 3 and replans < 4:
            replanned = known_map.path(
                (truth.x_m, truth.y_m), target.view_xy_m
            )
            if replanned:
                waypoints = replanned[1:] or [target.view_xy_m]
                waypoint_index = 0
                consecutive_collisions = 0
                replans += 1
                continue
        if consecutive_collisions >= 6:
            break
    reference_ok, reference_distance, limit_m = _within_capture_reference(
        truth,
        target.target_id,
        reference_views,
        verified,
        args.nearby_capture_distance,
    )
    verified_pose = verified.get(target.target_id)
    desired_yaw = (
        verified_pose.yaw_rad
        if verified_pose is not None and reference_distance <= 0.32
        else math.atan2(
            target.target_xy_m[1] - truth.y_m,
            target.target_xy_m[0] - truth.x_m,
        )
    )
    truth, heading = _align_camera(
        known_map,
        truth,
        target,
        normalize_angle(desired_yaw),
        args,
        rng,
        motor_gain,
        turn_floor,
        straight_bias,
    )
    heading.update({
        "target_id": target.target_id,
        "reference_distance_m": reference_distance,
        "reference_limit_m": limit_m,
        "drive_steps": steps,
        "collision_frames": collisions,
        "replans": replans,
        "verified_view": target.target_id in verified,
        "final_truth_pose": {
            "x_m": truth.x_m,
            "y_m": truth.y_m,
            "yaw_deg": math.degrees(truth.yaw_rad),
        },
        "waypoint_index": waypoint_index,
        "waypoint_count": len(waypoints),
        "active_waypoint": list(waypoints[waypoint_index]),
        "captured": bool(reference_ok and heading["heading_ok"]),
    })
    return truth, heading


def simulate_seed(known_map, targets, verified, args, seed):
    rng = np.random.default_rng(int(seed))
    truth = FIELD_START_POSE
    pending = list(targets)
    reference_views = {
        target.target_id: tuple(target.view_xy_m) for target in targets
    }
    captures = []
    selected_legs = []
    attempts = {}
    retry_origins = {}
    while pending:
        choices = []
        for index, target in enumerate(pending):
            route_proposal = _inspection_path_for_target(
                known_map,
                truth,
                target,
                args.target_turn_cost_m_per_rad,
            )
            if route_proposal is not None:
                path = route_proposal["path"]
                route_metrics = route_proposal["metrics"]
                choices.append((
                    route_metrics["score"],
                    route_metrics["length_m"],
                    target.target_id,
                    index,
                    path,
                    route_metrics["departure_turn_rad"],
                    route_metrics["arrival_turn_rad"],
                    route_proposal["staged_camera_approach"],
                ))
        if not choices:
            break
        choices = _prefer_displaced_retry_choices(
            choices, truth, retry_origins, args.target_retry_min_displacement
        )
        (
            _score,
            _length,
            _target_id,
            index,
            path,
            planned_departure_turn_rad,
            planned_arrival_turn_rad,
            staged_camera_approach,
        ) = min(choices)
        target = pending.pop(index)
        truth, result = _drive_target(
            known_map,
            truth,
            target,
            path,
            reference_views,
            verified,
            args,
            rng,
        )
        attempts[target.target_id] = attempts.get(target.target_id, 0) + 1
        result["attempt"] = attempts[target.target_id]
        result["planned_departure_turn_deg"] = math.degrees(
            planned_departure_turn_rad
        )
        result["planned_arrival_turn_deg"] = math.degrees(
            planned_arrival_turn_rad
        )
        selected_legs.append({
            "target_id": target.target_id,
            "attempt": attempts[target.target_id],
            "path_length_m": _length,
            "departure_turn_deg": math.degrees(planned_departure_turn_rad),
            "arrival_turn_deg": math.degrees(planned_arrival_turn_rad),
            "staged_camera_approach": bool(staged_camera_approach),
        })
        # A failed target returns only after the vehicle has visited another
        # area, up to the same configured attempt budget as production.
        if (
            not result["captured"]
            and attempts[target.target_id] < args.max_target_navigation_attempts
        ):
            retry_origins[target.target_id] = (truth.x_m, truth.y_m)
            pending.append(target)
            continue
        retry_origins.pop(target.target_id, None)
        captures.append(result)
    return {
        "seed": int(seed),
        "target_count": len(targets),
        "capture_count": sum(item["captured"] for item in captures),
        "maximum_heading_error_deg": max(
            (item["heading_error_deg"] for item in captures), default=math.inf
        ),
        "maximum_reference_distance_m": max(
            (item["reference_distance_m"] for item in captures), default=math.inf
        ),
        "first_selected_leg": selected_legs[0] if selected_legs else None,
        "maximum_planned_arrival_turn_deg": max(
            (item["arrival_turn_deg"] for item in selected_legs),
            default=math.inf,
        ),
        "near_uturn_selected_leg_count": sum(
            item["arrival_turn_deg"] >= 150.0 for item in selected_legs
        ),
        "selected_legs": selected_legs,
        "captures": captures,
        "passed": len(captures) == len(targets) and all(
            item["captured"] for item in captures
        ),
    }


def run(
    seed_count=100,
    output="car_sim/output/known_map_closed_loop.json",
    turn_cost_m_per_rad=0.35,
    target_limit=None,
    clearance_m=0.30,
):
    args = build_argument_parser().parse_args([
        "--map", "maps/ble_scene_17.pgm",
        "--map-yaml", "maps/ble_scene_17.yaml",
        "--clearance", str(clearance_m),
        "--standoff", "1.00",
        "--max-targets", "24",
        "--views-per-wall", "3",
        "--views-per-region", "1",
        "--min-target-spacing", "0.40",
        "--information-view-distance", "2.20",
        "--target-turn-cost-m-per-rad", str(turn_cost_m_per_rad),
        "--max-target-navigation-attempts", "4",
        "--nearby-capture-distance", "0.35",
        "--waypoint-tolerance", "0.18",
        "--heading-tolerance-deg", "7",
        "--speed", "330",
        "--max-drive-speed", "450",
        "--start-kick-speed", "420",
        "--slow-speed", "240",
        "--navigation-fine-control-distance", "0.60",
        "--turn-speed", "2400",
        "--relocalization-max-turn-speed", "3000",
        "--navigation-fine-turn-speed", "3000",
        "--navigation-fine-turn-min-speed", "3000",
        "--navigation-precision-turn-ttl-ms", "180",
        "--navigation-heading-seed-frames", "5",
        "--navigation-timeout-capture-max-heading-error-deg", "45",
    ])
    known_map = KnownOccupancyMap.load(
        args.map, args.map_yaml, clearance_m=args.clearance
    )
    raw_targets = known_map.inspection_targets(
        standoff_m=args.standoff,
        maximum_targets=args.max_targets,
        views_per_wall=args.views_per_wall,
        views_per_region=args.views_per_region,
        preferred_view_xy=(FIELD_START_POSE.x_m, FIELD_START_POSE.y_m),
    )
    semantic = _load_semantic("maps/ble_scene_17_semantics_qwen.json")
    targets = _spaced_targets(
        raw_targets, min_spacing_m=args.min_target_spacing
    )
    targets, skipped_components = _skip_classified_interior_targets(
        targets, semantic
    )
    # inspect_map.sh deliberately ignores old verified photo poses: those
    # poses came from earlier, occasionally wrong localization estimates.
    verified = {}
    targets, information_views = _information_view_targets(
        known_map, targets, args.information_view_distance
    )
    if target_limit is not None:
        ordered_preview = order_targets(
            known_map,
            (FIELD_START_POSE.x_m, FIELD_START_POSE.y_m),
            targets,
            start_yaw_rad=FIELD_START_POSE.yaw_rad,
            turn_cost_m_per_rad=args.target_turn_cost_m_per_rad,
        )
        targets = [
            target for target, _path in ordered_preview[:int(target_limit)]
        ]
    seeds = [
        simulate_seed(known_map, targets, verified, args, seed)
        for seed in range(int(seed_count))
    ]
    failed = [item for item in seeds if not item["passed"]]
    target_statistics = {}
    for target in targets:
        rows = [
            capture
            for seed in seeds
            for capture in seed["captures"]
            if capture["target_id"] == target.target_id
        ]
        errors = [item["heading_error_deg"] for item in rows]
        target_statistics[target.target_id] = {
            "capture_count": sum(item["captured"] for item in rows),
            "expected_capture_count": len(seeds),
            "maximum_heading_error_deg": max(errors, default=math.inf),
            "p95_heading_error_deg": (
                float(np.percentile(errors, 95.0)) if errors else math.inf
            ),
            "verified_view": target.target_id in verified,
            "reference_view": list(target.view_xy_m),
        }
    result = {
        "map": args.map,
        "field_start_pose": {
            "x_m": FIELD_START_POSE.x_m,
            "y_m": FIELD_START_POSE.y_m,
            "yaw_deg": math.degrees(FIELD_START_POSE.yaw_rad),
        },
        "raw_target_count": len(raw_targets),
        "deduplicated_target_count": len(targets),
        "removed_duplicate_count": len(raw_targets) - len(targets),
        "skipped_semantic_components": sorted(skipped_components),
        "information_view_count": len(information_views),
        "verified_target_count": len(verified),
        "target_ids": [item.target_id for item in targets],
        "seed_count": len(seeds),
        "passed_seed_count": len(seeds) - len(failed),
        "failed_seed_count": len(failed),
        "total_captures": sum(item["capture_count"] for item in seeds),
        "expected_total_captures": len(targets) * len(seeds),
        "maximum_heading_error_deg": max(
            item["maximum_heading_error_deg"] for item in seeds
        ),
        "maximum_first_leg_arrival_turn_deg": max(
            float(item["first_selected_leg"]["arrival_turn_deg"])
            for item in seeds if item["first_selected_leg"] is not None
        ),
        "total_near_uturn_selected_legs": sum(
            item["near_uturn_selected_leg_count"] for item in seeds
        ),
        "target_statistics": target_statistics,
        "representative_seed": seeds[0] if seeds else None,
        "passed": bool(
            not failed
            and all(
                item["first_selected_leg"] is not None
                and float(item["first_selected_leg"]["arrival_turn_deg"])
                < 90.0
                for item in seeds
            )
        ),
        "failed_seeds": failed[:8],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="real-PGM closed-loop inspection")
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument(
        "--output", default="car_sim/output/known_map_closed_loop.json"
    )
    parser.add_argument("--turn-cost-m-per-rad", type=float, default=0.35)
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--clearance", type=float, default=0.30)
    cli = parser.parse_args()
    result = run(
        cli.seed_count,
        cli.output,
        cli.turn_cost_m_per_rad,
        cli.target_limit,
        cli.clearance,
    )
    summary = dict(result)
    summary.pop("failed_seeds", None)
    summary.pop("target_statistics", None)
    summary.pop("representative_seed", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
