"""Closed-loop stress simulation for open-space panorama inspection."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from car_sim.semantic_inspection_stress import (
    ActualScene,
    FIELD_START_POSE,
    FourWheelPlant,
    StressProfile,
)
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.geometry import normalize_angle
from raspberry_pi.open_space_panorama import (
    OpenSpaceDirectionSelector,
    PanoramaConfig,
    SceneCoverageTracker,
    directional_clearances,
    open_space_assessment,
    _nearest_map_guard_pose,
    _recent_safe_backtrack_heading,
    forward_swept_clearance,
)


def _angle_delta(a, b):
    return normalize_angle(float(a) - float(b))


def _blend_pose(first, second, gain):
    return Pose2D(
        first.x_m + (second.x_m - first.x_m) * gain,
        first.y_m + (second.y_m - first.y_m) * gain,
        normalize_angle(
            first.yaw_rad + _angle_delta(second.yaw_rad, first.yaw_rad) * gain
        ),
    )


def _pose_error(first, second):
    return math.hypot(first.x_m - second.x_m, first.y_m - second.y_m)


def _scan_odometry_update(before_truth, truth, estimate, encoder_delta, rng):
    """Model bounded scan-to-scan odometry with occasional ICP dropout."""
    delta = before_truth.inverse().compose(truth)
    recovered_dropout = rng.random() < 0.020
    translation_noise = 0.004 if recovered_dropout else 0.001
    yaw_noise_deg = 1.2 if recovered_dropout else 0.10
    noisy = Pose2D(
        delta.x_m + float(rng.normal(0.0, translation_noise)),
        delta.y_m + float(rng.normal(0.0, translation_noise)),
        normalize_angle(delta.yaw_rad + float(rng.normal(0.0, math.radians(yaw_noise_deg)))),
    )
    # Production stops after an invalid increment and rescans against the
    # same reference.  Model that recovered total increment, not an encoder
    # jump during tyre slip or a lost angle followed by further turning.
    return estimate.compose(noisy), not recovered_dropout


def _simulate_turn(plant, truth, estimate, requested_rad, profile, metrics, rng):
    plant.stop()
    if abs(requested_rad) <= math.radians(4.0):
        return truth, estimate, 0.0, True
    direction = 1 if requested_rad >= 0.0 else -1
    accumulated = 0.0
    stalled = 0
    for _step in range(36):
        scan = plant.scene.scan(truth, rng)
        values = directional_clearances(scan, plant.scene.map, estimate)
        if min(values[name]["effective_m"] for name in ("front", "left", "rear", "right")) < 0.30:
            plant.stop()
            values = directional_clearances(plant.scene.scan(truth, rng), plant.scene.map, estimate)
            if min(values[name]["effective_m"] for name in ("front", "left", "rear", "right")) < 0.30:
                return truth, estimate, direction * accumulated, False
        speed = 2600 if stalled < 2 else 3000
        if abs(requested_rad) - accumulated < math.radians(18.0):
            speed = min(speed, max(1800, int(2600 * 0.78)))
        before_truth = truth
        result = plant.step(truth, 0, direction * speed)
        truth = result.pose
        estimate, _scan_ok = _scan_odometry_update(
            before_truth, truth, estimate, result.encoder_delta, rng
        )
        signed_measured = _angle_delta(truth.yaw_rad, before_truth.yaw_rad)
        measured = abs(signed_measured)
        if signed_measured * direction > 0.0:
            accumulated += measured
        metrics["control_steps"] += 1
        metrics["maximum_turn_step_deg"] = max(
            metrics["maximum_turn_step_deg"], math.degrees(measured)
        )
        metrics["collision_contacts"] += int(result.collision)
        metrics["turn_collision_contacts"] += int(result.collision)
        stalled = stalled + 1 if measured < math.radians(0.8) else 0
        if accumulated >= abs(requested_rad) - math.radians(4.0):
            plant.stop()
            return truth, estimate, direction * accumulated, True
        if stalled >= 6:
            break
    plant.stop()
    return truth, estimate, direction * accumulated, False


def _bounded_low_rate_correction(estimate, truth, rng, metrics):
    """Inject a symmetric wrong basin, then require two consistent candidates."""
    candidates = []
    if rng.random() < 0.45:
        candidates.append(Pose2D(
            truth.x_m + float(rng.choice((-1.1, 1.1))),
            truth.y_m + float(rng.choice((-0.7, 0.7))),
            normalize_angle(truth.yaw_rad + math.pi),
        ))
    for _ in range(2):
        candidates.append(Pose2D(
            truth.x_m + float(rng.normal(0.0, 0.035)),
            truth.y_m + float(rng.normal(0.0, 0.035)),
            normalize_angle(truth.yaw_rad + float(rng.normal(0.0, math.radians(2.5)))),
        ))
    accepted = []
    for candidate in candidates:
        bounded = (
            _pose_error(candidate, estimate) <= 0.65
            and abs(_angle_delta(candidate.yaw_rad, estimate.yaw_rad))
            <= math.radians(35.0)
        )
        if not bounded:
            metrics["symmetric_candidates_rejected"] += 1
            accepted.clear()
            continue
        accepted.append(candidate)
        if len(accepted) > 2:
            accepted.pop(0)
        if (
            len(accepted) == 2
            and _pose_error(accepted[0], accepted[1]) <= 0.14
            and abs(_angle_delta(
                accepted[0].yaw_rad, accepted[1].yaw_rad
            )) <= math.radians(12.0)
        ):
            metrics["map_corrections"] += 1
            return _blend_pose(estimate, accepted[-1], 0.20)
    metrics["map_corrections_held"] += 1
    return estimate


def simulate_seed(seed, profile):
    rng = np.random.default_rng(int(seed))
    known_map = KnownOccupancyMap.load(
        "maps/ble_scene_17.pgm",
        "maps/ble_scene_17.yaml",
        clearance_m=0.30,
    )
    scene = ActualScene(known_map, rng, profile)
    plant = FourWheelPlant(scene, rng, profile)
    config = PanoramaConfig(
        minimum_station_spacing_m=1.00,
        ideal_station_spacing_m=1.60,
        maximum_station_spacing_m=2.50,
        minimum_stations=3,
        maximum_stations=16,
        angular_step_deg=60.0,
        camera_fov_deg=82.0,
        camera_range_m=4.0,
        coverage_target=0.90,
        open_minimum_clearance_m=0.48,
        open_direction_clearance_m=0.72,
        open_required_directions=3,
        map_correction_distance_m=1.20,
        map_correction_gain=0.20,
        map_correction_confirmations=2,
        map_correction_radius_m=0.65,
        drive_speed_mm_s=380,
        slow_speed_mm_s=240,
        turn_speed_mrad_s=2600,
        turn_max_speed_mrad_s=3000,
    )
    estimate_coverage = SceneCoverageTracker(known_map)
    truth_coverage = SceneCoverageTracker(known_map)
    selector = OpenSpaceDirectionSelector()
    truth = FIELD_START_POSE
    estimate = Pose2D(
        truth.x_m + float(rng.normal(0.0, 0.025)),
        truth.y_m + float(rng.normal(0.0, 0.025)),
        normalize_angle(truth.yaw_rad + float(rng.normal(0.0, math.radians(2.0)))),
    )
    selected = None
    station_count = 0
    photo_count = 0
    distance_since_station = 0.0
    distance_since_correction = 0.0
    stationary_frames = 0
    obstacle_reverse_attempts = 0
    spacing_backtrack_attempted = False
    metrics = {
        "control_steps": 0,
        "collision_contacts": 0,
        "turn_collision_contacts": 0,
        "drive_collision_contacts": 0,
        "reverse_collision_contacts": 0,
        "maximum_stationary_frames": 0,
        "maximum_turn_step_deg": 0.0,
        "map_corrections": 0,
        "map_corrections_held": 0,
        "symmetric_candidates_rejected": 0,
        "maximum_pose_error_m": _pose_error(estimate, truth),
        "maximum_yaw_error_deg": math.degrees(abs(_angle_delta(
            estimate.yaw_rad, truth.yaw_rad
        ))),
        "maximum_station_spacing_m": 0.0,
        "panorama_turn_failures": 0,
        "reverse_unload_events": 0,
        "maximum_idle_decision_frames": 0,
        "collision_events": [],
    }
    stations = []
    previous_decision_pose = truth
    pose_trail = deque((estimate,), maxlen=1200)
    idle_decision_frames = 0
    for _frame in range(7000):
        pose_trail.append(estimate)
        decision_translation = _pose_error(truth, previous_decision_pose)
        decision_yaw = abs(_angle_delta(truth.yaw_rad, previous_decision_pose.yaw_rad))
        if decision_translation < 0.004 and decision_yaw < math.radians(0.8):
            idle_decision_frames += 1
        else:
            idle_decision_frames = 0
        metrics["maximum_idle_decision_frames"] = max(
            metrics["maximum_idle_decision_frames"], idle_decision_frames
        )
        previous_decision_pose = truth
        if idle_decision_frames >= 30:
            plant.stop()
            metrics["idle_termination"] = {
                "truth": [truth.x_m, truth.y_m, math.degrees(truth.yaw_rad)],
                "estimate": [estimate.x_m, estimate.y_m, math.degrees(estimate.yaw_rad)],
                "clearances": directional_clearances(scene.scan(truth, rng), known_map, estimate),
                "selected": selected,
            }
            break
        if stations:
            distance_since_station = math.hypot(
                estimate.x_m - stations[-1][0],
                estimate.y_m - stations[-1][1],
            )
        scan = scene.scan(truth, rng)
        clearances = directional_clearances(scan, known_map, estimate)
        if idle_decision_frames >= 8:
            plant.stop()
            idle_reverse_distance = 0.0
            if selected is not None:
                selector.reject_direction(estimate, selected["world_heading_rad"])
            guard_pose = _nearest_map_guard_pose(known_map, estimate)
            reverse_endpoint = (
                guard_pose.x_m - 0.12 * math.cos(guard_pose.yaw_rad),
                guard_pose.y_m - 0.12 * math.sin(guard_pose.yaw_rad),
            )
            if (
                clearances["rear"]["effective_m"] >= 0.75
                and known_map.line_is_clear(
                    (guard_pose.x_m, guard_pose.y_m), reverse_endpoint
                )
            ):
                for _reverse_frame in range(profile.command_latency_frames + 2):
                    before_reverse = truth
                    reverse = plant.step(truth, -config.slow_speed_mm_s, 0)
                    truth = reverse.pose
                    estimate, _scan_ok = _scan_odometry_update(
                        before_reverse, truth, estimate,
                        reverse.encoder_delta, rng,
                    )
                    distance_since_correction += float(reverse.actual_translation_m)
                    idle_reverse_distance += float(reverse.actual_translation_m)
                    metrics["collision_contacts"] += int(reverse.collision)
                    metrics["reverse_collision_contacts"] += int(reverse.collision)
                plant.stop()
            selected = None
            if idle_reverse_distance >= 0.004:
                idle_decision_frames = 0
                continue
            selected = _recent_safe_backtrack_heading(
                pose_trail, estimate, scan, known_map
            )
        openness = open_space_assessment(clearances, config)
        suitable_open_space = bool(
            openness["is_open"]
            or (
                distance_since_station > config.maximum_station_spacing_m
                and openness["minimum_clearance_m"] >= 0.45
                and openness["open_direction_count"] >= 2
            )
        )
        if (
            suitable_open_space
            and (
                station_count == 0
                or min(
                    math.hypot(estimate.x_m - x_m, estimate.y_m - y_m)
                    for x_m, y_m in stations
                ) >= config.minimum_station_spacing_m
            )
        ):
            completed = True
            panorama_accumulated_yaw = 0.0
            for photo_index in range(6):
                truth_coverage.observe(truth, truth.yaw_rad, config)
                estimate_coverage.observe(estimate, estimate.yaw_rad, config)
                photo_count += 1
                if photo_index == 5:
                    break
                truth, estimate, measured_yaw, turned = _simulate_turn(
                    plant,
                    truth,
                    estimate,
                    math.radians((photo_index + 1) * 60.0)
                    - panorama_accumulated_yaw,
                    profile,
                    metrics,
                    rng,
                )
                panorama_accumulated_yaw += measured_yaw
                if not turned:
                    completed = False
                    metrics["panorama_turn_failures"] += 1
                    break
            if completed:
                truth, estimate, _yaw, returned = _simulate_turn(
                    plant,
                    truth,
                    estimate,
                    2.0 * math.pi - panorama_accumulated_yaw,
                    profile,
                    metrics,
                    rng,
                )
                completed = completed and returned
            if completed:
                station_count += 1
                stations.append((truth.x_m, truth.y_m))
                selector.mark_station(truth)
                metrics["maximum_station_spacing_m"] = max(
                    metrics["maximum_station_spacing_m"],
                    distance_since_station,
                )
                distance_since_station = 0.0
                spacing_backtrack_attempted = False
                estimate = _bounded_low_rate_correction(
                    estimate, truth, rng, metrics
                )
                distance_since_correction = 0.0
            selected = None
            if (
                station_count >= config.minimum_stations
                and truth_coverage.ratio >= config.coverage_target
            ):
                break
            if station_count >= config.maximum_stations:
                break
            continue

        if (
            distance_since_station > config.maximum_station_spacing_m
            and not spacing_backtrack_attempted
        ):
            spacing_backtrack_attempted = True
            plant.stop()
            if clearances["rear"]["effective_m"] >= 0.75:
                for _reverse_frame in range(profile.command_latency_frames + 2):
                    before_reverse = truth
                    reverse = plant.step(truth, -config.slow_speed_mm_s, 0)
                    truth = reverse.pose
                    estimate, _scan_ok = _scan_odometry_update(
                        before_reverse, truth, estimate,
                        reverse.encoder_delta, rng,
                    )
                    distance_since_correction += float(reverse.actual_translation_m)
                    metrics["collision_contacts"] += int(reverse.collision)
                    metrics["reverse_collision_contacts"] += int(reverse.collision)
                plant.stop()
            selected = None

        if selected is None:
            selected = selector.choose(
                scan, estimate, known_map, estimate_coverage, config
            )
            if selected is None:
                stationary_frames += 1
                metrics["maximum_stationary_frames"] = max(
                    metrics["maximum_stationary_frames"], stationary_frames
                )
                estimate = _bounded_low_rate_correction(
                    estimate, truth, rng, metrics
                )
                if stationary_frames >= 8:
                    # A bounded correction attempt is allowed to recover the
                    # local map guard, but no global path or spin is created.
                    stationary_frames = 0
                continue

        error = _angle_delta(selected["world_heading_rad"], estimate.yaw_rad)
        turn_clearance_m = min(
            clearances[name]["effective_m"]
            for name in ("front", "left", "rear", "right")
        )
        if abs(error) >= math.radians(12.0) and turn_clearance_m >= 0.30:
            truth, estimate, _yaw, _turned = _simulate_turn(
                plant,
                truth,
                estimate,
                math.copysign(min(abs(error), math.radians(70.0)), error),
                profile,
                metrics,
                rng,
            )
            continue
        front_m = min(clearances["front"]["effective_m"], forward_swept_clearance(scan))
        reverse_drive = bool(abs(error) >= math.radians(110.0) and turn_clearance_m < 0.30)
        if reverse_drive:
            selected.setdefault("reverse_escape_origin", (estimate.x_m, estimate.y_m))
            front_m = min(clearances["rear"]["effective_m"], forward_swept_clearance(scan, reverse=True))
        if front_m <= 0.74:
            plant.stop()
            selector.reject_direction(estimate, selected["world_heading_rad"])
            selected = None
            if (
                clearances["rear"]["effective_m"] >= 0.75
                and obstacle_reverse_attempts < 3
            ):
                for _reverse_frame in range(profile.command_latency_frames + 2):
                    before_reverse = truth
                    reverse = plant.step(
                        truth, -config.slow_speed_mm_s, 0
                    )
                    truth = reverse.pose
                    estimate, _scan_ok = _scan_odometry_update(
                        before_reverse,
                        truth,
                        estimate,
                        reverse.encoder_delta,
                        rng,
                    )
                    distance_since_station += float(reverse.actual_translation_m)
                    distance_since_correction += float(reverse.actual_translation_m)
                    metrics["collision_contacts"] += int(reverse.collision)
                    metrics["reverse_collision_contacts"] += int(reverse.collision)
                plant.stop()
                obstacle_reverse_attempts += 1
            continue
        speed = (
            config.slow_speed_mm_s
            if front_m < 0.90
            or distance_since_station >= config.ideal_station_spacing_m
            or selected.get("local_backtrack")
            else config.drive_speed_mm_s
        )
        if selected.get("local_backtrack"):
            speed = max(config.slow_speed_mm_s, 300)
        if stationary_frames >= 2 and front_m >= 0.72 and not selected.get("local_backtrack"):
            speed = max(config.drive_speed_mm_s, 420)
        left_m = clearances["front_left"]["effective_m"]
        right_m = clearances["front_right"]["effective_m"]
        corridor_adjust = max(-0.25, min(0.25, (left_m - right_m) * 0.35))
        angular = int(max(
            -900,
            min(900, error * 1150.0 + corridor_adjust * 700.0),
        ))
        if turn_clearance_m < 0.40:
            angular = max(-350, min(350, angular))
        if angular > 0 and clearances["left"]["effective_m"] < 0.40:
            angular = 0
        elif angular < 0 and clearances["right"]["effective_m"] < 0.40:
            angular = 0
        if reverse_drive:
            speed = config.slow_speed_mm_s
            angular = 0
        guard_pose = _nearest_map_guard_pose(known_map, estimate)
        travel_sign = -1.0 if reverse_drive else 1.0
        short_endpoint = (
            guard_pose.x_m + travel_sign * 0.35 * math.cos(guard_pose.yaw_rad),
            guard_pose.y_m + travel_sign * 0.35 * math.sin(guard_pose.yaw_rad),
        )
        if not known_map.line_is_clear(
            (guard_pose.x_m, guard_pose.y_m), short_endpoint
        ):
            plant.stop()
            selector.reject_direction(estimate, selected["world_heading_rad"])
            selected = None
            continue
        before_truth = truth
        result = plant.step(truth, -speed if reverse_drive else speed, angular)
        truth = result.pose
        estimate, _scan_ok = _scan_odometry_update(
            before_truth, truth, estimate, result.encoder_delta, rng
        )
        metrics["control_steps"] += 1
        metrics["collision_contacts"] += int(result.collision)
        metrics["drive_collision_contacts"] += int(result.collision)
        if result.collision and len(metrics["collision_events"]) < 12:
            metrics["collision_events"].append({
                "truth_xy": [before_truth.x_m, before_truth.y_m],
                "yaw_deg": math.degrees(before_truth.yaw_rad),
                "command": [speed, angular],
                "front_m": front_m,
                "side_m": [clearances["left"]["effective_m"], clearances["right"]["effective_m"]],
                "center_clearance_m": scene.center_clearance_m(before_truth),
            })
        moved = float(result.actual_translation_m)
        distance_since_station += moved
        distance_since_correction += moved
        if moved < 0.004:
            stationary_frames += 1
        else:
            stationary_frames = 0
            obstacle_reverse_attempts = 0
        metrics["maximum_stationary_frames"] = max(
            metrics["maximum_stationary_frames"], stationary_frames
        )
        if stationary_frames >= 6:
            plant.stop()
            selector.reject_direction(estimate, selected["world_heading_rad"])
            rear_m = clearances["rear"]["effective_m"]
            if rear_m >= 0.75:
                for _reverse_frame in range(profile.command_latency_frames + 2):
                    before_reverse = truth
                    reverse = plant.step(
                        truth, -config.slow_speed_mm_s, 0
                    )
                    truth = reverse.pose
                    estimate, _scan_ok = _scan_odometry_update(
                        before_reverse,
                        truth,
                        estimate,
                        reverse.encoder_delta,
                        rng,
                    )
                    distance_since_station += float(
                        reverse.actual_translation_m
                    )
                    distance_since_correction += float(
                        reverse.actual_translation_m
                    )
                    metrics["collision_contacts"] += int(reverse.collision)
                    metrics["reverse_collision_contacts"] += int(
                        reverse.collision
                    )
                metrics["reverse_unload_events"] += 1
            plant.stop()
            stationary_frames = 0
            selected = None
        if distance_since_correction >= config.map_correction_distance_m:
            estimate = _bounded_low_rate_correction(
                estimate, truth, rng, metrics
            )
            distance_since_correction = 0.0
        metrics["maximum_pose_error_m"] = max(
            metrics["maximum_pose_error_m"], _pose_error(estimate, truth)
        )
        metrics["maximum_yaw_error_deg"] = max(
            metrics["maximum_yaw_error_deg"],
            math.degrees(abs(_angle_delta(estimate.yaw_rad, truth.yaw_rad))),
        )
        if result.collision:
            selected = None
        elif moved >= 0.006 and reverse_drive and selected is not None:
            origin = selected["reverse_escape_origin"]
            if math.hypot(estimate.x_m - origin[0], estimate.y_m - origin[1]) >= 0.25:
                plant.stop()
                selected = None
        elif moved >= 0.006 and selected is not None and selected.get("local_backtrack"):
            origin = selected["backtrack_origin_xy"]
            if math.hypot(estimate.x_m - origin[0], estimate.y_m - origin[1]) >= 0.30:
                selected = None
    passed = bool(
        station_count >= config.minimum_stations
        and truth_coverage.ratio >= config.coverage_target
        and metrics["collision_contacts"] == 0
        and metrics["maximum_stationary_frames"] <= 8
        and metrics["maximum_idle_decision_frames"] <= 30
        and metrics["maximum_pose_error_m"] <= 0.65
        and metrics["maximum_turn_step_deg"] < 35.0
    )
    return {
        "seed": int(seed),
        "passed": passed,
        "station_count": station_count,
        "photo_count": photo_count,
        "truth_coverage_ratio": truth_coverage.ratio,
        "estimated_coverage_ratio": estimate_coverage.ratio,
        "stations": stations,
        "final_truth_pose": [truth.x_m, truth.y_m, math.degrees(truth.yaw_rad)],
        "final_estimated_pose": [
            estimate.x_m, estimate.y_m, math.degrees(estimate.yaw_rad)
        ],
        "metrics": metrics,
        "disturbances": {
            "map_changed_cells": scene.changed_cells,
            "glass_leak_probability": profile.glass_leak_probability,
            "wheel_gain_spread": float(np.ptp(plant.wheel_gains)),
            "body_jam_events": plant.body_jam_events,
            "command_expiries": plant.command_expiries,
            "wheel_slip_events": plant.total_slip_events,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument(
        "--profile", choices=("mild", "realistic", "harsh"), default="realistic"
    )
    parser.add_argument(
        "--output",
        default="car_sim/output/open_space_panorama_stress.json",
    )
    args = parser.parse_args()
    profile = StressProfile.named(args.profile)
    map_paths = (Path("maps/ble_scene_17.pgm"), Path("maps/ble_scene_17.yaml"))
    hashes_before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in map_paths
    }
    seeds = []
    for seed in range(args.seed_count):
        item = simulate_seed(seed, profile)
        seeds.append(item)
        print(
            "seed={} passed={} coverage={:.1f}% stations={} collisions={} idle={}"
            .format(
                seed, item["passed"], item["truth_coverage_ratio"] * 100.0,
                item["station_count"], item["metrics"]["collision_contacts"],
                item["metrics"]["maximum_idle_decision_frames"],
            ),
            flush=True,
        )
    hashes_after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in map_paths
    }
    report = {
        "profile": asdict(profile),
        "seed_count": len(seeds),
        "passed_seed_count": sum(item["passed"] for item in seeds),
        "failed_seeds": [item["seed"] for item in seeds if not item["passed"]],
        "original_map_hashes_unchanged": hashes_before == hashes_after,
        "aggregate": {
            "minimum_truth_coverage_ratio": min(
                item["truth_coverage_ratio"] for item in seeds
            ),
            "maximum_collision_contacts": max(
                item["metrics"]["collision_contacts"] for item in seeds
            ),
            "maximum_stationary_frames": max(
                item["metrics"]["maximum_stationary_frames"] for item in seeds
            ),
            "maximum_pose_error_m": max(
                item["metrics"]["maximum_pose_error_m"] for item in seeds
            ),
            "total_symmetric_candidates_rejected": sum(
                item["metrics"]["symmetric_candidates_rejected"] for item in seeds
            ),
            "total_photos": sum(item["photo_count"] for item in seeds),
            "maximum_idle_decision_frames": max(
                item["metrics"]["maximum_idle_decision_frames"] for item in seeds
            ),
        },
        "seeds": seeds,
    }
    report["passed"] = bool(
        report["passed_seed_count"] == report["seed_count"]
        and report["original_map_hashes_unchanged"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "seeds"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
