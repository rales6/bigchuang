"""Run the production panorama state machine against a disturbed plant.

Hardware and scan odometry are simulated; navigation, capture sequencing,
candidate acceptance and saving execute the Raspberry Pi production code.
This is not an evaluation of real ICP or Qwen model accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from car_sim.semantic_inspection_stress import ActualScene, FourWheelPlant, StressProfile, FIELD_START_POSE
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.map_camera_inspection import InspectionReport, build_argument_parser
from raspberry_pi.mapping.geometry import Pose2D, normalize_angle
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.open_space_panorama import PanoramaConfig, SceneCoverageTracker, run_open_space_panorama


def simulate(seed, profile, inject_body_returns=False, camera_delay_s=2.1, map_match_delay_s=.25,
             lidar_offset_x_m=.14, lidar_offset_y_m=0.):
    rng = np.random.default_rng(seed)
    # Cast rays from the installed sensor, not the body center. Retain the
    # conservative plant radius (20 cm rather than the nominal 16.3 cm).
    config = PanoramaConfig(lidar_offset_x_m=lidar_offset_x_m, lidar_offset_y_m=lidar_offset_y_m)
    known_map = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml",
                                      clearance_m=config.map_clearance_m(.22))
    scene = ActualScene(known_map, rng, profile)
    plant = FourWheelPlant(scene, rng, profile)
    truth = FIELD_START_POSE
    elapsed = 0.0
    command = (0, 0)
    expires_at = 0.0
    steps = 0
    collisions = 0
    symmetric_candidates = 0
    maximum_pose_error = 0.0
    direction_errors = []
    photo_names = []
    truth_coverage = SceneCoverageTracker(known_map)

    def stop():
        nonlocal command
        command = (0, 0)
        plant.stop()

    def twist(linear, angular, ttl_ms=350):
        nonlocal command, expires_at
        command = (linear, angular)
        expires_at = elapsed + ttl_ms / 1000.0

    def scans():
        nonlocal truth, elapsed, steps, collisions
        while True:
            linear, angular = command if elapsed < expires_at else (0, 0)
            if steps:
                result = plant.step(truth, linear, angular)
                truth = result.pose
                collisions += int(result.collision)
            elapsed += profile.dt_s
            steps += 1
            sensor_pose = truth.compose(Pose2D(config.lidar_offset_x_m, config.lidar_offset_y_m, 0.))
            frame = scene.scan(sensor_pose, rng)
            angles, distances = frame.angles_rad, frame.distances_m
            if inject_body_returns:
                # Fixed sensor-frame clusters observed in the real stationary
                # evidence, absent from the former independent speckle model.
                # Single-return lidar: occluded rays must REPLACE the room
                # returns, not coexist with a fictitious clear ray behind them.
                degrees = np.degrees(angles) % 360.0
                distances = distances.copy()
                distances[(degrees >= 147) & (degrees <= 223)] = 0.082
                distances[(degrees >= 249) & (degrees <= 270)] = 0.121
            yield SimpleNamespace(angles_rad=angles, distances_m=distances, timestamp_s=elapsed, truth_pose=truth)

    def relative(reference, current):
        delta = reference.truth_pose.inverse().compose(current.truth_pose)
        if rng.random() < 0.02:
            return MatchResult(Pose2D(), math.inf, 0, False, 0.0)
        transform = Pose2D(
            delta.x_m + float(rng.normal(0, 0.001)),
            delta.y_m + float(rng.normal(0, 0.001)),
            normalize_angle(delta.yaw_rad + float(rng.normal(0, math.radians(0.10)))),
        )
        return MatchResult(transform, 0.01, 100, True, 0.95)

    def localize(_scan, _pose, radius_m, budget_s=.35, yaw_radius_deg=15.):
        nonlocal symmetric_candidates, elapsed
        elapsed += min(map_match_delay_s, budget_s)
        if map_match_delay_s > budget_s:
            return None
        if rng.random() < 0.45:
            symmetric_candidates += 1
            candidate = Pose2D(truth.x_m + 1.1, truth.y_m - 0.7, normalize_angle(truth.yaw_rad + math.pi))
        else:
            candidate = Pose2D(truth.x_m + float(rng.normal(0, 0.025)), truth.y_m + float(rng.normal(0, 0.025)), normalize_angle(truth.yaw_rad + float(rng.normal(0, math.radians(2)))))
        return SimpleNamespace(pose=candidate, rmse_m=0.06, inlier_ratio=0.85)

    def capture(path):
        nonlocal elapsed
        if command != (0, 0):
            raise AssertionError("photo taken while motion was commanded")
        truth_coverage.observe(truth, truth.yaw_rad, config)
        photo_names.append(path.name)
        elapsed += camera_delay_s
        return {"backend": "simulation", "brightness": 100, "contrast": 30}

    def sleep(seconds):
        nonlocal elapsed
        elapsed += seconds

    with tempfile.TemporaryDirectory() as directory:
        args = build_argument_parser().parse_args([
            "--open-space-panorama", "--map", "maps/ble_scene_17.pgm",
            "--map-yaml", "maps/ble_scene_17.yaml", "--max-runtime-min", "25",
            "--lidar-offset-x", str(config.lidar_offset_x_m),
            "--lidar-offset-y", str(config.lidar_offset_y_m), "--stop-distance", "0.42",
            "--panorama-self-return-radius", "0.19" if inject_body_returns else "0",
            "--clearance", str(config.map_clearance_m(.22)),
        ])
        report = InspectionReport(directory, args.map)
        original_add = report.add

        def add(record):
            nonlocal maximum_pose_error
            maximum_pose_error = max(maximum_pose_error, math.hypot(record["x_m"] - truth.x_m, record["y_m"] - truth.y_m))
            yaw_error = normalize_angle(math.radians(record["yaw_deg"]) - truth.yaw_rad)
            direction_errors.append(abs(math.degrees(yaw_error)))
            original_add(record)

        report.add = add
        iterator = scans()
        initial_scan = next(iterator)
        initial_pose = Pose2D(truth.x_m + float(rng.normal(0, 0.025)), truth.y_m + float(rng.normal(0, 0.025)), normalize_angle(truth.yaw_rad + float(rng.normal(0, math.radians(2)))))
        log = io.StringIO()
        with redirect_stdout(log), patch("raspberry_pi.open_space_panorama.time.monotonic", side_effect=lambda: elapsed), patch(
            "raspberry_pi.open_space_panorama.time.sleep", side_effect=sleep
        ), patch("raspberry_pi.open_space_panorama._render_live_map"):
            summary = run_open_space_panorama(
                args=args, known_map=known_map,
                localizer=SimpleNamespace(relative_motion=relative, localize_bounded=localize),
                initial_pose=initial_pose, initial_scan=initial_scan, scan_iterator=iterator,
                client=SimpleNamespace(stop=stop, set_twist=twist, allow_twist_to_expire=lambda: None),
                camera=SimpleNamespace(capture=capture), report=report,
                online_qwen=SimpleNamespace(url=""), should_stop=lambda: steps >= 7000,
            )
        maximum_pose_error = max(maximum_pose_error, math.hypot(summary["final_pose"]["x_m"] - truth.x_m, summary["final_pose"]["y_m"] - truth.y_m))
        motion_events = [json.loads(line) for line in
                         (report.directory / 'panorama_motion.jsonl').read_text(encoding='utf-8').splitlines()]
        checks = {
            "no_collision": collisions == 0,
            "completed_motion": bool(summary.get('motion_complete')) and summary.get('finish_reason') == 'coverage_complete',
            "coverage": truth_coverage.ratio >= config.coverage_target,
            "three_completed_stations": summary["station_count"] >= 3,
            "photo_directions": bool(direction_errors) and max(direction_errors) <= 12,
            "no_overwritten_images": len(photo_names) == len(set(photo_names)),
            "bounded_pose_error": maximum_pose_error <= 0.65,
        }
        return {
            "seed": seed, "passed": all(checks.values()), "checks": checks,
            "finish_reason": summary.get("finish_reason"),
            "station_count": summary["station_count"], "attempt_count": summary["station_attempt_count"],
            "photo_count": len(photo_names), "collisions": collisions,
            "camera_delay_s": camera_delay_s, "map_match_delay_s": map_match_delay_s,
            "body_length_m": config.body_length_m, "body_width_m": config.body_width_m,
            "map_clearance_m": config.map_clearance_m(.22),
            "plant_radius_m": profile.robot_radius_m,
            "lidar_offset_x_m": config.lidar_offset_x_m,
            "lidar_offset_y_m": config.lidar_offset_y_m,
            "reverse_attempt_count": sum(e.get('event') == 'reverse_escape_result' for e in motion_events),
            "reverse_denied_reasons": dict(Counter(e['rear_escape']['reason'] for e in motion_events
                                                    if e.get('event') == 'reverse_escape_denied')),
            'escape_budget_exhaustion_reasons': dict(Counter(e.get('exhaustion_reason') for e in motion_events
                if e.get('event') == 'escape_turn_budget_exhausted')),
            'heading_check_results': dict(Counter(e.get('result') for e in motion_events
                if e.get('event') == 'heading_calibration')),
            "mechanical_body_jam_events": plant.body_jam_events,
            "mechanical_body_jam_releases": plant.body_jam_releases,
            "mechanical_body_jam_active_at_finish": plant.body_jammed,
            "truth_coverage": truth_coverage.ratio, "estimated_coverage": summary["coverage"]["coverage_ratio"],
            "provisional_exploration_coverage": summary['provisional_photo_coverage']['coverage_ratio'],
            "maximum_photo_direction_error_deg": max(direction_errors, default=180),
            "maximum_pose_error_m": maximum_pose_error, "symmetric_candidates": symmetric_candidates,
            "simulated_seconds": elapsed, "scan_steps": steps, "log_tail": log.getvalue().splitlines()[-6:],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--profile", choices=("mild", "realistic", "harsh"), default="realistic")
    parser.add_argument("--output", default="simulation_runs/open_space_panorama_runtime.json")
    parser.add_argument("--inject-body-returns", action="store_true")
    parser.add_argument('--disable-mechanical-jam', action='store_true',
                        help='Diagnostic ablation only; not full disturbance acceptance.')
    parser.add_argument("--camera-delay-seconds", type=float, default=2.1)
    parser.add_argument("--map-match-delay-seconds", type=float, default=.25)
    parser.add_argument("--lidar-offset-x", type=float, default=.14)
    parser.add_argument("--lidar-offset-y", type=float, default=0.)
    args = parser.parse_args()
    paths = (Path("maps/ble_scene_17.pgm"), Path("maps/ble_scene_17.yaml"))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    results = []
    profile = StressProfile.named(args.profile)
    if args.disable_mechanical_jam:
        profile = replace(profile, body_jam_probability_per_forward_step=0.)
    for seed in range(args.seed_count):
        result = simulate(seed, profile, args.inject_body_returns,
                          args.camera_delay_seconds, args.map_match_delay_seconds,
                          args.lidar_offset_x, args.lidar_offset_y)
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    unchanged = all(hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)] for p in paths)
    report = {"production_state_machine": True, "odometry_model": "synthetic_bounded_scan_matching_not_real_ICP", "injected_body_returns": args.inject_body_returns, "profile": args.profile, "map_hashes_unchanged": unchanged, "map_hashes": before, "passed": unchanged and all(r["passed"] for r in results), "results": results}
    report['mechanical_jam_disabled_for_diagnosis'] = args.disable_mechanical_jam
    report['full_disturbance_acceptance'] = report['passed'] and not args.disable_mechanical_jam
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
