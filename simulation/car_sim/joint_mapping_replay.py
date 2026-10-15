"""Real SLAM sensor replay on the user's PGM; NOT a full navigation proof.

Truth drives a prescribed route above the central table, and only generates
laser rays/evaluation. It is never attached to scans passed to LidarSlam.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from raspberry_pi.config import MappingConfig
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.lidar import LaserScan
from raspberry_pi.mapping import LidarSlam, Pose2D, filter_scan_sector
from raspberry_pi.joint_mapping import atomic_json, angle_distance


def route():
    start = Pose2D(-1.0, .65, -math.pi / 2)
    yield from [(start, True)] * 10
    for y in np.linspace(.65, 0, 34)[1:]:
        yield Pose2D(-1, float(y), -math.pi / 2), False
    yield from [(Pose2D(-1, 0, -math.pi / 2), True)] * 10
    for yaw in np.linspace(-math.pi / 2, 0, 46)[1:]:
        yield Pose2D(-1, 0, float(yaw)), False
    yield from [(Pose2D(-1, 0, 0), True)] * 10
    for end in (-.4, .2, .8):
        for x in np.linspace(end - .6, end, 31)[1:]:
            yield Pose2D(float(x), 0, 0), False
        yield from [(Pose2D(end, 0, 0), True)] * 10


def raycast(known, pose, angles):
    ranges = np.arange(.12, 8.001, .015)
    headings = angles + pose.yaw_rad
    xs = pose.x_m + np.cos(headings)[:, None] * ranges
    ys = pose.y_m + np.sin(headings)[:, None] * ranges
    cols = np.floor((xs - known.origin_x_m) / known.resolution_m).astype(int)
    rows = np.floor((ys - known.origin_y_m) / known.resolution_m).astype(int)
    valid = (cols >= 0) & (cols < known.width) & (rows >= 0) & (rows < known.height)
    hit = valid & known.occupied[np.clip(rows, 0, known.height - 1), np.clip(cols, 0, known.width - 1)]
    return np.where(hit.any(axis=1), ranges[hit.argmax(axis=1)], 8.)


def replay(map_path, seed, noise=.008, dropout=.02):
    known = KnownOccupancyMap.load(map_path, clearance_m=.2)
    slam = LidarSlam(MappingConfig(manhattan_enabled=False, mapping_max_distance_m=4.))
    rng = np.random.default_rng(seed)
    trajectory = list(route())
    origin = trajectory[0][0]
    errors, yaw_errors, rows = [], [], []
    rejected = 0
    for i, (truth, stationary) in enumerate(trajectory):
        col, row = known.world_to_grid(((truth.x_m, truth.y_m),))[0]
        if not known.traversable[row, col]:
            raise ValueError("prescribed route violates 0.2 m PGM clearance at {}".format(truth))
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        distances = raycast(known, truth, angles) + rng.normal(0, noise, len(angles))
        distances[rng.random(len(angles)) < dropout] = np.nan
        scan = LaserScan(angles, distances, i * .2 + 1)
        update = slam.process(scan, mapping_scan=filter_scan_sector(scan, 0, 160), stationary_mapping=stationary)
        expected = origin.inverse().compose(truth)
        error = math.hypot(update.pose.x_m - expected.x_m, update.pose.y_m - expected.y_m)
        yaw_error = math.degrees(angle_distance(update.pose.yaw_rad, expected.yaw_rad))
        errors.append(error)
        yaw_errors.append(yaw_error)
        rejected += not update.accepted
        rows.append({"truth_local": [expected.x_m, expected.y_m, expected.yaw_rad],
                     "estimated": [update.pose.x_m, update.pose.y_m, update.pose.yaw_rad],
                     "accepted": update.accepted})
    return {"seed": seed, "frames": len(rows), "rejected": rejected,
            "max_position_error_m": max(errors), "final_position_error_m": errors[-1],
            "max_yaw_error_deg": max(yaw_errors), "fusions": slam.map_integrated_count,
            "passed": max(errors) < .2 and max(yaw_errors) < 10 and rejected < len(rows) * .05,
            "trajectory": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="maps/ble_scene_17.pgm")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", default="simulation_runs/joint_mapping_replay.json")
    args = parser.parse_args()
    runs = []
    for seed in range(args.seeds):
        result = replay(args.map, seed)
        runs.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "trajectory"}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, {"kind": "production_slam_prescribed_route_sensor_replay",
                         "limitations": "Not exact field replay, no wheel/controller/rolling-scan/glass model; initial position inferred approximately from user description. Old PGM only generates sensor truth, never initializes SLAM.",
                         "runs": runs, "passed": all(r["passed"] for r in runs)})
    if not all(r["passed"] for r in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
