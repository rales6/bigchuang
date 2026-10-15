"""Accelerated end-to-end localization stress test for production LidarSlam.

The synthetic vehicle follows one complete non-holonomic loop through a
rectangular room.  Every generated 360-degree scan is passed to the real scan
filter, adjacent/local/global ICP selection and physical-motion rejection
gates.  No wall-clock sleeps or browser are involved.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from raspberry_pi.config import LidarConfig, MappingConfig
from raspberry_pi.mapping import LidarSlam, Pose2D
from raspberry_pi.mapping.geometry import normalize_angle

from .headless_autonomous import Rectangle, _scan


def _loop_pose(index, scans):
    phase = math.pi + math.tau * float(index) / float(scans)
    x_m = 2.6 + 1.55 * math.cos(phase)
    y_m = 2.1 + 1.05 * math.sin(phase)
    # Tangent heading makes the trajectory physically plausible for a
    # differential-drive vehicle rather than teleporting it sideways.
    yaw_rad = math.atan2(
        1.05 * math.cos(phase),
        -1.55 * math.sin(phase),
    )
    return Pose2D(x_m, y_m, yaw_rad)


def run_localization_stress(scans=240, dt_s=0.20, seed=7, output=None):
    scans = max(40, int(scans))
    dt_s = max(0.05, float(dt_s))
    room = (5.2, 4.2)
    obstacles = (Rectangle(2.25, 1.72, 0.70, 0.76),)
    rng = np.random.default_rng(seed)
    slam = LidarSlam(
        MappingConfig(
            min_match_points=35,
            manhattan_enabled=True,
            global_match_every_updates=3,
        ),
        LidarConfig(max_distance_m=6.0),
    )

    first_truth = _loop_pose(0, scans)
    origin_inverse = first_truth.inverse()
    accepted = 0
    rejected = 0
    rejection_counts = Counter()
    rejection_streak = 0
    maximum_rejection_streak = 0
    position_errors = []
    yaw_errors = []
    maximum_truth_step_m = 0.0
    previous_truth = None

    for index in range(scans + 1):
        truth = _loop_pose(index, scans)
        if previous_truth is not None:
            maximum_truth_step_m = max(
                maximum_truth_step_m,
                math.hypot(
                    truth.x_m - previous_truth.x_m,
                    truth.y_m - previous_truth.y_m,
                ),
            )
        previous_truth = truth
        scan = _scan(
            truth,
            index * dt_s,
            room,
            obstacles,
            ray_count=300,
            noise_m=0.005,
            dropout=0.02,
            rng=rng,
        )
        update = slam.process(scan, stationary_mapping=False)
        expected = origin_inverse.compose(truth)
        if update.accepted:
            accepted += 1
            rejection_streak = 0
            position_errors.append(math.hypot(
                update.pose.x_m - expected.x_m,
                update.pose.y_m - expected.y_m,
            ))
            yaw_errors.append(abs(normalize_angle(
                update.pose.yaw_rad - expected.yaw_rad
            )))
        else:
            rejected += 1
            rejection_streak += 1
            maximum_rejection_streak = max(
                maximum_rejection_streak,
                rejection_streak,
            )
            rejection_counts[update.rejection_reason or "unknown"] += 1

    result = {
        "seed": int(seed),
        "scans": scans + 1,
        "simulated_s": scans * dt_s,
        "maximum_truth_step_m": maximum_truth_step_m,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_ratio": accepted / float(scans + 1),
        "maximum_consecutive_rejections": maximum_rejection_streak,
        "rejection_reason_counts": dict(rejection_counts),
        "position_rmse_m": float(math.sqrt(np.mean(
            np.square(position_errors)
        ))),
        "position_p95_m": float(np.percentile(position_errors, 95)),
        "yaw_rmse_deg": math.degrees(float(math.sqrt(np.mean(
            np.square(yaw_errors)
        )))),
        "yaw_p95_deg": math.degrees(float(np.percentile(yaw_errors, 95))),
    }
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="accelerated production LidarSlam localization stress test"
    )
    parser.add_argument("--scans", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="car_sim/output/localization_stress.json",
    )
    return parser


def main():
    args = build_parser().parse_args()
    result = run_localization_stress(
        scans=args.scans,
        dt_s=args.dt,
        seed=args.seed,
        output=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
