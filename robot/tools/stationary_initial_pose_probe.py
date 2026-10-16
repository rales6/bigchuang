#!/usr/bin/env python3
"""Read-only lidar + stereo-camera initial-pose probe for the release map.

This program never creates an ESP32 client and therefore cannot send chassis
or arm commands.  It refuses to run while a known motion mission owns the
lidar, captures three stationary scans plus the two current camera frames,
and lets DeepSeek select only among metric pose candidates produced by lidar.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from raspberry_pi.config import LidarConfig
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.mapping import Pose2D
from raspberry_pi.object_search_mission import MjpegCameraCapture

from deepseek_visual_localization import (
    DeepSeekVisionClient,
    candidate_scene_descriptions,
    load_config,
)
from relationship_localization_guard import (
    _constrained_start_localize,
    parse_start_pose_prior,
)


MOTION_MODULES = (
    "autonomous_mapping",
    "joint_mapping",
    "map_camera_inspection",
    "object_search_mission",
    "motion_calibration",
    "open_space_panorama",
)


def _pose_row(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    ambiguity_margin = float(result.ambiguity_margin)
    return {
        "x_m": round(float(result.pose.x_m), 4),
        "y_m": round(float(result.pose.y_m), 4),
        "yaw_deg": round(math.degrees(float(result.pose.yaw_rad)), 2),
        "rmse_m": round(float(result.rmse_m), 4),
        "inlier_ratio": round(float(result.inlier_ratio), 4),
        "score": round(float(result.score), 4),
        "ambiguity_margin": (
            round(ambiguity_margin, 4) if math.isfinite(ambiguity_margin) else None
        ),
        "reliable": bool(result.reliable),
    }


def _angle_error(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _motion_preflight(lidar_port: str) -> dict[str, Any]:
    processes = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    active = [
        line.strip()
        for line in processes.splitlines()
        if any("-m raspberry_pi." + name in line for name in MOTION_MODULES)
    ]
    if active:
        raise RuntimeError("motion task active; refusing read-only probe: " + repr(active))
    holders = subprocess.run(
        ["fuser", lidar_port], capture_output=True, text=True, check=False
    )
    if holders.returncode != 1:
        raise RuntimeError(
            "lidar is busy or holder check failed: "
            + holders.stdout.strip()
            + holders.stderr.strip()
        )
    return {
        "motion_tasks": [],
        "lidar_holders": [],
        "esp32_client_created": False,
        "hardware_commands_sent": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    release_default = Path(__file__).resolve().parent
    parser.add_argument("--release", type=Path, default=release_default)
    parser.add_argument("--config", type=Path, default=Path(
        os.environ.get(
            "DEEPSEEK_VISION_CONFIG",
            "/home/robot/dachuang/car1.0/secrets/deepseek_vision.json",
        )
    ))
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--camera-url", default="http://127.0.0.1:8080")
    parser.add_argument("--scan-count", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    release = args.release.resolve()
    output = release / "stationary_pose_runs" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    lock = (release / ".stationary_initial_pose_probe.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    safety = _motion_preflight(args.lidar_port)

    def timeout_handler(_signum, _frame):
        raise TimeoutError("stationary initial-pose probe exceeded timeout")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max(30, int(args.timeout)))
    report: dict[str, Any] = {
        "schema": "stationary_initial_pose_probe.v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "safety": safety,
        "release": str(release),
        "output": str(output),
    }
    lidar = None
    try:
        known_map = KnownOccupancyMap.load(
            release / "ble_scene_17.pgm",
            release / "ble_scene_17.yaml",
            clearance_m=0.30,
        )
        localizer = ScanMapLocalizer(known_map, lidar_offset_x_m=0.14)
        prior = parse_start_pose_prior(os.environ.get("SEARCH_START_POSE_HINT"))
        report["start_pose_prior"] = None if prior is None else {
            "x_m": prior.x_m,
            "y_m": prior.y_m,
            "yaw_deg": prior.yaw_deg,
            "radius_m": prior.radius_m,
            "yaw_tolerance_deg": prior.yaw_tolerance_deg,
        }

        camera_path = output / "current_left.jpg"
        camera_info = MjpegCameraCapture(args.camera_url, timeout_s=15.0).capture(camera_path)
        right_path = camera_path.with_name(str(camera_info["camera_2_image"]))
        report["camera"] = {
            **camera_info,
            "left": str(camera_path),
            "right": str(right_path),
        }

        lidar = N10LidarDriver(LidarConfig(port=args.lidar_port))
        iterator = iter(lidar.scans())
        scans = [next(iterator) for _ in range(max(1, min(5, args.scan_count)))]
        prior_results = []
        if prior is not None:
            for scan in scans:
                prior_results.append(
                    _constrained_start_localize(localizer, scan, prior, Pose2D)
                )
        report["prior_lidar_results"] = [_pose_row(item) for item in prior_results]

        valid_prior = [item for item in prior_results if item is not None and item.reliable]
        prior_consistent = False
        if len(valid_prior) == len(scans) and valid_prior:
            anchor = valid_prior[0].pose
            prior_consistent = all(
                math.hypot(item.pose.x_m - anchor.x_m, item.pose.y_m - anchor.y_m) <= 0.25
                and _angle_error(item.pose.yaw_rad, anchor.yaw_rad) <= math.radians(15.0)
                for item in valid_prior[1:]
            )
        report["prior_lidar_consistent"] = prior_consistent

        selected_scan = scans[-1]
        radar_candidates = localizer.global_localization_candidates(selected_scan, maximum=6)
        best_prior = min(valid_prior, key=lambda item: item.rmse_m) if valid_prior else None
        if best_prior is not None and not any(
            math.hypot(item.pose.x_m - best_prior.pose.x_m, item.pose.y_m - best_prior.pose.y_m) < 0.25
            and _angle_error(item.pose.yaw_rad, best_prior.pose.yaw_rad) < math.radians(15.0)
            for item in radar_candidates
        ):
            radar_candidates.insert(0, best_prior)
        radar_candidates = radar_candidates[:6]
        report["radar_candidates"] = [
            {"candidate_id": f"C{index:02d}", **(_pose_row(item) or {})}
            for index, item in enumerate(radar_candidates, start=1)
        ]

        config = load_config(args.config, module_directory=release)
        report["deepseek"] = {
            "configured": config.configured,
            "model": config.model,
            "config_source": str(config.source_path),
            "api_key_logged": False,
        }
        descriptions = candidate_scene_descriptions(
            radar_candidates,
            json.loads(config.semantic_map.read_text(encoding="utf-8")),
            config,
        )
        report["candidate_descriptions"] = descriptions
        if config.configured and radar_candidates:
            ranked = DeepSeekVisionClient(config).rank(
                camera_path, right_path, descriptions
            )
            report["deepseek"]["result"] = ranked.to_dict()
            reliable = bool(
                ranked.selected_candidate_id != "uncertain"
                and ranked.confidence >= config.minimum_confidence
                and ranked.margin >= config.minimum_margin
                and ranked.consistent_repeats == config.repeat_count
            )
            report["deepseek"]["reliable"] = reliable
            if reliable:
                index = int(ranked.selected_candidate_id[1:]) - 1
                winner = radar_candidates[index]
                report["final"] = {
                    "status": "lidar_pose_confirmed_by_stereo_deepseek",
                    "candidate_id": ranked.selected_candidate_id,
                    "pose": _pose_row(winner),
                }
            else:
                report["final"] = {
                    "status": "visual_uncertain",
                    "pose": _pose_row(best_prior) if prior_consistent else None,
                }
        else:
            report["final"] = {
                "status": "deepseek_not_configured",
                "pose": _pose_row(best_prior) if prior_consistent else None,
            }
    except Exception as exc:
        report["final"] = {"status": "probe_failed", "error": str(exc)}
        raise
    finally:
        signal.alarm(0)
        if lidar is not None:
            lidar.close()
        report["safety"]["hardware_commands_sent"] = False
        (output / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "output": str(output),
            "final": report.get("final"),
            "hardware_commands_sent": False,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
