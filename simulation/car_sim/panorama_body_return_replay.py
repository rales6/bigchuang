"""Replay stationary field scans through real ICP and the capture state machine.

No physical actuators are imported or commanded. This is a stationary
regression, not ground-truth validation of a full moving inspection route.
"""
from contextlib import redirect_stdout
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping.geometry import Pose2D
from raspberry_pi.map_camera_inspection import InspectionReport, build_argument_parser
from raspberry_pi.open_space_panorama import (
    _filter_body_returns, _bounded_motion, directional_clearances,
    forward_swept_clearance, run_open_space_panorama,
)


def replay(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    scans = [LaserScan(np.asarray(r["angles_rad"], dtype=float),
                       np.asarray(r["distances_m"], dtype=float), r["timestamp_s"])
             for r in rows if r["type"] == "scan"]
    known = KnownOccupancyMap.load("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml", clearance_m=0.30)
    # The final displayed pose from the run preceding acquisition, not truth.
    reference_pose = Pose2D(-1.1668, 0.3546, math.radians(-64.02))
    filtered = [_filter_body_returns(s, 0.19, 0.14, 0.0) for s in scans]
    raw_blocked = sum(forward_swept_clearance(s, offset_x_m=0.14) <= 0.72 for s in scans)
    front_safe = [forward_swept_clearance(s, offset_x_m=0.14) > 0.72 for s in filtered]
    turn_safe = [min(v[n]["effective_m"] for n in ("front", "left", "rear", "right")) >= 0.30
                 for v in (directional_clearances(s, known, reference_pose) for s in filtered)]
    front_clear, turn_clear = sum(front_safe), sum(turn_safe)
    # Evaluate bounded STOP/rescan windows, not a permission to drive through
    # the frames with residual external/ambiguous close returns.
    windows = len(scans) - 3
    resumable_front = sum(any(front_safe[i:i + 4]) for i in range(windows))
    resumable_turn = sum(any(turn_safe[i:i + 4]) for i in range(windows))
    localizer = ScanMapLocalizer(known)
    pose = reference_pose
    rejected = 0
    drift = 0.0
    for previous, current in zip(filtered, filtered[1:]):
        pose, _match, accepted = _bounded_motion(localizer, previous, current, pose)
        rejected += int(not accepted)
        drift = max(drift, math.hypot(pose.x_m - reference_pose.x_m, pose.y_m - reference_pose.y_m))
    photos = []
    hardware_commands = []

    def forbidden_twist(*_args, **_kwargs):
        hardware_commands.append(True)
        raise AssertionError("Stationary regression must stop immediately after first photo")

    with tempfile.TemporaryDirectory() as directory:
        args = build_argument_parser().parse_args(["--open-space-panorama", "--panorama-self-return-radius", "0.19"])
        with redirect_stdout(io.StringIO()), patch("raspberry_pi.open_space_panorama._render_live_map"), patch(
            "raspberry_pi.open_space_panorama.time.sleep"
        ):
            summary = run_open_space_panorama(
                args=args, known_map=known, localizer=ScanMapLocalizer(known),
                initial_pose=reference_pose, initial_scan=scans[0], scan_iterator=iter(scans[1:]),
                client=SimpleNamespace(stop=lambda: None, set_twist=forbidden_twist),
                camera=SimpleNamespace(capture=lambda p: (photos.append(p.name), {"backend": "offline"})[1]),
                report=InspectionReport(directory, args.map), online_qwen=SimpleNamespace(url=""),
                should_stop=lambda: bool(photos),
            )
    checks = {
        "old_logic_reproduces_block": raw_blocked / len(scans) >= 0.90,
        "stopped_rescan_front_recovers": resumable_front / windows >= 0.90,
        "stopped_rescan_turn_recovers": resumable_turn / windows >= 0.90,
        "real_icp_available": rejected == 0,
        "stationary_pose_drift_bounded": drift <= 0.15,
        "production_first_photo": summary["photo_count"] == 1,
        "no_actuator_commands": not hardware_commands,
    }
    return {
        "passed": all(checks.values()), "checks": checks, "scan_count": len(scans),
        "source": str(path), "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "old_front_blocked_frames": raw_blocked, "filtered_front_clear_frames": front_clear,
        "filtered_turn_clear_frames": turn_clear, "real_icp_rejected_frames": rejected,
        "stopped_rescan_window_count": windows,
        "resumable_front_windows": resumable_front, "resumable_turn_windows": resumable_turn,
        "maximum_stationary_position_drift_m": drift,
        "masked_points_range": [min(s.body_return_count for s in filtered), max(s.body_return_count for s in filtered)],
        "production_photo_count": summary["photo_count"],
        "limitation": "stationary field replay only; no moving trajectory truth or real camera inference",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scans")
    parser.add_argument("--output", default="simulation_runs/panorama_body_return_field_replay.json")
    args = parser.parse_args()
    result = replay(args.scans)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True), flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
