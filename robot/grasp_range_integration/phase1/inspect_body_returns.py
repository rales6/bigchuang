"""Archive raw stationary N10 frames and mask diagnostics; never connect BLE."""

import argparse
import json
from pathlib import Path

import numpy as np
from raspberry_pi.config import LidarConfig
from raspberry_pi.lidar import N10LidarDriver
from grasp_range_integration.phase1.rotation_guard import assess_rotation, body_mask_components, BODY_MASK_PROFILE


def describe(scan):
    legacy, patch = body_mask_components(scan.angles_rad, scan.distances_m)
    indices = np.flatnonzero(np.isfinite(scan.distances_m) & (scan.distances_m < .30))
    return {"timestamp_s": float(scan.timestamp_s),
            "angles_rad": np.asarray(scan.angles_rad).tolist(),
            "distances_m": [float(v) if np.isfinite(v) else None for v in scan.distances_m],
            "legacy_mask_indices": np.flatnonzero(legacy).tolist(),
            "installation_patch_indices": np.flatnonzero(patch).tolist(),
            "near_points": [{"index": int(i), "bearing_deg": float((np.degrees(scan.angles_rad[i])+180)%360-180),
                             "distance_m": float(scan.distances_m[i]),
                             "classification": "base_body" if legacy[i] else "installation_edge" if patch[i] else "unmasked_near"}
                            for i in indices],
            "rotation_without_map": assess_rotation(scan)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan-count", type=int, default=8)
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--replay", help="Existing RAW recorded scans; no hardware is opened")
    args = parser.parse_args()
    if not 1 <= args.scan_count <= 30:
        parser.error("scan-count must be in 1..30")
    output = Path(args.output)
    if output.exists():
        parser.error("output exists; raw audit data must not be overwritten")
    records = []
    if args.replay:
        from raspberry_pi.lidar.base import LaserScan
        data=json.loads(Path(args.replay).read_text(encoding="utf-8"))
        for frame in data["frames"][:args.scan_count]:
            records.append(describe(LaserScan(np.asarray(frame["raw_angles_rad"],dtype=float),
                                             np.asarray(frame["raw_distances_m"],dtype=float),0.)))
    else:
        lidar = N10LidarDriver(LidarConfig(port=args.lidar_port, baudrate=230400))
        try:
            scans = iter(lidar.scans())
            for _ in range(args.scan_count):
                records.append(describe(next(scans)))
        finally:
            lidar.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump({"body_mask_profile": BODY_MASK_PROFILE, "base_connected": False,
                   "motion_sent": False, "source": args.replay or args.lidar_port,
                   "audit_kind": "recorded_raw_replay" if args.replay else "stationary_lidar_capture",
                   "frames": records}, stream, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"output": str(output), "base_connected": False, "motion_sent": False,
                      "frames": [{"base": len(f["legacy_mask_indices"]),
                                  "edge": len(f["installation_patch_indices"]),
                                  "unmasked_near": [p for p in f["near_points"] if p["classification"]=="unmasked_near"]}
                                 for f in records]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
