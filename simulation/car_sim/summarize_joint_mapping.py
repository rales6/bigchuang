"""Summarize immutable closed-loop reports, including true no-motion intervals."""
import argparse
import json
import math
from pathlib import Path

from raspberry_pi.joint_mapping import atomic_json, angle_distance


def summarize(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    previous = None
    still_since = None
    longest_still = 0.
    for frame in report["trajectory"]:
        if previous:
            stationary = math.dist(previous["truth"][:2], frame["truth"][:2]) < .002 and angle_distance(previous["truth"][2], frame["truth"][2]) < math.radians(.5)
            if stationary:
                still_since = previous["timestamp"] if still_since is None else still_since
                longest_still = max(longest_still, frame["timestamp"] - still_since)
            else:
                still_since = None
        previous = frame
    return {"report": str(path), "seed": report["seed"], "frames": report["frames"],
            "distance_m": report["distance_m"], "max_error_m": report["max_error_m"],
            "max_yaw_error_deg": report["max_yaw_error_deg"], "collision": report["collision"],
            "longest_no_translation_or_rotation_s": longest_still,
            "completed": report["completed_before_step_limit"], "passed": report["passed"],
            "error": report["error"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", default="simulation_runs/joint_mapping_validation_summary.json")
    args = parser.parse_args()
    results = [summarize(path) for path in args.reports]
    payload = {"runs": results, "all_passed": all(r["passed"] and r["longest_no_translation_or_rotation_s"] <= 15 for r in results)}
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
