"""Safe local entry: inspect paths, check the repository, run offline suites.

No subcommand deploys software or opens cameras/motors. Runtime entry points
are documented by feature instead of being launched by a maintenance command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "mapping": "robot/raspberry_pi/mapping",
    "annotation": "robot/tools/build_relationship_semantic_map.py",
    "navigation": "robot/grasp_range_integration/phase1",
    "perception": "services/perception",
    "pursuit": "services/track/pursuit",
    "grasp": "services/grasp",
    "web": "services/track/final_work",
    "android": "apps/android",
    "firmware": "firmware/car",
    "simulation": "simulation/car_sim",
    "map": "maps/current",
}
SUITES = {
    "regression": ["grasp_range_integration.tests." + name for name in (
        "test_grasp_pose_correction", "test_robot_skills", "test_grasp_hull_recheck",
        "test_pursuit_loss_recovery", "test_track_web_phase1_pick", "test_task_video",
        "test_carry_localization", "test_phase1_startup_route_recovery",
        "test_phase1_adaptive_turn", "test_phase1_midrun_recheck", "test_phase1_midrun_view_change",
        "test_phase1_startup_active_recovery", "test_phase1_fresh_motion_feedback",
        "test_phase1_startup_drive")],
    "firmware": ["test_" + name for name in (
        "actuator_bus", "application_integration", "drive_and_indicators", "legacy_hardware",
        "protocol_v2")],
    "simulation": ["test_" + name for name in (
        "headless_autonomous", "mapping_report", "semantic_inspection_stress", "post_mapping_semantic")],
    "annotation": ["test_" + name for name in (
        "semantic_release", "relationship_semantic_map", "semantic_overview_render", "qwen_candidate_map")],
}


def environment():
    paths = [ROOT, ROOT/"robot", ROOT/"firmware", ROOT/"firmware/car",
             ROOT/"simulation", ROOT/"robot/tests", ROOT/"services/perception/src"]
    temporary = ROOT / ".local" / "test-tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    return {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths)),
            "TEMP": str(temporary), "TMP": str(temporary), "TMPDIR": str(temporary),
            "PYTHONIOENCODING": "utf-8", "FINAL_WORK_ENABLE_HARDWARE": "0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    paths = sub.add_parser("paths", help="Show functional source locations")
    paths.add_argument("feature", nargs="?", choices=sorted(PATHS))
    sub.add_parser("check", help="Check layout, syntax, active-map hashes and credential hygiene")
    test = sub.add_parser("test", help="Run explicitly offline test suites")
    test.add_argument("--suite", choices=["smoke", "all-offline", *SUITES], default="smoke")
    args = parser.parse_args()
    if args.command == "paths":
        selected = {args.feature: PATHS[args.feature]} if args.feature else PATHS
        print(json.dumps({k: str(ROOT/v) for k,v in selected.items()}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "check":
        return subprocess.call([sys.executable, str(ROOT/"tools/maintenance/check_repository.py")],
                               cwd=ROOT, env=environment())
    suites = ["smoke", *SUITES] if args.suite == "all-offline" else [args.suite]
    for suite in suites:
        if suite == "smoke":
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
        else:
            names = SUITES[suite]
            if suite in {"firmware", "simulation", "annotation"}:
                missing = [name for name in names if not (ROOT/"robot/tests"/(name+".py")).is_file()]
                if missing:
                    print("Suite definition needs correction: " + repr(missing), file=sys.stderr)
                    return 2
            command = [sys.executable, "-m", "unittest", *names]
        print("Offline suite: " + suite, flush=True)
        result = subprocess.call(command, cwd=ROOT, env=environment())
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
