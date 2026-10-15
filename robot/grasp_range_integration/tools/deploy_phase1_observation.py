"""Deploy only the phase-one adapter; never start hardware or alter core code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import time

import paramiko


ROOT = Path(__file__).resolve().parents[2]
CAR = "/home/dachuang/dachuang/car1.0"
RELEASE = CAR + "/ble_scene_17_relationship_20260919"
PYTHON = CAR + "/.venv-pi/bin/python3"
PI_HOST_KEY_SHA256 = "1c624d9d6c33aa0482a4f043a4b17b295b9b64dd232c3e3fadce4d6436594ee6"
FILES = [
    "grasp_range_integration/docs/body_occlusion_20261003.md",
    "grasp_range_integration/tests/fixtures/body_occlusion_20261003.json",
    "grasp_range_integration/phase1/run_phase1_find_only.sh",
    "grasp_range_integration/phase1/rotation_guard.py",
    "grasp_range_integration/phase1/temporary_obstacle_evidence.py",
    "grasp_range_integration/phase1/navigation_avoidance.py",
    "grasp_range_integration/phase1/navigation_freshness.py",
    "grasp_range_integration/tests/fixtures/navigation_start_20260926_193306.jsonl",
    "grasp_range_integration/tests/fixtures/midrun_stop_20260927_214129.json",
    "grasp_range_integration/tests/fixtures/locked_view_20261001_194340.json",
    "grasp_range_integration/tests/fixtures/locked_view_20261001_210758.json",
    "grasp_range_integration/tests/fixtures/dispatch_glass_20261001_204249.json",
    "grasp_range_integration/phase1/motion_interlock.py",
    "grasp_range_integration/phase1/motion_evidence.py",
    "grasp_range_integration/phase1/red_target_deepseek.py",
    "grasp_range_integration/phase1/colour_target_review.py",
    "grasp_range_integration/phase1/pursuit_entry_contract.py",
    "grasp_range_integration/phase1/verified_search.py",
    "grasp_range_integration/phase1/ble_startup_recovery.py",
    "grasp_range_integration/phase1/plan_phase1_deepseek.sh",
    "grasp_range_integration/phase1/prepare_phase1.py",
    "grasp_range_integration/phase1/glass_boundary_guard.py",
    "grasp_range_integration/phase1/glass_boundaries.json",
    "grasp_range_integration/tools/render_glass_boundary.py",
    "grasp_range_integration/debug_reports/glass_boundary_left_side.png",
    "grasp_range_integration/tests/fixtures/rotation_desk_20260926_180247.json",
    "grasp_range_integration/phase1/inspect_body_returns.py",
    "grasp_range_integration/phase1/startup_localization.py",
    "grasp_range_integration/phase1/startup_active_recovery.py",
    "grasp_range_integration/phase1/startup_route_recovery.py",
    "grasp_range_integration/phase1/startup_exploration.py",
    "grasp_range_integration/phase1/startup_drive.py",
    "grasp_range_integration/phase1/startup_drive_experience.py",
    "grasp_range_integration/phase1/startup_prior.py",
    "grasp_range_integration/phase1/fresh_motion_feedback.py",
    "grasp_range_integration/phase1/startup_visual_review.py",
    "grasp_range_integration/tests/fixtures/startup_stationary_20260926.json",
    "grasp_range_integration/tests/fixtures/body_mask_audit_20260926.json",
    "grasp_range_integration/tests/fixtures/body_mask_stationary_20260926_150426.json",
    "grasp_range_integration/tests/fixtures/body_confirmed_stop_20260927_162135.json",
    "grasp_range_integration/tests/fixtures/rotation_edge_20261001_173725.json",
    "grasp_range_integration/phase1/observation_recovery.py",
    "grasp_range_integration/phase1/turn_policy.py",
    "grasp_range_integration/phase1/adaptive_turn.py",
    "grasp_range_integration/phase1/turn_start_recovery.py",
    "grasp_range_integration/phase1/turn_output_boost.py",
    "grasp_range_integration/phase1/turn_calibration.py",
    "grasp_range_integration/phase1/motion_experience.py",
    "grasp_range_integration/phase1/arrival_visibility.py",
    "grasp_range_integration/phase1/localization_performance.py",
    "grasp_range_integration/phase1/localization_handoff.py",
    "grasp_range_integration/phase1/turn_range_feedback.py",
    "grasp_range_integration/phase1/turn_feedback_recovery.py",
    "grasp_range_integration/phase1/visibility_route_planner.py",
    "grasp_range_integration/phase1/relationship_find_only_entry.py",
    "grasp_range_integration/phase1/deepseek_planner.py",
    "grasp_range_integration/phase1/probe_bounded_turn.py",
    "grasp_range_integration/phase1/probe_live_rotation.py",
    "grasp_range_integration/tools/plot_rotation_evidence.py",
    "grasp_range_integration/tests/fixtures/rotation_probe_20260926_170547.json",
    "grasp_range_integration/tests/fixtures/rotation_probe_20260926_170942.json",
    "grasp_range_integration/tests/fixtures/rotation_probe_20260926_172012.json",
    "grasp_range_integration/debug_reports/rotation_20260926_172012.png",
    "grasp_range_integration/README.md",
    "grasp_range_integration/CODE_PATHS_PI_3090.txt",
    "grasp_range_integration/track_web_integration/pi/map_navigation_adapter.py",
    "grasp_range_integration/track_web_integration/pi/robot_bridge.py",
    "grasp_range_integration/track_web_integration/gpu/plan_adaptation.py",
    "grasp_range_integration/PHASE1_OBSERVATION_RECOVERY_PLAN_20260925.md",
] + [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "grasp_range_integration/tests").glob("test_*.py"))]
PROTECTED = ["raspberry_pi/object_search_mission.py", "raspberry_pi/map_camera_inspection.py",
             "raspberry_pi/known_map_navigation.py", "raspberry_pi/esp32/client.py",
             "raspberry_pi/mapping/adaptive_speed.py", "raspberry_pi/mapping/scan_matcher.py",
             "ble_scene_17_relationship_20260919/deepseek_visual_localization.py",
             "ble_scene_17_relationship_20260919/relationship_localization_guard.py"]


def remote(client, command, timeout=60):
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output, errors = stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"remote command failed ({code}): {output}\n{errors}")
    return output + errors


def mkdir(sftp, path):
    current = ""
    for part in PurePosixPath(path).parts[1:]:
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def read_optional(sftp, path):
    try:
        with sftp.open(path, "rb") as stream:
            return stream.read()
    except FileNotFoundError:
        return None


def protected_hashes(sftp):
    return {name: hashlib.sha256(read_optional(sftp, CAR + "/" + name) or b"").hexdigest() for name in PROTECTED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Deploy and verify file hashes only; do not run remote tests or compile checks")
    parser.add_argument("--test-module", action="append",
                        help="Run only these software regression modules instead of full discovery")
    parser.add_argument("--host", default="dachuang.local",
                        help="Explicit Pi hostname/IP when mDNS is unavailable")
    parser.add_argument("--only", action="append", choices=FILES,
                        help="Deploy only this allowlisted release file; may be repeated")
    args = parser.parse_args()
    selected_files = list(dict.fromkeys(args.only or FILES))
    password = os.environ.get("PI_SSH_PASSWORD")
    if not password:
        raise RuntimeError("Set PI_SSH_PASSWORD in the environment; it is never saved or printed")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username="dachuang", password=password,
                   look_for_keys=False, allow_agent=False, timeout=15, auth_timeout=15, banner_timeout=15)
    try:
        actual_host_key = hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()
        if actual_host_key != PI_HOST_KEY_SHA256:
            raise RuntimeError("Pi SSH host key mismatch; refusing deployment")
        remote(client, f"test -d {shlex.quote(RELEASE + '/grasp_range_integration/phase1')} && test -x {shlex.quote(PYTHON)}")
        processes = remote(client, "ps -eo pid,args")
        active = [line for line in processes.splitlines() if any(name in line for name in
                  ("relationship_find_only_entry.py", "object_search_mission.py", "probe_bounded_turn",
                   "probe_live_rotation"))]
        print(json.dumps({"release": RELEASE, "active_missions": active,
                          "host_key_sha256": actual_host_key}, indent=2))
        if not args.deploy:
            return
        if active:
            raise RuntimeError("Active mission detected; deployment refused without stopping it")
        with client.open_sftp() as sftp:
            if sftp.normalize(RELEASE) != RELEASE:
                raise RuntimeError("Release resolves to another directory; inspect before deployment")
            before = protected_hashes(sftp)
            backup = RELEASE + "/grasp_range_integration/deployment_backups/observation_" + time.strftime("%Y%m%d_%H%M%S")
            mkdir(sftp, backup)
            old, hashes = {}, {}
            for name in selected_files:
                if not name.startswith("grasp_range_integration/"):
                    raise RuntimeError("Refusing to change a protected path")
                data = (ROOT / name).read_bytes()
                previous = read_optional(sftp, RELEASE + "/" + name)
                old[name] = previous
                if previous is not None:
                    saved = backup + "/original/" + name
                    mkdir(sftp, str(PurePosixPath(saved).parent))
                    with sftp.open(saved, "wb") as stream:
                        stream.write(previous)
                staged = backup + "/staged/" + name
                mkdir(sftp, str(PurePosixPath(staged).parent))
                with sftp.open(staged, "wb") as stream:
                    stream.write(data)
                hashes[name] = hashlib.sha256(data).hexdigest()
                if hashlib.sha256(read_optional(sftp, staged)).hexdigest() != hashes[name]:
                    raise RuntimeError("Staging hash mismatch: " + name)
            activated = []
            try:
                for name in selected_files:
                    target = RELEASE + "/" + name
                    mkdir(sftp, str(PurePosixPath(target).parent))
                    sftp.posix_rename(backup + "/staged/" + name, target)
                    activated.append(name)
                if not args.skip_tests:
                    environment = "PYTHONPATH=" + shlex.quote(RELEASE + ":" + CAR)
                    test_args = (" ".join(shlex.quote(name) for name in args.test_module) + " -q"
                                 if args.test_module else "discover -s grasp_range_integration/tests -q")
                    command = f"cd {shlex.quote(RELEASE)} && {environment} {shlex.quote(PYTHON)} -m unittest {test_args}"
                    print(remote(client, command, timeout=120))
                    command = f"cd {shlex.quote(RELEASE)} && {environment} {shlex.quote(PYTHON)} -m compileall -q grasp_range_integration/phase1"
                    remote(client, command)
                    remote(client, "bash -n " + shlex.quote(RELEASE + "/grasp_range_integration/phase1/run_phase1_find_only.sh"))
                for name, digest in hashes.items():
                    if hashlib.sha256(read_optional(sftp, RELEASE + "/" + name)).hexdigest() != digest:
                        raise RuntimeError("Deployed hash mismatch: " + name)
                if protected_hashes(sftp) != before:
                    raise RuntimeError("Protected source hash changed")
            except Exception:
                # Keep the failed version recoverable too; restore only this
                # deployment's exact file targets, never delete a directory.
                for name in reversed(activated):
                    target = RELEASE + "/" + name
                    failed = backup + "/failed/" + name
                    mkdir(sftp, str(PurePosixPath(failed).parent))
                    sftp.posix_rename(target, failed)
                    if old[name] is not None:
                        sftp.posix_rename(backup + "/original/" + name, target)
                raise
            record = {"deployed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "release": RELEASE,
                      "backup": backup, "sha256": hashes, "protected_sha256": before,
                      "hardware_started": False, "tests_run": not args.skip_tests,
                      "tests_passed": None if args.skip_tests else True}
            for name in (backup + "/DEPLOYMENT_RECORD.json", RELEASE + "/grasp_range_integration/DEPLOYMENT_RECORD.json"):
                with sftp.open(name, "wb") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))
            print(json.dumps(record, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
