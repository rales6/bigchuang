"""Atomically deploy the confirmed-start inspection route fix to the Pi.

The script never starts motors, lidar, or cameras.  It refuses deployment
while an inspection/mapping task is active, keeps timestamped backups, syntax
checks staged files, verifies hashes, and runs a read-only route calculation
on the Pi after activation.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shlex
import time

import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/dachuang/dachuang/car1.0"
FILES = (
    "raspberry_pi/known_map_navigation.py",
    "raspberry_pi/map_camera_inspection.py",
    "raspberry_pi/open_space_panorama.py",
    "raspberry_pi/lidar/n10_driver.py",
    "raspberry_pi/esp32/client.py",
    "inspect_map.sh",
)


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def command(client, value, timeout=90):
    _stdin, stdout, stderr = client.exec_command(value, timeout=timeout)
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(error or output or "remote command failed")
    return output


def remote_route_check(client):
    code = """
import json, math
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from types import SimpleNamespace
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.map_camera_inspection import (
    _final_heading_timeout_seconds,
    _information_view_targets,
    _inspection_path_for_target,
    _navigation_command_ttl,
    _spaced_targets,
    build_argument_parser,
    _panorama_result_message,
)
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.lidar.n10_driver import N10LidarDriver
from raspberry_pi.esp32 import Esp32Client
from raspberry_pi.open_space_panorama import (
    PanoramaConfig, _bounded_motion, _filter_body_returns, forward_swept_clearance,
    _MotionAuditClient, MINIMUM_MOTION_TTL_MS,
)
panorama_args = build_argument_parser().parse_args(['--open-space-panorama'])
config = PanoramaConfig.from_args(panorama_args)
config.validate()
assert abs(config.swept_half_width_m - .16) < 1e-9
assert abs(config.map_clearance_m(.01) - .22) < 1e-9
assert MINIMUM_MOTION_TTL_MS == 100
with tempfile.TemporaryDirectory() as temporary, patch(
        'raspberry_pi.open_space_panorama.time.monotonic', return_value=10.):
    sent = []
    audit = _MotionAuditClient(SimpleNamespace(stop=lambda: None,
        set_bounded_twist=lambda *a, **kw: sent.append(kw)), config, Path(temporary))
    audit.scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), 9.546127428)
    assert not audit.set_twist(420, 342, 500) and not sent
    assert audit.current_reason == 'scan_budget_too_short'
    audit.scan = LaserScan(audit.scan.angles_rad, audit.scan.distances_m, 9.95)
    assert audit.set_twist(420, 342, 500)
    assert 100 <= sent[-1]['ttl_ms'] <= 450
assert hasattr(N10LidarDriver, 'set_scan_watchdog')
assert hasattr(Esp32Client, 'set_bounded_twist')
assert hasattr(ScanMapLocalizer, 'localize_bounded')
assert .4 <= panorama_args.panorama_startup_window_seconds <= 1.
assert panorama_args.panorama_progress_window_seconds == 1.8
assert panorama_args.open_space_panorama
body_args = build_argument_parser().parse_args(['--open-space-panorama',
    '--panorama-self-return-radius', '.19', '--panorama-self-return-half-width', '.125'])
PanoramaConfig.from_args(body_args).validate()
raw_body_scan = LaserScan(np.radians([180., -104., -90., 80., -90.]),
    np.array([.082, .121, .121, .10, .25]), 0.)
filtered = _filter_body_returns(raw_body_scan, .19, .14, 0., .125)
assert filtered.body_return_count == 3
assert np.isnan(filtered.distances_m[:3]).all()
assert np.isfinite(raw_body_scan.distances_m).all()
assert np.array_equal(filtered.distances_m[3:], raw_body_scan.distances_m[3:])
# Confirmed installed-sensor mask is deliberately independent of collision
# dimensions; clipping it to approximate dimensions caused the field stop.
physical_filtered = _filter_body_returns(raw_body_scan, .19, .14, 0., .125)
assert physical_filtered.body_return_count == 3
assert np.isfinite(physical_filtered.distances_m[3:]).all()
assert forward_swept_clearance(filtered, offset_x_m=.14) < .30
occluded = _filter_body_returns(LaserScan(np.radians([180., -104.]), np.array([.082, .121]), 0.), .19, .14)
assert forward_swept_clearance(occluded, offset_x_m=.14, reverse=True) == 0.
motion = MatchResult(Pose2D(.03, .04, 0), .01, 100, True, .99)
assert not hasattr(motion, 'translation_m')
localizer = SimpleNamespace(relative_motion=lambda *_: motion)
pose, _, accepted = _bounded_motion(localizer, None, None, Pose2D())
assert accepted and abs(pose.x_m - .03) < 1e-9
assert abs(localizer._panorama_motion_diagnostic['translation_m'] - .05) < 1e-9
scan = LaserScan(np.linspace(-math.pi, math.pi, 180, endpoint=False), np.full(180, 2.5), 0.)
_, actual_match, accepted = _bounded_motion(ScanMapLocalizer(None), scan, scan, Pose2D())
assert accepted and isinstance(actual_match, MatchResult)
message = _panorama_result_message({'station_count': 0, 'photo_count': 0,
    'coverage': {'coverage_ratio': 0}, 'motion_complete': False,
    'finish_reason': 'odometry_unavailable_after_stopped_recovery'})
assert '任务未完成' in message and '巡检完成：' not in message
assert _navigation_command_ttl(700, 450) == 350
assert _navigation_command_ttl(700, 450, 1.20, .42) == 700
assert _final_heading_timeout_seconds(8.0, math.pi) >= 18.0
assert build_argument_parser().parse_args(['--defer-online-qwen']).defer_online_qwen
k = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml', clearance_m=.30)
start = (-.815, .565)
start_yaw = math.radians(-126.)
t = k.inspection_targets(standoff_m=1.00, maximum_targets=24, views_per_wall=3,
                         views_per_region=1, preferred_view_xy=start)
t = _spaced_targets(t, .40)
t, _wide = _information_view_targets(k, t, 2.20)
pose = Pose2D(start[0], start[1], start_yaw)
proposals = []
for target in t:
    proposal = _inspection_path_for_target(k, pose, target, .35)
    if proposal:
        proposals.append((proposal['metrics']['score'], target, proposal))
proposals.sort(key=lambda item: item[0])
_score, target, proposal = proposals[0]
metrics = proposal['metrics']
assert math.degrees(metrics['arrival_turn_rad']) < 90.0
print(json.dumps({'target_id': target.target_id, 'view': target.view_xy_m,
                  'yaw_deg': math.degrees(target.yaw_rad),
                  'path_length_m': metrics['length_m'],
                  'arrival_turn_deg': math.degrees(metrics['arrival_turn_rad']),
                  'staged_camera_approach': proposal['staged_camera_approach'],
                  'panorama_real_match_contract': True,
                  'short_ttl_deferred_and_fresh_scan_resumes': True,
                  'physical_footprint_floor': True,
                  'body_return_filter_and_external_obstacle_guard': True,
                  'target_count': len(proposals)}))
""".strip()
    output = command(
        client,
        "cd " + shlex.quote(REMOTE) + " && .venv-pi/bin/python -c "
        + shlex.quote(code),
    )
    result = json.loads(output.strip().splitlines()[-1])
    if (
        int(result["target_count"]) < 1
        or float(result["path_length_m"]) > 1.50
        or float(result["arrival_turn_deg"]) >= 90.0
    ):
        raise RuntimeError("Pi route verification failed: " + json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="dachuang.local")
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        username="dachuang",
        password=os.environ.get("PI_SSH_PASSWORD") or getpass.getpass("Pi password: "),
        timeout=20,
    )
    try:
        active = command(
            client,
            "pgrep -af '[r]aspberry_pi.(autonomous_mapping|joint_mapping|map_camera_inspection)' || true",
        ).strip()
        print("ACTIVE_MOTION_TASKS=" + (active or "none"))
        with client.open_sftp() as sftp:
            map_hashes = {}
            for relative in ("maps/ble_scene_17.pgm", "maps/ble_scene_17.yaml"):
                with sftp.open(REMOTE + "/" + relative, "rb") as source:
                    map_hashes[relative] = hashlib.sha256(source.read()).hexdigest()
                if map_hashes[relative] != hashlib.sha256((ROOT / relative).read_bytes()).hexdigest():
                    raise RuntimeError("Pi map differs from simulation map; refused overwrite: " + relative)
            print("SIMULATION_MAP_MATCH=OK")
            differences = []
            for relative in FILES:
                try:
                    with sftp.open(REMOTE + "/" + relative, "rb") as remote_file:
                        remote_hash = digest(remote_file.read())
                except FileNotFoundError:
                    remote_hash = None
                local_hash = digest((ROOT / relative).read_bytes())
                if remote_hash != local_hash:
                    differences.append(relative)
            print("FILES_TO_UPDATE=" + json.dumps(differences))
            if not args.deploy:
                return
            if active:
                raise RuntimeError("motion task is active; deployment refused")
            if not differences:
                print("ALREADY_CURRENT")
                print("PI_ROUTE=" + json.dumps(remote_route_check(client)))
                return

            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup = REMOTE + "/backups/inspection_route_" + stamp
            staged = []
            for relative in differences:
                destination = REMOTE + "/" + relative
                saved = backup + "/" + relative
                command(client, "mkdir -p " + shlex.quote(posixpath.dirname(saved)))
                try:
                    sftp.stat(destination)
                except FileNotFoundError:
                    pass  # New module: there is no previous file to back up.
                else:
                    command(client, "cp -p " + shlex.quote(destination) + " " + shlex.quote(saved))
                temporary = destination + ".route-staged"
                content = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
                with sftp.open(temporary, "wb") as staged_file:
                    staged_file.write(content)
                with sftp.open(temporary, "rb") as staged_file:
                    if digest(staged_file.read()) != digest(content):
                        raise RuntimeError("upload hash mismatch: " + relative)
                staged.append((temporary, destination))

            python_files = [source for source, _dest in staged if source.endswith(".py.route-staged")]
            if python_files:
                command(
                    client,
                    "cd " + shlex.quote(REMOTE)
                    + " && .venv-pi/bin/python -m py_compile "
                    + " ".join(map(shlex.quote, python_files)),
                )
            shell_files = [source for source, _dest in staged if source.endswith(".sh.route-staged")]
            for source in shell_files:
                command(client, "bash -n " + shlex.quote(source))
            for source, destination in staged:
                sftp.posix_rename(source, destination)
            command(client, "chmod +x " + shlex.quote(REMOTE + "/inspect_map.sh"))

            for relative in differences:
                with sftp.open(REMOTE + "/" + relative, "rb") as remote_file:
                    if digest(remote_file.read()) != digest((ROOT / relative).read_bytes()):
                        raise RuntimeError("activated hash mismatch: " + relative)
            for relative, expected in map_hashes.items():
                with sftp.open(REMOTE + "/" + relative, "rb") as source:
                    if hashlib.sha256(source.read()).hexdigest() != expected:
                        raise RuntimeError("original map changed during deployment: " + relative)
            print("ORIGINAL_MAP_HASHES_UNCHANGED=OK")

        route = remote_route_check(client)
        print("BACKUP=" + backup)
        print("PI_ROUTE=" + json.dumps(route))
        print("DEPLOYED_WITHOUT_STARTING_MOTION")
    finally:
        client.close()


if __name__ == "__main__":
    main()
