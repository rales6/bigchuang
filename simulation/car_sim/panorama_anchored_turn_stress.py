"""Old-map ray simulation using REAL production ICP, not synthetic odometry.

This is a turning-only regression, not whole-route/collision acceptance.
The plant has fixed XY and models slow startup, finite TTL and braking inertia.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.open_space_panorama import PanoramaConfig, _turn_increment


def simulate(seed=0, requested_deg=60., startup_delay_s=.35, noisy=False,
             glass=False, duplicate=False, false_zero=False):
    known = KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml')
    localizer = ScanMapLocalizer(known, lidar_offset_x_m=.14)
    anchor = Pose2D(-.935, .625, math.radians(-128))
    rng = np.random.default_rng(seed)
    clock, yaw, velocity, command, until = 100., 0., 0., 0., 0.
    start_command = None
    last_scan = None
    duplicate_used, false_used = False, False
    commands, states = [], []

    def advance(duration):
        nonlocal clock, yaw, velocity
        remaining = duration
        while remaining > 1e-8:
            step = min(.01, remaining)
            target = command * .52 if clock < until else 0.
            if start_command is None or clock - start_command < startup_delay_s:
                target = 0.
            velocity += (target - velocity) * (1. - math.exp(-step / (.18 if target else .10)))
            yaw += velocity * step
            clock += step
            remaining -= step

    class Client:
        def stop(self):
            nonlocal command, until
            command, until = 0., clock
            advance(.08)  # Stop ACK plus motor braking tail.

        def set_twist(self, linear, angular, ttl_ms):
            nonlocal command, until, start_command
            assert linear == 0
            command, until = angular / 1000., clock + ttl_ms / 1000.
            if start_command is None:
                start_command = clock
            commands.append({'t': clock, 'angular': angular, 'ttl_ms': ttl_ms})
            advance(.12)  # Link ACK latency does not reset the command TTL.

    def make_scan():
        nonlocal last_scan
        angles, distances = localizer.predicted_range_profile(anchor.compose(Pose2D(yaw_rad=yaw)), 360)
        if noisy:
            distances += rng.normal(0., .008, len(distances))
            distances[rng.random(len(distances)) < .06] = np.nan
        if glass:
            world_angles = (angles + anchor.yaw_rad + yaw + math.pi) % (2. * math.pi) - math.pi
            distances[(world_angles > -2.8) & (world_angles < -2.6)] += .8
        last_scan = LaserScan(angles, distances, clock)
        return last_scan

    def scans():
        nonlocal duplicate_used
        while True:
            advance(.12)
            if duplicate and not duplicate_used and abs(yaw) > math.radians(10):
                duplicate_used = True
                yield last_scan
            else:
                yield make_scan()

    original_fit = localizer.relative_turn_motion_from_anchor
    def fit(reference, current):
        nonlocal false_used
        if false_zero and not false_used and abs(yaw) > math.radians(10):
            false_used = True
            result = MatchResult(Pose2D(), .005, 300, True, .99)
            localizer._panorama_turn_model_diagnostic = {
                'model': 'injected_wrong_zero',
                'range_evidence': localizer.turn_range_evidence(reference, current, result.transform)}
            return result
        return original_fit(reference, current)
    localizer.relative_turn_motion_from_anchor = fit

    def status(pose, state, scan):
        states.append({'state': state, 'true_yaw_deg': math.degrees(yaw),
                       'estimated_yaw_deg': math.degrees((pose.yaw_rad - anchor.yaw_rad + math.pi)
                                                        % (2. * math.pi) - math.pi),
                       'diagnostic': localizer._panorama_motion_diagnostic})

    initial_scan = make_scan()
    with patch('raspberry_pi.open_space_panorama.time.monotonic', side_effect=lambda: clock):
        _, pose, measured, success = _turn_increment(Client(), iter(scans()), localizer,
            initial_scan, anchor, math.radians(requested_deg), PanoramaConfig(), status,
            motion_is_safe=lambda *_: True)
    # Compare to the last acquired scan, not motor motion after its timestamp.
    last = states[-1] if states else {'true_yaw_deg': math.degrees(yaw), 'estimated_yaw_deg': 0.}
    error = abs(last['true_yaw_deg'] - last['estimated_yaw_deg'])
    return {'seed': seed, 'requested_deg': requested_deg, 'success': success,
            'true_yaw_deg_at_last_status': last['true_yaw_deg'],
            'estimated_yaw_deg': last['estimated_yaw_deg'], 'heading_error_deg': error,
            'measured_deg': math.degrees(measured), 'command_count': len(commands),
            'elapsed_simulated_s': clock - 100., 'false_zero_injected': false_used,
            'duplicate_injected': duplicate_used, 'noise': noisy, 'glass_outliers': glass,
            'pending_motion': getattr(localizer, '_panorama_pending_motion', None) is not None,
            'passed': success and error <= 3. and abs(abs(math.degrees(measured)) - abs(requested_deg)) <= 15.,
            'states': states}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='simulation_runs/panorama_anchored_turn_20260918.json')
    args = parser.parse_args()
    maps = [Path('maps/ble_scene_17.pgm'), Path('maps/ble_scene_17.yaml')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in maps}
    cases = [dict(requested_deg=60.), dict(requested_deg=-60.), dict(startup_delay_s=.65),
             dict(noisy=True), dict(noisy=True, glass=True), dict(duplicate=True), dict(false_zero=True)]
    results = [simulate(i, **case) for i, case in enumerate(cases)]
    unchanged = all(hashes[str(p)] == hashlib.sha256(p.read_bytes()).hexdigest() for p in maps)
    report = {'real_production_icp': True, 'old_map': 'ble_scene_17',
              'scope': 'turning_only_fixed_xy_not_full_route_acceptance',
              'map_hashes': hashes, 'map_hashes_unchanged': unchanged,
              'passed': unchanged and all(r['passed'] for r in results), 'results': results}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({**report, 'results': [{k:v for k,v in r.items() if k != 'states'} for r in results]}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
