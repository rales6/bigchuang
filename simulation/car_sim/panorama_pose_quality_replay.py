"""Offline field acceptance replay and sparse recorded-scan ICP checks.

Stop-event scans are sparse, not a complete drive recording. This tool cannot
reconstruct physical truth, evaluate collisions, or certify full-route success.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.analyze_panorama_pose_drift import rows
from raspberry_pi.known_map_navigation import KnownOccupancyMap, ScanMapLocalizer, LocalizationResult
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.open_space_panorama import _bounded_motion, _correction_rejection_reason


def old_accepted(match):
    translation = math.hypot(match.transform.x_m, match.transform.y_m)
    return bool(match.converged and math.isfinite(translation)
                and math.isfinite(match.transform.yaw_rad)
                and translation <= .38 and abs(match.transform.yaw_rad) <= math.radians(60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    nav = rows(root / 'panorama_navigation.jsonl')
    recorded = []
    seen = set()
    for row in nav:
        fit = row.get('scan_odometry')
        if not fit or fit.get('translation_m') is None or fit.get('yaw_deg') is None:
            continue
        key = (row.get('scan_acquired_monotonic_s'), fit['translation_m'], fit['yaw_deg'])
        if key in seen:
            continue
        seen.add(key)
        # Recorded diagnostic omits XY components. Only quality/envelope
        # policy is replayed, not integration or a counterfactual trajectory.
        match = MatchResult(Pose2D(fit['translation_m'], 0, math.radians(fit['yaw_deg'])),
            fit.get('rmse_m', math.inf), fit.get('correspondences', 0),
            fit.get('rejection_reason') != 'scan_match_failed', fit.get('inlier_ratio', 0))
        turning = isinstance(fit.get('turn_model'), dict)
        localizer = SimpleNamespace(relative_motion=lambda *_: match,
            _panorama_rotation_only=turning, _panorama_turn_model_diagnostic=fit.get('turn_model'))
        _, _, accepted = _bounded_motion(localizer, object(), object(), Pose2D())
        recorded.append({'time': row['updated_at'], 'state': row['state'],
                         'old_accepted': old_accepted(match), 'new_accepted': accepted,
                         **localizer._panorama_motion_diagnostic})
    candidates = []
    for row in rows(root / 'panorama_motion.jsonl'):
        if row.get('event') != 'map_correction_validation':
            continue
        xy = row['candidate_pose']
        pose = Pose2D(xy['x_m'], xy['y_m'], math.radians(xy['yaw_deg']))
        hint = Pose2D(pose.x_m - row['candidate_translation_delta_m'], pose.y_m,
                      pose.yaw_rad - math.radians(row['candidate_yaw_delta_deg']))
        candidate = LocalizationResult(pose, row['candidate_rmse_m'], row['candidate_inlier_ratio'], 0,
            row.get('candidate_ambiguity_margin') if row.get('candidate_ambiguity_margin') is not None else math.inf)
        candidates.append({'old_result': row['result'], 'new_rejection_reason':
            _correction_rejection_reason(candidate, hint, .65, row['yaw_limit_deg']),
            'inliers': candidate.inlier_ratio, 'delta_m': row['candidate_translation_delta_m']})

    localizer = ScanMapLocalizer(KnownOccupancyMap.load('maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml'),
                                lidar_offset_x_m=.14)
    frames = []
    for row in rows(root / 'panorama_stop_events.jsonl'):
        if not all(k in row for k in ('raw_angles_rad', 'filtered_distances_m', 'source_timestamp_s')):
            continue
        distances = [math.nan if d is None else d for d in row['filtered_distances_m']]
        scan = LaserScan(np.asarray(row['raw_angles_rad']), np.asarray(distances), row['source_timestamp_s'])
        if not frames or scan.timestamp_s != frames[-1][0].timestamp_s:
            frames.append((scan, row))
    raw_pairs = []
    for (before, _), (current, event) in zip(frames, frames[1:]):
        gap = current.timestamp_s - before.timestamp_s
        if not 0 < gap <= .75:
            continue
        nearest = min(nav, key=lambda r: abs(r['monotonic_s'] - event['monotonic_s']))
        turning = 'TURN' in nearest['state']
        localizer._panorama_pending_motion = None
        localizer._panorama_rotation_only = turning
        _, fit, accepted = _bounded_motion(localizer, before, current, Pose2D())
        raw_pairs.append({'reference_timestamp_s': before.timestamp_s, 'current_timestamp_s': current.timestamp_s,
            'gap_s': gap, 'turn_context_inferred_from_nearest_logged_state': turning,
            'old_accepted': old_accepted(fit), 'new_accepted': accepted,
            **localizer._panorama_motion_diagnostic})
    known_bad = [r for r in recorded if abs((r.get('translation_m') or 0) - .3352229345) < .00001]
    weak = [r for r in candidates if abs(r['inliers'] - .38125) < .00001]
    regression_passed = bool(known_bad and weak and all(not r['new_accepted'] for r in known_bad)
                             and all(r['new_rejection_reason'] == 'candidate_quality_low' for r in weak))
    report = {'field_regressions_passed': regression_passed, 'full_route_acceptance': False,
        'limits': ['recorded scalar fits are policy-only replay', 'raw scans are sparse stop-event samples',
                   'turn context inferred, no independent physical ground truth',
                   'no trajectory reconstruction or collision/full-mission evaluation'],
        'recorded_fit_count': len(recorded), 'recorded_new_rejection_reasons': dict(Counter(
            r['rejection_reason'] for r in recorded if not r['new_accepted'])),
        'recorded_candidates': candidates, 'known_33cm_failure': known_bad,
        'raw_frame_count': len(frames), 'raw_pair_count': len(raw_pairs),
        'raw_pair_new_rejection_reasons': dict(Counter(r['rejection_reason'] for r in raw_pairs if not r['new_accepted'])),
        'raw_pairs': raw_pairs}
    (root / 'pose_quality_replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw_pairs', 'recorded_candidates')}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if regression_passed else 1)


if __name__ == '__main__':
    main()
