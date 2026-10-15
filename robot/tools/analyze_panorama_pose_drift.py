"""Summarize recorded pose estimates, not independent physical ground truth."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


def rows(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A live download may contain one unfinished final record.
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    nav = rows(root / 'panorama_navigation.jsonl')
    motion = rows(root / 'panorama_motion.jsonl')
    captures = json.loads((root / 'inspection_manifest.json').read_text(encoding='utf-8')).get('captures', [])
    totals = defaultdict(float)
    jumps = []
    for a, b in zip(nav, nav[1:]):
        moved = math.hypot(b['x_m'] - a['x_m'], b['y_m'] - a['y_m'])
        totals[b['state']] += moved
        jumps.append({'time': b['updated_at'], 'state': b['state'], 'translation_m': moved,
                      'from': [a['x_m'], a['y_m'], a['yaw_deg']],
                      'to': [b['x_m'], b['y_m'], b['yaw_deg']],
                      'motion': b.get('odometry'), 'command': b.get('motion_command')})
    stations = defaultdict(list)
    for row in captures:
        stations[row.get('panorama_station_id', row.get('station_id', row.get('target_id', '').split('_d')[0]))].append(row)
    station_summary = []
    for key, photos in stations.items():
        first, last = photos[0], photos[-1]
        station_summary.append({
            'station': key, 'photos': len(photos),
            'first_pose': [first['x_m'], first['y_m'], first['yaw_deg']],
            'last_pose': [last['x_m'], last['y_m'], last['yaw_deg']],
            'estimated_shift_m': math.hypot(last['x_m'] - first['x_m'], last['y_m'] - first['y_m']),
            'low_confidence': sum(bool(p.get('pose_uncertain', p.get('low_confidence_position', False))) for p in photos),
            'new_verified_coverage': sum(p.get('new_coverage_samples', 0) for p in photos),
        })
    models = Counter()
    samples = []
    for row in nav:
        for key, value in row.items():
            if isinstance(value, dict) and 'turn_model' in value:
                model = value.get('turn_model')
                if isinstance(model, dict):
                    models[model.get('model')] += 1
                samples.append({'time': row['updated_at'], 'state': row['state'], 'field': key, **value})
    validation = [r for r in motion if r.get('event') == 'map_correction_validation']
    heading = [r for r in motion if r.get('event') == 'heading_calibration']
    summary = {
        'not_ground_truth': True, 'navigation_records': len(nav),
        'first_pose': {k: nav[0][k] for k in ('updated_at', 'x_m', 'y_m', 'yaw_deg')},
        'last_pose': {k: nav[-1][k] for k in ('updated_at', 'x_m', 'y_m', 'yaw_deg', 'pose_uncertain', 'coverage_ratio')},
        'estimated_travel_by_state_m': dict(totals),
        'largest_pose_updates': sorted(jumps, key=lambda r:r['translation_m'], reverse=True)[:8],
        'map_validation_counts': dict(Counter(r['result'] for r in validation)),
        'confirmed_candidates': [r for r in validation if r['result'] == 'confirmed_window'],
        'heading_results': dict(Counter(r['result'] for r in heading)),
        'stations': station_summary, 'turn_model_status_samples': dict(models),
        'largest_match_translations': sorted(samples, key=lambda r:r.get('translation_m') or 0, reverse=True)[:8],
        'navigation_keys': list(nav[-1]),
    }
    (root / 'pose_drift_analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
