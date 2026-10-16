"""Read-only field evidence snapshot; no robot, camera or service commands."""
import json
import os
from pathlib import Path
import time

from tools.deploy_task_localization_and_start_inventory import (
    connect, python, remote_hashes, ROOT, PI_ROOT,
)


def main():
    output = ROOT / 'simulation_runs' / ('field_pose_drift_' + time.strftime('%Y%m%d_%H%M%S'))
    client = connect('pi.example.invalid', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
    try:
        source = json.loads(python(client, 'python3', r'''
import json, pathlib, re
root = pathlib.Path(ROOT_PATH)
runs = [p for p in (root/'inspections/ble_scene_17').iterdir()
        if p.is_dir() and re.fullmatch(r'\d{8}_\d{6}', p.name)]
latest = max(runs, key=lambda p:p.name)
logs = sorted((root/'inspection_logs').glob('inspect_map_*.log'))
print(json.dumps({'run':str(latest), 'log':str(logs[-1]),
                  'files':{p.name:p.stat().st_size for p in latest.iterdir() if p.is_file()}}))
'''.replace('ROOT_PATH', repr(PI_ROOT))))
        source['deployed_hashes'] = remote_hashes(client, PI_ROOT, [
            'raspberry_pi/open_space_panorama.py', 'raspberry_pi/known_map_navigation.py',
            'raspberry_pi/mapping/scan_matcher.py', 'inspect_map.sh',
        ])
        output.mkdir(parents=True, exist_ok=False)
        names = [
            'navigation_diagnostics.jsonl', 'localization_diagnostics.jsonl',
            'inspection_manifest.json', 'inspection_plan.png', 'panorama_motion.jsonl',
            'panorama_summary.json', 'panorama_motion_summary.json', 'live_pose.json',
            'panorama_coverage.json', 'map_conflicts.jsonl', 'panorama_stopped_scan.jsonl',
            'panorama_navigation.jsonl', 'panorama_map_conflicts.jsonl', 'panorama_stop_events.jsonl',
        ]
        with client.open_sftp() as sftp:
            for name in names:
                if name in source['files']:
                    sftp.get(source['run'] + '/' + name, str(output / name))
            sftp.get(source['log'], str(output / Path(source['log']).name))
            manifest_path = output / 'inspection_manifest.json'
            if manifest_path.exists() and manifest_path.stat().st_size:
                captures = json.loads(manifest_path.read_text(encoding='utf-8')).get('captures', [])
                for row in captures[:1] + captures[-2:]:
                    for name in (row.get('image'), row.get('camera_2_image')):
                        if name and Path(name).name == name and name in source['files']:
                            sftp.get(source['run'] + '/' + name, str(output / name))
        (output / 'source.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'source': source['run'], 'output': str(output),
                          'downloaded': [p.name for p in output.iterdir()]}, ensure_ascii=False))
    finally:
        client.close()


if __name__ == '__main__':
    main()
