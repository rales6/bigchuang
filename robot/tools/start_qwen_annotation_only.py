"""Run a frozen saved-photo snapshot on the existing 3090 Qwen service.

No Pi write/deployment, motor/camera command, service restart or map update.
All source photo coordinates remain in an audit manifest; this run exports
image semantics without guessed map positions because poses are unverified.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time

from tools.deploy_task_localization_and_start_inventory import (
    connect, python, mkdirs, sha, ROOT, GPU_ROOT,
)


def start(snapshot):
    snapshot = snapshot.resolve()
    metadata = json.loads((snapshot / 'backup_manifest.json').read_text(encoding='utf-8'))
    run = metadata['inspection_run']
    if not re.fullmatch(r'\d{8}_\d{6}', run):
        raise ValueError('Invalid backed-up run ID')
    records = {r['path']: r['sha256'] for r in metadata['files']}
    photos = snapshot / 'pi_snapshot/inspections/ble_scene_17' / run
    manifest = json.loads((photos / 'inspection_manifest.json').read_text(encoding='utf-8'))
    captures = manifest.get('captures', [])
    if not captures:
        raise ValueError('No backed-up inspection captures')
    names = set()
    for capture in captures:
        for key in ('image', 'camera_2_image'):
            name = capture.get(key)
            if name:
                if Path(name).name != name or '/' in name or '\\' in name:
                    raise ValueError('Unsafe photo basename')
                names.add(name)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    job = '/home/gpu/car_inspection_jobs/object_inventory_' + stamp
    local_output = ROOT / 'semantic_maps' / ('qwen_reannotation_' + stamp)
    client = connect('198.51.100.20', 2020, 'sjtu', os.environ['QWEN_SSH_PASSWORD'])
    report = {'timestamp': stamp, 'run_id': run, 'backup': str(snapshot), 'job': job,
              'job_started': False, 'robot_motion_started': False,
              'existing_service_restarted': False, 'capture_count': len(captures),
              'photo_count': len(names), 'coordinate_policy': 'image_semantics_only_unverified_poses',
              'source_coordinates_preserved_in': job + '/photos/source_original_manifest.json'}
    try:
        health = json.loads(python(client, GPU_ROOT + '/.venv/bin/python', """
import requests,json
s=requests.Session();s.trust_env=False
r=s.get('http://127.0.0.1:8010/agent/health',timeout=15);r.raise_for_status()
print(json.dumps(r.json()))
"""))
        if not health.get('ok') or not health.get('model_loaded') or health.get('mock'):
            raise RuntimeError('Real Qwen model is not ready')
        active = json.loads(python(client, 'python3', """
import pathlib,json
active=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(bytes([0]))
  if any(a.endswith(b'/run_semantic_object_inventory.py') for a in args): active.append(p.parent.name)
 except OSError: pass
print(json.dumps(active))
"""))
        if active:
            raise RuntimeError('Another inventory is already running; do not duplicate: ' + str(active))
        local_output.mkdir(parents=True, exist_ok=False)
        report['qwen_health'] = health
        with client.open_sftp() as sftp:
            def upload(content, relative):
                target = job + '/' + relative
                mkdirs(sftp, str(Path(target).parent).replace('\\', '/'))
                with sftp.open(target, 'wb') as stream:
                    stream.write(content)
                with sftp.open(target, 'rb') as stream:
                    if sha(stream.read()) != sha(content):
                        raise RuntimeError('Upload hash mismatch: ' + relative)
                return sha(content)

            def saved_file(source, relative):
                content = source.read_bytes()
                if sha(content) != records[source.relative_to(snapshot).as_posix()]:
                    raise RuntimeError('Backup file changed: ' + str(source))
                return upload(content, relative)

            report['photo_hashes'] = {name: saved_file(photos / name, 'photos/' + name)
                                      for name in sorted(names)}
            saved_file(photos / 'inspection_manifest.json', 'photos/source_original_manifest.json')
            for capture in captures:
                capture['source_recorded_pose'] = {k: capture.get(k) for k in ('x_m', 'y_m', 'yaw_deg')}
                capture.update(x_m=None, y_m=None, yaw_deg=None, low_confidence_pose=True,
                               coordinate_policy='unverified_pose_no_map_projection')
            manifest['coordinate_policy'] = 'image_semantics_only_unverified_poses'
            report['processing_manifest_sha256'] = upload(
                json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'), 'photos/inspection_manifest.json')
            report['map_hashes'] = {name: saved_file(snapshot / 'pi_snapshot/maps' / name, 'maps/' + name)
                                   for name in ('ble_scene_17.pgm', 'ble_scene_17.yaml')}
            code_paths = ['scripts/run_semantic_object_inventory.py', 'scripts/finalize_semantic_inventory.py',
                          'src/qwen_grounded_tracker/__init__.py',
                          'src/qwen_grounded_tracker/inspection_annotation.py',
                          'src/qwen_grounded_tracker/semantic_object_catalog.py',
                          'src/qwen_grounded_tracker/semantic_inventory_quality.py']
            report['job_code'] = {name: saved_file(snapshot / 'local_development_snapshot/3090' / name, name)
                                  for name in code_paths}
        argv = [GPU_ROOT + '/.venv/bin/python', '-u', job + '/scripts/run_semantic_object_inventory.py',
                '--inspection', job + '/photos', '--map', job + '/maps/ble_scene_17.pgm',
                '--map-yaml', job + '/maps/ble_scene_17.yaml', '--output', job + '/output',
                '--ground-url', 'http://127.0.0.1:8010/agent/ground',
                '--queries', '桌子,椅子,柜子,门,窗户,玻璃墙,扫帚,簸箕,楼梯,插座,开关,标牌,灯具',
                '--detail-queries', '插座,开关,标牌,灯具', '--timeout', '90']
        report['command'] = argv
        report['max_expected_http_calls'] = len(names) * 17
        launched = json.loads(python(client, 'python3', """
import pathlib,subprocess,json
job=pathlib.Path(JOB)
log=open(job/'inventory.log','ab',buffering=0)
p=subprocess.Popen(ARGV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
quality_log=open(job/'quality.log','ab',buffering=0)
q=subprocess.Popen([PYTHON,'-u',str(job/'scripts/finalize_semantic_inventory.py'),'--job',str(job),
                    '--inference-pid',str(p.pid),'--timeout','21600'],
                   stdin=subprocess.DEVNULL,stdout=quality_log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'quality_pid':q.pid}))
""".replace('JOB', repr(job)).replace('ARGV', repr(argv)).replace('PYTHON', repr(GPU_ROOT + '/.venv/bin/python'))))
        report.update(launched, job_started=True, quality_stage_started=True,
                      log=job + '/inventory.log', remote_output=job + '/output',
                      quality_log=job + '/quality.log', quality_output=job + '/quality_filtered')
        print(json.dumps({**report, 'local_report': str(local_output / 'deployment_report.json')},
                         ensure_ascii=False, indent=2), flush=True)
        return report
    finally:
        if local_output.exists():
            (local_output / 'deployment_report.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    start(args.snapshot)
