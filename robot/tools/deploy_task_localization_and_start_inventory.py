"""Deploy independent task modules and start one photo-only Qwen inventory job.

No motor command, camera capture, service restart or original map replacement.
Existing destination files get timestamped backups and atomic replacement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shlex
import time

import paramiko

ROOT = Path(__file__).resolve().parents[1]
PI_ROOT = '/home/dachuang/dachuang/car1.0'
GPU_ROOT = '/home/sjtu/3090'
READ_ONLY_INSPECTION = ['raspberry_pi/map_camera_inspection.py', 'raspberry_pi/open_space_panorama.py', 'inspect_map.sh']


def connect(host, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password,
                   timeout=15, banner_timeout=15, auth_timeout=15)
    return client


def command(client, code, timeout=45):
    _, stdout, stderr = client.exec_command(code, timeout=timeout)
    output, error = stdout.read().decode('utf-8', 'replace'), stderr.read().decode('utf-8', 'replace')
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error or output)
    return output.strip()


def python(client, interpreter, code, timeout=45):
    return command(client, shlex.join([interpreter, '-c', code]), timeout)


def mkdirs(sftp, path):
    parts = []
    while path not in ('', '/'):
        parts.append(path)
        path = posixpath.dirname(path)
    for folder in reversed(parts):
        try:
            sftp.stat(folder)
        except FileNotFoundError:
            sftp.mkdir(folder)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def install(client, root, files, backup):
    manifest = []
    with client.open_sftp() as sftp:
        for local, relative in files:
            destination = root + '/' + relative
            mkdirs(sftp, posixpath.dirname(destination))
            saved = backup + '/' + relative
            mkdirs(sftp, posixpath.dirname(saved))
            try:
                with sftp.open(destination, 'rb') as stream:
                    old = stream.read()
                with sftp.open(saved, 'wb') as stream:
                    stream.write(old)
                old_sha = sha(old)
            except FileNotFoundError:
                old_sha = None
            content = local.read_bytes()
            temporary = destination + '.task-semantic-staged'
            with sftp.open(temporary, 'wb') as stream:
                stream.write(content)
            with sftp.open(temporary, 'rb') as stream:
                if sha(stream.read()) != sha(content):
                    raise RuntimeError('Upload hash mismatch: ' + relative)
            if relative.endswith('.py'):
                python(client, 'python3', 'import ast; ast.parse(open(' + repr(temporary) + ",encoding='utf-8').read())")
            # Check destination has not changed since the backup was made.
            try:
                with sftp.open(destination, 'rb') as stream:
                    latest_sha = sha(stream.read())
            except FileNotFoundError:
                latest_sha = None
            if latest_sha != old_sha:
                raise RuntimeError('Destination changed during deployment: ' + relative)
            sftp.posix_rename(temporary, destination)
            manifest.append({'path': destination, 'sha256': sha(content), 'previous_sha256': old_sha})
    return manifest


def remote_hashes(client, root, paths):
    return json.loads(python(client, 'python3',
        'import hashlib,json; from pathlib import Path; root=Path(' + repr(root) + '); paths=' + repr(paths)
        + "; print(json.dumps({p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}))"))


def pi_python(client):
    return python(client, 'python3', 'import os; paths=' + repr([
        PI_ROOT + '/.venv-pi/bin/python', PI_ROOT + '/.venv/bin/python'])
        + "; print(next((p for p in paths if os.access(p,os.X_OK)), 'python3'))")


def attach_quality_stage(report_path):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not report.get('job_started'):
        raise ValueError('No inference job recorded')
    job = report['job']
    if not job.startswith('/home/sjtu/car_inspection_jobs/object_inventory_'):
        raise ValueError('Unexpected job path')
    client = connect('202.121.181.124', 2020, 'sjtu', os.environ['QWEN_SSH_PASSWORD'])
    try:
        if report.get('quality_stage_started'):
            print(json.dumps({'already_attached': True, 'pid': report['quality_pid']}, ensure_ascii=False), flush=True)
            return report
        stamp = time.strftime('%Y%m%d_%H%M%S')
        files = ['scripts/finalize_semantic_inventory.py', 'src/qwen_grounded_tracker/semantic_inventory_quality.py']
        report['quality_gpu_files'] = install(client, GPU_ROOT,
            [(ROOT / '3090' / name, name) for name in files + ['scripts/run_semantic_object_inventory.py']],
            GPU_ROOT + '/backups/inventory_quality_' + stamp)
        report['quality_job_files'] = install(client, job,
            [(ROOT / '3090' / name, name) for name in files], job + '/quality_code_backup')
        argv = [GPU_ROOT + '/.venv/bin/python', '-u', job + '/scripts/finalize_semantic_inventory.py',
                '--job', job, '--inference-pid', str(report['pid'])]
        code = ('import subprocess,json; log=open(' + repr(job + '/quality.log') + ", 'ab',buffering=0); "
            + 'p=subprocess.Popen(' + repr(argv) + ',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
            + "print(json.dumps({'pid':p.pid}))")
        pid = json.loads(python(client, 'python3', code))['pid']
        report.update(quality_stage_started=True, quality_pid=pid, quality_log=job + '/quality.log',
                      quality_output=job + '/quality_filtered')
        print(json.dumps({'quality_stage_started': True, 'pid': pid, 'output': report['quality_output']}, ensure_ascii=False), flush=True)
        return report
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        client.close()


def deploy_and_start(run_id):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    local_output = ROOT / 'semantic_maps' / ('qwen_reannotation_' + stamp)
    local_output.mkdir(parents=True, exist_ok=True)
    report = {'timestamp': stamp, 'run_id': run_id, 'robot_motion_started': False,
              'task_localization_live_enabled': False, 'job_started': False}
    clients = []
    try:
        pi = connect('dachuang.local', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
        clients.append(pi)
        gpu = connect('202.121.181.124', 2020, 'sjtu', os.environ['QWEN_SSH_PASSWORD'])
        clients.append(gpu)
        active_jobs = command(gpu, "pgrep -af '[r]un_semantic_object_inventory.py' || true")
        if active_jobs:
            raise RuntimeError('Existing inventory job detected; not starting duplicate: ' + active_jobs)
        health = json.loads(python(gpu, GPU_ROOT + '/.venv/bin/python',
            "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8010/agent/health',timeout=8))))"))
        if health.get('ok') is not True or health.get('mock') is not False or health.get('model_loaded') is not True:
            raise RuntimeError('Real Qwen service is not ready')
        report['qwen_health'] = health
        protected = READ_ONLY_INSPECTION + ['maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml']
        before = remote_hashes(pi, PI_ROOT, protected)
        pi_paths = ['raspberry_pi/task_semantic_localization.py', 'raspberry_pi/task_photo_observer.py',
                    'raspberry_pi/docs/task_semantic_localization.md', 'raspberry_pi/examples/task_semantic_frames.json',
                    'tools/task_semantic_localization_probe.py']
        pi_backup = PI_ROOT + '/backups/task_semantic_' + stamp
        report['pi_files'] = install(pi, PI_ROOT, [(ROOT / p, p) for p in pi_paths], pi_backup)
        report['pi_backup'] = pi_backup
        interpreter = pi_python(pi)
        report['pi_import_check'] = command(pi, 'cd ' + shlex.quote(PI_ROOT) + ' && '
            + shlex.join([interpreter, '-c', 'from raspberry_pi.task_semantic_localization import TaskSemanticLocalizer; from raspberry_pi.task_photo_observer import QwenTaskPhotoObserver; print("TASK_ONLY_IMPORT_OK")']))
        if remote_hashes(pi, PI_ROOT, protected) != before:
            raise RuntimeError('Protected inspection/map file changed')
        report['protected_pi_hashes'] = before
        gpu_paths = ['scripts/run_semantic_object_inventory.py', 'src/qwen_grounded_tracker/semantic_object_catalog.py',
                     'docs/SEMANTIC_OBJECT_MEMORY.md']
        gpu_backup = GPU_ROOT + '/backups/object_inventory_' + stamp
        report['gpu_files'] = install(gpu, GPU_ROOT, [(ROOT / '3090' / p, p) for p in gpu_paths], gpu_backup)
        report['gpu_backup'] = gpu_backup
        report['gpu_import_check'] = command(gpu, shlex.join([GPU_ROOT + '/.venv/bin/python',
            GPU_ROOT + '/scripts/run_semantic_object_inventory.py', '--help']))
        job = '/home/sjtu/car_inspection_jobs/object_inventory_' + stamp
        report['job'] = job
        with pi.open_sftp() as source, gpu.open_sftp() as destination:
            photos = PI_ROOT + '/inspections/ble_scene_17/' + run_id
            with source.open(photos + '/inspection_manifest.json', 'rb') as stream:
                manifest_bytes = stream.read()
            captures = json.loads(manifest_bytes)['captures']
            if not captures:
                raise RuntimeError('No photo captures in selected inspection')
            names = ['inspection_manifest.json']
            for capture in captures:
                for name in [capture.get('image'), capture.get('camera_2_image')]:
                    if name:
                        if posixpath.basename(name) != name:
                            raise RuntimeError('Unsafe photo basename')
                        names.append(name)
            source_hashes = {}
            for name in sorted(set(names)):
                with source.open(photos + '/' + name, 'rb') as stream:
                    data = stream.read()
                target = job + '/photos/' + name
                mkdirs(destination, posixpath.dirname(target))
                with destination.open(target, 'wb') as stream:
                    stream.write(data)
                with destination.open(target, 'rb') as stream:
                    if sha(stream.read()) != sha(data):
                        raise RuntimeError('Photo copy hash mismatch')
                source_hashes[name] = sha(data)
            for name in ['ble_scene_17.pgm', 'ble_scene_17.yaml']:
                with source.open(PI_ROOT + '/maps/' + name, 'rb') as stream:
                    data = stream.read()
                mkdirs(destination, job + '/maps')
                with destination.open(job + '/maps/' + name, 'wb') as stream:
                    stream.write(data)
                with destination.open(job + '/maps/' + name, 'rb') as stream:
                    if sha(stream.read()) != sha(data):
                        raise RuntimeError('Remote map copy hash mismatch')
                if sha(data) != before['maps/' + name]:
                    raise RuntimeError('Map changed during copy')
        report['photo_hashes'] = source_hashes
        report['capture_count'] = len(captures)
        # Snapshot exact inference code so later deployments cannot change this job.
        isolated_paths = ['scripts/run_semantic_object_inventory.py', 'src/qwen_grounded_tracker/__init__.py',
                          'src/qwen_grounded_tracker/inspection_annotation.py',
                          'src/qwen_grounded_tracker/semantic_object_catalog.py']
        report['job_code'] = install(gpu, job, [(ROOT / '3090' / p, p) for p in isolated_paths], job + '/code_backup')
        argv = [GPU_ROOT + '/.venv/bin/python', '-u', job + '/scripts/run_semantic_object_inventory.py',
            '--inspection', job + '/photos', '--map', job + '/maps/ble_scene_17.pgm',
            '--map-yaml', job + '/maps/ble_scene_17.yaml', '--output', job + '/output',
            '--ground-url', 'http://127.0.0.1:8010/agent/ground',
            '--queries', '桌子,椅子,柜子,门,窗户,玻璃墙,扫帚,簸箕,楼梯,插座,开关,标牌,灯具',
            '--detail-queries', '插座,开关,标牌,灯具', '--timeout', '90']
        report['command'] = argv
        report['max_expected_http_calls'] = sum(bool(c.get('image')) + bool(c.get('camera_2_image')) for c in captures) * 17
        # A unique job directory and process group; no system service restart.
        code = ('import subprocess,json; log=open(' + repr(job + '/inventory.log') + ", 'ab',buffering=0); "
                + 'p=subprocess.Popen(' + repr(argv) + ',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True); '
                + "print(json.dumps({'pid':p.pid}))")
        launched = json.loads(python(gpu, 'python3', code))
        report.update(job_started=True, pid=launched['pid'], log=job + '/inventory.log',
                      remote_output=job + '/output', existing_service_restarted=False)
        print(json.dumps({'job_started': True, 'pid': report['pid'], 'job': job,
                          'captures': len(captures), 'max_http_calls': report['max_expected_http_calls'],
                          'local_report': str(local_output / 'deployment_report.json')}, ensure_ascii=False), flush=True)
        return report
    finally:
        (local_output / 'deployment_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        for client in clients:
            client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='20260917_190706')
    parser.add_argument('--existing-report', type=Path, help='Only attach quality filter to this already-started job')
    args = parser.parse_args()
    if re.fullmatch(r'\d{8}_\d{6}', args.run_id) is None:
        parser.error('Expected YYYYMMDD_HHMMSS run ID')
    if args.existing_report:
        attach_quality_stage(args.existing_report)
    else:
        result = deploy_and_start(args.run_id)
        attach_quality_stage(ROOT / 'semantic_maps' / ('qwen_reannotation_' + result['timestamp']) / 'deployment_report.json')
