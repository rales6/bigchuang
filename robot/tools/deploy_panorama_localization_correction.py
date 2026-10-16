"""Deploy the reviewed localization revision only; never start robot motion."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

from tools.deploy_task_localization_and_start_inventory import (
    connect, install, remote_hashes, python, pi_python, PI_ROOT, ROOT,
)


def main():
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = PI_ROOT + '/backups/panorama_localization_' + stamp
    report_path = ROOT / 'simulation_runs' / ('panorama_localization_deployment_' + stamp + '.json')
    sources = ['raspberry_pi/known_map_navigation.py', 'raspberry_pi/open_space_panorama.py']
    # Keep subsequent local navigation work OUT of this reviewed deployment.
    # The snapshot was downloaded and hash-verified after the first install.
    source_root = ROOT / 'backups/panorama_localization_deployed_20260918_162110'
    if not all((source_root / p).is_file() for p in sources):
        raise RuntimeError('Frozen localization deployment snapshot missing; do not deploy evolving local files')
    protected = ['raspberry_pi/map_camera_inspection.py', 'inspect_map.sh',
                 'maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml']
    baseline = ROOT / 'simulation_runs/field_localization_20260918_155858/deployed_sources'
    expected = {p: hashlib.sha256((baseline / Path(p).name).read_bytes()).hexdigest() for p in sources}
    client = connect('pi.example.invalid', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
    try:
        active = json.loads(python(client, 'python3', '''
import json, pathlib
names = {'raspberry_pi.map_camera_inspection', 'raspberry_pi.object_search_mission',
         'raspberry_pi.autonomous_mapping', 'raspberry_pi.joint_mapping'}
scripts = {n.rsplit('.', 1)[-1] + '.py' for n in names}
active = []
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().decode(errors='replace').split('\\0')
        if any(a in names or a.rsplit('/', 1)[-1] in scripts for a in args):
            active.append({'pid': int(path.parent.name), 'argv': args})
    except (OSError, ValueError):
        pass
print(json.dumps(active))
'''))
        if active:
            raise RuntimeError('Robot mission still running; deployment not performed: ' + json.dumps(active))
        before = remote_hashes(client, PI_ROOT, sources + protected)
        for p in sources:
            current_sha = hashlib.sha256((source_root / p).read_bytes()).hexdigest()
            if before[p] not in (expected[p], current_sha):
                raise RuntimeError('Remote file changed since reviewed snapshot: ' + p)
        # Install the backward-compatible matcher first, then its new caller.
        files = [(source_root / p, p) for p in sources]
        doc = 'raspberry_pi/docs/panorama_map_correction_20260918.md'
        files.append((ROOT / doc, doc))
        manifest = install(client, PI_ROOT, files, backup)
        after = remote_hashes(client, PI_ROOT, sources + protected)
        if any(before[p] != after[p] for p in protected):
            raise RuntimeError('Protected files changed during deployment')
        interpreter = pi_python(client)
        check = python(client, interpreter,
            'import os,inspect; os.chdir(' + repr(PI_ROOT) + '); '
            'from raspberry_pi.known_map_navigation import ScanMapLocalizer; '
            'from raspberry_pi.open_space_panorama import _CorrectionObservation,_stopped_heading_calibration; '
            "assert 'yaw_radius_deg' in inspect.signature(ScanMapLocalizer.localize_bounded).parameters; "
            "print('LOCALIZATION_IMPORT_OK_NO_MOTOR_COMMAND')")
        report = {'backup': backup, 'files': manifest, 'before': before, 'after': after,
                  'source_root': str(source_root),
                  'interpreter': interpreter, 'check': check, 'robot_started': False}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'deployed': True, 'backup': backup, 'report': str(report_path),
                          'check': check, 'protected_files_unchanged': True, 'robot_started': False}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
