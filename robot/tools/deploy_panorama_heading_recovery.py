"""Install hash-reviewed heading recovery; never start motors or cameras."""
import ast
import hashlib
import json
import os
import time

from tools.deploy_panorama_repeat_motion import ACTIVE_CHECK
from tools.deploy_task_localization_and_start_inventory import (
    ROOT, PI_ROOT, connect, install, remote_hashes, python, pi_python,
)


def main():
    reviewed = json.loads((ROOT / 'simulation_runs/panorama_heading_recovery_20260918_local_manifest.json').read_text(encoding='utf-8'))
    if not reviewed['unit_tests_passed']:
        raise RuntimeError('Reviewed tests not passed')
    sources = ['raspberry_pi/mapping/scan_matcher.py', 'raspberry_pi/known_map_navigation.py',
               'raspberry_pi/open_space_panorama.py']
    expected = {row['path']: row['sha256'] for row in reviewed['source_hashes']}
    previous = json.loads((ROOT / reviewed['previous_deployment_receipt']).read_text(encoding='utf-8'))
    protected = {path: previous['after'][path] for path in
                 ['maps/ble_scene_17.pgm', 'maps/ble_scene_17.yaml',
                  'raspberry_pi/map_camera_inspection.py', 'inspect_map.sh']}
    for path in sources:
        content = (ROOT / path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected[path]:
            raise RuntimeError('Source changed after review: ' + path)
        ast.parse(content.decode('utf-8'))
    stamp = time.strftime('%Y%m%d_%H%M%S')
    receipt = ROOT / 'simulation_runs' / ('panorama_heading_recovery_deployment_' + stamp + '.json')
    backup = PI_ROOT + '/backups/panorama_heading_recovery_' + stamp
    report = {'deployed': False, 'robot_started': False, 'backup': backup,
              'unit_tests': reviewed['unit_tests'], 'full_disturbance_acceptance': False}
    client = connect('pi.example.invalid', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
    try:
        active = json.loads(python(client, 'python3', ACTIVE_CHECK))
        if active:
            raise RuntimeError('Active mission; no replacement: ' + json.dumps(active))
        before = remote_hashes(client, PI_ROOT, sources + list(protected))
        report['before'] = before
        for path in sources:
            if before[path] not in (previous['after'][path], expected[path]):
                raise RuntimeError('Unexpected remote source: ' + path)
        for path, digest in protected.items():
            if before[path] != digest:
                raise RuntimeError('Protected remote baseline changed: ' + path)
        report['files'] = install(client, PI_ROOT, [(ROOT / path, path) for path in sources], backup)
        after = remote_hashes(client, PI_ROOT, sources + list(protected))
        report['after'] = after
        if any(after[path] != expected[path] for path in sources):
            raise RuntimeError('Installed source hash mismatch')
        if any(after[path] != digest for path, digest in protected.items()):
            raise RuntimeError('Protected files changed')
        code = '''
import math, os
from types import SimpleNamespace
os.chdir(ROOT_PATH)
from raspberry_pi.open_space_panorama import _turn_pulse_ttl_ms, _recover_pending_turn_heading, PanoramaConfig
from raspberry_pi.known_map_navigation import ScanMapLocalizer
from raspberry_pi.mapping import Pose2D
assert callable(ScanMapLocalizer.recover_turn_heading)
assert _turn_pulse_ttl_ms(math.radians(20), 3000, 0, True) == 120
reference = SimpleNamespace(timestamp_s=1.)
anchor = Pose2D(.1, .2, .3)
localizer = SimpleNamespace(_panorama_pending_motion=(reference, anchor), _panorama_pending_rotation_only=True)
_, pose, accepted = _recover_pending_turn_heading(SimpleNamespace(stop=lambda:None), iter(()),
    localizer, reference, anchor, PanoramaConfig())
assert not accepted and pose == anchor and localizer._panorama_pending_motion[0] is reference
print('HEADING_RECOVERY_CHECK_OK_NO_MOTOR_COMMAND')
'''.replace('ROOT_PATH', repr(PI_ROOT))
        report['check'] = python(client, pi_python(client), code)
        report['deployed'] = True
        print(json.dumps({'deployed': True, 'robot_started': False, 'backup': backup,
                          'receipt': str(receipt), 'check': report['check']}, ensure_ascii=False))
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
