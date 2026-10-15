"""Deploy the user's explicitly requested second revision, without motor commands."""
import ast
import hashlib
import json
import os
import time

from tools.deploy_task_localization_and_start_inventory import (
    ROOT, PI_ROOT, connect, install, remote_hashes, python, pi_python,
)


SOURCES = [
    'raspberry_pi/mapping/scan_matcher.py',
    'raspberry_pi/known_map_navigation.py',
    'raspberry_pi/open_space_panorama.py',
]
BASELINE = {
    SOURCES[0]: 'b216a36802376f5575c3b49d9ebdbd1a53de9212ffa2ab6576ac638484fb6638',
    SOURCES[1]: 'fd434afd5c7640630a24423fc32b91b51fa3b443c6eed10713d099c8947dbc80',
    SOURCES[2]: '7b06c6d998c9c33e8997087e210f31e8320d8e1ab5ba3f4ed9f3bc91e76a9e40',
}
ACTIVE_CHECK = '''
import json, pathlib
names = {'raspberry_pi.map_camera_inspection', 'raspberry_pi.object_search_mission',
         'raspberry_pi.autonomous_mapping', 'raspberry_pi.joint_mapping',
         'raspberry_pi.open_space_panorama'}
scripts = {n.rsplit('.', 1)[-1] + '.py' for n in names}
scripts.update({'inspect_map.sh', 'map_and_annotate.sh', 'search_object.sh', 'natural_search.sh'})
active = []
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args = path.read_bytes().decode(errors='replace').split('\\0')
        if any(a in names or a.rsplit('/', 1)[-1] in scripts for a in args):
            active.append({'pid': int(path.parent.name), 'argv': args})
    except (OSError, ValueError):
        pass
print(json.dumps(active))
'''


def main():
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = PI_ROOT + '/backups/panorama_repeat_motion_' + stamp
    receipt = ROOT / 'simulation_runs' / ('panorama_repeat_motion_deployment_' + stamp + '.json')
    evidence = ROOT / 'simulation_runs/panorama_repeat_motion_20260918_local_manifest.json'
    reviewed = json.loads(evidence.read_text(encoding='utf-8-sig'))
    expected = {item['path']: item['sha256'] for item in reviewed['source_hashes']}
    for path in SOURCES:
        content = (ROOT / path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected[path]:
            raise RuntimeError('Local source changed after tested revision: ' + path)
        ast.parse(content.decode('utf-8'))
    previous = json.loads((ROOT / reviewed['previous_deployment_receipt']).read_text(encoding='utf-8'))
    # Preserve the Pi's recorded baseline, not a different local entrypoint.
    protected = {item['path']: previous['after'][item['path']]
                 for item in reviewed['protected_files']}
    report = {
        'timestamp': stamp, 'backup': backup, 'robot_started': False,
        'full_disturbance_acceptance': False, 'unit_tests': reviewed['unit_tests'],
        'user_requested_deployment_of_unpassed_full_simulation': True,
        'deployed': False, 'files': [],
    }
    client = connect('dachuang.local', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
    try:
        if json.loads(python(client, 'python3', ACTIVE_CHECK)):
            raise RuntimeError('Robot mission is running; no files replaced and no stop command sent')
        before = remote_hashes(client, PI_ROOT, SOURCES + list(protected))
        report['before'] = before
        for path in SOURCES:
            if before[path] not in (BASELINE[path], expected[path]):
                raise RuntimeError('Unexpected remote source; refusing to overwrite: ' + path)
        for path, digest in protected.items():
            if before[path] != digest:
                raise RuntimeError('Protected file differs from reviewed revision: ' + path)
        # New matcher first, backward-compatible caller second, controller last.
        report['files'] = install(client, PI_ROOT, [(ROOT / p, p) for p in SOURCES], backup)
        after = remote_hashes(client, PI_ROOT, SOURCES + list(protected))
        report['after'] = after
        if any(after[p] != expected[p] for p in SOURCES):
            raise RuntimeError('Installed source hash mismatch')
        if any(before[p] != after[p] for p in protected):
            raise RuntimeError('Protected files changed')
        interpreter = pi_python(client)
        report['interpreter'] = interpreter
        check = '''
import os, math
os.chdir(ROOT_PATH)
import numpy as np
from unittest.mock import Mock
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import IcpScanMatcher
from raspberry_pi.known_map_navigation import ScanMapLocalizer
from raspberry_pi.open_space_panorama import PanoramaConfig, OpenSpaceDirectionSelector, _EscapeTurnBudget, _observe_photo_coverage
assert callable(ScanMapLocalizer.relative_turn_motion)
assert callable(OpenSpaceDirectionSelector.record_translation)
reference = np.random.default_rng(30).uniform(-2, 2, (120, 2))
truth = Pose2D(0, 0, math.radians(20))
match = IcpScanMatcher().match_rotation(reference, truth.inverse().transform_points(reference))
assert match.converged and abs(match.transform.yaw_rad - truth.yaw_rad) < math.radians(.5)
assert match.transform.x_m == 0 and match.transform.y_m == 0
verified, provisional = Mock(), Mock()
provisional.observe.return_value = 3
assert _observe_photo_coverage(verified, provisional, Pose2D(), 0, PanoramaConfig(), True) == (0, 3)
verified.observe.assert_not_called()
print('PANORAMA_SECOND_REVISION_IMPORT_AND_ROTATION_OK_NO_MOTOR_COMMAND')
'''.replace('ROOT_PATH', repr(PI_ROOT))
        report['check'] = python(client, interpreter, check, timeout=90)
        report['protected_files_unchanged'] = True
        report['deployed'] = True
        print(json.dumps({'deployed': True, 'backup': backup, 'receipt': str(receipt),
                          'check': report['check'], 'robot_started': False,
                          'protected_files_unchanged': True}, ensure_ascii=False), flush=True)
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
