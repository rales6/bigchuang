"""Deploy the hash-pinned pose acceptance revision; never start robot motion."""
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
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = PI_ROOT + '/backups/panorama_pose_quality_' + stamp
    receipt = ROOT / 'simulation_runs' / ('panorama_pose_quality_deployment_' + stamp + '.json')
    reviewed = json.loads((ROOT / 'simulation_runs/panorama_pose_quality_20260918_local_manifest.json').read_text(encoding='utf-8-sig'))
    if not reviewed.get('unit_tests_passed') or not reviewed.get('field_regressions_passed'):
        raise RuntimeError('Reviewed unit tests / field regressions did not pass')
    previous = json.loads((ROOT / reviewed['previous_deployment_receipt']).read_text(encoding='utf-8'))
    expected = {item['path']: item['sha256'] for item in reviewed['source_hashes']}
    sources = ['raspberry_pi/known_map_navigation.py', 'raspberry_pi/open_space_panorama.py']
    protected = {item['path']: previous['after'][item['path']] for item in reviewed['protected_files']}
    matcher = 'raspberry_pi/mapping/scan_matcher.py'
    protected[matcher] = previous['after'][matcher]
    for path in sources:
        content = (ROOT / path).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected[path]:
            raise RuntimeError('Local source changed since tested revision: ' + path)
        ast.parse(content.decode('utf-8'))
    report = {'timestamp': stamp, 'backup': backup, 'deployed': False,
              'robot_started': False, 'files': [], 'unit_tests': reviewed['unit_tests'],
              'field_regressions_passed': True, 'full_disturbance_acceptance': False,
              'user_requested_deployment_of_unpassed_full_simulation': True}
    client = connect('pi.example.invalid', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
    try:
        active = json.loads(python(client, 'python3', ACTIVE_CHECK))
        report['active_missions_before_deployment'] = active
        if active:
            raise RuntimeError('Robot mission running; no files replaced or stop command sent: ' + json.dumps(active))
        before = remote_hashes(client, PI_ROOT, sources + list(protected))
        report['before'] = before
        for path in sources:
            if before[path] not in (previous['after'][path], expected[path]):
                raise RuntimeError('Unexpected remote source; refusing overwrite: ' + path)
        for path, digest in protected.items():
            if before[path] != digest:
                raise RuntimeError('Protected Pi baseline changed: ' + path)
        # Dataclass/matcher caller first; panorama controller second.
        report['files'] = install(client, PI_ROOT, [(ROOT / path, path) for path in sources], backup)
        after = remote_hashes(client, PI_ROOT, sources + list(protected))
        report['after'] = after
        if any(after[path] != expected[path] for path in sources):
            raise RuntimeError('Installed source hash mismatch')
        if any(before[path] != after[path] for path in protected):
            raise RuntimeError('Protected files changed')
        interpreter = pi_python(client)
        report['interpreter'] = interpreter
        code = '''
import os, math
os.chdir(ROOT_PATH)
from types import SimpleNamespace
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.known_map_navigation import LocalizationResult, ScanMapLocalizer
from raspberry_pi.open_space_panorama import _bounded_motion, _correction_rejection_reason
assert callable(ScanMapLocalizer.relative_turn_motion)
weak = MatchResult(Pose2D(.3352229345, 0, .0188), .115125127, 69, True, .3365853658)
localizer = SimpleNamespace(relative_motion=lambda *_:weak, _panorama_rotation_only=True)
origin = Pose2D(.2, -.1, .4)
pose, _, accepted = _bounded_motion(localizer, object(), object(), origin)
assert not accepted and pose == origin
assert localizer._panorama_motion_diagnostic['rejection_reason'] == 'scan_match_quality_low'
assert localizer._panorama_pending_rotation_only is True
localizer._panorama_rotation_only = False
stronger = MatchResult(Pose2D(.2, 0, .02), .03, 120, True, .9)
localizer.relative_turn_motion = lambda *_:stronger
localizer._panorama_turn_model_diagnostic = {'translation_verified':False}
pose, _, accepted = _bounded_motion(localizer, object(), object(), origin)
assert not accepted and pose == origin
assert localizer._panorama_motion_diagnostic['rejection_reason'] == 'turn_translation_unverified'
candidate = LocalizationResult(Pose2D(.5, 0, .1), .0848, .38125, .7, .1, .9, .06, .9)
assert _correction_rejection_reason(candidate, Pose2D(), .65, 35) == 'candidate_quality_low'
good = LocalizationResult(Pose2D(.4, 0, .1), .04, .85, .8, .1, .9, .06, .9)
assert _correction_rejection_reason(good, Pose2D(), .65, 35) == ''
print('POSE_QUALITY_AND_TURN_RECOVERY_CHECK_OK_NO_MOTOR_COMMAND')
'''.replace('ROOT_PATH', repr(PI_ROOT))
        report['check'] = python(client, interpreter, code, timeout=90)
        report['protected_files_unchanged'] = True
        report['deployed'] = True
        print(json.dumps({'deployed': True, 'backup': backup, 'receipt': str(receipt),
                          'check': report['check'], 'protected_files_unchanged': True,
                          'robot_started': False}, ensure_ascii=False), flush=True)
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
