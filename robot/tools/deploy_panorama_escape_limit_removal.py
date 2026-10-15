"""Deploy only the reviewed escape cutoff removal; never start the robot."""
import ast
import hashlib
import json
import os
import time

from tools.deploy_panorama_repeat_motion import ACTIVE_CHECK
from tools.deploy_task_localization_and_start_inventory import (
    ROOT, PI_ROOT, connect, install, python, pi_python, remote_hashes,
)


def main():
    manifest = json.loads((ROOT/'simulation_runs/panorama_escape_limit_removal_20260918_local_manifest.json').read_text(encoding='utf-8'))
    if not manifest['unit_tests_passed']:
        raise RuntimeError('Tests not passed')
    source = 'raspberry_pi/open_space_panorama.py'
    expected = manifest['source_sha256']
    content = (ROOT/source).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Reviewed source changed')
    ast.parse(content.decode('utf-8'))
    previous = json.loads((ROOT/manifest['previous_deployment_receipt']).read_text(encoding='utf-8'))
    protected = {path:previous['after'][path] for path in (
        'maps/ble_scene_17.pgm','maps/ble_scene_17.yaml','inspect_map.sh',
        'raspberry_pi/map_camera_inspection.py','raspberry_pi/known_map_navigation.py',
        'raspberry_pi/mapping/scan_matcher.py')}
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = PI_ROOT+'/backups/panorama_escape_limit_removal_'+stamp
    receipt = ROOT/'simulation_runs'/('panorama_escape_limit_removal_deployment_'+stamp+'.json')
    report = {'deployed':False,'robot_started':False,'backup':backup,
              'unit_tests':manifest['unit_tests'],'full_disturbance_acceptance':False}
    client = connect('dachuang.local',22,'dachuang',os.environ['PI_SSH_PASSWORD'])
    try:
        active = json.loads(python(client,'python3',ACTIVE_CHECK))
        if active:
            raise RuntimeError('Active mission; no replacement: '+json.dumps(active))
        before = remote_hashes(client,PI_ROOT,[source]+list(protected))
        report['before'] = before
        if before[source] not in (previous['after'][source],expected):
            raise RuntimeError('Unexpected remote controller baseline')
        if any(before[path] != digest for path,digest in protected.items()):
            raise RuntimeError('Protected remote file changed')
        report['files'] = install(client,PI_ROOT,[(ROOT/source,source)],backup)
        after = remote_hashes(client,PI_ROOT,[source]+list(protected))
        report['after'] = after
        if after[source] != expected or any(after[path] != digest for path,digest in protected.items()):
            raise RuntimeError('Installed/protected file mismatch')
        code = '''
import os, math
os.chdir(ROOT_PATH)
from raspberry_pi.open_space_panorama import _local_motion_action, _turn_pulse_ttl_ms
assert _local_motion_action(.1,.61,.5,math.pi,0,.22)=='turn'
assert _turn_pulse_ttl_ms(math.radians(20),3000,0,True,4)==220
assert _turn_pulse_ttl_ms(math.radians(20),3000,1,False,4)==120
print('ESCAPE_CUTOFF_REMOVAL_CHECK_OK_NO_MOTOR_COMMAND')
'''.replace('ROOT_PATH',repr(PI_ROOT))
        report['check'] = python(client,pi_python(client),code)
        report['deployed'] = True
        print(json.dumps({'deployed':True,'robot_started':False,'backup':backup,
                          'receipt':str(receipt),'check':report['check']},ensure_ascii=False))
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        receipt.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        client.close()


if __name__=='__main__':
    main()
