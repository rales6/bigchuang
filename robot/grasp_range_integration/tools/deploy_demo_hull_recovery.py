"""Deploy demo carry-record handoff and fresh hull convergence retries, no motion."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shlex
import time

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote
from grasp_range_integration.tools.deploy_preview_camera_sharing import idle
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes

ROOT = Path(__file__).resolve().parents[2]
TRACK = '/home/dachuang/dachuang/track'
FILES = {
    'robot_skills.py': '7fd4b2ff3a53601fcca358419f76f46dd020dd5caaf8ada47b57dc2770243416',
    'robot_bridge.py': '64f987160be593e61c364328b7c0915acc4c06994154bbd7b2e6a0600322cbbc',
    'pursuit_recovery.py': '92d3c449584d9917039482cef1f4c11ad869ee4cdb18ed8acf038ec9efc6d972',
    'pursuit_session.py': 'fd65a63bf57d187a72cc7a46c47c8b6775c6e2b8972c30f526f6edd912eed748',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grasp-hull-recheck', action='store_true',
                        help='Deploy only pre-enqueue hull-jitter recovery and its two callers')
    args = parser.parse_args()
    selected = ({
        'robot_bridge.py': '3771838991b681639be556037a5fe784dca4ee190d9b9a925ff5fc51244f06e6',
        'pursuit_session.py': '7845c63e3a4a221ab7ad8a5024c8ed95d44e25977b0ca0d846a6312b6db86e05',
        'grasp_execution.py': '01a33a6b051b4e8f3a5007d292347ca918d95943cf65c0717f46de6d953a51ea',
    } if args.grasp_hull_recheck else FILES)
    kind = 'grasp_hull_recheck' if args.grasp_hull_recheck else 'demo_hull'
    client = _connect(HOSTS['pi'])
    backup = TRACK+'/final_work/deployment_backups/'+kind+'_'+time.strftime('%Y%m%d_%H%M%S')
    changed = []
    stopped = False
    report = {'backup': backup, 'files': {}, 'vehicle_task_started': False, 'arm_motion_started': False}
    try:
        idle(client)
        arm = json.loads(_remote(client, 'curl -fsS http://127.0.0.1:8080/api/arm-teach/exec'))
        queue = json.loads(_remote(client, 'curl -fsS http://127.0.0.1:8080/api/arm-teach/state'))
        if (arm.get('status') or {}).get('active') or queue.get('pending', 0):
            raise RuntimeError('Arm active or queue pending; postpone deployment')
        with client.open_sftp() as s:
            before = protected_hashes(s)
            frozen = {path: sha(_read(s, path)) for path in (
                TRACK+'/pursuit/hull_micro_adjust.py', '/home/dachuang/grab_visual/arm_teach.py')}
            ledger_path = TRACK+'/final_work/data/robot_skills_state.json'
            ledger = _read(s, ledger_path)
            prepared = []
            for name, expected in selected.items():
                destination = TRACK+'/final_work/pi/'+name
                old = _read(s, destination)
                desired = (ROOT/'grasp_range_integration/track_web_integration/pi'/name).read_bytes().replace(b'\r\n', b'\n')
                if old != desired and sha(old) != expected:
                    raise RuntimeError('Unexpected remote edit: '+destination)
                prepared.append((name, destination, old, desired))
            _mkdir(s, backup)
            with s.open(backup+'/robot_skills_state.before.json', 'wb') as f:
                f.write(ledger)
            for name, destination, old, desired in prepared:
                with s.open(backup+'/'+name+'.old', 'wb') as f:
                    f.write(old)
                with s.open(backup+'/'+name+'.staged', 'wb') as f:
                    f.write(desired)
            if args.grasp_hull_recheck:
                test_data = (ROOT/'grasp_range_integration/tests/test_grasp_hull_recheck.py').read_bytes()
                with s.open(backup+'/test_grasp_hull_recheck.py', 'wb') as f:
                    f.write(test_data)
        # Compile only: never import/start the hardware modules.
        compile_code = "import ast; from pathlib import Path; [ast.parse(p.read_text()) for p in Path("+repr(backup)+").glob('*.staged')]; print('syntax_ok')"
        _remote(client, HOSTS['pi']['python']+' -c '+shlex.quote(compile_code))
        idle(client)
        _remote(client, 'sudo -n systemctl stop final-work-pi-bridge.service')
        stopped = True
        with client.open_sftp() as s:
            for name, destination, old, desired in prepared:
                if _read(s, destination) != old:
                    raise RuntimeError('Concurrent remote edit: '+destination)
                staged = backup+'/'+name+'.staged'
                s.chmod(staged, 0o644)
                s.posix_rename(staged, destination)
                changed.append((destination, old))
                if _read(s, destination) != desired:
                    raise RuntimeError('Deployment checksum mismatch')
                report['files'][destination] = sha(desired)
            if protected_hashes(s) != before or any(sha(_read(s, p)) != digest for p, digest in frozen.items()):
                raise RuntimeError('Frozen core changed')
            if _read(s, ledger_path) != ledger:
                raise RuntimeError('Carry ledger changed during deployment')
        if args.grasp_hull_recheck:
            # Use deployed final_work modules, not stale car1 release copies.
            # All requests/arm/motion methods inside this suite are mocked.
            test_code = '''import importlib,importlib.util,sys,unittest
import grasp_range_integration.track_web_integration
import final_work.pi
prefix='grasp_range_integration.track_web_integration.pi'
sys.modules[prefix]=final_work.pi
for name in ('grasp_execution','robot_bridge','pursuit_session','motion_session'):
    sys.modules[prefix+'.'+name]=importlib.import_module('final_work.pi.'+name)
spec=importlib.util.spec_from_file_location('grasp_hull_recheck_tests',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(module))
if not result.wasSuccessful():raise SystemExit(1)
print('deployed_mocked_tests_passed='+str(result.testsRun))
'''
            env = 'PYTHONPATH='+shlex.quote(TRACK+':/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919:/home/dachuang/dachuang/car1.0')
            test_output = _remote(client, env+' '+HOSTS['pi']['python']+' -c '+shlex.quote(test_code)+' '+shlex.quote(backup+'/test_grasp_hull_recheck.py'))
            print(test_output)
            report['deployed_mocked_tests_passed'] = int(re.search(
                r'deployed_mocked_tests_passed=(\d+)', test_output).group(1))
        _remote(client, 'sudo -n systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service')
        _remote(client, 'curl --retry 6 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8780/health >/dev/null')
        stopped = False
        _remote(client, "curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8780/actions/preview/start")
        ready_code = '''import json,time
from urllib.request import urlopen
for attempt in range(30):
    with urlopen('http://127.0.0.1:8080/api/status',timeout=2) as response: status=json.load(response)
    cameras=status.get('cameras') or {}
    if status.get('ble') is True and all((cameras.get(side) or {}).get('frames',0)>0 and 0<=float((cameras.get(side) or {}).get('age_s') or 0)<1 for side in ('left','right')):break
    time.sleep(.5)
else:raise RuntimeError('preview cameras/BLE not ready after reload')
print('preview_ready_no_recording_started')
'''
        _remote(client, HOSTS['pi']['python']+' -c '+shlex.quote(ready_code))
        report.update(bridge_reloaded=True, frozen_core_unchanged=True, carry_ledger_preserved_until_new_pick=True)
        report['preview_and_camera_owner_restored'] = True
        # The caller saves this JSON receipt with apply_patch. Keep deployment
        # independent of the local diagnostic file's ownership or sandbox.
        print(json.dumps(report, ensure_ascii=False))
    except BaseException:
        if changed and not stopped:
            _remote(client, 'sudo -n systemctl stop final-work-pi-bridge.service')
            stopped = True
        with client.open_sftp() as s:
            for destination, old in reversed(changed):
                staged = destination+'.demo-hull-rollback'
                with s.open(staged, 'wb') as f:
                    f.write(old)
                s.chmod(staged, 0o644)
                s.posix_rename(staged, destination)
        if stopped:
            _remote(client, 'sudo -n systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service')
        raise
    finally:
        client.close()


if __name__ == '__main__':
    main()
