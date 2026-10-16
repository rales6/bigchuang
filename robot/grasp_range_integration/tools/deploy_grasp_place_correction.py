"""Back up and activate the audited S0 offset and pre-lower BLE handoff.

No chassis/arm task is started. Only idle bridge/preview services reload.
The user's pulse-offset choice belongs in start_grab_web_shared.sh; this
tool refuses unaudited remote edits and preserves carry/video records.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import time

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote
from grasp_range_integration.tools.deploy_preview_camera_sharing import idle
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'grasp_range_integration/track_web_integration/pi'
TRACK = '/home/robot/dachuang/track'
GRAB = '/home/robot/grab_visual'
ARM_HASH = '545786dc33865efb5b57c0fcdd582159634d14c198b4de87f3a2af72a14b2a63'
EXPECTED = {
    TRACK+'/final_work/pi/robot_skills.py': '9bd9091afa5d8a162a09eae973c389a5c2fd2dd26395a947bba027d061e88b92',
    TRACK+'/final_work/pi/placement_extension.py': '7cc45a104966a0481e771ab82a080e87ec33224281584a69bf498decab9670da',
    GRAB+'/placement_extension.py': '7cc45a104966a0481e771ab82a080e87ec33224281584a69bf498decab9670da',
    GRAB+'/arm_teach.py': ARM_HASH,
    GRAB+'/grasp_pose_correction.py': None,
    TRACK+'/final_work/pi/grasp_pose_correction.py': None,
    GRAB+'/start_web.sh': '21a1baa95d25514d629b0f658195fef4de328ba937218d6e272b4b62730abda3',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def render_arm_patch(original):
    start = original.index('def _s2_preclose_nudge_target(target: dict) -> dict | None:')
    end = original.index('def _run_grasp_sequence(', start)
    helper = '''def _s2_preclose_nudge_target(target: dict) -> dict | None:
    """Apply operator-calibrated S0 and mirrored S2/S3 offsets once."""
    from grasp_pose_correction import corrected_grasp_pose
    return corrected_grasp_pose(target, SERVOS)


'''
    desired = original[:start] + helper + original[end:]
    old_message = 'message=f"Model pose reached; s2/s3 pre-close nudge to s2={nudge_target[\'s2\']}"'
    new_message = 'message=f"Model pose reached; pre-close correction s0={nudge_target[\'s0\']} s2={nudge_target[\'s2\']}"'
    if desired.count(old_message) != 1:
        raise RuntimeError('Pre-close message anchor changed')
    desired = desired.replace(old_message, new_message, 1)
    # Everything outside the one pure helper and its display string stays
    # byte-for-byte identical (movement sequence, model and servo limits).
    ast.parse(desired)
    return desired


REMOTE_TEST = '''import ast,importlib,importlib.util,json,math,re,sys,tempfile,threading,types,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
import grasp_range_integration.track_web_integration
import final_work.pi
prefix='grasp_range_integration.track_web_integration.pi'
sys.modules[prefix]=final_work.pi
for name in ('robot_skills','region_navigation','placement_extension','grasp_pose_correction'):
    sys.modules[prefix+'.'+name]=importlib.import_module('final_work.pi.'+name)
from final_work.pi.robot_skills import RobotSkills
from final_work.pi.placement_extension import run_placement,validate_pose
from final_work.pi.region_navigation import save_json
# Load only the hardware-free placement/carry tests, not GPU compiler tests.
tree=ast.parse(Path(sys.argv[1]).read_text())
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SkillTests')
exec(compile(ast.Module(body=[node],type_ignores=[]),sys.argv[1],'exec'),globals())
names=['test_placement_order_and_preserve_mirrored_nudge','test_cancel_before_lower_does_not_release',
 'test_bad_pose_not_clamped_silently','test_stale_ble_recovers_with_ping_before_any_arm_motion',
 'test_link_timeout_is_proven_pre_lower_retryable','test_cancel_during_handoff_never_moves_or_authorises_retry',
 'test_lowering_fault_and_failed_stop_do_not_release_or_escape_queue',
 'test_swallowed_heartbeat_fault_prevents_release','test_placement_error_reconciles_carry_only_before_lower',
 'test_photo_io_error_does_not_abort_or_repeat_successful_release',
 'test_place_waits_handoff_and_pre_lower_retry_has_a_new_id',
 'test_persisted_pre_lower_failure_can_be_read_without_repeating_motion',
 'test_persisted_lower_or_release_remains_unknown_on_service_restart',
 'test_placement_unknown_does_not_reissue_release',
 'test_arrival_requires_current_job_visual_and_floor_clearance',
 'test_active_or_unknown_placement_still_requires_reconciliation']
suite=unittest.TestSuite(SkillTests(name) for name in names)
spec=importlib.util.spec_from_file_location('pose_tests',sys.argv[2]); module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module); suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
result=unittest.TextTestRunner(verbosity=1).run(suite)
if not result.wasSuccessful():raise SystemExit(1)
# Execute just the actual deployed grasp function with FAKE arm methods.
arm_tree=ast.parse(Path('/home/robot/grab_visual/arm_teach.py').read_text())
nodes=[n for n in arm_tree.body if isinstance(n,ast.FunctionDef) and n.name in
 ('_s2_preclose_nudge_target','_run_grasp_sequence')]
assert len(nodes)==2
calls=[]; status={}
servos=[{'key':'s0','min':500,'max':2500,'home':1500},
 {'key':'s1','min':800,'max':1700,'home':1700},{'key':'s2','min':1500,'max':2200,'home':2000}]
ns={'SERVOS':servos,'SERVO_KEYS':('s0','s1','s2'),'_state_lock':threading.RLock(),
 '_exec_status':status,'CLAW_CLOSE':0,'_heartbeat_wait':lambda *a:None,
 'send_home':lambda *a:calls.append(('home',None)) or 'OK',
 'send_grasp_pose':lambda send,p:calls.append(('pose',dict(p))),
 'send_claw':lambda send,p:calls.append(('claw',p))}
sys.path.insert(0,'/home/robot/grab_visual')
exec(compile(ast.Module(body=nodes,type_ignores=[]),'deployed_grasp_function','exec'),ns)
with patch.dict('os.environ',{'ARM_GRASP_S0_NUDGE_US':'-30','ARM_GRASP_S2_NUDGE_US':'-20'}):
 ns['_run_grasp_sequence'](lambda *a:'OK',{'s0':1410,'s1':969,'s2':1720})
assert status['phase']=='done' and status['params']=={'s0':1380,'s1':969,'s2':1700},status
assert calls==[('home',None),('pose',{'s0':1410,'s1':969,'s2':1720}),
 ('pose',{'s0':1380,'s1':969,'s2':1700}),('claw',0),('pose',{'s0':1500,'s1':1700,'s2':2000})],calls
print('deployed_mocked_tests_passed='+str(result.testsRun+1))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    client = _connect(HOSTS['pi'])
    backup = TRACK+'/final_work/deployment_backups/grasp_place_'+time.strftime('%Y%m%d_%H%M%S')
    prepared, changed = [], []
    stopped = False
    report = {'backup': backup, 'files': {}, 'vehicle_task_started': False,
              'arm_motion_started_during_deployment': False, 'recording_started': False}
    try:
        idle(client)
        arm = json.loads(_remote(client, 'curl -fsS http://127.0.0.1:8080/api/arm-teach/exec'))
        queue = json.loads(_remote(client, 'curl -fsS http://127.0.0.1:8080/api/arm-teach/state'))
        if (arm.get('status') or {}).get('active') or queue.get('pending'):
            raise RuntimeError('Arm busy; no deployment')
        with client.open_sftp() as s:
            protected = protected_hashes(s)
            frozen = {p: sha(_read(s, p)) for p in (
                TRACK+'/pursuit/hull_micro_adjust.py', GRAB+'/web_app.py')}
            ledger_path = TRACK+'/final_work/data/robot_skills_state.json'
            ledger = _read(s, ledger_path)
            video_root = TRACK+'/final_work/task_videos/latest'
            video_stats = {name: (s.stat(video_root+'/'+name).st_size,
                                  s.stat(video_root+'/'+name).st_mtime)
                           for name in ('left.avi','right.avi','session.json','timeline.jsonl')}
            for destination, expected in EXPECTED.items():
                try: old = _read(s, destination)
                except FileNotFoundError: old = None
                name = destination.rsplit('/', 1)[-1]
                if name == 'arm_teach.py':
                    if old is None or sha(old) != ARM_HASH:
                        raise RuntimeError('Arm source changed; re-audit before patching')
                    desired = render_arm_patch(old.decode()).encode()
                else:
                    local_name = 'start_grab_web_shared.sh' if name == 'start_web.sh' else name
                    desired = (SOURCE/local_name).read_bytes().replace(b'\r\n',b'\n')
                if old != desired and (sha(old) if old is not None else None) != expected:
                    raise RuntimeError('Unexpected remote edit: '+destination)
                stage = backup+'/'+str(len(prepared))+'_'+name
                prepared.append((destination, old, desired, stage))
            if not args.deploy:
                print(json.dumps({'checked': [p[0] for p in prepared], 'activated': False}))
                return
            _mkdir(s, backup)
            with s.open(backup+'/carry_ledger.before.json','wb') as f: f.write(ledger)
            for destination, old, desired, stage in prepared:
                if old is not None:
                    with s.open(stage+'.old','wb') as f: f.write(old)
                with s.open(stage+'.staged','wb') as f: f.write(desired)
                if _read(s,stage+'.staged') != desired: raise RuntimeError('Staging mismatch')
            for name in ('test_robot_skills.py','test_grasp_pose_correction.py'):
                with s.open(backup+'/'+name,'wb') as f:
                    f.write((ROOT/'grasp_range_integration/tests'/name).read_bytes())
        code = ('import ast; from pathlib import Path; [ast.parse(p.read_text()) for p in Path('+repr(backup)+
                ").glob('*.py.staged')]; print('syntax_ok')")
        _remote(client, HOSTS['pi']['python']+' -c '+shlex.quote(code))
        _remote(client, 'bash -n '+shlex.quote(next(p[3]+'.staged' for p in prepared if p[0].endswith('start_web.sh'))))
        idle(client)
        _remote(client, 'sudo -n systemctl stop final-work-pi-bridge.service')
        stopped = True
        _remote(client, 'cd '+shlex.quote(TRACK)+' && '+HOSTS['pi']['python']+' -c '+
                shlex.quote('from final_work.pi.motion_session import _stop_preview; _stop_preview()'))
        with client.open_sftp() as s:
            for destination, old, desired, stage in prepared:
                try: now = _read(s, destination)
                except FileNotFoundError: now = None
                if now != old: raise RuntimeError('Concurrent edit: '+destination)
                s.chmod(stage+'.staged', 0o755 if destination.endswith('.sh') else 0o644)
                s.posix_rename(stage+'.staged',destination)
                changed.append((destination, old))
                if _read(s,destination) != desired: raise RuntimeError('Activation mismatch')
                report['files'][destination]=sha(desired)
            if protected_hashes(s) != protected or any(sha(_read(s,p)) != digest for p,digest in frozen.items()):
                raise RuntimeError('Protected core changed')
            if _read(s,ledger_path) != ledger: raise RuntimeError('Carry ledger changed')
        env = 'PYTHONPATH='+shlex.quote(TRACK+':'+GRAB+':/home/robot/dachuang/car1.0/ble_scene_17_relationship_20260919:/home/robot/dachuang/car1.0')
        output = _remote(client, env+' '+HOSTS['pi']['python']+' -c '+shlex.quote(REMOTE_TEST)+' '+
                         shlex.quote(backup+'/test_robot_skills.py')+' '+shlex.quote(backup+'/test_grasp_pose_correction.py'))
        print(output)
        report['deployed_mocked_tests_passed']=int(re.search(r'deployed_mocked_tests_passed=(\d+)',output).group(1))
        _remote(client,'sudo -n systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service')
        _remote(client,'curl --retry 6 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8780/health >/dev/null')
        stopped = False
        _remote(client,"curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8780/actions/preview/start")
        ready_code = '''import json,time
from urllib.request import urlopen
for attempt in range(30):
 try:
  with urlopen('http://127.0.0.1:8080/api/status',timeout=2) as r:status=json.load(r)
  cameras=status.get('cameras') or {}
  if status.get('ble') is True and all((cameras.get(k) or {}).get('frames',0)>0 and 0<=float((cameras.get(k) or {}).get('age_s',99))<1 for k in ('left','right')):break
 except Exception:pass
 time.sleep(.5)
else:raise RuntimeError('Preview not ready after reload')
print('preview_ready')
'''
        _remote(client, HOSTS['pi']['python']+' -c '+shlex.quote(ready_code))
        with client.open_sftp() as s:
            if _read(s,ledger_path) != ledger: raise RuntimeError('Ledger modified during reload')
            after_stats = {name: (s.stat(video_root+'/'+name).st_size,s.stat(video_root+'/'+name).st_mtime)
                           for name in video_stats}
            if after_stats != video_stats: raise RuntimeError('Previous video changed')
        report.update(bridge_and_preview_reloaded=True, carry_ledger_preserved=True,
                      previous_stereo_recording_preserved=True, navigation_and_model_unchanged=True)
        print(json.dumps(report,ensure_ascii=False))
    except BaseException:
        if changed and not stopped:
            _remote(client,'sudo -n systemctl stop final-work-pi-bridge.service')
            _remote(client,'cd '+shlex.quote(TRACK)+' && '+HOSTS['pi']['python']+' -c '+
                    shlex.quote('from final_work.pi.motion_session import _stop_preview; _stop_preview()'))
        with client.open_sftp() as s:
            for destination,old in reversed(changed):
                if old is None:
                    s.remove(destination)  # Only new files owned by this deployment.
                else:
                    stage=destination+'.grasp-place-rollback'
                    with s.open(stage,'wb') as f:f.write(old)
                    s.chmod(stage,0o755 if destination.endswith('.sh') else 0o644)
                    s.posix_rename(stage,destination)
        if stopped or changed:
            _remote(client,'sudo -n systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service')
            _remote(client,'curl --retry 6 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8780/health >/dev/null')
            _remote(client,"curl -fsS -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8780/actions/preview/start")
        raise
    finally:
        client.close()


if __name__ == '__main__':
    main()
