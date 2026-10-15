"""Narrow audited carry-handoff deployment; never starts robot motion."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
from datetime import datetime, timezone, timedelta
SHANGHAI = timezone(timedelta(hours=8))

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes

ROOT = Path(__file__).resolve().parents[2]
CAR = '/home/dachuang/dachuang/car1.0'
RELEASE = CAR+'/ble_scene_17_relationship_20260919'
TRACK = '/home/dachuang/dachuang/track'
FILES = [
    ('grasp_range_integration/track_web_integration/pi/region_navigation.py',
     TRACK+'/final_work/pi/region_navigation.py',
     '966fd29a45913aed7e6b1faaa0f12129a473fd2127687573af1bc9ca28b62b40'),
    ('grasp_range_integration/track_web_integration/pi/carry_localization.py',
     TRACK+'/final_work/pi/carry_localization.py',None),
    ('grasp_range_integration/phase1/startup_active_recovery.py',
     RELEASE+'/grasp_range_integration/phase1/startup_active_recovery.py',
     '0e59356d6d0f86907b0fb8da85d6d5ab7c8494bf582bdb9cfab39e46974b5eb0'),
    ('grasp_range_integration/phase1/startup_route_recovery.py',
     RELEASE+'/grasp_range_integration/phase1/startup_route_recovery.py',
     '8c449e5a9ea20319495d9ec65be650b52e56a40ce17ff6ec64dac26bb75791af'),
]


def optional(sftp,path):
    try:return _read(sftp,path)
    except FileNotFoundError:return None


def sha(data):return hashlib.sha256(data).hexdigest() if data is not None else None


def idle(client, *, ack_empty=False):
    markers=('relationship_find_only_entry.py','run_visual_pursuit.py','pursuit_session',
             'hull_micro_adjust','bounded_motion.py','region_navigation --run-job')
    active=[row for row in _remote(client,'ps -eo pid=,args=').splitlines()
            if any(marker in row for marker in markers)]
    if active:raise RuntimeError('Active robot task; deployment refused: '+repr(active))
    state=json.loads(_remote(client,'curl -fsS http://127.0.0.1:8780/state'))
    managed=state.get('managed') or {}
    if managed.get('running') and managed.get('kind')!='camera_preview':
        raise RuntimeError('Bridge has active managed task')
    if ((state.get('grasp') or {}).get('data') or {}).get('arm_teach_exec',{}).get('active'):
        raise RuntimeError('Arm task still active')
    if (state.get('robot_skills') or {}).get('status') in ('carrying','placing','placement_unknown') and not ack_empty:
        raise RuntimeError('User confirmation of physical reset required before clearing old carry ledger')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy',action='store_true')
    parser.add_argument('--ack-empty',action='store_true',help='Human explicitly confirmed robot reset and empty claw')
    args=parser.parse_args()
    c=_connect(HOSTS['pi'])
    activated=[]
    try:
        state=idle(c,ack_empty=args.ack_empty)
        stamp=datetime.now(SHANGHAI).strftime('%Y%m%d_%H%M%S')
        backup=TRACK+'/final_work/deployment_backups/carry_handoff_'+stamp
        with c.open_sftp() as s:
            protected=protected_hashes(s)
            prepared=[]
            for local,remote,expected in FILES:
                old=optional(s,remote);new=(ROOT/local).read_bytes()
                if sha(old) not in {expected,sha(new)}:
                    raise RuntimeError('Unexpected remote version: '+remote)
                prepared.append((remote,old,new))
            if not args.deploy:
                print(json.dumps({'audited':[p for p,_,_ in prepared],'hardware_started':False}));return
            _mkdir(s,backup)
            for index,(remote,old,new) in enumerate(prepared):
                if old is not None:
                    with s.open(backup+f'/{index}.original','wb') as f:f.write(old)
                with s.open(backup+f'/{index}.staged.py','wb') as f:f.write(new)
                if sha(_read(s,backup+f'/{index}.staged.py'))!=sha(new):raise RuntimeError('Staging mismatch')
            test=backup+'/test_carry_localization.py'
            with s.open(test,'wb') as f:f.write((ROOT/'grasp_range_integration/tests/test_carry_localization.py').read_bytes())
        py=HOSTS['pi']['python']
        _remote(c,shlex.quote(py)+' -m py_compile '+' '.join(shlex.quote(backup+f'/{i}.staged.py') for i in range(len(prepared))))
        idle(c,ack_empty=args.ack_empty)
        try:
            with c.open_sftp() as s:
                for index,(remote,old,new) in enumerate(prepared):
                    if optional(s,remote)!=old:raise RuntimeError('Concurrent edit: '+remote)
                    s.posix_rename(backup+f'/{index}.staged.py',remote);activated.append((index,remote,old))
            code="""import importlib,importlib.util,sys,unittest,subprocess
sys.modules['grasp_range_integration.track_web_integration.pi']=importlib.import_module('final_work.pi')
for name in ('carry_localization','region_navigation'):
 sys.modules['grasp_range_integration.track_web_integration.pi.'+name]=importlib.import_module('final_work.pi.'+name)
spec=importlib.util.spec_from_file_location('carry_regression',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
suite=unittest.defaultTestLoader.loadTestsFromModule(module)
result=unittest.TextTestRunner(verbosity=1).run(suite);print('CARRY_SOFTWARE_TESTS',result.testsRun,'PASSED',result.wasSuccessful(),flush=True)
if not result.wasSuccessful():sys.exit(1)
# Hook-install tests monkeypatch process-global classes. Match production's
# new-process startup and prevent mock controller leakage between modules.
for name in ('test_phase1_startup_active_recovery','test_phase1_startup_route_recovery','test_phase1_midrun_view_change','test_phase1_startup_drive','test_phase1_fresh_motion_feedback'):
 child=subprocess.run([sys.executable,'-m','unittest','grasp_range_integration.tests.'+name,'-q'],capture_output=True,text=True,timeout=25)
 print(name,'EXIT',child.returncode,child.stderr[-1800:],flush=True)
 if child.returncode:sys.exit(child.returncode)
"""
            # Match the mission cwd: track/ also has an old raspberry_pi
            # snapshot, which must not shadow the frozen car1.0 runtime.
            cmd='cd '+shlex.quote(CAR)+' && PYTHONPATH='+shlex.quote(CAR+':'+RELEASE+':'+TRACK)+' '+shlex.quote(py)+' -c '+shlex.quote(code)+' '+shlex.quote(test)
            print(_remote(c,cmd))
            with c.open_sftp() as s:
                if protected_hashes(s)!=protected:raise RuntimeError('Frozen core changed')
                for remote,old,new in prepared:
                    if sha(_read(s,remote))!=sha(new):raise RuntimeError('Deployed hash mismatch')
                ledger=TRACK+'/final_work/data/robot_skills_state.json'
                if args.ack_empty:
                    idle(c,ack_empty=True)
                    previous=optional(s,ledger)
                    if previous is not None:
                        with s.open(backup+'/carry_state.original.json','wb') as f:f.write(previous)
                    cleared={'status':'empty','reason':'human_confirmed_full_reset_before_new_test',
                             'reset_at':datetime.now(SHANGHAI).isoformat(),
                             'previous_state_backup':backup+'/carry_state.original.json'}
                    with s.open(backup+'/carry_state.staged.json','wb') as f:f.write(json.dumps(cleared).encode())
                    s.posix_rename(backup+'/carry_state.staged.json',ledger)
            # Read new ledger in the service; no HOME/release/motion API calls.
            if args.ack_empty:
                print(_remote(c,'sudo -n systemctl restart final-work-pi-bridge.service'))
                print(_remote(c,'sudo -n systemctl restart final-work-pi-tunnel.service'))
            record={'backup':backup,'files':{p:sha(n) for p,o,n in prepared},
                    'protected_sha256':protected,'software_tests_passed':True,'hardware_started':False}
            with c.open_sftp() as s:
                with s.open(backup+'/DEPLOYMENT_RECORD.json','wb') as f:f.write(json.dumps(record,indent=2).encode())
            print(json.dumps(record,ensure_ascii=False))
        except Exception:
            with c.open_sftp() as s:
                for index,remote,old in reversed(activated):
                    s.posix_rename(remote,backup+f'/{index}.failed.py')
                    if old is not None:s.posix_rename(backup+f'/{index}.original',remote)
            raise
    finally:c.close()


if __name__=='__main__':main()
