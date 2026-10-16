"""Deploy audited task adapters and recording, never launch a robot task."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote
from grasp_range_integration.tools.deploy_carry_handoff import idle, optional, SHANGHAI, RELEASE, TRACK, CAR
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes
from grasp_range_integration.tools.validate_live_observation import MODULES

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'diagnostics/latest_repair_audit/manifest.json'
REMOTE_TESTS=tuple(n for n in MODULES if n not in {
    'test_robot_skills','test_track_web_phase1_pick','test_phase1_pursuit_entry_contract'})


def sha(data):return hashlib.sha256(data).hexdigest() if data is not None else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy',action='store_true')
    args=parser.parse_args()
    rows=json.loads(MANIFEST.read_text(encoding='utf-8'))
    clients={};prepared=[];activated=[];stopped=[]
    stamp=datetime.now(SHANGHAI).strftime('%Y%m%d_%H%M%S')
    backups={side:HOSTS[side]['root']+'/final_work/deployment_backups/latest_task_repair_'+stamp
             for side in ('pi','gpu')}
    try:
        clients={side:_connect(HOSTS[side]) for side in ('pi','gpu')}
        idle(clients['pi'])
        with clients['pi'].open_sftp() as s:protected=protected_hashes(s)
        for row in rows:
            side,target=row['side'],row['remote']
            local=(ROOT/row['local']).resolve()
            allowed=(RELEASE+'/grasp_range_integration/phase1/',TRACK+'/final_work/pi/') if side=='pi' else (HOSTS[side]['root']+'/final_work/gpu/',)
            if not any(target.startswith(prefix) for prefix in allowed) or not local.is_relative_to(ROOT/'grasp_range_integration'):
                raise RuntimeError('Manifest target outside adapter scope')
            with clients[side].open_sftp() as s:old=optional(s,target)
            new=local.read_bytes()
            if sha(old) not in {row['expected'],sha(new)}:raise RuntimeError('Concurrent remote edit: '+target)
            prepared.append((side,target,old,new))
        # Updated regression files only; originals are backed up too.
        for name in REMOTE_TESTS:
            target=RELEASE+'/grasp_range_integration/tests/'+name+'.py'
            with clients['pi'].open_sftp() as s:old=optional(s,target)
            prepared.append(('pi',target,old,(ROOT/'grasp_range_integration/tests'/f'{name}.py').read_bytes()))
        if not args.deploy:
            print(json.dumps(dict(audited=len(prepared),hardware_started=False)));return
        for i,(side,target,old,new) in enumerate(prepared):
            with clients[side].open_sftp() as s:
                _mkdir(s,backups[side])
                if old is not None:
                    with s.open(backups[side]+f'/{i}.original','wb') as f:f.write(old)
                with s.open(backups[side]+f'/{i}.staged','wb') as f:f.write(new)
                if sha(_read(s,backups[side]+f'/{i}.staged'))!=sha(new):raise RuntimeError('Staging mismatch')
        # Static syntax checks before activation (no hardware access).
        for side in clients:
            staged=[shlex.quote(backups[side]+f'/{i}.staged') for i,(host,target,_,_) in enumerate(prepared)
                    if host==side and target.endswith('.py')]
            _remote(clients[side],shlex.quote(HOSTS[side]['python'])+' -m py_compile '+' '.join(staged))
        for i,(side,target,old,new) in enumerate(prepared):
            if target.endswith('.sh'):_remote(clients[side],'bash -n '+shlex.quote(backups[side]+f'/{i}.staged'))
        idle(clients['pi'])
        _remote(clients['gpu'],'bash /home/gpu/track/final_work/stop_3090.sh');stopped.append('gpu')
        _remote(clients['pi'],'sudo -n systemctl stop final-work-pi-bridge.service');stopped.append('pi')
        for i,(side,target,old,new) in enumerate(prepared):
            with clients[side].open_sftp() as s:
                if optional(s,target)!=old:raise RuntimeError('Concurrent edit: '+target)
                _mkdir(s,str(PurePosixPath(target).parent))
                s.posix_rename(backups[side]+f'/{i}.staged',target)
                activated.append((i,side,target,old))
                if sha(_read(s,target))!=sha(new):raise RuntimeError('Activated hash mismatch')
        # Tests use installed Pi adapters, not a stale track/ core snapshot.
        # A pure GPU helper is copied only to the backup test sandbox on Pi.
        with clients['pi'].open_sftp() as s:
            _mkdir(s,backups['pi']+'/gpu')
            with s.open(backups['pi']+'/gpu/task_recording.py','wb') as f:
                f.write((ROOT/'grasp_range_integration/track_web_integration/gpu/task_recording.py').read_bytes())
        code='''import importlib,sys,types,unittest
parent=types.ModuleType('grasp_range_integration.track_web_integration');parent.__path__=[]
sys.modules[parent.__name__]=parent
pi=types.ModuleType(parent.__name__+'.pi');pi.__path__=['/home/robot/dachuang/track/final_work/pi']
sys.modules[pi.__name__]=pi
gpu=types.ModuleType(parent.__name__+'.gpu');gpu.__path__=[sys.argv[1]]
sys.modules[gpu.__name__]=gpu
suite=unittest.defaultTestLoader.loadTestsFromName('grasp_range_integration.tests.'+sys.argv[2])
result=unittest.TextTestRunner(verbosity=0).run(suite)
print('SOFTWARE_TEST',sys.argv[2],result.testsRun,result.wasSuccessful(),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
'''
        prefix='cd '+shlex.quote(CAR)+' && PYTHONPATH='+shlex.quote(CAR+':'+RELEASE+':'+TRACK)+' '+shlex.quote(HOSTS['pi']['python'])
        results=[]
        for name in REMOTE_TESTS:
            command=(prefix+' -c '+shlex.quote(code)+' '+shlex.quote(backups['pi']+'/gpu')+' '+shlex.quote(name)
                     if name in {'test_task_video','test_pursuit_loss_recovery'} else
                     prefix+' -m unittest grasp_range_integration.tests.'+name+' -q')
            output=_remote(clients['pi'],command)
            print(output.strip(),flush=True);results.append(output.strip())
        _remote(clients['gpu'],'cd /home/gpu/track && PYTHONPATH=/home/gpu/track '+
                shlex.quote(HOSTS['gpu']['python'])+" -c 'import final_work.gpu.app;print(\"GPU_APP_IMPORT_OK\")'")
        with clients['pi'].open_sftp() as s:
            if protected_hashes(s)!=protected:raise RuntimeError('Protected core changed')
        _remote(clients['pi'],'sudo -n systemctl start final-work-pi-bridge.service')
        _remote(clients['pi'],'sudo -n systemctl start final-work-pi-tunnel.service')
        _remote(clients['gpu'],'bash /home/gpu/track/final_work/run_3090_background.sh')
        _remote(clients['pi'],'systemctl is-active final-work-pi-bridge.service final-work-pi-tunnel.service')
        # Health/read-only status checks never call recording/start or a task.
        _remote(clients['pi'],'curl --retry 6 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8780/health >/dev/null')
        idle(clients['pi']);stopped.clear()
        report=dict(backups=backups,hardware_started=False,protected_unchanged=True,software_tests=results,
                    files={target:sha(new) for _,target,_,new in prepared},services_reloaded=True)
        for side in clients:
            with clients[side].open_sftp() as s:
                with s.open(backups[side]+'/deployment.json','w') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2))
        (ROOT/'diagnostics/latest_task_repair_deployment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(backups=backups,files=len(prepared),services_reloaded=True,hardware_started=False)))
    except BaseException:
        for i,side,target,old in reversed(activated):
            with clients[side].open_sftp() as s:
                s.posix_rename(target,backups[side]+f'/{i}.failed')
                if old is not None:s.posix_rename(backups[side]+f'/{i}.original',target)
        if 'pi' in stopped:
            _remote(clients['pi'],'sudo -n systemctl start final-work-pi-bridge.service')
            _remote(clients['pi'],'sudo -n systemctl start final-work-pi-tunnel.service')
        if 'gpu' in stopped:_remote(clients['gpu'],'bash /home/gpu/track/final_work/run_3090_background.sh')
        raise
    finally:
        for c in clients.values():c.close()


if __name__=='__main__':main()
