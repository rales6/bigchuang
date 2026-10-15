"""Deploy explicit empty-gripper metadata reset; no chassis/arm commands."""
import hashlib
import json
from pathlib import Path
import shlex
import time

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote
from grasp_range_integration.tools.deploy_preview_camera_sharing import idle
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'grasp_range_integration/track_web_integration'
FILES=[
 ('pi','pi/robot_skills.py','/home/dachuang/dachuang/track/final_work/pi/robot_skills.py','f6247a6627d5b3dc25c8fcc5f6b633341f7c8233835f7a5a755b4c93e1401d59'),
 ('pi','pi/robot_bridge.py','/home/dachuang/dachuang/track/final_work/pi/robot_bridge.py','9c3d0299d6c3b3d280f47b832cbe9eed523a04e54da57efcd52742fdd5f9f73b'),
 ('gpu','gpu/app.py','/home/sjtu/track/final_work/gpu/app.py','01c5240688dc31df1cdb5b3afdfeb9346594859bdcbd39bce7c359a24e501959'),
 ('gpu','static/empty-gripper.js','/home/sjtu/track/final_work/static/empty-gripper.js',None),
]


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    clients={};changed=[];stopped=[]
    stamp=time.strftime('%Y%m%d_%H%M%S')
    report={'backups':{},'files':{},'vehicle_task_started':False,'metadata_reset_requested':False}
    try:
        clients={side:_connect(HOSTS[side]) for side in ('pi','gpu')}
        idle(clients['pi'])
        prepared=[]
        with clients['pi'].open_sftp() as s:
            before=protected_hashes(s)
            ledger=_read(s,'/home/dachuang/dachuang/track/final_work/data/robot_skills_state.json')
        for side,name,destination,expected in FILES:
            with clients[side].open_sftp() as s:
                try:old=_read(s,destination)
                except FileNotFoundError:old=None
            desired=(SOURCE/name).read_bytes().replace(b'\r\n',b'\n')
            if old!=desired and (sha(old) if old is not None else None)!=expected:
                raise RuntimeError('Unexpected remote edit: '+destination)
            prepared.append((side,name,destination,old,desired))
        for side,client in clients.items():
            backup=HOSTS[side]['root']+'/final_work/deployment_backups/empty_gripper_'+stamp
            report['backups'][side]=backup
            with client.open_sftp() as s:_mkdir(s,backup)
        with clients['pi'].open_sftp() as s:
            with s.open(report['backups']['pi']+'/robot_skills_state.before.json','wb') as f:f.write(ledger)
        for i,(side,name,destination,old,desired) in enumerate(prepared):
            with clients[side].open_sftp() as s:
                if old is not None:
                    with s.open(report['backups'][side]+f'/{i}.old','wb') as f:f.write(old)
                with s.open(report['backups'][side]+f'/{i}.staged','wb') as f:f.write(desired)
        idle(clients['pi'])
        _remote(clients['gpu'],'bash /home/sjtu/track/final_work/stop_3090.sh')
        stopped.append('gpu')
        # Earlier deployments had stale PID files. Require the actual HTTP
        # listener to exit; target only the verified final_work web process.
        stop_code="""from pathlib import Path
import os,signal,time
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:a=(p/'cmdline').read_bytes().split(b'\\0')
 except OSError:continue
 if b'final_work.gpu.app' in a and b'-m' in a:
  os.kill(int(p.name),signal.SIGTERM)
  for _ in range(30):
   if not p.exists():break
   time.sleep(.1)
"""
        _remote(clients['gpu'],HOSTS['gpu']['python']+' -c '+shlex.quote(stop_code))
        _remote(clients['pi'],'sudo -n systemctl stop final-work-pi-bridge.service');stopped.append('pi')
        for i,(side,name,destination,old,desired) in enumerate(prepared):
            with clients[side].open_sftp() as s:
                if old is not None and _read(s,destination)!=old:raise RuntimeError('Concurrent edit')
                staged=report['backups'][side]+f'/{i}.staged'
                s.chmod(staged,0o644);s.posix_rename(staged,destination)
                changed.append((side,destination,old))
                if _read(s,destination)!=desired:raise RuntimeError('Deployment hash mismatch')
                report['files'][destination]=sha(desired)
        with clients['pi'].open_sftp() as s:
            if protected_hashes(s)!=before:raise RuntimeError('Protected core changed')
        _remote(clients['pi'],'sudo -n systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service')
        _remote(clients['gpu'],'bash /home/sjtu/track/final_work/run_3090_background.sh')
        _remote(clients['pi'],'curl --retry 6 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8780/health >/dev/null')
        _remote(clients['gpu'],'curl --retry 6 --retry-connrefused --retry-delay 1 -fsS http://127.0.0.1:8030/api/conversations >/dev/null')
        stopped.clear()
        report['frozen_core_unchanged']=True
        report['services_reloaded']=True
        (ROOT/'diagnostics/empty_gripper_deployment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))
    except BaseException:
        for side,destination,old in reversed(changed):
            with clients[side].open_sftp() as s:
                if old is None:s.remove(destination)
                else:
                    staged=destination+'.reset-rollback'
                    with s.open(staged,'wb') as f:f.write(old)
                    s.chmod(staged,0o644);s.posix_rename(staged,destination)
        if 'pi' in stopped:_remote(clients['pi'],'sudo -n systemctl start final-work-pi-bridge.service final-work-pi-tunnel.service')
        if 'gpu' in stopped:_remote(clients['gpu'],'bash /home/sjtu/track/final_work/run_3090_background.sh')
        raise
    finally:
        for c in clients.values():c.close()


if __name__=='__main__':main()
