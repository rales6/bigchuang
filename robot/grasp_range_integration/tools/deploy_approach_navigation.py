"""Audited region-transit adapter deployment. Never sends motion commands."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote
from grasp_range_integration.tools.deploy_carry_handoff import idle, optional, SHANGHAI, CAR, RELEASE, TRACK
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes

ROOT=Path(__file__).resolve().parents[2]
DEPLOYMENT_NAME='approach_navigation'
FILES={
    'phase1/relationship_find_only_entry.py':('594d163cbfdccf0d33749cc94d760e8dec79dd95d0e354ebb4fd1ccaeeffc754',
        '6dd52dbb03a9df15f585bcb88425cd6bfa51162c95645e4dcf42c88638ceb439'),
    'phase1/visibility_route_planner.py':('cdd6bfe2c145f05840bd8fd594ad5ce14cc99dd9cd3506704f04bca62988d4cf',
        'c26061bb6fd1432b5143748540f1e836028b285185f28b0e3e159bb1b7ff74c2'),
    'phase1/localization_handoff.py':('96ae530134b3b0c7a1cb308f5e085f3f569bd3c13f6f9a0ae82f2b65c41175a2',
        'c472f57b60bba92cc3e050915b67700e361c3989fdf94633c5d8b55318aa89c9'),
    'tests/test_phase1_approach_navigation.py':(None,
        '96787dc0f1f030c208f989cf4b4a671cc8ff34a4ff8292220e7b113fba6843fe'),
    'phase1/startup_active_recovery.py':('0872d5469823c1c75894c0bde058883ce0dab2248e03b847f0f52559eef4973e',),
    'tests/test_phase1_midrun_view_change.py':('0b99945da8e18a36152c173662f5e08c196e99096cf7096ff0be060d6878b1d6',),
}
TESTS=('test_phase1_approach_navigation','test_phase1_midrun_view_change',
       'test_phase1_bounded_navigation','test_phase1_navigation_recovery_contract',
       'test_phase1_startup_active_recovery','test_phase1_startup_route_recovery',
       'test_visibility_route_planner','test_phase1_observation_recovery',
       'test_phase1_fresh_motion_feedback','test_phase1_navigation_avoidance',
       'test_phase1_arrival_visibility')


def sha(data):return hashlib.sha256(data).hexdigest() if data is not None else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deploy',action='store_true')
    args=parser.parse_args()
    c=_connect(HOSTS['pi']);activated=[]
    backup=TRACK+'/final_work/deployment_backups/'+DEPLOYMENT_NAME+'_'+datetime.now(SHANGHAI).strftime('%Y%m%d_%H%M%S')
    try:
        idle(c)
        with c.open_sftp() as s:
            before=protected_hashes(s);prepared=[]
            for name,expected in FILES.items():
                target=RELEASE+'/grasp_range_integration/'+name
                old=optional(s,target);new=(ROOT/'grasp_range_integration'/name).read_bytes()
                if sha(old) not in {*expected,sha(new)}:raise RuntimeError('Unexpected remote version: '+target)
                prepared.append((name,target,old,new))
            if not args.deploy:
                print(json.dumps({'audited':[n for n,_,_,_ in prepared],'hardware_started':False}));return
            _mkdir(s,backup)
            for i,(name,target,old,new) in enumerate(prepared):
                if old is not None:
                    with s.open(backup+f'/{i}.original','wb') as f:f.write(old)
                with s.open(backup+f'/{i}.staged.py','wb') as f:f.write(new)
                if sha(_read(s,backup+f'/{i}.staged.py'))!=sha(new):raise RuntimeError('Staging hash mismatch')
        py=HOSTS['pi']['python']
        _remote(c,shlex.quote(py)+' -m py_compile '+' '.join(
            shlex.quote(backup+f'/{i}.staged.py') for i,(name,_,_,_) in enumerate(prepared)
            if name.endswith('.py')))
        idle(c)
        try:
            with c.open_sftp() as s:
                for i,(name,target,old,new) in enumerate(prepared):
                    if optional(s,target)!=old:raise RuntimeError('Concurrent edit: '+target)
                    _mkdir(s,str(Path(target).parent).replace('\\','/'))
                    s.posix_rename(backup+f'/{i}.staged.py',target);activated.append((i,target,old))
            # Match actual mission cwd, not track's obsolete core snapshot.
            prefix='cd '+shlex.quote(CAR)+' && PYTHONPATH='+shlex.quote(CAR+':'+RELEASE+':'+TRACK)+' '+shlex.quote(py)
            results=[]
            for name in TESTS:
                output=_remote(c,prefix+' -m unittest grasp_range_integration.tests.'+name+' -q')
                results.append({'module':name,'output':output})
            with c.open_sftp() as s:
                after=protected_hashes(s)
                if before!=after:raise RuntimeError('Protected source changed')
                for _,target,_,new in prepared:
                    if sha(_read(s,target))!=sha(new):raise RuntimeError('Activated hash mismatch')
                record={'backup':backup,'protected_unchanged':True,'hardware_started':False,
                        'files':{name:sha(new) for name,_,_,new in prepared},'software_tests':results}
                with s.open(backup+'/deployment.json','w') as f:f.write(json.dumps(record,ensure_ascii=False,indent=2))
        except BaseException:
            with c.open_sftp() as s:
                for i,target,old in reversed(activated):
                    if old is not None:
                        with s.open(backup+f'/{i}.rollback','wb') as f:f.write(old)
                        s.posix_rename(backup+f'/{i}.rollback',target)
                    else:s.remove(target)
            raise
        idle(c)
        _remote(c,'sudo -n systemctl restart final-work-pi-bridge.service')
        _remote(c,'systemctl is-active final-work-pi-bridge.service final-work-pi-tunnel.service')
        print(json.dumps(record,ensure_ascii=False,indent=2))
    finally:c.close()


if __name__=='__main__':main()
