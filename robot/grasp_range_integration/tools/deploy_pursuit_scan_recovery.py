"""Deploy the pursuit reacquisition adapters only; never start a robot task."""
import hashlib
import json
from pathlib import Path
import time

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes
from grasp_range_integration.tools.deploy_preview_camera_sharing import idle

ROOT = Path(__file__).resolve().parents[2]
TRACK = '/home/dachuang/dachuang/track'
SOURCE = ROOT/'grasp_range_integration/track_web_integration/pi'
FILES = {
    'pursuit_recovery.py': '8351b32fb6f8391eabe229d68b74d1d7444c0ebebce0d410a17f49e7bc95ecd8',
    'pursuit_session.py': '558260f3255dd5aade0c6a36bfae5f6b6a39c642570b89627f0a620831a31156',
    'pursuit_scan_recovery.py': None,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    client = _connect(HOSTS['pi'])
    backup = TRACK+'/final_work/deployment_backups/pursuit_scan_recovery_'+time.strftime('%Y%m%d_%H%M%S')
    report = dict(backup=backup, files={}, vehicle_task_started=False, services_restarted=False)
    changed = []
    try:
        idle(client)
        with client.open_sftp() as sftp:
            protected = protected_hashes(sftp)
            # Include frozen pursuit and grab code, beyond the navigation core.
            frozen = [TRACK+'/pursuit/run_visual_pursuit.py', TRACK+'/pursuit/pursuit_controller.py',
                      '/home/dachuang/grab_visual/web_app.py']
            frozen_before = {p: sha(_read(sftp, p)) for p in frozen}
            prepared = []
            for name, expected in FILES.items():
                destination = TRACK+'/final_work/pi/'+name
                desired = (SOURCE/name).read_bytes().replace(b'\r\n', b'\n')
                try:
                    old = _read(sftp, destination)
                except FileNotFoundError:
                    old = None
                if old != desired and (sha(old) if old is not None else None) != expected:
                    raise RuntimeError('Unexpected remote edit: '+destination)
                prepared.append((name, destination, old, desired))
            _mkdir(sftp, backup)
            try:
                for name, destination, old, desired in prepared:
                    if old != desired:
                        if old is not None:
                            with sftp.open(backup+'/'+name+'.old', 'wb') as f:f.write(old)
                        staged = destination+'.reacquire-new'
                        with sftp.open(staged, 'wb') as f:f.write(desired)
                        sftp.chmod(staged, 0o644)
                        if _read(sftp, staged) != desired:raise RuntimeError('Upload mismatch')
                        if old is not None and _read(sftp, destination) != old:raise RuntimeError('Concurrent remote edit')
                        sftp.posix_rename(staged, destination)
                        changed.append((destination, old))
                    if _read(sftp, destination) != desired:raise RuntimeError('Deployment mismatch')
                    report['files'][destination] = sha(desired)
                if protected_hashes(sftp) != protected or any(sha(_read(sftp,p)) != h for p,h in frozen_before.items()):
                    raise RuntimeError('Frozen files changed')
            except Exception:
                for destination, old in reversed(changed):
                    if old is None:sftp.remove(destination)
                    else:
                        staged=destination+'.reacquire-rollback'
                        with sftp.open(staged,'wb') as f:f.write(old)
                        sftp.chmod(staged,0o644);sftp.posix_rename(staged,destination)
                raise
            report['frozen_core_unchanged'] = True
            with sftp.open(backup+'/deployment.json','w') as f:
                f.write(json.dumps(report,ensure_ascii=False,indent=2))
        (ROOT/'diagnostics/pursuit_scan_recovery_deployment.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))
    finally:
        client.close()


if __name__=='__main__':main()
