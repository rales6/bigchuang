"""Deploy shared preview adapters and reload idle preview; no task or tests."""
import hashlib
import json
from pathlib import Path
import shlex
import time

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote
from grasp_range_integration.tools.deploy_phase1_observation import protected_hashes

ROOT = Path(__file__).resolve().parents[2]
TRACK = '/home/dachuang/dachuang/track'
SOURCE = ROOT / 'grasp_range_integration/track_web_integration/pi'
FILES = [
    ('camera_proxy.py', TRACK+'/final_work/pi/camera_proxy.py',
     'ee76e10b5ea4b19888e4db36ee443a9e19be44ab92ea110b2d14951b15b270c2'),
    ('preview_launcher.py', TRACK+'/final_work/pi/preview_launcher.py', None),
    ('start_grab_web_shared.sh', '/home/dachuang/grab_visual/start_web.sh',
     '91793c7a98ee0fc290afe604d632eb3a4403d3065be15d90a0a6314aca2a6fcc'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def idle(client):
    markers = ('relationship_find_only_entry.py', 'run_visual_pursuit.py',
               'pursuit_session', 'hull_micro_adjust', 'bounded_motion.py',
               'region_navigation --run-job')
    rows = _remote(client, 'ps -eo pid=,args=').splitlines()
    if any(any(marker in row for marker in markers) for row in rows):
        raise RuntimeError('Robot task is active; postpone preview deployment')
    state = json.loads(_remote(client, 'curl -fsS http://127.0.0.1:8780/state'))
    managed = state.get('managed') or {}
    if managed.get('running') and managed.get('kind') != 'camera_preview':
        raise RuntimeError('Managed task is active')
    grasp = (state.get('grasp') or {}).get('data') or {}
    if grasp.get('arm_teach_exec', {}).get('active') or grasp.get('grasp', {}).get('active'):
        raise RuntimeError('Arm task is active')
    if (state.get('task_recording') or {}).get('recording'):
        raise RuntimeError('Task recording is active')
    # Preview changes do not reset or overwrite the carrying ledger.


def main():
    client = _connect(HOSTS['pi'])
    backup = TRACK+'/final_work/deployment_backups/preview_launcher_'+time.strftime('%Y%m%d_%H%M%S')
    changed = []
    report = {'backup': backup, 'files': {}, 'tests_run': False, 'vehicle_task_started': False}
    try:
        idle(client)
        with client.open_sftp() as sftp:
            before = protected_hashes(sftp)
            prepared = []
            for name, destination, expected in FILES:
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
                    if old == desired:
                        continue
                    if old is not None:
                        with sftp.open(backup+'/'+name+'.old', 'wb') as f:
                            f.write(old)
                    staged = destination+'.camera-sharing-new'
                    with sftp.open(staged, 'wb') as f:
                        f.write(desired)
                    sftp.chmod(staged, sftp.stat(destination).st_mode & 0o777 if old is not None else 0o644)
                    if _read(sftp, staged) != desired:
                        raise RuntimeError('Upload checksum mismatch: '+destination)
                    if old is not None and _read(sftp, destination) != old:
                        raise RuntimeError('Remote file changed during upload: '+destination)
                    sftp.posix_rename(staged, destination)
                    changed.append((destination, old))
                for name, destination, old, desired in prepared:
                    if _read(sftp, destination) != desired:
                        raise RuntimeError('Deployed checksum mismatch: '+destination)
                    report['files'][destination] = sha(desired)
                if protected_hashes(sftp) != before:
                    raise RuntimeError('Protected core changed during deployment')
            except Exception:
                for destination, old in reversed(changed):
                    if old is None:
                        sftp.remove(destination)
                    else:
                        staged = destination+'.camera-sharing-rollback'
                        with sftp.open(staged, 'wb') as f:
                            f.write(old)
                        sftp.chmod(staged, sftp.stat(destination).st_mode & 0o777)
                        sftp.posix_rename(staged, destination)
                raise
            report['frozen_core_unchanged'] = True
        idle(client)
        code = 'from final_work.pi.motion_session import _stop_preview,_restore_preview; _stop_preview(); _restore_preview()'
        output = _remote(client, 'cd '+shlex.quote(TRACK)+' && '+shlex.quote(HOSTS['pi']['python'])+' -c '+shlex.quote(code))
        print(output)
        report['preview_reloaded'] = True
        Path(ROOT/'diagnostics/preview_launcher_deployment.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))
    finally:
        client.close()


if __name__ == '__main__':
    main()
