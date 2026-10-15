"""Deploy reviewed region/release adapters; never dispatch a robot action.

Dry-run by default. --apply --reload is for explicitly authorised service
maintenance. Holding recovery additionally requires an operator confirmation
and matching immutable web history, Pi carry ID and live arm execution pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import time

from tools.maintenance.assemble_workspace import connect

ROOT = Path(__file__).resolve().parents[2]
BASES = {'pi': '/home/dachuang/dachuang/track', 'gpu': '/home/sjtu/track'}
PYTHONS = {'pi': '/home/dachuang/dachuang/.venv/bin/python', 'gpu': '/home/sjtu/3090/.venv/bin/python'}
LEDGER = BASES['pi'] + '/final_work/data/robot_skills_state.json'
FILES = [
    ('pi', 'pi/robot_skills.py', BASES['pi'] + '/final_work/pi/robot_skills.py'),
    ('pi', 'pi/region_navigation.py', BASES['pi'] + '/final_work/pi/region_navigation.py'),
    ('pi', 'pi/region_localization.py', BASES['pi'] + '/final_work/pi/region_localization.py'),
    ('pi', 'pi/placement_extension.py', BASES['pi'] + '/final_work/pi/placement_extension.py'),
    ('pi', 'pi/placement_extension.py', '/home/dachuang/grab_visual/placement_extension.py'),
    ('gpu', 'gpu/robot_skill_catalog.py', BASES['gpu'] + '/final_work/gpu/robot_skill_catalog.py'),
    ('gpu', 'gpu/qwen_hub.py', BASES['gpu'] + '/final_work/gpu/qwen_hub.py'),
    ('gpu', 'gpu/planner.py', BASES['gpu'] + '/final_work/gpu/planner.py'),
    ('pi', 'phase1/adaptive_turn.py', '/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919/grasp_range_integration/phase1/adaptive_turn.py'),
    ('pi', 'phase1/motion_interlock.py', '/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919/grasp_range_integration/phase1/motion_interlock.py'),
    ('pi', 'phase1/relationship_find_only_entry.py', '/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919/grasp_range_integration/phase1/relationship_find_only_entry.py'),
    ('pi', 'phase1/fresh_motion_feedback.py', '/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919/grasp_range_integration/phase1/fresh_motion_feedback.py'),
    ('pi', 'phase1/startup_active_recovery.py', '/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919/grasp_range_integration/phase1/startup_active_recovery.py'),
]

# Audited against BOTH the local pre-edit file and live Pi on 2026-10-15.
# Never accept a freshly observed remote hash just to bypass version review.
REVIEWED_BASELINES = {
    ('pi', 'pi/region_navigation.py'): '784b84a4b272b92291e8456ba6423e1684621b4e67e2db88e965adbb2703eab0',
    ('pi', 'pi/region_localization.py'): '66b6383127e8c715bf4b790eb6ca251ca307c470288a46d308978375a2e2d3d6',
    ('pi', 'phase1/adaptive_turn.py'): '17402118ce03fbbeecf4620b2634ac815304cd4516126c8ca10ccdadf1ac16f2',
    ('pi', 'phase1/motion_interlock.py'): '904cbde7fa5b23564d862ae419f83d5aa099efd9d97c39bc22a7a1971095cbb4',
    ('pi', 'phase1/fresh_motion_feedback.py'): 'c08a38d8b8fbc35d82a7b753609b1b1937b44c1f3a716bf260ba0eec7a38f2ca',
}


def sha(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def read(sftp, path):
    try:
        with sftp.open(path, 'rb') as handle:
            return handle.read()
    except FileNotFoundError:
        return None


def write(sftp, path, data):
    with sftp.open(path, 'wb') as handle:
        handle.write(data)
    if read(sftp, path) != data:
        raise RuntimeError('Upload verification failed: ' + path)


def run(client, command):
    _, output, errors = client.exec_command(command, timeout=35)
    result, error = output.read().decode('utf-8', 'replace'), errors.read().decode('utf-8', 'replace')
    if output.channel.recv_exit_status():
        raise RuntimeError(error or result or 'Remote command failed')
    return result


def python(client, role, code):
    return run(client, shlex.quote(PYTHONS[role]) + ' -c ' + shlex.quote(code))


def get(client, role, url):
    return json.loads(python(client, role,
        'import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen(' + repr(url) + ',timeout=6))))'))


def idle(pi):
    state = get(pi, 'pi', 'http://127.0.0.1:8780/state')
    managed = state.get('managed') or {}
    if managed.get('running') and managed.get('kind') != 'camera_preview':
        raise RuntimeError('Robot task is still running; not reloading')
    if (state.get('robot_skills') or {}).get('status') in {'placing', 'placement_unknown'}:
        raise RuntimeError('Release still pending/unknown; not reloading')
    arm = get(pi, 'pi', 'http://127.0.0.1:8080/api/arm-teach/exec')
    if (arm.get('status') or {}).get('active'):
        raise RuntimeError('Arm queue active; not reloading')
    return state, arm


def recovered_holding(pi, gpu, original, arm):
    """Reconcile metadata only; never send BLE or a placement request."""
    if original.get('status') != 'empty':
        raise RuntimeError('Expected the audited empty ledger; do not overwrite another carry')
    expected = (original.get('previous_carry') or {}).get('carry_id')
    code = """import sqlite3,json
c=sqlite3.connect('file:/home/sjtu/track/final_work/data/history.sqlite3?mode=ro',uri=True)
result=None
for task_id,plan,execution in c.execute('SELECT id,plan_json,execution_json FROM tasks ORDER BY created_at DESC LIMIT 80'):
 p=json.loads(plan);e=json.loads(execution)
 for step in p.get('steps',[]):
  state=(e.get('steps') or {}).get(step.get('id'),{});detail=state.get('detail') or {};carry=detail.get('carry') or {}
  if step.get('action')=='model_grasp' and state.get('status')=='completed' and carry.get('carry_id')==EXPECTED:
   result={'task_id':task_id,'carry':carry,'execution':detail.get('execution')};break
 if result:break
print(json.dumps(result))
""".replace('EXPECTED', repr(expected))
    proof = json.loads(python(gpu, 'gpu', code))
    if not proof or not expected:
        raise RuntimeError('No matching completed model grasp in web history')
    ground = proof['carry'].get('ground_pose') or {}
    execution = arm.get('status') or {}
    live = get(pi, 'pi', 'http://127.0.0.1:8080/api/arm-teach/state')
    arm_state = live.get('state') or {}
    if (execution.get('phase') != 'done' or execution.get('params') != ground
            or (proof.get('execution') or {}).get('params') != ground
            or arm_state.get('claw_value') != 0 or arm_state.get('known') is not True
            or live.get('pending') != 0):
        raise RuntimeError('Live arm history differs from confirmed held-object grasp')
    specs = {row['key']: row for row in live.get('servos') or []}
    for key in ('s0', 's1', 's2'):
        value, spec = ground.get(key), specs[key]
        if not isinstance(value, int) or not spec['min'] <= value <= spec['max']:
            raise RuntimeError('Ground pose outside current servo limits')
    return {**proof['carry'], 'status': 'carrying', 'physical_pick_verified': False,
            'restored_at': time.time(), 'restored_from_task': proof['task_id'],
            'restored_reason': 'operator_confirmed_holding_and_unmodified_arm',
            'operator_confirmed_holding': True, 'live_execution_pose_matched': True}


def wait_health(client, role, url):
    deadline = time.monotonic() + 30
    last = None
    while time.monotonic() < deadline:
        try:
            value = get(client, role, url)
            if value.get('ok'):
                return value
        except Exception as exc:
            last = exc
        time.sleep(.5)
    raise RuntimeError('Service did not become healthy: ' + str(last))


def stop_verified_web_processes(client):
    """Bound shutdown of ONLY this user's exact module/port, not model servers.

MJPEG/SSE clients may keep an old Uvicorn worker draining after its listener
closes. Recheck PID start time and command tokens before escalation, so a
reused PID or an unrelated service can never be signalled by this cleanup.
"""
    code = """import os,pathlib,signal,time,json
def identity(pid):
 try:
  p=pathlib.Path('/proc')/str(pid)
  if p.stat().st_uid!=os.getuid():return None
  tokens=p.joinpath('cmdline').read_bytes().split(bytes([0]));tokens=[x.decode() for x in tokens if x]
  if '-m' not in tokens or tokens[tokens.index('-m')+1]!='final_work.gpu.app':return None
  if '--port' not in tokens or tokens[tokens.index('--port')+1]!='8030':return None
  stat=p.joinpath('stat').read_text().split(')',1)[1].split()
  return None if stat[0]=='Z' else stat[19]
 except (OSError,ValueError,IndexError):return None
targets={int(p.name):identity(int(p.name)) for p in pathlib.Path('/proc').iterdir() if p.name.isdigit()}
targets={pid:start for pid,start in targets.items() if start is not None}
for pid,start in targets.items():
 if identity(pid)==start:
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
deadline=time.monotonic()+5
while any(identity(pid)==start for pid,start in targets.items()) and time.monotonic()<deadline:time.sleep(.25)
forced=[]
for pid,start in targets.items():
 if identity(pid)==start:
  try:os.kill(pid,signal.SIGKILL);forced.append(pid)
  except ProcessLookupError:pass
deadline=time.monotonic()+5
while any(identity(pid)==start for pid,start in targets.items()) and time.monotonic()<deadline:time.sleep(.25)
assert not any(identity(pid)==start for pid,start in targets.items()),'Web process still active'
print(json.dumps({'stopped_web_pids':list(targets),'draining_workers_terminated':forced}))
"""
    print(python(client, 'gpu', code).strip(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--restore-confirmed-holding', action='store_true')
    options = parser.parse_args()
    if options.reload and not options.apply:
        parser.error('--reload requires --apply')
    if options.apply and not options.reload:
        parser.error('These runtime changes require an explicitly authorised reload')
    clients = {}
    prepared, activated, stopped = [], [], []
    stamp = time.strftime('%Y%m%d_%H%M%S')
    original_ledger = restored_ledger = None
    ledger_changed = False
    try:
        for role in ('pi', 'gpu'):
            clients[role] = connect(role)
        pi, gpu = clients['pi'], clients['gpu']
        state, arm = idle(pi)
        with pi.open_sftp() as sftp:
            original_ledger = read(sftp, LEDGER)
        if options.restore_confirmed_holding:
            restored = recovered_holding(pi, gpu, json.loads(original_ledger), arm)
            restored_ledger = json.dumps(restored, ensure_ascii=False, indent=2).encode('utf-8')
            print('holding_proof_verified: latest carry ID and executed ground pose match; metadata only', flush=True)
        manifest = json.loads((ROOT / 'docs/maintenance/source_manifest.json').read_text(encoding='utf-8'))['files']
        for index, (role, relative, remote) in enumerate(FILES):
            phase_module = relative.startswith('phase1/')
            source_key = ('robot/grasp_range_integration/' if phase_module else 'services/track/final_work/') + relative
            local = ROOT / source_key
            desired = local.read_bytes()
            if not phase_module:
                mirror = ROOT / 'robot/grasp_range_integration/track_web_integration' / relative
                if mirror.read_bytes() != desired:
                    raise RuntimeError('Maintenance source mismatch: ' + relative)
            allowed = {sha(desired), manifest.get(source_key, {}).get('source_sha256')}
            if (role, relative) in REVIEWED_BASELINES:
                allowed.add(REVIEWED_BASELINES[(role, relative)])
            if remote == '/home/dachuang/grab_visual/placement_extension.py':
                allowed.add(manifest.get('services/grasp/pi/placement_extension.py', {}).get('source_sha256'))
            backup = BASES[role] + '/final_work/deployment_backups/runtime_' + stamp
            stage = backup + '/' + str(index) + '_' + PurePosixPath(remote).name
            with clients[role].open_sftp() as sftp:
                current = read(sftp, remote)
                if sha(current) not in allowed or (current is None and relative != 'pi/region_localization.py'):
                    raise RuntimeError('Remote source changed since assembly; refusing overwrite: ' + remote)
                prepared.append((role, remote, current, desired, stage))
                if options.apply:
                    run(clients[role], 'mkdir -p ' + shlex.quote(backup))
                    if current is not None:
                        write(sftp, stage + '.original', current)
                    write(sftp, stage, desired)
            print('reviewed ' + role + ': ' + relative, flush=True)
        if not options.apply:
            print('dry_run_ok; no writes or service changes')
            return
        # Compile all staged files before stopping any service.
        for role in clients:
            paths = [stage for host, _, _, _, stage in prepared if host == role]
            run(clients[role], shlex.quote(PYTHONS[role]) + ' -m py_compile ' + ' '.join(map(shlex.quote, paths)))
        idle(pi)
        old_web_pid = run(gpu, 'cat /home/sjtu/track/final_work/agent_web.pid').strip()
        if not old_web_pid.isdigit():
            raise RuntimeError('Web PID file invalid; refusing an unverified reload')
        run(gpu, 'bash /home/sjtu/track/final_work/stop_3090.sh')
        stopped.append('gpu')
        stop_verified_web_processes(gpu)
        idle(pi)  # Catch a task started before the web listener finished exiting.
        run(pi, 'sudo -n systemctl stop final-work-pi-bridge.service')
        stopped.append('pi')
        print('services_stopped; no motion dispatched', flush=True)
        for role, remote, current, desired, stage in prepared:
            with clients[role].open_sftp() as sftp:
                if read(sftp, remote) != current:
                    raise RuntimeError('Concurrent remote source edit: ' + remote)
                sftp.posix_rename(stage, remote)
                activated.append((role, remote, current, desired, stage))
                if read(sftp, remote) != desired:
                    raise RuntimeError('Activation hash differs: ' + remote)
        if restored_ledger is not None:
            backup = BASES['pi'] + '/final_work/deployment_backups/runtime_' + stamp
            with pi.open_sftp() as sftp:
                if read(sftp, LEDGER) != original_ledger:
                    raise RuntimeError('Carry ledger changed during deployment')
                write(sftp, backup + '/robot_skills_state.json.original', original_ledger)
                write(sftp, backup + '/robot_skills_state.json.restored', restored_ledger)
                write(sftp, LEDGER + '.runtime_update', restored_ledger)
                sftp.posix_rename(LEDGER + '.runtime_update', LEDGER)
                ledger_changed = True
        # Import checks only: no instance, camera, BLE or hardware action.
        python(pi, 'pi', "import os;os.chdir('/home/dachuang/dachuang/track');import sys;sys.path[:0]=['/home/dachuang/dachuang/car1.0','/home/dachuang/dachuang/car1.0/ble_scene_17_relationship_20260919',os.getcwd()];from final_work.pi.region_localization import localize_region;from final_work.pi.robot_skills import RobotSkills;print('pi_imports_ok')")
        run(pi, 'sudo -n systemctl start final-work-pi-bridge.service')
        run(pi, 'sudo -n systemctl start final-work-pi-tunnel.service')
        wait_health(pi, 'pi', 'http://127.0.0.1:8780/health')
        run(gpu, 'bash /home/sjtu/track/final_work/run_3090_background.sh')
        wait_health(gpu, 'gpu', 'http://127.0.0.1:8030/api/health')
        new_web_pid = run(gpu, 'cat /home/sjtu/track/final_work/agent_web.pid').strip()
        if new_web_pid == old_web_pid:
            raise RuntimeError('Web PID did not change; cannot confirm loaded code')
        wait_health(gpu, 'gpu', 'http://127.0.0.1:18780/health')
        stopped.clear()
        report = {'backup': 'runtime_' + stamp, 'files': [{'host': role, 'path': remote, 'sha256': sha(desired)}
            for role, remote, _, desired, _ in prepared], 'services_reloaded': True,
            'old_web_pid': old_web_pid, 'new_web_pid': new_web_pid,
            'holding_metadata_restored': ledger_changed, 'hardware_commands_sent': False}
        report_path = ROOT / '.local/deployments' / ('runtime_' + stamp + '.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        # Do not overwrite a concurrent operator/task edit during rollback.
        for role, remote, current, desired, stage in reversed(activated):
            with clients[role].open_sftp() as sftp:
                if read(sftp, remote) == desired:
                    sftp.posix_rename(remote, stage + '.failed')
                    if current is not None:
                        sftp.posix_rename(stage + '.original', remote)
        if ledger_changed:
            with clients['pi'].open_sftp() as sftp:
                if read(sftp, LEDGER) == restored_ledger:
                    write(sftp, LEDGER + '.runtime_rollback', original_ledger)
                    sftp.posix_rename(LEDGER + '.runtime_rollback', LEDGER)
        if 'pi' in stopped:
            run(clients['pi'], 'sudo -n systemctl start final-work-pi-bridge.service')
            run(clients['pi'], 'sudo -n systemctl start final-work-pi-tunnel.service')
        if 'gpu' in stopped:
            run(clients['gpu'], 'bash /home/sjtu/track/final_work/run_3090_background.sh')
        raise
    finally:
        for client in clients.values():
            client.close()


if __name__ == '__main__':
    main()
