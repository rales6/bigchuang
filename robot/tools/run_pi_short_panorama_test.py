"""User-supervised, finite route test; status/stop target this test only."""
import argparse
import json
import os
import time

from tools.deploy_panorama_repeat_motion import ACTIVE_CHECK
from tools.deploy_task_localization_and_start_inventory import (
    ROOT, PI_ROOT, connect, python, pi_python, remote_hashes,
)


SUPERVISOR = r'''
import fcntl, json, os, pathlib, re, shlex, signal, subprocess, sys, time
root = pathlib.Path(ROOT_PATH)
status_path = pathlib.Path(STATUS_PATH)
lock = open('/tmp/car1_map_camera_inspection.lock', 'a')
def save(payload):
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(temporary, status_path)
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    save({'state':'refused_existing_inspection_lock'})
    sys.exit(2)
text = (root/'inspect_map.sh').read_text(encoding='utf-8')
command = text.split('python3 -u -m raspberry_pi.map_camera_inspection',1)[1].split('2>&1',1)[0]
command = command.replace('\\\n', ' ')
command = re.sub(r'\$\{[A-Z_]+:-([^}]+)\}', lambda m:m.group(1), command)
arguments = shlex.split(command)
def set_argument(name, value):
    if name in arguments:
        arguments[arguments.index(name)+1] = str(value)
    else:
        arguments.extend([name,str(value)])
set_argument('--online-qwen-url', '')
set_argument('--max-runtime-min', 1.)
set_argument('--panorama-drive-speed', 300)
set_argument('--start-kick-speed', 300)
if '--free-cameras' in arguments:
    arguments.remove('--free-cameras')
child = None
stop_requested = False
def stop(_sig=None, _frame=None):
    global stop_requested
    stop_requested = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGTERM)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
if stop_requested:
    save({'state':'cancelled_before_start'})
    sys.exit(0)
started = time.monotonic()
child = subprocess.Popen([INTERPRETER, '-u', '-m', 'raspberry_pi.map_camera_inspection']+arguments,
    cwd=root, stdin=subprocess.DEVNULL)
save({'state':'running','supervisor_pid':os.getpid(),'child_pid':child.pid,
      'maximum_wall_seconds':150,'motion_runtime_minutes':1.,'arguments':arguments})
print('短时路线测试已启动；巡检限时一分钟，总进程监督150秒；Qwen关闭，防撞保留。',flush=True)
stop_time = None
forced_kill = False
while child.poll() is None:
    if (time.monotonic()-started >= 150 or stop_requested) and stop_time is None:
        stop()
        stop_time = time.monotonic()
        print('短时测试请求停车退出。',flush=True)
    if stop_time is not None and time.monotonic()-stop_time >= 15:
        child.kill()  # Only this owned process; firmware command TTL expires.
        forced_kill = True
        break
    time.sleep(.1)
code = child.wait()
save({'state':'finished','exit_code':code,'elapsed_s':time.monotonic()-started,
      'supervisor_pid':os.getpid(),'child_pid':child.pid,'forced_kill':forced_kill})
print('短时测试结束，退出码={}，强制结束={}。'.format(code,forced_kill),flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--start', action='store_true')
    group.add_argument('--status', action='store_true')
    group.add_argument('--stop', action='store_true')
    parser.add_argument('--receipt')
    args = parser.parse_args()
    client = connect('dachuang.local', 22, 'dachuang', os.environ['PI_SSH_PASSWORD'])
    try:
        if args.start:
            active = json.loads(python(client, 'python3', ACTIVE_CHECK))
            if active:
                raise RuntimeError('Active mission; no second launch: ' + json.dumps(active))
            manifest = json.loads((ROOT/'simulation_runs/panorama_heading_recovery_20260918_local_manifest.json').read_text(encoding='utf-8'))
            expected = {r['path']:r['sha256'] for r in manifest['source_hashes']}
            actual = remote_hashes(client, PI_ROOT, list(expected))
            if actual != expected:
                raise RuntimeError('Pi does not have reviewed heading recovery')
            stamp = time.strftime('%Y%m%d_%H%M%S')
            log = PI_ROOT + '/inspection_logs/inspect_map_' + stamp + '.log'
            status = PI_ROOT + '/inspection_logs/heading_short_' + stamp + '_status.json'
            token = 'heading_short_' + stamp
            interpreter = pi_python(client)
            supervisor = SUPERVISOR.replace('ROOT_PATH',repr(PI_ROOT)).replace('STATUS_PATH',repr(status)).replace('INTERPRETER',repr(interpreter))
            code = '''
import json, os, subprocess
with open(LOG_PATH,'xb') as log:
    process = subprocess.Popen([PYTHON_PATH,'-u','-c',SUPERVISOR_CODE,TOKEN],
        cwd=ROOT_PATH,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':process.pid,'log':LOG_PATH,'status_path':STATUS_PATH,'token':TOKEN}))
'''
            for key,value in [('LOG_PATH',log),('PYTHON_PATH',interpreter),('SUPERVISOR_CODE',supervisor),
                              ('TOKEN',token),('ROOT_PATH',PI_ROOT),('STATUS_PATH',status)]:
                code = code.replace(key,repr(value))
            receipt = json.loads(python(client,'python3',code))
            receipt['robot_launch_requested'] = True
            path = ROOT/'simulation_runs'/('heading_short_test_'+stamp+'.json')
            path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
            receipt['local_receipt'] = str(path)
            print(json.dumps(receipt,ensure_ascii=False))
        else:
            if not args.receipt:
                raise RuntimeError('Use the receipt of this specific short test')
            receipt = json.loads((ROOT/args.receipt).read_text(encoding='utf-8'))
            code = '''
import json, os, pathlib, signal
status_path=pathlib.Path(STATUS_PATH)
status=json.loads(status_path.read_text()) if status_path.exists() else {'state':'starting'}
if STOP and status.get('state') not in ('finished','cancelled_before_start','refused_existing_inspection_lock'):
    cmdline_path=pathlib.Path('/proc')/str(PID)/'cmdline'
    if cmdline_path.exists() and TOKEN.encode() in cmdline_path.read_bytes().split(b'\\0'):
        os.kill(PID,signal.SIGTERM)
        status['stop_requested']=True
    else:
        status['stop_not_sent']='owned_supervisor_not_running'
log=pathlib.Path(LOG_PATH)
print(json.dumps({'status':status,'log_tail':log.read_text(encoding='utf-8',errors='replace')[-5000:] if log.exists() else ''},ensure_ascii=False))
'''
            for key,value in [('STATUS_PATH',receipt['status_path']),('LOG_PATH',receipt['log']),
                              ('PID',receipt['supervisor_pid']),('TOKEN',receipt['token']),('STOP',args.stop)]:
                code = code.replace(key,repr(value))
            print(python(client,'python3',code))
    finally:
        client.close()


if __name__ == '__main__':
    main()
