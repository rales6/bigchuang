"""Acquire Pi lidar evidence without importing or commanding the chassis."""
import getpass
import gzip
import json
from pathlib import Path
import time

import paramiko


REMOTE_CODE = r'''
import fcntl, json, math, os, signal, subprocess, time
lock = open('/tmp/car1_map_camera_inspection.lock', 'a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)
active = [line for line in processes.splitlines() if any(
    ('-m raspberry_pi.' + name) in line for name in
    ('autonomous_mapping', 'joint_mapping', 'map_camera_inspection', 'object_search_mission', 'motion_calibration'))]
if active:
    raise RuntimeError('Motion task active; no acquisition: ' + repr(active))
holders = subprocess.run(['fuser', '/dev/ttyACM0'], capture_output=True, text=True)
if holders.returncode != 1:
    raise RuntimeError('Lidar busy or holder check failed: ' + holders.stdout + holders.stderr)
print(json.dumps({'type': 'preflight', 'motion_tasks': active, 'lidar_holders': [], 'chassis_commands': 'none'}), flush=True)
from raspberry_pi.config import LidarConfig
from raspberry_pi.lidar import N10LidarDriver
def timeout(*args):
    raise TimeoutError('bounded stationary acquisition timeout')
signal.signal(signal.SIGALRM, timeout)
signal.alarm(40)
lidar = None
count = 0
try:
    lidar = N10LidarDriver(LidarConfig(port='/dev/ttyACM0'))
    first = None
    for scan in lidar.scans():
        if first is None:
            first = scan.timestamp_s
        finite = lambda values: [float(v) if math.isfinite(float(v)) else None for v in values]
        print(json.dumps({'type': 'scan', 'timestamp_s': scan.timestamp_s,
                          'angles_rad': finite(scan.angles_rad),
                          'distances_m': finite(scan.distances_m)}), flush=True)
        count += 1
        if count >= 100 or scan.timestamp_s - first >= 20:
            break
finally:
    signal.alarm(0)
    if lidar is not None:
        lidar.close()
    print(json.dumps({'type': 'closed', 'scans': count, 'chassis_commands': 'none'}), flush=True)
'''


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / 'simulation_runs' / ('stationary_field_' + time.strftime('%Y%m%d_%H%M%S'))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('dachuang.local', username='dachuang', password=getpass.getpass('Pi password: '), timeout=20)
    try:
        directory.mkdir(parents=True, exist_ok=False)
        with client.open_sftp() as sftp:
            for name in ('ble_scene_17.pgm', 'ble_scene_17.yaml'):
                sftp.get('/home/dachuang/dachuang/car1.0/maps/' + name, str(directory / name))
        stdin, stdout, stderr = client.exec_command('cd /home/dachuang/dachuang/car1.0 && .venv-pi/bin/python -u -', timeout=55)
        stdin.write(REMOTE_CODE)
        stdin.flush()
        stdin.channel.shutdown_write()
        count = 0
        with gzip.open(directory / 'scans.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for line in stdout:
                row = json.loads(line)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if row['type'] == 'scan':
                    count += 1
                else:
                    print(json.dumps(row), flush=True)
        code = stdout.channel.recv_exit_status()
        error = stderr.read().decode()
        print('OUTPUT=' + str(directory))
        print('SCANS=' + str(count))
        if code:
            raise RuntimeError(error or 'remote acquisition failed')
    finally:
        client.close()


if __name__ == '__main__':
    main()
