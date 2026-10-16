"""Bounded live test using existing 3090 Qwen API; never controls the robot."""
import os
from pathlib import Path
import shlex
import time

import paramiko


def main():
    root = Path(__file__).resolve().parents[1]
    snapshot = root / 'backups/semantic_inspection_20260917_195037/pi_snapshot'
    inspection = snapshot / 'inspections/ble_scene_17/20260917_190706'
    job = '/home/gpu/car_inspection_jobs/object_inventory_probe_' + time.strftime('%Y%m%d_%H%M%S')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('198.51.100.20', port=2020, username='sjtu',
                   password=os.environ['QWEN_SSH_PASSWORD'], timeout=15)
    try:
        with client.open_sftp() as sftp:
            sftp.mkdir(job)
            for relative in ['scripts', 'src', 'src/qwen_grounded_tracker', 'photos', 'maps']:
                sftp.mkdir(job + '/' + relative)
            pairs = [(root / ('3090/' + name), name) for name in [
                'scripts/run_semantic_object_inventory.py',
                'src/qwen_grounded_tracker/__init__.py',
                'src/qwen_grounded_tracker/inspection_annotation.py',
                'src/qwen_grounded_tracker/semantic_object_catalog.py']]
            pairs += [(inspection / name, 'photos/' + name) for name in [
                'inspection_manifest.json', 'panorama_s01_d000.jpg', 'panorama_s01_d000_camera_2.jpg']]
            pairs += [(snapshot / ('maps/' + name), 'maps/' + name) for name in ['ble_scene_17.pgm', 'ble_scene_17.yaml']]
            for source, destination in pairs:
                sftp.put(str(source), job + '/' + destination)
        command = ['/home/gpu/3090/.venv/bin/python', '-u', job + '/scripts/run_semantic_object_inventory.py',
            '--inspection', job + '/photos', '--map', job + '/maps/ble_scene_17.pgm',
            '--map-yaml', job + '/maps/ble_scene_17.yaml', '--output', job + '/output',
            '--ground-url', 'http://127.0.0.1:8010/agent/ground', '--queries', '插座', '--limit', '1', '--timeout', '90']
        print('REMOTE_JOB=' + job, flush=True)
        _, stdout, stderr = client.exec_command(shlex.join(command), timeout=240)
        for line in stdout:
            print(line.rstrip(), flush=True)
        print(stderr.read().decode(), flush=True)
        code = stdout.channel.recv_exit_status()
        output = root / 'semantic_maps/ble_scene_17_20260917_190706/live_qwen_probe'
        output.mkdir(parents=True, exist_ok=True)
        with client.open_sftp() as sftp:
            for name in ['semantic_object_memory.json', 'inventory_report.json', 'object_inventory.csv']:
                sftp.get(job + '/output/' + name, str(output / name))
            for name in sftp.listdir(job + '/output/checkpoints'):
                sftp.get(job + '/output/checkpoints/' + name, str(output / name))
        print('LOCAL_OUTPUT=' + str(output), flush=True)
        return code
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
