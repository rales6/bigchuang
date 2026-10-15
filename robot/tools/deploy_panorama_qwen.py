"""Back up and deploy the panorama prompt, preserving the current Qwen hub."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

from tools.deploy_object_search import connect
from tools.deploy_inspection_route_fix import command

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "/home/sjtu/3090/scripts/run_qwen_landmark_server.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    content = (ROOT / "3090/scripts/run_qwen_landmark_server.py").read_bytes().replace(b"\r\n", b"\n")
    expected = hashlib.sha256(content).hexdigest()
    client = connect("202.121.181.124", "sjtu", os.environ["RTX_SSH_PASSWORD"], port=2020)
    backup = "/home/sjtu/3090/backups/panorama_" + time.strftime("%Y%m%d_%H%M%S")
    try:
        with client.open_sftp() as sftp:
            previous = sftp.open(SOURCE, "rb").read()
            command(client, "mkdir -p " + shlex.quote(backup))
            saved = backup + "/run_qwen_landmark_server.py"
            command(client, "cp -p " + shlex.quote(SOURCE) + " " + shlex.quote(saved))
            staged = SOURCE + ".panorama-staged"
            with sftp.open(staged, "wb") as destination:
                destination.write(content)
            if hashlib.sha256(sftp.open(staged, "rb").read()).hexdigest() != expected:
                raise RuntimeError("staged upload hash mismatch")
            check = "import runpy; d=runpy.run_path(" + repr(staged) + "); p=d['build_prompt']({'inspection_mode':'panorama_sector'}); assert 'map_memory_allowed=false' in p; assert 'LEFT/RIGHT' in p; print('PANORAMA_PROMPT_OK')"
            print(command(client, "/home/sjtu/3090/.venv/bin/python -c " + shlex.quote(check)).strip(), flush=True)
            if sftp.open(SOURCE, "rb").read() != previous:
                raise RuntimeError("remote source changed during staging")
            sftp.chmod(staged, sftp.stat(SOURCE).st_mode & 0o777)
            sftp.posix_rename(staged, SOURCE)
            if hashlib.sha256(sftp.open(SOURCE, "rb").read()).hexdigest() != expected:
                raise RuntimeError("activated upload hash mismatch")

        restart = r'''
import json, os, signal, subprocess, time, urllib.request
pid = PID
argv = open(f'/proc/{pid}/cmdline','rb').read().split(b'\0')
argv = [os.fsdecode(a) for a in argv if a]
assert '--existing-script' in argv and argv[argv.index('--existing-script')+1] == SOURCE
assert '--port' in argv and argv[argv.index('--port')+1] == '8010'
assert 'final_work.gpu.qwen_hub' in argv
cwd = os.readlink(f'/proc/{pid}/cwd')
env = dict(os.fsdecode(a).split('=',1) for a in open(f'/proc/{pid}/environ','rb').read().split(b'\0') if b'=' in a)
os.kill(pid, signal.SIGTERM)
deadline = time.monotonic()+20
while os.path.exists(f'/proc/{pid}') and time.monotonic()<deadline:
    time.sleep(.25)
if os.path.exists(f'/proc/{pid}'):
    raise RuntimeError('Qwen hub did not stop gracefully; no second hub started')
with open(LOG,'ab',buffering=0) as log:
    process = subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(60):
    if process.poll() is not None:
        raise RuntimeError('Qwen hub exited; inspect '+LOG)
    try:
        data = json.load(urllib.request.urlopen('http://127.0.0.1:8010/agent/health',timeout=2))
        if data.get('ok') and not data.get('mock'):
            print(json.dumps({'pid':process.pid,'health':data,'log':LOG}))
            break
    except Exception:
        pass
    time.sleep(.5)
else:
    raise RuntimeError('Qwen hub health timeout')
'''.replace("PID", str(args.pid)).replace("SOURCE", repr(SOURCE)).replace("LOG", repr(backup + "/qwen_hub.log"))
        print("BACKUP=" + backup, flush=True)
        print(command(client, "/usr/bin/python3 -c " + shlex.quote(restart), timeout=100).strip(), flush=True)
        print("QWEN_PANORAMA_DEPLOYED_NO_ROBOT_MOTION")
    finally:
        client.close()


if __name__ == "__main__":
    main()
