from __future__ import annotations

import os
import sys
import time

import paramiko


def command(client, value: str, timeout: int = 30) -> str:
    _stdin, stdout, stderr = client.exec_command(value, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if code:
        raise RuntimeError(error or output or f"remote command failed: {code}")
    return output


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "dachuang.local",
        username="dachuang",
        password=os.environ["PI_SSH_PASSWORD"],
        timeout=45,
        auth_timeout=45,
        banner_timeout=45,
    )
    try:
        state = command(
            client,
            "cd ~/dachuang/car1.0 && "
            "test -f search_runs/block_command.txt && "
            "curl -fsS --max-time 3 http://127.0.0.1:8080/ >/dev/null && "
            "! pgrep -f '[o]bject_search_mission' >/dev/null && "
            "echo PRECHECK_OK",
        )
        print(state.strip(), flush=True)
        launch = (
            "cd ~/dachuang/car1.0 && source .venv-pi/bin/activate && "
            "export PYTHONUNBUFFERED=1 SEARCH_TEST_MAX_DISTANCE=0 && "
            "timeout --signal=INT --kill-after=15s 600s "
            "./natural_search.sh @search_runs/block_command.txt real block qwen_fallback "
            "2>&1 | tee search_runs/live_block_search_fused.log"
        )
        _stdin, stdout, _stderr = client.exec_command(launch, timeout=630)
        while not stdout.channel.exit_status_ready():
            if stdout.channel.recv_ready():
                text = stdout.channel.recv(8192).decode("utf-8", errors="replace")
                if text:
                    print(text, end="", flush=True)
            time.sleep(0.15)
        while stdout.channel.recv_ready():
            print(stdout.channel.recv(8192).decode("utf-8", errors="replace"), end="", flush=True)
        print(f"LIVE_BLOCK_EXIT={stdout.channel.recv_exit_status()}", flush=True)
        print(command(
            client,
            "cd ~/dachuang/car1.0 && python3 - <<'PY'\n"
            "import json, pathlib\n"
            "paths=sorted(pathlib.Path('search_runs/ble_scene_17').glob('*/mission.json'))\n"
            "p=paths[-1]; d=json.load(open(p,encoding='utf-8'))\n"
            "print('MISSION',p)\n"
            "print(json.dumps({'result':d.get('result'),'summary':d.get('summary'),'events':d.get('events',[])},ensure_ascii=False))\n"
            "PY",
        ))
    finally:
        try:
            command(
                client,
                "pids=$(pgrep -f '[o]bject_search_mission' || true); "
                "if [ -n \"$pids\" ]; then kill -INT $pids; sleep 2; fi",
                timeout=10,
            )
        except Exception as exc:
            print(f"cleanup warning: {exc}", file=sys.stderr)
        client.close()


if __name__ == "__main__":
    main()
