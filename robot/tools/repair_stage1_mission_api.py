from __future__ import annotations

import os
from pathlib import Path
import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/sjtu/3090_stage1"


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("202.121.181.124", port=2020, username="sjtu", password=os.environ["RTX_SSH_PASSWORD"], timeout=45, auth_timeout=45, banner_timeout=45)
    with c.open_sftp() as s:
        s.put(str(ROOT / "3090/remote_stage1_snapshot/run_stage1_perception_server.py"), f"{REMOTE}/scripts/run_stage1_perception_server.py")
    command = (
        f"cd {REMOTE} && /home/sjtu/3090/.venv/bin/python -m py_compile scripts/run_stage1_perception_server.py && "
        "nohup /home/sjtu/3090/.venv/bin/python -u scripts/run_stage1_perception_server.py "
        "--model-path /home/sjtu/3090/models/qwen/Qwen3-VL-2B-Instruct --device cuda --max-new-tokens 900 "
        "--yolo-model /home/sjtu/3090/models/yolo/yolo11n-seg.pt --yolo-device 0 --yolo-confidence 0.25 "
        "--yolo-image-size 960 --pi-url http://172.18.14.74:8766 --host 0.0.0.0 --port 8011 "
        f">{REMOTE}/stage1_8011.log 2>&1 </dev/null &"
    )
    _, out, err = c.exec_command(command, timeout=30)
    if out.channel.recv_exit_status():
        raise RuntimeError(err.read().decode("utf-8", errors="replace"))
    verify = (
        "for i in $(seq 1 90); do body=$(curl -fsS http://127.0.0.1:8011/openapi.json 2>/dev/null || true); "
        "if echo \"$body\" | grep -q '/mission/next' && echo \"$body\" | grep -q '/mission/command'; "
        "then echo STAGE1_MISSION_API_OK; exit 0; fi; sleep 2; done; "
        f"tail -80 {REMOTE}/stage1_8011.log; exit 1"
    )
    _, out, err = c.exec_command(verify, timeout=210)
    code = out.channel.recv_exit_status()
    output = out.read().decode("utf-8", errors="replace")
    error = err.read().decode("utf-8", errors="replace")
    c.close()
    print(output)
    if code:
        raise RuntimeError(error or "8011 restart failed")


if __name__ == "__main__":
    main()
