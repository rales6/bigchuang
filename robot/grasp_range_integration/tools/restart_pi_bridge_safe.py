"""Restart the idle final_work Pi bridge after a verified adapter deployment."""

from __future__ import annotations

import hashlib
import json
import os
import time

import paramiko


HOST = "192.0.2.13"
HOST_KEY_SHA256 = "1c624d9d6c33aa0482a4f043a4b17b295b9b64dd232c3e3fadce4d6436594ee6"
TRACK = "/home/robot/dachuang/track"
PID_FILE = TRACK + "/final_work/pi_bridge.pid"


def _run(client: paramiko.SSHClient, command: str, timeout: int = 15) -> str:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    errors = stderr.read().decode("utf-8", "replace")
    if stdout.channel.recv_exit_status():
        raise RuntimeError(f"远端服务命令失败：{errors[-400:]} {output[-400:]}")
    return output


def main() -> None:
    password = os.environ.get("PI_SSH_PASSWORD")
    if not password:
        raise RuntimeError("缺少 PI_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="dachuang", password=password,
                   look_for_keys=False, allow_agent=False, timeout=15,
                   auth_timeout=15, banner_timeout=15)
    try:
        fingerprint = hashlib.sha256(
            client.get_transport().get_remote_server_key().asbytes()
        ).hexdigest()
        if fingerprint != HOST_KEY_SHA256:
            raise RuntimeError("SSH 主机密钥不匹配")
        state = json.loads(_run(client,
            "curl --fail --silent --show-error http://127.0.0.1:8780/state"))
        if (state.get("managed") or {}).get("running"):
            raise RuntimeError("桥接仍有管理中的任务，拒绝重启")
        with client.open_sftp() as sftp:
            with sftp.open(PID_FILE, "rb") as stream:
                pid_text = stream.read().decode("ascii").strip()
            if not pid_text.isdecimal():
                raise RuntimeError("Pi 桥接 PID 文件无效")
            pid = int(pid_text)
            with sftp.open(f"/proc/{pid}/cmdline", "rb") as stream:
                cmdline = stream.read().replace(b"\0", b" ").decode("utf-8", "replace")
        if "final_work.pi.robot_bridge" not in cmdline:
            raise RuntimeError("PID 不属于预期的 Pi 桥接进程，拒绝停止")
        _run(client, f"kill -TERM {pid}")
        for _ in range(20):
            with client.open_sftp() as sftp:
                try:
                    with sftp.open(f"/proc/{pid}/cmdline", "rb") as stream:
                        still = stream.read()
                except FileNotFoundError:
                    still = b""
            if b"final_work.pi.robot_bridge" not in still:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("旧桥接没有按时退出；未启动第二个实例")
        output = _run(client, f"cd {TRACK} && bash final_work/run_pi_background.sh", timeout=30)
        for _ in range(20):
            try:
                health = json.loads(_run(client,
                    "curl --fail --silent --show-error http://127.0.0.1:8780/health"))
            except RuntimeError:
                time.sleep(0.25)
                continue
            if health.get("ok") is True:
                print(json.dumps({"restart": output.strip(), "health": health,
                                  "hardware_task_started": False}, ensure_ascii=False))
                return
        raise RuntimeError("新 Pi 桥接健康检查未通过")
    finally:
        client.close()


if __name__ == "__main__":
    main()
