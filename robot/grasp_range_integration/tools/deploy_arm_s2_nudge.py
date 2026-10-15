"""Apply a narrowly scoped, backed-up pre-close s2/s3 nudge to the Pi arm service.

This tool never sends an arm or chassis command and never restarts a service.
The firmware mirrors servo 3 from servo 2 (s3=3100-s2), so an independent
servo-3 command is not possible through the taught #ARMG protocol.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import time

import paramiko


HOST = "172.18.5.31"
HOST_KEY = "1c624d9d6c33aa0482a4f043a4b17b295b9b64dd232c3e3fadce4d6436594ee6"
SOURCE = "/home/dachuang/grab_visual/arm_teach.py"
EXPECTED_SHA256 = "1deb955ea5af8a58c7722c130948c56872fe21f4d228aeb26c606506a02f09a4"
MARKER = "def _s2_preclose_nudge_target(target: dict) -> dict | None:"


def render_patch(original: str) -> str:
    if MARKER in original:
        raise RuntimeError("s2/s3 微调已存在，拒绝重复叠加")
    helper_anchor = "def _run_grasp_sequence(ble_send, params: dict) -> None:\n"
    helper = '''def _s2_preclose_nudge_target(target: dict) -> dict | None:
    """One small absolute-pose correction after model motion, before claw close.

    Ground-grasp teaching samples used s2=1650..1850 versus HOME=2000.
    The firmware derives s3=3100-s2, so decreasing s2 mirrors servo 3.
    Override with ARM_GRASP_S2_NUDGE_US=0 to disable or a calibrated signed
    value within +/-25 us; this is not an independent servo-3 command.
    """
    raw = os.environ.get("ARM_GRASP_S2_NUDGE_US", "-10")
    try:
        delta = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("ARM_GRASP_S2_NUDGE_US must be an integer") from exc
    if not -25 <= delta <= 25:
        raise ValueError("ARM_GRASP_S2_NUDGE_US must be within -25..25 us")
    if delta == 0:
        return None
    spec = next(item for item in SERVOS if item["key"] == "s2")
    desired = int(target["s2"]) + delta
    if not spec["min"] <= desired <= spec["max"]:
        raise ValueError("s2 nudge would cross the configured servo limit")
    return {**target, "s2": desired}


'''
    start_anchor = '    try:\n        set_status(active=True, phase="home", message='
    start_replacement = ('    try:\n'
                         '        nudge_target = _s2_preclose_nudge_target(target)\n'
                         '        set_status(active=True, phase="home", message=')
    settle_anchor = ('        _heartbeat_wait(ble_send, 1.5)\n\n'
                     '        set_status(phase="grasp", message=')
    settle_replacement = '''        _heartbeat_wait(ble_send, 1.5)
        if aborted():
            raise RuntimeError("user stopped before pre-close nudge")
        if nudge_target is not None:
            set_status(phase="lower_nudge", params=dict(nudge_target),
                       message=f"Model pose reached; s2/s3 pre-close nudge to s2={nudge_target['s2']}")
            send_grasp_pose(ble_send, nudge_target)
            _heartbeat_wait(ble_send, 0.4)
            if aborted():
                raise RuntimeError("user stopped after pre-close nudge")

        set_status(phase="grasp", message='''
    for anchor, replacement in ((helper_anchor, helper + helper_anchor),
                                (start_anchor, start_replacement),
                                (settle_anchor, settle_replacement)):
        if original.count(anchor) != 1:
            raise RuntimeError(f"机械臂源码锚点数量不是 1：{anchor[:55]!r}")
        original = original.replace(anchor, replacement, 1)
    compile(original, SOURCE, "exec")
    return original


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remote_json(client: paramiko.SSHClient, command: str) -> dict:
    _stdin, stdout, stderr = client.exec_command(command, timeout=12)
    result = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    if stdout.channel.recv_exit_status():
        raise RuntimeError(f"远端状态检查失败：{error[-300:]}")
    return json.loads(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true", help="Backup and atomically replace the audited source")
    args = parser.parse_args()
    password = os.environ.get("PI_SSH_PASSWORD")
    if not password:
        raise RuntimeError("缺少 PI_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="dachuang", password=password, look_for_keys=False,
                   allow_agent=False, timeout=15, auth_timeout=15, banner_timeout=15)
    try:
        if _sha(client.get_transport().get_remote_server_key().asbytes()) != HOST_KEY:
            raise RuntimeError("SSH 主机密钥不匹配")
        with client.open_sftp() as sftp:
            with sftp.open(SOURCE, "rb") as stream:
                original_bytes = stream.read()
            if _sha(original_bytes) != EXPECTED_SHA256:
                raise RuntimeError("远端机械臂源码与审阅版本不一致，拒绝覆盖")
            patched = render_patch(original_bytes.decode("utf-8"))
            patched_bytes = patched.encode("utf-8")
            if not args.deploy:
                diff = difflib.unified_diff(original_bytes.decode("utf-8").splitlines(),
                                            patched.splitlines(), fromfile="arm_teach.py",
                                            tofile="arm_teach.py + s2 nudge", lineterm="")
                print("\n".join(diff))
                return
            execution = _remote_json(client,
                "curl --fail --silent --show-error http://127.0.0.1:8080/api/arm-teach/exec")
            if (execution.get("status") or {}).get("active"):
                raise RuntimeError("机械臂正在执行动作，拒绝替换源码")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = "/home/dachuang/grab_visual/deployment_backups"
            try:
                sftp.stat(backup_dir)
            except FileNotFoundError:
                sftp.mkdir(backup_dir)
            backup = f"{backup_dir}/arm_teach_before_s2_nudge_{stamp}.py"
            staged = f"/home/dachuang/grab_visual/.arm_teach_s2_nudge_{stamp}.py"
            with sftp.open(backup, "wb") as stream:
                stream.write(original_bytes)
            with sftp.open(staged, "wb") as stream:
                stream.write(patched_bytes)
            sftp.chmod(staged, sftp.stat(SOURCE).st_mode & 0o777)
            with sftp.open(staged, "rb") as stream:
                if _sha(stream.read()) != _sha(patched_bytes):
                    raise RuntimeError("暂存文件哈希不匹配")
            sftp.posix_rename(staged, SOURCE)
            with sftp.open(SOURCE, "rb") as stream:
                if _sha(stream.read()) != _sha(patched_bytes):
                    raise RuntimeError("部署后文件哈希不匹配；原文件备份已保留")
            print(json.dumps({"source": SOURCE, "backup": backup,
                              "before_sha256": EXPECTED_SHA256,
                              "after_sha256": _sha(patched_bytes),
                              "arm_or_chassis_started": False,
                              "service_restarted": False}, ensure_ascii=False, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
