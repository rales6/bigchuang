"""Deploy only the final_work integration adapters; never start a robot task.

The current remote files are compared with the copies audited on 2026-09-29.
An unexpected remote edit aborts rather than overwriting another student's
work.  All replaced files are backed up before any activation.  This script
does not restart either service; activation is a separate, visible operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import time

import paramiko


ROOT = Path(__file__).resolve().parents[1] / "track_web_integration"
HOSTS = {
    "pi": {
        "host": "192.0.2.13", "port": 22, "user": "dachuang",
        "password_env": "PI_SSH_PASSWORD",
        "host_key_sha256": "1c624d9d6c33aa0482a4f043a4b17b295b9b64dd232c3e3fadce4d6436594ee6",
        "root": "/home/robot/dachuang/track",
        "python": "/home/robot/dachuang/.venv/bin/python",
        "files": {
            "map_navigation_adapter.py": "147dcaa8da229beb82f6e67a25187d3e9a469f65394d27303cfc104c13fb651b",
            "robot_bridge.py": "bcc2b8c8d60cdeab525b25eea6d1fa9c1e6ea97b8b0f5d5e46e01be08af0e022",
        },
        "known_predecessors": {
            "robot_bridge.py": {
                "f8fbe688adbe47b598747a9a0dfa19b302d48fefcbc99c1fe6c5bfecc481f449",
                "f495e8c392ac015118189bb222541950b8a48efccefb520ad3f4588bd6d32ad5",
            },
            "map_navigation_adapter.py": {
                "2e1eacae0ce37d4386e742298205e2eb904fcf4589f72f7110d5c05ae55d9080",
                "d98e57ad3ea326bafe592fc9e2bc3ccd3463e42d3a8727c2ef6669f024b21b94",
                "9a2aa0fecf8399ab4fab522f963a12db719e9821723454b9a20ef556cb52842b",
                "033cc9c7e5286e9f6a6d8017eefb3d9eb3d68940d7fc28ba995b4ab9ec29f6fe",
            },
        },
    },
    "gpu": {
        "host": "198.51.100.20", "port": 2020, "user": "sjtu",
        "password_env": "GPU_SSH_PASSWORD",
        "host_key_sha256": "11f798721ea24418ee4048fc6294714dd0592c5a10367084e4605c8a04b2d5b0",
        "root": "/home/gpu/track",
        "python": "/home/gpu/3090/.venv/bin/python",
        "files": {
            "plan_adaptation.py": "b758f20ce0aa76ad31225a3ad82827959b396da5851ebb812cd18804efc6bf7c",
        },
    },
}


def _remote(client, command: str) -> str:
    _stdin, stdout, stderr = client.exec_command(command, timeout=30)
    output = stdout.read().decode("utf-8", "replace")
    errors = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"远端校验失败（{code}）：{output[-500:]} {errors[-500:]}")
    return output


def _mkdir(sftp, path: str):
    current = ""
    for part in PurePosixPath(path).parts[1:]:
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def _read(sftp, path: str) -> bytes:
    with sftp.open(path, "rb") as stream:
        return stream.read()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _connect(spec: dict) -> paramiko.SSHClient:
    password = os.environ.get(spec["password_env"])
    if not password:
        raise RuntimeError("缺少部署口令环境变量：" + spec["password_env"])
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(spec["host"], port=spec["port"], username=spec["user"],
                   password=password, look_for_keys=False, allow_agent=False,
                   timeout=15, auth_timeout=15, banner_timeout=15)
    fingerprint = _sha(client.get_transport().get_remote_server_key().asbytes())
    if fingerprint != spec["host_key_sha256"]:
        client.close()
        raise RuntimeError("SSH 主机密钥指纹不匹配；拒绝部署到未知设备")
    return client


def _destination(name: str, spec: dict) -> str:
    side = "pi" if name == "pi" else "gpu"
    return spec["root"] + "/final_work/" + side + "/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--only", action="append",
                        choices=["pi/robot_bridge.py", "pi/map_navigation_adapter.py",
                                 "gpu/plan_adaptation.py"],
                        help="Limit checks and deployment to this adapter; may be repeated")
    args = parser.parse_args()
    selected = set(args.only or [f"{side}/{filename}" for side, spec in HOSTS.items()
                                 for filename in spec["files"]])
    clients = {}
    prepared = []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    try:
        for name, spec in HOSTS.items():
            if not any(item.startswith(name + "/") for item in selected):
                continue
            client = _connect(spec)
            clients[name] = client
            if name == "pi":
                process_lines = _remote(client, "ps -eo pid=,args=").splitlines()
                active = [line.strip() for line in process_lines if any(marker in line for marker in (
                    "relationship_find_only_entry.py", "object_search_mission.py",
                    "run_visual_pursuit.py", "pursuit_session", "hull_micro_adjust.py",
                    "probe_bounded_turn", "bounded_motion.py"))]
                if active:
                    raise RuntimeError("树莓派仍有实车任务，拒绝替换集成层：" + str(active))
            with client.open_sftp() as sftp:
                for filename, original_hash in spec["files"].items():
                    if f"{name}/{filename}" not in selected:
                        continue
                    remote_path = _destination(name, spec) + filename
                    local_path = ROOT / name / filename
                    current = _read(sftp, remote_path)
                    desired = local_path.read_bytes()
                    allowed = {original_hash, _sha(desired)} | spec.get(
                        "known_predecessors", {}).get(filename, set())
                    if _sha(current) not in allowed:
                        raise RuntimeError(f"远端已被其他人修改，拒绝覆盖：{remote_path}")
                    prepared.append((name, remote_path, local_path,
                                     _sha(current), _sha(desired)))
        print(json.dumps({"checked": [
            {"host": name, "remote_path": remote_path, "local_path": str(local_path),
             "remote_sha256": current_hash, "desired_sha256": desired_hash}
            for name, remote_path, local_path, current_hash, desired_hash in prepared
        ], "active_robot_task": False,
                          "services_restarted": False}, ensure_ascii=False, indent=2))
        if not args.deploy:
            return
        activated = []
        try:
            for name, remote_path, local_path, current_hash, desired_hash in prepared:
                if current_hash == desired_hash:
                    continue
                spec = HOSTS[name]
                backup_root = spec["root"] + "/final_work/deployment_backups/phase1_pick_" + stamp
                with clients[name].open_sftp() as sftp:
                    _mkdir(sftp, backup_root)
                    backup = backup_root + "/" + local_path.name + ".original"
                    staged = backup_root + "/" + local_path.name + ".staged"
                    with sftp.open(backup, "wb") as stream:
                        stream.write(_read(sftp, remote_path))
                    with sftp.open(staged, "wb") as stream:
                        stream.write(local_path.read_bytes())
                    if _sha(_read(sftp, staged)) != desired_hash:
                        raise RuntimeError("暂存文件哈希不匹配：" + remote_path)
                    sftp.posix_rename(staged, remote_path)
                    activated.append((name, remote_path, backup))
            for name, remote_path, _local_path, _current_hash, desired_hash in prepared:
                with clients[name].open_sftp() as sftp:
                    if _sha(_read(sftp, remote_path)) != desired_hash:
                        raise RuntimeError("目标文件哈希不匹配：" + remote_path)
            for name, spec in HOSTS.items():
                if name not in clients:
                    continue
                files = [path for host, path, *_ in prepared if host == name]
                command = (shlex.quote(spec["python"]) + " -m py_compile " +
                           " ".join(shlex.quote(path) for path in files))
                _remote(clients[name], command)
        except Exception:
            for name, remote_path, backup in reversed(activated):
                with clients[name].open_sftp() as sftp:
                    failed = backup + ".failed"
                    sftp.posix_rename(remote_path, failed)
                    sftp.posix_rename(backup, remote_path)
            raise
        print(json.dumps({"deployed": [path for _name, path, _backup in activated],
                          "backup_suffix": "phase1_pick_" + stamp,
                          "remote_syntax_checked": True,
                          "services_restarted": False,
                          "hardware_started": False}, ensure_ascii=False, indent=2))
    finally:
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    main()
