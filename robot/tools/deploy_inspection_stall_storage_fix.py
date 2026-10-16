"""Safely deploy the simulated inspection stall/storage fix to the Pi."""

from __future__ import annotations

import getpass
import os
from pathlib import Path
import shlex
import time

import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/robot/dachuang/car1.0"
FILES = (
    "raspberry_pi/map_camera_inspection.py",
    "raspberry_pi/mapping/adaptive_speed.py",
)


def run(client: paramiko.SSHClient, command: str, timeout: int = 120) -> str:
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if code:
        raise RuntimeError(error or output or f"remote command failed with {code}")
    return output


def main() -> None:
    password = os.environ.get("PI_SSH_PASSWORD") or getpass.getpass(
        "Pi SSH password: "
    )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "pi.example.invalid",
        username="dachuang",
        password=password,
        timeout=45,
        auth_timeout=45,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = f"{REMOTE_ROOT}/backups/inspection_stall_storage_{timestamp}"
    try:
        active = run(
            client,
            "pgrep -af '[r]aspberry_pi.map_camera_inspection' || true",
        ).strip()
        if active:
            raise RuntimeError(
                "inspection is still active; refusing hot replacement:\n" + active
            )
        run(client, f"mkdir -p {shlex.quote(backup)}")
        with client.open_sftp() as sftp:
            for relative in FILES:
                remote = f"{REMOTE_ROOT}/{relative}"
                destination = f"{backup}/{relative}"
                run(
                    client,
                    "mkdir -p " + shlex.quote(str(Path(destination).parent).replace("\\", "/")),
                )
                run(
                    client,
                    f"cp -p {shlex.quote(remote)} {shlex.quote(destination)}",
                )
                temporary = remote + ".codex-upload"
                sftp.put(str(ROOT / relative), temporary)
                run(client, f"mv {shlex.quote(temporary)} {shlex.quote(remote)}")

        verify = run(
            client,
            f"cd {shlex.quote(REMOTE_ROOT)} && "
            "source .venv-pi/bin/activate && "
            "python3 -m py_compile "
            "raspberry_pi/map_camera_inspection.py "
            "raspberry_pi/mapping/adaptive_speed.py && "
            "grep -q '20260905_155436' raspberry_pi/mapping/adaptive_speed.py && "
            "grep -q 'inspection_output_directory_disappeared' "
            "raspberry_pi/map_camera_inspection.py && "
            "curl -fsS --max-time 10 http://192.0.2.20:8010/health",
            timeout=90,
        )
        print("PI_INSPECTION_FIX_DEPLOYED")
        print(f"BACKUP={backup}")
        print("QWEN_HEALTH=" + verify.strip()[:800])
    finally:
        client.close()


if __name__ == "__main__":
    main()
