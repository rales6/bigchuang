from __future__ import annotations

import os

import paramiko


def main() -> None:
    password = os.environ.get("RTX_SSH_PASSWORD")
    if not password:
        raise RuntimeError("RTX_SSH_PASSWORD is required")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "202.121.181.124",
        port=2020,
        username="sjtu",
        password=password,
        timeout=90,
        auth_timeout=90,
        banner_timeout=90,
    )
    command = (
        "pids=$(pgrep -f '[v]alidate_search_task_qwen.py' || true); "
        "if [ -n \"$pids\" ]; then kill $pids; sleep 2; fi; "
        "pgrep -af '[v]alidate_search_task_qwen.py' || true"
    )
    _, stdout, stderr = client.exec_command(command, timeout=30)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    client.close()
    if code:
        raise RuntimeError(error or f"cleanup failed with {code}")
    print(output or "RTX_SEARCH_VALIDATION_PROCESS_NOT_RUNNING")


if __name__ == "__main__":
    main()
