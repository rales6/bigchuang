from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/sjtu/3090_stage1"


def main() -> None:
    password = os.environ["RTX_SSH_PASSWORD"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        "202.121.181.124", port=2020, username="sjtu", password=password,
        timeout=45, auth_timeout=45, banner_timeout=45,
    )
    command = (
        "pid=$(pgrep -f '[r]un_stage1_perception_server.py.*--port 8011' | head -1); "
        "test -n \"$pid\"; "
        "echo PID=$pid; "
        "ps -o ppid= -p $pid; "
        "tr '\\0' ' ' </proc/$pid/cmdline; echo; "
        "readlink -f /proc/$pid/cwd"
    )
    _, stdout, stderr = client.exec_command(command, timeout=30)
    code = stdout.channel.recv_exit_status()
    details = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if code:
        client.close()
        raise RuntimeError(error or details or "8011 process not found")
    lines = [line.strip() for line in details.splitlines() if line.strip()]
    pid = int(lines[0].split("=", 1)[1])
    parent_pid = int(lines[1])
    cmdline = lines[2]
    cwd = lines[3]
    if cwd != REMOTE or "--port 8011" not in cmdline:
        client.close()
        raise RuntimeError(f"unexpected 8011 process: cwd={cwd}, cmd={cmdline}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup = f"{REMOTE}/backup_before_mission_api_{timestamp}"
    check = (
        f"mkdir -p {shlex.quote(backup)} && "
        f"cp {REMOTE}/scripts/run_stage1_perception_server.py {shlex.quote(backup)}/ && "
        f"test -f {REMOTE}/src/qwen_grounded_tracker/search_task_planner.py || true && "
        f"cp -f {REMOTE}/src/qwen_grounded_tracker/search_task_planner.py {shlex.quote(backup)}/ 2>/dev/null || true"
    )
    _, out, err = client.exec_command(check, timeout=30)
    if out.channel.recv_exit_status():
        client.close()
        raise RuntimeError(err.read().decode("utf-8", errors="replace"))

    with client.open_sftp() as sftp:
        sftp.put(
            str(ROOT / "3090/remote_stage1_snapshot/run_stage1_perception_server.py"),
            f"{REMOTE}/scripts/run_stage1_perception_server.py",
        )
        for name in ("mission_supervisor.py", "search_task_planner.py"):
            sftp.put(
                str(ROOT / f"3090/src/qwen_grounded_tracker/{name}"),
                f"{REMOTE}/src/qwen_grounded_tracker/{name}",
            )

    python_path = "/home/sjtu/3090/.venv/bin/python"
    compile_command = (
        f"cd {REMOTE} && PYTHONPATH=src {python_path} -m py_compile "
        "scripts/run_stage1_perception_server.py "
        "src/qwen_grounded_tracker/mission_supervisor.py "
        "src/qwen_grounded_tracker/search_task_planner.py"
    )
    _, out, err = client.exec_command(compile_command, timeout=45)
    if out.channel.recv_exit_status():
        client.close()
        raise RuntimeError(err.read().decode("utf-8", errors="replace"))

    # Restart exactly the identified 8011 process.  Do not accept a still-live
    # old listener as a successful deployment: first allow graceful shutdown,
    # then terminate the same PID forcefully if it did not exit, and finally
    # verify that the new PID differs from the old one.
    launch = (
        f"cd {REMOTE}; kill -TERM {pid}; "
        "for i in $(seq 1 20); do kill -0 " + str(pid) + " 2>/dev/null || break; sleep 0.5; done; "
        f"if kill -0 {pid} 2>/dev/null; then kill -KILL {pid}; fi; "
        f"for i in $(seq 1 20); do kill -0 {pid} 2>/dev/null || break; sleep 0.25; done; "
        f"nohup {cmdline} >{REMOTE}/stage1_8011.log 2>&1 </dev/null & echo NEW_PID=$!"
    )
    _, out, err = client.exec_command(launch, timeout=45)
    if out.channel.recv_exit_status():
        client.close()
        raise RuntimeError(err.read().decode("utf-8", errors="replace"))
    # Localhost verifies route registration without requiring public port routing.
    verify = (
        "for i in $(seq 1 60); do "
        "body=$(curl -fsS http://127.0.0.1:8011/openapi.json 2>/dev/null || true); "
        "newpid=$(pgrep -f '[r]un_stage1_perception_server.py.*--port 8011' | head -1); "
        "if echo \"$body\" | grep -q '/mission/next' && echo \"$body\" | grep -q '/mission/command' "
        f"&& [ -n \"$newpid\" ] && [ \"$newpid\" != \"{pid}\" ]; "
        "then echo STAGE1_MISSION_API_OK new_pid=$newpid; exit 0; fi; sleep 2; done; exit 1"
    )
    _, out, err = client.exec_command(verify, timeout=140)
    code = out.channel.recv_exit_status()
    output = out.read().decode("utf-8", errors="replace")
    error = err.read().decode("utf-8", errors="replace")
    client.close()
    if code:
        raise RuntimeError(error or "8011 did not expose mission APIs after restart")
    print(output.strip())
    print(f"BACKUP={backup}")


if __name__ == "__main__":
    main()
