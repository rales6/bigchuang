from __future__ import annotations

import os
import paramiko


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("198.51.100.20", port=2020, username="sjtu", password=os.environ["RTX_SSH_PASSWORD"], timeout=45, auth_timeout=45, banner_timeout=45)
    command = (
        "echo PROCESS; pgrep -af 'run_stage1_perception_server.py.*8011' || true; "
        "echo PORT; ss -ltnp 2>/dev/null | grep ':8011' || true; "
        "echo LOG; tail -120 /home/gpu/3090_stage1/stage1_8011.log 2>/dev/null || true; "
        "echo BACKUPS; ls -dt /home/gpu/3090_stage1/backup_before_mission_api_* 2>/dev/null | head -3"
    )
    _, out, err = client.exec_command(command, timeout=40)
    code = out.channel.recv_exit_status()
    print(out.read().decode("utf-8", errors="replace"))
    print(err.read().decode("utf-8", errors="replace"))
    client.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
