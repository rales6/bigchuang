from __future__ import annotations

import os
import paramiko


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("dachuang.local", username="dachuang", password=os.environ["PI_SSH_PASSWORD"], timeout=45)
    command = (
        "cd ~/dachuang/car1.0 && source .venv-pi/bin/activate && "
        "curl -fsS --max-time 10 http://192.168.55.33:8011/health >/dev/null && "
        "./natural_search.sh '去柜子边上找一个红色瓶子' simulate"
    )
    _, out, err = c.exec_command(command, timeout=300)
    code = out.channel.recv_exit_status()
    output = out.read().decode("utf-8", errors="replace")
    error = err.read().decode("utf-8", errors="replace")
    c.close()
    print(output)
    print(error)
    if code:
        raise RuntimeError(f"end-to-end validation failed with {code}")
    if '"status": "executable"' not in output:
        raise RuntimeError("3090 did not return executable decision")
    print("PI_TO_3090_TO_PI_SIMULATION_OK")


if __name__ == "__main__":
    main()
