from __future__ import annotations

import os
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/dachuang/dachuang/car1.0"


def main() -> None:
    password = os.environ.get("PI_SSH_PASSWORD")
    if not password:
        raise RuntimeError("PI_SSH_PASSWORD is required")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("dachuang.local", username="dachuang", password=password, timeout=15)
    files = (
        "raspberry_pi/search_route_planner.py",
        "plan_search_route.sh",
        "maps/search_decision_cabinet_red_bottle.json",
        "maps/search_route_cabinet_red_bottle_sim.json",
        "maps/search_route_cabinet_red_bottle_sim.png",
    )
    with client.open_sftp() as sftp:
        for relative in files:
            sftp.put(str(ROOT / relative), f"{REMOTE_ROOT}/{relative}")
    command = (
        f"cd {REMOTE_ROOT} && "
        "source .venv-pi/bin/activate && "
        "chmod +x plan_search_route.sh && "
        "python3 -m py_compile raspberry_pi/search_route_planner.py && "
        "python3 -m raspberry_pi.search_route_planner "
        "--map maps/ble_scene_17.pgm --map-yaml maps/ble_scene_17.yaml "
        "--decision maps/search_decision_cabinet_red_bottle.json "
        "--output maps/search_route_cabinet_red_bottle_pi_check.json "
        "--start-x 0 --start-y 0 >/tmp/search_route_check.log && "
        "python3 -m json.tool maps/search_route_cabinet_red_bottle_pi_check.json >/dev/null"
    )
    _, stdout, stderr = client.exec_command(command, timeout=90)
    code = stdout.channel.recv_exit_status()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    client.close()
    if code:
        raise RuntimeError(error or f"remote validation failed with exit code {code}")
    print("PI_SEARCH_ROUTE_SYNC_AND_OFFLINE_VALIDATION_OK")


if __name__ == "__main__":
    main()
