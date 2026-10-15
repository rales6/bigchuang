from __future__ import annotations

import os
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]


def connect(host: str, username: str, password: str, port: int = 22):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        port=port,
        username=username,
        password=password,
        timeout=45,
        auth_timeout=45,
        banner_timeout=45,
    )
    return client


def upload(client, files: list[tuple[str, str]]) -> None:
    with client.open_sftp() as sftp:
        for local, remote in files:
            sftp.put(str(ROOT / local), remote)


def run_checked(client, command: str, timeout: int = 120) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if code:
        raise RuntimeError(error or output or f"remote command failed with {code}")
    return output


def main() -> None:
    pi_password = os.environ.get("PI_SSH_PASSWORD")
    rtx_password = os.environ.get("RTX_SSH_PASSWORD")
    if not pi_password or not rtx_password:
        raise RuntimeError("PI_SSH_PASSWORD and RTX_SSH_PASSWORD are required")

    pi = connect("dachuang.local", "dachuang", pi_password)
    upload(
        pi,
        [
            ("raspberry_pi/known_map_navigation.py", "/home/dachuang/dachuang/car1.0/raspberry_pi/known_map_navigation.py"),
            ("raspberry_pi/search_route_planner.py", "/home/dachuang/dachuang/car1.0/raspberry_pi/search_route_planner.py"),
            ("raspberry_pi/object_search_mission.py", "/home/dachuang/dachuang/car1.0/raspberry_pi/object_search_mission.py"),
            ("raspberry_pi/object_search_simulation.py", "/home/dachuang/dachuang/car1.0/raspberry_pi/object_search_simulation.py"),
            ("raspberry_pi/search_dashboard.py", "/home/dachuang/dachuang/car1.0/raspberry_pi/search_dashboard.py"),
            ("raspberry_pi/realtime_map.py", "/home/dachuang/dachuang/car1.0/raspberry_pi/realtime_map.py"),
            ("raspberry_pi/submit_search_command.py", "/home/dachuang/dachuang/car1.0/raspberry_pi/submit_search_command.py"),
            ("search_object.sh", "/home/dachuang/dachuang/car1.0/search_object.sh"),
            ("natural_search.sh", "/home/dachuang/dachuang/car1.0/natural_search.sh"),
            ("search_dashboard.sh", "/home/dachuang/dachuang/car1.0/search_dashboard.sh"),
            ("maps/search_decision_cabinet_red_bottle.json", "/home/dachuang/dachuang/car1.0/maps/search_decision_cabinet_red_bottle.json"),
        ],
    )
    pi_output = run_checked(
        pi,
        "cd ~/dachuang/car1.0 && source .venv-pi/bin/activate && "
        "chmod +x search_object.sh natural_search.sh search_dashboard.sh && python3 -m compileall -q raspberry_pi && "
        "./search_object.sh maps/search_decision_cabinet_red_bottle.json simulate && "
        "python3 -c \"import json,pathlib; p=sorted(pathlib.Path('search_runs/ble_scene_17').iterdir())[-1]/'mission.json'; d=json.load(open(p)); assert d['result']=='simulation_passed'; print('PI_SIMULATION_OK',p)\"",
        timeout=180,
    )
    pi.close()

    rtx = connect("202.121.181.124", "sjtu", rtx_password, port=2020)
    run_checked(rtx, "mkdir -p /home/sjtu/3090/src/qwen_grounded_tracker /home/sjtu/3090/remote_stage1_snapshot")
    upload(
        rtx,
        [
            ("3090/src/qwen_grounded_tracker/mission_supervisor.py", "/home/sjtu/3090/src/qwen_grounded_tracker/mission_supervisor.py"),
            ("3090/src/qwen_grounded_tracker/search_task_planner.py", "/home/sjtu/3090/src/qwen_grounded_tracker/search_task_planner.py"),
            ("3090/remote_stage1_snapshot/run_stage1_perception_server.py", "/home/sjtu/3090/remote_stage1_snapshot/run_stage1_perception_server.py"),
        ],
    )
    rtx_output = run_checked(
        rtx,
        "cd /home/sjtu/3090 && PYTHONPATH=src /usr/bin/python3 -m py_compile "
        "src/qwen_grounded_tracker/mission_supervisor.py "
        "src/qwen_grounded_tracker/search_task_planner.py "
        "remote_stage1_snapshot/run_stage1_perception_server.py && "
        "PYTHONPATH=src /usr/bin/python3 -c \"from qwen_grounded_tracker.mission_supervisor import MissionSupervisor; d=MissionSupervisor().next({'remaining_view_indices':[0],'latest_event':{'type':'observation_complete'},'detections':[]}); assert d.action=='navigate_next_view'; print('RTX_SUPERVISOR_IMPORT_OK')\"",
        timeout=120,
    )
    rtx.close()
    print(pi_output.strip())
    print(rtx_output.strip())
    print("OBJECT_SEARCH_DEPLOYMENT_OK")


if __name__ == "__main__":
    main()
