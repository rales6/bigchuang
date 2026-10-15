from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import shlex
import time

import paramiko


ROOT = Path(__file__).resolve().parents[1]


def connect(host: str, port: int, username: str, password: str) -> paramiko.SSHClient:
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


def run(client: paramiko.SSHClient, command: str, timeout: int = 120) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    if code:
        raise RuntimeError(error or output or f"remote command failed with {code}")
    return output


def inspect(client: paramiko.SSHClient, root: str, role: str) -> str:
    command = (
        f"echo ROLE={shlex.quote(role)}; hostname; echo ADDRESSES=$(hostname -I); "
        f"test -d {shlex.quote(root)} && echo ROOT_OK={shlex.quote(root)} || echo ROOT_MISSING={shlex.quote(root)}; "
        f"find {shlex.quote(root)} -maxdepth 4 -type f "
        r"\( -name 'run_pi_task_gateway.py' -o -name 'run_semantic_task_planner.py' "
        r"-o -name 'semantic_task_planner.py' -o -name 'scene_change_memory.py' \) "
        "-print 2>/dev/null | sort; "
        "echo PROCESSES; "
        "pgrep -af '[r]un_pi_task_gateway.py|[r]un_semantic_task_planner.py' || true; "
        "echo LISTENERS; ss -ltnp 2>/dev/null | grep -E ':(8010|802[0-9]|8030|8767)[[:space:]]' || true"
        "; for port in 8010 8020 8021 8022 8023 8024 8025 8026 8027 8028 8029 8030 8767; do "
        "pid=$(ss -ltnp 2>/dev/null | grep -E \"0\\.0\\.0\\.0:$port[[:space:]]|127\\.0\\.0\\.1:$port[[:space:]]\" "
        r"| sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1); "
        "if test -n \"$pid\"; then echo PORT_${port}_PID=$pid; "
        "ps -o pid=,ppid=,lstart=,cmd= -p $pid; "
        "echo CWD=$(readlink -f /proc/$pid/cwd); "
        "echo CMDLINE=$(tr '\\0' ' ' </proc/$pid/cmdline); fi; done"
    )
    return run(client, command, timeout=60)


def backup_and_upload(
    client: paramiko.SSHClient,
    remote_root: str,
    files: list[tuple[Path, str]],
    backup: str,
) -> None:
    destinations = [f"{remote_root}/{relative}" for _local, relative in files]
    parent_dirs = sorted({str(Path(path).parent).replace("\\", "/") for path in destinations})
    run(
        client,
        "mkdir -p " + " ".join(shlex.quote(item) for item in [backup, *parent_dirs]),
    )
    with client.open_sftp() as sftp:
        for local, relative in files:
            remote = f"{remote_root}/{relative}"
            backup_target = f"{backup}/{relative}"
            backup_parent = str(Path(backup_target).parent).replace("\\", "/")
            run(client, f"mkdir -p {shlex.quote(backup_parent)}")
            run(
                client,
                f"if test -f {shlex.quote(remote)}; then cp -p {shlex.quote(remote)} {shlex.quote(backup_target)}; fi",
            )
            temporary = remote + ".codex-upload"
            sftp.put(str(local), temporary)
            run(client, f"mv {shlex.quote(temporary)} {shlex.quote(remote)}")


def deploy(pi: paramiko.SSHClient, rtx: paramiko.SSHClient) -> tuple[str, str]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pi_root = "/home/dachuang/dachuang/car1.0"
    rtx_root = "/home/sjtu/3090"
    pi_files = [
        (ROOT / "3090/scripts/run_pi_task_gateway.py", "3090/scripts/run_pi_task_gateway.py"),
    ]
    rtx_files = [
        (ROOT / "3090/src/qwen_grounded_tracker/scene_change_memory.py", "src/qwen_grounded_tracker/scene_change_memory.py"),
        (ROOT / "3090/src/qwen_grounded_tracker/semantic_task_planner.py", "src/qwen_grounded_tracker/semantic_task_planner.py"),
        (ROOT / "3090/src/qwen_grounded_tracker/task_execution.py", "src/qwen_grounded_tracker/task_execution.py"),
        (ROOT / "3090/src/qwen_grounded_tracker/task_dry_run.py", "src/qwen_grounded_tracker/task_dry_run.py"),
        (ROOT / "3090/scripts/run_semantic_task_planner.py", "scripts/run_semantic_task_planner.py"),
        (ROOT / "3090/scripts/run_semantic_task_demo.py", "scripts/run_semantic_task_demo.py"),
        (ROOT / "3090/examples/demo_cleanup_scene.json", "examples/demo_cleanup_scene.json"),
        (ROOT / "3090/docs/SEMANTIC_TASK_PLANNING.md", "docs/SEMANTIC_TASK_PLANNING.md"),
    ]
    pi_backup = f"{pi_root}/backups/semantic_tasks_{timestamp}"
    rtx_backup = f"{rtx_root}/backups/semantic_tasks_{timestamp}"
    backup_and_upload(pi, pi_root, pi_files, pi_backup)
    backup_and_upload(rtx, rtx_root, rtx_files, rtx_backup)

    pi_python = run(
        pi,
        f"cd {shlex.quote(pi_root)} && "
        "if test -x .venv-pi/bin/python; then echo .venv-pi/bin/python; "
        "elif test -x .venv/bin/python; then echo .venv/bin/python; else echo python3; fi",
    ).strip()
    pi_verify = run(
        pi,
        f"cd {shlex.quote(pi_root)} && {pi_python} -m py_compile 3090/scripts/run_pi_task_gateway.py && "
        f"PYTHONPATH=3090/src {pi_python} -c "
        + shlex.quote(
            "import importlib.util; "
            "s=importlib.util.spec_from_file_location('gateway','3090/scripts/run_pi_task_gateway.py'); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "g=m.PiTaskGateway('http://127.0.0.1:8766',False,''); "
            "assert g.health()['ok']; print('PI_TASK_GATEWAY_DRY_IMPORT_OK')"
        ),
    )
    rtx_python = run(
        rtx,
        f"cd {shlex.quote(rtx_root)} && "
        "if test -x .venv/bin/python; then echo .venv/bin/python; else echo /usr/bin/python3; fi",
    ).strip()
    rtx_verify = run(
        rtx,
        f"cd {shlex.quote(rtx_root)} && PYTHONPATH=src {rtx_python} -m py_compile "
        "src/qwen_grounded_tracker/scene_change_memory.py "
        "src/qwen_grounded_tracker/semantic_task_planner.py "
        "src/qwen_grounded_tracker/task_execution.py "
        "src/qwen_grounded_tracker/task_dry_run.py scripts/run_semantic_task_planner.py && "
        f"PYTHONPATH=src {rtx_python} scripts/run_semantic_task_demo.py "
        "--backend rules --instruction '把蓝色水瓶放到工作桌上' "
        "--output outputs/deployment_semantic_task_dry_run.json >/tmp/semantic_task_deploy.log && "
        f"PYTHONPATH=src {rtx_python} -c "
        + shlex.quote(
            "import json; "
            "d=json.load(open('outputs/deployment_semantic_task_dry_run.json')); "
            "assert d['dry_run']['completed']; "
            "assert d['dry_run']['grasp_attempts']==d['dry_run']['grasp_successes']==1; "
            "print('RTX_NATURAL_TASK_DRY_RUN_OK')"
        ),
        timeout=180,
    )
    pi_ip = run(pi, "hostname -I | awk '{print $1}'").strip()
    if not pi_ip:
        raise RuntimeError("Raspberry Pi has no LAN address")

    pi_launch = run(
        pi,
        f"cd {shlex.quote(pi_root)} && mkdir -p logs && "
        "pid=$(ss -ltnp 2>/dev/null | grep -E ':8767[[:space:]]' "
        r"| sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1); "
        "if test -n \"$pid\"; then "
        "tr '\\0' ' ' </proc/$pid/cmdline | grep -q 'run_pi_task_gateway.py' || exit 44; "
        "kill -TERM $pid; for i in $(seq 1 20); do kill -0 $pid 2>/dev/null || break; sleep 0.2; done; fi; "
        f"nohup {pi_python} 3090/scripts/run_pi_task_gateway.py --host 0.0.0.0 --port 8767 "
        "--map-url http://127.0.0.1:8766 >logs/pi_task_gateway_8767.log 2>&1 </dev/null & "
        "newpid=$!; for i in $(seq 1 40); do "
        "body=$(curl -fsS http://127.0.0.1:8767/api/task/v1/health 2>/dev/null || true); "
        "echo \"$body\" | grep -q '\"hardware_enabled\": false' && "
        "echo PI_TASK_GATEWAY_SERVICE_OK pid=$newpid ip=" + shlex.quote(pi_ip) + " && exit 0; "
        "sleep 0.25; done; tail -40 logs/pi_task_gateway_8767.log; exit 1",
        timeout=90,
    )

    planner_command = (
        f"env PYTHONPATH=src {rtx_python} scripts/run_semantic_task_planner.py "
        "--backend rules --semantic-map examples/demo_cleanup_scene.json "
        "--scene-change-memory outputs/scene_change_overlay.json "
        f"--robot-url http://{pi_ip}:8767 --host 0.0.0.0 --port 8023"
    )
    rtx_launch = run(
        rtx,
        f"cd {shlex.quote(rtx_root)} && mkdir -p logs outputs && "
        "pid=$(ss -ltnp 2>/dev/null | grep -E ':8023[[:space:]]' "
        r"| sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1); "
        "if test -n \"$pid\"; then "
        "tr '\\0' ' ' </proc/$pid/cmdline | grep -q 'run_semantic_task_planner.py' || exit 45; "
        "kill -TERM $pid; for i in $(seq 1 20); do kill -0 $pid 2>/dev/null || break; sleep 0.2; done; fi; "
        f"nohup {planner_command} >logs/semantic_task_planner_8023.log 2>&1 </dev/null & "
        "newpid=$!; for i in $(seq 1 60); do "
        "body=$(curl -fsS http://127.0.0.1:8023/health 2>/dev/null || true); "
        "echo \"$body\" | grep -q '\"status\":\"ok\"' && "
        "echo RTX_SEMANTIC_PLANNER_SERVICE_OK pid=$newpid && exit 0; "
        "sleep 0.5; done; tail -60 logs/semantic_task_planner_8023.log; exit 1",
        timeout=120,
    )
    e2e_code = (
        "import json,urllib.request\n"
        "def post(path,payload):\n"
        "    b=json.dumps(payload,ensure_ascii=False).encode()\n"
        "    r=urllib.request.Request('http://127.0.0.1:8023'+path,data=b,headers={'Content-Type':'application/json'},method='POST')\n"
        "    return json.loads(urllib.request.urlopen(r,timeout=30).read())\n"
        "p=post('/plan',{'instruction':'把蓝色水瓶放到工作桌上'})['plan']\n"
        "e=post('/execute/start',{'plan':p,'auto_run':True})['execution']\n"
        "assert e['status']=='completed',e\n"
        "print('RTX_PI_NATURAL_TASK_E2E_OK steps='+str(e['total_steps']))"
    )
    e2e = run(
        rtx,
        f"cd {shlex.quote(rtx_root)} && {rtx_python} -c {shlex.quote(e2e_code)}",
        timeout=120,
    )
    arm_health = run(
        rtx,
        "body=$(curl -fsS http://127.0.0.1:8020/health 2>/dev/null || true); "
        "test -n \"$body\"; echo ARM_GRASP_SERVICE_HEALTH_OK; echo \"$body\" | head -c 800",
        timeout=45,
    )
    return (
        f"{pi_verify.strip()}\n{pi_launch.strip()}\nPI_BACKUP={pi_backup}",
        f"{rtx_verify.strip()}\n{rtx_launch.strip()}\n{e2e.strip()}\n{arm_health.strip()}\nRTX_BACKUP={rtx_backup}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy semantic natural-language task components")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    pi_password = os.environ.get("PI_SSH_PASSWORD") or getpass.getpass("Pi SSH password: ")
    rtx_password = os.environ.get("RTX_SSH_PASSWORD") or getpass.getpass("RTX SSH password: ")
    pi = connect("dachuang.local", 22, "dachuang", pi_password)
    rtx = connect("202.121.181.124", 2020, "sjtu", rtx_password)
    try:
        print(inspect(pi, "/home/dachuang/dachuang/car1.0", "raspberry_pi"))
        print(inspect(rtx, "/home/sjtu/3090", "rtx_3090"))
        if args.inspect_only:
            return
        pi_result, rtx_result = deploy(pi, rtx)
        print(pi_result)
        print(rtx_result)
        print("SEMANTIC_TASK_DEPLOYMENT_OK")
    finally:
        pi.close()
        rtx.close()


if __name__ == "__main__":
    main()
