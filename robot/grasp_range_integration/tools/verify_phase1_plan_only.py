"""Run one explicitly approved DeepSeek text plan on Pi; never start hardware."""

from __future__ import annotations

import base64
import argparse
import hashlib
import json
import os
import shlex

import paramiko


INSTRUCTION = "去柜子边上夹取一个红色的小物块"
HOST = "192.0.2.13"
HOST_KEY_SHA256 = "1c624d9d6c33aa0482a4f043a4b17b295b9b64dd232c3e3fadce4d6436594ee6"
PYTHON = "/home/robot/dachuang/.venv/bin/python"
TRACK = "/home/robot/dachuang/track"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-approved-plan", action="store_true",
                        help="仅在用户明确授权后发送固定指令做一次规划")
    parser.add_argument("--inspect-decision", metavar="ABSOLUTE_PATH",
                        help="只读取已有规划文件，不再调用 DeepSeek")
    parser.add_argument("--inspect-service", action="store_true",
                        help="只读取 Pi 服务启动脚本和进程，不调用 DeepSeek")
    parser.add_argument("--offline-tests", action="store_true",
                        help="仅运行已部署代码的离线单元测试")
    args = parser.parse_args()
    if not (args.run_approved_plan or args.inspect_decision or
            args.inspect_service or args.offline_tests):
        parser.error("必须明确选择操作；默认不调用外部 API")
    password = os.environ.get("PI_SSH_PASSWORD")
    if not password:
        raise RuntimeError("缺少 PI_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="dachuang", password=password,
                   look_for_keys=False, allow_agent=False, timeout=15,
                   auth_timeout=15, banner_timeout=15)
    try:
        key_hash = hashlib.sha256(
            client.get_transport().get_remote_server_key().asbytes()
        ).hexdigest()
        if key_hash != HOST_KEY_SHA256:
            raise RuntimeError("树莓派 SSH 主机密钥不匹配")
        if args.offline_tests:
            car = "/home/robot/dachuang/car1.0"
            release = car + "/ble_scene_17_relationship_20260919"
            command = (
                "cd " + shlex.quote(car) + " && PYTHONPATH=" +
                shlex.quote(release + ":" + car) + " " +
                shlex.quote(car + "/.venv-pi/bin/python3") +
                " -m unittest grasp_range_integration.tests.test_phase1_ble_startup_recovery"
                " grasp_range_integration.tests.test_phase1_verified_search"
                " grasp_range_integration.tests.test_track_web_phase1_pick -q"
            )
            _stdin, stdout, stderr = client.exec_command(command, timeout=45)
            output = stdout.read().decode("utf-8", "replace")
            errors = stderr.read().decode("utf-8", "replace")
            if stdout.channel.recv_exit_status():
                raise RuntimeError("Pi 离线测试失败：" + output[-600:] + errors[-800:])
            print((output + errors).strip())
            return
        if args.inspect_service:
            _stdin, stdout, stderr = client.exec_command("ps -eo pid=,args=", timeout=10)
            if stdout.channel.recv_exit_status():
                raise RuntimeError(stderr.read().decode("utf-8", "replace"))
            lines = stdout.read().decode("utf-8", "replace").splitlines()
            print("\n".join(line for line in lines if
                            "final_work" in line or "reverse_tunnel" in line))
            _stdin, stdout, stderr = client.exec_command(
                "curl --fail --silent --show-error http://127.0.0.1:8780/state", timeout=10)
            if stdout.channel.recv_exit_status():
                raise RuntimeError(stderr.read().decode("utf-8", "replace"))
            state = json.loads(stdout.read().decode("utf-8"))
            print(json.dumps({"managed": state.get("managed")}, ensure_ascii=False))
            with client.open_sftp() as sftp:
                with sftp.open(TRACK + "/final_work/run_pi_background.sh", "rb") as stream:
                    print(stream.read().decode("utf-8", "replace"))
            return
        if args.inspect_decision:
            expected_root = TRACK + "/final_work/data/navigation/"
            if not args.inspect_decision.startswith(expected_root):
                raise ValueError("只能读取网页规划目录内的决策文件")
            with client.open_sftp() as sftp:
                with sftp.open(args.inspect_decision, "rb") as stream:
                    decision = json.load(stream)
            print(json.dumps({
                "status": decision.get("status"),
                "target": decision.get("target"),
                "anchor": decision.get("anchor"),
                "phase_control": decision.get("phase_control"),
                "planning_audit": {key: (decision.get("planning_audit") or {}).get(key)
                                   for key in ("motion_started", "camera_started", "arm_started")},
            }, ensure_ascii=False))
            return
        encoded = base64.b64encode(INSTRUCTION.encode("utf-8")).decode("ascii")
        code = (
            "import base64,json; "
            "from final_work.pi.map_navigation_adapter import MapNavigationAdapter; "
            f"instruction=base64.b64decode('{encoded}').decode('utf-8'); "
            "decision,path=MapNavigationAdapter().prepare_phase1_pick(instruction); "
            "print(json.dumps({'status':decision.get('status'),"
            "'target':decision.get('target'),"
            "'search_region_id':(decision.get('search_region') or {}).get('id'),"
            "'phase_control':decision.get('phase_control'),"
            "'decision_path':str(path),"
            "'motion_started':(decision.get('planning_audit') or {}).get('motion_started')},"
            "ensure_ascii=False))"
        )
        command = ("cd " + shlex.quote(TRACK) +
                   " && FINAL_WORK_ALLOW_DEEPSEEK_SEMANTIC_UPLOAD=1 " +
                   shlex.quote(PYTHON) + " -c " + shlex.quote(code))
        _stdin, stdout, stderr = client.exec_command(command, timeout=155)
        output = stdout.read().decode("utf-8", "replace")
        errors = stderr.read().decode("utf-8", "replace")
        result = stdout.channel.recv_exit_status()
        if result:
            raise RuntimeError("只规划验证失败：" + errors[-700:] + output[-300:])
        print(output.strip())
    finally:
        client.close()


if __name__ == "__main__":
    main()
