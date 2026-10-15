"""Backed-up integration deployment. Never starts a wheel or arm job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import time

from grasp_range_integration.tools.deploy_track_web_pick import _connect, HOSTS, _read, _mkdir, _remote

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "track_web_integration"
MANIFEST = Path(__file__).with_name("robot_skills_deploy_manifest.json")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def idle(client):
    lines = _remote(client, "ps -eo pid=,args=").splitlines()
    markers = ("relationship_find_only_entry.py", "verified_search", "pursuit_session",
               "run_visual_pursuit.py", "hull_micro_adjust", "bounded_motion", "region_navigation --run-job")
    tasks = [line for line in lines if any(word in line for word in markers)]
    if tasks:
        raise RuntimeError("有实车任务在运行，拒绝替换代码：" + repr(tasks))
    code = "import json,urllib.request; v=json.load(urllib.request.urlopen('http://127.0.0.1:8780/state',timeout=5)); print(json.dumps({'managed':v.get('managed'),'skills':v.get('robot_skills'),'grasp':v.get('grasp')}))"
    result = json.loads(_remote(client, shlex.quote(HOSTS['pi']['python']) + " -c " + shlex.quote(code)))
    managed = result.get("managed") or {}
    if managed.get("running") and managed.get("kind") != "camera_preview":
        raise RuntimeError("桥服务有正在运行的非预览任务")
    execution = ((result.get("grasp") or {}).get("data") or {}).get("arm_teach_exec") or {}
    if execution.get("active"):
        raise RuntimeError("机械臂仍在运行，拒绝服务重载")
    if (result.get("grasp") or {}).get("connected"):
        probe = "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8080/api/arm-teach/exec',timeout=4))))"
        arm = json.loads(_remote(client, shlex.quote(HOSTS['pi']['python']) + " -c " + shlex.quote(probe)))
        if (arm.get("status") or {}).get("active"):
            raise RuntimeError("机械臂执行队列仍在运行，拒绝重载")
    if (result.get("skills") or {}).get("status") in {"carrying", "placing", "placement_unknown"}:
        raise RuntimeError("有尚未释放的携带任务，拒绝部署中覆盖交接服务")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--pi-host", help="Current Pi IPv4/IPv6; pinned host key must still match")
    parser.add_argument("--activate", action="store_true", help="Reload ONLY idle web/bridge services")
    args = parser.parse_args()
    if args.pi_host:
        HOSTS["pi"]["host"] = args.pi_host
    if args.activate and not args.deploy:
        parser.error("--activate requires --deploy")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clients, prepared, activated = {}, [], []
    stamp = time.strftime("%Y%m%d_%H%M%S")
    protected = {}
    stopped = []
    try:
        for side in ("pi", "gpu"):
            clients[side] = _connect(HOSTS[side])
        idle(clients["pi"])
        with clients["pi"].open_sftp() as sftp:
            for name in ("object_search_mission.py", "known_map_navigation.py"):
                path = "/home/dachuang/dachuang/car1.0/raspberry_pi/" + name
                protected[path] = sha(_read(sftp, path))
        files = []
        for side, names in (("pi", ["robot_bridge.py", "map_navigation_adapter.py"]),
                            ("gpu", ["planner.py", "robot_client.py", "qwen_hub.py", "app.py", "plan_adaptation.py"])):
            for name in names:
                meta = manifest[side][name]
                files.append((side, meta["path"], (SOURCE / side / name).read_bytes(), meta["sha256"]))
        for side, names in (("pi", ["region_navigation.py", "region_localization.py", "carry_localization.py", "robot_skills.py", "skill_planning.py", "placement_extension.py"]),
                            ("gpu", ["robot_skill_catalog.py"])):
            for name in names:
                remote = HOSTS[side]["root"] + "/final_work/" + side + "/" + name
                files.append((side, remote, (SOURCE / side / name).read_bytes(), None))
        for name, local in (("entry", "relationship_find_only_entry.py"), ("visibility", "visibility_route_planner.py")):
            meta = manifest["pi"][name]
            files.append(("pi", meta["path"], (ROOT / "phase1" / local).read_bytes(), meta["sha256"]))
        files.append(("pi", "/home/dachuang/grab_visual/placement_extension.py",
                      (SOURCE / "pi/placement_extension.py").read_bytes(), None))
        for side in clients:
            files.append((side, HOSTS[side]["root"] + "/final_work/ROBOT_SKILLS.md",
                          (ROOT / "ROBOT_SKILLS.md").read_bytes(), None))
        arm_meta = manifest["pi"]["arm"]
        with clients["pi"].open_sftp() as sftp:
            original = _read(sftp, arm_meta["path"])
        marker = b"    app.register_blueprint(bp)"
        addition = (b"    from placement_extension import register_placement\n"
                    b"    register_placement(bp, __import__('sys').modules[__name__], ble_send, jpegs)\n")
        if original.count(marker) != 1:
            raise RuntimeError("机械臂插件注册锚点不是一个")
        desired = original if addition in original else original.replace(marker, addition + marker, 1)
        files.append(("pi", arm_meta["path"], desired, arm_meta["sha256"]))
        js_meta = manifest["gpu"]["js"]
        with clients["gpu"].open_sftp() as sftp:
            original = _read(sftp, js_meta["path"])
        text = original.decode("utf-8")
        if "place_on_floor:" not in text:
            text, count = re.subn(r"(locate_destination:\s*'[^']*',\s*)(release:)",
                                 r"\1place_on_floor: '地面放置', \2", text)
            if count != 1:
                raise RuntimeError("网页动作标签锚点不是一个")
        files.append(("gpu", js_meta["path"], text.encode("utf-8"), js_meta["sha256"]))
        for side, remote, desired, expected in files:
            with clients[side].open_sftp() as sftp:
                try:
                    current = _read(sftp, remote)
                except FileNotFoundError:
                    current = None
                if current is not None and sha(current) not in {sha(desired), expected}:
                    raise RuntimeError("远端版本与审核版本不同，拒绝覆盖：" + remote)
                if current is None and expected is not None:
                    raise RuntimeError("审核文件缺失：" + remote)
                backup_root = HOSTS[side]["root"] + "/final_work/deployment_backups/robot_skills_" + stamp
                staged = backup_root + "/" + str(len(prepared)) + "_" + Path(remote).name
                prepared.append((side, remote, desired, current, staged))
                if args.deploy:
                    _mkdir(sftp, backup_root)
                    if current is not None:
                        with sftp.open(staged + ".original", "wb") as handle:
                            handle.write(current)
                    with sftp.open(staged, "wb") as handle:
                        handle.write(desired)
                    if sha(_read(sftp, staged)) != sha(desired):
                        raise RuntimeError("暂存哈希不一致：" + remote)
        if not args.deploy:
            print(json.dumps({"checked": [item[1] for item in prepared], "hardware_started": False}))
            return
        # Compile staged files BEFORE replacing or reloading anything.
        for side in clients:
            paths = [staged for host, remote, desired, current, staged in prepared
                     if host == side and remote.endswith(".py")]
            _remote(clients[side], shlex.quote(HOSTS[side]["python"]) + " -m py_compile " +
                    " ".join(shlex.quote(path) for path in paths))
        idle(clients["pi"])
        if args.activate:
            _remote(clients["gpu"], "bash /home/sjtu/track/final_work/stop_3090.sh")
            stopped.append("gpu")
            _remote(clients["pi"], "sudo -n systemctl stop final-work-pi-bridge.service")
            stopped.append("pi")
        for side, remote, desired, current, staged in prepared:
            with clients[side].open_sftp() as sftp:
                # Recheck for concurrent edits immediately before replacement.
                try: now = _read(sftp, remote)
                except FileNotFoundError: now = None
                if now != current:
                    raise RuntimeError("暂存期间远端被修改：" + remote)
                _mkdir(sftp, str(Path(remote).parent).replace("\\", "/"))
                sftp.posix_rename(staged, remote)
                activated.append((side, remote, current, staged))
                if sha(_read(sftp, remote)) != sha(desired):
                    raise RuntimeError("激活哈希错误：" + remote)
        with clients["pi"].open_sftp() as sftp:
            for path, expected in protected.items():
                if sha(_read(sftp, path)) != expected:
                    raise RuntimeError("冻结导航文件发生变化：" + path)
        if args.activate:
            _remote(clients["pi"], "sudo -n systemctl start final-work-pi-bridge.service")
            # Requires=bridge stops the reverse tunnel during a bridge reload.
            _remote(clients["pi"], "sudo -n systemctl start final-work-pi-tunnel.service")
            _remote(clients["gpu"], "bash /home/sjtu/track/final_work/run_3090_background.sh")
            stopped.clear()
        print(json.dumps({"deployed": [item[1] for item in prepared], "backup": "robot_skills_" + stamp,
                          "frozen_navigation_unchanged": True, "services_reloaded": args.activate,
                          "hardware_started": False}, ensure_ascii=False, indent=2))
    except Exception:
        for side, remote, original, staged in reversed(activated):
            with clients[side].open_sftp() as sftp:
                # Keep failed/new files recoverable; do not recursively delete.
                sftp.posix_rename(remote, staged + ".failed")
                if original is not None:
                    sftp.posix_rename(staged + ".original", remote)
        if "pi" in stopped:
            _remote(clients["pi"], "sudo -n systemctl start final-work-pi-bridge.service")
            _remote(clients["pi"], "sudo -n systemctl start final-work-pi-tunnel.service")
        if "gpu" in stopped:
            _remote(clients["gpu"], "bash /home/sjtu/track/final_work/run_3090_background.sh")
        raise
    finally:
        for client in clients.values(): client.close()


if __name__ == "__main__":
    main()
