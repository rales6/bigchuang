"""Check Pi parity; only deploy explicitly after a passing closed-loop report.

Never starts robot motion. Existing map/inspection launcher remain untouched.
"""
import argparse
import ast
import difflib
import getpass
import hashlib
import json
from pathlib import Path
import posixpath
import shlex
import time

import paramiko

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/robot/dachuang/car1.0"
FILES = ("raspberry_pi/autonomous_mapping.py", "raspberry_pi/joint_mapping.py",
         "raspberry_pi/mapping/explorer.py", "map_and_annotate.sh", "raspberry_pi/docs/joint_mapping.md")
DEPENDENCIES = [p.relative_to(ROOT).as_posix() for p in (ROOT / "raspberry_pi/mapping").glob("*.py") if p.relative_to(ROOT).as_posix() not in FILES]
DEPENDENCIES += ["raspberry_pi/config.py", "raspberry_pi/map_camera_inspection.py",
                 "raspberry_pi/known_map_navigation.py",
                 "raspberry_pi/lidar/n10_driver.py", "raspberry_pi/lidar/base.py"]


def command(client, value):
    _, stdout, stderr = client.exec_command(value, timeout=60)
    output, error = stdout.read().decode(), stderr.read().decode()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error or output)
    return output


def digest(data):
    # Cross-platform checkout line endings do not change Python semantics.
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def mapping_config_equivalent(first, second):
    # The Pi lacks an unrelated optional visual-localization dataclass. Do
    # not overwrite it merely to deploy mapping. Everything else must match.
    def canonical(content):
        tree = ast.parse(content.decode())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ClassDef) and node.name == "VisualLocalizationConfig")]
        return ast.dump(tree, include_attributes=False)
    return canonical(first) == canonical(second)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="pi.example.invalid")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--show-config-diff", action="store_true")
    parser.add_argument("--show-source-diff", action="store_true")
    parser.add_argument("--report", default="simulation_runs/joint_closed_loop_long/report.json")
    args = parser.parse_args()
    if args.deploy:
        report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
        if not report.get("passed"):
            raise RuntimeError("closed-loop acceptance did not pass; deployment refused")
        if not report.get("source_sha256"):
            raise RuntimeError("report lacks tested source hashes")
        for relative, expected in report["source_sha256"].items():
            if digest((ROOT / relative).read_bytes()) != expected:
                raise RuntimeError("source changed after simulation: " + relative)
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username="dachuang", password=getpass.getpass("Pi password: "), timeout=20)
    try:
        with client.open_sftp() as sftp:
            if args.show_source_diff:
                for relative in ("raspberry_pi/autonomous_mapping.py", "raspberry_pi/mapping/explorer.py"):
                    with sftp.open(REMOTE + "/" + relative, "rb") as remote:
                        before = remote.read().decode().splitlines(True)
                    print("".join(difflib.unified_diff(before, (ROOT / relative).read_text(encoding="utf-8").splitlines(True), fromfile="Pi/" + relative, tofile="local/" + relative)))
            mismatches = []
            for relative in DEPENDENCIES:
                with sftp.open(REMOTE + "/" + relative, "rb") as remote:
                    remote_content = remote.read()
                    local_content = (ROOT / relative).read_bytes()
                    differs = digest(remote_content) != digest(local_content)
                    if differs and relative == "raspberry_pi/config.py" and mapping_config_equivalent(remote_content, local_content):
                        print("CONFIG_MAPPING_EQUIVALENT: unrelated VisualLocalizationConfig difference preserved")
                        differs = False
                    if differs:
                        mismatches.append(relative)
                        if args.show_config_diff and relative == "raspberry_pi/config.py":
                            print("".join(difflib.unified_diff(remote_content.decode().splitlines(True), local_content.decode().splitlines(True), fromfile="Pi/config.py", tofile="local/config.py")))
            print("DEPENDENCY_MISMATCHES=" + json.dumps(mismatches))
            active = command(client, "pgrep -af '[r]aspberry_pi.(autonomous_mapping|joint_mapping|map_camera_inspection)' || true").strip()
            print("ACTIVE_MOTION_TASKS=" + (active or "none"))
            if not args.deploy:
                return
            if mismatches or active:
                raise RuntimeError("remote differs from tested dependencies or has active motion; refusing deployment")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            backup = REMOTE + "/backups/joint_mapping_" + stamp
            staged = []
            for relative in FILES:
                destination = REMOTE + "/" + relative
                saved = backup + "/" + relative
                command(client, "mkdir -p " + shlex.quote(posixpath.dirname(saved)))
                try:
                    sftp.stat(destination)
                    command(client, "cp -p " + shlex.quote(destination) + " " + shlex.quote(saved))
                except FileNotFoundError:
                    pass
                temporary = destination + ".joint-staged"
                content = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
                with sftp.open(temporary, "wb") as remote:
                    remote.write(content)
                with sftp.open(temporary, "rb") as remote:
                    if hashlib.sha256(remote.read()).digest() != hashlib.sha256(content).digest():
                        raise RuntimeError("upload hash mismatch: " + relative)
                staged.append((temporary, destination))
            # Syntax-check staged Python without activating it.
            compile_files = [p for p, target in staged if target.endswith(".py")]
            command(client, "python3 -m py_compile " + " ".join(map(shlex.quote, compile_files)))
            command(client, "bash -n " + shlex.quote(REMOTE + "/map_and_annotate.sh.joint-staged"))
            for temporary, destination in staged:
                sftp.posix_rename(temporary, destination)
            print("BACKUP=" + backup)
            print(command(client, "cd " + shlex.quote(REMOTE) + " && .venv-pi/bin/python -m raspberry_pi.joint_mapping --help").splitlines()[0])
            print("DEPLOYED_WITHOUT_STARTING_MOTION")
    finally:
        client.close()


if __name__ == "__main__":
    main()
