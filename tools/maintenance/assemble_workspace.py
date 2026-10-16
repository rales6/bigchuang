"""Assemble the local monorepo from existing work and READ-ONLY remote sources.

Never deploys, restarts services, invokes a robot API, or deletes a source.
Only a new, history-only Git clone may be assembled. Passwords come from
environment variables, and SSH host keys are pinned before reading files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OLD = WORKSPACE / "car1.0"
SKIP = {".git", ".venv", ".venv-win", "venv", "__pycache__", ".pytest_cache",
        ".gradle", ".kotlin", ".tooling", ".idea", ".claude", ".codex", ".agents",
        "build", "dist", "logs", "log", "outputs", "output", "runs", "results",
        "backups", "legacy_backup", "deployment_backups", "deployment_snapshots",
        "remote_stage1_snapshot", "diagnostics", "search_runs", "simulation_runs",
        "debug_reports", "complete_test_output", "stationary_pose_runs", "secrets",
        "runtime_config", "models", "samples", "arm_teach_dataset",
        "arm_mapping_dataset", "gripper_dataset", "gripper_training", "问题信息"}
EXTENSIONS = {".py", ".md", ".sh", ".ps1", ".json", ".yaml", ".yml", ".toml",
              ".txt", ".ini", ".cfg", ".conf", ".service", ".timer", ".html",
              ".css", ".js", ".kt", ".kts", ".xml", ".properties", ".jar",
              ".bat", ".cmd", ".png", ".jpg", ".jpeg", ".svg", ".ico",
              ".webp", ".dxf", ".dwg", ".csv", ".ttf", ".pgm"}
REMOTE = {
    "pi": {"host": os.environ.get("PI_HOST", "pi.example.invalid"), "port": int(os.environ.get("PI_SSH_PORT", "22")), "user": os.environ.get("PI_USER", "robot"),
           "password_env": "PI_SSH_PASSWORD",
           "key": os.environ.get("PI_SSH_HOST_KEY_SHA256", "")},
    "gpu": {"host": os.environ.get("GPU_HOST", "gpu.example.invalid"), "port": int(os.environ.get("GPU_SSH_PORT", "22")), "user": os.environ.get("GPU_USER", "gpu"),
            "password_env": "GPU_SSH_PASSWORD",
            "key": os.environ.get("GPU_SSH_HOST_KEY_SHA256", "")},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def allowed(name, is_directory=False):
    if name in SKIP or name.startswith(("backup_", "backup-", "debug_")):
        return False
    if any(token in name for token in (".bak", ".before_", ".old", ".staged")):
        return False
    if name.startswith(".env") or name in {"local.properties", "hardware_token", "deepseek_vision.json"}:
        return False
    if is_directory:
        return True
    return Path(name).suffix.lower() in EXTENSIONS or name in {"gradlew", "Dockerfile", "__init__.py"}


def redact(data, relative):
    """Do not import code to inspect it. Remove known embedded credentials."""
    if Path(relative).suffix not in {".py", ".sh", ".json", ".md", ".txt", ".yaml", ".yml", ".conf"}:
        return data, 0
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data, 0
    count = 0
    # Old one-shot deployment scripts embedded the user's GPU SSH password.
    gpu_password = os.environ.get("GPU_SSH_PASSWORD", "")
    if len(gpu_password) >= 8 and gpu_password in text:
        text = text.replace(gpu_password, "SET_GPU_SSH_PASSWORD")
        count += 1
    text, replaced = re.subn(r"sk-[A-Za-z0-9_-]{16,}", "REDACTED_API_KEY_USE_LOCAL_CONFIG", text)
    count += replaced
    # Python's source remains valid; do not silently ship a working Pi password.
    text, replaced = re.subn(r"(password\s*=\s*)(['\"])1\2", r"\1'SET_PI_SSH_PASSWORD'", text)
    count += replaced
    return text.encode("utf-8") if count else data, count


class Assembly:
    def __init__(self):
        self.files = {}
        self.excluded = []

    def put(self, source, destination, data, origin):
        destination = Path(destination)
        target = (ROOT / destination).resolve()
        if not target.is_relative_to(ROOT) or ".git" in destination.parts:
            raise ValueError("Destination leaves the new repository")
        original_hash = digest(data)
        data, redactions = redact(data, str(destination))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        previous = self.files.get(destination.as_posix())
        self.files[destination.as_posix()] = {
            "source": str(source), "origin": origin, "source_sha256": original_hash,
            "local_sha256": digest(data), "bytes": len(data), "redactions": redactions,
            "previous_source": previous["source"] if previous else None,
        }

    def local_tree(self, source, destination):
        source = Path(source)
        if not source.exists():
            return
        for directory, dirs, files in os.walk(source):
            dirs[:] = sorted(d for d in dirs if allowed(d, True))
            for name in sorted(files):
                if not allowed(name):
                    continue
                path = Path(directory) / name
                if path.is_symlink() or path.stat().st_size > 10 * 1024 * 1024:
                    continue
                self.put(path, Path(destination) / path.relative_to(source), path.read_bytes(), "local")

    def remote_tree(self, sftp, source, destination, role, *, recurse=True):
        try:
            items = sftp.listdir_attr(source)
        except FileNotFoundError:
            self.excluded.append({"source": source, "reason": "not_present", "role": role})
            return
        for item in sorted(items, key=lambda entry: entry.filename):
            name = item.filename
            remote = str(PurePosixPath(source) / name)
            local = Path(destination) / name
            if stat.S_ISDIR(item.st_mode):
                if recurse and allowed(name, True):
                    self.remote_tree(sftp, remote, local, role)
            elif stat.S_ISREG(item.st_mode) and allowed(name) and item.st_size <= 10 * 1024 * 1024:
                with sftp.open(remote, "rb") as stream:
                    data = stream.read()
                self.put(remote, local, data, role)


def connect(role, pi_host=None):
    import paramiko
    spec = dict(REMOTE[role])
    if role == "pi" and pi_host:
        spec["host"] = pi_host
    if spec["host"].endswith(".invalid") or not re.fullmatch(r"[0-9a-f]{64}", spec["key"]):
        raise RuntimeError("Configure the device host and independently verified SSH host-key SHA256 before connecting")
    password = os.environ.get(spec["password_env"])
    if not password:
        raise RuntimeError("Missing " + spec["password_env"])
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(spec["host"], port=spec["port"], username=spec["user"], password=password,
                   allow_agent=False, look_for_keys=False, timeout=15, banner_timeout=15, auth_timeout=15)
    if digest(client.get_transport().get_remote_server_key().asbytes()) != spec["key"]:
        client.close()
        raise RuntimeError("Unexpected SSH host key; no files read")
    return client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--pi-host")
    args = parser.parse_args()
    if not args.apply:
        parser.error("One-time assembly requires --apply; keep the original workspace as backup")
    manifest = ROOT / "docs/maintenance/source_manifest.json"
    if manifest.exists():
        raise RuntimeError("Assembly already exists; do not overwrite subsequent edits")
    assembly = Assembly()
    for name in ("raspberry_pi", "grasp_range_integration", "tests", "tools", "stereo_calibration"):
        assembly.local_tree(OLD / name, Path("robot") / name)
    assembly.local_tree(OLD / "car", "firmware/car")
    assembly.local_tree(WORKSPACE / "car/docs", "firmware/docs")
    assembly.local_tree(OLD / "3090", "services/perception")
    assembly.local_tree(OLD / "car_sim", "simulation/car_sim")
    assembly.local_tree(WORKSPACE / "RobotCompanionAndroid", "apps/android")
    assembly.local_tree(WORKSPACE / "CAD", "hardware/cad")
    for path in OLD.iterdir():
        if path.is_file() and allowed(path.name) and not path.name.startswith("debug_"):
            assembly.put(path, Path("robot") / path.name, path.read_bytes(), "local")

    clients = {}
    try:
        for role in ("pi", "gpu"):
            clients[role] = connect(role, args.pi_host)
        with clients["pi"].open_sftp() as sftp:
            base = "/home/robot/dachuang/car1.0"
            assembly.remote_tree(sftp, base + "/raspberry_pi", "robot/raspberry_pi", "pi")
            release = base + "/ble_scene_17_relationship_20260919"
            assembly.remote_tree(sftp, release + "/grasp_range_integration", "robot/grasp_range_integration", "pi")
            # Only active release inputs/results; no historical run directories.
            for item in sftp.listdir_attr(release):
                if (stat.S_ISREG(item.st_mode) and Path(item.filename).suffix in {".pgm", ".yaml", ".json", ".png"}
                        and item.filename != "UPLOAD_RECORD.json"):
                    with sftp.open(release + "/" + item.filename, "rb") as stream:
                        assembly.put(release + "/" + item.filename, Path("maps/current") / item.filename, stream.read(), "pi")
            # Runtime root supports its own transport implementation: preserve
            # that boundary instead of merging two different raspberry_pi trees.
            track = "/home/robot/dachuang/track"
            for name in ("pursuit", "integrations", "raspberry_pi", "car"):
                assembly.remote_tree(sftp, track + "/" + name, Path("services/track") / name, "pi")
            for name in ("pi", "static", "tools", "tests"):
                assembly.remote_tree(sftp, track + "/final_work/" + name,
                                     Path("services/track/final_work") / name, "pi")
            assembly.remote_tree(sftp, track + "/final_work", "services/track/final_work", "pi", recurse=False)
            assembly.remote_tree(sftp, "/home/robot/grab_visual", "services/grasp/pi", "pi", recurse=False)
            # Canonical development adapters match actual running Pi adapters.
            assembly.remote_tree(sftp, track + "/final_work/pi",
                                 "robot/grasp_range_integration/track_web_integration/pi", "pi")
            # Reference DB is a dependency of CURRENT localisation, not an old run.
            assembly.remote_tree(sftp, base + "/maps/visual_reference_db", "maps/current/visual_reference_db", "pi")
        with clients["gpu"].open_sftp() as sftp:
            track = "/home/gpu/track"
            for name in ("gpu", "static", "tools"):
                assembly.remote_tree(sftp, track + "/" + name, Path("services/track") / name, "gpu")
            for name in ("gpu", "static"):
                assembly.remote_tree(sftp, track + "/final_work/" + name,
                                     Path("services/track/final_work") / name, "gpu")
            assembly.remote_tree(sftp, track + "/final_work/gpu",
                                 "robot/grasp_range_integration/track_web_integration/gpu", "gpu")
            assembly.remote_tree(sftp, "/home/gpu/grab_visual", "services/grasp/gpu", "gpu", recurse=False)
            for name in ("src", "scripts", "requirements", "configs"):
                assembly.remote_tree(sftp, "/home/gpu/3090/" + name,
                                     Path("services/perception") / name, "gpu")
            for name in ("model_state.json", "model.pt"):
                path = "/home/gpu/grab_visual/arm_teach_artifacts/" + name
                info = sftp.stat(path)
                if info.st_size > 10 * 1024 * 1024:
                    raise RuntimeError("Unexpectedly large active arm artifact")
                with sftp.open(path, "rb") as stream:
                    assembly.put(path, Path("services/grasp/gpu/arm_teach_artifacts") / name, stream.read(), "gpu")
    finally:
        for client in clients.values():
            client.close()

    # Keep the original Git attributes, but not the conflicted ignore file.
    shutil.copy2(OLD / ".gitattributes", ROOT / ".gitattributes")
    report = {"schema": "robot_car.source_manifest.v1", "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "original_worktree": str(OLD), "original_branch": "miller", "history_head": "0fea8a9",
              "remote_access": "read_only_sftp", "remote_files_modified": False,
              "files": assembly.files, "excluded": assembly.excluded}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(assembly.files), "bytes": sum(f["bytes"] for f in assembly.files.values()),
                      "manifest": str(manifest), "redacted_files": sum(bool(f["redactions"]) for f in assembly.files.values()),
                      "excluded": assembly.excluded}, ensure_ascii=False))


if __name__ == "__main__":
    main()
