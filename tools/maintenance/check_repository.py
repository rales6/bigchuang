"""Read-only local repository verification. Never imports hardware modules."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", "runtime", "build", ".gradle"}
REQUIRED = ["apps/android/gradlew", "apps/android/app/build.gradle.kts", "firmware/car/main.py",
            "robot/raspberry_pi/object_search_mission.py", "robot/grasp_range_integration/phase1/relationship_find_only_entry.py",
            "services/track/final_work/pi/robot_bridge.py", "services/track/final_work/gpu/app.py",
            "services/track/pursuit/hull_micro_adjust.py", "services/grasp/pi/arm_teach.py",
            "services/grasp/gpu/arm_teach_model.py", "simulation/car_sim/server.py",
            "maps/current/ble_scene_17.pgm", "maps/current/semantic_runtime.json"]


def external_runtime_artifact(name):
    path = Path(name)
    return 'runtime' in path.parts or path.suffix in {'.pt', '.pth', '.onnx', '.safetensors'}


def source_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not any(part in IGNORED or part.startswith(".venv-") for part in path.relative_to(ROOT).parts):
            yield path


def verify():
    errors, warnings = [], []
    for name in REQUIRED:
        if not (ROOT/name).is_file():
            errors.append("Missing component: " + name)
    ignore = (ROOT/".gitignore").read_text(encoding="utf-8")
    if re.search(r"^(<<<<<<<|=======|>>>>>>>)", ignore, re.M):
        errors.append(".gitignore contains unresolved merge markers")
    python_count = 0
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py":
            python_count += 1
            try:
                source = path.read_text(encoding="utf-8-sig")
                ast.parse(source, filename=relative)
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"Invalid Python: {relative}: {exc}")
        if path.suffix in {".py", ".sh", ".json", ".md", ".txt", ".yml", ".yaml", ".conf"}:
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeError:
                continue
            if re.search(r"sk-[A-Za-z0-9_-]{16,}", text):
                errors.append("Possible API key: " + relative)
            if path.suffix == ".py" and re.search(r"password\s*=\s*(['\"])1\1", text):
                errors.append("Hard-coded Pi password: " + relative)
    manifest_path = ROOT/"docs/maintenance/source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, meta in manifest["files"].items():
        if external_runtime_artifact(name):
            # Provenance includes the device's learned state and model file;
            # neither is a source dependency in a clean Git clone. Do not
            # force-add runtime state just to satisfy a provenance manifest.
            continue
        if not (ROOT/name).is_file():
            errors.append("Synced source disappeared: " + name)
    active = ROOT/"maps/current"
    map_manifest = json.loads((active/"manifest.json").read_text(encoding="utf-8"))
    checked_maps = 0
    for name, expected in map_manifest.get("files", {}).items():
        if Path(name).suffix not in {".pgm", ".yaml", ".json", ".png"}:
            continue
        path = active/name
        if not path.is_file():
            errors.append("Current map dependency missing: " + name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append("Current map hash differs from device release manifest: " + name)
        else:
            checked_maps += 1
    # Catch accidental nested repositories without modifying Git or its index.
    nested = [str(p.relative_to(ROOT)) for p in ROOT.rglob(".git") if p != ROOT/".git"]
    if nested:
        errors.append("Nested Git directories: " + repr(nested))
    return {"python_files_checked": python_count, "current_map_files_verified": checked_maps,
            "source_manifest_entries": len(manifest["files"]), "errors": errors, "warnings": warnings,
            "hardware_imported": False, "remote_connections": False}


def main():
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
