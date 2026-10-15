"""Build a stable semantic-map release from audited source artifacts.

The release keeps the successful inspection map and the later Qwen
re-annotation as separate, immutable artifacts. Runtime geometry comes from
the reviewed 20260917 layout; later annotation batches add evidence but cannot
overwrite those positions or promote them to navigation/localization anchors.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_relationship_semantic_map import build_memory


DEFAULT_OUTPUT = ROOT / "maps" / "semantic_release_20260919"
DEFAULT_MAP = ROOT / "maps" / "ble_scene_17.pgm"
DEFAULT_YAML = ROOT / "maps" / "ble_scene_17.yaml"
DEFAULT_LOCALIZATION = ROOT / "maps" / "ble_scene_17_semantics_qwen.json"
DEFAULT_OBSERVATION = (
    ROOT / "semantic_maps" / "ble_scene_17_20260917_190706"
    / "semantic_observation_map.json"
)
DEFAULT_OBSERVATION_IMAGE = DEFAULT_OBSERVATION.with_suffix(".png")
DEFAULT_QWEN = (
    ROOT / "semantic_maps" / "qwen_reannotation_20260918_142739"
    / "results" / "quality_filtered" / "semantic_object_memory.json"
)
DEFAULT_QWEN_IMAGE = (
    ROOT / "semantic_maps" / "qwen_reannotation_20260918_142739"
    / "qwen_semantic_map.png"
)
DEFAULT_QWEN_REPORT = DEFAULT_QWEN_IMAGE.with_suffix(".json")
DEFAULT_REVIEWED_LAYOUT = (
    ROOT / "semantic_maps" / "ble_scene_17_20260917_190706"
    / "complete_objects" / "supplemented_object_map.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_release(
    *,
    map_path: Path,
    yaml_path: Path,
    localization_path: Path,
    observation_path: Path,
    observation_image: Path,
    qwen_path: Path,
    qwen_image: Path,
    qwen_report: Path,
    output: Path,
    reviewed_layout_path: Path | None = None,
) -> dict[str, Any]:
    sources = [
        map_path,
        yaml_path,
        localization_path,
        observation_path,
        observation_image,
        qwen_path,
        qwen_image,
        qwen_report,
    ]
    if reviewed_layout_path is not None:
        sources.append(reviewed_layout_path)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    map_hash = sha256(map_path)
    yaml_hash = sha256(yaml_path)
    memory = load_json(qwen_path)
    observation = load_json(observation_path)
    if memory.get("schema") != "semantic_object_memory.v1":
        raise ValueError("Qwen source must use semantic_object_memory.v1")
    if memory.get("map_sha256") != map_hash:
        raise ValueError("Qwen memory and successful occupancy map hashes differ")
    if memory.get("map_yaml_sha256") != yaml_hash:
        raise ValueError("Qwen memory and successful map YAML hashes differ")
    if str(observation.get("inspection_run")) != "20260917_190706":
        raise ValueError("Observation map is not the successful 20260917_190706 run")

    runtime = (
        build_memory(map_path, yaml_path, reviewed_layout_path, [qwen_path])
        if reviewed_layout_path is not None
        else deepcopy(memory)
    )
    objects = runtime.get("objects") or []
    if not isinstance(objects, list):
        raise ValueError("Semantic memory objects must be a list")
    for item in objects:
        if not isinstance(item, dict):
            continue
        # Re-assert the safety boundary even if a future experimental file is
        # accidentally supplied: only reviewed records may be promoted by the
        # established review pipeline, never by this publisher.
        if item.get("semantic_reviewed") is not True:
            item["position_verified"] = False
            item["approach_verified"] = False
            item["navigation_allowed"] = False
            item["localization_allowed"] = False
            item["grasp_allowed"] = False
            item["requires_visual_recheck"] = True

    stations = observation.get("stations") or []
    runtime["release"] = {
        "schema": "semantic_map_release.v1",
        "release_id": "ble_scene_17_semantic_release_20260919",
        "successful_inspection_run": "20260917_190706",
        "qwen_reannotation_run": "20260918_142739",
        "occupancy_map": "ble_scene_17.pgm",
        "occupancy_yaml": "ble_scene_17.yaml",
        "observation_map": "semantic_observation_20260917_190706.json",
        "qwen_audit_map": "qwen_reannotation_20260918.json",
        "combined_map": "semantic_relationship_map_20260919.png",
        "localization_landmarks": "semantic_localization_landmarks.json",
        "usage": {
            "metric_pose_source": "lidar_occupancy_map",
            "semantic_localization": "verified_landmarks_only",
            "task_search": "estimated_regions_require_live_visual_recheck",
            "direct_navigation_to_unverified_object": False,
            "direct_grasp_from_map_coordinate": False,
        },
    }
    runtime["observation_stations"] = stations
    runtime["inspection_coverage"] = observation.get("coverage")
    runtime["source_artifacts"] = {
        "successful_observation_sha256": sha256(observation_path),
        "qwen_quality_filtered_sha256": sha256(qwen_path),
        "occupancy_map_sha256": map_hash,
        "occupancy_yaml_sha256": yaml_hash,
    }

    output.mkdir(parents=True, exist_ok=True)
    copies = {
        map_path: output / "ble_scene_17.pgm",
        yaml_path: output / "ble_scene_17.yaml",
        localization_path: output / "semantic_localization_landmarks.json",
        observation_path: output / "semantic_observation_20260917_190706.json",
        observation_image: output / "semantic_observation_20260917_190706.png",
        qwen_report: output / "qwen_reannotation_20260918.json",
        qwen_image: output / "qwen_reannotation_20260918.png",
        qwen_path: output / "qwen_quality_filtered_20260918.json",
        ROOT / "tools" / "build_relationship_semantic_map.py": output / "build_relationship_semantic_map.py",
        ROOT / "tools" / "build_semantic_release.py": output / "build_semantic_release.py",
        ROOT / "tools" / "semantic_release_complete_test.py": output / "semantic_release_complete_test.py",
        ROOT / "raspberry_pi" / "search_route_planner.py": output / "search_route_planner_relationship.py",
        ROOT / "tools" / "run_semantic_release_complete_test.sh": output / "run_complete_test.sh",
        ROOT / "tools" / "relationship_object_search_entry.py": output / "relationship_object_search_entry.py",
        ROOT / "tools" / "relationship_localization_guard.py": output / "relationship_localization_guard.py",
        ROOT / "tools" / "deepseek_visual_localization.py": output / "deepseek_visual_localization.py",
        ROOT / "tools" / "deepseek_vision_config.example.json": output / "deepseek_vision_config.example.json",
        ROOT / "tools" / "stationary_initial_pose_probe.py": output / "stationary_initial_pose_probe.py",
        ROOT / "tools" / "replay_stationary_deepseek.py": output / "replay_stationary_deepseek.py",
        ROOT / "tools" / "natural_search_relationship.sh": output / "natural_search_relationship.sh",
        ROOT / "tools" / "companion_service_relationship.py": output / "companion_service_relationship.py",
        ROOT / "tools" / "robot-companion-relationship-override.conf": output / "robot-companion-relationship-override.conf",
    }
    if reviewed_layout_path is not None:
        copies[reviewed_layout_path] = output / "reviewed_layout_20260917.json"
    for source, target in copies.items():
        shutil.copy2(source, target)
    # Human-readable archive aliases. The YAML keeps its original
    # `image: ble_scene_17.pgm` reference, which resolves to the canonical
    # copy above and preserves the source hash used by semantic memory.
    shutil.copy2(map_path, output / "occupancy_success_20260917_190706.pgm")
    shutil.copy2(yaml_path, output / "occupancy_success_20260917_190706.yaml")
    write_json(output / "semantic_runtime.json", runtime)

    verified = [
        item for item in objects
        if isinstance(item, dict)
        and item.get("position_verified") is True
        and item.get("localization_allowed") is True
    ]
    manifest = {
        "schema": "semantic_map_release_manifest.v1",
        "release_id": runtime["release"]["release_id"],
        "map_sha256": map_hash,
        "map_yaml_sha256": yaml_hash,
        "candidate_count": len(objects),
        "search_region_count": len(runtime.get("search_regions") or []),
        "observation_station_count": len(stations),
        "verified_localization_landmark_count": len(verified),
        "files": {
            target.name: sha256(target) for target in copies.values()
        },
    }
    for alias in (
        output / "occupancy_success_20260917_190706.pgm",
        output / "occupancy_success_20260917_190706.yaml",
    ):
        manifest["files"][alias.name] = sha256(alias)
    manifest["files"]["semantic_runtime.json"] = sha256(
        output / "semantic_runtime.json"
    )
    combined = output / "semantic_relationship_map_20260919.png"
    combined_report = combined.with_suffix(".json")
    for generated in (combined, combined_report):
        if generated.is_file():
            manifest["files"][generated.name] = sha256(generated)
    release_readme = output / "README.md"
    if release_readme.is_file():
        manifest["files"][release_readme.name] = sha256(release_readme)
    write_json(output / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--map", type=Path, default=DEFAULT_MAP)
    value.add_argument("--map-yaml", type=Path, default=DEFAULT_YAML)
    value.add_argument("--localization-map", type=Path, default=DEFAULT_LOCALIZATION)
    value.add_argument("--observation-map", type=Path, default=DEFAULT_OBSERVATION)
    value.add_argument("--observation-image", type=Path, default=DEFAULT_OBSERVATION_IMAGE)
    value.add_argument("--qwen-memory", type=Path, default=DEFAULT_QWEN)
    value.add_argument("--qwen-image", type=Path, default=DEFAULT_QWEN_IMAGE)
    value.add_argument("--qwen-report", type=Path, default=DEFAULT_QWEN_REPORT)
    value.add_argument("--reviewed-layout", type=Path, default=DEFAULT_REVIEWED_LAYOUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    result = build_release(
        map_path=args.map,
        yaml_path=args.map_yaml,
        localization_path=args.localization_map,
        observation_path=args.observation_map,
        observation_image=args.observation_image,
        qwen_path=args.qwen_memory,
        qwen_image=args.qwen_image,
        qwen_report=args.qwen_report,
        output=args.output,
        reviewed_layout_path=args.reviewed_layout,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
