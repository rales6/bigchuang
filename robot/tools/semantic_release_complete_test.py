"""Safe end-to-end validation for a relationship-aware semantic release.

This test verifies the package, resolves semantic search regions, plans A*
observation routes and runs the closed-loop mission simulator. It never opens
the ESP32 link, lidar or cameras and never sends a hardware command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_root_for(script: Path) -> Path:
    # Repository tool: <root>/tools/script.py. Packaged copy:
    # <root>/ble_scene_17_relationship_20260919/script.py.
    parent = script.resolve().parent
    if (parent.parent / "raspberry_pi").is_dir():
        return parent.parent
    raise RuntimeError("Cannot locate car1.0 project root")


def verify_manifest(release: Path) -> dict:
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked, errors = {}, []
    for relative, expected in manifest.get("files", {}).items():
        path = release / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
            continue
        actual = digest(path)
        checked[relative] = actual
        if actual != expected:
            errors.append(f"sha256:{relative}")
    return {"passed": not errors, "checked_count": len(checked), "errors": errors}


def folded(value) -> str:
    return "".join(str(value or "").lower().split())


def find_region(memory: dict, query: str) -> dict:
    objects = {str(item["id"]): item for item in memory.get("objects", [])}
    needle = folded(query)
    candidates = []
    for region in memory.get("search_regions", []):
        item = objects.get(str(region.get("id")), {})
        values = [region.get("query"), item.get("name"), item.get("category")]
        values.extend(item.get("aliases") or [])
        score = sum(1 for value in values if needle and needle in folded(value))
        if score:
            candidates.append((score, float(item.get("confidence") or 0.0), region, item))
    if not candidates:
        raise ValueError(f"No semantic search region for: {query}")
    _score, _confidence, region, item = max(candidates, key=lambda row: (row[0], row[1]))
    return {"region": region, "object": item}


def gpu_health(url: str) -> dict:
    endpoint = url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(endpoint, timeout=4) as response:
            body = response.read().decode("utf-8", errors="replace")
        return {"available": True, "endpoint": endpoint, "response": body[:1000]}
    except (OSError, urllib.error.URLError) as exc:
        return {"available": False, "endpoint": endpoint, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--anchor", default="玻璃门")
    parser.add_argument("--target", default="扫帚")
    parser.add_argument("--gpu-url", default="http://192.0.2.20:8011")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    release = args.release.resolve()
    root = (args.project_root or project_root_for(Path(__file__))).resolve()
    sys.path.insert(0, str(root))
    from raspberry_pi.known_map_navigation import KnownOccupancyMap
    from raspberry_pi import object_search_simulation

    planner_path = release / "search_route_planner_relationship.py"
    specification = importlib.util.spec_from_file_location(
        "semantic_release_route_planner", planner_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load packaged route planner: {planner_path}")
    planner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(planner)
    build_search_route = planner.build_search_route
    render_search_route = planner.render_search_route
    # The installed Pi simulator may import an older planner. Redirect only
    # this test process to the packaged implementation; no installed module is
    # edited or activated.
    object_search_simulation.build_search_route = build_search_route
    SimulatedScenario = object_search_simulation.SimulatedScenario
    simulate_scenario = object_search_simulation.simulate_scenario

    output = (args.output_dir or release / "complete_test_output").resolve()
    output.mkdir(parents=True, exist_ok=True)
    verification = verify_manifest(release)
    memory = json.loads((release / "semantic_runtime.json").read_text(encoding="utf-8"))
    observation = json.loads(
        (release / "semantic_observation_20260917_190706.json").read_text(encoding="utf-8")
    )
    if memory.get("schema") != "semantic_object_memory.v1":
        raise ValueError("Unexpected semantic runtime schema")
    unsafe = [
        item.get("id") for item in memory.get("objects", [])
        if item.get("navigation_allowed") is not False
        or item.get("localization_allowed") is not False
        or item.get("grasp_allowed") is not False
    ]
    map_path = release / "ble_scene_17.pgm"
    yaml_path = release / "ble_scene_17.yaml"
    known_map = KnownOccupancyMap.load(map_path, yaml_path, clearance_m=0.30)
    station = observation["stations"][0]
    requested_start = (float(station["x_m"]), float(station["y_m"]))
    start_cell = known_map._nearest_traversable(
        tuple(known_map.world_to_grid((requested_start,))[0]), maximum_radius=15
    )
    if start_cell is None:
        raise RuntimeError("No traversable simulated start near inspection station S1")
    start = tuple(float(value) for value in known_map.grid_to_world((start_cell,))[0])

    route_matrix = []
    for region in memory.get("search_regions", []):
        request = {
            "mode": "semantic_search_region",
            "center_xy_m": region["map_xy"],
            "radius_m": region.get("radius_m", 1.2),
            "maximum_views": 8,
            "require_live_visual_recheck": True,
        }
        route = build_search_route(known_map, start, request)
        route_matrix.append({
            "id": region["id"],
            "query": region["query"],
            "view_count": len(route.get("views", [])),
            "rejected_count": len(route.get("rejected_unreachable_views", [])),
            "passed": bool(route.get("views")),
        })

    selected = find_region(memory, args.anchor)
    region, anchor = selected["region"], selected["object"]
    decision = {
        "schema": "robot_object_search_plan.v1",
        "task_id": "semantic-release-complete-test",
        "instruction": f"在{args.anchor}附近寻找{args.target}",
        "status": "executable",
        "intent": "find_object",
        "summary": "关系语义地图安全搜索区完整仿真",
        "target": {
            "query": args.target,
            "class_name": args.target,
            "detection_mode": "qwen_fallback",
            "live_visual_confirmation_required": True,
        },
        "anchor": {
            "query": args.anchor,
            "search_region_id": region["id"],
            "name": anchor["name"],
            "map_xy": region["map_xy"],
            "position_verified": False,
            "requires_visual_recheck": True,
        },
        "route_request": {
            "mode": "semantic_search_region",
            "center_xy_m": region["map_xy"],
            "radius_m": region.get("radius_m", 1.2),
            "maximum_views": 8,
            "require_live_visual_recheck": True,
        },
        "warnings": ["本任务为无硬件命令仿真，搜索区中心不是物体或夹取坐标。"],
    }
    decision_path = output / "complete_test_decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    route = build_search_route(known_map, start, decision["route_request"])
    route_path = output / "complete_test_route.json"
    route_path.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_search_route(known_map, route, output / "complete_test_route.png", decision["route_request"])

    scenarios = (
        SimulatedScenario("target_at_third_view", target_view_index=2),
        SimulatedScenario("temporary_obstacle_then_target", target_view_index=3, obstacle_before_view_index=1),
        SimulatedScenario("target_absent"),
        SimulatedScenario("localization_failure", localization_failure=True),
    )
    results = [simulate_scenario(known_map, decision, scenario) for scenario in scenarios]
    expected = {
        "target_at_third_view": "success",
        "temporary_obstacle_then_target": "success",
        "target_absent": "failure",
        "localization_failure": "failure",
    }
    simulation_passed = all(expected[row["scenario"]] == row["result"] for row in results)
    health = gpu_health(args.gpu_url)
    offline_passed = (
        verification["passed"] and not unsafe and all(row["passed"] for row in route_matrix)
        and bool(route.get("views")) and simulation_passed
    )
    report = {
        "schema": "semantic_release_complete_test.v1",
        "release": str(release),
        "hardware_commands_sent": False,
        "existing_runtime_activated": False,
        "package_verification": verification,
        "semantic_object_count": len(memory.get("objects", [])),
        "search_region_count": len(memory.get("search_regions", [])),
        "scene_relation_count": len(memory.get("scene_relations", [])),
        "unsafe_promoted_objects": unsafe,
        "simulated_start_xy_m": list(start),
        "route_matrix": route_matrix,
        "selected_decision": decision,
        "selected_route_view_count": len(route.get("views", [])),
        "simulation_results": results,
        "offline_chain_passed": offline_passed,
        "gpu_service": health,
        "full_online_chain_ready": offline_passed and health["available"],
        "next_action": (
            "Run the same task against 3090 and live dual cameras while stopped."
            if health["available"] else
            "Restore the 3090 service/network, then rerun with --require-gpu before any real motion."
        ),
    }
    report_path = output / "complete_test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "offline_chain_passed": report["offline_chain_passed"],
        "full_online_chain_ready": report["full_online_chain_ready"],
        "route_regions_passed": sum(row["passed"] for row in route_matrix),
        "route_regions_total": len(route_matrix),
        "selected_route_view_count": report["selected_route_view_count"],
        "gpu_service": health,
        "hardware_commands_sent": False,
    }, ensure_ascii=False, indent=2))
    if not offline_passed or (args.require_gpu and not health["available"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
