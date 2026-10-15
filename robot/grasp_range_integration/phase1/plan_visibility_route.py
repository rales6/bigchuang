"""Offline visibility-zone route preview; never connects to hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raspberry_pi.known_map_navigation import KnownOccupancyMap

from .deepseek_planner import validate_phase1_decision
from .visibility_route_planner import install_visibility_planner


def main() -> int:
    parser = argparse.ArgumentParser(description="离线生成目标可见区路线，不连接任何硬件")
    parser.add_argument("--decision", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--start-x", type=float, required=True)
    parser.add_argument("--start-y", type=float, required=True)
    parser.add_argument("--clearance", type=float, default=0.30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    validate_phase1_decision(decision, require_executable=True)
    # Exercise the same process-local route binding as the real-car entry.
    from relationship_object_search_entry import install_relationship_planner

    install_relationship_planner()
    install_visibility_planner()
    import raspberry_pi.object_search_mission as mission

    known_map = KnownOccupancyMap.load(args.map, args.map_yaml, clearance_m=args.clearance)
    route = mission.build_search_route(
        known_map,
        (args.start_x, args.start_y),
        decision["route_request"],
    )
    if route.get("planning_policy") != "semantic_visibility_regions.v1":
        raise RuntimeError("实车入口未绑定目标可见区规划器")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(route, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"目标可见区路线已保存：{output}")
    print(f"观察次数：{len(route['views'])}；第一观察移动距离：{route['views'][0]['path_length_m']:.2f}m")
    print("本命令未连接底盘、雷达、相机或机械臂。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
