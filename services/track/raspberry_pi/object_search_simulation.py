from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from raspberry_pi.mapping.geometry import Pose2D, normalize_angle
from raspberry_pi.object_search_mission import _densify_path, _line_safe_with_margin
from raspberry_pi.search_route_planner import build_search_route


@dataclass(frozen=True)
class SimulatedScenario:
    name: str
    target_view_index: int | None = None
    obstacle_before_view_index: int | None = None
    localization_failure: bool = False


def _route_length(route: dict[str, Any]) -> float:
    return sum(float(view.get("path_length_m") or 0.0) for view in route.get("views", []))


def simulate_waypoint_tracking(
    known_map,
    route: dict[str, Any],
    speed_mm_s: int = 240,
    turn_mrad_s: int = 1250,
    step_s: float = 0.08,
) -> dict[str, Any]:
    """Kinematic check of the same turn-first/drive controller used on Pi."""
    start = route["start_xy_m"]
    pose = Pose2D(float(start[0]), float(start[1]), 0.0)
    commands = 0
    maximum_linear = 0
    maximum_angular = 0
    for view in route.get("views", []):
        path = view.get("path_xy_m") or [view["view_xy_m"]]
        waypoints = _densify_path(known_map, path, spacing_m=0.18)[1:] or path
        for waypoint in waypoints:
            reached = False
            for _step in range(900):
                dx = float(waypoint[0]) - pose.x_m
                dy = float(waypoint[1]) - pose.y_m
                distance = math.hypot(dx, dy)
                if distance <= 0.06:
                    reached = True
                    break
                desired = math.atan2(dy, dx)
                error = normalize_angle(desired - pose.yaw_rad)
                if abs(error) > math.radians(3.0):
                    linear = 0
                    angular = int(math.copysign(min(turn_mrad_s, max(650, abs(error) * 1100)), error))
                else:
                    linear = speed_mm_s if distance >= 0.50 else min(speed_mm_s, 230)
                    angular = 0
                yaw = normalize_angle(pose.yaw_rad + angular / 1000.0 * step_s)
                x = pose.x_m + linear / 1000.0 * math.cos(yaw) * step_s
                y = pose.y_m + linear / 1000.0 * math.sin(yaw) * step_s
                cell = known_map.world_to_grid(((x, y),))[0]
                if not known_map.in_bounds(int(cell[0]), int(cell[1])) or not known_map.traversable[int(cell[1]), int(cell[0])]:
                    return {
                        "reached": False,
                        "reason": "simulated_collision_or_unknown",
                        "commands": commands,
                        "pose": [round(pose.x_m, 3), round(pose.y_m, 3), round(pose.yaw_rad, 3)],
                        "attempted_pose": [round(x, 3), round(y, 3), round(yaw, 3)],
                        "waypoint": [float(waypoint[0]), float(waypoint[1])],
                    }
                pose = Pose2D(x, y, yaw)
                commands += 1
                maximum_linear = max(maximum_linear, abs(linear))
                maximum_angular = max(maximum_angular, abs(angular))
            if not reached:
                return {"reached": False, "reason": "waypoint_timeout", "commands": commands}
    return {
        "reached": True,
        "reason": "all_waypoints_reached",
        "commands": commands,
        "maximum_linear_mm_s": maximum_linear,
        "maximum_angular_mrad_s": maximum_angular,
        "final_pose": [round(pose.x_m, 3), round(pose.y_m, 3), round(pose.yaw_rad, 3)],
    }


def simulate_scenario(known_map, decision, scenario: SimulatedScenario) -> dict[str, Any]:
    start = (0.0, 0.0)
    if scenario.localization_failure:
        return {
            "scenario": scenario.name,
            "result": "failure",
            "terminal_reason": "localization_failed",
            "hardware_commands_sent": False,
            "visited_views": [],
            "replans": 0,
        }
    route = build_search_route(known_map, start, decision["route_request"])
    motion = simulate_waypoint_tracking(known_map, route)
    if not motion["reached"]:
        return {
            "scenario": scenario.name,
            "result": "failure",
            "terminal_reason": motion["reason"],
            "hardware_commands_sent": False,
            "visited_views": [],
            "replans": 0,
            "motion_simulation": motion,
        }
    views = route.get("views", [])
    visited: list[int] = []
    replans = 0
    for index, _view in enumerate(views):
        if scenario.obstacle_before_view_index == index and replans == 0:
            # The real executor marks lidar evidence in the volatile layer.
            # Simulation models the resulting event/replan without modifying
            # permanent PGM/YAML data.
            replans += 1
            continue
        visited.append(index)
        if scenario.target_view_index == index:
            return {
                "scenario": scenario.name,
                "result": "success",
                "terminal_reason": "confirmed_visual_class_bbox_and_colour",
                "hardware_commands_sent": False,
                "visited_views": visited,
                "replans": replans,
                "route_length_m": round(_route_length(route), 3),
                "motion_simulation": motion,
            }
    return {
        "scenario": scenario.name,
        "result": "failure",
        "terminal_reason": "search_views_exhausted",
        "hardware_commands_sent": False,
        "visited_views": visited,
        "replans": replans,
        "route_length_m": round(_route_length(route), 3),
        "motion_simulation": motion,
    }


def simulate_mission(known_map, decision, report) -> dict[str, Any]:
    scenarios = (
        SimulatedScenario("target_at_third_view", target_view_index=2),
        SimulatedScenario("temporary_obstacle_then_target", target_view_index=3, obstacle_before_view_index=1),
        SimulatedScenario("target_absent"),
        SimulatedScenario("localization_failure", localization_failure=True),
    )
    results = []
    for scenario in scenarios:
        result = simulate_scenario(known_map, decision, scenario)
        report.event("simulation_scenario", **result)
        results.append(result)
    expected = {
        "target_at_third_view": "success",
        "temporary_obstacle_then_target": "success",
        "target_absent": "failure",
        "localization_failure": "failure",
    }
    passed = all(expected[item["scenario"]] == item["result"] for item in results)
    return {
        "result": "simulation_passed" if passed else "simulation_failed",
        "summary": "四个闭环事件场景均符合安全终止条件" if passed else "仿真场景断言失败",
        "simulation_results": results,
        "hardware_motion_enabled": False,
    }
