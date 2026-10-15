from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

from qwen_grounded_tracker.semantic_task_planner import SemanticTaskPlan, validate_plan
from qwen_grounded_tracker.scene_change_memory import SceneChangeMemory


@dataclass
class DryRunResult:
    completed: bool
    final_robot_xy: list[int]
    carrying: str | None
    trace: list[dict[str, Any]]
    path_cells: list[list[int]]
    navigation_goals_reached: int
    grasp_attempts: int
    grasp_successes: int
    grasp_model_probe: dict[str, Any]
    suspicious_rechecks: list[dict[str, Any]]
    collision_contacts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "final_robot_xy": self.final_robot_xy,
            "carrying": self.carrying,
            "trace": self.trace,
            "path_cells": self.path_cells,
            "navigation_goals_reached": self.navigation_goals_reached,
            "grasp_attempts": self.grasp_attempts,
            "grasp_successes": self.grasp_successes,
            "grasp_model_probe": self.grasp_model_probe,
            "suspicious_rechecks": self.suspicious_rechecks,
            "collision_contacts": self.collision_contacts,
        }


def _grid(scene: dict[str, Any]) -> list[str]:
    grid = scene.get("occupancy_grid", {}).get("rows", [])
    if not isinstance(grid, list) or not grid or not all(isinstance(row, str) for row in grid):
        raise ValueError("Scene occupancy_grid.rows must be a non-empty string array")
    width = len(grid[0])
    if width == 0 or any(len(row) != width for row in grid):
        raise ValueError("Scene occupancy grid must be rectangular")
    return grid


def _astar(grid: list[str], start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    width, height = len(grid[0]), len(grid)
    def free(node: tuple[int, int]) -> bool:
        x, y = node
        return 0 <= x < width and 0 <= y < height and grid[y][x] != "#"
    if not free(start) or not free(goal):
        raise ValueError(f"Start or goal is occupied: {start} -> {goal}")
    queue: list[tuple[int, int, tuple[int, int]]] = [(0, 0, start)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost = {start: 0}
    serial = 0
    while queue:
        _, _, current = heapq.heappop(queue)
        if current == goal:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0] + dx, current[1] + dy)
            if not free(nxt):
                continue
            new_cost = cost[current] + 1
            if new_cost >= cost.get(nxt, 10**9):
                continue
            cost[nxt] = new_cost
            parent[nxt] = current
            serial += 1
            heuristic = abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
            heapq.heappush(queue, (new_cost + heuristic, serial, nxt))
    raise ValueError(f"No collision-free path: {start} -> {goal}")


def _find_object(scene: dict[str, Any], landmark_id: str | None, query: str) -> dict[str, Any] | None:
    objects = scene.get("objects", [])
    if landmark_id:
        match = next((obj for obj in objects if obj.get("id") == landmark_id), None)
        if match:
            return match
    folded = query.lower()
    groups = (
        ("垃圾", "trash", "rubbish", "waste"),
        ("簸箕", "dustpan"),
        ("垃圾桶", "trash bin", "trash_bin"),
        ("瓶", "bottle"),
        ("桌", "table", "desk"),
        ("柜", "cabinet"),
        ("椅", "chair"),
    )
    aliases = [alias for group in groups if any(alias in folded for alias in group) for alias in group]
    if not aliases:
        aliases = [part for part in folded.replace("或", " ").split() if part]
    for obj in objects:
        text = " ".join(str(obj.get(key, "")).lower() for key in ("id", "name", "category"))
        if any(alias in text or text in alias for alias in aliases):
            return obj
    return None


class SymbolicDryRunExecutor:
    """Checks plan ordering and routes without sending any hardware command."""

    def __init__(self, change_memory: SceneChangeMemory | None = None) -> None:
        self.change_memory = change_memory

    def execute(self, plan: SemanticTaskPlan, scene: dict[str, Any]) -> DryRunResult:
        validate_plan(plan)
        grid = _grid(scene)
        pose = scene.get("robot_pose", {}).get("grid_xy", [1, 1])
        robot_xy = (int(pose[0]), int(pose[1]))
        carrying: str | None = None
        active_target: dict[str, Any] | None = None
        destination: dict[str, Any] | None = None
        trace: list[dict[str, Any]] = []
        all_paths: list[list[int]] = []
        navigation_goals_reached = 0
        grasp_attempts = grasp_successes = 0
        suspicious_rechecks: list[dict[str, Any]] = []
        model = dict(scene.get("grasp_model") or {})
        grasp_model_probe = {
            "available": bool(model.get("available", False)),
            "tested": bool(model.get("available", False)),
            "usable": bool(model.get("available", False) and model.get("smoke_test_ok", True)),
            "mode": "model_smoke_test" if model.get("available", False) else "skipped_not_installed",
            "grasp_result_policy": "default_success_in_simulation",
        }

        def route_to(goal_raw: Any) -> None:
            nonlocal robot_xy, navigation_goals_reached
            if not isinstance(goal_raw, (list, tuple)) or len(goal_raw) < 2:
                raise RuntimeError("Navigation target has no grid approach point")
            goal = (int(round(float(goal_raw[0]))), int(round(float(goal_raw[1]))))
            path = _astar(grid, robot_xy, goal)
            all_paths.extend([[x, y] for x, y in path[1:]])
            robot_xy = goal
            navigation_goals_reached += 1

        for step in plan.steps:
            target_id = step.target.get("landmark_id")
            query = str(step.target.get("query") or "")
            skipped = False

            if step.action == "patrol_search":
                active_target = _find_object(scene, target_id, query)
                require_recheck = bool(step.target.get("require_scene_recheck"))
                if require_recheck:
                    for suspicious in step.target.get("suspicious_locations", []):
                        if not isinstance(suspicious, dict):
                            continue
                        route_to(
                            suspicious.get("approach_xy")
                            or suspicious.get("grid_xy")
                        )
                        detected = _find_object(
                            {"objects": scene.get("simulated_detections", [])},
                            target_id,
                            query,
                        )
                        expected_id = str(suspicious.get("expected_item_id") or target_id or "")
                        observed_id = str((detected or {}).get("id") or "") or None
                        change_type = "rechecked" if observed_id == expected_id else (
                            "moved" if observed_id else "missing"
                        )
                        record = {
                            "suspicious_location_id": suspicious.get("id"),
                            "change_type": change_type,
                            "expected_item_id": expected_id or None,
                            "observed_item_id": observed_id,
                            "robot_xy": list(robot_xy),
                        }
                        suspicious_rechecks.append(record)
                        if self.change_memory is not None:
                            self.change_memory.record(
                                change_type=change_type,
                                expected_item_id=expected_id or None,
                                expected_query=query,
                                map_xy=suspicious.get("map_xy"),
                                grid_xy=suspicious.get("grid_xy"),
                                observed_item_id=observed_id,
                                confidence=float((detected or {}).get("confidence") or 0.0),
                                evidence={"source": "symbolic_dry_run"},
                            )
                        if detected is not None:
                            active_target = detected
                elif active_target is not None and step.condition == "if_target_missing_or_stale":
                    skipped = True
                else:
                    detected = _find_object(
                        {"objects": scene.get("simulated_detections", [])},
                        target_id,
                        query,
                    )
                    if detected is None:
                        raise RuntimeError("Dry-run patrol could not find the requested target")
                    scene.setdefault("objects", []).append(dict(detected))
                    active_target = detected

            elif step.action == "locate_object":
                active_target = active_target or _find_object(scene, target_id, query)
                if active_target is None:
                    raise RuntimeError("Target was not located")

            elif step.action == "locate_destination":
                destination = _find_object(scene, target_id, query)
                if destination is None:
                    raise RuntimeError("Destination was not located")

            elif step.action == "navigate_to":
                obj = destination if carrying else active_target
                if obj is None:
                    obj = _find_object(scene, target_id, query)
                if obj is None:
                    raise RuntimeError(f"No navigation target for {step.id}")
                goal_raw = obj.get("approach_xy") or obj.get("grid_xy")
                if not goal_raw:
                    raise RuntimeError(f"Object {obj.get('id')} has no approach point")
                route_to(goal_raw)

            elif step.action == "grasp":
                if active_target is None:
                    raise RuntimeError("Cannot grasp without an active target")
                grasp_attempts += 1
                # The physical grasp model is outside this PC simulator. If a
                # local model is declared, only its smoke-test status is read;
                # the simulated grasp itself follows the requested success
                # assumption and never invents arm geometry.
                if grasp_model_probe["tested"] and not grasp_model_probe["usable"]:
                    raise RuntimeError("Configured grasp model failed its smoke test")
                carrying = str(active_target.get("id") or "target")
                grasp_successes += 1

            elif step.action == "verify_grasp" and carrying is None:
                raise RuntimeError("Grasp verification failed in dry run")

            elif step.action == "release":
                if carrying is None or destination is None:
                    raise RuntimeError("Cannot release without payload and destination")
                carrying = None

            trace.append(
                {
                    "step_id": step.id,
                    "action": step.action,
                    "status": "skipped" if skipped else "ok",
                    "robot_xy": list(robot_xy),
                    "carrying": carrying,
                    "hardware_command_sent": False,
                    "suspicious_rechecks": list(suspicious_rechecks)
                    if step.action == "patrol_search" else [],
                    "grasp_model_probe": grasp_model_probe
                    if step.action == "grasp" else None,
                }
            )

        return DryRunResult(
            completed=bool(trace and trace[-1]["status"] == "ok"),
            final_robot_xy=list(robot_xy),
            carrying=carrying,
            trace=trace,
            path_cells=all_paths,
            navigation_goals_reached=navigation_goals_reached,
            grasp_attempts=grasp_attempts,
            grasp_successes=grasp_successes,
            grasp_model_probe=grasp_model_probe,
            suspicious_rechecks=suspicious_rechecks,
        )
