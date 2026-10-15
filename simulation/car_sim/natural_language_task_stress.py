"""PC-only stress simulation for natural-language semantic robot tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
QWEN_SRC = ROOT / "3090" / "src"
if str(QWEN_SRC) not in sys.path:
    sys.path.insert(0, str(QWEN_SRC))

from qwen_grounded_tracker.scene_change_memory import SceneChangeMemory
from qwen_grounded_tracker.semantic_task_planner import SemanticTaskPlanner
from qwen_grounded_tracker.task_dry_run import SymbolicDryRunExecutor, _astar


SCENE_PATH = ROOT / "3090" / "examples" / "demo_cleanup_scene.json"


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _free_cells(rows: list[str]) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y, row in enumerate(rows)
        for x, value in enumerate(row)
        if value != "#"
    ]


def _with_dynamic_obstacles(scene: dict[str, Any], rng) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    baseline_rows = list(result["occupancy_grid"]["rows"])
    rows = [list(row) for row in baseline_rows]
    start = tuple(result["robot_pose"]["grid_xy"])
    protected = {start, (12, 10)}  # persistent bottle-change checkpoint
    for item in list(result.get("objects", [])) + list(result.get("simulated_detections", [])):
        point = item.get("approach_xy") or item.get("grid_xy")
        if point:
            protected.add((int(point[0]), int(point[1])))
    candidates = [cell for cell in _free_cells(baseline_rows) if cell not in protected]
    rng.shuffle(candidates)
    accepted = []
    for x, y in candidates:
        if len(accepted) >= 8:
            break
        previous = rows[y][x]
        rows[y][x] = "#"
        candidate_rows = ["".join(row) for row in rows]
        try:
            for goal in protected:
                _astar(candidate_rows, start, goal)
        except ValueError:
            rows[y][x] = previous
            continue
        accepted.append([x, y])
    result["occupancy_grid"]["rows"] = ["".join(row) for row in rows]
    result["session_change_layer"] = {
        "dynamic_obstacle_cells": accepted,
        "baseline_map_mutated": False,
    }
    return result


def _randomize_detection(scene: dict[str, Any], object_id: str, rng) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    detection = next(item for item in result["simulated_detections"] if item["id"] == object_id)
    free = _free_cells(result["occupancy_grid"]["rows"])
    protected = {
        tuple(item.get("approach_xy") or item.get("grid_xy"))
        for item in result.get("objects", [])
        if item.get("approach_xy") or item.get("grid_xy")
    }
    choices = [item for item in free if item not in protected and 2 <= item[0] <= 21 and 2 <= item[1] <= 13]
    x, y = choices[int(rng.integers(0, len(choices)))]
    detection["grid_xy"] = [x, y]
    detection["approach_xy"] = [x, y]
    detection["confidence"] = float(rng.uniform(0.82, 0.96))
    return result


def _run_task(
    instruction: str,
    scene: dict[str, Any],
    memory: SceneChangeMemory | None = None,
) -> dict[str, Any]:
    plan = SemanticTaskPlanner().plan(instruction, scene)
    result = SymbolicDryRunExecutor(memory).execute(plan, copy.deepcopy(scene))
    payload = result.to_dict()
    payload.update(
        instruction=instruction,
        intent=plan.intent,
        plan_source=plan.source,
        action_sequence=[step.action for step in plan.steps],
        require_scene_recheck=any(
            bool(step.target.get("require_scene_recheck")) for step in plan.steps
        ),
    )
    payload["passed"] = bool(
        payload["completed"]
        and payload["collision_contacts"] == 0
        and payload["grasp_attempts"] == payload["grasp_successes"]
    )
    return payload


def simulate_seed(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    baseline = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    baseline_digest = _digest(baseline)
    scene = _with_dynamic_obstacles(baseline, rng)
    scene = _randomize_detection(scene, "trash_01", rng)
    scene = _randomize_detection(scene, "bottle_01", rng)
    scene["grasp_model"] = {
        "available": (ROOT / "3090" / "models" / "grasp").exists(),
        "smoke_test_ok": True,
    }

    regular_tasks = [
        _run_task("去工作桌旁边", scene),
        _run_task("看看灰色柜子附近有什么", scene),
        _run_task("把蓝色水瓶放到工作桌上", scene),
        _run_task("帮我处理地上的垃圾", scene),
    ]

    bottle = next(item for item in scene["simulated_detections"] if item["id"] == "bottle_01")
    changed_scene = copy.deepcopy(scene)
    changed_scene["objects"].append({
        "id": "bottle_01",
        "name": "蓝色水瓶",
        "category": "bottle",
        "grid_xy": [12, 9],
        "approach_xy": [12, 10],
        "mobility": "frequently_moved",
        "confidence": 0.90,
    })
    memory = SceneChangeMemory(map_id=str(baseline.get("map_id") or "unknown"))
    memory.record(
        change_type="moved",
        expected_item_id="bottle_01",
        expected_query="蓝色水瓶",
        grid_xy=[12, 10],
        observed_item_id="bottle_01",
        confidence=float(bottle["confidence"]),
        evidence={"new_grid_xy": bottle["grid_xy"], "source": "simulated_camera"},
        timestamp_s=float(seed + 1),
    )
    overlay_scene = memory.overlay(changed_scene)
    revisit_task = _run_task("去蓝色水瓶旁边看看它还在不在", overlay_scene, memory)
    tasks = regular_tasks + [revisit_task]
    return {
        "seed": int(seed),
        "passed": bool(
            all(task["passed"] for task in tasks)
            and revisit_task["require_scene_recheck"]
            and len(revisit_task["suspicious_rechecks"]) >= 1
            and _digest(baseline) == baseline_digest
            and memory.snapshot()["baseline_map_mutated"] is False
        ),
        "baseline_map_digest_before": baseline_digest,
        "baseline_map_digest_after": _digest(baseline),
        "dynamic_obstacle_count": len(scene["session_change_layer"]["dynamic_obstacle_cells"]),
        "tasks": tasks,
        "scene_change_overlay": memory.snapshot(),
    }


def run(seed_count: int, output: str | Path) -> dict[str, Any]:
    seeds = [simulate_seed(seed) for seed in range(max(1, int(seed_count)))]
    tasks = [task for seed in seeds for task in seed["tasks"]]
    result = {
        "schema": "natural_language_task_stress.v1",
        "seed_count": len(seeds),
        "passed_seed_count": sum(seed["passed"] for seed in seeds),
        "passed": all(seed["passed"] for seed in seeds),
        "aggregate": {
            "total_tasks": len(tasks),
            "completed_tasks": sum(task["completed"] for task in tasks),
            "navigation_goals_reached": sum(task["navigation_goals_reached"] for task in tasks),
            "grasp_attempts": sum(task["grasp_attempts"] for task in tasks),
            "grasp_successes": sum(task["grasp_successes"] for task in tasks),
            "collision_contacts": sum(task["collision_contacts"] for task in tasks),
            "suspicious_rechecks": sum(len(task["suspicious_rechecks"]) for task in tasks),
            "baseline_map_mutations": sum(
                seed["baseline_map_digest_before"] != seed["baseline_map_digest_after"]
                for seed in seeds
            ),
            "grasp_model_available": any(
                task["grasp_model_probe"]["available"] for task in tasks
            ),
        },
        "representative_seed": seeds[0],
        "failed_seeds": [seed for seed in seeds if not seed["passed"]][:10],
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test natural-language robot tasks")
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument(
        "--output", default="car_sim/output/natural_language_task_stress.json"
    )
    args = parser.parse_args()
    result = run(args.seed_count, args.output)
    summary = dict(result)
    summary.pop("representative_seed", None)
    summary.pop("failed_seeds", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
