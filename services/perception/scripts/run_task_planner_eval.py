from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.semantic_task_planner import (
    QwenTransformersBackend,
    SemanticTaskPlanner,
)


def _evaluate(case: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    interpretation = plan.get("model_interpretation") or {}
    actual_intent = interpretation.get("intent") or plan.get("intent")
    if actual_intent != case.get("expected_intent"):
        errors.append(f"intent={actual_intent!r}, expected={case.get('expected_intent')!r}")
    steps = plan.get("steps") or []
    actions = [step.get("action") for step in steps]
    for action in case.get("required_actions", []):
        if action not in actions:
            errors.append(f"missing action {action}")
    for action in case.get("forbidden_actions", []):
        if action in actions:
            errors.append(f"forbidden action {action}")
    targets = [step.get("target") or {} for step in steps]
    resolved_target_ids = {item.get("landmark_id") for item in targets}
    for forbidden_id in case.get("forbidden_target_ids", []):
        if forbidden_id in resolved_target_ids:
            errors.append(f"forbidden target id {forbidden_id} was accepted")
    expected_target = case.get("expected_target_id")
    if expected_target and expected_target not in resolved_target_ids:
        errors.append(f"target id {expected_target} not resolved")
    expected_destination = case.get("expected_destination_id")
    if expected_destination:
        destination_steps = [
            step for step in steps if step.get("action") in {"locate_destination", "release"}
        ]
        if expected_destination not in {
            (step.get("target") or {}).get("landmark_id") for step in destination_steps
        }:
            errors.append(f"destination id {expected_destination} not resolved")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate semantic task planning on several Chinese tasks.")
    parser.add_argument("--backend", choices=("rules", "qwen"), default="qwen")
    parser.add_argument("--model-path", default="models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scene", default="examples/demo_cleanup_scene.json")
    parser.add_argument("--cases", default="examples/task_eval_cases.json")
    parser.add_argument("--output", default="outputs/task_planner_eval.json")
    args = parser.parse_args()

    def rooted(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    scene = json.loads(rooted(args.scene).read_text(encoding="utf-8"))
    cases = json.loads(rooted(args.cases).read_text(encoding="utf-8"))
    backend = None
    if args.backend == "qwen":
        backend = QwenTransformersBackend(str(rooted(args.model_path)), device=args.device)
    planner = SemanticTaskPlanner(backend)
    results = []
    for case in cases:
        started = time.perf_counter()
        plan = planner.plan(str(case["instruction"]), scene).to_dict()
        errors = _evaluate(case, plan)
        results.append(
            {
                "name": case["name"],
                "instruction": case["instruction"],
                "passed": not errors,
                "errors": errors,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "source": plan["source"],
                "model_interpretation": plan.get("model_interpretation", {}),
                "actions": [step["action"] for step in plan["steps"]],
                "warnings": plan.get("warnings", []),
            }
        )
    report = {
        "schema": "task_planner_eval.v1",
        "backend": args.backend,
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
        "all_passed": all(item["passed"] for item in results),
        "results": results,
    }
    output = rooted(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
