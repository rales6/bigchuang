from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.semantic_task_planner import (
    QwenTransformersBackend,
    SemanticTaskPlanner,
)
from qwen_grounded_tracker.task_dry_run import SymbolicDryRunExecutor


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan and dry-run a semantic robot task.")
    parser.add_argument("--instruction", default="帮我处理地上的垃圾")
    parser.add_argument("--scene", default="examples/demo_cleanup_scene.json")
    parser.add_argument("--backend", choices=("rules", "qwen"), default="rules")
    parser.add_argument("--model-path", default="models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/semantic_task_demo.json")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = ROOT / scene_path
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    backend = None
    if args.backend == "qwen":
        model_path = Path(args.model_path).expanduser()
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        backend = QwenTransformersBackend(str(model_path), device=args.device)

    plan = SemanticTaskPlanner(backend).plan(args.instruction, scene)
    dry_run_scene = json.loads(json.dumps(scene, ensure_ascii=False))
    try:
        dry_run = SymbolicDryRunExecutor().execute(plan, dry_run_scene).to_dict()
    except Exception as exc:
        dry_run = {
            "completed": False,
            "error": str(exc),
            "trace": [],
            "path_cells": [],
            "hardware_command_sent": False,
        }
    payload = {"plan": plan.to_dict(), "dry_run": dry_run}
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_path}")
    return 0 if dry_run.get("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
