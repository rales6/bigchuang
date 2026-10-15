"""Create a DeepSeek phase-1 decision without starting any hardware."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .deepseek_planner import (
    DeepSeekTextClient,
    compile_phase1_decision,
    load_deepseek_config,
    sanitised_audit,
)


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"{label}不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label}不是有效JSON：{path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label}顶层必须是JSON对象：{path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用DeepSeek生成第一阶段寻物决策；本命令只写JSON，不启动底盘、摄像头或机械臂。"
    )
    parser.add_argument("instruction", help="自然语言任务")
    parser.add_argument(
        "--semantic-map",
        default="maps/semantic_release_20260919/semantic_runtime.json",
    )
    parser.add_argument(
        "--deepseek-config",
        default="secrets/deepseek_vision.json",
        help="沿用已有DeepSeek配置；密钥不会写入输出或日志",
    )
    parser.add_argument(
        "--output",
        default="search_runs/phase1/latest_deepseek_decision.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    semantic_path = Path(args.semantic_map)
    semantic_map = _load_object(semantic_path, "语义地图")
    config = load_deepseek_config(args.deepseek_config)
    contract = DeepSeekTextClient(config).plan(args.instruction, semantic_map)
    decision = compile_phase1_decision(
        args.instruction,
        contract,
        semantic_map,
        minimum_confidence=config.minimum_plan_confidence,
    )
    decision["planning_audit"] = {
        **sanitised_audit(config),
        "planned_at": datetime.now(timezone.utc).isoformat(),
        "semantic_map": str(semantic_path),
        "motion_started": False,
        "camera_started": False,
        "arm_started": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"第一阶段决策已保存：{output}")
    print(f"状态：{decision['status']}；本命令未启动任何硬件。")
    if decision.get("clarification_question"):
        print("需要确认：" + str(decision["clarification_question"]))
    return 0 if decision.get("status") == "executable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
