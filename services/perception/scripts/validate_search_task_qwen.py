from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultralytics import YOLO  # noqa: E402

from qwen_grounded_tracker.search_task_planner import (  # noqa: E402
    InstructionCoordinator,
    SearchTaskPlanner,
)
from qwen_grounded_tracker.semantic_task_planner import QwenTransformersBackend  # noqa: E402


def main() -> None:
    semantic = json.loads(
        Path("/home/sjtu/car_runtime/ble_scene_17_semantics_qwen.json").read_text(
            encoding="utf-8"
        )
    )
    yolo = YOLO("models/yolo/yolo11n-seg.pt")
    backend = QwenTransformersBackend(
        "models/qwen/Qwen3-VL-2B-Instruct",
        device="cuda",
        max_new_tokens=220,
        max_time_seconds=90,
    )
    coordinator = InstructionCoordinator(
        SearchTaskPlanner(list(yolo.names.values()), backend), semantic
    )
    commands = [
        "去柜子边上找一个红色瓶子",
        "去马路上找一个红色物体",
        "去簸箕旁找一个红色物品",
        "簸箕在地图的右下角",
        "去柜子边找一个红色小物块",
    ]
    expected = [
        "executable",
        "rejected",
        "needs_clarification",
        "executable",
        "rejected",
    ]
    actual = []
    for instruction in commands:
        decision = coordinator.submit(instruction)
        actual.append(decision["status"])
        print(
            json.dumps(
                {
                    "instruction": instruction,
                    "status": decision["status"],
                    "reason": decision["reason"],
                    "question": decision.get("clarification_question"),
                    "anchor": decision.get("anchor"),
                    "target": decision.get("target"),
                    "source": decision.get("source"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if decision["status"] == "executable":
            coordinator.complete(decision["task_id"], "validation_only")
    if actual != expected:
        raise RuntimeError(f"Qwen validation states mismatch: {actual} != {expected}")
    print("REAL_QWEN_SEARCH_VALIDATION_OK", flush=True)


if __name__ == "__main__":
    main()
