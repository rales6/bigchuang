"""Validate an already prepared phase-1 decision before real execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deepseek_planner import validate_phase1_decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("decision")
    args = parser.parse_args()
    path = Path(args.decision)
    decision = json.loads(path.read_text(encoding="utf-8"))
    validate_phase1_decision(decision, require_executable=True)
    target = decision["target"]
    anchor = decision["anchor"]
    print(
        "第一阶段门检通过：target={} {}，region={}，终止条件=找到并停车，夹取=禁用".format(
            target.get("color"), target.get("class_name"), anchor.get("search_region_id")
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
