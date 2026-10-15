# -*- coding: utf-8 -*-
"""Print a compact diagnostic summary for the latest pursuit run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def resolve_run_dir(value: str | None, log_root: Path) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    pointer = log_root / "LATEST.txt"
    if not pointer.exists():
        raise FileNotFoundError(f"找不到 {pointer}，尚无任务日志")
    return Path(pointer.read_text(encoding="utf-8").strip()).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总最近一次视觉追踪任务日志")
    parser.add_argument("run_dir", nargs="?", help="任务目录；省略则读取 LATEST.txt")
    parser.add_argument("--log-root", default="logs/pursuit")
    args = parser.parse_args()
    run_dir = resolve_run_dir(args.run_dir, Path(args.log_root).expanduser())
    metadata_path = run_dir / "run.json"
    telemetry_path = run_dir / "telemetry.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) \
        if metadata_path.exists() else {}
    if telemetry_path.exists():
        with telemetry_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []

    states = Counter(row.get("state", "") for row in rows)
    sources = Counter(row.get("src", "") or "none" for row in rows)
    tracked = sum(row.get("tracking") == "1" for row in rows)
    both = sum(row.get("tracking") == "1" and row.get("tracking2") == "1"
               for row in rows)

    def values(key: str) -> list[float]:
        return [value for row in rows if (value := _number(row, key)) is not None]

    age = values("perception_age_ms")
    perception_fps = values("perception_fps")
    upload_fps = values("upload_fps")
    left_fps = values("left_fps")
    right_fps = values("right_fps")

    def mean(items: list[float]) -> str:
        return f"{sum(items) / len(items):.2f}" if items else "n/a"

    duration = rows[-1].get("t_s", "0") if rows else metadata.get("duration_s", 0)
    print(f"任务目录: {run_dir}")
    print(f"指令: {metadata.get('instruction', 'n/a')}")
    print(f"模式/结果: {metadata.get('mode', 'n/a')} / "
          f"{metadata.get('status', 'n/a')} / {metadata.get('final_result', '')}")
    print(f"时长/采样行: {duration}s / {len(rows)}")
    print(f"状态占比(秒): {dict(states)}")
    print(f"主路追踪率: {tracked}/{len(rows)}; 双路同时追踪: {both}/{len(rows)}")
    print(f"追踪来源: {dict(sources)}")
    print(f"平均感知: {mean(perception_fps)} fps; 平均年龄: {mean(age)} ms; "
          f"最大年龄: {max(age):.1f} ms" if age else
          f"平均感知: {mean(perception_fps)} fps; 感知年龄: n/a")
    print(f"平均采集 L/R: {mean(left_fps)}/{mean(right_fps)} fps; "
          f"平均上传: {mean(upload_fps)} fps")
    print(f"详细终端日志: {run_dir / 'console.log'}")
    print(f"逐秒遥测: {telemetry_path}" + ("" if telemetry_path.exists()
                                         else "（启动阶段失败，文件未建立）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
