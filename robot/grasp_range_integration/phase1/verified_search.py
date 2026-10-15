"""Run the existing find-only mission and verify its handoff to visual pursuit.

The protected mission may return process exit code zero after a normal *failed*
search.  A following grasp task must therefore inspect this run's mission.json
and the final, region-eligible stereo observation before taking over hardware.
This module never calls the arm or controls the chassis itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image

from .pursuit_entry_contract import assess_pursuit_entry

from .deepseek_planner import validate_phase1_decision


RELEASE_DIR = Path(__file__).resolve().parents[2]
CAR_ROOT = Path(os.environ.get("CAR_ROOT", RELEASE_DIR.parent)).resolve()
SEARCH_SCRIPT = RELEASE_DIR / "grasp_range_integration/phase1/run_phase1_find_only.sh"
OUTPUT_ROOT = CAR_ROOT / "search_runs/ble_scene_17"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return value


def verify_handoff(decision: dict[str, Any], mission_dir: Path,
                   *, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Require a fresh stereo success, not just a zero subprocess exit code."""
    validate_phase1_decision(decision, require_executable=True)
    mission_dir = mission_dir.resolve(strict=True)
    if mission_dir.parent != output_root.resolve(strict=True):
        raise ValueError("寻物结果目录不在指定任务输出根目录内")
    mission = _read_object(mission_dir / "mission.json")
    if (mission.get("instruction") != decision.get("instruction")
            or mission.get("result") != "success"
            or mission.get("hardware_motion_enabled") is not True
            or not mission.get("finished_at")):
        raise ValueError("本次寻物未确认成功、指令不匹配，或底盘并非实车模式")
    events = mission.get("events") or []
    if not isinstance(events, list):
        raise ValueError("寻物事件记录无效")
    terminals = [event for event in events if isinstance(event, dict)
                 and event.get("type") == "supervisor_directive"
                 and (event.get("directive") or {}).get("terminal") is True]
    if (not terminals or (terminals[-1].get("directive") or {}).get("action")
            != "report_success"):
        raise ValueError("缺少最终双摄成功指令")
    observations = [event for event in events if isinstance(event, dict)
                    and event.get("type") == "observation_complete"]
    if not observations:
        raise ValueError("寻物成功但没有停车观察记录")
    observed = observations[-1]
    if (observed.get("dual_camera_confirmed") is not True
            or set(observed.get("complete_cameras") or []) != {"left", "right"}):
        raise ValueError("最后一次观察缺少完整双摄确认")
    target = decision.get("target") or {}
    pairs: dict[str, set[str]] = {}
    eligible_items = []
    for item in observed.get("detections") or []:
        if not isinstance(item, dict):
            continue
        evidence = item.get("search_evidence") or {}
        if (item.get("class_name") != target.get("class_name")
                or item.get("task_region_eligible") is not True
                or item.get("dual_camera_confirmed") is not True
                or item.get("confirmed_match") is not True
                or not all(evidence.get(key) is True for key in
                           ("source_ok", "class_ok", "colour_ok", "bbox_complete"))):
            continue
        pair_id = str(item.get("stereo_pair_id") or "")
        camera = str(item.get("camera_id") or "")
        if pair_id and camera in {"left", "right"}:
            pairs.setdefault(pair_id, set()).add(camera)
            eligible_items.append(item)
    if not any(cameras == {"left", "right"} for cameras in pairs.values()):
        raise ValueError("找不到同一目标的左右双摄合格检测")
    archived = observed.get("archived_images") or {}
    images = {}
    for camera in ("left", "right"):
        relative = Path(str(archived.get(camera) or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("停车观察照片路径不可信")
        path = (mission_dir / relative).resolve(strict=True)
        if not path.is_relative_to(mission_dir) or not path.is_file():
            raise ValueError("停车观察照片缺失或越界")
        images[camera] = str(path)
    if images["left"] == images["right"]:
        raise ValueError("左右摄像头不能引用同一张照片")
    with Image.open(images["left"]) as left_image, Image.open(images["right"]) as right_image:
        image_sizes = {"left": left_image.size, "right": right_image.size}
    pursuit_entry = assess_pursuit_entry(eligible_items, image_sizes)
    if pursuit_entry is None or not pursuit_entry["ready"]:
        raise ValueError("双摄结果尚未满足后续追踪入口视野：" + str(
            (pursuit_entry or {}).get("reason") or "missing_confirmed_pair"))
    confirmed = pursuit_entry["stereo_pair_id"]
    return {
        "schema": "phase1_verified_handoff.v1",
        "mission_dir": str(mission_dir),
        "decision_task_id": decision.get("task_id"),
        "instruction": decision.get("instruction"),
        "target": target,
        "stereo_pair_id": confirmed,
        "observation_id": observed.get("observation_id"),
        "images": images,
        "pursuit_entry": pursuit_entry,
        "chassis_mission_exited": True,
        "object_range_established": False,
        "arm_grasp_authorized": False,
        "next_stage": "fresh_visual_pursuit_and_hull_alignment",
    }


def run(decision_path: Path, *, runner=subprocess.Popen,
        output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    decision_path = decision_path.resolve(strict=True)
    decision = _read_object(decision_path)
    validate_phase1_decision(decision, require_executable=True)
    output_root.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in output_root.iterdir() if path.is_dir()}
    command = ["bash", str(SEARCH_SCRIPT), str(decision_path), "--execute-find-only"]
    process = runner(command, cwd=str(CAR_ROOT), stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                     errors="replace", bufsize=1)
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    if returncode != 0:
        raise RuntimeError(f"第一阶段寻物进程退出码={returncode}；不启动视觉追踪或机械臂")
    fresh = [path for path in output_root.iterdir()
             if path.is_dir() and path.name not in before and (path / "mission.json").is_file()]
    if len(fresh) != 1:
        raise RuntimeError(f"本次寻物结果目录应恰好一个，实际={len(fresh)}；拒绝复用旧成功记录")
    handoff = verify_handoff(decision, fresh[0], output_root=output_root)
    destination = fresh[0] / "phase1_verified_handoff.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(handoff, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return handoff


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--execute-verified-search", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_verified_search:
        parser.error("实车寻物必须显式添加 --execute-verified-search")
    try:
        result = run(args.decision)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"寻物交接拒绝：{exc}", file=sys.stderr, flush=True)
        return 2
    print("寻物交接验证通过：" + json.dumps({
        "mission_dir": result["mission_dir"],
        "target": result["target"],
        "next_stage": result["next_stage"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
