from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


ALLOWED_ACTIONS = {
    "inspect_semantic_map",
    "patrol_search",
    "locate_object",
    "navigate_to",
    "align_for_grasp",
    "grasp",
    "verify_grasp",
    "locate_destination",
    "release",
    "verify_completion",
    "return_home",
    "report_result",
}
ALLOWED_FAILURE_POLICIES = {"retry", "replan", "safe_stop", "report_failure"}
NAVIGATION_ACTIONS = {"patrol_search", "navigate_to", "return_home"}
MANIPULATION_ACTIONS = {"align_for_grasp", "grasp", "verify_grasp", "release"}
MANDATORY_CONSTRAINTS = [
    "任何底盘运动都同时检查激光雷达净空和摄像头动态障碍物",
    "定位或感知失效时停止，不盲目前进",
    "机械臂动作前底盘必须停止且目标在可达工作空间内",
    "每次抓取和投放后都必须通过视觉复核",
]


@dataclass
class PlanStep:
    id: str
    action: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    target: dict[str, Any] = field(default_factory=dict)
    sensors: list[str] = field(default_factory=list)
    success_condition: str = ""
    on_failure: str = "safe_stop"
    max_retries: int = 1
    condition: str = "always"


@dataclass
class SemanticTaskPlan:
    schema: str
    task_id: str
    instruction: str
    intent: str
    summary: str
    assumptions: list[str]
    global_constraints: list[str]
    steps: list[PlanStep]
    source: str = "rules"
    warnings: list[str] = field(default_factory=list)
    model_interpretation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextGenerationBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


def summarize_semantic_map(semantic_map: dict[str, Any] | None) -> dict[str, Any]:
    data = semantic_map if isinstance(semantic_map, dict) else {}
    objects = data.get("objects") or data.get("landmarks") or []
    compact_objects: list[dict[str, Any]] = []
    if isinstance(objects, list):
        for obj in objects[:80]:
            if not isinstance(obj, dict):
                continue
            compact_objects.append(
                {
                    "id": obj.get("id"),
                    "name": obj.get("name") or obj.get("label"),
                    "category": obj.get("category") or obj.get("type") or obj.get("role"),
                    "map_xy": obj.get("map_xy") or obj.get("xy"),
                    "approach_xy": obj.get("approach_xy"),
                    "mobility": obj.get("mobility", "unknown"),
                    "confidence": obj.get("confidence") or obj.get("stability"),
                }
            )
    return {
        "map_id": data.get("map_id") or data.get("schema") or "unknown",
        "robot_pose": data.get("robot_pose", {}),
        "objects": compact_objects,
        "search_regions": data.get("search_regions", [])[:20],
        "suspicious_locations": [
            {
                "id": item.get("id"),
                "expected_item_id": item.get("expected_item_id"),
                "expected_query": item.get("expected_query"),
                "map_xy": item.get("map_xy"),
                "grid_xy": item.get("grid_xy"),
                "last_change_type": item.get("last_change_type"),
                "requires_recheck_on_every_visit": bool(
                    item.get("requires_recheck_on_every_visit", True)
                ),
            }
            for item in (data.get("suspicious_locations") or [])[:40]
            if isinstance(item, dict)
        ],
    }


def build_qwen_planning_prompt(
    instruction: str,
    semantic_map: dict[str, Any] | None,
) -> str:
    map_summary = summarize_semantic_map(semantic_map)
    schema_example = {
        "schema": "robot_task_plan.v1",
        "task_id": "task_short_id",
        "instruction": instruction,
        "intent": "pick_and_place",
        "summary": "一句中文摘要",
        "assumptions": ["必要且可核验的假设"],
        "global_constraints": MANDATORY_CONSTRAINTS,
        "steps": [
            {
                "id": "step_01",
                "action": "inspect_semantic_map",
                "description": "中文动作说明",
                "depends_on": [],
                "target": {"query": "目标名称", "landmark_id": None},
                "sensors": ["semantic_map"],
                "success_condition": "可观测的成功条件",
                "on_failure": "replan",
                "max_retries": 1,
                "condition": "always",
            }
        ],
    }
    return f"""
你是移动机械臂小车的高层任务规划器。你只做任务分解，不直接输出电机转速、舵机角度、
任意代码或 shell 命令。低层导航、避障和机械臂控制由确定性控制器完成。

用户指令：{instruction}

建图完成后的语义地图摘要：
{json.dumps(map_summary, ensure_ascii=False, separators=(',', ':'))}

可用动作白名单：{sorted(ALLOWED_ACTIONS)}
失败策略白名单：{sorted(ALLOWED_FAILURE_POLICIES)}

规划规则：
1. 先利用地图中已有的稳定物体；垃圾等动态目标不可靠或不存在时，按 search_regions 巡检并用摄像头搜索。
1a. suspicious_locations 是独立变化标记，不是原图修订；任务经过相关地点时必须用摄像头复查预期物品是否仍在。
2. 所有 patrol_search、navigate_to、return_home 步骤的 sensors 必须同时包含 lidar、camera、semantic_map。
3. 定位融合应在步骤描述或成功条件中体现：地图/雷达给全局位姿，摄像头给目标识别与末端对准。
4. 抓取前必须 locate_object、navigate_to、align_for_grasp；抓取后必须 verify_grasp。
5. 投放前必须 locate_destination、navigate_to；投放后必须 verify_completion。
6. 目标或目的地缺失时采用 patrol_search/replan；感知失效、有人或近障碍时 safe_stop。
7. 只引用地图里真实存在的 landmark_id；未找到时用 null 和 query，禁止编造坐标。
8. 最多 20 步，depends_on 只能引用前面的步骤。不要输出思维过程。

严格只输出一个 JSON 对象，不要 Markdown。JSON 结构示例：
{json.dumps(schema_example, ensure_ascii=False, separators=(',', ':'))}
""".strip()


def build_qwen_intent_prompt(
    instruction: str,
    semantic_map: dict[str, Any] | None,
) -> str:
    """Ask the small model for semantics only; deterministic code compiles the steps."""
    map_summary = summarize_semantic_map(semantic_map)
    return f"""
你是移动机械臂小车的自然语言语义解析器。不要规划动作步骤，不要输出控制量、代码、
Markdown 或解释。只从用户指令和地图摘要中提取任务契约。

用户指令：{instruction}
地图摘要：{json.dumps(map_summary, ensure_ascii=False, separators=(',', ':'))}

intent 只能是 cleanup、pick_and_place、navigate、inspect、unsupported 之一。
cleanup=清理垃圾并放入清洁容器；pick_and_place=把一个物体搬到指定位置；
navigate=移动到目标附近；inspect=到目标附近观察；unsupported=机器人现有能力无法完成。
landmark_id 只能引用地图摘要中真实存在的 id；未知时必须为 null，禁止编造坐标或 id。
垃圾等经常移动的目标默认 needs_scene_patrol=true；地图中已有可靠目标时可为 false。

语义示例：
- “去工作桌旁边” → intent=navigate, target_query=工作桌
- “看看柜子附近有什么” → intent=inspect, target_query=柜子
- “把蓝色水瓶放到工作桌上” → intent=pick_and_place, target_query=蓝色水瓶, destination_query=工作桌
- “帮我处理地上的垃圾” → intent=cleanup, target_query=地面垃圾, destination_query=簸箕或垃圾桶
- “给我唱一首歌” → intent=unsupported

严格只输出一个紧凑 JSON 对象，并按顺序包含这些键：intent、target_query、
target_landmark_id、destination_query、destination_landmark_id、requested_count、
needs_scene_patrol、return_home、confidence。所有值必须来自本次用户指令和地图，
不得照抄上面的示例；没有目的地或标注 id 时使用 null（字符串查询未知时用空字符串）。
""".strip()


def parse_qwen_intent(raw: str) -> dict[str, Any]:
    payload = _extract_json_object(raw)
    intent = str(payload.get("intent") or "").strip().lower()
    if intent not in {"cleanup", "pick_and_place", "navigate", "inspect", "unsupported"}:
        raise ValueError(f"Unsupported Qwen intent: {intent!r}")
    target_query = str(payload.get("target_query") or "").strip()
    if not target_query and intent != "unsupported":
        raise ValueError("Qwen intent is missing target_query")
    return {
        "intent": intent,
        "target_query": target_query,
        "target_landmark_id": payload.get("target_landmark_id") or None,
        "destination_query": str(payload.get("destination_query") or "").strip(),
        "destination_landmark_id": payload.get("destination_landmark_id") or None,
        "requested_count": max(1, min(20, int(payload.get("requested_count") or 1))),
        "needs_scene_patrol": bool(payload.get("needs_scene_patrol", False)),
        "return_home": bool(payload.get("return_home", False)),
        "confidence": max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
    }


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Qwen output does not contain a valid JSON object")


def _normalise_step(raw: dict[str, Any], index: int) -> PlanStep:
    return PlanStep(
        id=str(raw.get("id") or f"step_{index:02d}"),
        action=str(raw.get("action") or ""),
        description=str(raw.get("description") or ""),
        depends_on=[str(item) for item in raw.get("depends_on", []) if str(item)],
        target=dict(raw.get("target") or {}),
        sensors=[str(item) for item in raw.get("sensors", []) if str(item)],
        success_condition=str(raw.get("success_condition") or ""),
        on_failure=str(raw.get("on_failure") or "safe_stop"),
        max_retries=max(0, min(3, int(raw.get("max_retries", 1)))),
        condition=str(raw.get("condition") or "always"),
    )


def validate_plan(plan: SemanticTaskPlan) -> None:
    if plan.schema != "robot_task_plan.v1":
        raise ValueError("Unsupported task plan schema")
    if not plan.steps or len(plan.steps) > 20:
        raise ValueError("Task plan must contain 1-20 steps")

    seen: set[str] = set()
    completed_actions: list[str] = []
    for step in plan.steps:
        if not step.id or step.id in seen:
            raise ValueError(f"Duplicate or empty step id: {step.id!r}")
        if step.action not in ALLOWED_ACTIONS:
            raise ValueError(f"Action is not allowed: {step.action!r}")
        if step.on_failure not in ALLOWED_FAILURE_POLICIES:
            raise ValueError(f"Failure policy is not allowed: {step.on_failure!r}")
        unknown_dependencies = set(step.depends_on) - seen
        if unknown_dependencies:
            raise ValueError(
                f"Step {step.id} has forward/unknown dependencies: {sorted(unknown_dependencies)}"
            )
        sensor_set = {item.lower() for item in step.sensors}
        if step.action in NAVIGATION_ACTIONS:
            required = {"lidar", "camera", "semantic_map"}
            if not required.issubset(sensor_set):
                raise ValueError(f"Step {step.id} navigation lacks lidar/camera/semantic_map")
        if step.action in MANIPULATION_ACTIONS and "camera" not in sensor_set:
            raise ValueError(f"Step {step.id} manipulation lacks camera verification")
        if step.action == "navigate_to":
            has_known_landmark = bool(step.target.get("landmark_id"))
            has_localisation = any(
                action in completed_actions
                for action in ("inspect_semantic_map", "patrol_search", "locate_object", "locate_destination")
            )
            if not has_known_landmark and not has_localisation:
                raise ValueError(f"Step {step.id} navigates to an unknown target before localisation")
        prerequisites = {
            "align_for_grasp": {"navigate_to"},
            "grasp": {"align_for_grasp"},
            "verify_grasp": {"grasp"},
            "locate_destination": {"verify_grasp"},
            "release": {"grasp", "locate_destination", "navigate_to"},
            "verify_completion": {"release"},
        }.get(step.action, set())
        missing = prerequisites - set(completed_actions)
        if missing:
            raise ValueError(f"Step {step.id} lacks semantic prerequisites: {sorted(missing)}")
        seen.add(step.id)
        completed_actions.append(step.action)

    if plan.intent in {"cleanup", "pick_and_place"}:
        required = {
            "locate_object",
            "navigate_to",
            "align_for_grasp",
            "grasp",
            "verify_grasp",
            "locate_destination",
            "release",
            "verify_completion",
        }
        missing = required - set(completed_actions)
        if missing:
            raise ValueError(f"Pick-and-place plan is incomplete: {sorted(missing)}")


def parse_qwen_plan(raw: str, instruction: str) -> SemanticTaskPlan:
    payload = _extract_json_object(raw)
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Task plan steps must be an array")
    plan = SemanticTaskPlan(
        schema=str(payload.get("schema") or ""),
        task_id=str(payload.get("task_id") or "task_qwen"),
        instruction=str(payload.get("instruction") or instruction),
        intent=str(payload.get("intent") or "unknown"),
        summary=str(payload.get("summary") or ""),
        assumptions=[str(item) for item in payload.get("assumptions", [])],
        global_constraints=[str(item) for item in payload.get("global_constraints", [])],
        steps=[_normalise_step(item, index) for index, item in enumerate(raw_steps, 1) if isinstance(item, dict)],
        source="qwen",
    )
    for constraint in MANDATORY_CONSTRAINTS:
        if constraint not in plan.global_constraints:
            plan.global_constraints.append(constraint)
    validate_plan(plan)
    return plan


def plan_from_dict(payload: dict[str, Any]) -> SemanticTaskPlan:
    """Rebuild and validate a plan received through an API boundary."""
    if not isinstance(payload, dict):
        raise ValueError("Task plan payload must be a JSON object")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("Task plan steps must be an array")
    plan = SemanticTaskPlan(
        schema=str(payload.get("schema") or ""),
        task_id=str(payload.get("task_id") or ""),
        instruction=str(payload.get("instruction") or ""),
        intent=str(payload.get("intent") or "unknown"),
        summary=str(payload.get("summary") or ""),
        assumptions=[str(item) for item in payload.get("assumptions", [])],
        global_constraints=[str(item) for item in payload.get("global_constraints", [])],
        steps=[
            _normalise_step(item, index)
            for index, item in enumerate(raw_steps, 1)
            if isinstance(item, dict)
        ],
        source=str(payload.get("source") or "api"),
        warnings=[str(item) for item in payload.get("warnings", [])],
        model_interpretation=dict(payload.get("model_interpretation") or {}),
    )
    validate_plan(plan)
    return plan


def _object_matches(obj: dict[str, Any], aliases: tuple[str, ...]) -> bool:
    text = " ".join(
        str(obj.get(key, "")).lower() for key in ("id", "name", "label", "category", "type", "role")
    )
    for alias in aliases:
        folded = alias.lower()
        # Short English aliases such as "bin" must not match "cabinet".
        if folded.isascii() and re.fullmatch(r"[a-z0-9_ -]+", folded):
            escaped_alias = re.escape(folded).replace("\\ ", "[ _-]+")
            pattern = rf"(?<![a-z0-9]){escaped_alias}(?![a-z0-9])"
            if re.search(pattern, text):
                return True
        elif folded in text:
            return True
    return False


def _find_landmark(
    semantic_map: dict[str, Any] | None,
    aliases: tuple[str, ...],
) -> dict[str, Any] | None:
    data = semantic_map if isinstance(semantic_map, dict) else {}
    objects = data.get("objects") or data.get("landmarks") or []
    if not isinstance(objects, list):
        return None
    candidates = [item for item in objects if isinstance(item, dict) and _object_matches(item, aliases)]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("confidence") or item.get("stability") or 0.0))


def _find_landmark_by_query(
    semantic_map: dict[str, Any] | None,
    query: str,
) -> dict[str, Any] | None:
    folded = query.lower().strip()
    alias_groups = (
        ("垃圾", "trash", "rubbish", "waste"),
        ("簸箕", "dustpan"),
        ("垃圾桶", "trash bin", "trash_bin", "dustbin"),
        ("桌", "table", "desk"),
        ("柜", "cabinet", "cupboard"),
        ("瓶", "bottle"),
        ("椅", "chair"),
        ("墙角", "corner"),
        ("扫帚", "broom"),
        ("纸箱", "box", "carton"),
    )
    aliases: tuple[str, ...] = tuple(
        alias
        for group in alias_groups
        if any(alias in folded for alias in group)
        for alias in group
    )
    if aliases:
        match = _find_landmark(semantic_map, aliases)
        if match is not None:
            return match
    data = semantic_map if isinstance(semantic_map, dict) else {}
    objects = data.get("objects") or data.get("landmarks") or []
    candidates = []
    for item in objects if isinstance(objects, list) else []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get(key, "")).lower() for key in ("id", "name", "label", "type", "role"))
        if folded and (folded in text or text in folded):
            candidates.append(item)
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("confidence") or item.get("stability") or 0.0))


def _landmark_id_matches_query(
    semantic_map: dict[str, Any] | None,
    landmark_id: Any,
    query: str,
) -> bool:
    if not landmark_id or not query.strip():
        return False
    data = semantic_map if isinstance(semantic_map, dict) else {}
    objects = data.get("objects") or data.get("landmarks") or []
    if not isinstance(objects, list):
        return False
    selected = next(
        (
            item
            for item in objects
            if isinstance(item, dict) and str(item.get("id")) == str(landmark_id)
        ),
        None,
    )
    if selected is None:
        return False
    folded = query.lower().strip()
    alias_groups = (
        ("垃圾", "trash", "rubbish", "waste"),
        ("簸箕", "dustpan"),
        ("垃圾桶", "trash bin", "trash_bin", "dustbin"),
        ("桌", "table", "desk"),
        ("柜", "cabinet", "cupboard"),
        ("瓶", "bottle"),
        ("椅", "chair"),
        ("墙角", "corner"),
        ("扫帚", "broom"),
        ("纸箱", "box", "carton"),
    )
    aliases = tuple(
        alias
        for group in alias_groups
        if any(alias in folded for alias in group)
        for alias in group
    )
    if aliases:
        return _object_matches(selected, aliases)
    selected_text = " ".join(
        str(selected.get(key, "")).lower()
        for key in ("id", "name", "label", "category", "type", "role")
    )
    return folded in selected_text or selected_text in folded


def _explicit_instruction_contract(
    instruction: str,
    semantic_map: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a conservative lexical contract only for unambiguous command forms."""
    text = instruction.strip()
    folded = text.lower()
    intent: str | None = None
    target_query = text
    destination_query = ""
    needs_scene_patrol = False

    if any(word in folded for word in ("唱歌", "讲笑话", "天气", "打电话", "sing", "joke", "weather")):
        intent = "unsupported"
    elif any(word in folded for word in ("垃圾", "trash", "rubbish", "waste")):
        intent = "cleanup"
        target_query = "地面垃圾"
        destination_query = "簸箕或垃圾桶"
        needs_scene_patrol = True
    else:
        transfer = re.search(r"(?:把|将)?\s*(.+?)\s*(?:放到|放在|送到|搬到|移到)\s*(.+)", text)
        if transfer:
            intent = "pick_and_place"
            target_query = transfer.group(1).strip()
            destination_query = transfer.group(2).strip().rstrip("。！？")
            needs_scene_patrol = True
        elif any(word in folded for word in ("看看", "查看", "观察", "检查", "有什么", "inspect")):
            intent = "inspect"
        elif re.search(r"^(?:请)?(?:去|前往|到|导航到|go to|navigate to)", folded):
            intent = "navigate"

    if intent is None:
        return None
    target = _find_landmark_by_query(semantic_map, target_query)
    destination = _find_landmark_by_query(semantic_map, destination_query) if destination_query else None
    return {
        "intent": intent,
        "target_query": target_query,
        "target_landmark_id": target.get("id") if target else None,
        "destination_query": destination_query,
        "destination_landmark_id": destination.get("id") if destination else None,
        "needs_scene_patrol": needs_scene_patrol,
    }


def _map_has_landmark_id(semantic_map: dict[str, Any] | None, landmark_id: Any) -> bool:
    if not landmark_id:
        return False
    data = semantic_map if isinstance(semantic_map, dict) else {}
    objects = data.get("objects") or data.get("landmarks") or []
    return isinstance(objects, list) and any(
        isinstance(item, dict) and str(item.get("id")) == str(landmark_id) for item in objects
    )


def _suspicious_locations_for_target(
    semantic_map: dict[str, Any] | None,
    query: str,
    landmark: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    data = semantic_map if isinstance(semantic_map, dict) else {}
    query_folded = str(query or "").lower().strip()
    landmark_id = str((landmark or {}).get("id") or "")
    landmark_xy = (landmark or {}).get("map_xy") or (landmark or {}).get("xy")
    result = []
    for item in data.get("suspicious_locations", []) or []:
        if not isinstance(item, dict) or not item.get("requires_recheck_on_every_visit", True):
            continue
        expected_id = str(item.get("expected_item_id") or "")
        expected_query = str(item.get("expected_query") or "").lower().strip()
        semantic_match = bool(
            (landmark_id and expected_id == landmark_id)
            or (
                query_folded and expected_query
                and (query_folded in expected_query or expected_query in query_folded)
            )
        )
        item_xy = item.get("map_xy")
        spatial_match = False
        if (
            isinstance(landmark_xy, (list, tuple)) and len(landmark_xy) >= 2
            and isinstance(item_xy, (list, tuple)) and len(item_xy) >= 2
        ):
            try:
                spatial_match = (
                    (float(landmark_xy[0]) - float(item_xy[0])) ** 2
                    + (float(landmark_xy[1]) - float(item_xy[1])) ** 2
                ) ** 0.5 <= 0.90
            except (TypeError, ValueError):
                spatial_match = False
        if semantic_match or spatial_match:
            result.append(dict(item))
    return result


def _target_reference_with_recheck(
    semantic_map: dict[str, Any] | None,
    query: str,
    landmark_id: Any,
    landmark: dict[str, Any] | None,
) -> dict[str, Any]:
    suspicions = _suspicious_locations_for_target(semantic_map, query, landmark)
    reference = {
        "query": query,
        "landmark_id": landmark_id,
        "expected_map_xy": (landmark or {}).get("map_xy") or (landmark or {}).get("xy"),
        "expected_grid_xy": (landmark or {}).get("grid_xy"),
        "require_scene_recheck": bool(suspicions),
        "suspicious_location_ids": [str(item.get("id")) for item in suspicions],
        "suspicious_locations": suspicions,
    }
    return reference


def _step(
    number: int,
    action: str,
    description: str,
    *,
    depends_on: list[str] | None = None,
    target: dict[str, Any] | None = None,
    sensors: list[str] | None = None,
    success_condition: str,
    on_failure: str = "safe_stop",
    condition: str = "always",
    max_retries: int = 1,
) -> PlanStep:
    return PlanStep(
        id=f"step_{number:02d}",
        action=action,
        description=description,
        depends_on=list(depends_on or []),
        target=dict(target or {}),
        sensors=list(sensors or []),
        success_condition=success_condition,
        on_failure=on_failure,
        max_retries=max_retries,
        condition=condition,
    )


def build_rule_plan(
    instruction: str,
    semantic_map: dict[str, Any] | None,
    semantic_contract: dict[str, Any] | None = None,
) -> SemanticTaskPlan:
    text = instruction.strip()
    folded = text.lower()
    contract = semantic_contract if isinstance(semantic_contract, dict) else {}
    contract_intent = str(contract.get("intent") or "")
    cleanup_task = contract_intent == "cleanup" or any(
        word in folded for word in ("垃圾", "trash", "rubbish", "waste")
    )
    unsupported_task = contract_intent == "unsupported" or any(
        word in folded for word in ("唱歌", "讲笑话", "天气", "打电话", "sing", "joke", "weather")
    )
    if unsupported_task:
        return build_unsupported_plan(text, contract)

    rule_pick = any(
        word in folded
        for word in ("捡", "拿", "抓", "夹", "搬", "放到", "放在", "送到", "pick", "grab", "place")
    )
    manipulation_task = cleanup_task or contract_intent == "pick_and_place" or rule_pick

    if not manipulation_task:
        inferred_contract = dict(contract)
        if not contract_intent:
            if any(word in folded for word in ("看看", "查看", "观察", "检查", "有什么", "inspect")):
                inferred_contract["intent"] = "inspect"
            elif any(word in folded for word in ("去", "前往", "到", "导航", "navigate", "go to")):
                inferred_contract["intent"] = "navigate"
            else:
                return build_unsupported_plan(text)
        return _build_generic_navigation_plan(text, semantic_map, inferred_contract)

    rule_target_query = text
    rule_destination_query = "任务指定的放置位置"
    if not cleanup_task and not contract.get("target_query"):
        match = re.search(r"(?:把|将)?\s*(.+?)\s*(?:放到|放在|送到|搬到|移到)\s*(.+)", text)
        if match:
            rule_target_query = match.group(1).strip()
            rule_destination_query = match.group(2).strip().rstrip("。！？")
    target_query = str(
        contract.get("target_query") or ("地面上的垃圾" if cleanup_task else rule_target_query)
    )
    destination_query = str(
        contract.get("destination_query")
        or ("簸箕或垃圾桶的开口" if cleanup_task else rule_destination_query)
    )
    target_landmark = _find_landmark_by_query(semantic_map, target_query)
    destination = _find_landmark_by_query(semantic_map, destination_query)
    contract_target_id = contract.get("target_landmark_id")
    contract_destination_id = contract.get("destination_landmark_id")
    target_id_is_consistent = _landmark_id_matches_query(
        semantic_map, contract_target_id, target_query
    )
    destination_id_is_consistent = _landmark_id_matches_query(
        semantic_map, contract_destination_id, destination_query
    )
    target_ref = _target_reference_with_recheck(
        semantic_map,
        target_query,
        contract_target_id
        if target_id_is_consistent
        else (target_landmark.get("id") if target_landmark else None),
        target_landmark,
    )
    destination_ref = _target_reference_with_recheck(
        semantic_map,
        destination_query,
        contract_destination_id
        if destination_id_is_consistent
        else (destination.get("id") if destination else None),
        destination,
    )
    target_label = target_query or "目标物体"
    destination_label = destination_query or "放置位置"
    nav_sensors = ["lidar", "camera", "semantic_map", "odometry"]
    steps = [
        _step(
            1,
            "inspect_semantic_map",
            f"读取占据栅格、机器人位姿和物品标注，确认{target_label}与{destination_label}的位置。",
            target=target_ref,
            sensors=["semantic_map", "lidar"],
            success_condition="地图坐标系有效，并得到目标/目的地候选或确认需要巡检",
            on_failure="report_failure",
        ),
        _step(
            2,
            "patrol_search",
            f"沿地图中可达的搜索区域巡检；雷达定位和避障，摄像头寻找{target_label}。",
            depends_on=["step_01"],
            target=target_ref,
            sensors=nav_sensors,
            success_condition=f"摄像头发现{target_label}且其位置可投影到地图",
            on_failure="replan",
            condition=(
                "if_target_missing_stale_or_suspicious"
                if target_ref["require_scene_recheck"]
                else (
                    "if_target_missing_or_stale"
                    if bool(contract.get("needs_scene_patrol", target_landmark is None))
                    else "if_target_stale"
                )
            ),
            max_retries=2,
        ),
        _step(
            3,
            "locate_object",
            f"融合视觉目标方位、雷达地图位姿和里程计，确定{target_label}的地图位置与置信度。",
            depends_on=["step_01", "step_02"],
            target=target_ref,
            sensors=["camera", "lidar", "semantic_map", "odometry"],
            success_condition=f"{target_label}定位置信度达标且存在可达抓取接近点",
            on_failure="retry",
            max_retries=2,
        ),
        _step(
            4,
            "navigate_to",
            f"规划到{target_label}抓取接近点的无碰路径并低速导航，持续执行雷达与视觉双重避障。",
            depends_on=["step_03"],
            target=target_ref,
            sensors=nav_sensors,
            success_condition="到达抓取接近点且前方净空满足安全阈值",
            on_failure="replan",
            max_retries=2,
        ),
        _step(
            5,
            "align_for_grasp",
            f"停止底盘，使用摄像头闭环微调车体和机械臂，使{target_label}位于夹爪可达区。",
            depends_on=["step_04"],
            target=target_ref,
            sensors=["camera", "lidar"],
            success_condition="底盘静止、目标居中且深度/尺度满足抓取范围",
            on_failure="retry",
            max_retries=2,
        ),
        _step(
            6,
            "grasp",
            f"对{target_label}执行受限的预抓取、下降和闭合夹爪动作。",
            depends_on=["step_05"],
            target=target_ref,
            sensors=["camera", "arm_feedback"],
            success_condition="夹爪闭合且负载/位置反馈正常",
            on_failure="safe_stop",
        ),
        _step(
            7,
            "verify_grasp",
            f"抬升少量距离并用视觉与夹爪反馈确认{target_label}已被稳定夹持。",
            depends_on=["step_06"],
            target=target_ref,
            sensors=["camera", "arm_feedback"],
            success_condition="目标随夹爪移动且夹持稳定",
            on_failure="retry",
            max_retries=1,
        ),
        _step(
            8,
            "locate_destination",
            f"优先读取地图中的{destination_label}标注，并用摄像头复核当前位置和可放置区域。",
            depends_on=["step_07"],
            target=destination_ref,
            sensors=["semantic_map", "camera", "lidar"],
            success_condition="目的地存在、位置置信度达标且开口可见",
            on_failure="replan",
            max_retries=2,
        ),
        _step(
            9,
            "navigate_to",
            f"保持夹爪稳定，规划到{destination_label}投放接近点的路径并执行双重避障导航。",
            depends_on=["step_08"],
            target=destination_ref,
            sensors=nav_sensors,
            success_condition="安全到达投放接近点并保持垃圾夹持",
            on_failure="safe_stop",
            max_retries=1,
        ),
        _step(
            10,
            "release",
            f"底盘停止后将夹爪移动到{destination_label}的安全放置区域并松开{target_label}。",
            depends_on=["step_09"],
            target=destination_ref,
            sensors=["camera", "arm_feedback"],
            success_condition="夹爪已打开且目标脱离夹爪",
            on_failure="safe_stop",
        ),
        _step(
            11,
            "verify_completion",
            f"用摄像头确认{target_label}已位于{destination_label}，并检查机械臂已安全退出。",
            depends_on=["step_10"],
            target=destination_ref,
            sensors=["camera", "semantic_map"],
            success_condition=f"{target_label}位于{destination_label}，原抓取位置已清理",
            on_failure="report_failure",
        ),
        _step(
            12,
            "report_result",
            "停止底盘和机械臂，向用户报告任务结果与任何异常。",
            depends_on=["step_11"],
            sensors=["camera", "lidar", "arm_feedback"],
            success_condition="安全停止并返回结构化结果",
            on_failure="safe_stop",
        ),
    ]
    warnings = []
    if contract_target_id and not _map_has_landmark_id(semantic_map, contract_target_id):
        warnings.append(f"忽略 Qwen 编造或过期的目标标注 id：{contract_target_id}")
    elif contract_target_id and not target_id_is_consistent:
        warnings.append(
            f"忽略与目标“{target_label}”语义不一致的标注 id：{contract_target_id}"
        )
    if contract_destination_id and not _map_has_landmark_id(semantic_map, contract_destination_id):
        warnings.append(f"忽略 Qwen 编造或过期的目的地标注 id：{contract_destination_id}")
    elif contract_destination_id and not destination_id_is_consistent:
        warnings.append(
            f"忽略与目的地“{destination_label}”语义不一致的标注 id：{contract_destination_id}"
        )
    if destination is None and not destination_id_is_consistent:
        warnings.append(f"地图中没有可靠的目的地“{destination_label}”，执行前必须补充标注或搜索。")
    plan = SemanticTaskPlan(
        schema="robot_task_plan.v1",
        task_id="cleanup_floor_trash" if cleanup_task else "semantic_pick_and_place",
        instruction=text,
        intent="cleanup" if cleanup_task else "pick_and_place",
        summary=(
            "巡检并定位地面垃圾，安全抓取后运送到簸箕或垃圾桶。"
            if cleanup_task
            else f"定位并抓取{target_label}，安全运送到{destination_label}。"
        ),
        assumptions=[f"{target_label}尺寸和重量在夹爪能力范围内", "地图与当前场景未发生大幅变化"],
        global_constraints=list(MANDATORY_CONSTRAINTS),
        steps=steps,
        source="rules",
        warnings=warnings,
        model_interpretation=dict(contract),
    )
    validate_plan(plan)
    return plan


def build_unsupported_plan(
    instruction: str,
    semantic_contract: dict[str, Any] | None = None,
) -> SemanticTaskPlan:
    contract = semantic_contract if isinstance(semantic_contract, dict) else {}
    step = _step(
        1,
        "report_result",
        "保持底盘和机械臂停止，并报告当前机器人能力无法安全完成该指令。",
        sensors=["lidar", "camera"],
        success_condition="机器人保持停止，用户收到不支持原因",
        on_failure="safe_stop",
    )
    plan = SemanticTaskPlan(
        schema="robot_task_plan.v1",
        task_id="unsupported_task",
        instruction=instruction,
        intent="unsupported",
        summary="当前能力无法安全执行该自然语言任务。",
        assumptions=[],
        global_constraints=list(MANDATORY_CONSTRAINTS),
        steps=[step],
        source="qwen_compiled" if contract else "rules",
        warnings=["任务未映射到导航、观察、抓取或清理能力，未生成运动动作。"],
        model_interpretation=dict(contract),
    )
    validate_plan(plan)
    return plan


def _build_generic_navigation_plan(
    instruction: str,
    semantic_map: dict[str, Any] | None,
    semantic_contract: dict[str, Any] | None = None,
) -> SemanticTaskPlan:
    contract = semantic_contract if isinstance(semantic_contract, dict) else {}
    query = str(contract.get("target_query") or instruction)
    matched = _find_landmark_by_query(semantic_map, query)
    contract_id = contract.get("target_landmark_id")
    contract_id_is_consistent = _landmark_id_matches_query(semantic_map, contract_id, query)
    target = _target_reference_with_recheck(
        semantic_map,
        query,
        contract_id
        if contract_id_is_consistent
        else (matched.get("id") if matched else None),
        matched,
    )
    steps = [
        _step(
            1,
            "inspect_semantic_map",
            "解析指令并在语义地图中检索目标。",
            target=target,
            sensors=["semantic_map"],
            success_condition="得到可靠目标或确认需要视觉搜索",
            on_failure="report_failure",
        ),
        _step(
            2,
            "patrol_search",
            "目标未知时沿安全巡检点搜索并建立视觉-地图关联。",
            depends_on=["step_01"],
            target=target,
            sensors=["lidar", "camera", "semantic_map"],
            success_condition="目标被识别并定位",
            on_failure="report_failure",
            condition=(
                "if_target_missing_stale_or_suspicious"
                if target["require_scene_recheck"]
                else "if_target_missing_or_stale"
            ),
        ),
        _step(
            3,
            "navigate_to",
            "在雷达与视觉双重避障下导航到目标安全观察点。",
            depends_on=["step_01", "step_02"],
            target=target,
            sensors=["lidar", "camera", "semantic_map", "odometry"],
            success_condition="到达目标安全距离并停止",
            on_failure="replan",
        ),
        _step(
            4,
            "report_result",
            "报告到达结果并保持安全停止。",
            depends_on=["step_03"],
            sensors=["camera", "lidar"],
            success_condition="已报告结果",
            on_failure="safe_stop",
        ),
    ]
    plan = SemanticTaskPlan(
        schema="robot_task_plan.v1",
        task_id="semantic_inspection"
        if contract.get("intent") == "inspect"
        else "generic_semantic_navigation",
        instruction=instruction,
        intent="inspect" if contract.get("intent") == "inspect" else "navigate",
        summary=f"搜索“{query}”并导航到安全观察点。",
        assumptions=["目标可以被当前传感器识别"],
        global_constraints=list(MANDATORY_CONSTRAINTS),
        steps=steps,
        source="rules",
        model_interpretation=dict(contract),
    )
    if contract_id and not _map_has_landmark_id(semantic_map, contract_id):
        plan.warnings.append(f"忽略 Qwen 编造或过期的目标标注 id：{contract_id}")
    elif contract_id and not contract_id_is_consistent:
        plan.warnings.append(f"忽略与目标“{query}”语义不一致的标注 id：{contract_id}")
    validate_plan(plan)
    return plan


class QwenTransformersBackend:
    """Lazy local Qwen backend used by the standalone 3090 planning service."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        local_files_only: bool = True,
        max_new_tokens: int = 320,
        max_time_seconds: float = 60.0,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.max_time_seconds = max_time_seconds
        self.model: Any | None = None
        self.processor: Any | None = None
        self.torch: Any | None = None

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        if self.local_files_only and not Path(self.model_path).expanduser().exists():
            raise FileNotFoundError(f"Qwen model not found: {self.model_path}")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        use_cuda = self.device.lower().startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        dtype = torch.bfloat16 if use_cuda else torch.float32
        kwargs = {"local_files_only": self.local_files_only, "low_cpu_mem_usage": True}
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, dtype=dtype, **kwargs
            )
        except TypeError:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, torch_dtype=dtype, **kwargs
            )
        self.model.to("cuda" if use_cuda else "cpu")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=self.local_files_only
        )
        self.torch = torch

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        assert self.model is not None and self.processor is not None and self.torch is not None
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                max_time=self.max_time_seconds,
                do_sample=False,
            )
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, output_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


class SemanticTaskPlanner:
    def __init__(self, backend: TextGenerationBackend | None = None) -> None:
        self.backend = backend

    def plan(
        self,
        instruction: str,
        semantic_map: dict[str, Any] | None = None,
    ) -> SemanticTaskPlan:
        text = instruction.strip()
        if not text:
            raise ValueError("Instruction must not be empty")
        if self.backend is None:
            return build_rule_plan(text, semantic_map)
        try:
            prompt = build_qwen_intent_prompt(text, semantic_map)
            contract = parse_qwen_intent(self.backend.generate(prompt))
            explicit = _explicit_instruction_contract(text, semantic_map)
            corrected = bool(explicit and explicit["intent"] != contract["intent"])
            if corrected:
                raw_intent = contract["intent"]
                contract.update(explicit)
                contract["model_raw_intent"] = raw_intent
                contract["corrected_by_rules"] = True
            if contract["intent"] == "unsupported":
                plan = build_unsupported_plan(text, contract)
                if corrected:
                    plan.warnings.append(
                        f"Qwen 意图“{contract['model_raw_intent']}”与明确指令句式冲突，已纠正为“unsupported”。"
                    )
                return plan
            plan = build_rule_plan(text, semantic_map, contract)
            plan.source = "qwen_compiled"
            if corrected:
                plan.warnings.append(
                    f"Qwen 意图“{contract['model_raw_intent']}”与明确指令句式冲突，"
                    f"已纠正为“{contract['intent']}”。"
                )
            return plan
        except Exception as exc:
            plan = build_rule_plan(text, semantic_map)
            plan.source = "rules_fallback"
            plan.warnings.append(f"Qwen 计划不可用，已安全回退到规则规划器：{exc}")
            return plan
