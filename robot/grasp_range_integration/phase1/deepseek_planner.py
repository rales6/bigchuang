"""DeepSeek-backed, find-only task planning.

DeepSeek extracts a semantic contract.  Deterministic code resolves that
contract against the immutable semantic map and creates the decision consumed
by ``raspberry_pi.object_search_mission``.  Model-provided coordinates, paths,
motor commands and arm commands are deliberately ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
import uuid


PLACEHOLDER_MARKERS = ("请填写", "your_", "replace_", "changeme", "sk-xxx")
ALLOWED_INTENTS = {"find_object_near_anchor", "unsupported"}
ALLOWED_CURRENT_ACTIONS = {"find_and_stop"}
ALLOWED_STEPS = {
    "localize",
    "plan_search_route",
    "navigate_to_search_area",
    "capture_at_stop",
    "visual_detect_and_filter",
    "continue_nearby_search",
    "stop_and_report",
}
FORBIDDEN_ACTION_MARKERS = (
    "grasp", "pick", "arm", "servo", "manipulat", "夹取", "抓取", "机械臂", "舵机",
)
COLOUR_ALIASES = {
    "红色": "red", "红": "red", "red": "red",
    "蓝色": "blue", "蓝": "blue", "blue": "blue",
    "绿色": "green", "绿": "green", "green": "green",
    "黄色": "yellow", "黄": "yellow", "yellow": "yellow",
}
CLASS_ALIASES = {
    "小物块": "block", "物块": "block", "方块": "block", "积木": "block",
    "block": "block", "瓶子": "bottle", "水瓶": "bottle", "bottle": "bottle",
    "杯子": "cup", "杯": "cup", "cup": "cup",
}
ANCHOR_ALIASES = {
    "柜子": ("柜子", "沿墙柜子", "cabinet", "cupboard"),
    "桌子": ("桌子", "中央桌子", "table", "desk"),
    "门": ("门", "door"),
    "窗户": ("窗户", "窗", "window"),
    "玻璃墙": ("玻璃墙", "glass wall", "glass_wall"),
}


@dataclass(frozen=True)
class DeepSeekPlanConfig:
    enabled: bool
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_s: float
    minimum_plan_confidence: float
    source_path: Path

    @property
    def configured(self) -> bool:
        key = self.api_key.strip().lower()
        return bool(
            self.enabled
            and key
            and not any(marker in key for marker in PLACEHOLDER_MARKERS)
        )


def load_deepseek_config(path: str | Path) -> DeepSeekPlanConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"DeepSeek配置不存在：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek配置不是有效JSON：{source}") from exc
    if not isinstance(payload, dict):
        raise ValueError("DeepSeek配置顶层必须是JSON对象")
    config = DeepSeekPlanConfig(
        enabled=bool(payload.get("enabled", True)),
        api_key=str(payload.get("api_key") or ""),
        base_url=str(payload.get("base_url") or "https://api.deepseek.com").rstrip("/"),
        model=str(payload.get("model") or "deepseek-flash"),
        timeout_s=max(10.0, min(180.0, float(payload.get("timeout_s") or 90.0))),
        minimum_plan_confidence=max(
            0.50,
            min(0.95, float(payload.get("minimum_plan_confidence") or 0.68)),
        ),
        source_path=source,
    )
    if not config.configured:
        raise ValueError("DeepSeek未启用或API密钥仍为空/占位值")
    return config


def _extract_json(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw.strip()):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw.strip()[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("DeepSeek响应中没有可解析的JSON对象")


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def parse_deepseek_contract(raw: str) -> dict[str, Any]:
    payload = _extract_json(raw)
    target = payload.get("target")
    if not isinstance(target, dict):
        target = {}
    intent = str(payload.get("intent") or "unsupported").strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = "unsupported"
    current_action = str(payload.get("current_phase_action") or "find_and_stop").strip().lower()
    if current_action not in ALLOWED_CURRENT_ACTIONS:
        current_action = "find_and_stop"
    return {
        "intent": intent,
        "anchor_query": str(payload.get("anchor_query") or "").strip(),
        "target": {
            "query": str(target.get("query") or "").strip(),
            "color": str(target.get("color") or "").strip().lower(),
            "class_name": str(target.get("class_name") or "").strip().lower(),
            "count": max(1, min(10, int(target.get("count") or 1))),
        },
        "current_phase_action": current_action,
        "requested_final_action": str(
            payload.get("requested_final_action") or "none"
        ).strip().lower(),
        "confidence": _clamp_confidence(payload.get("confidence")),
        "needs_clarification": bool(payload.get("needs_clarification", False)),
        "clarification_question": str(payload.get("clarification_question") or "").strip(),
        "reason": str(payload.get("reason") or "").strip(),
    }


def _map_summary(semantic_map: dict[str, Any]) -> dict[str, Any]:
    objects = {
        str(item.get("id")): item
        for item in (semantic_map.get("objects") or [])
        if isinstance(item, dict) and item.get("id")
    }
    regions = []
    for region in semantic_map.get("search_regions") or []:
        if not isinstance(region, dict):
            continue
        linked = objects.get(str(region.get("id")), {})
        regions.append({
            "id": region.get("id"),
            "query": region.get("query"),
            "name": linked.get("name"),
            "aliases": linked.get("aliases") or [],
            "category": linked.get("category"),
        })
    return {"map_id": semantic_map.get("map_id"), "available_search_regions": regions}


def build_prompt(instruction: str, semantic_map: dict[str, Any]) -> str:
    return f"""
你是移动小车第一阶段任务规划器。本阶段的唯一职责是：理解用户目标，选择语义搜索区域，
让小车到附近用相机找到目标后停车。即使用户说“夹取/抓取”，本阶段也绝不执行机械臂、
夹爪、目标精对准或夹取；只把最终动作记为 requested_final_action="grasp"，留给以后阶段。

用户指令：{instruction}
可用语义搜索区域（不含坐标）：{json.dumps(_map_summary(semantic_map), ensure_ascii=False, separators=(',', ':'))}

规则：
1. 只提取语义；禁止输出或猜测坐标、路径、速度、电机、舵机命令。
2. intent只能是find_object_near_anchor或unsupported。
3. current_phase_action必须是find_and_stop。
4. 目标类别用简短英文；“物块/方块/积木”统一为block。用户没有指定颜色时color填空字符串，不要猜颜色。
5. anchor_query使用用户提到的参照物名称，例如“柜子”。
6. 信息不足时needs_clarification=true；不要编造参照物。

严格只输出JSON对象：
{{"intent":"find_object_near_anchor","anchor_query":"柜子","target":{{"query":"红色小物块","color":"red","class_name":"block","count":1}},"current_phase_action":"find_and_stop","requested_final_action":"grasp","confidence":0.95,"needs_clarification":false,"clarification_question":"","reason":"去柜子搜索区域寻找红色小物块，找到即停车"}}
""".strip()


class DeepSeekTextClient:
    def __init__(self, config: DeepSeekPlanConfig) -> None:
        self.config = config

    def plan(self, instruction: str, semantic_map: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": build_prompt(instruction, semantic_map)}],
        }
        request = urllib_request.Request(
            self.config.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.config.timeout_s) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, TimeoutError, urllib_error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"DeepSeek规划请求失败：{exc}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek响应缺少choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek规划响应为空")
        return parse_deepseek_contract(content)


def _explicit_alias(text: str, aliases: dict[str, str]) -> str:
    folded = text.lower()
    for phrase, canonical in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase.lower() in folded:
            return canonical
    return ""


def _explicit_anchor(text: str) -> str:
    folded = text.lower()
    for canonical, aliases in ANCHOR_ALIASES.items():
        if any(alias.lower() in folded for alias in aliases):
            return canonical
    return ""


def _entity_text(item: dict[str, Any]) -> str:
    values = [
        item.get("id"), item.get("query"), item.get("name"), item.get("category"),
        item.get("type"), item.get("label"), *(item.get("aliases") or []),
    ]
    return " ".join(str(value).strip().lower() for value in values if value)


def _anchor_terms(query: str) -> set[str]:
    folded = query.strip().lower()
    terms = {folded} if folded else set()
    for canonical, aliases in ANCHOR_ALIASES.items():
        candidates = {canonical.lower(), *(alias.lower() for alias in aliases)}
        if folded in candidates or any(value in folded for value in candidates):
            terms.update(candidates)
    return {term for term in terms if term}


def resolve_search_region(
    semantic_map: dict[str, Any], anchor_query: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    objects = {
        str(item.get("id")): item
        for item in (semantic_map.get("objects") or [])
        if isinstance(item, dict) and item.get("id")
    }
    terms = _anchor_terms(anchor_query)
    candidates: list[tuple[int, float, dict[str, Any], dict[str, Any]]] = []
    for region in semantic_map.get("search_regions") or []:
        if not isinstance(region, dict):
            continue
        xy = region.get("map_xy")
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            continue
        linked = objects.get(str(region.get("id")), {})
        text = _entity_text({**linked, **region})
        score = sum(1 for term in terms if term in text)
        if score:
            candidates.append((score, float(linked.get("confidence") or 0.0), region, linked))
    if not candidates:
        return None
    _, _, region, linked = max(candidates, key=lambda row: (row[0], row[1]))
    return dict(region), dict(linked)


def _base_target(contract: dict[str, Any], instruction: str) -> dict[str, Any]:
    model_target = contract.get("target") or {}
    explicit_colour = _explicit_alias(instruction, COLOUR_ALIASES)
    explicit_class = _explicit_alias(instruction, CLASS_ALIASES)
    colour = explicit_colour or str(model_target.get("color") or "").strip().lower()
    class_name = explicit_class or str(model_target.get("class_name") or "").strip().lower()
    query = str(model_target.get("query") or "").strip()
    if not query:
        query = " ".join(value for value in (colour, class_name) if value) or instruction
    return {
        "query": query,
        "color": colour,
        "class_name": class_name,
        "generic_object": False,
        "requested_count": max(1, min(10, int(model_target.get("count") or 1))),
        "detection_mode": "qwen_fallback" if class_name == "block" else "qwen_fallback",
    }


def _sanitise_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Keep semantic fields only; discard any model-suggested control data."""
    target = contract.get("target")
    if not isinstance(target, dict):
        target = {}
    return {
        "intent": str(contract.get("intent") or "unsupported"),
        "anchor_query": str(contract.get("anchor_query") or ""),
        "target": {
            "query": str(target.get("query") or ""),
            "color": str(target.get("color") or ""),
            "class_name": str(target.get("class_name") or ""),
            "count": max(1, min(10, int(target.get("count") or 1))),
        },
        "current_phase_action": "find_and_stop",
        "requested_final_action": str(contract.get("requested_final_action") or "none"),
        "confidence": _clamp_confidence(contract.get("confidence")),
        "needs_clarification": bool(contract.get("needs_clarification", False)),
        "clarification_question": str(contract.get("clarification_question") or ""),
        "reason": str(contract.get("reason") or ""),
    }


def _non_executable_decision(
    instruction: str,
    contract: dict[str, Any],
    target: dict[str, Any],
    reason: str,
    question: str,
) -> dict[str, Any]:
    return {
        "schema": "robot_object_search_plan.v1",
        "task_id": "deepseek_phase1_" + uuid.uuid4().hex[:10],
        "instruction": instruction,
        "status": "needs_clarification",
        "intent": "find_object",
        "reason": reason,
        "clarification_question": question,
        "target": target,
        "anchor": {},
        "route_request": {},
        "steps": [],
        "capabilities": {"grasp_enabled": False, "arm_commands_allowed": False},
        "source": "deepseek_validated",
        "warnings": [],
        "model_interpretation": _sanitise_contract(contract),
        "phase_control": {
            "phase": "phase1_find_only",
            "terminal_condition": "target_found_and_vehicle_stopped",
            "deferred_actions": ["grasp"],
            "hardware_scope": ["localization", "base_navigation", "camera_detection"],
        },
    }


def compile_phase1_decision(
    instruction: str,
    contract: dict[str, Any],
    semantic_map: dict[str, Any],
    *,
    minimum_confidence: float = 0.68,
) -> dict[str, Any]:
    """Compile model semantics into a map-grounded, find-only decision."""
    text = instruction.strip()
    if not text:
        raise ValueError("instruction must not be empty")
    target = _base_target(contract, text)
    explicit_anchor = _explicit_anchor(text)
    anchor_query = explicit_anchor or str(contract.get("anchor_query") or "").strip()

    if contract.get("intent") != "find_object_near_anchor":
        return _non_executable_decision(
            text, contract, target, "DeepSeek未能确认这是目标附近寻物任务。",
            "请重新说明要在哪个已标注物体附近寻找什么物品。",
        )
    if bool(contract.get("needs_clarification")) or float(contract.get("confidence") or 0.0) < minimum_confidence:
        return _non_executable_decision(
            text, contract, target, "DeepSeek对任务语义的置信度不足，拒绝启动底盘。",
            str(contract.get("clarification_question") or "请确认参照物和目标类别。"),
        )
    if not target["class_name"]:
        return _non_executable_decision(
            text, contract, target, "目标物品类别不完整。", "请说明要找的物品类别；颜色可以不指定。",
        )
    if not anchor_query:
        return _non_executable_decision(
            text, contract, target, "没有可用于地图检索的参照物。", "请说明目标位于哪个已标注物体附近。",
        )

    resolved = resolve_search_region(semantic_map, anchor_query)
    if resolved is None:
        return _non_executable_decision(
            text, contract, target, f"语义地图没有与‘{anchor_query}’匹配的搜索区域。",
            f"请确认‘{anchor_query}’是否是地图中的已标注区域。",
        )
    region, linked = resolved
    xy = region["map_xy"]
    requested_final_action = str(contract.get("requested_final_action") or "none").lower()
    if any(word in text for word in ("夹取", "抓取", "拿取")):
        requested_final_action = "grasp"
    warnings = [
        "柜子位置仅作为未核实语义搜索区域；必须到达观察点后用实时双摄确认目标。",
        "用户请求中的夹取已明确延后；本阶段找到目标后停车，不发送任何机械臂或夹爪命令。",
    ]
    decision = {
        "schema": "robot_object_search_plan.v1",
        "task_id": "deepseek_phase1_" + uuid.uuid4().hex[:10],
        "instruction": text,
        "status": "executable",
        "intent": "find_object",
        "reason": "DeepSeek已解析目标与参照物；确定性代码已将参照物绑定到语义搜索区域。本阶段找到后停车。",
        "clarification_question": "",
        "target": target,
        "anchor": {
            "query": anchor_query,
            "landmark_id": None,
            "search_region_id": str(region.get("id") or ""),
            "name": str(region.get("query") or linked.get("name") or anchor_query),
            "map_xy": [float(xy[0]), float(xy[1])],
            "region_hint": "",
            "confidence": float(linked.get("confidence") or 0.0),
            "position_verified": False,
            "requires_visual_recheck": True,
        },
        "route_request": {
            "mode": "semantic_search_region",
            "center_xy_m": [float(xy[0]), float(xy[1])],
            "radius_m": max(0.4, float(region.get("radius_m") or 1.2)),
            "maximum_views": 8,
            "require_live_visual_recheck": True,
            "planning_policy": "semantic_visibility_regions.v1",
            "inspect_current_view_first": True,
            "inspect_transit_views": False,
            "camera_horizontal_fov_deg": 82.0,
            "recognition_min_distance_m": 0.45,
            "recognition_max_distance_m": 1.80,
            "minimum_work_area_coverage": 0.25,
            "stop_on_visibility_entry": True,
            "transit_observation_spacing_m": 0.75,
            "visibility_bands": [
                {"name": "high", "confidence": 0.85, "standoff_m": 0.90, "fov_margin_deg": 10.0},
                {"name": "medium", "confidence": 0.70, "standoff_m": 1.10, "fov_margin_deg": 7.0},
                {"name": "low", "confidence": 0.55, "standoff_m": 1.30, "fov_margin_deg": 4.0},
            ],
            "minimum_work_area_standoff_m": 0.80,
        },
        "steps": [
            {"id": "step_01", "action": "localize", "executor": "raspberry_pi", "success": "已知地图定位通过一致性验证"},
            {"id": "step_02", "action": "plan_search_route", "executor": "raspberry_pi", "success": "A*生成语义区域内的可达观察点"},
            {"id": "step_03", "action": "navigate_to_search_area", "executor": "raspberry_pi", "success": "到达首个可达观察点", "safety": ["lidar_clearance", "temporary_obstacles", "short_ttl"]},
            {"id": "step_04", "action": "capture_at_stop", "executor": "raspberry_pi", "success": "停车稳定后采集双摄图像"},
            {"id": "step_05", "action": "visual_detect_and_filter", "executor": "3090", "success": "类别、可选颜色、边界框和双摄证据通过", "target": target},
            {"id": "step_06", "action": "continue_nearby_search", "executor": "3090+raspberry_pi", "success": "未找到时继续区域内剩余观察点"},
            {"id": "step_07", "action": "stop_and_report", "executor": "raspberry_pi", "success": "找到目标或观察点耗尽后停车并报告"},
        ],
        "capabilities": {
            "detection_mode": target["detection_mode"],
            "qwen_visual_fallback": True,
            "colour_filter": target["color"],
            "class_supported": False,
            "grasp_enabled": False,
            "arm_commands_allowed": False,
        },
        "source": "deepseek_validated",
        "warnings": warnings,
        "model_interpretation": _sanitise_contract(contract),
        "phase_control": {
            "phase": "phase1_find_only",
            "current_action": "find_and_stop",
            "requested_final_action": requested_final_action,
            "terminal_condition": "target_found_and_vehicle_stopped",
            "deferred_actions": ["grasp"] if requested_final_action == "grasp" else [],
            "hardware_scope": ["localization", "base_navigation", "camera_detection"],
        },
    }
    validate_phase1_decision(decision, require_executable=True)
    return decision


def validate_phase1_decision(
    decision: dict[str, Any], *, require_executable: bool = True
) -> None:
    """Fail closed if a decision exceeds the first-stage authority."""
    if decision.get("schema") != "robot_object_search_plan.v1":
        raise ValueError("决策schema不受支持")
    if require_executable and decision.get("status") != "executable":
        raise ValueError(f"决策不可执行：{decision.get('status')}")
    if decision.get("status") != "executable":
        return
    phase = decision.get("phase_control") or {}
    if phase.get("phase") != "phase1_find_only":
        raise ValueError("缺少phase1_find_only执行边界")
    if phase.get("terminal_condition") != "target_found_and_vehicle_stopped":
        raise ValueError("第一阶段终止条件不是找到并停车")
    capabilities = decision.get("capabilities") or {}
    if capabilities.get("grasp_enabled") is not False or capabilities.get("arm_commands_allowed") is not False:
        raise ValueError("第一阶段禁止启用机械臂或夹取")
    route = decision.get("route_request") or {}
    if route.get("mode") != "semantic_search_region":
        raise ValueError("第一阶段只允许semantic_search_region路线")
    if route.get("planning_policy") != "semantic_visibility_regions.v1":
        raise ValueError("第一阶段必须按目标可见区域规划，禁止恢复为目标点规划")
    if route.get("inspect_current_view_first") is not True:
        raise ValueError("第一阶段必须在移动前先检查当前视野")
    if route.get("stop_on_visibility_entry") is not True:
        raise ValueError("第一阶段必须在进入有效视野后停车识别")
    if not isinstance(route.get("center_xy_m"), list) or len(route["center_xy_m"]) != 2:
        raise ValueError("搜索区域中心格式无效")
    actions = [str(step.get("action") or "") for step in (decision.get("steps") or [])]
    if not actions or any(action not in ALLOWED_STEPS for action in actions):
        raise ValueError("决策包含第一阶段以外的步骤")
    action_text = " ".join(actions).lower()
    if any(marker in action_text for marker in FORBIDDEN_ACTION_MARKERS):
        raise ValueError("决策包含机械臂或夹取步骤")
    target = decision.get("target") or {}
    if not target.get("class_name"):
        raise ValueError("目标类别为空")
    anchor = decision.get("anchor") or {}
    if not anchor.get("search_region_id") or anchor.get("position_verified") is not False:
        raise ValueError("必须使用未核实语义搜索区域并进行实时复核")


def sanitised_audit(config: DeepSeekPlanConfig) -> dict[str, Any]:
    """Return safe configuration metadata; the API key is never included."""
    return {
        "provider": "deepseek",
        "model": config.model,
        "base_url": config.base_url,
        "config_source": str(config.source_path),
        "api_key_logged": False,
    }
