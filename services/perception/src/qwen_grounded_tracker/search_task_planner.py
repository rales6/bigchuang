from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


DECISION_STATES = {"executable", "needs_clarification", "rejected"}
REGION_HINTS = {
    "右下角": "bottom_right",
    "右下": "bottom_right",
    "左下角": "bottom_left",
    "左下": "bottom_left",
    "右上角": "top_right",
    "右上": "top_right",
    "左上角": "top_left",
    "左上": "top_left",
    "中间": "center",
    "中央": "center",
}
COLOURS = {
    "红色": "red",
    "红": "red",
    "蓝色": "blue",
    "蓝": "blue",
    "绿色": "green",
    "绿": "green",
    "黄色": "yellow",
    "黄": "yellow",
    "黑色": "black",
    "黑": "black",
    "白色": "white",
    "白": "white",
    "灰色": "gray",
    "灰": "gray",
}
CLASS_ALIASES = {
    "瓶子": "bottle",
    "水瓶": "bottle",
    "瓶": "bottle",
    "杯子": "cup",
    "杯": "cup",
    "碗": "bowl",
    "书": "book",
    "手机": "cell phone",
    "电话": "cell phone",
    "遥控器": "remote",
    "剪刀": "scissors",
    "球": "sports ball",
    "玩具熊": "teddy bear",
    "泰迪熊": "teddy bear",
    "背包": "backpack",
    "手提包": "handbag",
    "箱子": "suitcase",
    "行李箱": "suitcase",
    "椅子": "chair",
    "花盆": "potted plant",
    "苹果": "apple",
    "橙子": "orange",
    "香蕉": "banana",
    "积木": "block",
    "物块": "block",
    "方块": "block",
    "小块": "block",
}
GENERIC_OBJECT_WORDS = ("物品", "物体", "东西", "目标", "object", "item")
OUTDOOR_WORDS = (
    "马路",
    "公路",
    "高速",
    "街道",
    "室外",
    "校园外",
    "road",
    "street",
    "highway",
    "outdoor",
)
SEARCH_WORDS = ("找", "寻找", "搜索", "发现", "看看有没有", "find", "search", "look for")
SAFETY_ONLY_CLASSES = {"person", "car", "motorcycle", "bus", "truck", "bicycle", "dog"}


class TextBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class SearchTaskDecision:
    schema: str
    task_id: str
    instruction: str
    status: str
    intent: str
    reason: str
    clarification_question: str = ""
    target: dict[str, Any] = field(default_factory=dict)
    anchor: dict[str, Any] = field(default_factory=dict)
    route_request: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    source: str = "rules"
    warnings: list[str] = field(default_factory=list)
    model_interpretation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fold(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _entity_text(item: dict[str, Any]) -> str:
    annotation = item.get("qwen_annotation")
    if not isinstance(annotation, dict):
        annotation = {}
    values = [
        item.get("id"),
        item.get("target_id"),
        item.get("name"),
        item.get("label"),
        item.get("category"),
        item.get("semantic_label"),
        annotation.get("semantic_label"),
        annotation.get("name_zh"),
        annotation.get("category"),
        annotation.get("description_zh"),
    ]
    return " ".join(_fold(value) for value in values if value)


def semantic_landmarks(semantic_map: dict[str, Any] | None) -> list[dict[str, Any]]:
    data = semantic_map if isinstance(semantic_map, dict) else {}
    result: list[dict[str, Any]] = []
    original = data.get("objects") or data.get("landmarks") or []
    if isinstance(original, list):
        for item in original:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            candidate.setdefault("id", candidate.get("target_id"))
            candidate.setdefault("map_xy", candidate.get("xy"))
            result.append(candidate)
    entities = data.get("entities")
    if isinstance(entities, dict):
        for key, item in entities.items():
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            annotation = candidate.get("qwen_annotation")
            if not isinstance(annotation, dict):
                annotation = {}
            candidate["id"] = str(candidate.get("target_id") or key)
            candidate["name"] = str(
                annotation.get("name_zh")
                or candidate.get("name")
                or candidate.get("semantic_label")
                or candidate["id"]
            )
            candidate["category"] = str(
                annotation.get("category")
                or annotation.get("semantic_label")
                or candidate.get("semantic_label")
                or candidate.get("kind")
                or "unknown"
            )
            if candidate.get("target_x_m") is not None and candidate.get("target_y_m") is not None:
                candidate["map_xy"] = [
                    float(candidate["target_x_m"]),
                    float(candidate["target_y_m"]),
                ]
            candidate["confidence"] = float(
                annotation.get("confidence")
                or candidate.get("semantic_label_confidence")
                or candidate.get("qwen_candidate_confidence")
                or 0.0
            )
            candidate["needs_human_review"] = bool(annotation.get("needs_human_review", False))
            result.append(candidate)
    deduplicated: dict[str, dict[str, Any]] = {}
    for item in result:
        identity = str(item.get("id") or "")
        if identity:
            deduplicated[identity] = item
    return list(deduplicated.values())


def summarize_search_map(semantic_map: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "map_id": (semantic_map or {}).get("map_id") or (semantic_map or {}).get("map") or "unknown",
        "landmarks": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "category": item.get("category"),
                "map_xy": item.get("map_xy"),
                "confidence": item.get("confidence"),
                "needs_human_review": item.get("needs_human_review", False),
            }
            for item in semantic_landmarks(semantic_map)
        ],
    }


def find_landmark(semantic_map: dict[str, Any] | None, query: str) -> dict[str, Any] | None:
    folded = _fold(query)
    if not folded:
        return None
    aliases = {folded}
    translations = {
        "柜子": ("cabinet", "cupboard"),
        "桌子": ("table", "desk"),
        "椅子": ("chair",),
        "墙": ("wall",),
        "簸箕": ("dustpan",),
        "垃圾桶": ("trash bin", "dustbin"),
        "门": ("door",),
        "窗": ("window",),
    }
    for word, values in translations.items():
        if word in folded:
            aliases.update(values)
    candidates = []
    for item in semantic_landmarks(semantic_map):
        text = _entity_text(item)
        score = sum(1 for alias in aliases if alias and alias in text)
        if score:
            candidates.append((score, float(item.get("confidence") or 0.0), item))
    if not candidates:
        return None
    return dict(max(candidates, key=lambda value: (value[0], value[1]))[2])


def _extract_region_hint(text: str) -> str:
    folded = _fold(text)
    for word, region in REGION_HINTS.items():
        if word in folded:
            return region
    return ""


def _extract_colour(text: str) -> str:
    folded = _fold(text)
    for word, colour in COLOURS.items():
        if word in folded:
            return colour
    return ""


def _extract_class(text: str) -> tuple[str, bool]:
    folded = _fold(text)
    for word, label in sorted(CLASS_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if word in folded:
            return label, False
    generic = any(word in folded for word in GENERIC_OBJECT_WORDS)
    return ("any_yolo_object" if generic else ""), generic


def _extract_anchor(text: str) -> str:
    folded = str(text or "").strip()
    patterns = (
        r"(?:去|到|前往)?\s*([^，。！？,]{1,24}?)(?:边上|旁边|附近|周围|旁)",
        r"(?:near|around)\s+([a-z0-9 _-]{1,30})",
    )
    for pattern in patterns:
        match = re.search(pattern, folded, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"^(?:去|到|前往)", "", candidate).strip()
            return candidate
    return ""


def _rule_contract(instruction: str) -> dict[str, Any]:
    folded = _fold(instruction)
    target_class, generic = _extract_class(instruction)
    return {
        "intent": "find_object" if any(word in folded for word in SEARCH_WORDS) else "unknown",
        "target_query": instruction,
        "target_color": _extract_colour(instruction),
        "target_class": target_class,
        "generic_object": generic,
        "anchor_query": _extract_anchor(instruction),
        "region_hint": _extract_region_hint(instruction),
        "environment": "outside_mapped_area" if any(word in folded for word in OUTDOOR_WORDS) else "mapped_indoor",
        "requested_count": 1,
        "confidence": 1.0,
    }


def build_search_contract_prompt(
    instruction: str,
    semantic_map: dict[str, Any] | None,
    pending: dict[str, Any] | None = None,
) -> str:
    return f"""
你是室内移动小车的寻物指令语义解析器。只提取语义，不输出路径坐标、电机速度、代码或解释。
用户指令：{instruction}
上一条等待补充的寻物契约：{json.dumps(pending or {}, ensure_ascii=False, separators=(',', ':'))}
当前永久语义地图：{json.dumps(summarize_search_map(semantic_map), ensure_ascii=False, separators=(',', ':'))}

intent只能是find_object、navigate、unsupported。environment只能是mapped_indoor、outside_mapped_area、unknown。
region_hint只能是top_left、top_right、bottom_left、bottom_right、center或空字符串。
target_class使用简短英文类别；泛指“物品/物体”填any_yolo_object。anchor_query是“柜子、桌子、簸箕”等
参照物名称。地图没有参照物且用户没有给方位时，不得编造landmark_id或坐标。

严格只输出JSON：
{{"intent":"find_object","target_query":"红色瓶子","target_color":"red","target_class":"bottle","generic_object":false,"anchor_query":"柜子","region_hint":"","environment":"mapped_indoor","requested_count":1,"confidence":0.9}}
""".strip()


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Qwen output has no JSON object")


def parse_search_contract(raw: str) -> dict[str, Any]:
    payload = _extract_json(raw)
    intent = str(payload.get("intent") or "unknown").strip().lower()
    if intent not in {"find_object", "navigate", "unsupported"}:
        raise ValueError(f"unsupported search intent: {intent}")
    region = str(payload.get("region_hint") or "").strip().lower()
    if region not in {"", "top_left", "top_right", "bottom_left", "bottom_right", "center"}:
        region = ""
    environment = str(payload.get("environment") or "unknown").strip().lower()
    if environment not in {"mapped_indoor", "outside_mapped_area", "unknown"}:
        environment = "unknown"
    return {
        "intent": intent,
        "target_query": str(payload.get("target_query") or "").strip(),
        "target_color": str(payload.get("target_color") or "").strip().lower(),
        "target_class": str(payload.get("target_class") or "").strip().lower(),
        "generic_object": bool(payload.get("generic_object", False)),
        "anchor_query": str(payload.get("anchor_query") or "").strip(),
        "region_hint": region,
        "environment": environment,
        "requested_count": max(1, min(10, int(payload.get("requested_count") or 1))),
        "confidence": max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
    }


class SearchTaskPlanner:
    def __init__(
        self,
        yolo_classes: list[str] | set[str] | tuple[str, ...],
        backend: TextBackend | None = None,
    ) -> None:
        self.yolo_classes = {_fold(value) for value in yolo_classes if _fold(value)}
        self.backend = backend

    def parse(
        self,
        instruction: str,
        semantic_map: dict[str, Any] | None,
        pending: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str, list[str]]:
        rules = _rule_contract(instruction)
        warnings: list[str] = []
        if self.backend is None:
            return rules, "rules", warnings
        try:
            model = parse_search_contract(
                self.backend.generate(build_search_contract_prompt(instruction, semantic_map, pending))
            )
            # Safety and explicitly stated lexical facts override model guesses.
            for key in ("target_color", "target_class", "anchor_query", "region_hint"):
                if rules.get(key):
                    model[key] = rules[key]
            if rules["environment"] == "outside_mapped_area":
                model["environment"] = "outside_mapped_area"
            if rules["intent"] == "find_object":
                model["intent"] = "find_object"
            return model, "qwen_validated", warnings
        except Exception as exc:
            warnings.append(f"Qwen语义解析失败，使用保守规则：{exc}")
            return rules, "rules_fallback", warnings

    def plan(
        self,
        instruction: str,
        semantic_map: dict[str, Any] | None,
        pending: dict[str, Any] | None = None,
    ) -> SearchTaskDecision:
        text = instruction.strip()
        if not text:
            raise ValueError("instruction must not be empty")
        supplement = _rule_contract(text)
        # A reply such as "簸箕在地图右下角" answers the previous location
        # question; it is not a new request to visually find a dustpan.  This
        # deterministic dialogue gate runs before Qwen so the model cannot
        # replace the original red-object target while filling context.
        location_only_followup = bool(
            pending
            and supplement.get("intent") == "unknown"
            and supplement.get("region_hint")
            and not supplement.get("target_color")
            and not supplement.get("target_class")
        )
        if location_only_followup:
            merged = dict(pending)
            merged["region_hint"] = supplement["region_hint"]
            contract = merged
            source = "deterministic_context_merge"
            warnings = []
        else:
            contract, source, warnings = self.parse(text, semantic_map, pending)

        task_id = "search_" + uuid.uuid4().hex[:10]
        target_class = _fold(contract.get("target_class"))
        target_colour = _fold(contract.get("target_color"))
        anchor_query = str(contract.get("anchor_query") or "").strip()
        region_hint = str(contract.get("region_hint") or "").strip()
        # Short Chinese forms such as "柜子边找..." are easy for a regex to
        # confuse with the target phrase.  If parsing did not return an
        # anchor, recover an explicitly named permanent landmark directly
        # from the semantic map instead of needlessly asking the user again.
        if not anchor_query:
            instruction_folded = _fold(text)
            for landmark_item in semantic_landmarks(semantic_map):
                names = {
                    _fold(landmark_item.get("name")),
                    _fold(landmark_item.get("category")),
                }
                annotation = landmark_item.get("qwen_annotation")
                if isinstance(annotation, dict):
                    names.update(
                        {
                            _fold(annotation.get("name_zh")),
                            _fold(annotation.get("semantic_label")),
                        }
                    )
                explicit = [name for name in names if len(name) >= 2 and name in instruction_folded]
                if explicit:
                    anchor_query = max(explicit, key=len)
                    break
        target = {
            "query": str(contract.get("target_query") or text),
            "color": target_colour,
            "class_name": target_class,
            "generic_object": bool(contract.get("generic_object", target_class == "any yolo object")),
            "requested_count": int(contract.get("requested_count") or 1),
        }
        generic_supported = target_class.replace(" ", "_") == "any_yolo_object" and bool(target_colour)
        if target_class in self.yolo_classes:
            detection_mode = "yolo_primary"
            detector_role = "YOLO geometry followed by deterministic colour/size filtering"
        elif generic_supported:
            detection_mode = "hybrid_generic"
            detector_role = "YOLO/Qwen candidates followed by deterministic colour filtering"
        else:
            detection_mode = "qwen_fallback"
            detector_role = "Qwen target grounding followed by deterministic class/colour evidence checks"
        target["detection_mode"] = detection_mode
        capability = {
            "yolo_class_count": len(self.yolo_classes),
            "required_class": target_class,
            "class_supported": target_class in self.yolo_classes,
            "generic_colour_filter_supported": target_class.replace(" ", "_") == "any_yolo_object" and bool(target_colour),
            "qwen_visual_fallback": True,
            "detection_mode": detection_mode,
            "colour_filter": target_colour or None,
            "detector_role": detector_role,
        }

        if contract.get("environment") == "outside_mapped_area":
            return SearchTaskDecision(
                "robot_object_search_plan.v1", task_id, text, "rejected", "find_object",
                "目标区域位于当前室内永久地图之外，现有小车不能进入马路或未知室外区域。",
                target=target, capabilities=capability, source=source, warnings=warnings,
                model_interpretation=contract,
            )
        if contract.get("intent") not in {"find_object", "navigate"}:
            return SearchTaskDecision(
                "robot_object_search_plan.v1", task_id, text, "rejected", str(contract.get("intent")),
                "该指令不属于当前阶段支持的室内导航或寻物任务。",
                target=target, capabilities=capability, source=source, warnings=warnings,
                model_interpretation=contract,
            )
        if not target_class:
            return SearchTaskDecision(
                "robot_object_search_plan.v1", task_id, text, "needs_clarification", "find_object",
                "没有足够信息确定应由YOLO筛选哪类目标。",
                clarification_question="请说明要找的物品类别；若类别不限，至少说明颜色，例如‘红色物品’。",
                target=target, capabilities=capability, source=source, warnings=warnings,
                model_interpretation=contract,
            )
        if detection_mode == "qwen_fallback":
            warnings.append(
                f"YOLO权重没有‘{target_class}’类别；到达目标区域并停车后改用Qwen视觉定位，"
                "仍需边界框、类别和颜色证据通过确定性校验。"
            )

        landmark = find_landmark(semantic_map, anchor_query)
        if landmark is None and anchor_query and not region_hint:
            return SearchTaskDecision(
                "robot_object_search_plan.v1", task_id, text, "needs_clarification", "find_object",
                f"地图里没有可确认的‘{anchor_query}’标注，无法确定安全搜索区域。",
                clarification_question=f"请补充{anchor_query}在地图中的大致方位，例如‘{anchor_query}在地图右下角’。",
                target=target,
                anchor={"query": anchor_query, "landmark_id": None, "region_hint": ""},
                capabilities=capability, source=source, warnings=warnings,
                model_interpretation=contract,
            )
        if landmark is None and not anchor_query and not region_hint:
            return SearchTaskDecision(
                "robot_object_search_plan.v1", task_id, text, "needs_clarification", "find_object",
                "没有给出参照物或地图区域，整图盲搜代价过高且容易重复绕行。",
                clarification_question="请说明目标大约在哪个已标注物体附近，或在地图的哪个方位。",
                target=target, capabilities=capability, source=source, warnings=warnings,
                model_interpretation=contract,
            )

        if landmark is not None:
            anchor = {
                "query": anchor_query,
                "landmark_id": landmark.get("id"),
                "name": landmark.get("name"),
                "map_xy": landmark.get("map_xy"),
                "region_hint": "",
                "confidence": landmark.get("confidence"),
            }
            route_request = {
                "mode": "around_landmark",
                "anchor_id": landmark.get("id"),
                "anchor_xy_m": landmark.get("map_xy"),
                "view_distance_m": 0.75,
                "maximum_views": 6,
            }
        else:
            anchor = {
                "query": anchor_query,
                "landmark_id": None,
                "name": anchor_query,
                "map_xy": None,
                "region_hint": region_hint,
                "confidence": 0.35,
            }
            route_request = {
                "mode": "coarse_region",
                "region_hint": region_hint,
                "maximum_views": 8,
            }

        steps = [
            {"id": "step_01", "action": "localize", "executor": "raspberry_pi", "success": "已知地图定位通过跨位置一致性验证"},
            {"id": "step_02", "action": "plan_search_route", "executor": "raspberry_pi", "success": "A*生成全部可达观察点；不可达点被剔除"},
            {"id": "step_03", "action": "navigate_to_search_area", "executor": "raspberry_pi", "success": "先到达目标附近首个可达观察点", "safety": ["lidar_clearance", "temporary_obstacles", "short_ttl"]},
            {"id": "step_04", "action": "capture_at_stop", "executor": "raspberry_pi", "success": "到达目标附近并停车稳定后才保存当前帧"},
            {"id": "step_05", "action": "visual_detect_and_filter", "executor": "3090", "success": "所选YOLO或Qwen检测模式的类别、边界框和颜色证据同时满足", "target": target},
            {"id": "step_06", "action": "continue_nearby_search", "executor": "3090+raspberry_pi", "success": "当前视角未找到时继续检查目标附近剩余观察点"},
            {"id": "step_07", "action": "stop_and_report", "executor": "raspberry_pi", "success": "找到目标或附近搜索点耗尽后停车并反馈"},
        ]
        return SearchTaskDecision(
            "robot_object_search_plan.v1", task_id, text, "executable", "find_object",
            "指令位于永久地图内，搜索区域可确定，且当前视觉能力能够进行候选筛选。",
            target=target, anchor=anchor, route_request=route_request, steps=steps,
            capabilities=capability, source=source, warnings=warnings,
            model_interpretation=contract,
        )


class InstructionCoordinator:
    """Command-first state: user work preempts idle annotation immediately."""

    def __init__(self, planner: SearchTaskPlanner, semantic_map: dict[str, Any]) -> None:
        self.planner = planner
        self.semantic_map = semantic_map
        self.lock = threading.Lock()
        self.active: dict[str, Any] | None = None
        self.pending_contract: dict[str, Any] | None = None
        self.idle_job: dict[str, Any] | None = None
        self.annotation_keys: set[str] = set()
        self.generation = 0

    def submit(self, instruction: str) -> dict[str, Any]:
        with self.lock:
            preempted = None
            if self.active:
                preempted = self.active.get("task_id")
            elif self.idle_job:
                preempted = self.idle_job.get("task_id")
            self.active = None
            self.idle_job = None
            self.generation += 1
            decision = self.planner.plan(instruction, self.semantic_map, self.pending_contract)
            payload = decision.to_dict()
            payload["generation"] = self.generation
            payload["preempted_task_id"] = preempted
            if decision.status == "needs_clarification":
                self.pending_contract = dict(decision.model_interpretation)
            else:
                self.pending_contract = None
            if decision.status == "executable":
                self.active = payload
            return payload

    def complete(self, task_id: str, result: str) -> dict[str, Any]:
        with self.lock:
            if not self.active or self.active.get("task_id") != task_id:
                return {"ok": False, "reason": "task_not_active"}
            self.active = None
            return {"ok": True, "task_id": task_id, "result": result}

    def request_idle_annotation(self, dedup_key: str) -> dict[str, Any]:
        with self.lock:
            if self.active:
                return {"accepted": False, "reason": "user_command_active"}
            if dedup_key in self.annotation_keys:
                return {"accepted": False, "reason": "duplicate_annotation"}
            task_id = "idle_" + uuid.uuid4().hex[:10]
            self.idle_job = {"task_id": task_id, "dedup_key": dedup_key, "started_at": time.time()}
            return {"accepted": True, **self.idle_job}

    def finish_idle_annotation(self, task_id: str, committed: bool) -> dict[str, Any]:
        with self.lock:
            if not self.idle_job or self.idle_job.get("task_id") != task_id:
                return {"ok": False, "reason": "idle_task_not_active"}
            if committed:
                self.annotation_keys.add(str(self.idle_job["dedup_key"]))
            self.idle_job = None
            return {"ok": True, "committed": bool(committed)}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "generation": self.generation,
                "active": self.active,
                "pending_clarification": self.pending_contract,
                "idle_annotation": self.idle_job,
                "completed_annotation_keys": len(self.annotation_keys),
                "priority_policy": "new_user_command_preempts_idle_annotation",
            }
