"""Compatibility wrapper that adds task planning to the existing 8010 model.

The original semantic-map FastAPI application is loaded unchanged, so its
existing routes remain available.  A serialised text-generation backend shares
the same Qwen model and lock with landmark inference, avoiding a fifth GPU copy.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import importlib.util
import json
from pathlib import Path
import re
import sys
import threading
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import File, Form, UploadFile
import cv2
import numpy as np


COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
}


def _stereo_order_geometry(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Estimate endpoint order for the robot's inward-toed stereo rig.

    This is deliberately not the conventional rectified-parallel convention.
    Both cameras yaw toward the centre (内八); the operator-confirmed rig has
    positive background-dominant ``x_A - x_B`` when endpoint A is the physical
    right camera.  The mapping below encodes that physical mounting fact.
    """
    try:
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=1800, fastThreshold=10)
        key_left, desc_left = orb.detectAndCompute(gray_left, None)
        key_right, desc_right = orb.detectAndCompute(gray_right, None)
        if desc_left is None or desc_right is None:
            return {"order": "uncertain", "confidence": 0.0, "matches": 0}
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_left, desc_right)
        good = []
        height = max(left.shape[0], right.shape[0])
        for match in sorted(pairs, key=lambda item: item.distance)[:500]:
            a = key_left[match.queryIdx].pt
            b = key_right[match.trainIdx].pt
            if match.distance <= 64 and abs(a[1] - b[1]) <= max(18.0, height * 0.08):
                good.append(a[0] - b[0])
        if len(good) < 18:
            return {"order": "uncertain", "confidence": 0.0, "matches": len(good)}
        median_dx = float(np.median(np.asarray(good, dtype=np.float32)))
        disparity_spread = float(np.median(np.abs(np.asarray(good) - median_dx)))
        coherence = abs(median_dx) / (abs(median_dx) + disparity_spread + 1e-6)
        confidence = min(
            0.94,
            max(0.0, len(good) / 90.0)
            * max(0.0, min(1.0, abs(median_dx) / 18.0))
            * (0.55 + 0.45 * coherence),
        )
        if abs(median_dx) < 2.5:
            order = "uncertain"
            confidence *= 0.35
        else:
            # Operator-validated inward-toed rig convention.  Do not replace
            # this with the sign rule for a parallel/rectified stereo pair.
            order = "swapped" if median_dx > 0 else "normal"
        return {
            "order": order,
            "confidence": round(confidence, 3),
            "matches": len(good),
            "median_disparity_px": round(median_dx, 2),
            "median_disparity_spread_px": round(disparity_spread, 2),
            "rig": "inward_toed_operator_validated",
        }
    except Exception as exc:
        return {"order": "uncertain", "confidence": 0.0, "matches": 0, "error": str(exc)}


def _load_existing(path: Path):
    spec = importlib.util.spec_from_file_location("final_work_existing_qwen_landmarks", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载现有 Qwen 服务：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SharedLandmarker:
    def __init__(self, inner: Any, lock: threading.RLock) -> None:
        self.inner = inner
        self.lock = lock

    def detect(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            return self.inner.detect(*args, **kwargs)


class SharedTextBackend:
    def __init__(self, shared: SharedLandmarker, max_new_tokens: int = 320) -> None:
        self.shared = shared
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        with self.shared.lock:
            landmarker = self.shared.inner
            if landmarker.mock:
                raise RuntimeError("mock Qwen hub cannot generate task intent")
            landmarker._load()
            model = landmarker.model
            processor = landmarker.processor
            torch = landmarker.torch
            if model is None or processor is None or torch is None:
                raise RuntimeError("Qwen model is not initialized")
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    max_time=60.0,
                    do_sample=False,
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            return processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]


class SharedAgentPlanner:
    """Qwen intent extraction plus deterministic, UTF-8-safe plan compilation."""

    CONSTRAINTS = [
        "底盘运动时必须同时检查雷达净空与相机动态障碍物",
        "定位或感知失效时立即停止，不得盲目前进",
        "机械臂动作前底盘必须停止，目标必须位于可达工作空间",
        "抓取和放置后必须通过视觉复核",
    ]

    def __init__(self, backend: SharedTextBackend) -> None:
        self.backend = backend

    @staticmethod
    def _extract(raw: str) -> dict[str, Any]:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
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
        raise ValueError("Qwen 输出中没有 JSON 对象")

    @staticmethod
    def _map_summary(semantic_map: dict[str, Any]) -> dict[str, Any]:
        items = semantic_map.get("landmarks") or semantic_map.get("objects") or []
        compact = []
        for item in items[:50] if isinstance(items, list) else []:
            if isinstance(item, dict):
                compact.append({
                    "id": item.get("id"),
                    "name": item.get("name") or item.get("label"),
                    "map_xy": item.get("map_xy") or item.get("xy"),
                    "confidence": item.get("confidence") or item.get("stability"),
                })
        return {"landmarks": compact, "search_regions": semantic_map.get("search_regions", [])[:12]}

    @staticmethod
    def _direct_motion_contract(instruction: str) -> dict[str, Any] | None:
        text = instruction.strip().lower()
        patterns = (
            ("forward", r"(?:向前|前进)(?:行动|移动|走)?\s*([0-9]+(?:\.[0-9]+)?)?\s*(米|m|厘米|cm)?"),
            ("backward", r"(?:向后|后退)(?:行动|移动|走)?\s*([0-9]+(?:\.[0-9]+)?)?\s*(米|m|厘米|cm)?"),
            ("left", r"(?:向左转|左转)\s*([0-9]+(?:\.[0-9]+)?)?\s*(?:度|°|deg)?"),
            ("right", r"(?:向右转|右转)\s*([0-9]+(?:\.[0-9]+)?)?\s*(?:度|°|deg)?"),
        )
        for direction, pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            value = float(match.group(1) or (30.0 if direction in {"left", "right"} else 0.30))
            if direction in {"forward", "backward"}:
                unit = (match.group(2) or "米").lower()
                if unit in {"厘米", "cm"}:
                    value /= 100.0
                if not 0.05 <= value <= 1.50:
                    raise ValueError("单次直行距离必须在 0.05..1.50m")
                return {"intent": "drive", "direction": direction, "distance_m": value, "angle_deg": 0.0, "confidence": 1.0}
            if not 5.0 <= value <= 180.0:
                raise ValueError("单次转向角度必须在 5..180°")
            return {"intent": "drive", "direction": direction, "distance_m": 0.0, "angle_deg": value, "confidence": 1.0}
        return None

    def _prompt(self, instruction: str, semantic_map: dict[str, Any]) -> str:
        return f"""你是移动机械臂小车的高层任务理解器。只解析用户意图和实体，不输出电机速度、舵机角度、shell 命令或思维过程。

用户指令：{instruction}
已知语义地图摘要：{json.dumps(self._map_summary(semantic_map), ensure_ascii=False)}

intent 只能是 inspect、navigate、search、pick、pick_and_place、drive、unsupported：
- inspect：观察某个位置或物体并汇报；
- navigate：前往某位置；
- search：寻找并靠近某物，但不抓取；
- pick：寻找并抓取某物，但没有要求搬运到另一位置；
- pick_and_place：寻找、抓取并放到描述位置；
- drive：明确要求小车前进、后退、左转或右转；
- unsupported：超出移动、观察、寻找、抓取和放置能力。

严格只返回一个 JSON 对象，字段为：
{{"intent":"...","target_query":"...","search_near_query":"...","destination_query":"...","direction":"forward|backward|left|right","distance_m":0.0,"angle_deg":0.0,"target_landmark_id":null,"search_near_landmark_id":null,"destination_landmark_id":null,"confidence":0.0}}
target_query 是要识别/抓取的物体，search_near_query 是“在某物旁边/附近”所描述的搜索锚点，destination_query 只用于抓取后的放置位置，三者不得互换。
例如“在黄色柜子旁边找到并夹取一个绿色小木块”必须解析为 intent=pick、target_query=绿色小木块、search_near_query=黄色柜子、destination_query=""。
字段必须直接保留用户所说的中文，不得乱码；地图没有对应 id 时必须为 null，禁止编造坐标和 id。"""

    @staticmethod
    def _step(index: int, action: str, description: str, target: dict[str, Any], depends: list[str], sensors: list[str]) -> dict[str, Any]:
        return {
            "id": f"step_{index:02d}", "action": action,
            "description": description, "depends_on": depends,
            "target": target, "sensors": sensors,
            "success_condition": "动作由对应适配器确认完成",
            "on_failure": "safe_stop" if action not in {"inspect_semantic_map", "report_result"} else "report_failure",
            "max_retries": 1, "condition": "always",
        }

    def _compile(self, instruction: str, contract: dict[str, Any], source: str, warning: str = "") -> dict[str, Any]:
        intent = str(contract.get("intent") or "unsupported").lower()
        if intent not in {"inspect", "navigate", "search", "pick", "pick_and_place", "drive", "unsupported"}:
            intent = "unsupported"
        target_query = str(contract.get("target_query") or "").strip() or instruction
        destination_query = str(contract.get("destination_query") or "").strip()
        search_near_query = str(contract.get("search_near_query") or "").strip()
        target = {"query": target_query, "landmark_id": contract.get("target_landmark_id") or None}
        destination = {"query": destination_query, "landmark_id": contract.get("destination_landmark_id") or None}
        search_anchor = {
            "query": search_near_query or target_query,
            "landmark_id": contract.get("search_near_landmark_id") or None,
            "target_query": target_query,
        }
        specs: list[tuple[str, str, dict[str, Any], list[str]]] = [
            ("inspect_semantic_map", "读取语义地图并核对目标与可通行区域", target, ["semantic_map"]),
        ]
        if intent == "drive":
            direction = str(contract.get("direction") or "")
            motion_target = {
                "direction": direction,
                "distance_m": float(contract.get("distance_m") or 0.0),
                "angle_deg": float(contract.get("angle_deg") or 0.0),
            }
            labels = {"forward": "前进", "backward": "后退", "left": "左转", "right": "右转"}
            amount = f"{motion_target['distance_m']:.2f}m" if direction in {"forward", "backward"} else f"{motion_target['angle_deg']:.0f}°"
            specs = [
                ("drive_relative", f"雷达保护下{labels.get(direction, '移动')} {amount}", motion_target, ["lidar"]),
                ("report_result", "汇报底盘运动结果并保持停车", motion_target, ["lidar"]),
            ]
        elif intent == "inspect":
            specs += [
                ("patrol_search", "选择安全观测点并获取目标区域视觉", target, ["lidar", "camera", "semantic_map"]),
                ("report_result", "汇报视觉观察与地图标注结果", target, ["camera", "semantic_map"]),
            ]
        elif intent == "navigate":
            specs += [
                ("navigate_to", "导航到文本描述位置附近并安全停车", target, ["lidar", "camera", "semantic_map"]),
                ("report_result", "汇报到达状态", target, ["camera", "lidar"]),
            ]
        elif intent == "search":
            specs += [
                ("patrol_search", f"由 AI 在安全候选中确认观察顺序并搜索{('“' + search_near_query + '”附近的') if search_near_query else ''}目标", search_anchor, ["lidar", "camera", "semantic_map"]),
                ("locate_object", "使用 Qwen 初次定位并由 YOLO 持续追踪目标", target, ["camera"]),
                ("align_for_grasp", "移动小车直到目标进入双目示教区域", target, ["camera", "lidar"]),
                ("report_result", "汇报目标位置和接近结果", target, ["camera", "lidar"]),
            ]
        elif intent == "pick":
            specs += [
                ("patrol_search", f"由 AI 在安全候选中确认观察顺序并搜索{('“' + search_near_query + '”附近的') if search_near_query else ''}目标", search_anchor, ["lidar", "camera", "semantic_map"]),
                ("locate_object", "使用 Qwen 定位指定目标并持续保持身份", target, ["camera"]),
                ("align_for_grasp", "视觉追踪靠近并对准抓取区域", target, ["camera", "lidar"]),
                ("grasp", "机械臂执行视觉伺服抓取", target, ["camera"]),
                ("verify_grasp", "通过双目视觉确认抓取成功", target, ["camera"]),
                ("report_result", "汇报抓取结果并保持停止", target, ["camera", "lidar"]),
            ]
        elif intent == "pick_and_place":
            specs += [
                ("patrol_search", "在安全观测点搜索待抓取物", target, ["lidar", "camera", "semantic_map"]),
                ("navigate_to", "导航到目标物附近", target, ["lidar", "camera", "semantic_map"]),
                ("locate_object", "使用 Qwen 定位目标并持续保持身份", target, ["camera"]),
                ("align_for_grasp", "视觉追踪靠近并对准抓取区域", target, ["camera", "lidar"]),
                ("grasp", "机械臂执行视觉伺服抓取", target, ["camera"]),
                ("verify_grasp", "通过双目视觉确认抓取成功", target, ["camera"]),
                ("locate_destination", "在语义地图中定位放置位置", destination, ["semantic_map", "camera"]),
                ("navigate_to", "携带物体导航到放置位置附近", destination, ["lidar", "camera", "semantic_map"]),
                ("release", "机械臂在安全位置释放物体", destination, ["camera"]),
                ("verify_completion", "通过视觉确认物体已放到目标位置", destination, ["camera"]),
                ("report_result", "汇报搬运结果并保持停止", destination, ["camera", "lidar"]),
            ]
        else:
            specs = [("report_result", "报告当前系统无法安全执行该指令", {}, ["camera", "lidar"])]
        steps = []
        for index, (action, description, step_target, sensors) in enumerate(specs, 1):
            depends = [] if index == 1 else [f"step_{index - 1:02d}"]
            steps.append(self._step(index, action, description, step_target, depends, sensors))
        summaries = {
            "inspect": f"观察“{target_query}”并将可靠特征关联到地图。",
            "navigate": f"安全前往“{target_query}”附近。",
            "search": f"搜索“{target_query}”并通过双目视觉持续追踪靠近。",
            "pick": f"在“{search_near_query or '指定区域'}”附近寻找并抓取“{target_query}”。",
            "pick_and_place": f"寻找并抓取“{target_query}”，然后放到“{destination_query or '指定位置'}”。",
            "drive": "执行一次有距离/角度上限并受雷达保护的底盘运动。",
            "unsupported": "当前能力无法安全执行这条指令。",
        }
        return {
            "schema": "robot_task_plan.v1", "task_id": f"final_work_{intent}",
            "instruction": instruction, "intent": intent, "summary": summaries[intent],
            "assumptions": [], "global_constraints": list(self.CONSTRAINTS),
            "steps": steps, "source": source,
            "warnings": [warning] if warning else [], "model_interpretation": contract,
        }

    def plan(self, instruction: str, semantic_map: dict[str, Any]) -> dict[str, Any]:
        direct = self._direct_motion_contract(instruction)
        if direct is not None:
            return self._compile(instruction, direct, "deterministic_motion_parser")
        try:
            raw = self.backend.generate(self._prompt(instruction, semantic_map))
            contract = self._extract(raw)
            return self._compile(instruction, contract, "qwen_compiled")
        except Exception as exc:
            folded = instruction.lower()
            if any(word in folded for word in ("放到", "放在", "搬到", "拿到")):
                intent = "pick_and_place"
            elif any(word in folded for word in ("夹取", "抓取", "拿起", "捡起")):
                intent = "pick"
            elif any(word in folded for word in ("找", "寻找", "靠近", "追踪")):
                intent = "search"
            elif any(word in folded for word in ("去", "前往", "移动到", "到达")):
                intent = "navigate"
            else:
                intent = "inspect"
            return self._compile(
                instruction,
                {"intent": intent, "target_query": instruction, "confidence": 0.0},
                "rules_fallback",
                f"Qwen 意图解析失败，已安全降级：{type(exc).__name__}: {exc}",
            )


def build_app(args: argparse.Namespace):
    existing_path = Path(args.existing_script).expanduser()
    existing = _load_existing(existing_path)
    source_root = Path(args.source_root).expanduser()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from qwen_grounded_tracker.grounding.qwen_grounder import Qwen3VLTargetGrounder
    from qwen_grounded_tracker.domain import GroundingResult
    from qwen_grounded_tracker.search_task_planner import InstructionCoordinator, SearchTaskPlanner

    lock = threading.RLock()
    inner = existing.QwenLandmarker(
        model_path=str(Path(args.model_path).expanduser()),
        device=args.device,
        local_files_only=True,
        mock=args.mock,
    )
    shared = SharedLandmarker(inner, lock)
    existing.LANDMARKER = shared
    existing.PI_MAP_URL = args.pi_url.strip().rstrip("/")
    text_backend = SharedTextBackend(shared)
    planner = SharedAgentPlanner(text_backend)
    search_coordinator = InstructionCoordinator(SearchTaskPlanner(COCO_CLASSES, text_backend), {})
    search_request_lock = threading.Lock()
    grounder = Qwen3VLTargetGrounder(
        model_path=inner.model_path,
        device=args.device,
        local_files_only=True,
        max_new_tokens=160,
        max_time_seconds=30.0,
        min_visual_tokens=128,
        max_visual_tokens=512,
        minimum_box_area_ratio=0.0002,
    )

    def shared_ground(image: np.ndarray, instruction: str, reference: np.ndarray | None):
        with lock:
            if inner.mock:
                return GroundingResult(False, None, "unknown", 0.0, message="mock")
            inner._load()
            grounder.model = inner.model
            grounder.processor = inner.processor
            grounder.torch = inner.torch
            return grounder.ground(image, instruction, reference_image=reference)

    @existing.app.get("/agent/health")
    async def agent_health() -> JSONResponse:
        return JSONResponse({
            "ok": True,
            "service": "shared_qwen_landmark_and_agent",
            "model_path": inner.model_path,
            "model_loaded": inner.model is not None,
            "mock": inner.mock,
            "map_navigation_contract": True,
        })

    @existing.app.post("/agent/plan")
    async def agent_plan(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            instruction = str(payload.get("instruction") or "").strip()
            if not instruction:
                raise ValueError("instruction 不能为空")
            semantic_map = payload.get("semantic_map")
            if not isinstance(semantic_map, dict):
                semantic_map = {}
            result = await asyncio.to_thread(planner.plan, instruction, semantic_map)
            return JSONResponse({"ok": True, "plan": result})
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    def infer_camera_order(left_value: np.ndarray, right_value: np.ndarray) -> dict[str, Any]:
        # Pillow is needed only for the multimodal request. Keeping it lazy
        # lets geometry diagnostics run without loading the full vision stack.
        from PIL import Image

        geometry = _stereo_order_geometry(left_value, right_value)
        with lock:
            if inner.mock:
                return {"order": geometry.get("order", "uncertain"), "confidence": geometry.get("confidence", 0.0), "source": "geometry_mock", "geometry": geometry}
            inner._load()
            if inner.model is None or inner.processor is None or inner.torch is None:
                raise RuntimeError("Qwen model is not initialized")
            left_rgb = Image.fromarray(cv2.cvtColor(left_value, cv2.COLOR_BGR2RGB))
            right_rgb = Image.fromarray(cv2.cvtColor(right_value, cv2.COLOR_BGR2RGB))
            prompt = f"""你正在校准一对朝向车辆前方、呈内八字略微向中间汇聚的双目摄像头。
图像 A 来自软件当前标记的 left 端点，图像 B 来自软件当前标记的 right 端点。
请比较两图中相同近处物体的水平视差、遮挡边界和视角。车辆后方向前看时，物理左摄像头才应标记为 left。
注意：这不是平行或已经极线校正的双目，禁止套用“正视差=A为左”的常规规则；视差符号会受汇聚角和物距影响。
辅助几何结果已经使用该小车经操作员确认的内八字安装约定完成 normal/swapped 映射。
确定当前端点是 normal（A=物理左、B=物理右）、swapped（A=物理右、B=物理左）还是 uncertain。
几何匹配的辅助证据为：{json.dumps(geometry, ensure_ascii=False)}。
辅助证据可能受运动物体或低纹理影响，必须同时检查图像；证据冲突时返回 uncertain。
严格只输出 JSON：{{"order":"normal|swapped|uncertain","confidence":0.0,"reason":"简短依据"}}"""
            messages = [{"role": "user", "content": [
                {"type": "image", "image": left_rgb},
                {"type": "text", "text": "图像 A：当前软件 left 端点"},
                {"type": "image", "image": right_rgb},
                {"type": "text", "text": "图像 B：当前软件 right 端点"},
                {"type": "text", "text": prompt},
            ]}]
            inputs = inner.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(inner.model.device)
            with inner.torch.inference_mode():
                output_ids = inner.model.generate(
                    **inputs, max_new_tokens=140, max_time=45.0, do_sample=False,
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            raw = inner.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]
        parsed = SharedAgentPlanner._extract(raw)
        ai_order = str(parsed.get("order") or "uncertain").strip().lower()
        if ai_order not in {"normal", "swapped", "uncertain"}:
            ai_order = "uncertain"
        try:
            ai_confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            ai_confidence = 0.0
        geometry_order = str(geometry.get("order") or "uncertain")
        geometry_confidence = float(geometry.get("confidence") or 0.0)
        if ai_order in {"normal", "swapped"} and ai_confidence >= 0.55:
            if geometry_order == "uncertain" or geometry_order == ai_order:
                order = ai_order
                confidence = max(ai_confidence, geometry_confidence)
                source = "qwen_geometry_consensus" if geometry_order == ai_order else "qwen_visual"
            elif geometry_confidence >= 0.70:
                order, confidence, source = "uncertain", min(ai_confidence, geometry_confidence), "evidence_conflict"
            else:
                order, confidence, source = ai_order, ai_confidence * 0.8, "qwen_visual_geometry_weak"
        elif geometry_order in {"normal", "swapped"} and geometry_confidence >= 0.72:
            order, confidence, source = geometry_order, geometry_confidence * 0.85, "geometry_ai_uncertain"
        else:
            order, confidence, source = "uncertain", max(ai_confidence, geometry_confidence), "insufficient_evidence"
        return {
            "order": order,
            "confidence": round(float(confidence), 3),
            "source": source,
            "reason": str(parsed.get("reason") or "")[:300],
            "geometry": geometry,
            "raw_text": raw[:1000],
        }

    @existing.app.post("/agent/camera-order")
    async def agent_camera_order(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            images = []
            for key in ("left_image_b64", "right_image_b64"):
                raw = base64.b64decode(str(payload.get(key) or ""), validate=True)
                if len(raw) < 1000 or len(raw) > 8 * 1024 * 1024:
                    raise ValueError("摄像头图像大小无效")
                value = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if value is None:
                    raise ValueError("摄像头图像解码失败")
                images.append(value)
            result = await asyncio.to_thread(infer_camera_order, images[0], images[1])
            return JSONResponse({"ok": True, **result})
        except (ValueError, binascii.Error) as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @existing.app.post("/mission/command")
    async def mission_command(request: Request) -> JSONResponse:
        """Compatibility route for the frozen car1.0 natural-search client.

        Semantic data is supplied by the Pi adapter for every request, so a
        newly published read-only release is used without restarting Qwen.
        """
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            instruction = str(payload.get("instruction") or "").strip()
            if not instruction:
                raise ValueError("instruction 不能为空")
            semantic_map = payload.get("semantic_map")
            if not isinstance(semantic_map, dict):
                semantic_map = {}

            def submit() -> dict[str, Any]:
                with search_request_lock:
                    search_coordinator.semantic_map = semantic_map
                    return search_coordinator.submit(instruction)

            decision = await asyncio.to_thread(submit)
            return JSONResponse({"ok": True, "decision": decision})
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @existing.app.post("/mission/select-observation")
    async def mission_select_observation(request: Request) -> JSONResponse:
        """Rank prevalidated observation candidates without inventing coordinates."""
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            candidates = payload.get("candidates")
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
                raise ValueError("candidates 必须包含 1 到 16 个安全候选点")
            safe_candidates = []
            for expected, item in enumerate(candidates):
                if not isinstance(item, dict) or int(item.get("index", -1)) != expected:
                    raise ValueError("候选点索引无效")
                xy = item.get("view_xy_m")
                if not isinstance(xy, list) or len(xy) != 2:
                    raise ValueError("候选点坐标无效")
                safe_candidates.append({
                    "index": expected,
                    "view_xy_m": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
                    "path_length_from_start_m": round(float(item.get("path_length_from_start_m") or 0.0), 3),
                    "distance_to_anchor_m": round(float(item.get("distance_to_anchor_m") or 0.0), 3),
                    "look_at_yaw_deg": round(float(item.get("look_at_yaw_deg") or 0.0), 1),
                })
            target_context = payload.get("target") if isinstance(payload.get("target"), dict) else {}
            anchor_context = payload.get("anchor") if isinstance(payload.get("anchor"), dict) else {}
            semantic_context = payload.get("semantic_context") if isinstance(payload.get("semantic_context"), dict) else {}
            prompt = f"""你负责为双目移动机器人确认观察点。低层程序已经通过占据地图净空、目标可见射线和 A* 可达性生成候选；你只能排序候选编号，禁止修改或生成坐标。

任务：{str(payload.get('instruction') or '')[:500]}
视觉目标：{json.dumps(target_context, ensure_ascii=False)}
语义锚点：{json.dumps(anchor_context, ensure_ascii=False)}
相关场景语义：{json.dumps(semantic_context, ensure_ascii=False)}
已通过安全校验的候选：{json.dumps(safe_candidates, ensure_ascii=False)}

请优先选择更可能直接看到目标所在区域、遮挡更少、观察方向符合场景关系且路径不过度绕行的位置；不要单纯选择最近点。其余点作为失败后的复查顺序。
confidence 必须是你对首选观察点优于其他候选的真实置信度：证据充分可为 0.70~0.95，差异不明显应低于 0.45；禁止保留占位值 0.0。
严格只输出 JSON：{{"ranked_indices":[候选编号],"confidence":0.73,"reason":"结合具体编号、观察方向和路径的依据","candidate_reasons":{{"候选编号":"该点的具体优缺点"}}}}。ranked_indices 只能包含上述编号，每个编号最多一次；示例数字不可照抄。"""
            raw = await asyncio.to_thread(text_backend.generate, prompt)
            parsed = SharedAgentPlanner._extract(raw)
            ranked = []
            for value in parsed.get("ranked_indices") or []:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(safe_candidates) and index not in ranked:
                    ranked.append(index)
            if not ranked:
                raise ValueError("Qwen 未返回有效候选编号")
            ranked.extend(index for index in range(len(safe_candidates)) if index not in ranked)
            try:
                confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            return JSONResponse({
                "ok": True,
                "ranked_indices": ranked,
                "confidence": confidence,
                "confirmed": confidence >= 0.45,
                "reason": str(parsed.get("reason") or "")[:500],
                "candidate_reasons": parsed.get("candidate_reasons") if isinstance(parsed.get("candidate_reasons"), dict) else {},
                "source": "shared_qwen_safe_candidate_ranking",
                "raw_text": raw[:1200],
            })
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @existing.app.get("/mission/state")
    async def mission_state() -> JSONResponse:
        return JSONResponse({"ok": True, "state": search_coordinator.snapshot()})

    @existing.app.post("/mission/task/complete")
    async def mission_complete(request: Request) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            return JSONResponse({"ok": False, "error": "请求必须是 JSON 对象"}, status_code=400)
        result = search_coordinator.complete(
            str(payload.get("task_id") or ""),
            str(payload.get("result") or "completed"),
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 409)

    @existing.app.post("/agent/ground")
    async def agent_ground(
        image: UploadFile = File(...),
        instruction: str = Form(...),
        reference: UploadFile | None = File(None),
    ) -> JSONResponse:
        try:
            image_value = cv2.imdecode(
                np.frombuffer(await image.read(), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            reference_value = None
            if reference is not None:
                reference_value = cv2.imdecode(
                    np.frombuffer(await reference.read(), dtype=np.uint8), cv2.IMREAD_COLOR
                )
            if image_value is None:
                raise ValueError("图像解码失败")
            result = await asyncio.to_thread(
                shared_ground, image_value, instruction.strip(), reference_value
            )
            return JSONResponse({
                "ok": True,
                "found": result.found,
                "bbox": None if result.bbox is None else [
                    result.bbox.x1, result.bbox.y1, result.bbox.x2, result.bbox.y2
                ],
                "target_name": result.target_name,
                "confidence": result.confidence,
                "raw_text": result.raw_text,
                "message": result.message,
                "candidates": [
                    {
                        "target_name": candidate.target_name,
                        "bbox": [candidate.bbox.x1, candidate.bbox.y1, candidate.bbox.x2, candidate.bbox.y2],
                        "confidence": candidate.confidence,
                        "camera_view": candidate.camera_view,
                    }
                    for candidate in result.candidates
                ],
            })
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return existing.app


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared Qwen semantic-map and Agent hub")
    parser.add_argument("--existing-script", default="/home/sjtu/3090/scripts/run_qwen_landmark_server.py")
    parser.add_argument("--source-root", default="/home/sjtu/3090/src")
    parser.add_argument("--model-path", default="/home/sjtu/3090/models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pi-url", default="http://172.18.14.74:8766")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    app = build_app(args)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
