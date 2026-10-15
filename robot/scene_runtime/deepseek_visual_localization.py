#!/usr/bin/env python3
"""DeepSeek Vision ranking for lidar pose candidates.

The model is never allowed to invent metric coordinates.  Lidar supplies a
small set of candidate poses, deterministic code describes the reviewed map
geometry expected from each candidate, and DeepSeek may only select a
candidate id or abstain.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import math
import mimetypes
import os
from pathlib import Path
import re
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from PIL import Image


PLACEHOLDER_MARKERS = ("请填写", "your_", "replace_", "changeme", "sk-xxx")


@dataclass(frozen=True)
class DeepSeekVisionConfig:
    enabled: bool
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_s: float
    minimum_confidence: float
    minimum_margin: float
    repeat_count: int
    maximum_candidates: int
    maximum_landmark_distance_m: float
    camera_horizontal_fov_deg: float
    camera_bearing_margin_deg: float
    landmark_categories: tuple[str, ...]
    stable_anchor_categories: tuple[str, ...]
    context_only_categories: tuple[str, ...]
    semantic_map: Path
    occupancy_map: Path
    occupancy_yaml: Path
    source_path: Path

    @property
    def configured(self) -> bool:
        key = self.api_key.strip().lower()
        return bool(
            self.enabled
            and key
            and not any(marker in key for marker in PLACEHOLDER_MARKERS)
        )


@dataclass(frozen=True)
class VisionRankResult:
    selected_candidate_id: str
    confidence: float
    second_candidate_id: str
    second_confidence: float
    observed_structures: tuple[dict[str, Any], ...]
    evidence: str
    uncertain_reason: str
    consistent_repeats: int

    @property
    def margin(self) -> float:
        return max(0.0, self.confidence - self.second_confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected_candidate_id,
            "confidence": round(self.confidence, 4),
            "second_candidate_id": self.second_candidate_id,
            "second_confidence": round(self.second_confidence, 4),
            "margin": round(self.margin, 4),
            "observed_structures": list(self.observed_structures),
            "evidence": self.evidence,
            "uncertain_reason": self.uncertain_reason,
            "consistent_repeats": self.consistent_repeats,
        }


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _resolve_runtime_path(value: Any, module_directory: Path) -> Path:
    path = Path(str(value or "semantic_runtime.json"))
    return path if path.is_absolute() else module_directory / path


def _tuple_setting(payload: dict[str, Any], name: str, default: list[str]) -> tuple[str, ...]:
    value = payload.get(name) or default
    return tuple(str(item) for item in value)


def load_config(path: str | Path, *, module_directory: str | Path) -> DeepSeekVisionConfig:
    source = Path(path)
    payload: dict[str, Any] = {}
    if source.is_file():
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (OSError, json.JSONDecodeError):
            payload = {}
    categories = payload.get("landmark_categories") or [
        "glass_wall", "stairs", "door", "window", "cabinet", "sign", "table"
    ]
    return DeepSeekVisionConfig(
        enabled=bool(payload.get("enabled", True)),
        api_key=str(payload.get("api_key") or ""),
        base_url=str(payload.get("base_url") or "https://api.deepseek.com").rstrip("/"),
        model=str(payload.get("model") or "deepseek-flash"),
        timeout_s=max(10.0, min(180.0, float(payload.get("timeout_s") or 90.0))),
        minimum_confidence=max(0.50, min(0.99, float(payload.get("minimum_confidence") or 0.72))),
        minimum_margin=max(0.05, min(0.80, float(payload.get("minimum_margin") or 0.20))),
        repeat_count=max(1, min(3, int(payload.get("repeat_count") or 2))),
        maximum_candidates=max(2, min(8, int(payload.get("maximum_candidates") or 6))),
        maximum_landmark_distance_m=max(
            2.0, min(8.0, float(payload.get("maximum_landmark_distance_m") or 5.5))
        ),
        camera_horizontal_fov_deg=max(
            45.0, min(150.0, float(payload.get("camera_horizontal_fov_deg") or 82.0))
        ),
        camera_bearing_margin_deg=max(
            0.0, min(30.0, float(payload.get("camera_bearing_margin_deg") or 8.0))
        ),
        landmark_categories=tuple(str(item) for item in categories),
        stable_anchor_categories=_tuple_setting(
            payload,
            "stable_anchor_categories",
            ["glass_wall", "stairs", "door", "window"],
        ),
        context_only_categories=_tuple_setting(
            payload,
            "context_only_categories",
            ["table", "chair", "desk"],
        ),
        semantic_map=_resolve_runtime_path(payload.get("semantic_map"), Path(module_directory)),
        occupancy_map=_resolve_runtime_path(
            payload.get("occupancy_map") or "ble_scene_17.pgm", Path(module_directory)
        ),
        occupancy_yaml=_resolve_runtime_path(
            payload.get("occupancy_yaml") or "ble_scene_17.yaml", Path(module_directory)
        ),
        source_path=source,
    )


def _normalise_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _occupancy_geometry(config: DeepSeekVisionConfig):
    """Load the immutable ROS map for visibility diagnostics, if available."""
    if not config.occupancy_map.is_file() or not config.occupancy_yaml.is_file():
        return None
    try:
        yaml_text = config.occupancy_yaml.read_text(encoding="utf-8")
        resolution_match = re.search(r"^resolution:\s*([-+0-9.eE]+)", yaml_text, re.MULTILINE)
        origin_match = re.search(
            r"^origin:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)",
            yaml_text,
            re.MULTILINE,
        )
        if resolution_match is None or origin_match is None:
            return None
        image = Image.open(config.occupancy_map).convert("L")
        return (
            image,
            float(resolution_match.group(1)),
            float(origin_match.group(1)),
            float(origin_match.group(2)),
        )
    except (OSError, ValueError):
        return None


def _map_occlusion(
    geometry,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> str:
    """Classify a sight ray without treating its occupied endpoint as a wall."""
    if geometry is None:
        return "unknown"
    image, resolution, origin_x, origin_y = geometry
    dx, dy = end_xy[0] - start_xy[0], end_xy[1] - start_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= 0.30:
        return "clear"
    steps = max(2, int(math.ceil(distance / max(0.02, resolution))))
    end_margin = max(2, int(math.ceil(0.22 / resolution)))
    consecutive_occupied = 0
    for index in range(2, max(2, steps - end_margin)):
        fraction = index / steps
        x = start_xy[0] + fraction * dx
        y = start_xy[1] + fraction * dy
        column = int(math.floor((x - origin_x) / resolution))
        world_row = int(math.floor((y - origin_y) / resolution))
        row = image.height - 1 - world_row
        if not (0 <= column < image.width and 0 <= row < image.height):
            return "unknown"
        if image.getpixel((column, row)) <= 65:
            consecutive_occupied += 1
            if consecutive_occupied >= 2:
                return "blocked"
        else:
            consecutive_occupied = 0
    return "clear"


def candidate_scene_descriptions(
    candidates: list[Any],
    semantic_map: dict[str, Any],
    config: DeepSeekVisionConfig,
) -> list[dict[str, Any]]:
    """Describe expected fixed structures without exposing writable poses to the model."""
    objects = [
        item for item in (semantic_map.get("objects") or [])
        if isinstance(item, dict)
        and str(item.get("category") or "") in config.landmark_categories
        and isinstance(item.get("map_xy"), list)
        and len(item["map_xy"]) >= 2
        and item.get("semantic_reviewed") is True
    ]
    object_by_id = {
        str(item.get("id")): item for item in objects if item.get("id")
    }
    transparent_subjects = {
        str(relation.get("subject_id"))
        for relation in (semantic_map.get("scene_relations") or [])
        if isinstance(relation, dict)
        and str(relation.get("relation")) == "behind"
        and str((object_by_id.get(str(relation.get("object_id"))) or {}).get("category"))
        == "glass_wall"
    }
    geometry = _occupancy_geometry(config)
    descriptions = []
    for index, candidate in enumerate(candidates, start=1):
        pose = candidate.pose if hasattr(candidate, "pose") else candidate
        structures: list[dict[str, Any]] = []
        context_only: list[dict[str, Any]] = []
        for item in objects:
            dx = float(item["map_xy"][0]) - float(pose.x_m)
            dy = float(item["map_xy"][1]) - float(pose.y_m)
            distance = math.hypot(dx, dy)
            relative = math.degrees(_normalise_angle(math.atan2(dy, dx) - pose.yaw_rad))
            uncertainty = max(0.0, float(item.get("position_uncertainty_m") or 0.0))
            category = str(item.get("category") or "")
            if distance > config.maximum_landmark_distance_m + uncertainty:
                continue
            bearing_uncertainty = min(
                28.0,
                math.degrees(math.atan2(uncertainty, max(0.20, distance))),
            )
            fov_limit = (
                config.camera_horizontal_fov_deg / 2.0
                + config.camera_bearing_margin_deg
                + bearing_uncertainty
            )
            if abs(relative) > fov_limit:
                continue
            side = "center" if abs(relative) <= 18.0 else ("left" if relative > 0 else "right")
            occlusion = _map_occlusion(
                geometry,
                (float(pose.x_m), float(pose.y_m)),
                (float(item["map_xy"][0]), float(item["map_xy"][1])),
            )
            through_glass = str(item.get("id") or "") in transparent_subjects
            evidence_role = (
                "context_only"
                if category in config.context_only_categories
                else (
                    "stable_anchor"
                    if category in config.stable_anchor_categories
                    else "supporting_structure"
                )
            )
            row = {
                "name": str(item.get("name") or item.get("category") or "structure"),
                "category": category,
                "side": side,
                "relative_bearing_deg": round(relative, 1),
                "bearing_uncertainty_deg": round(bearing_uncertainty, 1),
                "distance_m": round(distance, 2),
                "map_uncertainty_m": round(uncertainty, 2),
                "evidence_role": evidence_role,
                "map_ray": occlusion,
                "visible_through_glass_relation": through_glass,
                # The current cameras are low and tilted toward the floor.
                # Absence in a frame is not reliable without 3-D extents.
                "absence_is_negative": False,
            }
            (context_only if evidence_role == "context_only" else structures).append(row)
        role_order = {"stable_anchor": 0, "supporting_structure": 1}
        structures.sort(key=lambda row: (
            role_order.get(row["evidence_role"], 2),
            row["distance_m"],
            abs(row["relative_bearing_deg"]),
        ))
        context_only.sort(key=lambda row: row["distance_m"])
        lidar_rmse = float(getattr(candidate, "rmse_m", math.inf))
        descriptions.append({
            "candidate_id": f"C{index:02d}",
            "expected_visible_structures": structures[:8],
            "context_only_structures": context_only[:4],
            "visibility_model": {
                "dual_camera_horizontal_fov_deg": config.camera_horizontal_fov_deg,
                "camera_bearing_margin_deg": config.camera_bearing_margin_deg,
                "low_downward_camera": True,
                "absence_requires_explicit_flag": True,
            },
            "lidar_reliable": bool(getattr(candidate, "reliable", False)),
            "lidar_rmse_m": round(lidar_rmse, 4) if math.isfinite(lidar_rmse) else None,
            "lidar_inlier_ratio": round(float(getattr(candidate, "inlier_ratio", 0.0)), 4),
            "lidar_score": round(float(getattr(candidate, "score", 0.0)), 4),
        })
    return descriptions


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return "data:{};base64,{}".format(
        mime,
        base64.b64encode(path.read_bytes()).decode("ascii"),
    )


def _json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        begin, end = cleaned.find("{"), cleaned.rfind("}")
        if begin < 0 or end <= begin:
            raise ValueError("DeepSeek did not return a JSON object")
        value = json.loads(cleaned[begin:end + 1])
    if not isinstance(value, dict):
        raise ValueError("DeepSeek response is not a JSON object")
    return value


def parse_rank_response(text: str, allowed_ids: set[str]) -> VisionRankResult:
    value = _json_object(text)
    selected = str(value.get("selected_candidate_id") or "uncertain")
    second = str(value.get("second_candidate_id") or "uncertain")
    if selected not in allowed_ids:
        selected = "uncertain"
    if second not in allowed_ids or second == selected:
        second = "uncertain"
    structures = value.get("observed_structures") or []
    if not isinstance(structures, list):
        structures = []
    return VisionRankResult(
        selected_candidate_id=selected,
        confidence=_clamp(value.get("confidence")),
        second_candidate_id=second,
        second_confidence=_clamp(value.get("second_confidence")),
        observed_structures=tuple(item for item in structures[:12] if isinstance(item, dict)),
        evidence=str(value.get("evidence") or "")[:500],
        uncertain_reason=str(value.get("uncertain_reason") or "")[:500],
        consistent_repeats=1,
    )


class DeepSeekVisionClient:
    def __init__(self, config: DeepSeekVisionConfig) -> None:
        self.config = config

    def _once(
        self,
        left_image: Path,
        right_image: Path,
        candidates: list[dict[str, Any]],
    ) -> VisionRankResult:
        allowed = {str(item["candidate_id"]) for item in candidates}
        prompt = """
你是移动机器人视觉定位候选复核器。前两张图片是当前时刻的左、右摄像头画面。
下面 JSON 中每个 Cxx 都是雷达计算出的候选位姿，以及该位姿面向方向内理论上可能看见的
固定结构。候选已用双摄水平视场和占据地图射线做过筛选，但地图点位仍有不确定性，玻璃墙
后的结构可能被占据图标成 blocked。请综合玻璃墙、门、楼梯/栏杆、窗户和柜体的左右关系、
相对方向与组合结构判断，不能仅凭地板、颜色、亮度或一个普通柜子判断。

只把 expected_visible_structures 中 evidence_role=stable_anchor 的“实际出现”当强证据；
supporting_structure 只能辅助。context_only_structures 是桌椅等易移动或易遮挡物，其未出现
绝不能扣分。当前相机位置低且朝向地面，所有 absence_is_negative=false 的结构即使没拍到也
不能否定候选。map_ray=blocked 也可能只是透明玻璃在雷达地图中的墙线；若
visible_through_glass_relation=true，应允许在玻璃后看到它。点位误差会造成左右方位偏差，
请结合 bearing_uncertainty_deg 容忍误差。如果一个候选同时匹配至少两个稳定结构、雷达质量
明显更高且没有明确空间矛盾，可以选择它；否则返回 uncertain。

你只能选择给出的 candidate_id，绝对不能生成坐标。证据不足、候选相似或画面遮挡时，
selected_candidate_id 必须为 uncertain。特别检查左右相机是否共同支持同一结构关系。

候选：
{candidates}

只返回 JSON：
{{
  "selected_candidate_id": "C01 或 uncertain",
  "confidence": 0.0,
  "second_candidate_id": "C02 或 uncertain",
  "second_confidence": 0.0,
  "observed_structures": [
    {{"name": "玻璃墙", "side": "center/left/right", "confidence": 0.0}}
  ],
  "evidence": "说明支持候选的多结构空间关系",
  "uncertain_reason": "若不确定则说明原因"
}}
""".strip().format(candidates=json.dumps(candidates, ensure_ascii=False))
        payload = {
            "model": self.config.model,
            # Flash enables long-form thinking by default.  Candidate ranking
            # needs a short deterministic JSON answer; disabling thinking also
            # prevents a small max_tokens budget being consumed before content.
            "thinking": {"type": "disabled"},
            "temperature": 0.0,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "CURRENT LEFT CAMERA"},
                    {"type": "image_url", "image_url": {"url": _data_url(left_image), "detail": "high"}},
                    {"type": "text", "text": "CURRENT RIGHT CAMERA"},
                    {"type": "image_url", "image_url": {"url": _data_url(right_image), "detail": "high"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        last_content_error = ""
        # DeepSeek documents an occasional empty-content condition for JSON
        # Output. Retry that specific safe failure once; never accept or infer
        # a candidate from an empty/malformed response.
        for response_attempt in range(2):
            if response_attempt:
                payload["messages"][0]["content"].append({
                    "type": "text",
                    "text": "上一次没有得到可解析结果。请立即只输出一个完整 JSON 对象，不要留空。",
                })
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
                raise RuntimeError(f"DeepSeek Vision request failed: {exc}") from exc
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("DeepSeek Vision response has no message content") from exc
            try:
                return parse_rank_response(str(content or ""), allowed)
            except (ValueError, json.JSONDecodeError) as exc:
                last_content_error = str(exc)
        raise RuntimeError(
            "DeepSeek Vision returned empty or malformed JSON twice: "
            + last_content_error
        )

    def rank(
        self,
        left_image: Path,
        right_image: Path,
        candidates: list[dict[str, Any]],
    ) -> VisionRankResult:
        results = [
            self._once(left_image, right_image, candidates)
            for _ in range(self.config.repeat_count)
        ]
        selected = results[0].selected_candidate_id
        consistent = sum(item.selected_candidate_id == selected for item in results)
        if selected == "uncertain" or consistent != len(results):
            return VisionRankResult(
                selected_candidate_id="uncertain",
                confidence=min(item.confidence for item in results),
                second_candidate_id="uncertain",
                second_confidence=max(item.second_confidence for item in results),
                observed_structures=results[0].observed_structures,
                evidence="；".join(item.evidence for item in results if item.evidence)[:500],
                uncertain_reason="重复复核结果不一致或均不确定",
                consistent_repeats=consistent,
            )
        return VisionRankResult(
            selected_candidate_id=selected,
            confidence=min(item.confidence for item in results),
            second_candidate_id=results[0].second_candidate_id,
            second_confidence=max(item.second_confidence for item in results),
            observed_structures=results[0].observed_structures,
            evidence="；".join(item.evidence for item in results if item.evidence)[:500],
            uncertain_reason="",
            consistent_repeats=consistent,
        )


def _load_semantic_map(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def install(mission_module, config_path: str | Path | None = None) -> bool:
    """Patch only the release process; protected source files stay untouched."""
    module_directory = Path(__file__).resolve().parent
    default_path = Path(
        os.environ.get(
            "DEEPSEEK_VISION_CONFIG",
            str(Path(os.environ.get("CAR_ROOT", module_directory.parent)) / "secrets" / "deepseek_vision.json"),
        )
    )
    config = load_config(config_path or default_path, module_directory=module_directory)
    if not config.configured:
        print(
            f"DeepSeek Vision 未启用：请填写 {config.source_path}；继续使用安全定位回退。",
            flush=True,
        )
        return False
    semantic_map = _load_semantic_map(config.semantic_map)
    if not semantic_map.get("objects"):
        print(f"DeepSeek Vision 语义地图不可用：{config.semantic_map}", flush=True)
        return False

    client = DeepSeekVisionClient(config)
    def check(args, candidate, scan, localizer, camera, recognizer, gpu, directory, attempt):
        image_path = Path(directory) / f"localization_deepseek_{attempt:02d}.jpg"
        camera_info = camera.capture(image_path)
        second_name = camera_info.get("camera_2_image")
        right_path = image_path
        if second_name:
            possible = image_path.with_name(str(second_name))
            if possible.is_file():
                right_path = possible
        radar_candidates = localizer.global_localization_candidates(
            scan, maximum=config.maximum_candidates
        )
        if not radar_candidates:
            radar_candidates = [candidate]
        elif not any(
            math.hypot(item.pose.x_m - candidate.pose.x_m, item.pose.y_m - candidate.pose.y_m) < 0.25
            and abs(_normalise_angle(item.pose.yaw_rad - candidate.pose.yaw_rad)) < math.radians(15.0)
            for item in radar_candidates
        ):
            radar_candidates.insert(0, candidate)
        radar_candidates = radar_candidates[:config.maximum_candidates]
        descriptions = candidate_scene_descriptions(radar_candidates, semantic_map, config)
        diagnostics: dict[str, Any] = {
            "engine": "deepseek_vision_candidate_ranker",
            "model": config.model,
            "capture": [str(image_path), str(right_path)],
            "camera_count": camera_info.get("camera_count", 1),
            "candidate_descriptions": descriptions,
            "config_source": str(config.source_path),
            "api_key_logged": False,
        }
        try:
            result = client.rank(image_path, right_path, descriptions)
        except Exception as exc:
            diagnostics["deepseek_error"] = str(exc)
            diagnostics["fallback"] = "fail_closed_uncertain"
            return "uncertain", candidate, None, diagnostics
        diagnostics["deepseek_result"] = result.to_dict()
        reliable = bool(
            result.selected_candidate_id != "uncertain"
            and result.confidence >= config.minimum_confidence
            and result.margin >= config.minimum_margin
            and result.consistent_repeats == config.repeat_count
        )
        diagnostics["deepseek_reliable"] = reliable
        if not reliable:
            return "uncertain", candidate, None, diagnostics

        index = int(result.selected_candidate_id[1:]) - 1
        winner = radar_candidates[index]
        if not winner.reliable:
            diagnostics["rejected_reason"] = "selected_lidar_candidate_not_reliable"
            return "uncertain", candidate, None, diagnostics
        match = mission_module.VisualPlaceMatch(
            target_id="deepseek:" + result.selected_candidate_id,
            pose=winner.pose,
            score=result.confidence,
            runner_up_score=result.second_confidence,
            margin=result.margin,
            inliers=0,
            good_matches=0,
            reference_images=(str(image_path), str(right_path)),
            manifest=str(config.semantic_map),
            reliable=True,
        )
        position_error = math.hypot(
            candidate.pose.x_m - winner.pose.x_m,
            candidate.pose.y_m - winner.pose.y_m,
        )
        yaw_error = abs(_normalise_angle(candidate.pose.yaw_rad - winner.pose.yaw_rad))
        diagnostics["radar_visual_position_error_m"] = round(position_error, 4)
        diagnostics["radar_visual_yaw_error_deg"] = round(math.degrees(yaw_error), 2)
        if (
            position_error <= args.visual_localization_max_position_error
            and yaw_error <= math.radians(args.visual_localization_max_yaw_error_deg)
        ):
            return "confirmed", winner, match, diagnostics

        guided = localizer.localize_near_position(
            scan,
            winner.pose,
            radius_m=max(0.45, args.visual_localization_max_position_error),
        )
        if guided is not None and guided.reliable:
            guided_error = math.hypot(
                guided.pose.x_m - winner.pose.x_m,
                guided.pose.y_m - winner.pose.y_m,
            )
            diagnostics["guided_position_error_m"] = round(guided_error, 4)
            diagnostics["guided_rmse_m"] = round(guided.rmse_m, 4)
            if guided_error <= args.visual_localization_max_position_error:
                return "corrected", guided, match, diagnostics
        return "conflict", candidate, match, diagnostics

    mission_module._visual_localization_check = check
    print(
        "DeepSeek Vision 已启用：只复核雷达候选编号，禁止生成坐标；"
        f"model={config.model} repeats={config.repeat_count}",
        flush=True,
    )
    return True
