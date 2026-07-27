from __future__ import annotations

import json
import re
from typing import Any

from qwen_grounded_tracker.domain import BBox, GroundedCandidate, GroundingResult


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            repaired = candidate
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            repaired = re.sub(r"(?<!\\)'", '"', repaired)
            repaired = re.sub(r"\bTrue\b", "true", repaired)
            repaired = re.sub(r"\bFalse\b", "false", repaired)
            repaired = re.sub(r"\bNone\b", "null", repaired)
            try:
                value = json.loads(repaired)
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def _extract_bbox_fallback(text: str) -> list[float] | None:
    match = re.search(
        r"(?:bbox_2d|bbox|box)\s*[\"']?\s*[:=]\s*\[\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        # Last-resort parser for answers like: [123, 45, 678, 900]
        generic = re.search(
            r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
            r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
            text,
        )
        if not generic:
            return None
        return [float(generic.group(i)) for i in range(1, 5)]
    return [float(match.group(i)) for i in range(1, 5)]


def _convert_bbox(
    values: list[float],
    coordinate_system: str,
    frame_width: int,
    frame_height: int,
) -> BBox:
    x1, y1, x2, y2 = values
    system = coordinate_system.lower().strip()

    if system in {"relative_1000", "normalized_1000", "0-1000", "1000"}:
        scale_x = frame_width / 1000.0
        scale_y = frame_height / 1000.0
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
    elif system in {"normalized", "relative", "0-1", "1"}:
        x1, x2 = x1 * frame_width, x2 * frame_width
        y1, y2 = y1 * frame_height, y2 * frame_height
    elif system not in {"pixel", "pixels", "absolute"}:
        max_value = max(abs(v) for v in values)
        if max_value <= 1.5:
            x1, x2 = x1 * frame_width, x2 * frame_width
            y1, y2 = y1 * frame_height, y2 * frame_height
        else:
            # The grounding prompt explicitly requests a relative 1000x1000 grid.
            x1, x2 = x1 * frame_width / 1000.0, x2 * frame_width / 1000.0
            y1, y2 = y1 * frame_height / 1000.0, y2 * frame_height / 1000.0

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return BBox(x1, y1, x2, y2).clamp(frame_width, frame_height)


def _candidate_from_payload(
    payload: dict[str, Any],
    frame_width: int,
    frame_height: int,
    default_coordinate_system: str,
    minimum_box_area_ratio: float,
    fallback_name: str,
) -> GroundedCandidate | None:
    candidate = payload.get("bbox_2d") or payload.get("bbox") or payload.get("box")
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 4:
        return None
    try:
        bbox_values = [float(candidate[i]) for i in range(4)]
    except (TypeError, ValueError):
        return None

    coordinate_system = str(payload.get("coordinate_system", default_coordinate_system))
    bbox = _convert_bbox(bbox_values, coordinate_system, frame_width, frame_height)
    area_ratio = bbox.area_ratio(frame_width, frame_height)
    if bbox.width < 3 or bbox.height < 3 or area_ratio < minimum_box_area_ratio:
        return None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    target_name = str(
        payload.get("target_name")
        or payload.get("label")
        or payload.get("name")
        or fallback_name
    )
    return GroundedCandidate(
        target_name=target_name,
        bbox=bbox,
        confidence=max(0.0, min(confidence, 1.0)),
    )


def parse_grounding_output(
    raw_text: str,
    frame_width: int,
    frame_height: int,
    minimum_box_area_ratio: float = 0.001,
) -> GroundingResult:
    payload = _extract_json_object(raw_text)

    found = True
    target_name = "target"
    confidence = 0.0
    coordinate_system = "relative_1000"
    bbox_values: list[float] | None = None

    if payload is not None:
        found = bool(payload.get("found", True))
        target_name = str(
            payload.get("target_name")
            or payload.get("label")
            or payload.get("name")
            or "target"
        )
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        coordinate_system = str(payload.get("coordinate_system", "relative_1000"))
        # 兼容多目标输出：Qwen 可以返回 targets 数组，也可以继续返回旧的单 bbox。
        targets = payload.get("targets")
        if isinstance(targets, list):
            candidates = []
            for index, item in enumerate(targets, start=1):
                if not isinstance(item, dict):
                    continue
                parsed = _candidate_from_payload(
                    payload=item,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    default_coordinate_system=coordinate_system,
                    minimum_box_area_ratio=minimum_box_area_ratio,
                    fallback_name=f"{target_name}_{index}",
                )
                if parsed is not None:
                    candidates.append(parsed)
            if found and candidates:
                candidates.sort(key=lambda item: item.bbox.x1)
                first = candidates[0]
                return GroundingResult(
                    found=True,
                    bbox=first.bbox,
                    target_name=first.target_name,
                    confidence=first.confidence,
                    raw_text=raw_text,
                    message=f"{len(candidates)} target candidates grounded",
                    candidates=candidates,
                )
        candidate = payload.get("bbox_2d") or payload.get("bbox") or payload.get("box")
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 4:
            try:
                bbox_values = [float(candidate[i]) for i in range(4)]
            except (TypeError, ValueError):
                bbox_values = None

    if bbox_values is None:
        bbox_values = _extract_bbox_fallback(raw_text)

    if not found or bbox_values is None:
        return GroundingResult(
            found=False,
            bbox=None,
            target_name=target_name,
            confidence=confidence,
            raw_text=raw_text,
            message="Qwen did not return a valid target box",
        )

    bbox = _convert_bbox(
        bbox_values,
        coordinate_system,
        frame_width,
        frame_height,
    )
    area_ratio = bbox.area_ratio(frame_width, frame_height)
    if bbox.width < 3 or bbox.height < 3 or area_ratio < minimum_box_area_ratio:
        return GroundingResult(
            found=False,
            bbox=None,
            target_name=target_name,
            confidence=confidence,
            raw_text=raw_text,
            message=f"Grounding box is too small: area_ratio={area_ratio:.6f}",
        )

    return GroundingResult(
        found=True,
        bbox=bbox,
        target_name=target_name,
        confidence=max(0.0, min(confidence, 1.0)),
        raw_text=raw_text,
        message="target grounded",
        candidates=[
            GroundedCandidate(
                target_name=target_name,
                bbox=bbox,
                confidence=max(0.0, min(confidence, 1.0)),
            )
        ],
    )
