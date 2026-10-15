"""Pure decision helpers for target selection and grasp verification."""

from __future__ import annotations

import math
import re
from typing import Any


# Exact equivalents of pressing the web UI's "前伸" (D20) button five times.
# Tuple entries: label, physical forward_mm, physical up_mm, duration_ms.
# The Pi serializer maps positive forward to the web UI's D command.
INITIAL_DEPLOY_SEQUENCE = tuple(
    (f"step{index}: D20 forward", 20, 0, 1000)
    for index in range(1, 6)
)


def autonomous_motion_ready(
    *, claw_detected_now: bool, claw_stable_frames: int,
    target_ok: bool, minimum_stable_frames: int = 3,
) -> bool:
    """Require a current stable claw detection and a valid target."""
    return bool(
        claw_detected_now
        and claw_stable_frames >= minimum_stable_frames
        and target_ok
    )


def autonomous_motion_ready(
    *, claw_detected_now: bool, claw_stable_frames: int,
    target_ok: bool, minimum_stable_frames: int = 3,
) -> bool:
    """Hard gate for live arm motion after deployment.

    A stale CSRT box is not sufficient: the production gripper model must see
    the claw in the current frame for several consecutive frames, while the
    requested target is also valid in that same camera.
    """
    return bool(
        claw_detected_now
        and claw_stable_frames >= minimum_stable_frames
        and target_ok
    )


def target_text_score(item: dict[str, Any], target_description: str) -> float:
    target = re.sub(r"\s+", "", str(target_description or "").lower())
    if not target:
        return 1.0
    text = " ".join(str(item.get(key, "")) for key in (
        "name", "type", "appearance", "reason", "description",
    )).lower()
    compact = re.sub(r"\s+", "", text)
    if target in compact:
        return 3.0
    tokens = [token for token in re.split(r"[\s,，。;；、]+", target) if token]
    token_score = sum(1.0 for token in tokens if token in compact)
    unique = set(target)
    char_score = sum(1.0 for char in unique if char in compact) / max(1, len(unique))
    return token_score + char_score


def select_requested_object(
    objects: list[dict[str, Any]], target_description: str,
) -> dict[str, Any] | None:
    pickable = [item for item in objects if item.get("pickable", True)]
    if not pickable:
        return None
    target = str(target_description or "").strip()
    if not target:
        return max(pickable, key=lambda item: float(item.get("confidence", 0) or 0))
    explicit = [item for item in pickable if item.get("matches_request") is True]
    candidates = explicit or [
        item for item in pickable if target_text_score(item, target) >= 0.80
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (
        target_text_score(item, target),
        float(item.get("confidence", 0) or 0),
    ))


def choose_camera_from_qwen(parsed: dict[str, Any], default: str = "left") -> str:
    """Choose Qwen's preferred half from a side-by-side camera comparison."""
    preferred = str(parsed.get("preferred_camera", "")).lower()
    if preferred in ("left", "right"):
        return preferred
    scored: list[tuple[float, str]] = []
    for camera in ("left", "right"):
        item = parsed.get(camera)
        if not isinstance(item, dict) or not item.get("found", False):
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        bbox = item.get("bbox_2d") or item.get("bbox") or []
        area = 0.0
        if isinstance(bbox, list) and len(bbox) == 4:
            area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
                0.0, float(bbox[3]) - float(bbox[1])
            )
        scored.append((confidence + min(area / 1_000_000.0, 0.25), camera))
    if scored:
        return max(scored)[1]
    bbox = parsed.get("bbox_2d") or parsed.get("bbox") or []
    if parsed.get("found") and isinstance(bbox, list) and len(bbox) == 4:
        center_x = (float(bbox[0]) + float(bbox[2])) * 0.5
        return "left" if center_x < 500.0 else "right"
    return default if default in ("left", "right") else "left"


def stored_tracking_boxes(dual_tracker: Any) -> dict[str, Any] | None:
    """Return stored tracker boxes without assuming optional convenience properties."""
    claw = getattr(dual_tracker, "claw", None)
    target = getattr(dual_tracker, "target", None)
    claw_ok = bool(getattr(claw, "initialized", False))
    target_ok = bool(getattr(target, "initialized", False))
    if not claw_ok and not target_ok:
        return None
    return {
        "claw_ok": claw_ok,
        "target_ok": target_ok,
        "claw_bbox": getattr(claw, "bbox", None),
        "target_bbox": getattr(target, "bbox", None),
    }



def jaw_geometry(keypoints: Any, target_bbox: Any | None = None) -> dict[str, Any]:
    """Return image-plane geometry for two jaw contact points."""
    visible: list[tuple[float, float]] = []
    for point in keypoints or []:
        if isinstance(point, dict):
            visibility = float(point.get("v", point.get("confidence", 0.0)) or 0.0)
            x, y = point.get("x", 0), point.get("y", 0)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
            visibility = float(point[2]) if len(point) > 2 else 2.0
        else:
            continue
        if visibility > 0:
            visible.append((float(x), float(y)))
    visible.sort(key=lambda point: point[0])
    result: dict[str, Any] = {
        "visible_count": len(visible), "points": visible, "ready": len(visible) >= 2,
    }
    if len(visible) < 2:
        return result
    left, right = visible[0], visible[-1]
    midpoint = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)
    result.update({
        "left": left, "right": right, "midpoint": midpoint,
        "span_px": math.hypot(right[0] - left[0], right[1] - left[1]),
        "opening_x_px": max(0.0, right[0] - left[0]),
    })
    if target_bbox is not None:
        target_cx = (float(target_bbox.x1) + float(target_bbox.x2)) * 0.5
        target_cy = (float(target_bbox.y1) + float(target_bbox.y2)) * 0.5
        target_width = max(1.0, float(target_bbox.x2) - float(target_bbox.x1))
        target_height = max(1.0, float(target_bbox.y2) - float(target_bbox.y1))
        # Put the jaw midpoint below the object center before closing so both
        # fingers descend far enough to wrap around the object. Horizontal
        # tolerance stays generous, while the vertical band is deliberately
        # lower and tighter to avoid closing from the previous high position.
        center_y_offset = max(12.0, min(26.0, target_height * 0.20))
        desired_grasp_cy = target_cy + center_y_offset
        result.update({
            "target_cx": target_cx, "target_cy": target_cy,
            "desired_grasp_cy": desired_grasp_cy,
            "center_y_offset_px": center_y_offset,
            "dx_px": target_cx - midpoint[0],
            "dy_px": desired_grasp_cy - midpoint[1],
            "tip_gap_px": float(target_bbox.y1) - midpoint[1],
            "target_between": left[0] <= target_cx <= right[0],
            "opening_ratio": (right[0] - left[0]) / target_width,
            "center_x_tolerance_px": max(26.0, min(52.0, target_width * 0.40)),
            "center_y_tolerance_px": max(32.0, min(80.0, target_height * 0.48)),
        })
    return result

def dual_camera_grasp_ready(
    parsed: dict[str, Any],
    *,
    left_claw: Any,
    right_claw: Any,
    frame_width: int = 640,
    frame_height: int = 480,
    minimum_confidence: float = 0.45,
    left_keypoints: Any = None,
    right_keypoints: Any = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Broad final-close gate using two independent camera views."""
    details: dict[str, Any] = {}
    failures: list[str] = []
    for camera, claw, keypoints in (("left", left_claw, left_keypoints), ("right", right_claw, right_keypoints)):
        item = parsed.get(camera)
        if not isinstance(item, dict):
            item = {}
        found = bool(item.get("target_found", item.get("found", False)))
        confidence = float(item.get("confidence", 0.0) or 0.0)
        target = item.get("target_bbox") or item.get("bbox_2d") or item.get("bbox") or []
        view = {"target_found": found, "confidence": confidence,
                "claw_found": claw is not None, "ready": False}
        details[camera] = view
        if claw is None:
            failures.append(f"{camera}: gripper not detected")
            continue
        if not found or not isinstance(target, list) or len(target) != 4:
            failures.append(f"{camera}: requested target not detected")
            continue
        if confidence < minimum_confidence:
            failures.append(f"{camera}: target confidence {confidence:.2f} < {minimum_confidence:.2f}")
            continue
        try:
            values = [float(value) for value in target]
        except (TypeError, ValueError):
            failures.append(f"{camera}: invalid target bbox")
            continue
        # Qwen 对左右拼接图返回的是「整幅拼接图」的 0..1000 归一化坐标，
        # 需先换算到整图像素再减去该半帧偏移；完全越界时回退按半帧坐标。
        camera_offset = 0.0 if camera == "left" else float(frame_width)
        if max(values) <= 1000.0:
            full_w = float(frame_width) * 2.0
            x1 = values[0] / 1000.0 * full_w - camera_offset
            x2 = values[2] / 1000.0 * full_w - camera_offset
            y1 = values[1] / 1000.0 * frame_height
            y2 = values[3] / 1000.0 * frame_height
            if x2 <= 0 or x1 >= float(frame_width):
                x1 = values[0] / 1000.0 * frame_width
                x2 = values[2] / 1000.0 * frame_width
                y1 = values[1] / 1000.0 * frame_height
                y2 = values[3] / 1000.0 * frame_height
        else:
            x1, y1, x2, y2 = values
            x1 -= camera_offset
            x2 -= camera_offset
        target_w = max(1.0, x2 - x1)
        target_h = max(1.0, y2 - y1)
        target_cx = (x1 + x2) * 0.5
        claw_cx = (float(claw.x1) + float(claw.x2)) * 0.5
        fingertip_gap = y1 - float(claw.y2)
        dx = target_cx - claw_cx
        x_limit = max(50.0, min(150.0, max(target_w * 1.50, float(claw.width) * 0.60)))
        gap_limit = max(45.0, min(130.0, target_h * 1.80))
        pose_geometry = jaw_geometry(keypoints, type("TargetBox", (), {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })()) if keypoints is not None else None
        if pose_geometry is not None:
            geometry_ready = (
                pose_geometry["visible_count"] >= 2
                and bool(pose_geometry.get("target_between"))
                and float(pose_geometry.get("opening_ratio", 0.0)) >= 0.55
                and abs(float(pose_geometry.get("dx_px", 9999.0)))
                    <= float(pose_geometry.get("center_x_tolerance_px", 0.0)) * 1.25
                and abs(float(pose_geometry.get("dy_px", 9999.0)))
                    <= float(pose_geometry.get("center_y_tolerance_px", 0.0))
            )
            view["jaw_points"] = pose_geometry
            view["center_rectangle"] = {
                "dx_px": round(float(pose_geometry.get("dx_px", 0.0)), 1),
                "dy_px": round(float(pose_geometry.get("dy_px", 0.0)), 1),
                "x_tolerance_px": round(float(pose_geometry.get("center_x_tolerance_px", 0.0)), 1),
                "y_tolerance_px": round(float(pose_geometry.get("center_y_tolerance_px", 0.0)), 1),
            }
            if pose_geometry["visible_count"] < 2:
                failures.append(f"{camera}: only {pose_geometry['visible_count']} jaw point visible")
            elif not pose_geometry.get("target_between"):
                failures.append(f"{camera}: target is not between jaw points")
            elif float(pose_geometry.get("opening_ratio", 0.0)) < 0.55:
                failures.append(f"{camera}: jaw opening is narrower than target")
        else:
            geometry_ready = abs(dx) <= x_limit and abs(fingertip_gap) <= gap_limit
        view.update({"target_bbox": [x1, y1, x2, y2], "dx_px": round(dx, 1),
                     "fingertip_gap_px": round(fingertip_gap, 1),
                     "x_limit_px": round(x_limit, 1), "gap_limit_px": round(gap_limit, 1),
                     "ready": geometry_ready})
        if not geometry_ready:
            failures.append(f"{camera}: offset dx={dx:.0f}px gap={fingertip_gap:.0f}px")
    ready = not failures and all(bool(details.get(c, {}).get("ready")) for c in ("left", "right"))
    reason = "both cameras confirm close geometry" if ready else "; ".join(failures)
    return ready, reason, details

def relative_target_state(claw_bbox: Any, target_bbox: Any) -> dict[str, float] | None:
    if claw_bbox is None or target_bbox is None:
        return None
    claw_x = (float(claw_bbox.x1) + float(claw_bbox.x2)) * 0.5
    claw_y = (float(claw_bbox.y1) + float(claw_bbox.y2)) * 0.5
    target_x = (float(target_bbox.x1) + float(target_bbox.x2)) * 0.5
    target_y = (float(target_bbox.y1) + float(target_bbox.y2)) * 0.5
    return {
        "claw_x": claw_x,
        "claw_y": claw_y,
        "target_x": target_x,
        "target_y": target_y,
        "rel_x": target_x - claw_x,
        "rel_y": target_y - claw_y,
    }


def grasp_motion_verified(
    before: dict[str, float] | None,
    after: dict[str, float] | None,
    *,
    minimum_motion_px: float = 6.0,
    maximum_relative_drift_px: float = 38.0,
) -> tuple[bool, str]:
    if before is None or after is None:
        return False, "claw or target not visible after lift"
    claw_dx = after["claw_x"] - before["claw_x"]
    claw_dy = after["claw_y"] - before["claw_y"]
    target_dx = after["target_x"] - before["target_x"]
    target_dy = after["target_y"] - before["target_y"]
    claw_motion = math.hypot(claw_dx, claw_dy)
    target_motion = math.hypot(target_dx, target_dy)
    relative_drift = math.hypot(
        after["rel_x"] - before["rel_x"],
        after["rel_y"] - before["rel_y"],
    )
    if claw_motion < minimum_motion_px:
        return False, f"lift motion too small to verify ({claw_motion:.1f}px)"
    dot = claw_dx * target_dx + claw_dy * target_dy
    motion_ratio = target_motion / max(claw_motion, 1.0)
    verified = (
        dot > 0
        and 0.30 <= motion_ratio <= 2.50
        and relative_drift <= maximum_relative_drift_px
    )
    reason = (
        f"co-motion claw={claw_motion:.1f}px target={target_motion:.1f}px "
        f"ratio={motion_ratio:.2f} drift={relative_drift:.1f}px"
    )
    return verified, reason
