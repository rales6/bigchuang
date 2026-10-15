#!/usr/bin/env python3
"""Safety guard for relationship-map startup localization.

This module is loaded only by the independent semantic-map release.  It keeps
the protected ``raspberry_pi`` package unchanged while adding two safeguards:

* a configurable, map-relative start-pose prior for the first global match;
* rejection of oversized detections and lone generic furniture labels as
  authoritative visual-localization evidence.

The start prior is deliberately an environment setting, not a permanent map
assumption.  Remove ``SEARCH_START_POSE_HINT`` when the robot is not staged in
the documented start area.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class StartPosePrior:
    x_m: float
    y_m: float
    yaw_deg: float
    radius_m: float = 1.35
    yaw_tolerance_deg: float = 55.0


def parse_start_pose_prior(value: str | None) -> StartPosePrior | None:
    """Parse ``x_m,y_m,yaw_deg[,radius_m[,yaw_tolerance_deg]]``."""
    text = str(value or "").strip()
    if not text:
        return None
    fields = [item.strip() for item in text.split(",")]
    if len(fields) not in {3, 4, 5}:
        raise ValueError(
            "SEARCH_START_POSE_HINT must be "
            "x_m,y_m,yaw_deg[,radius_m[,yaw_tolerance_deg]]"
        )
    numbers = [float(item) for item in fields]
    prior = StartPosePrior(
        numbers[0],
        numbers[1],
        numbers[2],
        numbers[3] if len(numbers) >= 4 else 1.35,
        numbers[4] if len(numbers) >= 5 else 55.0,
    )
    if not all(math.isfinite(item) for item in numbers):
        raise ValueError("SEARCH_START_POSE_HINT contains a non-finite number")
    if not 0.20 <= prior.radius_m <= 2.50:
        raise ValueError("start-pose radius must be within 0.20..2.50 m")
    if not 5.0 <= prior.yaw_tolerance_deg <= 90.0:
        raise ValueError("start-pose yaw tolerance must be within 5..90 degrees")
    return prior


def _bbox(det: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = det.get("bbox_1000") or det.get("bbox") or det.get("box")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return box if all(math.isfinite(value) for value in box) else None


def filter_localization_detections(
    detections: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop Qwen boxes that are scene-wide strips rather than real objects."""
    filtered: list[dict[str, Any]] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        box = _bbox(detection)
        if box is not None:
            x1, y1, x2, y2 = box
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            # Qwen localization boxes use a 0..1000 coordinate system.  The
            # observed false "cabinet" was [0, 0, 1000, 100]: a banner across
            # the whole image, explicitly contradicting the detector prompt.
            scene_wide_strip = width >= 850.0 and height <= 220.0
            almost_full_frame = width >= 900.0 and height >= 800.0
            if scene_wide_strip or almost_full_frame:
                continue
        filtered.append(detection)
    return filtered


def guarded_landmark_labels(
    original: Callable[[Iterable[dict[str, Any]]], set[str]],
    detections: Iterable[dict[str, Any]],
) -> set[str]:
    """Keep generic furniture from selecting a global radar basin alone."""
    labels = set(original(filter_localization_detections(detections)))
    generic = {"cabinet", "chair", "table", "desk", "shelf"}
    if labels and labels <= generic:
        return set()
    return labels


def _constrained_start_localize(localizer, scan, prior: StartPosePrior, pose_type):
    """Rank lidar poses only inside the user-confirmed startup envelope."""
    center = pose_type(prior.x_m, prior.y_m, math.radians(prior.yaw_deg))
    points = localizer.points(scan, maximum=220)
    range_profile = localizer.range_profile(scan)
    if len(points) < 80:
        return None
    coarse_lattice = localizer._pose_lattice(
        center,
        translation_radius_m=prior.radius_m,
        translation_step_m=0.22,
        yaw_radius_deg=prior.yaw_tolerance_deg,
        yaw_step_deg=10.0,
    )
    coarse = localizer._rank(
        points,
        coarse_lattice,
        keep=6,
        range_profile=range_profile,
    )
    if not coarse:
        return None
    refined_lattice = []
    for _score, pose, _metrics in coarse:
        refined_lattice.extend(localizer._pose_lattice(
            pose,
            translation_radius_m=0.18,
            translation_step_m=0.06,
            yaw_radius_deg=10.0,
            yaw_step_deg=2.0,
        ))
    ranked = localizer._rank(
        points,
        refined_lattice,
        keep=12,
        range_profile=range_profile,
    )
    result = localizer._result_from_ranked(ranked)
    if result is None or not result.reliable:
        return None
    position_error = math.hypot(
        result.pose.x_m - prior.x_m,
        result.pose.y_m - prior.y_m,
    )
    yaw_error = abs(
        (result.pose.yaw_rad - math.radians(prior.yaw_deg) + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )
    if (
        position_error > prior.radius_m + 0.10
        or yaw_error > math.radians(prior.yaw_tolerance_deg + 3.0)
    ):
        return None
    # Ambiguity outside the confirmed startup envelope is intentionally
    # excluded.  Preserve any stronger in-envelope margin reported by lidar.
    return replace(result, ambiguity_margin=max(result.ambiguity_margin, 0.20))


def install(mission_module) -> StartPosePrior | None:
    """Install release-local guards into an imported mission module."""
    original_labels = mission_module.detection_landmark_labels

    def labels(detections):
        return guarded_landmark_labels(original_labels, detections)

    mission_module.detection_landmark_labels = labels

    prior = parse_start_pose_prior(os.environ.get("SEARCH_START_POSE_HINT"))
    if prior is None:
        print(
            "关系语义图定位保护已启用：无启动位姿先验；仅过滤不可靠视觉标注。",
            flush=True,
        )
        return None

    localizer_type = mission_module.ScanMapLocalizer
    original_global_localize = localizer_type.global_localize
    pose_type = mission_module.Pose2D

    def guarded_global_localize(self, scan, hint=None):
        if hint is not None:
            return original_global_localize(self, scan, hint=hint)
        result = _constrained_start_localize(self, scan, prior, pose_type)
        if result is not None:
            print(
                "启动位姿先验命中：x={:.3f}m y={:.3f}m yaw={:.1f}° "
                "(rmse={:.3f}m, inlier={:.1%})".format(
                    result.pose.x_m,
                    result.pose.y_m,
                    math.degrees(result.pose.yaw_rad),
                    result.rmse_m,
                    result.inlier_ratio,
                ),
                flush=True,
            )
            return result
        print("启动位姿先验内未找到可靠雷达匹配；拒绝回退到全图猜测。", flush=True)
        return None

    localizer_type.global_localize = guarded_global_localize
    print(
        "关系语义图启动先验：x={:.2f}m y={:.2f}m yaw={:.1f}° "
        "radius={:.2f}m yaw_tolerance={:.1f}°".format(
            prior.x_m,
            prior.y_m,
            prior.yaw_deg,
            prior.radius_m,
            prior.yaw_tolerance_deg,
        ),
        flush=True,
    )
    return prior

