from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import cv2
import numpy as np


@dataclass
class LidarFeature:
    feature_id: str
    feature_type: str
    center_xy: list[float]
    yaw_rad: float | None
    score: float
    description: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def scan_to_local_points(scan, max_distance_m: float = 4.0) -> np.ndarray:
    angles = np.asarray(scan.angles_rad, dtype=np.float64)
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    valid = np.isfinite(angles) & np.isfinite(distances)
    valid &= distances > 0.08
    valid &= distances <= max_distance_m
    angles = angles[valid]
    distances = distances[valid]
    return np.stack((np.cos(angles) * distances, np.sin(angles) * distances), axis=1)


def scan_to_world_points(scan, pose, max_distance_m: float = 4.0) -> np.ndarray:
    return pose.transform_points(scan_to_local_points(scan, max_distance_m=max_distance_m))


def extract_lidar_features(scan, pose, max_distance_m: float = 4.0) -> list[LidarFeature]:
    """从当前雷达帧提取粗略几何参考点。

    这里先做轻量版本：把连续雷达点按距离断点分段，再用 PCA 判断长直线、
    紧凑簇和可能的 L 型角点。它不是完整 SLAM 特征提取器，但足够给 Qwen
    提供“雷达图里有哪些稳定候选”的上下文。
    """

    local = scan_to_local_points(scan, max_distance_m=max_distance_m)
    if len(local) < 12:
        return []
    world = pose.transform_points(local)
    segments = _segments(world)
    features: list[LidarFeature] = []
    line_features: list[LidarFeature] = []

    for index, segment in enumerate(segments):
        if len(segment) < 8:
            continue
        feature = _segment_feature(segment, index)
        if feature is None:
            continue
        features.append(feature)
        if feature.feature_type == "line":
            line_features.append(feature)

    features.extend(_corner_features(line_features))
    features.sort(key=lambda item: item.score, reverse=True)
    return features[:12]


def render_lidar_context(scan, pose, features: list[LidarFeature], size_px: int = 640) -> bytes:
    """生成给 Qwen 看的局部雷达俯视图 PNG。"""

    image = np.full((size_px, size_px, 3), 245, dtype=np.uint8)
    scale = size_px / 8.0
    center = np.array((size_px / 2.0, size_px / 2.0))
    points = scan_to_local_points(scan, max_distance_m=4.0)
    if len(points):
        pixels = np.column_stack((points[:, 1], -points[:, 0])) * scale + center
        for x, y in pixels.astype(int):
            if 0 <= x < size_px and 0 <= y < size_px:
                image[y, x] = (30, 30, 30)

    # 小车中心和朝向
    cv2.circle(image, tuple(center.astype(int)), 7, (0, 120, 255), -1)
    cv2.arrowedLine(
        image,
        tuple(center.astype(int)),
        tuple((center + np.array((0, -45))).astype(int)),
        (0, 120, 255),
        2,
        tipLength=0.25,
    )

    for item in features[:8]:
        # 转回以当前 pose 为中心的局部显示坐标
        world_xy = np.asarray(item.center_xy, dtype=np.float64)
        dx = world_xy[0] - float(pose.x_m)
        dy = world_xy[1] - float(pose.y_m)
        c = math.cos(-pose.yaw_rad)
        s = math.sin(-pose.yaw_rad)
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy
        pixel = np.array((local_y, -local_x)) * scale + center
        px, py = pixel.astype(int)
        if 0 <= px < size_px and 0 <= py < size_px:
            color = (40, 180, 40) if item.feature_type == "corner" else (255, 120, 30)
            cv2.circle(image, (px, py), 8, color, 2)
            cv2.putText(
                image,
                item.feature_id,
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return b""
    return bytes(encoded)


def _segments(points: np.ndarray, gap_m: float = 0.28) -> list[np.ndarray]:
    breaks = np.where(np.linalg.norm(np.diff(points, axis=0), axis=1) > gap_m)[0] + 1
    return [seg for seg in np.split(points, breaks) if len(seg)]


def _segment_feature(segment: np.ndarray, index: int) -> LidarFeature | None:
    center = np.mean(segment, axis=0)
    shifted = segment - center
    covariance = shifted.T @ shifted / max(1, len(segment) - 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    length = float(np.linalg.norm(np.max(segment, axis=0) - np.min(segment, axis=0)))
    if values[1] <= 1e-8:
        linearity = 99.0
    else:
        linearity = float(values[0] / values[1])

    if length >= 0.45 and linearity >= 8.0:
        yaw = math.atan2(float(vectors[1, 0]), float(vectors[0, 0]))
        return LidarFeature(
            feature_id=f"L{index}",
            feature_type="line",
            center_xy=[float(center[0]), float(center[1])],
            yaw_rad=yaw,
            score=min(1.0, 0.35 + length * 0.18 + min(linearity, 30.0) * 0.01),
            description=f"line length={length:.2f}m linearity={linearity:.1f}",
        )

    if 0.08 <= length <= 0.50 and len(segment) >= 10:
        return LidarFeature(
            feature_id=f"C{index}",
            feature_type="compact_cluster",
            center_xy=[float(center[0]), float(center[1])],
            yaw_rad=None,
            score=0.45,
            description=f"compact lidar cluster size={length:.2f}m",
        )
    return None


def _corner_features(lines: list[LidarFeature]) -> list[LidarFeature]:
    corners: list[LidarFeature] = []
    for i, left in enumerate(lines):
        for right in lines[i + 1 :]:
            if left.yaw_rad is None or right.yaw_rad is None:
                continue
            angle = abs(_normalize(left.yaw_rad - right.yaw_rad))
            angle = min(angle, math.pi - angle)
            distance = math.hypot(
                left.center_xy[0] - right.center_xy[0],
                left.center_xy[1] - right.center_xy[1],
            )
            if math.radians(55) <= angle <= math.radians(125) and distance <= 1.2:
                center = [
                    (left.center_xy[0] + right.center_xy[0]) / 2.0,
                    (left.center_xy[1] + right.center_xy[1]) / 2.0,
                ]
                corners.append(
                    LidarFeature(
                        feature_id=f"K{len(corners)}",
                        feature_type="corner",
                        center_xy=[float(center[0]), float(center[1])],
                        yaw_rad=None,
                        score=0.75,
                        description=f"L-shaped corner from {left.feature_id}+{right.feature_id}",
                    )
                )
    return corners


def _normalize(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
