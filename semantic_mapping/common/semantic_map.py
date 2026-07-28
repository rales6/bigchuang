from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import time
from typing import Any


def _now() -> float:
    return time.time()


@dataclass
class SemanticObservation:
    """一次视觉/雷达语义观测。"""

    timestamp_s: float
    robot_pose: dict[str, float]
    source: str
    confidence: float
    camera_bbox: list[float] = field(default_factory=list)
    lidar_feature_id: str = ""
    note: str = ""


@dataclass
class SemanticLandmark:
    """地图中的自然参考点或语义物体。"""

    landmark_id: str
    name: str
    landmark_type: str
    map_xy: list[float]
    yaw_rad: float | None = None
    visual_description: str = ""
    lidar_description: str = ""
    stability: float = 0.5
    first_seen_s: float = field(default_factory=_now)
    last_seen_s: float = field(default_factory=_now)
    seen_count: int = 1
    failed_match_count: int = 0
    observations: list[SemanticObservation] = field(default_factory=list)

    def distance_to(self, xy: list[float]) -> float:
        return math.hypot(self.map_xy[0] - xy[0], self.map_xy[1] - xy[1])


class SemanticMap:
    """保存自然参考点和语义物体的轻量 JSON 地图。"""

    def __init__(self) -> None:
        self.created_at_s = _now()
        self.updated_at_s = self.created_at_s
        self.landmarks: list[SemanticLandmark] = []
        self.events: list[dict[str, Any]] = []

    @classmethod
    def load(cls, path: str | Path) -> "SemanticMap":
        path = Path(path)
        result = cls()
        if not path.exists():
            return result
        raw = json.loads(path.read_text(encoding="utf-8"))
        result.created_at_s = float(raw.get("created_at_s", _now()))
        result.updated_at_s = float(raw.get("updated_at_s", result.created_at_s))
        result.events = list(raw.get("events", []))
        result.landmarks = []
        for item in raw.get("landmarks", []):
            observations = [
                SemanticObservation(**obs)
                for obs in item.get("observations", [])
                if isinstance(obs, dict)
            ]
            item = dict(item)
            item["observations"] = observations
            result.landmarks.append(SemanticLandmark(**item))
        return result

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at_s = _now()
        payload = {
            "schema": "semantic_map.v1",
            "created_at_s": self.created_at_s,
            "updated_at_s": self.updated_at_s,
            "landmarks": [
                {
                    **asdict(landmark),
                    "observations": [asdict(obs) for obs in landmark.observations[-20:]],
                }
                for landmark in self.landmarks
            ],
            "events": self.events[-200:],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def add_event(self, kind: str, **payload: Any) -> None:
        self.events.append({"time_s": _now(), "kind": kind, **payload})

    def upsert_landmark(
        self,
        *,
        name: str,
        landmark_type: str,
        map_xy: list[float],
        robot_pose: dict[str, float],
        confidence: float,
        source: str,
        camera_bbox: list[float] | None = None,
        lidar_feature_id: str = "",
        visual_description: str = "",
        lidar_description: str = "",
        yaw_rad: float | None = None,
        merge_radius_m: float = 0.45,
    ) -> SemanticLandmark:
        """新增或合并同类近距离地标。

        无标记自然环境中，Qwen 给出的语义可能会有波动，所以这里不按名称严格匹配；
        只要类型相同、地图坐标足够接近，就认为可能是同一个参考点。
        """

        confidence = max(0.0, min(1.0, float(confidence)))
        now_s = _now()
        bbox = list(camera_bbox or [])
        candidate = self._nearest_compatible(landmark_type, map_xy, merge_radius_m)
        observation = SemanticObservation(
            timestamp_s=now_s,
            robot_pose=robot_pose,
            source=source,
            confidence=confidence,
            camera_bbox=bbox,
            lidar_feature_id=lidar_feature_id,
            note=visual_description,
        )

        if candidate is None:
            landmark = SemanticLandmark(
                landmark_id=self._next_id(landmark_type),
                name=name or landmark_type,
                landmark_type=landmark_type,
                map_xy=[float(map_xy[0]), float(map_xy[1])],
                yaw_rad=yaw_rad,
                visual_description=visual_description,
                lidar_description=lidar_description,
                stability=0.35 + confidence * 0.45,
                observations=[observation],
            )
            self.landmarks.append(landmark)
            self.add_event("landmark_created", id=landmark.landmark_id, name=landmark.name)
            return landmark

        # 稳定地标更新时采用小步平均，避免一次错误观测把地图锚点拉偏。
        weight = 0.15 + 0.20 * confidence
        candidate.map_xy = [
            candidate.map_xy[0] * (1.0 - weight) + float(map_xy[0]) * weight,
            candidate.map_xy[1] * (1.0 - weight) + float(map_xy[1]) * weight,
        ]
        if yaw_rad is not None:
            candidate.yaw_rad = yaw_rad if candidate.yaw_rad is None else (
                candidate.yaw_rad * 0.8 + float(yaw_rad) * 0.2
            )
        candidate.name = name or candidate.name
        candidate.visual_description = visual_description or candidate.visual_description
        candidate.lidar_description = lidar_description or candidate.lidar_description
        candidate.last_seen_s = now_s
        candidate.seen_count += 1
        candidate.stability = min(1.0, candidate.stability + 0.04 + confidence * 0.04)
        candidate.observations.append(observation)
        self.add_event("landmark_seen", id=candidate.landmark_id, name=candidate.name)
        return candidate

    def query(self, text: str, limit: int = 5) -> list[SemanticLandmark]:
        text = text.strip().lower()
        if not text:
            return []
        scored: list[tuple[float, SemanticLandmark]] = []
        for landmark in self.landmarks:
            haystack = " ".join(
                [
                    landmark.name,
                    landmark.landmark_type,
                    landmark.visual_description,
                    landmark.lidar_description,
                ]
            ).lower()
            score = landmark.stability
            if text in haystack:
                score += 1.0
            for token in text.split():
                if token and token in haystack:
                    score += 0.25
            if score > landmark.stability:
                scored.append((score, landmark))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _nearest_compatible(
        self,
        landmark_type: str,
        map_xy: list[float],
        radius_m: float,
    ) -> SemanticLandmark | None:
        compatible = [
            item
            for item in self.landmarks
            if item.landmark_type == landmark_type and item.distance_to(map_xy) <= radius_m
        ]
        if not compatible:
            return None
        return min(compatible, key=lambda item: item.distance_to(map_xy))

    def _next_id(self, landmark_type: str) -> str:
        prefix = "".join(ch if ch.isalnum() else "_" for ch in landmark_type.lower()).strip("_")
        prefix = prefix or "landmark"
        return f"{prefix}_{len(self.landmarks) + 1:04d}"
