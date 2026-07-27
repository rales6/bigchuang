from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qwen_grounded_tracker.domain import (
    BBox,
    ObstacleDetection,
    ObstacleObservation,
)


class YoloObstacleDetector:
    """Optional semantic obstacle detector.

    The target itself is tracked by CSRT and is category-agnostic. YOLO is used
    only for semantic safety cues such as people and vehicles.
    """

    def __init__(
        self,
        enabled: bool,
        model_path: str,
        device: str = "cpu",
        image_size: int = 416,
        confidence: float = 0.25,
        frame_interval: int = 3,
        stop_labels: list[str] | None = None,
        danger_zone: dict[str, float] | None = None,
        exclude_target_iou: float = 0.60,
        safety_only: bool = True,
        report_clear_detections: bool = False,
    ) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.frame_interval = max(1, frame_interval)
        self.stop_labels = set(stop_labels or [])
        self.danger_zone = danger_zone or {"x1": 0.28, "y1": 0.42, "x2": 0.72, "y2": 1.0}
        self.exclude_target_iou = exclude_target_iou
        self.safety_only = safety_only
        self.report_clear_detections = report_clear_detections
        self.model: Any | None = None
        self.stop_class_ids: list[int] | None = None
        self.frame_index = 0
        self.last_observation = ObstacleObservation(status="not run")

    def _ensure_loaded(self) -> None:
        if not self.enabled or self.model is not None:
            return
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO obstacle model not found: {model_path.resolve()}"
            )
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        names = getattr(self.model, "names", {})
        if self.safety_only and self.stop_labels and isinstance(names, dict):
            self.stop_class_ids = [
                int(class_id)
                for class_id, label in names.items()
                if str(label) in self.stop_labels
            ]
            print(
                "[YOLO obstacles] safety-only classes: "
                f"{self.stop_class_ids or 'none matched'}"
            )
        print(f"[YOLO obstacles] model loaded: {model_path}")

    def detect(
        self,
        frame: np.ndarray,
        target_bbox: BBox | None = None,
    ) -> ObstacleObservation:
        if not self.enabled:
            return ObstacleObservation(status="YOLO obstacle detector disabled")

        self.frame_index += 1
        if self.frame_index % self.frame_interval != 0:
            return self.last_observation

        self._ensure_loaded()
        assert self.model is not None

        predict_kwargs: dict[str, Any] = {
            "imgsz": self.image_size,
            "conf": self.confidence,
            "device": self.device,
            "verbose": False,
        }
        if self.safety_only and self.stop_class_ids:
            # 只跑安全相关类别，避免 YOLO 把普通目标当作“识别结果”干扰 Qwen/CSRT 主目标链路。
            predict_kwargs["classes"] = self.stop_class_ids

        results = self.model.predict(frame, **predict_kwargs)
        if not results:
            self.last_observation = ObstacleObservation(status="no result")
            return self.last_observation

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self.last_observation = ObstacleObservation(status="no semantic obstacles")
            return self.last_observation

        frame_h, frame_w = frame.shape[:2]
        zone = BBox(
            self.danger_zone["x1"] * frame_w,
            self.danger_zone["y1"] * frame_h,
            self.danger_zone["x2"] * frame_w,
            self.danger_zone["y2"] * frame_h,
        )

        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        names = result.names

        masks_xy = None
        if getattr(result, "masks", None) is not None:
            masks_xy = result.masks.xy

        detections: list[ObstacleDetection] = []
        danger = False
        for index, values in enumerate(xyxy):
            bbox = BBox(*[float(v) for v in values]).clamp(frame_w, frame_h)
            label = str(names[int(classes[index])])
            confidence = float(confidences[index])

            if self.safety_only and label not in self.stop_labels:
                continue

            # The selected target is not treated as an obstacle, except person.
            if (
                target_bbox is not None
                and label != "person"
                and bbox.iou(target_bbox) >= self.exclude_target_iou
            ):
                continue

            cx, cy = bbox.center
            in_zone = zone.x1 <= cx <= zone.x2 and zone.y1 <= cy <= zone.y2
            is_danger = label in self.stop_labels and in_zone
            danger = danger or is_danger

            if self.safety_only and not is_danger and not self.report_clear_detections:
                continue

            contour = None
            if masks_xy is not None and index < len(masks_xy):
                polygon = np.asarray(masks_xy[index], dtype=np.int32)
                if polygon.ndim == 2 and polygon.shape[0] >= 3:
                    contour = polygon.reshape(-1, 1, 2)

            detections.append(
                ObstacleDetection(
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    contour=contour,
                    in_danger_zone=is_danger,
                )
            )

        if not detections:
            status = "no safety obstacles in danger zone"
        else:
            status = "danger" if danger else "clear by safety-only heuristic"
        self.last_observation = ObstacleObservation(
            detections=detections,
            danger=danger,
            status=status,
        )
        return self.last_observation
