from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


PARAMETER_LIMITS = {
    "s0": (500, 2500),
    "s1": (800, 1700),
    "s2": (1500, 2200),
}


class GraspModelError(RuntimeError):
    """The frozen grasp-model service rejected or could not process a request."""


@dataclass(frozen=True)
class ModelStatus:
    object_name: str
    num_samples: int
    hull_points: int
    test_mae: dict[str, Any]


class ArmTeachModelClient:
    """Read-only client for the existing 3090 arm-teach model on port 8022.

    This class exposes status and prediction only.  It deliberately has no BLE,
    chassis, servo, or execution method.
    """

    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_s) as response:
                value = json.loads(response.read().decode("utf-8", "replace"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = str(json.loads(detail).get("error") or detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise GraspModelError(f"模型服务拒绝请求 ({exc.code}): {detail}") from exc
        except (OSError, urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise GraspModelError(f"模型服务不可达或响应无效: {exc}") from exc
        if not isinstance(value, dict):
            raise GraspModelError("模型服务响应不是 JSON 对象")
        return value

    def status(self) -> ModelStatus:
        value = self._json("/api/model/status")
        if not value.get("ok") or not value.get("trained"):
            raise GraspModelError(str(value.get("error") or "模型尚未训练"))
        object_name = str(value.get("object_name") or "").strip()
        if not object_name:
            raise GraspModelError("模型状态缺少 object_name")
        return ModelStatus(
            object_name=object_name,
            num_samples=int(value.get("num_samples") or 0),
            hull_points=int(value.get("hull_points") or 0),
            test_mae=dict(value.get("test_mae") or {}),
        )

    @staticmethod
    def validate_parameters(parameters: Any) -> dict[str, int]:
        if not isinstance(parameters, dict):
            raise GraspModelError("预测响应缺少 parameters")
        checked: dict[str, int] = {}
        for name, (low, high) in PARAMETER_LIMITS.items():
            try:
                value = int(parameters[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise GraspModelError(f"预测参数 {name} 缺失或无效") from exc
            if not low <= value <= high:
                raise GraspModelError(
                    f"预测参数 {name}={value} 超出只读模型约定范围 [{low}, {high}]"
                )
            checked[name] = value
        return checked

    def predict(self, stitched_jpeg_b64: str, target: str) -> dict[str, Any]:
        status = self.status()
        requested = str(target).strip()
        if requested != status.object_name:
            raise GraspModelError(
                f"当前模型只训练了“{status.object_name}”，拒绝用于“{requested or '空目标'}”"
            )
        value = self._json(
            "/api/model_grasp/predict",
            {"image_b64": stitched_jpeg_b64, "target": requested},
        )
        if not value.get("ok"):
            raise GraspModelError(str(value.get("error") or "预测失败"))
        centers = value.get("centers")
        if not isinstance(centers, dict) or not all(k in centers for k in ("left", "right")):
            raise GraspModelError("预测结果没有双摄目标中心")
        return {
            "target": requested,
            "centers": centers,
            "parameters": self.validate_parameters(value.get("parameters")),
            "point": value.get("point"),
            "model": {
                "object_name": status.object_name,
                "num_samples": status.num_samples,
                "hull_points": status.hull_points,
                "test_mae": status.test_mae,
            },
        }

    def detect_centers(self, stitched_jpeg_b64: str, target: str) -> dict[str, Any]:
        """Run visual center detection only; never invoke the servo MLP or hardware."""
        requested = str(target).strip()
        value = self._json(
            "/api/teach/detect",
            {"image_b64": stitched_jpeg_b64, "target": requested},
        )
        if not value.get("ok"):
            raise GraspModelError(str(value.get("error") or "双摄中心检测失败"))
        centers = value.get("centers")
        if not isinstance(centers, dict) or not all(k in centers for k in ("left", "right")):
            raise GraspModelError("中心检测结果没有左右双摄坐标")
        return {"target": requested, "centers": centers}
