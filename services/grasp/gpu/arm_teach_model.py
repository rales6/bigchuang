"""机械臂示教夹取模型服务 (3090, 端口 8022, 独立系统)

独立于 arm_grasp_server(8020) / 队友的 arm-mapping 系统，自成一套：
  - 检测:   左右拼接帧 -> Qwen 双相机目标中心 (整幅 0..1000 约定, 已真机验证)
  - 区域:   左右各自二维凸包或25px外沿 -> 停车/抓取资格
  - 姿态:   4维中心 -> 原4D示教边界投影 -> MLP -> 三舵机脉宽
  - 执行由树莓派完成 (ASCII #ARMG 绝对脉宽指令, BLE 通道只认 ASCII)

端点:
  GET  /api/model/status            模型/凸包状态
  POST /api/teach/detect            {image_b64, target} -> {centers:{left:[x,y],right:[x,y]}}
  POST /api/model_grasp/predict     {image_b64, target} -> 舵机预测 + 凸包距离/入包标志
  双目各自二维凸包或25px外沿放行；原四维凸包用于姿态投影和诊断。

目标识别复用 arm_grasp_server(8020) 的分析端点，因此 8022 只加载轻量
位姿 MLP，不再额外加载第三份 Qwen；8020 同时使用生产版夹爪 YOLO。
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import sys
import threading
from urllib import error as urlerror
from urllib import request as urlrequest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from teaching_region import dual_view_region,GRASP_MARGIN_PX

ANALYSIS_URL = os.environ.get(
    "ARM_TEACH_ANALYSIS_URL",
    "http://127.0.0.1:8020/api/calibration/analyze",
)
ARTIFACT_DIR = Path(os.environ.get(
    "ARM_TEACH_ARTIFACTS", str(HERE / "arm_teach_artifacts")
))
MODEL_STATE_PATH = ARTIFACT_DIR / "model_state.json"
MODEL_WEIGHTS_PATH = ARTIFACT_DIR / "model.pt"

# 模型输出 = 三舵机绝对脉宽 (#ARMG{s0}{s1}{s2}! 指令空间)
# 归一化范围 = 固件各舵机脉宽限位 (s3 与 s2 镜像联动, 不训练)
PARAM_ORDER = ["s0", "s1", "s2"]
PARAM_RANGES = {
    "s0": (500.0, 2500.0),
    "s1": (800.0, 1700.0),
    "s2": (1500.0, 2200.0),
}
CAMERA_W, CAMERA_H = 640, 480

app = FastAPI(title="Arm Teach Model Service (8022)")
logger = logging.getLogger("arm_teach_model")

_artifacts_lock = threading.Lock()
_artifacts_cache: dict = {"mtime": -1.0, "state": None, "model": None}


def _normalized_target_box(item: dict, view_w: float, view_h: float, center) -> list | None:
    """Geometry from the SAME detection as the center; no second model call."""
    box=item.get('target_bbox')
    try:
        if isinstance(box,dict):
            box=[box[key] for key in ('x1','y1','x2','y2')]
        if not isinstance(box,(list,tuple)) or len(box)!=4:
            return None
        x1,y1,x2,y2=map(float,box)
        if (not all(math.isfinite(v) for v in (x1,y1,x2,y2,view_w,view_h))
                or not 0<=x1<x2<=view_w or not 0<=y1<y2<=view_h):
            return None
        normalized=[x1/view_w*CAMERA_W,y1/view_h*CAMERA_H,
                    x2/view_w*CAMERA_W,y2/view_h*CAMERA_H]
        if not (normalized[0]-2<=center[0]<=normalized[2]+2
                and normalized[1]-2<=center[1]<=normalized[3]+2):
            return None
        return [round(v,2) for v in normalized]
    except (KeyError,TypeError,ValueError,ZeroDivisionError):
        return None


def _detect_centers(image_b64: str, target: str, geometry: dict | None = None) -> dict:
    """让 8020 复用其 Qwen+生产 YOLO，并返回 640x480 标准中心。"""
    raw = base64.b64decode(str(image_b64 or ""))
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("invalid image")
    height, width = frame.shape[:2]
    if width < int(height * 1.8):
        raise ValueError("side-by-side image required")
    half_w = width // 2
    encoded: dict[str, str] = {}
    for camera, half in (
        ("left", frame[:, :half_w]),
        ("right", frame[:, half_w:]),
    ):
        ok, jpeg = cv2.imencode(".jpg", half, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"failed to encode {camera} frame")
        encoded[camera] = base64.b64encode(jpeg.tobytes()).decode("ascii")
    request_obj = urlrequest.Request(
        ANALYSIS_URL,
        data=json.dumps({
            "left_image_b64": encoded["left"],
            "right_image_b64": encoded["right"],
            "target_description": str(target or "红色小物块"),
            "use_qwen": True,
        }, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(request_obj, timeout=90.0) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"8020 analysis rejected request: {detail}") from exc
    if not isinstance(parsed, dict) or not parsed.get("ok"):
        raise RuntimeError(str(parsed.get("error") if isinstance(parsed, dict) else parsed))

    views = parsed.get("views") or {}
    centers: dict[str, list[float]] = {}
    for camera in ("left", "right"):
        item = views.get(camera) or {}
        center = item.get("target_center") or {}
        if not item.get("target_found") or not isinstance(center, dict):
            continue
        view_w = max(float(item.get("width") or half_w), 1.0)
        view_h = max(float(item.get("height") or height), 1.0)
        cx = float(center.get("x")) / view_w * CAMERA_W
        cy = float(center.get("y")) / view_h * CAMERA_H
        if -10.0 <= cx <= CAMERA_W + 10.0 and -10.0 <= cy <= CAMERA_H + 10.0:
            centers[camera] = [round(cx, 1), round(cy, 1)]
            if geometry is not None:
                box=_normalized_target_box(item,view_w,view_h,centers[camera])
                if box is not None:
                    geometry[camera]=box
    if "left" not in centers or "right" not in centers:
        raise RuntimeError("target not visible in both views")
    return centers


# ── 4D 凸包判断 (无 scipy: 投影梯度下降求凸组合) ─────────

def _project_to_hull(point: np.ndarray, hull: np.ndarray,
                     max_iter: int = 10000) -> tuple[np.ndarray, float]:
    """返回凸包中距离 point 最近的点及距离平方。

    解 min_λ 0.5*||Hλ - x||^2, s.t. λ >= 0, Σλ = 1 (凸组合可行性)。
    FISTA (加速近端梯度) + 精确单纯形投影。
    """
    h = torch.from_numpy(hull)
    x = torch.from_numpy(point)
    n = h.shape[0]
    gram = h @ h.T
    smoothness = max(torch.linalg.eigvalsh(gram)[-1].item(), 1e-6)
    step = 1.0 / smoothness

    def proj_simplex(v: torch.Tensor) -> torch.Tensor:
        """精确投影到 {λ: λ>=0, Σλ=1} (排序法, O(n log n))。"""
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(1, n + 1, dtype=torch.float64)
        cond = u - cssv / ind > 0
        rho = int(torch.nonzero(cond).max().item()) + 1 if cond.any() else 1
        theta = cssv[rho - 1] / float(rho)
        return torch.clamp(v - theta, min=0.0)

    lam = torch.full((n,), 1.0 / n, dtype=torch.float64)
    y = lam.clone()
    t = 1.0
    hx = h @ x
    last_projection = None
    stable_checks = 0
    for iteration in range(max_iter):
        gy = gram @ y - hx
        lam_new = proj_simplex(y - step * gy)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = lam_new + ((t - 1.0) / t_new) * (lam_new - lam)
        lam, t = lam_new, t_new
        # 凸包外的残差不可能趋近于零。旧实现因此每次都跑满 5 万轮，
        # 让近距反馈卡住几十秒；改为同时检测投影点是否已收敛。
        if iteration % 25 == 0:
            current_projection = h.T @ lam
            if float(((current_projection - x) ** 2).sum().item()) <= 1e-10:
                break
            if last_projection is not None and float(
                torch.max(torch.abs(current_projection - last_projection)).item()
            ) < 1e-9:
                stable_checks += 1
                if stable_checks >= 3:
                    break
            else:
                stable_checks = 0
            last_projection = current_projection
    projected = h.T @ lam
    residual = float(((projected - x) ** 2).sum().item())
    return projected.numpy(), residual


def _hull_min_residual(point: np.ndarray, hull: np.ndarray,
                       max_iter: int = 10000) -> float:
    """点到凸包的最小距离平方 (归一化坐标空间)。"""
    return _project_to_hull(point, hull, max_iter)[1]


def point_in_convex_hull(point: np.ndarray, hull: np.ndarray,
                         tol: float = 1e-8, max_iter: int = 10000) -> bool:
    """判断 point 是否在 hull 点集(每行一点)的凸包内 (残差 <= tol)。"""
    point = np.asarray(point, dtype=np.float64).reshape(-1)
    hull = np.asarray(hull, dtype=np.float64)
    if hull.ndim != 2 or hull.shape[0] == 0 or hull.shape[1] != point.shape[0]:
        return False
    if hull.shape[0] == 1:
        return bool(np.allclose(hull[0], point, atol=1e-4))
    return _hull_min_residual(point, hull, max_iter) <= tol


# ── MLP 模型与产物加载 ──────────────────────────────────

class ArmMLP(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _load_artifacts():
    """加载 model_state.json + model.pt, 按 mtime 缓存。"""
    with _artifacts_lock:
        try:
            mtime = MODEL_STATE_PATH.stat().st_mtime
        except OSError:
            raise RuntimeError("模型未训练: 先采集样本并运行 train_arm_mlp.py")
        if _artifacts_cache["mtime"] != mtime:
            state = json.loads(MODEL_STATE_PATH.read_text(encoding="utf-8"))
            model = ArmMLP()
            model.load_state_dict(torch.load(
                MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=True,
            ))
            model.eval()
            _artifacts_cache.update({"mtime": mtime, "state": state, "model": model})
        return _artifacts_cache["state"], _artifacts_cache["model"]


def _predict_params(centers: dict) -> dict:
    """中心 -> 三舵机脉宽 s0/s1/s2。

    严格四维凸包标志保留；停车/抓取资格按两眼独立二维25px外沿。
    余量内也只用最近已示教边界推理，不用外推舵机姿态。
    """
    left, right = centers["left"], centers["right"]
    point = np.asarray([
        left[0] / CAMERA_W, left[1] / CAMERA_H,
        right[0] / CAMERA_W, right[1] / CAMERA_H,
    ], dtype=np.float64)
    state, model = _load_artifacts()
    hull = np.asarray(state["hull"], dtype=np.float64)
    region=dual_view_region(centers,hull.tolist(),CAMERA_W,CAMERA_H)
    projected_point, residual = _project_to_hull(point, hull)
    dist_px = math.sqrt(max(residual, 0.0)) * CAMERA_W
    inside_hull = residual <= 1e-8
    projected = not inside_hull
    effective_point = projected_point if projected else point
    if projected:
        logger.warning(
            "target outside teaching hull by %.1fpx; using nearest calibrated boundary; "
            "observed=%s effective=%s",
            dist_px,
            np.round(point, 5).tolist(),
            np.round(effective_point, 5).tolist(),
        )
    means = np.asarray(state["input_means"], dtype=np.float64)
    stds = np.asarray(state["input_stds"], dtype=np.float64)
    with torch.no_grad():
        x = torch.from_numpy(
            (effective_point - means) / np.maximum(stds, 1e-6)
        ).float().unsqueeze(0)
        raw = model(x).numpy()[0].astype(np.float64)          # 0..1 归一化
    parameters = {}
    for index, name in enumerate(PARAM_ORDER):
        low, high = PARAM_RANGES[name]
        value = float(np.clip(raw[index], 0.0, 1.0)) * (high - low) + low
        parameters[name] = int(round(value))
    result = {
        "parameters": parameters,
        "point": point.tolist(),
        "effective_point": effective_point.tolist(),
        "projected_to_teaching_boundary": projected,
        "outside_distance_px": round(dist_px, 1),
        "inside_teaching_hull": inside_hull,
        "joint_hull_distance_px": round(dist_px, 1),
        "nearest_hull_centers": {
            "left": [round(float(projected_point[0] * CAMERA_W), 1),
                     round(float(projected_point[1] * CAMERA_H), 1)],
            "right": [round(float(projected_point[2] * CAMERA_W), 1),
                      round(float(projected_point[3] * CAMERA_H), 1)],
        },
        "object_name": state.get("object_name", ""),
        "num_samples": int(state.get("num_samples", 0)),
    }
    result.update(region)
    return result


# ── HTTP 端点 ───────────────────────────────────────────

@app.get("/api/model/status")
async def model_status() -> JSONResponse:
    try:
        state, _model = _load_artifacts()
        info = {
            "ok": True,
            "trained": True,
            "object_name": state.get("object_name", ""),
            "num_samples": state.get("num_samples", 0),
            "hull_points": len(state.get("hull", [])),
            "test_mae": state.get("test_mae", {}),
        }
    except RuntimeError as exc:
        info = {"ok": True, "trained": False, "error": str(exc)}
    return JSONResponse(info)


@app.post("/api/teach/detect")
async def teach_detect(payload: dict) -> JSONResponse:
    try:
        centers = _detect_centers(
            str(payload.get("image_b64", "")),
            str(payload.get("target", "红色小物块")),
        )
        return JSONResponse({"ok": True, "centers": centers})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/model_grasp/predict")
async def model_predict(payload: dict) -> JSONResponse:
    try:
        boxes={}
        centers = _detect_centers(
            str(payload.get("image_b64", "")),
            str(payload.get("target", "红色小物块")),
            geometry=boxes,
        )
        result = _predict_params(centers)
        result.update({"ok": True, "centers": centers,"target_boxes":boxes})
        return JSONResponse(result)
    except ValueError as exc:
        # 凸包外: 业务性拒绝, 不是服务器错误
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8022)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
