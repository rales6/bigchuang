"""3090 侧 WebSocket 服务 — 视觉伺服抓取。

提供 /ws/grasp 端点，接收树莓派的摄像头帧，运行双目标跟踪 + 视觉伺服，
通过 WebSocket 将 arm_cmd 发回树莓派。
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

# 添加 3090 src 路径
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "3090" / "src"
if not SRC.exists():
    # 兼容 sjtu-server 部署路径
    for alt in [
        Path.home() / "3090_new_method" / "src",
        Path.home() / "3090" / "src",
        Path.home() / "3090_new_method" / "3090" / "src",
    ]:
        if alt.exists():
            SRC = alt; break
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.config import load_config
from qwen_grounded_tracker.grounding.qwen_grounder import Qwen3VLTargetGrounder
from qwen_grounded_tracker.grounding.worker import GroundingWorker
from qwen_grounded_tracker.perception.yolo_obstacles import YoloObstacleDetector

try:
    from .arm_calibration import ArmVisualCalibration, DEFAULT_CALIBRATION
    from .dual_tracker import DualTracker, DualTrackResult
    from .visual_servo import ServoCommand, VisualServoController
    from .grasp_sequence import GraspSequence, Phase
    from .structured_log import EventLogger
    from .automation_logic import (
        INITIAL_DEPLOY_SEQUENCE,
        autonomous_motion_ready,
        choose_camera_from_qwen,
        dual_camera_grasp_ready,
        grasp_motion_verified,
        relative_target_state,
        select_requested_object,
        stored_tracking_boxes,
    )
except ImportError:
    # 兼容部署机直接执行 arm_grasp_server.py。
    from arm_calibration import ArmVisualCalibration, DEFAULT_CALIBRATION
    from dual_tracker import DualTracker, DualTrackResult
    from visual_servo import ServoCommand, VisualServoController
    from grasp_sequence import GraspSequence, Phase
    from structured_log import EventLogger
    from automation_logic import (
        INITIAL_DEPLOY_SEQUENCE,
        autonomous_motion_ready,
        choose_camera_from_qwen,
        dual_camera_grasp_ready,
        grasp_motion_verified,
        relative_target_state,
        select_requested_object,
        stored_tracking_boxes,
    )


app = FastAPI(title="Arm Visual Servo Grasp Server")
EVENT_LOG = EventLogger("3090", "grasp_server.jsonl")
_LABEL_SESSION = None
_LABEL_LOCK = threading.Lock()
_TRAIN_STATE = {"active": False, "status": "idle", "log": [], "result": "", "error": ""}
_GRIPPER_YOLO = None
_GRIPPER_YOLO_PATH = ""


# ── Qwen 抓取 prompt ─────────────────────────────────────

GRASP_QWEN_PROMPT = """你是机械臂抓取视觉系统，负责识别画面中可抓取的目标物体。

画面中可能包含:
- 机械臂爪尖贴有绿色标记 (不用管它)
- 待抓取的目标物体

请找出画面中最适合抓取的物体。用户想抓的目标通常是画面中位置明确、大小合适的物体。

规则:
- 返回至少一个可抓取物体的 bbox，如果没有适合抓取的物体，设置 objects_found=false
- 优先选择画面中位置明确、大小合适（不太小不太大）的物体
- bbox 使用 0-1000 标准化坐标

返回纯 JSON:
{
  "claw_found": false,
  "claw_bbox": null,
  "objects_found": true,
  "objects": [
    {
      "name": "瓶子",
      "bbox": [x1, y1, x2, y2],
      "camera_id": "left",
      "pickable": true,
      "confidence": 0.85,
      "reason": "位置明确，大小合适"
    }
  ],
  "scene_description": "目标物在前方偏右"
}"""


def build_grasp_prompt(target_description: str) -> str:
    target = str(target_description or "").strip()
    if not target:
        return GRASP_QWEN_PROMPT
    return GRASP_QWEN_PROMPT + f"""

用户明确要求抓取的目标是：{target}
- 只能选择与该描述匹配的物体，不能用其他更容易抓的物体替代。
- 每个 objects 元素额外返回 matches_request: true/false。
- 如果画面中没有匹配目标，objects_found=false，objects=[]。
"""


def build_camera_selection_prompt(target_description: str) -> str:
    return f"""Camera selection only. The image is side-by-side: LEFT camera on
the left half and RIGHT camera on the right half. Compare both views of the
requested target ({target_description}) after arm deployment. Locate the target
in every half where it is visible. Prefer the view where the requested target
and the black gripper are both visible, the target is clearer/larger/less
occluded, and there is enough nearby space for grasping. Return the normal
stereo left/right JSON fields with confidence. Do not use a gripper bbox to
initialize tracking and do not issue movement advice."""


# ── 图像编解码 ───────────────────────────────────────────

def build_dual_close_confirmation_prompt(target_description: str) -> str:
    return f"""Final grasp-close verification only. This is one side-by-side
image: LEFT camera is the left half and RIGHT camera is the right half. In
each half independently locate the requested target ({target_description}).
Do not confuse the black robot gripper, arm, shadow, cable, floor, or a
different object with the requested target. Coordinates must be 0..1000 and
relative to that camera half, not the full combined image. Return JSON only:
{{
  "left": {{"target_found": true, "target_bbox": [x1,y1,x2,y2], "confidence": 0.0}},
  "right": {{"target_found": true, "target_bbox": [x1,y1,x2,y2], "confidence": 0.0}}
}}
Use target_found=false and target_bbox=null when a half cannot genuinely see
the requested target. Do not issue movement advice."""

def _decode_jpeg(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpeg_b64(frame: np.ndarray, quality: int = 70) -> str:
    ok, enc = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("ascii")


# ── 共享 Qwen grounder (跨会话复用, 模型只加载一次) ──────

_SHARED_GROUNDER = None
_SHARED_GROUNDER_LOCK = threading.Lock()


class _SerializedGrounder:
    """串行化包装: 共享 grounder 同时只服务一个请求 (模型推理非线程安全)。"""

    def __init__(self, grounder) -> None:
        self._g = grounder
        self._lock = threading.Lock()

    def ground(self, *args, **kwargs):
        with self._lock:
            return self._g.ground(*args, **kwargs)


def _get_shared_grounder(qwen_cfg: dict):
    """获取共享 grounder 单例。模型加载耗时几十秒, 只做一次。"""
    global _SHARED_GROUNDER
    with _SHARED_GROUNDER_LOCK:
        if _SHARED_GROUNDER is None:
            model_path = str(qwen_cfg.get("model_path", ""))
            model_path = str(Path(model_path).expanduser())
            grounder = Qwen3VLTargetGrounder(
                model_path=model_path,
                device=str(qwen_cfg.get("device", "cuda")),
                local_files_only=bool(qwen_cfg.get("local_files_only", True)),
                max_new_tokens=int(qwen_cfg.get("max_new_tokens", 512)),
                max_time_seconds=float(qwen_cfg.get("max_time_seconds", 30.0)),
                min_visual_tokens=int(qwen_cfg.get("min_visual_tokens", 128)),
                max_visual_tokens=int(qwen_cfg.get("max_visual_tokens", 512)),
                minimum_box_area_ratio=float(qwen_cfg.get("minimum_box_area_ratio", 0.001)),
            )
            _SHARED_GROUNDER = _SerializedGrounder(grounder)
            print("[grasp] shared Qwen grounder loaded")
        return _SHARED_GROUNDER


# ── GraspSession ─────────────────────────────────────────

class GraspSession:
    """一次抓取任务的完整会话。"""

    def __init__(self, config: dict[str, Any],
                 calibration: ArmVisualCalibration | None = None) -> None:
        self.cal = calibration or DEFAULT_CALIBRATION
        self.cfg = config

        # Qwen 初始化 (共享单例)
        qwen_cfg = config.get("qwen", {})
        self.grounder = _get_shared_grounder(qwen_cfg)
        self.grounding_worker = GroundingWorker(self.grounder)

        # YOLO (可选)
        obstacle_cfg = config.get("obstacles", {})
        if obstacle_cfg.get("enabled", False):
            self.yolo = YoloObstacleDetector(
                enabled=True,
                model_path=str(obstacle_cfg.get("model_path", "")),
                device=str(obstacle_cfg.get("device", "0")),
                image_size=int(obstacle_cfg.get("image_size", 640)),
                confidence=float(obstacle_cfg.get("confidence", 0.25)),
                frame_interval=int(obstacle_cfg.get("frame_interval", 1)),
                stop_labels=list(obstacle_cfg.get("stop_labels", ["person"])),
            )
        else:
            self.yolo = None

        # 核心模块
        self.dual_tracker = DualTracker(self.cal, yolo_detector=self.yolo)
        self.servo = VisualServoController(self.cal)
        self.sequence = GraspSequence()
        # The active camera drives servo motion, but both camera views must
        # independently confirm the geometry before the final close by default.
        self._require_dual_close = bool(
            config.get("grasp", {}).get("require_dual_close", True)
        )

        # 状态
        self.frame_count: int = 0
        self.latest_overlay_b64: str = ""
        self.latest_status: dict[str, Any] = {}
        self._trained_gripper_enabled = (
            _production_gripper_pose_model_path().exists()
            or _production_gripper_model_path().exists()
        )
        self._pose_gripper_enabled = _production_gripper_pose_model_path().exists()
        self._claw_yolo_seen_now = False
        self._claw_yolo_stable_frames = 0
        self._claw_yolo_missed_frames = 0
        self._claw_keypoints = None
        self._jaw_two_stable_frames = 0
        self._jaw_reveal_frames = 0
        self._jaw_reveal_commands = 0
        self._jaw_reveal_lower_count = 0
        self._claw_loss_lower_count = 0

    def handle_start(self, target_description: str = "",
                     active_camera: str = "left",
                     available_cameras: list[str] | None = None) -> dict:
        """启动抓取任务。"""
        self.sequence = GraspSequence(target_description)
        self.sequence.start()
        self.dual_tracker.reset()
        self.servo.reset()
        self._qwen_submitted = False
        self._qwen_start_time = 0.0
        self._move_until = 0.0
        self._last_probe_t = 0.0
        self._motion_claw_bbox = None
        self._claw_yolo_seen_now = False
        self._claw_yolo_stable_frames = 0
        self._claw_yolo_missed_frames = 0
        self._claw_keypoints = None
        self._jaw_two_stable_frames = 0
        self._jaw_reveal_frames = 0
        self._jaw_reveal_commands = 0
        self._jaw_reveal_lower_count = 0
        self._claw_loss_lower_count = 0
        self._deploy_seq = None
        self._carry_verified = False
        self._grasp_reference = None
        self._lift_reference = None
        self._verification_samples = []
        self._grasp_retry_count = 0
        self._last_grasp_was_trial = False
        self._pending_grasp_cmd = None
        self._dual_confirm_mode = ""
        self._dual_confirm_submitted = False
        self._dual_confirm_started_at = 0.0
        self._dual_confirm_claws = (None, None)
        self._dual_confirm_block_until = 0.0
        self._camera_select_submitted = False
        self._last_logged_phase = ""
        self._last_logged_action = ""
        self._last_tracking_state = None
        cameras = [str(c) for c in (available_cameras or [active_camera])]
        self._available_cameras = list(dict.fromkeys(cameras)) or ["left"]
        self._active_camera = (active_camera if active_camera in self._available_cameras
                               else self._available_cameras[0])
        self._camera_switch_count = 0
        self._last_camera_switch_at = 0.0
        EVENT_LOG.event(
            "task_start", target=target_description,
            active_camera=self._active_camera,
            available_cameras=self._available_cameras,
        )
        return {"type": "ack", "message": "grasp sequence started",
                "phase": self.sequence.phase.value,
                "active_camera": self._active_camera}

    def _request_camera_switch(self, reason: str) -> dict | None:
        """Stop and let Qwen compare both cameras again after visual loss."""
        cameras = getattr(self, "_available_cameras", ["left"])
        current = getattr(self, "_active_camera", cameras[0])
        if len(cameras) < 2 or getattr(self, "_camera_switch_count", 0) >= 1:
            return None
        now = time.time()
        if now - getattr(self, "_last_camera_switch_at", 0.0) < 3.0:
            return None
        self._camera_switch_count += 1
        self._last_camera_switch_at = now
        self._move_until = 0.0
        self._motion_claw_bbox = None
        self._claw_yolo_seen_now = False
        self._claw_yolo_stable_frames = 0
        self._claw_yolo_missed_frames = 0
        self._claw_loss_lower_count = 0
        self.dual_tracker.reset()
        self.servo.reset()
        self._qwen_submitted = False
        self._camera_select_submitted = False
        self._qwen_start_time = 0.0
        self._last_probe_t = 0.0
        self.sequence.phase = Phase.CAMERA_SELECT
        self.sequence._phase_started_at = now
        EVENT_LOG.event(
            "camera_reselection_requested",
            current_camera=current,
            reason=reason,
            switch_count=self._camera_switch_count,
        )
        return {
            "action": "compare_cameras",
            "reason": reason,
        }

    def handle_frame(self, frame: np.ndarray) -> dict:
        """处理一帧摄像头画面，返回 armm_cmd 或状态。"""
        self.frame_count += 1
        # A close request first changes the Pi runner to a side-by-side frame.
        # Handle that composite before ordinary single-camera tracking.
        if getattr(self, "_pending_grasp_cmd", None) is not None:
            return self._handle_dual_close_confirmation(frame)

        # ── 运动+黑色 检测爪子 (所有阶段持续运行) ──────────
        # 相机静止、车不动 → 画面里移动过的黑色物体 = 爪子
        # A trained one-class YOLO model is authoritative when available.
        # Motion-dark detection remains a fallback and a physical-motion
        # verifier for installations that have not trained a model yet.
        gripper_observation = _detect_trained_gripper(frame)
        yolo_claw_bbox = (
            gripper_observation.bbox if gripper_observation is not None else None
        )
        if yolo_claw_bbox is not None:
            accepted = self._refresh_claw(frame, yolo_claw_bbox, trained_model=True)
            self._claw_yolo_seen_now = accepted
            self._claw_keypoints = (
                gripper_observation.keypoints if gripper_observation.pose else None
            )
            if accepted:
                self._claw_yolo_stable_frames += 1
                self._claw_yolo_missed_frames = 0
                self._motion_claw_bbox = yolo_claw_bbox
            else:
                self._claw_yolo_stable_frames = 0
                self._claw_yolo_missed_frames += 1
                self._motion_claw_bbox = None
                self._claw_keypoints = None
        elif self._trained_gripper_enabled:
            self._claw_yolo_seen_now = False
            self._claw_yolo_stable_frames = 0
            self._claw_yolo_missed_frames += 1
            self._motion_claw_bbox = None
            self._claw_keypoints = [] if self._pose_gripper_enabled else None
        else:
            self._claw_keypoints = None
            candidate = self.dual_tracker.detect_claw_motion_dark(frame)
            self._motion_claw_bbox = (
                candidate if candidate is not None and self._refresh_claw(frame, candidate)
                else None
            )

        # Motion acknowledgement gate shared by every phase. Camera frames are
        # still consumed above for claw motion detection, but no state may
        # advance until the issued arm command has had time to settle.
        if (
            self.sequence.phase != Phase.DEPLOY
            and time.time() < getattr(self, "_move_until", 0.0)
        ):
            tracking = None
            # 运动中仍逐帧推进两个 tracker。尤其在 LIFT 阶段，目标是否
            # 随夹爪共同移动正是抓取成功验证所需的证据。
            if self.dual_tracker.both_ready:
                tracking = self.dual_tracker.update(frame)
                if self._motion_claw_bbox is not None:
                    self._refresh_claw(
                        frame, self._motion_claw_bbox,
                        trained_model=self._trained_gripper_enabled,
                    )
                    tracking.claw_bbox = self._motion_claw_bbox
                    tracking.claw_ok = True
                    tracking.claw_lost_frames = 0
            return self._make_response(
                arm_cmd={"action": "idle", "reason": "arm moving; state held"},
                tracking=tracking,
            )

        # ── 根据序列阶段处理 ──────────────────────────────
        if self.sequence.phase == Phase.DEPLOY:
            return self._handle_deploy(frame)

        if self.sequence.phase == Phase.CAMERA_SELECT:
            return self._handle_camera_select(frame)

        if self.sequence.phase == Phase.QWEN_INIT:
            return self._handle_qwen_init(frame)

        if self.sequence.phase == Phase.CLAW_INIT:
            return self._handle_claw_init(frame)

        if self.sequence.phase in (Phase.ALIGN, Phase.APPROACH, Phase.FINE_ALIGN):
            return self._handle_servo_loop(frame)

        if self.sequence.phase == Phase.GRASP:
            return self._handle_grasp(frame)

        if self.sequence.phase == Phase.HOLD:
            return self._handle_hold(frame)

        if self.sequence.phase == Phase.LIFT:
            return self._handle_lift(frame)

        if self.sequence.phase == Phase.VERIFY:
            return self._handle_verify(frame)

        if self.sequence.phase == Phase.RETRACT:
            return self._handle_retract(frame)

        if self.sequence.phase == Phase.RELEASE:
            return self._handle_release(frame)

        if self.sequence.phase == Phase.CARRY:
            return self._make_response(
                arm_cmd={
                    "action": "carry",
                    "reason": "grasp verified; holding object until release command",
                },
            )

        if self.sequence.phase == Phase.DONE:
            return self._make_response(
                arm_cmd={"action": "done", "reason": "grasp complete"},
            )

        if self.sequence.phase == Phase.ERROR:
            return self._make_response(
                arm_cmd={
                    "action": "error",
                    "reason": self.sequence.status().error,
                    "emergency_stop": True,
                },
            )

        # IDLE
        return self._make_response(
            arm_cmd={"action": "idle", "reason": "waiting for command"},
        )

    # ── 各阶段处理 ────────────────────────────────────────

    def _refresh_claw(self, frame: np.ndarray, bbox, trained_model: bool = False) -> bool:
        """用检测到的 bbox 刷新爪子跟踪状态 (必要时重建 CSRT)。"""
        claw = self.dual_tracker.claw
        if not self.dual_tracker._validate_bbox(
            bbox, frame, is_claw=True,
            last_size=(claw.last_size if claw.initialized else (0, 0)),
            trained_claw=trained_model,
        ):
            EVENT_LOG.throttled(
                "claw_candidate_rejected", 1.0, "claw_candidate_rejected",
                bbox=_bbox_dict(bbox), previous_size=claw.last_size,
            )
            return False
        need_reinit = (not claw.initialized or claw.tracker is None
                       or claw.lost_frames > 5)
        claw.bbox = bbox
        claw.initialized = True
        claw.lost_frames = 0
        claw.quality_score = min(1.0, claw.quality_score + 0.1)
        claw.last_seen_at = time.time()
        claw.last_size = (bbox.width, bbox.height)
        if need_reinit:
            try:
                claw.tracker = self.dual_tracker._create_csrt()
                claw.tracker.init(frame, bbox.to_xywh())
            except Exception:
                pass
        return True

    def _handle_deploy(self, frame: np.ndarray) -> dict:
        """From the home pose, execute the web UI's D20 command five times."""
        deploy_seq = getattr(self, '_deploy_seq', None)
        if deploy_seq is None:
            # 首次进入：初始化序列
            self._deploy_seq = list(INITIAL_DEPLOY_SEQUENCE)
            self._deploy_idx = 0
            self._deploy_sent = False
            self._deploy_start_time = time.time()
            deploy_seq = self._deploy_seq

        idx = self._deploy_idx
        elapsed = time.time() - self._deploy_start_time

        if idx < len(deploy_seq):
            label, fwd, up, dur = deploy_seq[idx]
            if not self._deploy_sent:
                self._deploy_sent = True
                self._deploy_start_time = time.time()
                cmd = ServoCommand(action="move", forward_mm=fwd, up_mm=up,
                    left_deg=0, claw_open=-1, duration_ms=dur,
                    state="deploying", reason=label)
                return self._make_response(arm_cmd=cmd)
            wait_s = float(dur) / 1000.0 + 0.8
            if elapsed < wait_s:
                return self._make_response(arm_cmd={"action": "idle",
                    "reason": f"{label} | wait {elapsed:.1f}/{wait_s:.1f}s"})
            # 进入下一步
            self._deploy_idx = idx + 1
            self._deploy_sent = False
            self._deploy_start_time = time.time()
            return self._make_response(arm_cmd={"action": "idle",
                "reason": f"{label} done, moving to step {idx+2}/{len(deploy_seq)}"})

        # Always finish all five motions. Camera selection must happen only
        # after deployment, with both views compared by Qwen.
        self.sequence.on_deploy_done()
        self._camera_select_submitted = False
        return self._make_response(arm_cmd={
            "action": "compare_cameras",
            "reason": "five D20 deploy motions complete; send both camera views",
        })

    def _handle_camera_select(self, frame: np.ndarray) -> dict:
        """Ask Qwen to compare a labelled LEFT|RIGHT composite and lock one view."""
        if not self.grounding_worker.busy and not self._camera_select_submitted:
            self.grounding_worker.submit(
                frame=frame,
                instruction=build_camera_selection_prompt(
                    self.sequence.target_description,
                ),
                reference_image=None,
            )
            self._camera_select_submitted = True
            self._qwen_start_time = time.time()

        try:
            response = self.grounding_worker.poll()
        except Exception as exc:
            response = None
            EVENT_LOG.event("camera_selection_error", error=repr(exc))

        if response is None:
            elapsed = time.time() - getattr(self, "_qwen_start_time", time.time())
            if elapsed > 60.0:
                self._camera_select_submitted = False
                self._qwen_start_time = time.time()
            return self._make_response(arm_cmd={
                "action": "compare_cameras",
                "reason": f"Qwen comparing left/right views... {elapsed:.0f}s",
            })

        raw = response.result.raw_text
        try:
            parsed = _parse_qwen_json(raw)
            selected = choose_camera_from_qwen(
                parsed,
                default=getattr(self, "_active_camera", "left"),
            )
        except Exception as exc:
            EVENT_LOG.event(
                "camera_selection_parse_error", error=repr(exc), raw=raw[:4000],
            )
            selected = getattr(self, "_active_camera", "left")

        if selected not in getattr(self, "_available_cameras", [selected]):
            selected = self._available_cameras[0]
        old_camera = self._active_camera
        self._active_camera = selected
        self.dual_tracker.reset()
        self.servo.reset()
        self._motion_claw_bbox = None
        self._claw_yolo_seen_now = False
        self._claw_yolo_stable_frames = 0
        self._claw_yolo_missed_frames = 0
        self._qwen_submitted = False
        self._camera_select_submitted = False
        self._last_probe_t = 0.0
        self.sequence.phase = Phase.QWEN_INIT
        self.sequence._phase_started_at = time.time()
        EVENT_LOG.event(
            "camera_selected_by_qwen",
            old_camera=old_camera,
            selected_camera=selected,
            raw=raw[:4000],
        )
        return self._make_response(arm_cmd={
            "action": "switch_camera",
            "camera": selected,
            "reason": "Qwen selected the better grasp view",
        })

    def _cancel_dual_close(self, reason: str) -> dict:
        camera = getattr(self, "_active_camera", "left")
        self._pending_grasp_cmd = None
        self._dual_confirm_mode = ""
        self._dual_confirm_submitted = False
        self._dual_confirm_started_at = 0.0
        self._dual_confirm_claws = (None, None)
        self._dual_confirm_block_until = time.time() + 1.2
        self.sequence.retry_after_failed_grasp(
            f"dual-camera close confirmation deferred: {reason}"
        )
        self.servo.prepare_grasp_retry()
        EVENT_LOG.event("dual_close_deferred", camera=camera, reason=reason)
        return self._make_response(arm_cmd={
            "action": "switch_camera",
            "camera": camera,
            "reason": f"dual-camera close deferred: {reason}",
        })

    def _handle_dual_close_confirmation(self, frame: np.ndarray) -> dict:
        """Confirm final close from both cameras, then return to the servo view."""
        mode = getattr(self, "_dual_confirm_mode", "compare")
        camera = getattr(self, "_active_camera", "left")

        if mode == "issue":
            result = self.dual_tracker.update(frame)
            observation = _detect_trained_gripper(frame)
            claw = observation.bbox if observation is not None else None
            accepted = (
                claw is not None
                and self._refresh_claw(frame, claw, trained_model=True)
            )
            visible_count = sum(
                point.get("v", 0) > 0 for point in (observation.keypoints or [])
            ) if observation is not None and observation.pose else 2
            if not accepted or not result.target_ok or visible_count < 2:
                return self._cancel_dual_close("active view changed before close")
            self._claw_keypoints = observation.keypoints if observation.pose else None
            result.claw_bbox = claw
            result.claw_ok = True
            cmd = self._pending_grasp_cmd
            self._pending_grasp_cmd = None
            self._dual_confirm_mode = ""
            self._dual_confirm_submitted = False
            self.sequence.grasp_completed()
            EVENT_LOG.event(
                "dual_close_authorized", camera=camera,
                grasp_state=getattr(cmd, "state", ""),
            )
            return self._make_response(arm_cmd=cmd, tracking=result)

        h, w = frame.shape[:2]
        if w < int(h * 1.8):
            return self._make_response(arm_cmd={
                "action": "compare_cameras",
                "reason": "waiting for side-by-side final confirmation frame",
            })
        half_w = w // 2
        if not self.grounding_worker.busy and not self._dual_confirm_submitted:
            left_frame = frame[:, :half_w]
            right_frame = frame[:, half_w:]
            self._dual_confirm_claws = (
                _detect_trained_gripper(left_frame),
                _detect_trained_gripper(right_frame),
            )
            self.grounding_worker.submit(
                frame=frame,
                instruction=build_dual_close_confirmation_prompt(
                    self.sequence.target_description,
                ),
                reference_image=None,
            )
            self._dual_confirm_submitted = True
            self._dual_confirm_started_at = time.time()

        left_claw, right_claw = self._dual_confirm_claws
        try:
            response = self.grounding_worker.poll()
        except Exception as exc:
            EVENT_LOG.event("dual_close_qwen_error", error=repr(exc))
            return self._cancel_dual_close(f"Qwen error: {exc}")
        if response is None:
            elapsed = time.time() - self._dual_confirm_started_at
            if elapsed > 60.0:
                return self._cancel_dual_close("Qwen confirmation timed out")
            return self._make_response(arm_cmd={
                "action": "compare_cameras",
                "reason": f"checking both cameras before close... {elapsed:.0f}s",
            })

        raw = response.result.raw_text
        try:
            parsed = _parse_qwen_json(raw)
            ready, reason, details = dual_camera_grasp_ready(
                parsed,
                left_claw=left_claw.bbox if left_claw is not None else None,
                right_claw=right_claw.bbox if right_claw is not None else None,
                frame_width=half_w,
                frame_height=h,
                left_keypoints=(
                    left_claw.keypoints if left_claw is not None and left_claw.pose else None
                ),
                right_keypoints=(
                    right_claw.keypoints if right_claw is not None and right_claw.pose else None
                ),
            )
        except Exception as exc:
            EVENT_LOG.event(
                "dual_close_parse_error", error=repr(exc), raw=raw[:4000],
            )
            return self._cancel_dual_close(f"confirmation parse error: {exc}")
        EVENT_LOG.event(
            "dual_close_result", ready=ready, reason=reason,
            details=details, raw=raw[:4000],
        )
        if not ready:
            return self._cancel_dual_close(reason)

        self._dual_confirm_mode = "issue"
        return self._make_response(arm_cmd={
            "action": "switch_camera",
            "camera": camera,
            "reason": "both cameras confirmed; returning to active view to close",
        })

    def _handle_qwen_init(self, frame: np.ndarray) -> dict:
        """初始化识别。

        爪子: 运动+黑色检测 (handle_frame 持续运行) + 摆动探测
        目标: Qwen 识别 (唯一用途)
        """
        # ① 爪子未找到 → 摆动探测 (机械臂动几下, 运动检测锁定黑色爪子)
        # A previously grounded target is advanced on the current frame while
        # YOLO accumulates its stability gate.  This avoids asking Qwen again
        # just because the claw needed two more frames to become stable.
        if self.dual_tracker.target.initialized:
            current = self.dual_tracker.update(frame)
            if self._motion_claw_bbox is not None:
                self._refresh_claw(
                        frame, self._motion_claw_bbox,
                        trained_model=self._trained_gripper_enabled,
                    )
                current.claw_bbox = self._motion_claw_bbox
                current.claw_ok = self._claw_yolo_seen_now
            if autonomous_motion_ready(
                claw_detected_now=self._claw_yolo_seen_now,
                claw_stable_frames=self._claw_yolo_stable_frames,
                target_ok=current.target_ok,
            ):
                self._camera_switch_count = 0
                self.sequence.on_qwen_done(
                    True, True, self.sequence.target_description,
                )
                return self._make_response(
                    arm_cmd={"action": "idle", "reason": "claw and target locked"},
                    tracking=current,
                )
            if not current.target_ok:
                self.dual_tracker.target.reset()

        # Do not move merely to discover the claw.  The production model must
        # establish a stable lock in the selected camera while Qwen finds the
        # requested target.

        # ② Qwen 找目标 (只提交一次)
        if not self.grounding_worker.busy and not getattr(self, '_qwen_submitted', False):
            self.grounding_worker.submit(
                frame=frame,
                instruction=build_grasp_prompt(self.sequence.target_description),
                reference_image=None,
            )
            self._qwen_submitted = True
            self._qwen_start_time = time.time()

        # 检查 Qwen 是否完成
        try:
            from qwen_grounded_tracker.domain import BBox
            response = self.grounding_worker.poll()
        except Exception as exc:
            response = None
            print(f"[grasp qwen error] {exc}")

        if response is None:
            # Qwen 仍在运行; 检查超时 (60s) → 重置并重试 (不关共享 grounder)
            elapsed = time.time() - getattr(self, '_qwen_start_time', time.time())
            if elapsed > 60.0:
                self._qwen_submitted = False
                self._qwen_start_time = time.time()
            return self._make_response(
                arm_cmd={"action": "idle",
                         "reason": f"Qwen identifying target... {elapsed:.0f}s",
                         "qwen_elapsed_s": self.grounding_worker.elapsed_seconds},
            )

        # 解析 Qwen 结果
        result = response.result
        raw = result.raw_text
        target_found = False
        target_name = ""

        try:
            # 尝试从 raw_text 中解析 JSON
            parsed = _parse_qwen_json(raw)
            EVENT_LOG.event(
                "qwen_result",
                target=self.sequence.target_description,
                raw=raw[:4000],
                object_count=len(parsed.get("objects", [])),
            )
            objects_found = parsed.get("objects_found", False)

            if objects_found and parsed.get("objects"):
                objs = parsed["objects"]
                best = select_requested_object(
                    objs, self.sequence.target_description,
                )
                if best is not None:
                    if best.get("bbox"):
                        tb = best["bbox"]
                        h, w = frame.shape[:2]
                        target_bbox = BBox(
                            tb[0] / 1000.0 * w, tb[1] / 1000.0 * h,
                            tb[2] / 1000.0 * w, tb[3] / 1000.0 * h,
                        )
                        target_found = self.dual_tracker.initialize_target(
                            frame, target_bbox,
                        )
                        if target_found:
                            target_name = best.get("name", "object")
                            EVENT_LOG.event(
                                "target_selected",
                                requested=self.sequence.target_description,
                                selected=target_name,
                                bbox=_bbox_dict(target_bbox),
                            )
        except Exception as exc:
            print(f"[grasp qwen parse error] {exc}")
            EVENT_LOG.event("qwen_parse_error", error=repr(exc), raw=raw[:4000])

        # 当前相机看不到指定目标时，先在静止状态下切换另一相机重识别。
        if not target_found:
            switch = self._request_camera_switch(
                "requested target not found; trying other camera"
            )
            if switch is not None:
                return self._make_response(arm_cmd=switch)

        # 完成条件: 爪子(运动检测) + 目标(Qwen)
        claw_stable = autonomous_motion_ready(
            claw_detected_now=self._claw_yolo_seen_now,
            claw_stable_frames=self._claw_yolo_stable_frames,
            target_ok=target_found,
        )
        if target_found and claw_stable:
            # A successful lock completes this search round. A later genuine
            # loss may use the other camera again as a fresh fallback round.
            self._camera_switch_count = 0
            self.sequence.on_qwen_done(True, True, target_name)
        elif target_found:
            # 有目标无爪子：保持 QWEN_INIT，继续运动探测；不能误转到
            # CLAW_INIT（该阶段的前提恰好是已经找到爪子）。
            self._qwen_submitted = False
        else:
            # 无目标 → CLAW_INIT 阶段每10s重试 Qwen
            self.sequence.on_qwen_done(self.dual_tracker.claw_ready, False, "")

        return self._make_response(
            arm_cmd={"action": "idle",
                     "reason": "init: claw={}, target={}".format(
                         claw_stable, target_found,
                     )},
        )

    def _handle_claw_init(self, frame: np.ndarray) -> dict:
        """爪子已找到(绿斑), 等待目标物。

        每 10s 用 Qwen 重试找目标; 找到后进入 ALIGN。
        """
        # 用 YOLO 尝试找到可抓取物体
        if self.yolo is not None:
            try:
                obs = self.yolo.detect(frame)
                stop_labels = getattr(self.yolo, "stop_labels", set())
                candidates = [d for d in obs.detections if d.label not in stop_labels]
                selected = select_requested_object([
                    {
                        "name": d.label,
                        "confidence": d.confidence,
                        "pickable": True,
                        "_detection": d,
                    }
                    for d in candidates
                ], self.sequence.target_description)
                if selected is not None:
                    best = selected["_detection"]
                    if not self.dual_tracker.initialize_target(frame, best.bbox):
                        raise ValueError("YOLO target bbox rejected by tracker")
                    self.sequence.on_target_found(best.label)
                    return self._make_response(
                        arm_cmd={"action": "idle", "reason": f"target: {best.label}"})
            except Exception:
                pass

        # 定期用 Qwen 重试找目标
        last_try = getattr(self, '_claw_init_qwen_t', 0.0)
        now = time.time()
        if (now - last_try > 10.0
                and not self.grounding_worker.busy
                and not getattr(self, '_qwen_submitted', False)):
            self._qwen_submitted = True
            self._qwen_start_time = time.time()
            self.grounding_worker.submit(
                frame=frame,
                instruction=build_grasp_prompt(self.sequence.target_description),
                reference_image=None,
            )
            self._claw_init_qwen_t = now

        # 检查 Qwen 结果
        try:
            response = self.grounding_worker.poll()
            if response is not None:
                self._qwen_submitted = False
                self._claw_init_qwen_t = now
                parsed = _parse_qwen_json(response.result.raw_text)
                objs = parsed.get("objects", []) if parsed.get("objects_found") else []
                best = select_requested_object(
                    objs, self.sequence.target_description,
                )
                if best is not None:
                    if best.get("bbox"):
                        tb = best["bbox"]
                        h, w = frame.shape[:2]
                        target_bbox = BBox(
                            tb[0] / 1000.0 * w, tb[1] / 1000.0 * h,
                            tb[2] / 1000.0 * w, tb[3] / 1000.0 * h,
                        )
                        if self.dual_tracker.initialize_target(frame, target_bbox):
                            self.sequence.on_target_found(best.get("name", "object"))
                            return self._make_response(
                                arm_cmd={"action": "idle",
                                         "reason": f"target found: {best.get('name', 'object')}"})
        except Exception:
            pass

        return self._make_response(
            arm_cmd={"action": "idle",
                     "reason": "claw OK (green), waiting for target..."},
        )

    def _handle_servo_loop(self, frame: np.ndarray) -> dict:
        """视觉伺服主循环。

        顺序: 运动检测(handle_frame已做) → 移动门控 → CSRT+暗色校验
              → 丢失回退/摆动探测 → 安全区 → 伺服。
        """
        # ① 更新双目标跟踪 (爪子已被 handle_frame 的运动检测刷新)
        result = self.dual_tracker.update(frame)

        # 运动检测锁定 → 爪子位置以运动检测为准 (权威来源)
        if self._motion_claw_bbox is not None:
            result.claw_bbox = self._motion_claw_bbox
            result.claw_ok = True
            result.claw_lost_frames = 0

        # A CSRT prediction from an older frame never authorizes motion when a
        # promoted YOLO model is installed.
        if self._trained_gripper_enabled:
            result.claw_ok = bool(self._claw_yolo_seen_now)
            if self._claw_yolo_seen_now:
                result.claw_bbox = self._motion_claw_bbox
                result.claw_lost_frames = 0
            else:
                result.claw_lost_frames = self._claw_yolo_missed_frames

        # ② 移动门控: 机械臂仍在运动 → 不发新命令 (等停稳)
        if time.time() < getattr(self, "_move_until", 0.0):
            return self._make_response(
                arm_cmd={"action": "idle",
                         "reason": "arm moving, waiting to settle..."},
                tracking=result,
            )

        # ③ 爪子丢失 → 先回退, 再摆动探测
        if not result.claw_ok:
            # A small downward recovery often brings an arm that slipped above
            # the image boundary back into view. Do this only once per loss
            # episode; repeated descent without vision would be unsafe.
            if (self._claw_yolo_missed_frames >= 2
                    and self._claw_loss_lower_count < 1):
                self._claw_loss_lower_count += 1
                EVENT_LOG.event(
                    "claw_loss_lower_recovery",
                    missed_frames=self._claw_yolo_missed_frames,
                    lower_count=self._claw_loss_lower_count,
                )
                return self._make_response(
                    arm_cmd=ServoCommand(
                        action="move", forward_mm=0, up_mm=-1, left_deg=0,
                        claw_open=-1, duration_ms=900, state="recovering",
                        reason="claw lost; lower 1mm once to recover the view",
                    ),
                    tracking=result,
                )
            if self._claw_yolo_missed_frames >= 8:
                switch = self._request_camera_switch(
                    "claw left current view; trying other camera"
                )
                if switch is not None:
                    return self._make_response(arm_cmd=switch, tracking=result)
                self._qwen_submitted = False
                self.sequence.phase = Phase.QWEN_INIT
                return self._make_response(
                    arm_cmd={"action": "reground",
                             "reason": "YOLO claw lost; stationary re-grounding"},
                    tracking=result,
                )
            return self._make_response(
                arm_cmd={"action": "idle",
                         "reason": f"YOLO claw missing {result.claw_lost_frames} frames; motion blocked"},
                tracking=result,
            )

        # 爪子 OK → 清除回退计数和本轮丢失下降限制
        self._claw_loss_lower_count = 0
        self.servo.clear_rollback()

        # Target loss is a hard motion gate. Never approach a stale bbox.
        if not result.target_ok or result.needs_reground:
            switch = self._request_camera_switch(
                "target tracking lost; trying other camera"
            )
            if switch is not None:
                return self._make_response(arm_cmd=switch, tracking=result)
            self._qwen_submitted = False
            self.sequence.phase = Phase.QWEN_INIT
            return self._make_response(
                arm_cmd={"action": "reground", "reason": "target tracking lost"},
                tracking=result,
            )

        # Re-apply the same hard gate before every servo decision.  A single
        # missed YOLO frame resets stability, so motion resumes only after the
        # claw and target have again been valid together for three frames.
        if not autonomous_motion_ready(
            claw_detected_now=self._claw_yolo_seen_now,
            claw_stable_frames=self._claw_yolo_stable_frames,
            target_ok=result.target_ok,
        ):
            return self._make_response(
                arm_cmd={
                    "action": "idle",
                    "reason": (
                        "waiting for 3 stable YOLO claw frames with current target "
                        f"({self._claw_yolo_stable_frames}/3)"
                    ),
                },
                tracking=result,
            )

        # ④ 暗色校验: CSRT 跟踪漂移到非黑色区域 → 视为丢失
        # A Pose detection with fewer than two visible jaws is still a valid
        # claw lock, but it cannot define the opening midpoint. First lower 1mm
        # once to recover the jaw points, then use yaw-only probes. Never combine
        # vertical and yaw recovery in the same command.
        if self._claw_keypoints is not None:
            visible_points = [
                point for point in self._claw_keypoints
                if float(point.get("v", point.get("confidence", 0.0)) or 0.0) > 0
            ]
            if len(visible_points) < 2:
                self._jaw_two_stable_frames = 0
                self._jaw_reveal_frames += 1
                if self._jaw_reveal_frames < 2:
                    return self._make_response(
                        arm_cmd={
                            "action": "idle",
                            "reason": (
                                f"only {len(visible_points)} jaw point visible; "
                                "confirming before yaw-only reveal"
                            ),
                        },
                        tracking=result,
                    )
                self._jaw_reveal_frames = 0
                if self._jaw_reveal_lower_count == 0:
                    self._jaw_reveal_lower_count = 1
                    EVENT_LOG.event(
                        "jaw_points_loss_lower",
                        visible_points=len(visible_points),
                        attempt=self._jaw_reveal_lower_count,
                    )
                    return self._make_response(
                        arm_cmd=ServoCommand(
                            action="move", forward_mm=0, up_mm=-1,
                            left_deg=0, duration_ms=700,
                            state="recover_jaw_points_lower",
                            reason=(
                                f"only {len(visible_points)} jaw point visible; "
                                "lower 1mm once before yaw recovery"
                            ),
                        ),
                        tracking=result,
                    )
                if (self._jaw_reveal_commands > 0
                        and self._jaw_reveal_commands % 4 == 0
                        and self._jaw_reveal_lower_count < 2):
                    self._jaw_reveal_lower_count += 1
                    EVENT_LOG.event(
                        "jaw_reveal_lower",
                        visible_points=len(visible_points),
                        attempt=self._jaw_reveal_lower_count,
                    )
                    return self._make_response(
                        arm_cmd=ServoCommand(
                            action="move", forward_mm=0, up_mm=-1,
                            left_deg=0, duration_ms=700,
                            state="reveal_jaws_lower",
                            reason=(
                                "second jaw still hidden after four 1deg probes; "
                                f"lower 1mm ({self._jaw_reveal_lower_count}/2) "
                                "then continue yaw-only confirmation"
                            ),
                        ),
                        tracking=result,
                    )
                if visible_points:
                    base_direction = (
                        -1 if float(visible_points[0]["x"]) < frame.shape[1] * 0.5 else 1
                    )
                else:
                    base_direction = 1
                direction = (
                    base_direction
                    if self._jaw_reveal_commands % 2 == 0
                    else -base_direction
                )
                self._jaw_reveal_commands += 1
                EVENT_LOG.event(
                    "jaw_reveal_move",
                    visible_points=len(visible_points),
                    left_deg=direction,
                    attempt=self._jaw_reveal_commands,
                )
                return self._make_response(
                    arm_cmd=ServoCommand(
                        action="move", forward_mm=0, up_mm=0,
                        left_deg=direction, duration_ms=650,
                        state="reveal_jaws",
                        reason=(
                            f"only {len(visible_points)} jaw point visible; "
                            f"yaw-only reveal {direction:+d}deg"
                        ),
                    ),
                    tracking=result,
                )
            self._jaw_reveal_frames = 0
            self._jaw_two_stable_frames += 1
            if self._jaw_two_stable_frames < 2:
                return self._make_response(
                    arm_cmd={
                        "action": "idle",
                        "reason": "both jaw points found; confirming one more frame",
                    },
                    tracking=result,
                )
            self._jaw_reveal_commands = 0
            self._jaw_reveal_lower_count = 0
        if (not self._trained_gripper_enabled
                and not self.dual_tracker.is_dark_region(frame, result.claw_bbox)):
            result.claw_ok = False
            return self._make_response(
                arm_cmd={"action": "idle",
                         "reason": "legacy claw tracking drifted"},
                tracking=result,
            )

        # ⑤ 安全区检查: 爪尖贴边 → 先拉回中央
        if not self.servo.claw_in_safe_zone(result.claw_bbox, frame.shape):
            pull = self.servo.pull_back_cmd(result.claw_bbox, frame.shape)
            return self._make_response(arm_cmd=pull, tracking=result)

        # ⑥ 正常伺服
        servo_cmd = self.servo.compute(
            result.claw_bbox, result.target_bbox,
            frame.shape, dual_result=result,
            claw_keypoints=self._claw_keypoints,
        )

        # 更新序列状态
        phase_before_servo = self.sequence.phase
        advance = self.sequence.update_from_servo(servo_cmd)
        # Physical axes have separate small-step budgets. Keep only the global
        # budget when switching between horizontal, depth and vertical phases.
        if self.sequence.phase != phase_before_servo and advance != "error":
            self.servo.reset_step_counter()

        if advance == "advance":
            # 阶段切换 (对齐→抓取)
            self.servo.clear_motion_history()
            self._grasp_reference = relative_target_state(
                result.claw_bbox, result.target_bbox,
            )
            self._last_grasp_was_trial = servo_cmd.state == "trial_grasp"
            # Optionally defer the close for a calibrated stereo installation.
            # Normal operation trusts the Qwen-selected active camera.
            if (self._require_dual_close
                    and len(getattr(self, "_available_cameras", [])) >= 2):
                if time.time() < getattr(self, "_dual_confirm_block_until", 0.0):
                    self.sequence.retry_after_failed_grasp(
                        "waiting before dual-camera confirmation retry"
                    )
                    return self._make_response(
                        arm_cmd={
                            "action": "idle",
                            "reason": "waiting for a fresh frame before dual-camera retry",
                        },
                        tracking=result,
                    )
                self._pending_grasp_cmd = servo_cmd
                self._dual_confirm_mode = "compare"
                self._dual_confirm_submitted = False
                self._dual_confirm_started_at = 0.0
                self._dual_confirm_claws = (None, None)
                EVENT_LOG.event(
                    "dual_close_requested", camera=self._active_camera,
                    grasp_state=servo_cmd.state,
                )
                return self._make_response(
                    arm_cmd={
                        "action": "compare_cameras",
                        "reason": "servo ready; confirming close with both cameras",
                    },
                    tracking=result,
                )
            EVENT_LOG.event(
                "single_camera_close_authorized",
                camera=self._active_camera,
                grasp_state=servo_cmd.state,
                dx_px=servo_cmd.dx_px,
                dy_px=servo_cmd.dy_px,
            )
            self.sequence.grasp_completed()
            return self._make_response(
                arm_cmd=servo_cmd,
                tracking=result,
            )

        if advance == "error":
            return self._make_response(
                arm_cmd={"action": "error",
                         "reason": self.sequence.status().error},
                tracking=result,
            )

        # 检查超限
        if self.servo.attempts_exceeded:
            return self._make_response(
                arm_cmd={"action": "error",
                         "reason": "max attempts exceeded"},
                tracking=result,
            )

        # Detector confirmation frames are not physical attempts. Counting idle
        # frames here used to exhaust the budget after only a few real moves.
        if servo_cmd.action == "move":
            self.servo.note_executed()
        return self._make_response(arm_cmd=servo_cmd, tracking=result)

    def _handle_grasp(self, frame: np.ndarray) -> dict:
        """闭合爪子抓取。"""
        servo_cmd = ServoCommand(
            action="grasp",
            claw_open=self.cal.claw_closed_value,
            duration_ms=600,
            state="grasping",
            reason="closing claw",
        )
        self.sequence.grasp_completed()
        return self._make_response(arm_cmd=servo_cmd)

    def _handle_hold(self, frame: np.ndarray) -> dict:
        """保持抓取等待稳定。"""
        if self.sequence.hold_ready(hold_s=1.8):
            self.sequence.advance_to_lift()
            self._lift_reference = relative_target_state(
                self.dual_tracker.claw.bbox,
                self.dual_tracker.target.bbox,
            )
            self._verification_samples = []
            return self._make_response(
                arm_cmd=ServoCommand(
                    action="move",
                    up_mm=6,
                    left_deg=0,
                    forward_mm=0,
                    duration_ms=800,
                    state="lifting",
                    reason="lifting object",
                ),
            )
        return self._make_response(
            arm_cmd={"action": "idle", "reason": "holding..."},
        )

    def _handle_lift(self, frame: np.ndarray) -> dict:
        """等待抬升停稳，然后进入视觉抓取验证。"""
        self.sequence.advance_to_verify()
        return self._make_response(
            arm_cmd={"action": "idle", "reason": "lift settled; verifying grasp"},
        )

    def _retry_or_abort_grasp(self, reason: str,
                              tracking: DualTrackResult | None = None) -> dict:
        """Release and visually re-align a failed close, with a hard retry cap."""
        if self._grasp_retry_count < 2:
            self._grasp_retry_count += 1
            retry_kind = "trial grasp" if self._last_grasp_was_trial else "grasp"
            retry_reason = (
                f"{retry_kind} verification failed ({reason}); "
                f"release and visually re-align, retry {self._grasp_retry_count}/2"
            )
            EVENT_LOG.event(
                "grasp_retry",
                retry=self._grasp_retry_count,
                trial=self._last_grasp_was_trial,
                reason=reason,
            )
            self._verification_samples = []
            self._carry_verified = False
            self.sequence.retry_after_failed_grasp(retry_reason)
            self.servo.prepare_grasp_retry()
            return self._make_response(
                arm_cmd=ServoCommand(
                    action="release",
                    claw_open=self.cal.claw_open_value,
                    duration_ms=500,
                    state="retry_release",
                    reason=retry_reason,
                ),
                tracking=tracking,
            )
        self.sequence.abort("object did not move with claw after two retries")
        return self._make_response(
            arm_cmd={
                "action": "error",
                "reason": f"grasp verification failed after retries: {reason}",
                "emergency_stop": True,
            },
            tracking=tracking,
        )

    def _handle_verify(self, frame: np.ndarray) -> dict:
        """Require repeated visual evidence that the target followed the claw."""
        result = self.dual_tracker.update(frame)
        # The legacy tracker may reject the closed gripper because its box
        # changes shape after lifting. A current, accepted YOLO-Pose detection
        # with both jaw points is authoritative for claw visibility only; the
        # object must still pass the independent co-motion verification below.
        visible_jaws = 0
        for point in self._claw_keypoints or []:
            if isinstance(point, dict):
                visibility = float(
                    point.get("confidence", point.get("v", 0.0)) or 0.0
                )
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                visibility = float(point[2]) if len(point) > 2 else 1.0
            else:
                visibility = 0.0
            if visibility > 0.35:
                visible_jaws += 1
        if self._motion_claw_bbox is not None and visible_jaws >= 2:
            if not result.claw_ok:
                EVENT_LOG.event(
                    "verify_pose_claw_override",
                    visible_jaws=visible_jaws,
                )
            result.claw_bbox = self._motion_claw_bbox
            result.claw_ok = True
            result.claw_lost_frames = 0
        if not result.claw_ok or not result.target_ok:
            # A lifted object can be briefly hidden by the closing fingers.
            # Wait several observations, but never count occlusion as success.
            reason = "target/claw unavailable after lift"
            self._verification_samples.append(False)
            self._verification_samples = self._verification_samples[-5:]
            EVENT_LOG.event(
                "grasp_verify_sample",
                verified=False,
                reason=reason,
                samples=self._verification_samples,
            )
            if len(self._verification_samples) >= 5:
                return self._retry_or_abort_grasp(reason, result)
            return self._make_response(
                arm_cmd={"action": "idle", "reason": f"verifying: {reason}"},
                tracking=result,
            )

        current = relative_target_state(result.claw_bbox, result.target_bbox)
        verified, reason = grasp_motion_verified(self._lift_reference, current)
        self._verification_samples.append(bool(verified))
        self._verification_samples = self._verification_samples[-5:]
        EVENT_LOG.event(
            "grasp_verify_sample",
            verified=verified,
            reason=reason,
            samples=self._verification_samples,
        )
        if sum(self._verification_samples) >= 3:
            self._carry_verified = True
            self.sequence.advance_to_retract()
            return self._make_response(
                arm_cmd=ServoCommand(
                    action="move",
                    forward_mm=-8,
                    up_mm=0,
                    left_deg=0,
                    claw_open=self.cal.claw_closed_value,
                    duration_ms=800,
                    state="retracting",
                    reason="grasp verified; retracting to carry position",
                ),
                tracking=result,
            )
        if len(self._verification_samples) >= 5:
            return self._retry_or_abort_grasp(reason, result)
        return self._make_response(
            arm_cmd={"action": "idle", "reason": f"verifying: {reason}"},
            tracking=result,
        )
    def _handle_retract(self, frame: np.ndarray) -> dict:
        """后撤完成后保持夹持，等待显式释放命令。"""
        self.sequence.advance_to_carry()
        return self._make_response(
            arm_cmd={
                "action": "carry",
                "reason": "object secured; claw remains closed",
            },
        )

    def _handle_release(self, frame: np.ndarray) -> dict:
        """释放完成。"""
        self.sequence.release_completed()
        return self._make_response(
            arm_cmd={"action": "done", "reason": "grasp and release complete"},
        )

    def request_release(self) -> dict:
        """Release is allowed only after visual verification reached CARRY."""
        if self.sequence.phase != Phase.CARRY or not self._carry_verified:
            EVENT_LOG.event(
                "release_rejected",
                phase=self.sequence.phase.value,
                carry_verified=self._carry_verified,
            )
            return {
                "type": "error",
                "message": "release rejected: no verified carried object",
            }
        self.sequence.advance_to_release()
        EVENT_LOG.event("release_accepted")
        return self._make_response(
            arm_cmd=ServoCommand(
                action="release",
                claw_open=self.cal.claw_open_value,
                duration_ms=500,
                state="releasing",
                reason="explicit release command",
            ),
        )

    # ── 响应构造 ──────────────────────────────────────────

    def _make_response(self, arm_cmd, tracking: DualTrackResult | None = None) -> dict:
        # 移动门控: 发出 move 命令时记录机械臂预计停稳时间
        action = arm_cmd.action if isinstance(arm_cmd, ServoCommand) else arm_cmd.get("action")
        if action in ("move", "grasp", "release"):
            dur_ms = (arm_cmd.duration_ms if isinstance(arm_cmd, ServoCommand)
                      else arm_cmd.get("duration_ms", 900))
            self._move_until = time.time() + float(dur_ms) / 1000.0 + 0.4

        status = self.sequence.status()
        servo_status = self.servo.status()

        response = {
            "type": "result",
            "phase": status.phase,
            "step": status.step,
            "step_label": status.step_label,
            "elapsed_s": status.elapsed_s,
            "active_camera": getattr(self, "_active_camera", "left"),
            "arm_cmd": _serialize_cmd(arm_cmd),
            "servo": {
                "state": servo_status.state,
                "error": servo_status.last_error,
            },
        }

        if tracking is not None:
            response["tracking"] = {
                "claw_ok": tracking.claw_ok,
                "target_ok": tracking.target_ok,
                "claw_bbox": _bbox_dict(tracking.claw_bbox),
                "target_bbox": _bbox_dict(tracking.target_bbox),
                "claw_keypoints": self._claw_keypoints,
                "jaw_pose": self._claw_keypoints is not None,
            }
        else:
            # Expose newly grounded/stored boxes immediately, including while
            # the other tracker is still being acquired. The Pi web overlay
            # should not have to wait until the ALIGN phase for the first box.
            stored = stored_tracking_boxes(self.dual_tracker)
            if stored is not None:
                response["tracking"] = {
                    "claw_ok": stored["claw_ok"],
                    "target_ok": stored["target_ok"],
                    "claw_bbox": _bbox_dict(stored["claw_bbox"]),
                    "target_bbox": _bbox_dict(stored["target_bbox"]),
                    "claw_keypoints": self._claw_keypoints,
                    "jaw_pose": self._claw_keypoints is not None,
                }

        phase_changed = status.phase != self._last_logged_phase
        action_changed = action != self._last_logged_action
        tracking_state = None if tracking is None else (
            bool(tracking.claw_ok), bool(tracking.target_ok)
        )
        tracking_changed = tracking_state != self._last_tracking_state
        important = action not in ("idle", "carry")
        if phase_changed or action_changed or tracking_changed or important:
            EVENT_LOG.event(
                "decision",
                frame=self.frame_count,
                active_camera=getattr(self, "_active_camera", "left"),
                phase=status.phase,
                step=status.step_label,
                arm_cmd=response["arm_cmd"],
                tracking=response.get("tracking"),
            )
        else:
            EVENT_LOG.throttled(
                "heartbeat", 5.0, "heartbeat",
                frame=self.frame_count,
                phase=status.phase,
                tracking=response.get("tracking"),
            )
        self._last_logged_phase = status.phase
        self._last_logged_action = action
        if tracking is not None:
            self._last_tracking_state = tracking_state

        self.latest_status = response
        return response

    def close(self) -> None:
        # 不关闭共享 grounder (跨会话复用, 避免每次重新加载模型)
        pass


# ── 序列化辅助 ───────────────────────────────────────────

def _serialize_cmd(cmd) -> dict:
    """将 ServoCommand 或 dict 序列化为可 JSON 的 dict。"""
    if isinstance(cmd, dict):
        return {k: (int(v) if isinstance(v, (np.integer,)) else v)
                for k, v in cmd.items()}
    result = {}
    for field in ("action", "forward_mm", "up_mm", "left_deg",
                  "claw_open", "duration_ms", "state", "reason"):
        val = getattr(cmd, field, None)
        if isinstance(val, (np.integer,)):
            val = int(val)
        elif isinstance(val, (np.floating,)):
            val = float(val)
        result[field] = val
    return result


def _bbox_dict(bbox) -> dict | None:
    if bbox is None:
        return None
    return {
        "x1": float(bbox.x1), "y1": float(bbox.y1),
        "x2": float(bbox.x2), "y2": float(bbox.y2),
        "width": float(bbox.width), "height": float(bbox.height),
    }


def _parse_qwen_json(raw: str) -> dict:
    """从 Qwen 回复中提取 JSON — 使用健壮解析器。

    复用 run_qwen_landmark_server.py 的成熟逻辑:
    parse_json_object → recover_partial_json_object
    """
    import re
    text = raw.strip()

    # Step 1: 去除 markdown 包裹
    text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text = re.sub(r"```$", "", text).strip()

    # Step 2: 直接解析
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass

    # Step 3: 正则提取 { ... }
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass

    # Step 4: 从截断的 JSON 中恢复部分字段
    return _recover_partial_grasp_json(text)


def _recover_partial_grasp_json(text: str) -> dict:
    """从截断的 Qwen JSON 回复中尽力恢复字段。"""
    import re
    result: dict = {}

    # 恢复布尔字段
    result["claw_found"] = _recover_bool(text, "claw_found", False)
    result["objects_found"] = _recover_bool(text, "objects_found", False)

    # 恢复 claw_bbox
    claw_match = re.search(
        r'"claw_bbox"\s*:\s*\[([^\]]*)\]', text, flags=re.S
    )
    if claw_match:
        nums = re.findall(r"[\d.]+", claw_match.group(1))
        if len(nums) >= 4:
            result["claw_bbox"] = [float(v) for v in nums[:4]]

    # 恢复 objects 数组中的 dict
    objects = _recover_array_dicts(text, "objects")
    if objects:
        result["objects"] = objects

    # 恢复 scene_description
    desc_match = re.search(
        r'"scene_description"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.S
    )
    if desc_match:
        try:
            result["scene_description"] = json.loads(
                '"' + desc_match.group(1) + '"'
            )
        except json.JSONDecodeError:
            result["scene_description"] = desc_match.group(1)[:200]

    return result


def _recover_bool(text: str, key: str, default: bool = False) -> bool:
    import re
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*(true|false)', text, flags=re.I
    )
    if not match:
        return default
    return match.group(1).lower() == "true"


def _recover_array_dicts(text: str, key: str) -> list[dict]:
    """从截断文本中恢复数组中的对象字典。"""
    import re
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
    if not match:
        return []
    start = text.find("[", match.start()) + 1
    items: list[dict] = []
    idx = start
    while idx < len(text):
        obj_start = text.find("{", idx)
        arr_end = text.find("]", idx)
        if obj_start < 0 or (arr_end >= 0 and arr_end < obj_start):
            break
        obj_text = _balanced_json_fragment(text, obj_start)
        if not obj_text:
            break
        try:
            obj = json.loads(obj_text)
            if isinstance(obj, dict):
                items.append(obj)
        except json.JSONDecodeError:
            break
        idx = obj_start + len(obj_text)
    return items


def _balanced_json_fragment(text: str, start: int) -> str:
    """提取从 start 开始的平衡 JSON 对象字符串。"""
    if start < 0 or start >= len(text) or text[start] != "{":
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


# ── HTTP 端点 ────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "arm_grasp_server"})


def _label_session() -> GraspSession:
    global _LABEL_SESSION
    if _LABEL_SESSION is None:
        _LABEL_SESSION = GraspSession(app.state.tracker_config, app.state.arm_calibration)
    return _LABEL_SESSION


def _production_gripper_model_path() -> Path:
    """Legacy box model retained until a Pose model is explicitly promoted."""
    return Path.home() / "grab_visual" / "models" / "gripper_tip_best.pt"


def _production_gripper_pose_model_path() -> Path:
    return Path.home() / "grab_visual" / "models" / "gripper_jaw_pose_best.pt"


@dataclass
class GripperObservation:
    bbox: Any
    keypoints: list[dict] | None
    confidence: float
    pose: bool


def _detect_trained_gripper(frame: np.ndarray) -> GripperObservation | None:
    """Prefer the promoted two-keypoint Pose model, then fall back to bbox."""
    global _GRIPPER_YOLO, _GRIPPER_YOLO_PATH
    pose_path = _production_gripper_pose_model_path()
    legacy_path = _production_gripper_model_path()
    model_path = pose_path if pose_path.exists() else legacy_path
    if not model_path.exists() or _TRAIN_STATE.get("active"):
        return None
    is_pose = model_path == pose_path
    try:
        if _GRIPPER_YOLO is None or _GRIPPER_YOLO_PATH != str(model_path):
            from ultralytics import YOLO
            _GRIPPER_YOLO = YOLO(str(model_path))
            _GRIPPER_YOLO_PATH = str(model_path)
            EVENT_LOG.event("gripper_yolo_loaded", model=str(model_path), pose=is_pose)
        results = _GRIPPER_YOLO.predict(frame, imgsz=416, conf=0.30, device=0, verbose=False)
        result = results[0] if results else None
        boxes = result.boxes if result is not None else None
        if boxes is None or len(boxes) == 0:
            return None
        confidences = boxes.conf.detach().cpu().numpy()
        best = int(np.argmax(confidences))
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best].detach().cpu().numpy()]
        from qwen_grounded_tracker.domain import BBox
        bbox = BBox(x1, y1, x2, y2).clamp(frame.shape[1], frame.shape[0])
        keypoints: list[dict] | None = None
        if is_pose:
            keypoints = []
            pose_points = getattr(result, "keypoints", None)
            if pose_points is not None and pose_points.xy is not None:
                xy = pose_points.xy[best].detach().cpu().numpy()
                conf_tensor = getattr(pose_points, "conf", None)
                kp_conf = conf_tensor[best].detach().cpu().numpy() if conf_tensor is not None else np.ones(len(xy))
                for index in range(2):
                    if index < len(xy):
                        px, py = float(xy[index][0]), float(xy[index][1])
                        confidence = float(kp_conf[index]) if index < len(kp_conf) else 1.0
                    else:
                        px = py = confidence = 0.0
                    visible = confidence >= 0.25 and px > 0.0 and py > 0.0
                    keypoints.append({
                        "x": px if visible else 0.0, "y": py if visible else 0.0,
                        "v": 2 if visible else 0, "confidence": confidence,
                    })
                if len(keypoints) == 2 and all(p["v"] > 0 for p in keypoints) and keypoints[0]["x"] > keypoints[1]["x"]:
                    keypoints.reverse()
        return GripperObservation(
            bbox=bbox, keypoints=keypoints,
            confidence=float(confidences[best]), pose=is_pose,
        )
    except Exception as exc:
        EVENT_LOG.throttled("gripper_yolo_error", 5.0, "gripper_yolo_error", error=repr(exc))
        return None

@app.post("/api/gripper/qwen")
async def qwen_gripper_label(payload: dict) -> JSONResponse:
    """Ground the gripper tip in one labelled-camera image without moving the arm."""
    try:
        raw_b64 = str(payload.get("image_b64", ""))
        frame = _decode_jpeg(base64.b64decode(raw_b64))
        if frame is None:
            raise ValueError("invalid image")
        camera = str(payload.get("camera", "left"))
        instruction = (
            f"Locate the black robot gripper tip / two gripping fingers in this {camera} camera image. "
            "Do not select the whole robot arm, shadows, floor, cable, wheel, or background. "
            "Return one tight bbox around the two gripping fingers."
        )
        if _TRAIN_STATE.get("active"):
            return JSONResponse({"ok": False, "error": "YOLO training is active"}, status_code=409)
        with _LABEL_LOCK:
            session = _label_session()
            result = session.grounder.ground(frame, instruction, None)
        bbox = _bbox_dict(result.bbox)
        EVENT_LOG.event("gripper_qwen_label", camera=camera, found=result.found, bbox=bbox,
                        confidence=result.confidence, raw=result.raw_text[:2000])
        return JSONResponse({"ok": True, "found": bool(result.found), "bbox": bbox,
                             "confidence": float(result.confidence), "raw": result.raw_text})
    except Exception as exc:
        EVENT_LOG.event("gripper_qwen_label_error", error=repr(exc))
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


def _calibration_target_view(frame: np.ndarray, item: dict | None) -> dict:
    height, width = frame.shape[:2]
    view = {
        "width": int(width), "height": int(height),
        "target_found": False, "target_bbox": None,
        "target_center": None, "target_confidence": 0.0,
    }
    item = item if isinstance(item, dict) else {}
    box = item.get("target_bbox")
    if item.get("target_found") and isinstance(box, (list, tuple)) and len(box) >= 4:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
        y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
        pixel_box = {
            "x1": x1 * width / 1000.0, "y1": y1 * height / 1000.0,
            "x2": x2 * width / 1000.0, "y2": y2 * height / 1000.0,
        }
        pixel_box["width"] = pixel_box["x2"] - pixel_box["x1"]
        pixel_box["height"] = pixel_box["y2"] - pixel_box["y1"]
        if pixel_box["width"] >= 2 and pixel_box["height"] >= 2:
            view.update(
                target_found=True,
                target_bbox=pixel_box,
                target_center={
                    "x": (pixel_box["x1"] + pixel_box["x2"]) * 0.5,
                    "y": (pixel_box["y1"] + pixel_box["y2"]) * 0.5,
                },
                target_confidence=float(item.get("confidence", 0.0) or 0.0),
            )
    return view


def _calibration_add_jaws(view: dict, observation: GripperObservation | None) -> None:
    view["jaw_bbox"] = _bbox_dict(observation.bbox) if observation else None
    view["jaw_points"] = list(observation.keypoints or []) if observation else []
    view["jaw_confidence"] = float(observation.confidence) if observation else 0.0
    view["jaw_pose_model"] = bool(observation and observation.pose)


@app.post("/api/calibration/analyze")
async def analyze_arm_mapping_pair(payload: dict) -> JSONResponse:
    """Analyze a synchronized pair for annotation only; never move the arm."""
    if _TRAIN_STATE.get("active"):
        return JSONResponse({"ok": False, "error": "YOLO training is active"}, status_code=409)
    try:
        frames = {}
        for camera in ("left", "right"):
            raw = base64.b64decode(str(payload.get(f"{camera}_image_b64", "")))
            frame = _decode_jpeg(raw)
            if frame is None:
                raise ValueError(f"invalid {camera} image")
            frames[camera] = frame

        observations = {
            camera: _detect_trained_gripper(frame)
            for camera, frame in frames.items()
        }
        views = {
            camera: _calibration_target_view(frame, None)
            for camera, frame in frames.items()
        }
        raw_results = {}
        if bool(payload.get("use_qwen", True)):
            description = str(payload.get("target_description", "")).strip()
            if not description:
                raise ValueError("target_description is required for Qwen analysis")
            instruction = (
                f"Locate only the requested grasp target ({description}) in this single camera image. "
                "Do not select the robot arm, black gripper, shadow, floor, cable, wheel, or a different object. "
                "Return one tight bounding box around the target; report not found when it is not genuinely visible."
            )
            with _LABEL_LOCK:
                session = _label_session()
                for camera in ("left", "right"):
                    result = session.grounder.ground(frames[camera], instruction, None)
                    raw_results[camera] = str(result.raw_text or "")
                    bbox = _bbox_dict(result.bbox) if result.found and result.bbox else None
                    if bbox and bbox["width"] >= 2 and bbox["height"] >= 2:
                        views[camera].update(
                            target_found=True,
                            target_bbox=bbox,
                            target_center={
                                "x": (bbox["x1"] + bbox["x2"]) * 0.5,
                                "y": (bbox["y1"] + bbox["y2"]) * 0.5,
                            },
                            target_confidence=float(result.confidence),
                        )
        raw_text = json.dumps(raw_results, ensure_ascii=False)
        for camera in ("left", "right"):
            _calibration_add_jaws(views[camera], observations[camera])
        EVENT_LOG.event(
            "calibration_pair_analyzed",
            target=str(payload.get("target_description", ""))[:200],
            use_qwen=bool(payload.get("use_qwen", True)),
            left_target=views["left"]["target_found"],
            right_target=views["right"]["target_found"],
            left_jaws=len(views["left"]["jaw_points"]),
            right_jaws=len(views["right"]["jaw_points"]),
        )
        return JSONResponse({"ok": True, "views": views, "raw": raw_text})
    except Exception as exc:
        EVENT_LOG.event("calibration_pair_analyze_error", error=repr(exc))
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
def _append_train_log(message: str) -> None:
    _TRAIN_STATE["log"].append(str(message))
    _TRAIN_STATE["log"] = _TRAIN_STATE["log"][-120:]


def _run_gripper_training(zip_path: Path, epochs: int, imgsz: int) -> None:
    global _SHARED_GROUNDER, _LABEL_SESSION, _GRIPPER_YOLO, _GRIPPER_YOLO_PATH
    try:
        _TRAIN_STATE.update(active=True, status="preparing", error="", result="", log=[])
        # Explicit user-started training may unload Qwen, but never stops this HTTP service.
        _LABEL_SESSION = None
        _SHARED_GROUNDER = None
        _GRIPPER_YOLO = None
        _GRIPPER_YOLO_PATH = ""
        try:
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass
        work = Path(tempfile.mkdtemp(prefix="gripper_train_", dir="/tmp"))
        import zipfile
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(work)
        # Ultralytics resolves a relative dataset root against its global datasets
        # directory.  Pin it to this one-shot extraction directory instead.
        (work / "data.yaml").write_text(
            f"path: {work.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "kpt_shape: [2, 3]\nflip_idx: [1, 0]\n"
            "names:\n  0: gripper_jaws\n",
            encoding="utf-8",
        )
        output_root = Path.home() / "grab_visual" / "gripper_training"
        output_root.mkdir(parents=True, exist_ok=True)
        model_path = (
            _production_gripper_model_path()
            if _production_gripper_model_path().exists()
            else Path.home() / "3090" / "models" / "yolo" / "yolo11n-seg.pt"
        )
        _TRAIN_STATE["status"] = "training"
        code = (
            "from ultralytics import YOLO; "
            "m=YOLO('yolo11n-pose.yaml'); "
            f"m.load({str(model_path)!r}); "
            f"m.train(data={str(work / 'data.yaml')!r},epochs={epochs},imgsz={imgsz},batch=2,device=0,workers=2,"
            # Ultralytics AMP validation may try to download an unrelated
            # reference model.  This deployment is intentionally offline, so
            # use regular FP32 training and avoid that network-only self-test.
            f"project={str(output_root)!r},name='gripper_jaw_pose',exist_ok=True,cache=False,amp=False); "
            "print('TRAIN_DONE')"
        )
        proc = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            _append_train_log(line.rstrip())
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"YOLO training exited with code {rc}")
        best = output_root / "gripper_jaw_pose" / "weights" / "best.pt"
        _TRAIN_STATE.update(active=False, status="done", result=str(best))
    except Exception as exc:
        _TRAIN_STATE.update(active=False, status="error", error=str(exc))
        _append_train_log(repr(exc))


@app.post("/api/gripper/train")
async def start_gripper_training(payload: dict) -> JSONResponse:
    if _TRAIN_STATE.get("active"):
        return JSONResponse({"ok": False, "error": "training already active"}, status_code=409)
    try:
        data = base64.b64decode(str(payload.get("dataset_zip_b64", "")))
        if len(data) < 1000:
            raise ValueError("dataset zip is empty")
        upload_dir = Path.home() / "grab_visual" / "gripper_training" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        zip_path = upload_dir / f"dataset_{int(time.time())}.zip"
        zip_path.write_bytes(data)
        epochs = max(5, min(int(payload.get("epochs", 40)), 100))
        imgsz = max(320, min(int(payload.get("imgsz", 416)), 640))
        threading.Thread(target=_run_gripper_training, args=(zip_path, epochs, imgsz), daemon=True).start()
        return JSONResponse({"ok": True, "message": "training started; Qwen GPU memory will be released"})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/gripper/train/promote")
async def promote_gripper_pose_model() -> JSONResponse:
    if _TRAIN_STATE.get("active"):
        return JSONResponse({"ok": False, "error": "training is active"}, status_code=409)
    source = Path(str(_TRAIN_STATE.get("result", "")))
    if not source.exists():
        source = Path.home() / "grab_visual" / "gripper_training" / "gripper_jaw_pose" / "weights" / "best.pt"
    if not source.exists():
        return JSONResponse({"ok": False, "error": "finished pose weight not found"}, status_code=404)
    destination = _production_gripper_pose_model_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    global _GRIPPER_YOLO, _GRIPPER_YOLO_PATH
    _GRIPPER_YOLO = None
    _GRIPPER_YOLO_PATH = ""
    EVENT_LOG.event("gripper_pose_promoted", source=str(source), destination=str(destination))
    return JSONResponse({"ok": True, "model": str(destination)})

@app.get("/api/gripper/train/status")
async def gripper_training_status() -> JSONResponse:
    return JSONResponse(dict(_TRAIN_STATE))


# ── WebSocket 抓取端点 ────────────────────────────────────

@app.websocket("/ws/grasp")
async def grasp_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    EVENT_LOG.event("websocket_connected", peer=str(websocket.client))
    config = websocket.app.state.tracker_config
    cal = websocket.app.state.arm_calibration

    session = GraspSession(config, calibration=cal)

    try:
        # 等待启动消息
        first_msg = await websocket.receive_text()
        try:
            hello = json.loads(first_msg)
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps(
                {"type": "error", "message": "first message must be JSON"}))
            return

        msg_type = hello.get("type", "")
        if msg_type == "start":
            target_desc = hello.get("target_description", "")
            ack = session.handle_start(
                target_desc,
                active_camera=str(hello.get("active_camera", "left")),
                available_cameras=hello.get("available_cameras"),
            )
            await websocket.send_text(json.dumps(ack))
        else:
            await websocket.send_text(json.dumps(
                {"type": "error", "message": f"expected type=start, got {msg_type}"}))
            return

        # 帧处理循环
        while True:
            try:
                message = await websocket.receive()
            except Exception:
                break

            # 文本控制消息
            text = message.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue

                ctrl_type = payload.get("type", "")
                if ctrl_type == "stop":
                    EVENT_LOG.event("control", type="stop")
                    session.sequence.abort("user stop")
                    await websocket.send_text(json.dumps(
                        {
                            "type": "result",
                            "message": "stopped",
                            "arm_cmd": {
                                "action": "error",
                                "reason": "user stop",
                                "emergency_stop": True,
                            },
                        }))
                elif ctrl_type == "release":
                    EVENT_LOG.event("control", type="release")
                    await websocket.send_text(json.dumps(
                        session.request_release(), ensure_ascii=False))
                elif ctrl_type == "reset":
                    EVENT_LOG.event("control", type="reset")
                    session = GraspSession(config, calibration=cal)
                    await websocket.send_text(json.dumps(
                        {"type": "ack", "message": "reset"}))
                elif ctrl_type == "close":
                    await websocket.send_text(json.dumps(
                        {"type": "ack", "message": "closing"}))
                    break
                continue

            # 二进制 JPEG 帧
            data = message.get("bytes")
            if data is not None:
                frame = _decode_jpeg(data)
                if frame is None:
                    EVENT_LOG.event("bad_jpeg", bytes=len(data))
                    await websocket.send_text(json.dumps(
                        {"type": "error", "message": "bad JPEG"}))
                    continue

                result = session.handle_frame(frame)

                # 发送 servo 命令给树莓派
                await websocket.send_text(json.dumps(result, ensure_ascii=False))

    except WebSocketDisconnect:
        EVENT_LOG.event("websocket_disconnected", reason="client_disconnect")
    except Exception as exc:
        EVENT_LOG.event("websocket_error", error=repr(exc))
        raise
    finally:
        session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Arm Visual Servo Grasp Server")
    parser.add_argument("--config", default="configs/linux_3090_remote.yaml")
    parser.add_argument("--calibration", default="grab_target/arm_visual_calibration.json")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    app.state.tracker_config = load_config(config_path)

    cal_path = ROOT / args.calibration if not Path(args.calibration).is_absolute() else Path(args.calibration)
    if cal_path.exists():
        app.state.arm_calibration = ArmVisualCalibration.load(cal_path)
        print(f"[grasp] loaded calibration from {cal_path}")
    else:
        app.state.arm_calibration = DEFAULT_CALIBRATION
        print("[grasp] using default calibration")

    EVENT_LOG.event(
        "server_start",
        host=args.host,
        port=args.port,
        config=str(config_path),
        calibration=str(cal_path),
        calibration_exists=cal_path.exists(),
        log=str(EVENT_LOG.path),
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
