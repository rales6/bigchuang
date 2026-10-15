"""视觉伺服控制器 v2 — 优化版。

改进:
  - 自适应步长: 误差大→大步, 误差小→小步 (sigmoid 映射)
  - 死区滞后: 防止在阈值边界来回振荡
  - 立体深度辅助: 利用左右相机视差估算深度 (可选)
  - 夹爪安全位: 运动时保持张开, 只在抓取时闭合
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

try:
    from .automation_logic import jaw_geometry
except ImportError:
    from automation_logic import jaw_geometry


class ServoState(Enum):
    IDLE = "idle"
    QWEN_INIT = "qwen_init"
    ALIGNING_HORIZONTAL = "align_h"
    APPROACHING = "approach"
    FINE_ALIGNING = "fine_align"
    READY_TO_GRASP = "ready"
    ERROR = "error"


@dataclass
class ServoCommand:
    action: str = "idle"          # idle | move | grasp | release | error | reground
    forward_mm: int = 0
    up_mm: int = 0
    left_deg: int = 0
    claw_open: int = -1           # -1=不改变, 0=闭合, 100=全开
    duration_ms: int = 800
    state: str = "idle"
    reason: str = ""
    dx_px: float = 0.0
    dy_px: float = 0.0
    claw_area: float = 0.0
    target_area: float = 0.0


@dataclass
class ServoStatus:
    state: str = "idle"
    claw_center: tuple[float, float] = (0.0, 0.0)
    target_center: tuple[float, float] = (0.0, 0.0)
    dx_px: float = 0.0
    dy_px: float = 0.0
    area_ratio: float = 0.0
    h_aligned: bool = False
    v_aligned: bool = False
    z_ready: bool = False
    attempt: int = 0
    total_attempts: int = 0
    step_size_scale: float = 1.0
    last_error: str = ""


def _adaptive_gain(error_px: float, threshold_px: float) -> float:
    """自适应增益: 误差大时大步(≤1.0)，误差小时小步，带平滑过渡。"""
    if error_px <= 0:
        return 0.0
    ratio = error_px / max(threshold_px, 1.0)
    # sigmoid: 大步(ratio>3→0.95) 到 小步(ratio<1→0.3)
    gain = 0.30 + 0.65 / (1.0 + math.exp(-2.0 * (ratio - 1.8)))
    return gain


class VisualServoController:
    """像素误差 → 机械臂运动命令 — 优化版。

    安全机制:
      - 安全区: 爪尖必须在画面中央区域才允许伺服运动, 贴边先拉回
      - 步长限幅: 单步移动不超过爪子可见范围 (防跑出视野)
      - 记忆回退: 爪子丢失时反向执行上一步 (找回概率最大)
    """

    # 安全区 (相对画面比例): 爪尖中心必须在此区域内
    SAFE_X1, SAFE_X2 = 0.15, 0.85
    SAFE_Y1, SAFE_Y2 = 0.30, 0.92

    def __init__(self, calibration) -> None:
        self.cal = calibration
        self.state = ServoState.IDLE
        self._attempt_count: int = 0
        self._total_attempt_count: int = 0
        self._last_error: str = ""
        self._overshoot_count: int = 0
        # 死区滞后: 进入对齐状态后需要更大的误差才能退出
        self._was_aligned_h: bool = False
        self._was_aligned_v: bool = False
        self._was_aligned_z: bool = False
        self._last_dx_sign: int = 0  # 用于检测过冲
        # 记忆回退: 记录最近一次移动命令 (用于丢失时反向执行)
        self._last_move: dict | None = None
        self._rollback_count: int = 0
        # Keep a short reversible motion trail. If the claw leaves the camera
        # view, replaying this trail backwards is safer than guessing a
        # direction from an already missing bbox.
        self._motion_history: list[dict[str, int]] = []
        self._max_motion_history: int = 16
        self._grasp_height_confirm_frames: int = 0
        self._trial_grasp_stall_frames: int = 0
        self._last_grasp_height_abs_error: float | None = None
        self._approach_ready: bool = False
        self._depth_direction_sign: int = 1
        self._depth_feedback_pending: bool = False
        self._last_depth_abs_error: float | None = None
        # Slowly learn the real image response of yaw and vertical motion.
        # Tight bounds prevent a noisy frame from retuning an axis.
        self._yaw_deg_per_pixel: float = float(self.cal.deg_per_pixel)
        self._vertical_mm_per_pixel: float = float(self.cal.mm_per_pixel_vertical)
        self._pending_axis_response: dict[str, float] | None = None
        self._current_response_metrics: dict[str, float] | None = None
        self._depth_worse_frames: int = 0
        self._proximity_confirm_frames: int = 0
        self._depth_vote_sign: int = 0
        self._depth_vote_count: int = 0
        # Pose horizontal motion is coarse even at one degree.
        self._pose_h_vote_sign: int = 0
        self._pose_h_vote_count: int = 0
        self._pose_h_flip_count: int = 0
        self._pose_h_lower_count: int = 0
        self._pose_final_lower_done: bool = False


    # ── 安全区与找回 ──────────────────────────────────────

    def claw_in_safe_zone(self, claw_bbox, frame_shape) -> bool:
        """检查爪尖是否在安全区内。"""
        if claw_bbox is None:
            return False
        h, w = frame_shape[:2]
        cx = (claw_bbox.x1 + claw_bbox.x2) / 2.0 / w
        cy = (claw_bbox.y1 + claw_bbox.y2) / 2.0 / h
        return (self.SAFE_X1 <= cx <= self.SAFE_X2
                and self.SAFE_Y1 <= cy <= self.SAFE_Y2)

    def pull_back_cmd(self, claw_bbox, frame_shape) -> ServoCommand:
        """爪子贴边/在安全区外 → 小步拉回中央。"""
        h, w = frame_shape[:2]
        cx = (claw_bbox.x1 + claw_bbox.x2) / 2.0 / w
        cy = (claw_bbox.y1 + claw_bbox.y2) / 2.0 / h

        left_deg = 0.0
        up_mm = 0.0
        reason_parts = []

        # 水平贴边: 爪子太靠右 → 基座左转 (画面里爪尖右移需要左转还是右转?)
        # 眼在手上: 基座左转 → 手臂右移 → 画面爪尖向右 → 所以:
        #   爪尖太靠左(cx小) → 右转; 太靠右(cx大) → 左转
        if cx < self.SAFE_X1:
            left_deg = 8.0   # 爪尖太靠左 → 基座右转, 手臂摆到左边?
            reason_parts.append(f"claw too left ({cx:.2f})")
        elif cx > self.SAFE_X2:
            left_deg = -8.0
            reason_parts.append(f"claw too right ({cx:.2f})")

        # Installed yaw has the opposite sign to image X.
        left_deg = -left_deg

        # 垂直贴边: 爪尖太靠下 → 抬升; 太靠上 → 下降
        if cy > self.SAFE_Y2:
            up_mm = 8.0
            reason_parts.append(f"claw too low ({cy:.2f})")
        elif cy < self.SAFE_Y1:
            up_mm = -8.0
            reason_parts.append(f"claw too high ({cy:.2f})")

        if not reason_parts:
            return ServoCommand(action="idle", reason="claw in safe zone")

        # One axis per observation. Horizontal correction has priority because
        # left/right motion is also the only motion used to identify the claw.
        # Re-observe before issuing vertical correction: the arm has measurable
        # cross-axis coupling.
        if left_deg:
            up_mm = 0.0
        # Edge recovery must be gentler than ordinary alignment.  The real arm
        # previously crossed almost the full image on a 5 degree correction.
        left_limit = min(2.0, self.cal.max_left_per_step_deg)
        vertical_limit = min(1.0, self.cal.max_up_per_step_mm)
        left_deg = max(-left_limit, min(left_limit, left_deg))
        up_mm = max(-vertical_limit, min(vertical_limit, up_mm))

        return self._make_move(
            forward_mm=0, up_mm=int(up_mm), left_deg=int(left_deg),
            dx_px=0, dy_px=0, claw_area=0, target_area=0,
            reason="pull back: " + ", ".join(reason_parts),
        )

    def rollback_cmd(self) -> ServoCommand | None:
        """爪子丢失时沿运动历史逆序小步回退，且一次只动一个轴。"""
        if not self._motion_history or self._rollback_count >= 12:
            return None
        move = self._motion_history.pop()
        self._rollback_count += 1
        forward_mm = max(
            -self.cal.max_forward_per_step_mm,
            min(self.cal.max_forward_per_step_mm,
                -int(move.get("forward_mm", 0))),
        )
        up_mm = max(
            -self.cal.max_up_per_step_mm,
            min(self.cal.max_up_per_step_mm, -int(move.get("up_mm", 0))),
        )
        left_deg = max(
            -self.cal.max_left_per_step_deg,
            min(self.cal.max_left_per_step_deg,
                -int(move.get("left_deg", 0))),
        )
        # Horizontal motion is the safest way to recover visual location.
        # If the historical record contains coupled axes, correct only one and
        # observe the new frame before considering another correction.
        if left_deg:
            forward_mm = 0
            up_mm = 0
        elif forward_mm:
            up_mm = 0
        return ServoCommand(
            action="move",
            forward_mm=int(forward_mm),
            up_mm=int(up_mm),
            left_deg=int(left_deg),
            claw_open=-1,
            duration_ms=800,
            state="recovering",
            reason=f"visual recovery #{self._rollback_count}: reverse last move",
        )

    def record_move(self, forward_mm, up_mm, left_deg) -> None:
        """记录一次移动 (用于丢失时回退)。"""
        move = {
            "forward_mm": int(forward_mm),
            "up_mm": int(up_mm),
            "left_deg": int(left_deg),
        }
        if not any(move.values()):
            return
        self._last_move = move
        self._motion_history.append(move)
        if len(self._motion_history) > self._max_motion_history:
            self._motion_history = self._motion_history[-self._max_motion_history:]

    def clear_rollback(self) -> None:
        """视觉重新锁定后允许下一次恢复，但保留可逆运动轨迹。"""
        self._rollback_count = 0

    def clear_motion_history(self) -> None:
        """在抓取进入不可逆阶段前清除旧的对齐轨迹。"""
        self._motion_history.clear()
        self._last_move = None
        self._rollback_count = 0

    def probe_cmd(self) -> ServoCommand:
        """爪子丢失 → 小幅左右摆动。

        爪子移动 → 运动检测重新定位 (黑色+运动的物体=爪子)。
        方向交替, 幅度小, 不影响大致位置。
        """
        self._probe_dir = -getattr(self, "_probe_dir", 1)
        deg = min(5, int(self.cal.max_left_per_step_deg)) * self._probe_dir
        return ServoCommand(
            action="move",
            forward_mm=0, up_mm=0, left_deg=deg,
            claw_open=-1,
            duration_ms=1200,
            state="probing",
            reason=f"probe: wiggle {deg:+d} deg to relocate claw",
        )

    def compute(
        self,
        claw_bbox,
        target_bbox,
        frame_shape: tuple[int, int, int],
        dual_result=None,
        stereo_depth_mm: float | None = None,
        claw_keypoints=None,  # two jaw contact points; None means legacy bbox model  # 立体深度 (可选)
    ) -> ServoCommand:
        h, w = frame_shape[:2]

        if claw_bbox is None or target_bbox is None:
            return ServoCommand(
                action="reground" if self._total_attempt_count > 0 else "idle",
                state="error",
                reason=f"bbox missing: claw={claw_bbox is not None}, target={target_bbox is not None}",
            )

        # Pose geometry uses the two physical contact points. A one-point
        # observation cannot authorize depth or vertical motion.
        pose_geometry = None
        if claw_keypoints is not None:
            pose_geometry = jaw_geometry(claw_keypoints, target_bbox)
            if pose_geometry["visible_count"] < 2:
                return ServoCommand(
                    action="idle", state="need_two_jaw_points",
                    reason=(
                        f"only {pose_geometry['visible_count']} jaw point visible; "
                        "depth and vertical motion blocked"
                    ),
                )
        if pose_geometry is not None:
            claw_cx, claw_cy = pose_geometry["midpoint"]
            fingertip_y = claw_cy
        else:
            claw_cx = (claw_bbox.x1 + claw_bbox.x2) / 2.0
            claw_cy = (claw_bbox.y1 + claw_bbox.y2) / 2.0
            fingertip_y = float(claw_bbox.y2)
        target_cx = (target_bbox.x1 + target_bbox.x2) / 2.0
        target_cy = (target_bbox.y1 + target_bbox.y2) / 2.0
        dx_px = target_cx - claw_cx
        dy_px = target_cy - claw_cy
        target_height = float(target_bbox.height)
        fingertip_gap_px = float(target_bbox.y1 - fingertip_y)
        hover_gap_px = max(10.0, min(22.0, target_height * 0.16))
        hover_error_px = fingertip_gap_px - hover_gap_px
        response_approach_gap_px = max(32.0, min(60.0, target_height * 0.65))
        response_approach_error_px = fingertip_gap_px - response_approach_gap_px
        self._observe_axis_response(
            dx_px, hover_error_px, response_approach_error_px,
        )
        self._current_response_metrics = {
            "dx_px": dx_px,
            "hover_error_px": hover_error_px,
            "approach_error_px": response_approach_error_px,
        }

        # 面积
        claw_area = float(claw_bbox.width * claw_bbox.height)
        target_area = float(target_bbox.width * target_bbox.height)
        frame_area = float(w * h)
        target_area_ratio = target_area / max(frame_area, 1.0)

        if target_area_ratio < self.cal.min_target_area_ratio:
            return ServoCommand(
                action="idle", state="error",
                reason=f"target too small ({target_area_ratio:.4f} < {self.cal.min_target_area_ratio})",
                dx_px=dx_px, dy_px=dy_px, claw_area=claw_area, target_area=target_area,
            )

        area_ratio = claw_area / max(target_area, 1.0)

        # ── 自适应阈值 (带滞后) ────────────────────────────
        # Scale the deadband slightly for a large nearby target so tiny yaw
        # steps do not keep crossing its centre.
        if pose_geometry is not None:
            h_thresh = float(pose_geometry["center_x_tolerance_px"])
        else:
            h_thresh = max(
                self.cal.horizontal_alignment_px,
                min(28.0, float(target_bbox.width) * 0.22),
            )
        v_thresh = self.cal.vertical_alignment_px
        # 滞后: 进入对齐后阈值放宽
        h_thresh_effective = h_thresh * (1.6 if self._was_aligned_h else 1.0)
        v_thresh_effective = v_thresh * (1.6 if self._was_aligned_v else 1.0)

        abs_dx = abs(dx_px)
        abs_dy = abs(dy_px)

        h_aligned = abs_dx < h_thresh_effective
        if pose_geometry is not None:
            h_aligned = h_aligned and bool(pose_geometry.get("target_between"))
        v_aligned = abs(hover_error_px) < v_thresh_effective

        # 深度判断: 优先用立体深度
        # Always define coarse-depth thresholds; stereo callers may still need
        # them when depth is not yet within the ready range.
        approach_gap_px = max(32.0, min(60.0, target_height * 0.65))
        approach_tolerance_px = max(8.0, min(15.0, target_height * 0.15))

        if stereo_depth_mm is not None:
            z_ready = stereo_depth_mm < 40.0  # < 4cm 认为可抓取
        else:
            approach_gap_px = max(32.0, min(60.0, target_height * 0.65))
            approach_tolerance_px = max(
                8.0, min(15.0, target_height * 0.15),
            )
            approach_error_px = fingertip_gap_px - approach_gap_px
            # Without calibrated stereo this is only a 2-D proximity gate.
            # The promoted YOLO box covers the whole gripper and its bottom edge
            # jitters, so use a broad band and three stable observations.
            near_lower_px = -max(45.0, target_height * 0.95)
            near_upper_px = approach_gap_px + 30.0
            visually_near = near_lower_px <= fingertip_gap_px <= near_upper_px
            if not self._approach_ready:
                if visually_near:
                    self._proximity_confirm_frames += 1
                else:
                    self._proximity_confirm_frames = 0
                if self._proximity_confirm_frames >= 3:
                    self._approach_ready = True
                    self._depth_vote_sign = 0
                    self._depth_vote_count = 0
            # Once near proximity is acquired, keep it latched for this target.
            # Re-entering depth control from detector jitter caused D1/U1 shake.
            z_ready = self._approach_ready

        # 过冲检测
        if self._last_dx_sign != 0 and dx_px * self._last_dx_sign < 0:
            self._overshoot_count += 1
        self._last_dx_sign = 1 if dx_px > 0 else (-1 if dx_px < 0 else 0)

        # 更新滞后状态
        self._was_aligned_h = h_aligned
        self._was_aligned_v = v_aligned
        self._was_aligned_z = z_ready

        # ── 状态机 ─────────────────────────────────────────

        if (pose_geometry is not None
                and float(pose_geometry.get("opening_ratio", 0.0)) < 0.55):
            return ServoCommand(
                action="idle", state="jaw_opening_too_narrow",
                reason=(
                    "target is wider than the visible jaw opening; "
                    f"opening_ratio={pose_geometry.get('opening_ratio', 0.0):.2f}"
                ),
                dx_px=dx_px, dy_px=dy_px,
                claw_area=claw_area, target_area=target_area,
            )
        if h_aligned and pose_geometry is not None:
            self._pose_h_vote_sign = 0
            self._pose_h_vote_count = 0
            self._pose_h_flip_count = 0
        if not h_aligned:
            self._depth_vote_sign = 0
            self._depth_vote_count = 0
            self._grasp_height_confirm_frames = 0
            self._reset_trial_grasp_watch()
            self.state = ServoState.ALIGNING_HORIZONTAL
            if pose_geometry is not None:
                desired_pose_sign = 1 if dx_px > 0 else -1
                if desired_pose_sign == self._pose_h_vote_sign:
                    self._pose_h_vote_count += 1
                else:
                    if self._pose_h_vote_sign:
                        self._pose_h_flip_count += 1
                    self._pose_h_vote_sign = desired_pose_sign
                    self._pose_h_vote_count = 1
                required_votes = 2 if abs_dx >= 45.0 else 3
                if (self._pose_h_flip_count >= 3
                        and self._pose_h_lower_count < 2):
                    self._pose_h_flip_count = 0
                    self._pose_h_vote_sign = 0
                    self._pose_h_vote_count = 0
                    self._pose_h_lower_count += 1
                    return self._make_move(
                        forward_mm=0, up_mm=-1, left_deg=0,
                        dx_px=dx_px, dy_px=dy_px,
                        claw_area=claw_area, target_area=target_area,
                        learn_axis=True,
                        reason=(
                            "pose horizontal estimate alternated repeatedly; "
                            f"lower 1mm for a clearer jaw view "
                            f"({self._pose_h_lower_count}/2)"
                        ),
                    )
                if self._pose_h_vote_count < required_votes:
                    return ServoCommand(
                        action="idle", state="pose_h_confirm",
                        reason=(
                            "confirming pose horizontal direction "
                            f"({self._pose_h_vote_count}/{required_votes}, "
                            f"dx={dx_px:.1f}px)"
                        ),
                        dx_px=dx_px, dy_px=dy_px,
                        claw_area=claw_area, target_area=target_area,
                    )
                self._pose_h_vote_count = 0
            gain = _adaptive_gain(abs_dx, h_thresh)
            # 过冲时减半步长
            if self._overshoot_count > 1:
                gain *= 0.5
            # Installed yaw has the opposite sign to image X.
            left_deg = -dx_px * self._yaw_deg_per_pixel * gain
            slow_left_limit = min(1.0 if pose_geometry is not None else 3.0,
                                  self.cal.max_left_per_step_deg)
            left_deg = max(-slow_left_limit, min(slow_left_limit, left_deg))
            # 最小步长
            if 0 < abs(left_deg) < 1.0:
                left_deg = math.copysign(1.0, left_deg)
            return self._make_move(
                forward_mm=0, up_mm=0, left_deg=int(round(left_deg)),
                dx_px=dx_px, dy_px=dy_px,
                claw_area=claw_area, target_area=target_area,
                learn_axis=True,
                reason=(f"align h: dx={dx_px:.1f}px gain={gain:.2f} "
                        f"map={self._yaw_deg_per_pixel:.4f}deg/px "
                        f"→ left={left_deg:.0f}deg"),
            )

        if not z_ready:
            self._grasp_height_confirm_frames = 0
            self._reset_trial_grasp_watch()
            # Coarse depth correction happens before hover-height correction.
            # Every new frame still rechecks horizontal alignment first.
            self.state = ServoState.APPROACHING
            approach_error_px = fingertip_gap_px - approach_gap_px
            # Learn direction slowly. Detector-size jitter must not reverse
            # physical depth motion after only one or two observations.
            if self._depth_feedback_pending:
                if (self._last_depth_abs_error is not None
                        and abs(approach_error_px) > self._last_depth_abs_error + 6.0):
                    self._depth_worse_frames += 1
                else:
                    self._depth_worse_frames = max(0, self._depth_worse_frames - 1)
                if self._depth_worse_frames >= 4:
                    self._depth_direction_sign *= -1
                    self._depth_worse_frames = 0
                self._depth_feedback_pending = False

            desired_sign = 1 if approach_error_px > 0 else -1
            if desired_sign == self._depth_vote_sign:
                self._depth_vote_count += 1
            else:
                self._depth_vote_sign = desired_sign
                self._depth_vote_count = 1
            if self._depth_vote_count < 3:
                return ServoCommand(
                    action="idle", state="depth_confirm",
                    reason=(
                        "depth estimate is uncalibrated; waiting for consistent "
                        f"direction ({self._depth_vote_count}/3)"
                    ),
                    dx_px=dx_px, dy_px=dy_px,
                    claw_area=claw_area, target_area=target_area,
                )

            forward_mm = float(desired_sign * self._depth_direction_sign)
            self._depth_vote_sign = 0
            self._depth_vote_count = 0
            self._last_depth_abs_error = abs(approach_error_px)
            self._depth_feedback_pending = True

            # Never mix depth and vertical motion in one command. Forward and
            # vertical axes are mechanically coupled, so observe after every
            # depth step before deciding the next correction.
            up_mm = 0.0
            return self._make_move(
                forward_mm=int(round(forward_mm)),
                up_mm=int(round(up_mm)), left_deg=0,
                dx_px=dx_px, dy_px=dy_px,
                claw_area=claw_area, target_area=target_area,
                reason=f"approach: area_ratio={area_ratio:.2f} depth={stereo_depth_mm} → forward={forward_mm:.0f}mm",
            )

        # Pose closing uses a generous horizontal rectangle below the
        # object center. Once inside it, descend one last millimetre and
        # re-observe before authorizing the close.
        if pose_geometry is not None:
            center_dx = float(pose_geometry["dx_px"])
            center_dy = float(pose_geometry["dy_px"])
            center_x_tol = float(pose_geometry["center_x_tolerance_px"])
            center_x_close_tol = center_x_tol * 1.25
            center_y_tol = float(pose_geometry["center_y_tolerance_px"])
            center_y_offset = float(pose_geometry["center_y_offset_px"])
            if abs(center_dx) > center_x_close_tol:
                self._grasp_height_confirm_frames = 0
                return ServoCommand(
                    action="idle", state="pose_center_x_recheck",
                    reason=(
                        f"jaw/object center x outside rectangle: "
                        f"dx={center_dx:.1f}px tol={center_x_close_tol:.1f}px"
                    ),
                    dx_px=center_dx, dy_px=center_dy,
                    claw_area=claw_area, target_area=target_area,
                )
            if abs(center_dy) > center_y_tol:
                self._grasp_height_confirm_frames = 0
                self.state = ServoState.FINE_ALIGNING
                up_mm = -1 if center_dy > 0 else 1
                return self._make_move(
                    forward_mm=0, up_mm=up_mm, left_deg=0,
                    dx_px=center_dx, dy_px=center_dy,
                    claw_area=claw_area, target_area=target_area,
                    learn_axis=True,
                    reason=(
                        f"pose expanded center rectangle: dy={center_dy:.1f}px "
                        f"tol={center_y_tol:.1f}px offset={center_y_offset:.1f}px "
                        f"up={up_mm:+d}mm"
                    ),
                )
            if not self._pose_final_lower_done:
                self._pose_final_lower_done = True
                self._grasp_height_confirm_frames = 0
                self.state = ServoState.FINE_ALIGNING
                return self._make_move(
                    forward_mm=0, up_mm=-1, left_deg=0,
                    dx_px=center_dx, dy_px=center_dy,
                    claw_area=claw_area, target_area=target_area,
                    learn_axis=True,
                    reason=(
                        "inside expanded close rectangle; final 1mm descent "
                        "before close verification"
                    ),
                )
            self._grasp_height_confirm_frames += 1
            if self._grasp_height_confirm_frames < 2:
                return ServoCommand(
                    action="idle", state="pose_center_confirm",
                    reason=(
                        "jaw midpoint is inside the expanded target rectangle; "
                        "confirming one more frame"
                    ),
                    dx_px=center_dx, dy_px=center_dy,
                    claw_area=claw_area, target_area=target_area,
                )
            self.state = ServoState.READY_TO_GRASP
            return ServoCommand(
                action="grasp",
                claw_open=self.cal.claw_closed_value,
                duration_ms=600,
                state="pose_center_ready",
                reason=(
                    f"expanded pose rectangle ready: dx={center_dx:.1f}/"
                    f"{center_x_close_tol:.1f}px dy={center_dy:.1f}/"
                    f"{center_y_tol:.1f}px offset={center_y_offset:.1f}px"
                ),
                dx_px=center_dx, dy_px=center_dy,
                claw_area=claw_area, target_area=target_area,
            )
        # Establish hover after coarse depth. This is bidirectional: if the
        # claw moved too low, command a small rise. Horizontal alignment is
        # rechecked first on every frame.
        if not v_aligned:
            self._grasp_height_confirm_frames = 0
            self._reset_trial_grasp_watch()
            self.state = ServoState.FINE_ALIGNING
            up_mm = -hover_error_px * self._vertical_mm_per_pixel * 0.45
            slow_vertical_limit = min(1.0, self.cal.max_up_per_step_mm)
            up_mm = max(-slow_vertical_limit, min(slow_vertical_limit, up_mm))
            if 0 < abs(up_mm) < 1.0:
                up_mm = math.copysign(1.0, up_mm)
            return self._make_move(
                forward_mm=0, up_mm=int(round(up_mm)), left_deg=0,
                dx_px=dx_px, dy_px=dy_px,
                claw_area=claw_area, target_area=target_area,
                learn_axis=True,
                reason=(f"hover correction: gap={fingertip_gap_px:.1f}px "
                        f"wanted={hover_gap_px:.1f}px "
                        f"map={self._vertical_mm_per_pixel:.3f}mm/px "
                        f"up={up_mm:.0f}mm"),
            )

        # Close-range descent: move the fingertips from the hover point into
        # the upper portion of the target, 1 mm at a time.  Never combine this
        # descent with forward or horizontal motion.
        desired_grasp_gap_px = -max(4.0, min(12.0, target_height * 0.15))
        grasp_height_error_px = fingertip_gap_px - desired_grasp_gap_px
        grasp_height_tolerance_px = max(6.0, min(10.0, target_height * 0.10))
        abs_grasp_height_error = abs(grasp_height_error_px)

        # A detector can stop improving a few pixels before the ideal closing
        # height. When already close and otherwise aligned, a deterministic
        # trial close is allowed after several stalled observations. The
        # session verifies it after lifting and releases/re-aligns on failure.
        trial_grasp_band_px = max(
            grasp_height_tolerance_px + 8.0,
            min(22.0, target_height * 0.35),
        )
        if abs_grasp_height_error <= trial_grasp_band_px:
            if (self._last_grasp_height_abs_error is not None
                    and abs_grasp_height_error
                    >= self._last_grasp_height_abs_error - 1.0):
                self._trial_grasp_stall_frames += 1
            else:
                self._trial_grasp_stall_frames = 0
        else:
            self._trial_grasp_stall_frames = 0
        self._last_grasp_height_abs_error = abs_grasp_height_error

        if (claw_keypoints is None
                and abs_grasp_height_error > grasp_height_tolerance_px
                and self._trial_grasp_stall_frames >= 2):
            self._grasp_height_confirm_frames = 0
            self._reset_trial_grasp_watch()
            self.state = ServoState.READY_TO_GRASP
            return ServoCommand(
                action="grasp",
                claw_open=self.cal.claw_closed_value,
                duration_ms=600,
                state="trial_grasp",
                reason=(f"verified trial grasp after stalled near-target error: "
                        f"dx={dx_px:.1f}px height_error={grasp_height_error_px:.1f}px"),
                dx_px=dx_px, dy_px=dy_px,
                claw_area=claw_area, target_area=target_area,
            )

        if abs(grasp_height_error_px) > grasp_height_tolerance_px:
            self._grasp_height_confirm_frames = 0
            self.state = ServoState.FINE_ALIGNING
            up_mm = -1.0 if grasp_height_error_px > 0 else 1.0
            return self._make_move(
                forward_mm=0, up_mm=int(round(up_mm)), left_deg=0,
                dx_px=dx_px, dy_px=dy_px,
                claw_area=claw_area, target_area=target_area,
                learn_axis=True,
                reason=(f"slow descend: gap={fingertip_gap_px:.1f}px "
                        f"wanted={desired_grasp_gap_px:.1f}px up={up_mm:.0f}mm"),
            )

        self._grasp_height_confirm_frames += 1
        if self._grasp_height_confirm_frames < 2:
            return ServoCommand(
                action="idle", state="grasp_height_confirm",
                reason="grasp height reached; confirming one more frame",
                dx_px=dx_px, dy_px=dy_px,
                claw_area=claw_area, target_area=target_area,
            )
        self.state = ServoState.READY_TO_GRASP
        return ServoCommand(
            action="grasp",
            claw_open=self.cal.claw_closed_value,
            duration_ms=600,
            state="ready_to_grasp",
            reason=(f"overhead jaw-point grasp ready: dx={dx_px:.1f}px "
                    f"tip_gap={fingertip_gap_px:.1f}px area_ratio={area_ratio:.2f}"),
            dx_px=dx_px, dy_px=dy_px,
            claw_area=claw_area, target_area=target_area,
        )

        return ServoCommand(
            action="idle", state="idle", reason="no action needed",
            dx_px=dx_px, dy_px=dy_px, claw_area=claw_area, target_area=target_area,
        )

    def note_executed(self) -> None:
        self._attempt_count += 1
        self._total_attempt_count += 1

    @property
    def attempts_exceeded(self) -> bool:
        return (self._attempt_count >= self.cal.max_attempts_per_step
                or self._total_attempt_count >= self.cal.max_total_attempts)

    def reset_step_counter(self) -> None:
        self._attempt_count = 0
        self._overshoot_count = 0
        self._last_dx_sign = 0

    def prepare_grasp_retry(self) -> None:
        """Keep depth calibration, but clear close-range convergence memory."""
        self._grasp_height_confirm_frames = 0
        self._reset_trial_grasp_watch()
        self._was_aligned_h = False
        self._was_aligned_v = False
        self._depth_feedback_pending = False
        self._pose_final_lower_done = False
        self.reset_step_counter()

    def _reset_trial_grasp_watch(self) -> None:
        self._trial_grasp_stall_frames = 0
        self._last_grasp_height_abs_error = None

    def _observe_axis_response(self, dx_px: float, hover_error_px: float,
                               approach_error_px: float) -> None:
        pending = self._pending_axis_response
        self._pending_axis_response = None
        if not pending:
            return
        alpha = 0.12
        axis = pending["axis"]
        command = abs(pending["command"])
        if axis == "yaw":
            pixel_response = abs(dx_px - pending["dx_px"])
            if command >= 1.0 and pixel_response >= 3.0:
                sample = max(0.02, min(0.16, command / pixel_response))
                self._yaw_deg_per_pixel = (
                    (1.0 - alpha) * self._yaw_deg_per_pixel + alpha * sample
                )
        elif axis == "vertical":
            pixel_response = abs(hover_error_px - pending["hover_error_px"])
            if command >= 1.0 and pixel_response >= 2.0:
                sample = max(0.15, min(1.20, command / pixel_response))
                self._vertical_mm_per_pixel = (
                    (1.0 - alpha) * self._vertical_mm_per_pixel + alpha * sample
                )

    def _queue_axis_response(self, forward_mm: float, up_mm: float,
                             left_deg: float) -> None:
        metrics = self._current_response_metrics
        if not metrics:
            return
        if left_deg and not forward_mm and not up_mm:
            axis, command = "yaw", left_deg
        elif up_mm and not forward_mm and not left_deg:
            axis, command = "vertical", up_mm
        else:
            return
        self._pending_axis_response = {
            "axis": axis,
            "command": float(command),
            **metrics,
        }

    def reset(self) -> None:
        self.state = ServoState.IDLE
        self._attempt_count = 0
        self._total_attempt_count = 0
        self._overshoot_count = 0
        self._last_error = ""
        self._was_aligned_h = False
        self._was_aligned_v = False
        self._was_aligned_z = False
        self._last_dx_sign = 0
        self._last_move = None
        self._rollback_count = 0
        self._motion_history = []
        self._probe_dir = -1
        self._grasp_height_confirm_frames = 0
        self._reset_trial_grasp_watch()
        self._approach_ready = False
        self._depth_direction_sign = 1
        self._depth_feedback_pending = False
        self._last_depth_abs_error = None
        self._yaw_deg_per_pixel = float(self.cal.deg_per_pixel)
        self._vertical_mm_per_pixel = float(self.cal.mm_per_pixel_vertical)
        self._pending_axis_response = None
        self._current_response_metrics = None
        self._depth_worse_frames = 0
        self._proximity_confirm_frames = 0
        self._depth_vote_sign = 0
        self._depth_vote_count = 0
        self._pose_h_vote_sign = 0
        self._pose_h_vote_count = 0
        self._pose_h_flip_count = 0
        self._pose_h_lower_count = 0
        self._pose_final_lower_done = False

    def status(self) -> ServoStatus:
        return ServoStatus(
            state=self.state.value,
            last_error=self._last_error,
            attempt=self._attempt_count,
            total_attempts=self._total_attempt_count,
        )

    # ── 内部方法 ──────────────────────────────────────────

    def _estimate_forward(self, area_ratio: float,
                          stereo_depth_mm: float | None = None) -> float:
        """估算前进距离。有立体深度则直接用，否则基于面积比。"""
        if stereo_depth_mm is not None and stereo_depth_mm > 0:
            # 立体深度: 目标距离 - 爪子当前 reach (~250mm)
            target_dist = stereo_depth_mm
            current_reach = 250.0  # home position reach
            needed = target_dist - current_reach + 15.0  # +15mm 余量
            return max(5.0, min(needed, self.cal.max_forward_per_step_mm))

        deficit = self.cal.grasp_area_ratio_threshold - area_ratio
        if deficit <= 0:
            return 0.0
        forward = deficit / max(self.cal.area_gain_per_forward_mm, 1e-6)
        # 衰减
        if area_ratio > 0.25:
            forward *= 0.65
        if area_ratio > 0.35:
            forward *= 0.55
        if area_ratio > 0.45:
            forward *= 0.40
        return max(5.0, min(forward, self.cal.max_forward_per_step_mm))

    def _make_move(self, forward_mm, up_mm, left_deg,
                   dx_px, dy_px, claw_area, target_area, reason,
                   learn_axis=False) -> ServoCommand:
        duration = self.cal.step_duration(forward_mm, up_mm, left_deg)
        # 记录用于丢失回退
        self.record_move(forward_mm, up_mm, left_deg)
        if learn_axis:
            self._queue_axis_response(forward_mm, up_mm, left_deg)
        return ServoCommand(
            action="move",
            forward_mm=int(forward_mm),
            up_mm=int(up_mm),
            left_deg=int(left_deg),
            claw_open=-1,
            duration_ms=duration,
            state=self.state.value,
            reason=reason,
            dx_px=dx_px, dy_px=dy_px,
            claw_area=claw_area, target_area=target_area,
        )


def _adaptive_gap(current: float, target: float) -> float:
    """类似 _adaptive_gain 但针对比例值。"""
    if current >= target:
        return 0.0
    ratio = current / max(target, 1e-6)
    return 0.25 + 0.75 / (1.0 + math.exp(-4.0 * (ratio - 0.5)))
