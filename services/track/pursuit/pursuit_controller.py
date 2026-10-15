# -*- coding: utf-8 -*-
"""视觉追踪逼近状态机与运动控制律。

纯逻辑模块，不直接接触硬件：
- TRACKING：由 3090 返回的 bbox/深度计算视线偏角与逼近速度；
- 丢失后 SEARCHING：停车，按确定性角度序列原地旋转（用相邻圈 ICP
  确认已转过对应角度），每步由外部重新提交 Qwen 识别；转满仍未找到
  即确认停车（GIVE_UP）；
- 最终停车：双目深度与雷达前向距离交叉确认 ≤ stop_distance_m。

所有运动命令都带短 TTL，交给 run_visual_pursuit 层下发。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping.geometry import Pose2D
from raspberry_pi.mapping.scan_filter import sector_min_distance
from raspberry_pi.mapping.scan_matcher import IcpScanMatcher

from pursuit.pursuit_config import PursuitConfig, StereoConfig
from pursuit.visual_step_feedback import VisualStepFeedback
from pursuit.forward_reference import reference_error


class State(Enum):
    INIT = auto()          # 等待首次 Qwen 识别结果
    TRACKING = auto()      # 持续追踪逼近
    SEARCHING = auto()     # 目标丢失，雷达旋转重搜
    STOPPED = auto()       # 到达停止距离，任务成功
    GIVE_UP = auto()       # 重搜转满/障碍物过近，确认停车


@dataclass
class Perception:
    """3090 每帧返回的追踪结果。"""
    tracking: bool
    bbox: tuple[float, float, float, float] | None  # 整幅拼接图坐标
    depth_m: float | None
    depth_conf: float = 0.0
    status: str = ""
    # 第二路（另一摄像头）追踪结果：示教停车需要两路中心同时对齐
    tracking2: bool = False
    bbox2: tuple[float, float, float, float] | None = None
    status2: str = ""
    # 3090 双目被动重识别刚刚找回了原目标（短暂连续帧标志）
    reidentified: bool = False
    # 主路追踪来源（qwen/yolo/csrt），日志诊断用
    source: str = ""
    # 树莓派收到该结果的本地单调时钟与相邻结果周期。0 表示旧调用方
    # 未提供（测试/兼容模式），不启用年龄门控。
    frame_id: int = 0
    received_at: float = 0.0
    frame_interval_s: float = 0.0


@dataclass(frozen=True)
class Motion:
    linear_mm_s: int = 0
    angular_mrad_s: int = 0
    ttl_ms: int = 600
    reason: str = ""


@dataclass
class Decision:
    motion: Motion
    reground_now: bool = False       # 外部应停车拍照并重新提交 Qwen
    finished: bool = False
    result: str = ""                 # "" / "success" / "give_up"
    detail: str = ""


class PursuitController:
    def __init__(self, pursuit: PursuitConfig, stereo: StereoConfig,
                 frame_half_width: int,
                 simulate_motion: bool = False, frame_height: int | None = None) -> None:
        self.cfg = pursuit
        self.stereo = stereo
        self.frame_half_width = max(1, int(frame_half_width))
        self.frame_height = max(1,int(frame_height if frame_height is not None
                                     else self.frame_half_width*720/1280))
        # dry-run 下车实际不动，雷达 ICP 测不到转角：SEARCHING 旋转
        # 改为按指令角速度的时长估算，保证状态机可以走完。
        self._simulate_motion = bool(simulate_motion)
        # 运动链路（BLE）就绪标志：未就绪时跳过 stuck 卡死检测
        self._link_ready = True
        self.state = State.INIT
        self._last_tick: float | None = None

        # TRACKING 状态量
        self._stop_frames = 0
        self._lost_since = 0.0
        self._last_bbox: tuple[float, float, float, float] | None = None
        # 平滑漂移看门狗（2026-08-17 安全修复）：单帧 120px 跳变
        # 保护会被 19px/帧 的平滑漂移骗过（实测 1s 漂 475px）。对
        # 横向中心速度做 EMA（扣除车自身转向的像面期望），持续
        # 过大且追踪源未被外观确认（csrt/kf）→ 视为漂移，与跳变
        # 同样处理（停车等重新确认）。
        self._drift_ema = 0.0
        self._drift_frames = 0
        self._last_px_t: float | None = None
        self._last_cmd_angular = 0
        # 视觉丢失回退：只保存最近双目确认帧的目标平均水平
        # 位置，供“两路当前都丢失”时预测回退方向。
        self._recent_dual_centers: list[tuple[float, float]] = []
        self._recent_motion: list[tuple[float, int, int]] = []
        self._recent_visual_approach_at = 0.0
        self._last_dual_visual_signature: tuple[float, float, float, float] | None = None
        self._last_near_blind_at = 0.0
        self._last_history_frame_id = -1
        self._recovery_started_at: float | None = None
        self._recovery_pulse_until: float | None = None
        self._recovery_next_pulse_at = 0.0
        self._recovery_turn_sign = 0
        self._recovery_dual_frames = 0
        self._recovery_last_confirm_frame_id = -1
        self._recovery_from_blind = False
        self._recovery_pulses = 0
        self._recovery_swept_used = False
        self._recovery_swept_turn = 0
        self._recovery_swept_estimated_m = 0.0
        self._recovery_swept_pulses = 0
        self._recovery_swept_last_frame_id = -1
        self._turn_rollback_baseline: tuple[float, float] | None = None
        self._turn_rollback_active = False
        self._turn_rollback_edge_x: float | None = None
        # 转向 EMA 平滑（2026-08-17 流畅性）：服务端感知 ~7-10fps，
        # 角速度指令逐帧跳变会让车转向猛甩；0.5/0.5 EMA 摊到 ~120ms
        self._last_angular_out: int | None = None
        # 近距蠕动状态（2026-08-17 修复近距卡死）：脉冲/休息节奏
        self._creep_pulse_until: float | None = None
        self._creep_next_pulse_at = 0.0
        self._creep_backing = False
        self._creep_active_hold_s = 0.0
        # 视觉近距/低感知帧率下的时间占空比伺服。
        self._visual_pulse_until: float | None = None
        self._visual_next_pulse_at = 0.0
        self._visual_active_hold_s = 0.0
        self._visual_feedback = VisualStepFeedback()
        self._parking_step_sample = None
        self._stationary_recheck_pending = False
        self._stationary_recheck_count = 0
        self._stationary_recheck_deadline = 0.0
        self._stationary_recheck_near = False
        self._near_recheck_handoff = False
        self._reground_confirm_frames = 0
        self._reground_confirm_frame_id = -1
        self._reground_confirm_rx_at = 0.0
        self._reground_confirm_after = 0.0
        self._search_pulse_until = 0.0
        self._search_next_pulse_at = 0.0
        self._search_total_on_s = 0.0
        # 原地转向脉冲：先对准，再直行，避免直线脉冲把目标推离视野。
        self._align_turn_until: float | None = None
        self._align_next_turn_at = 0.0
        self._align_turn_sign = 0

        # 平滑起步（用户要求）：静止→先给 linear_breakaway_mm_s 冲量
        # 破静摩擦（保持 breakaway_hold_s），之后按 linear_accel_mm_s2
        # 爬升到目标速度；减速立即生效。
        self._smoothed_linear = 0.0
        self._kick_until: float | None = None
        # 雷达卡死检测：命令在走但 ICP 无位移 → 逐级提高冲量
        self._stall_icp = IcpScanMatcher(
            max_correspondence_m=self.cfg.search_icp_max_correspondence_m,
        )
        self._stall_ref_points: Any = None
        self._stall_ref_visual: tuple[float, float, float, float] | None = None
        self._stall_check_at: float | None = None
        self._stuck_windows = 0
        self._stuck_level = 0
        self._moving_windows = 0
        self._lidar_stop_frames = 0
        # 当前帧横向偏差绝对值（软门槛用）
        self._last_align_err: float | None = None
        self._last_bearing_errors = None

        # SEARCHING 状态量
        self._step_idx = 0
        self._rotating = False
        self._awaiting_ground = False
        # 重搜序列的中心偏移（丢失瞬间回瞄目标最后方位），度
        self._search_base_deg = 0.0
        # 搜索转满 GIVE_UP 后自动 Qwen 复检一次的挂起标记
        self._giveup_reground_pending = False
        # 旋转中服务端视觉追回的连续确认帧计数
        self._reacquire_frames = 0
        self._terminal_reid_frames = 0
        # 曾成功建立过目标身份后，旋转搜索以服务端实时双目重识别为主，
        # 不在每个角度都阻塞运行 Qwen。
        self._has_identity = False
        self._search_hold_until: float | None = None
        self._ref_points: Any = None
        self._yaw_since_start = 0.0
        self._icp = IcpScanMatcher(
            max_correspondence_m=self.cfg.search_icp_max_correspondence_m,
        )
        self._stall_deadline: float | None = None
        self._kick = False
        self._step_checked_obstacle = False
        self._rotation_fallback_deadline: float | None = None
        # ICP 持续不收敛（空旷/对称环境特征不足）时的退化计数：
        # 连续若干卡死窗口都没收敛过一次就改按时长估算转角，
        # 避免车在实车模式下无限原地旋转。
        self._icp_converged_once = False
        self._icp_fail_windows = 0

    def set_simulate_motion(self, value: bool) -> None:
        """dry-run 或运动链路未就绪时由主程序切换为时长估算转角。"""
        self._simulate_motion = bool(value)

    def set_link_ready(self, value: bool) -> None:
        """运动链路（BLE）是否就绪。未就绪时命令不下发、车不会动，
        stuck 卡死检测没有意义（否则 ESP32 恢复广播前的等待期会被
        误判卡死 GIVE_UP——2026-08-17 实测第八轮 36s 误杀）。
        就绪后自动恢复检测。"""
        self._link_ready = bool(value)

    # ---------------- 对外接口 ----------------

    def on_ground_result(self, found: bool, now: float) -> None:
        """首次识别或旋转重搜后的识别结果。"""
        self._parking_step_sample = None  # never carry an old target's step
        was_recheck = self._stationary_recheck_pending
        near_recheck = was_recheck and self._stationary_recheck_near
        self._stationary_recheck_pending = False
        if was_recheck:
            self._last_tick = now
            self._reset_visual_pulse()
            self._reset_align_turn()
            self._lost_since = 0.0
            self._last_bbox = None
            self._reset_recovery(clear_history=False)
            if found:
                self._has_identity = True
                self.state = State.TRACKING
                self._near_recheck_handoff = near_recheck
                self._reground_confirm_frames = 0
                self._reground_confirm_frame_id = -1
                self._reground_confirm_rx_at = 0.0
                self._reground_confirm_after = now
                self._stationary_recheck_deadline = now + self.cfg.lost_stationary_recheck_timeout_s
            else:
                # Failed stationary observation is not permission to turn or
                # reverse immediately. Preserve recent approach evidence.
                self.state = State.TRACKING
                self._recovery_from_blind = near_recheck
            return
        if found:
            self._has_identity = True
        if self.state in (State.STOPPED, State.GIVE_UP):
            if not found:
                return
            # 任务已结束（成功/卡死）：网页"重新识别"找到目标 → 重启追踪
            self.state = State.TRACKING
            self._lost_since = 0.0
            self._stop_frames = 0
            self._step_idx = 0
            self._rotating = False
            self._awaiting_ground = False
            self._stuck_level = 0
            self._stuck_windows = 0
            self._stall_ref_points = None
            self._stall_ref_visual = None
            self._stall_check_at = None
            self._giveup_reground_pending = False
            self._terminal_reid_frames = 0
            self._reset_step()
            self._reset_recovery(clear_history=True)
            return
        if found:
            self.state = State.TRACKING
            self._lost_since = 0.0
            self._stop_frames = 0
            self._step_idx = 0
            self._rotating = False
            self._awaiting_ground = False
            self._stuck_level = 0
            self._stuck_windows = 0
            self._moving_windows = 0
            self._last_bbox = None  # 新锁不跟旧锁比跳变
            self._stall_ref_points = None
            self._stall_ref_visual = None
            self._stall_check_at = None
            self._reset_step()
            self._reset_recovery(clear_history=True)
        elif self.state == State.INIT:
            # 初始画面就没有目标：进入旋转重搜流程
            self._enter_searching()
        elif self.state == State.TRACKING:
            # 测试网页用新指令重新识别但未找到：按目标丢失处理，旋转重搜
            self._enter_searching()
        else:
            # SEARCHING 中这一步没找到：转下一步
            self._step_idx += 1
            if self._step_idx >= len(self.cfg.search_rotation_steps_deg):
                self.state = State.GIVE_UP
                self._awaiting_ground = False
                # 转满仍未找到：GIVE_UP 停车后再自动 Qwen 复检一次
                # （目标可能在最后一步旋转中才进入画面）
                self._giveup_reground_pending = True
                return
            self._awaiting_ground = False
            self._rotating = False
            self._reset_step()

    def tick(self, perception: Perception | None, scan: LaserScan | None,
             now: float) -> Decision:
        if self._last_tick is None:
            self._last_tick = now
        dt = now - self._last_tick
        self._last_tick = now

        lidar_front = self._lidar_front(scan)
        # 旋转搜索中或搜索耗尽后，3090 的双目身份重识别已经同时满足
        # 指令属性、不可变初始外观和双目几何。直接用连续强标志恢复，
        # 不再停车后额外跑一次 Qwen（1~3 秒延迟会让移动目标再次离开）。
        # 普通残留 tracking 没有 reidentified 标志，不能唤醒这些状态。
        if self.state in (State.SEARCHING, State.GIVE_UP):
            trusted_reid = bool(
                perception is not None
                and perception.reidentified
                and perception.tracking
                and perception.tracking2)
            self._terminal_reid_frames = (
                self._terminal_reid_frames + 1 if trusted_reid else 0)
            required = 2 if self.state == State.SEARCHING else 3
            if self._terminal_reid_frames >= required:
                self.on_ground_result(True, now)
                self._terminal_reid_frames = 0
        # 硬停止：任何时刻前方物体过近立即停车（已到任务终点除外）。
        if (lidar_front is not None
                and lidar_front < self.cfg.lidar_hard_floor_m
                and self.state not in (State.STOPPED, State.GIVE_UP)):
            self.state = State.GIVE_UP
            decision = Decision(
                Motion(0, 0, self.cfg.ttl_near_ms, "hard floor"),
                finished=True, result="give_up",
                detail=f"前方 {lidar_front:.2f}m 物体过近，停车",
            )
        elif self.state in (State.STOPPED, State.GIVE_UP):
            reason = "finished" if self.state == State.STOPPED else "give up"
            reground = False
            if self.state == State.GIVE_UP and self._giveup_reground_pending:
                # 搜索转满后的自动 Qwen 复检（只触发一次）
                self._giveup_reground_pending = False
                reground = True
            decision = Decision(Motion(0, 0, self.cfg.ttl_near_ms, reason),
                                reground_now=reground,
                                finished=True,
                                result="success" if self.state == State.STOPPED else "give_up",
                                detail=reason)
        elif self.state == State.INIT:
            decision = Decision(Motion(0, 0, self.cfg.ttl_far_ms,
                                       "waiting initial ground"))
        elif self.state == State.TRACKING:
            decision = self._tick_tracking(
                perception, scan, lidar_front, dt, now)
        else:
            decision = self._tick_searching(perception, scan, now)
        # ---- 雷达停车兜底（2026-08-17 修复"示教停车未达成"）----
        # 示教/行测距停车都需要视觉锁稳定走到近距，而锁漂/第二路
        # 丢失时这些判定永远不触发（实测从未到 0.2m）。雷达对前进
        # 方向物体测距可靠（逼近时读数单调下降与车-目标距离一致），
        # 统一在 TRACKING 决策后加雷达 0.2m 停车判定：三种测距模式
        # 都不再依赖视觉锁的健康度，安全到达。
        if (not self.cfg.use_taught_stop
                and self.state == State.TRACKING and not decision.finished
                and lidar_front is not None
                and lidar_front <= self.cfg.stop_distance_m + 0.05):
            self._lidar_stop_frames += 1
            angular = int(max(-self.cfg.max_angular_near_mrad_s,
                              min(self.cfg.max_angular_near_mrad_s,
                                  decision.motion.angular_mrad_s)))
            if self._lidar_stop_frames >= self.cfg.stop_confirm_frames:
                self.state = State.STOPPED
                decision = Decision(
                    Motion(0, 0, self.cfg.ttl_near_ms, "stop reached"),
                    finished=True, result="success",
                    detail=f"雷达 {lidar_front:.2f}m ≤ "
                           f"{self.cfg.stop_distance_m}m，停车")
            else:
                decision = Decision(Motion(
                    0, angular, self.cfg.ttl_near_ms,
                    f"lidar {lidar_front:.2f}m near stop"))
        elif self.state == State.TRACKING:
            # 停车计数滞回：计数一旦开始（≤0.25m），雷达噪声抖动
            # （0.24↔0.26）不会重置计数——否则车在停车线边缘反复
            # 蠕动脉冲-停，永远凑不满 3 帧确认（2026-08-17 review）。
            # 只有明显退远（>0.30m）才重置。
            if lidar_front is None \
                    or lidar_front > self.cfg.stop_distance_m + 0.10:
                self._lidar_stop_frames = 0
        # ---- 雷达近距硬限速（2026-08-17 安全修复）----
        # 行测距在近距会失效（目标在画面中部时行 y 不随距离增加，
        # 速度降不下来，车全速冲撞；实测 0.68m 仍 280mm/s）。雷达
        # 测距可靠：<1.0m 限 100mm/s、<0.6m 限 50mm/s，对 TRACKING
        # 直行决策生效（SEARCHING 无直行不受影响）。
        if (not decision.finished
                and decision.motion.linear_mm_s > 0
                and lidar_front is not None):
            near_cap = 50 if lidar_front < 0.6 else \
                100 if lidar_front < 1.0 else None
            if near_cap is not None \
                    and decision.motion.linear_mm_s > near_cap:
                decision = Decision(Motion(
                    int(near_cap), decision.motion.angular_mrad_s,
                    decision.motion.ttl_ms,
                    f"lidar {lidar_front:.2f}m near cap"),
                    finished=decision.finished,
                    result=decision.result,
                    detail=decision.detail,
                    reground_now=decision.reground_now)
        decision = self._soft_align_gate(decision)
        decision = self._smooth_linear(decision, dt, now)
        decision, align_turn_active = self._alignment_turn_decision(
            decision, perception, now)
        # 近距蠕动逼近（2026-08-17 修复近距卡死）：低速巡航
        # （<160mm/s）低于静摩擦临界推不动，stuck 升级链反复触发、
        # 车在 0.34-0.54m 反复蠕动到 GIVE_UP。脉冲/休息交替，放在
        # 平滑之后直接覆盖（脉冲不经限速/平滑/align）。
        # 注意：雷达停车确认期（_lidar_stop_frames>0，lidar≤0.25m
        # 等待 3 帧确认）不蠕动——否则停车确认期的"停住"会被脉冲
        # 覆盖，车在 0.25m 处继续冲、过冲到 0.10m 触发硬停 GIVE_UP。
        lidar_creep = (not (self.cfg.use_taught_stop and self.cfg.pulse_drive_enabled)
                and not align_turn_active
                and self.state == State.TRACKING and not decision.finished
                and decision.motion.linear_mm_s > 0
                and self._lidar_stop_frames == 0
                and lidar_front is not None
                and lidar_front < self.cfg.creep_zone_m)
        if lidar_creep:
            decision = self._creep_decision(
                decision, lidar_front, now, perception)
        else:
            # 丢框/旧帧/框跳变等停车决策必须同时取消尚未结束的雷达
            # 蠕动周期，否则旧 pulse 会在下一 tick 重新把车向前推。
            if decision.motion.linear_mm_s <= 0:
                self._creep_pulse_until = None
                self._creep_next_pulse_at = 0.0
                self._creep_backing = False
                self._creep_active_hold_s = 0.0
        if (not align_turn_active and not lidar_creep
              and self.state == State.TRACKING and not decision.finished
              and self._visual_near_phase(perception)):
            # 雷达看不到低矮目标时也能近距小步逼近：每次只走 2~5cm，
            # 停稳期间等待新框，再决定下一步。
            decision = self._visual_pulse_decision(
                decision, perception, now, near=True)
        elif (not align_turn_active and not lidar_creep
              and self.state == State.TRACKING and not decision.finished):
            decision = self._blind_travel_gate(decision, perception, now)
        else:
            self._reset_visual_pulse()
        # 必须在蠕动/视觉脉冲覆盖之后，按“实际将要下发的线速度”判断
        # 卡死。旧顺序在 creep rest（实际 v=0）期间仍拿覆盖前的正速度
        # 累计卡死，车已经移动也会最终误进 GIVE_UP。
        if (not decision.finished
                and decision.motion.linear_mm_s > 0
                and self.state == State.TRACKING
                and self._link_ready
                and not self._simulate_motion
                and not decision.motion.reason.startswith("visual servo pulse")):
            stalled = self._check_stalled(scan, now, perception)
            if stalled is True:
                self._stuck_windows += 1
                self._moving_windows = 0
                if self._stuck_windows >= self.cfg.stuck_escalate_windows:
                    self._stuck_windows = 0
                    self._stuck_level += 1
                    self._smoothed_linear = 0.0
                    if self._stuck_level > self.cfg.stuck_max_escalations:
                        self.state = State.GIVE_UP
                        decision = Decision(
                            Motion(0, 0, self.cfg.ttl_near_ms, "stuck"),
                            finished=True, result="give_up",
                            detail="雷达判定实际前进脉冲连续未产生位移，停车",
                        )
                    else:
                        decision.motion = Motion(
                            decision.motion.linear_mm_s,
                            decision.motion.angular_mrad_s,
                            decision.motion.ttl_ms,
                            decision.motion.reason
                            + f"; lidar no-motion -> next level "
                              f"{self._stuck_level}")
            elif stalled is False:
                self._stuck_windows = 0
                self._moving_windows += 1
                if self._moving_windows >= 2:
                    self._moving_windows = 0
                    self._stuck_level = max(0, self._stuck_level - 1)
                    decision.motion = Motion(
                        decision.motion.linear_mm_s,
                        decision.motion.angular_mrad_s,
                        decision.motion.ttl_ms,
                        decision.motion.reason
                        + f"; progress observed -> level {self._stuck_level}")
            # None = 窗口未到期或雷达配准不确定；不能把它当作已移动
        # 转向 EMA 平滑（2026-08-17 流畅性）：服务端 ~7-10fps 的感知
        # 更新让角速度指令逐帧跳变，车转向猛甩。0.5/0.5 EMA 把跳变
        # 摊到 ~120ms；只作用于 TRACKING（搜索旋转是定速，不受影响）。
        if (not decision.finished and self.state == State.TRACKING
                and (decision.motion.linear_mm_s != 0
                     or decision.motion.angular_mrad_s != 0)):
            desired = decision.motion.angular_mrad_s
            # 高扭矩转向脉冲必须原样下发；若 EMA 把 1600 首帧削成
            # 800，就会重新落回实测无法克服侧向静摩擦的区间。
            align_kick = decision.motion.reason.startswith("align arc pulse")
            # A bounded reverse rollback already has its own small yaw cap;
            # EMA must not halve its first (and sometimes only) correction.
            if (not align_kick and decision.motion.linear_mm_s >= 0
                    and self._last_angular_out is not None) \
                    and abs(self._last_angular_out - desired) >= 10:
                desired = int(round(
                    0.5 * self._last_angular_out + 0.5 * desired))
            self._last_angular_out = desired
            if desired != decision.motion.angular_mrad_s:
                decision.motion = Motion(
                    decision.motion.linear_mm_s, desired,
                    decision.motion.ttl_ms, decision.motion.reason)
        else:
            self._last_angular_out = 0
        # 记录实际下发的角速度：漂移看门狗用它扣除车自身转向的像面期望
        self._last_cmd_angular = decision.motion.angular_mrad_s
        if (self._link_ready and not self._simulate_motion
                and (decision.motion.linear_mm_s or decision.motion.angular_mrad_s)):
            self._recent_motion.append((now, decision.motion.linear_mm_s,
                                        decision.motion.angular_mrad_s))
            del self._recent_motion[:-80]
            if (decision.motion.linear_mm_s > 0
                    and abs(decision.motion.angular_mrad_s) >= 100
                    and self._recent_dual_centers
                    and (self._turn_rollback_baseline is None or
                         now-self._turn_rollback_baseline[0]
                         > self.cfg.recovery_turn_rollback_max_age_s)):
                self._turn_rollback_baseline = (
                    now, self._recent_dual_centers[-1][1])
        return decision

    def _visual_near_phase(self, perception: Perception | None) -> bool:
        """是否进入不依赖雷达的视觉脉冲区。"""
        if perception is None or not perception.tracking \
                or perception.bbox is None:
            return False
        if self.cfg.use_taught_stop and perception.bbox2 is not None:
            return self._near_handoff_rows(perception, margin=80.0*self._frame_scale_y()) or max(self._taught_mask_distance(perception.bbox),
                       self._taught_mask_distance(perception.bbox2)) \
                <= self.cfg.taught_slow_band_px
        if self.cfg.use_visual_row_distance:
            cy = (perception.bbox[1] + perception.bbox[3]) / 2.0
            return cy >= self.cfg.visual_near_row_px
        return False

    def _frame_scale_y(self) -> float:
        return self.frame_height/720.0

    def _near_handoff_rows(self, perception: Perception | None, margin: float = 0.0) -> bool:
        if (not self.cfg.visual_align_handoff_enabled or perception is None
                or not perception.tracking or not perception.tracking2
                or perception.bbox is None or perception.bbox2 is None):
            return False
        return ((perception.bbox[1]+perception.bbox[3])/2 >= self.cfg.visual_align_handoff_left_row_px-margin
                and (perception.bbox2[1]+perception.bbox2[3])/2 >= self.cfg.visual_align_handoff_right_row_px-margin)

    def _close_to_taught_mask(self, perception: Perception | None) -> bool:
        if (not self.cfg.use_taught_stop or perception is None
                or not perception.tracking or perception.bbox is None
                or not perception.tracking2 or perception.bbox2 is None):
            return False
        return max(self._taught_mask_distance(perception.bbox),
                   self._taught_mask_distance(perception.bbox2)) \
            <= self.cfg.taught_close_pulse_band_px

    def _near_blind_guard(self, *boxes: tuple[float, float, float, float]) -> bool:
        """目标已经又大又靠画面下缘；再前进会冲进双目近端盲区。"""
        return any(
            (box[1] + box[3]) / 2.0 >= self.cfg.visual_blind_guard_row_px
            and box[3] - box[1] >= self.cfg.visual_blind_guard_box_height_px
            for box in boxes
        )

    def _visual_hold_cap(self, perception: Perception | None) -> float:
        if self._near_handoff_rows(perception, margin=80.0*self._frame_scale_y()):
            return self.cfg.visual_align_near_pulse_max_s
        if self._close_to_taught_mask(perception):
            distance = max(self._taught_mask_distance(perception.bbox),
                           self._taught_mask_distance(perception.bbox2))
            return (self.cfg.taught_close_pulse_max_s
                    if distance <= self.cfg.taught_fine_pulse_band_px
                    else self.cfg.taught_stationary_trial_max_s)
        return self.cfg.visual_feedback_max_s

    def _stop_at_blind_guard(self, detail: str | None = None) -> Decision:
        self.state = State.GIVE_UP
        return Decision(
            Motion(0, 0, self.cfg.ttl_near_ms, "near visual blind zone"),
            finished=True, result="give_up",
            detail=detail or "目标已接近双目近端盲区但未满足双目抓取掩码，安全停车；需检查相机顺序或目标位置",
        )

    def _reset_visual_pulse(self) -> None:
        self._visual_pulse_until = None
        self._visual_next_pulse_at = 0.0
        self._visual_active_hold_s = 0.0
        self._visual_feedback.cancel_pending()

    def _reset_align_turn(self) -> None:
        self._align_turn_until = None
        self._align_active_hold_s = 0.0
        self._align_next_turn_at = 0.0
        self._align_turn_sign = 0

    def _alignment_turn_decision(self, decision: Decision,
                                 perception: Perception | None,
                                 now: float) -> tuple[Decision, bool]:
        """偏差大时先做原地转向脉冲；返回 (decision, 是否接管)。"""
        eligible = (
            self.cfg.pulse_drive_enabled
            and not decision.finished
            and decision.motion.linear_mm_s > 0
            and decision.motion.angular_mrad_s != 0
            and self._last_align_err is not None
            and self._last_align_err >= self.cfg.align_turn_trigger_px
            and perception is not None
            and perception.tracking
        )
        if not eligible:
            self._reset_align_turn()
            return decision, False

        sign = 1 if decision.motion.angular_mrad_s > 0 else -1
        # 误差越过画面中心、转向方向反转时，立即取消旧周期并按新方向转。
        if self._align_turn_sign not in (0, sign):
            self._reset_align_turn()
        self._align_turn_sign = sign
        if now >= self._align_next_turn_at:
            pulse_s = min(
                self.cfg.align_turn_pulse_max_s,
                self.cfg.align_turn_pulse_s
                + self._stuck_level * self.cfg.align_turn_pulse_step_s,
            )
            if self._near_handoff_rows(perception,margin=80.0*self._frame_scale_y()):
                pulse_s=min(pulse_s,self.cfg.visual_align_near_pulse_max_s)
            self._align_turn_until = now + pulse_s
            self._align_active_hold_s=pulse_s
            self._align_next_turn_at = (
                now + pulse_s
                + self.cfg.align_turn_rest_s)
        if (self._align_turn_until is not None and self._near_handoff_rows(perception,margin=80.0*self._frame_scale_y())
                and self._align_active_hold_s>self.cfg.visual_align_near_pulse_max_s):
            started=self._align_turn_until-self._align_active_hold_s
            self._align_active_hold_s=self.cfg.visual_align_near_pulse_max_s
            self._align_turn_until=min(self._align_turn_until,started+self._align_active_hold_s)
            self._align_next_turn_at=min(self._align_next_turn_at,self._align_turn_until+self.cfg.align_turn_rest_s)
        pulsing = (self._align_turn_until is not None
                   and now < self._align_turn_until)
        if pulsing:
            angular = sign * self.cfg.align_turn_pulse_mrad_s
            return Decision(Motion(
                self.cfg.align_arc_linear_mm_s, angular,
                self.cfg.ttl_near_ms,
                f"align arc pulse err={self._last_align_err:.0f}px")), True
        return Decision(Motion(
            0, 0, self.cfg.ttl_near_ms,
            f"align arc rest/measure err={self._last_align_err:.0f}px")), True

    def _visual_pulse_decision(self, decision: Decision,
                               perception: Perception | None,
                               now: float, near: bool) -> Decision:
        """以可靠的较高电机速度做短脉冲，用占空比获得低平均速度。"""
        if decision.motion.linear_mm_s <= 0:
            self._reset_visual_pulse()
            return decision
        if (perception is not None and perception.received_at > 0.0
                and now - perception.received_at
                > self.cfg.perception_max_age_s):
            self._reset_visual_pulse()
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "visual wait fresh perception"))
        if now >= self._visual_next_pulse_at:
            signature = self._stall_visual_signature(perception)
            if self._link_ready and not self._simulate_motion and self._visual_feedback.baseline is not None:
                feedback_state=self._visual_feedback.complete(signature,
                    max_level=self.cfg.visual_feedback_max_levels)
                self._parking_step_sample=((now,self._visual_feedback.last_change_px)
                    if feedback_state!='unknown' and signature is not None else None)
            elif not self._link_ready or self._simulate_motion:
                self._visual_feedback.cancel_pending()
            if self._visual_feedback.exhausted:
                self.state = State.GIVE_UP
                return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                    "visual startup exhausted"), finished=True, result="give_up",
                    detail="405速度下受限起步脉冲仍无双目视觉进展，已停车；未持续提速")
            base_s = (self.cfg.visual_pulse_near_s if near
                      else self.cfg.visual_pulse_far_s)
            pulse_s = min(
                self.cfg.visual_pulse_max_s, self.cfg.visual_feedback_max_s,
                base_s + self._visual_feedback.level * self.cfg.visual_feedback_step_s,
            )
            if self._near_handoff_rows(perception,margin=80.0*self._frame_scale_y()):
                pulse_s=min(pulse_s,self.cfg.visual_align_near_pulse_max_s)
            if self._close_to_taught_mask(perception):
                pulse_s = min(pulse_s, self._visual_hold_cap(perception))
            self._visual_active_hold_s = pulse_s
            self._visual_feedback.baseline = (signature
                if self._link_ready and not self._simulate_motion else None)
            self._visual_pulse_until = now + pulse_s
            self._visual_next_pulse_at = (
                now + pulse_s + self.cfg.visual_pulse_rest_s)
        # 若脉冲中途才跨入近距带，也立刻截短剩余通电时间。
        if ((self._close_to_taught_mask(perception) or self._near_handoff_rows(perception,margin=80.0*self._frame_scale_y()))
                and self._visual_pulse_until is not None
                and self._visual_active_hold_s
                > self._visual_hold_cap(perception)):
            started = self._visual_pulse_until - self._visual_active_hold_s
            self._visual_active_hold_s = self._visual_hold_cap(perception)
            self._visual_pulse_until = min(
                self._visual_pulse_until,
                started + self._visual_active_hold_s)
            self._visual_next_pulse_at = min(
                self._visual_next_pulse_at,
                self._visual_pulse_until + self.cfg.visual_pulse_rest_s)
        pulsing = (self._visual_pulse_until is not None
                   and now < self._visual_pulse_until)
        if not pulsing:
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   f"visual servo rest/measure hold="
                                   f"{self._visual_active_hold_s:.2f}s "
                                   f"visual_level={self._visual_feedback.level} "
                                   f"change={self._visual_feedback.last_change_px}; "
                                   + decision.motion.reason))
        speed = min(550, self.cfg.visual_pulse_mm_s)
        # 大角度纠偏由独立的原地转向脉冲完成；直线脉冲保持直行，
        # 避免左右轮一边破静摩擦、一边仍卡住造成不可预测的偏航。
        return Decision(Motion(speed, 0, self.cfg.ttl_near_ms,
                               f"visual servo pulse hold="
                               f"{self._visual_active_hold_s:.2f}s "
                               f"visual_level={self._visual_feedback.level} "
                               f"change={self._visual_feedback.last_change_px}; "
                               + decision.motion.reason))

    def _blind_travel_gate(self, decision: Decision,
                           perception: Perception | None,
                           now: float) -> Decision:
        """按实际感知周期限制相邻观测之间的开环位移。"""
        if decision.motion.linear_mm_s <= 0 or perception is None:
            self._reset_visual_pulse()
            return decision
        # 该车低速连续命令难以克服静摩擦，高速连续命令又容易冲出
        # 视野。统一采用“短促起步—MSG_STOP 急停—重新观测”的步进
        # 方式；远/近仅使用不同的基础脉冲时长。
        if self.cfg.pulse_drive_enabled:
            return self._visual_pulse_decision(
                decision, perception, now, near=False)
        if perception.frame_interval_s <= 0.0:
            self._reset_visual_pulse()
            return decision
        period = max(0.02, min(1.0, perception.frame_interval_s))
        cap = int(self.cfg.max_blind_travel_mm / period)
        if cap >= decision.motion.linear_mm_s:
            self._reset_visual_pulse()
            return decision
        if cap < self.cfg.min_linear_mm_s:
            return self._visual_pulse_decision(
                decision, perception, now, near=False)
        cap = max(self.cfg.min_linear_mm_s, cap)
        decision.motion = Motion(
            cap, decision.motion.angular_mrad_s, decision.motion.ttl_ms,
            decision.motion.reason + f" (perception cap {cap})")
        return decision

    def _soft_align_gate(self, decision: Decision) -> Decision:
        """横向偏差大时按比例降低前进速度（边前进边修正，不停车）。

        用户要求：目标靠视觉左边就略微向左前进。偏差超过 deadband
        后每增大 advance_speed_scale_px 速度再降一档，最低保留
        advance_min_ratio，避免大偏差时全速冲。
        """
        if (decision.finished or decision.motion.linear_mm_s <= 0
                or self._last_align_err is None):
            return decision
        err = self._last_align_err
        if err <= self.cfg.advance_align_deadband_px:
            return decision
        ratio = max(
            self.cfg.advance_min_ratio,
            1.0 - (err - self.cfg.advance_align_deadband_px)
            / max(self.cfg.advance_speed_scale_px, 1e-3))
        scaled = int(round(
            decision.motion.linear_mm_s * ratio / 10.0) * 10.0)
        if scaled >= decision.motion.linear_mm_s:
            return decision
        decision.motion = Motion(
            scaled, decision.motion.angular_mrad_s,
            decision.motion.ttl_ms,
            decision.motion.reason + f" (align {ratio:.0%})")
        return decision

    @staticmethod
    def _stall_visual_signature(
            perception: Perception | None
            ) -> tuple[float, float, float, float] | None:
        if (perception is None or not perception.tracking
                or perception.bbox is None or not perception.tracking2
                or perception.bbox2 is None):
            return None
        a, b = perception.bbox, perception.bbox2
        return ((a[1] + a[3]) / 2.0, a[3] - a[1],
                (b[1] + b[3]) / 2.0, b[3] - b[1])

    @staticmethod
    def _visual_approach_progress(
            before: tuple[float, float, float, float] | None,
            after: tuple[float, float, float, float] | None) -> bool:
        if before is None or after is None:
            return False
        # 两目都看到目标往画面下方移动或变大，表明正在逼近。
        # 这种情况下即使 ICP 对地面/障碍配准得出“没动”，也不能
        # 继续延长下一次脉冲；目标自己靠近时同样以短脉冲为安全侧。
        return all(
            after[i] - before[i] >= 3.0
            or after[i + 1] - before[i + 1] >= 3.0
            for i in (0, 2)
        )

    def _check_stalled(self, scan: LaserScan | None,
                       now: float,
                       perception: Perception | None = None) -> bool | None:
        """雷达判断车是否没动：只用可信 ICP 位移更新脉冲档位。

        返回 True=确认卡住，False=确认移动，None=未到期或测量不确定。
        旧实现把无扫描/ICP 失败都算作已移动，会反复降低已升高的
        起步脉冲，尤其在低矮目标和空旷房间里造成静摩擦卡死。
        """
        if scan is None:
            return None
        if self._stall_check_at is not None and now < self._stall_check_at:
            return None
        points = scan.points(min_distance_m=self.cfg.lidar_min_range_m,
                             max_distance_m=self.cfg.lidar_max_range_m)
        visual = self._stall_visual_signature(perception)
        if self._stall_ref_points is None:
            self._stall_ref_points = points
            self._stall_ref_visual = visual
            self._stall_check_at = now + self.cfg.stuck_window_s
            return None
        match = self._stall_icp.match(self._stall_ref_points, points)
        visual_progress = self._visual_approach_progress(
            self._stall_ref_visual, visual)
        self._stall_ref_points = points
        self._stall_ref_visual = visual
        self._stall_check_at = now + self.cfg.stuck_window_s
        if visual_progress:
            return False
        if not match.converged or not math.isfinite(match.rmse_m):
            return None
        distance = math.hypot(match.transform.x_m, match.transform.y_m)
        yaw = abs(match.transform.yaw_rad)
        # 只有当前进位移远大于扫描匹配噪声时才认定“已移动”。
        # 直行命令下的纯转角不能解除前进卡死等级。
        if distance >= self.cfg.stuck_progress_confirm_m:
            return False
        if distance < self.cfg.stuck_move_eps_m \
                and yaw < self.cfg.stuck_yaw_eps_rad:
            return True
        return None

    def _smooth_linear(self, decision: Decision, dt: float,
                       now: float) -> Decision:
        """静止起步：先给 linear_breakaway_mm_s 冲量破静摩擦（保持
        breakaway_hold_s），之后按 linear_accel_mm_s2 爬升到目标速度；
        减速/停车立即生效。"""
        if decision.finished or decision.motion.linear_mm_s == 0:
            self._smoothed_linear = 0.0
            self._kick_until = None
            return decision
        if decision.motion.linear_mm_s < 0:
            # 丢失回退是已经限定时长的负速脉冲，不能套用正向
            # breakaway（否则负速会被瞬间改成正向 220mm/s）。
            self._smoothed_linear = 0.0
            self._kick_until = None
            return decision
        desired = float(decision.motion.linear_mm_s)
        if self._smoothed_linear == 0.0:
            # 起步冲量（卡死时逐级提高，上限为固件 550）
            breakaway = min(
                550.0,
                float(self.cfg.linear_breakaway_mm_s)
                + self._stuck_level * self.cfg.stuck_breakaway_step_mm_s)
            self._smoothed_linear = breakaway
            self._kick_until = now + self.cfg.linear_breakaway_hold_s
        if self._kick_until is not None and now < self._kick_until:
            # 冲量保持期内维持起步速度，不管目标速度；
            # 冲量瞬间车突然动、画面猛跳，转向同时进行目标容易横向
            # 出画（2026-08-17 实测丢锁点在冲量帧附近）→ 冲量期内
            # 转向限幅减半，直行为主，目标留在画面中央区域。
            smoothed = self._smoothed_linear
            if decision.motion.angular_mrad_s:
                limited = int(round(
                    decision.motion.angular_mrad_s / 2.0))
                if limited != decision.motion.angular_mrad_s:
                    decision.motion = Motion(
                        smoothed, limited,
                        decision.motion.ttl_ms,
                        decision.motion.reason)
                    return decision
        else:
            self._kick_until = None
            if self._smoothed_linear > desired:
                # 冲量结束按 decel 限速回落到目标速度（不做 min_linear
                # 托底；也不瞬降——瞬降和瞬升一样会让画面猛跳丢目标）
                self._smoothed_linear = max(
                    desired,
                    self._smoothed_linear
                    - self.cfg.linear_decel_mm_s2 * max(dt, 0.0))
            else:
                self._smoothed_linear = min(
                    desired,
                    self._smoothed_linear
                    + self.cfg.linear_accel_mm_s2 * max(dt, 0.0))
            smoothed = self._smoothed_linear
        # 10mm/s 步进取整，减少 BLE 命令抖动
        smoothed = int(round(smoothed / 10.0) * 10.0)
        if smoothed != int(desired):
            decision.motion = Motion(smoothed, decision.motion.angular_mrad_s,
                                     decision.motion.ttl_ms,
                                     decision.motion.reason)
        return decision

    def _creep_decision(self, decision: Decision, lidar_front: float,
                        now: float,
                        perception: Perception | None = None) -> Decision:
        """近距蠕动逼近（逐帧小步长前后移动）：短促冲量（>静摩擦
        临界，带 stuck 升级加成）→ 完全停止（测量）→ 循环。
        - 末段（<0.35m）脉冲更短更小（~4cm 步长），越近越精细；
        - 过冲（雷达 < 0.18m，已越过停车线且停车判定未触发）→
          后退小步修正，再重新前冲；
        - 转向持续进行（差速车"左右"靠转向实现）。
        休息期雷达/视觉测量稳定，0.25m 停车判定可靠触发。"""
        overshoot = lidar_front < 0.18
        if now >= self._creep_next_pulse_at:
            if overshoot:
                # 反向同样需要越过静摩擦：高速度只保持 0.08s，约退4cm。
                self._creep_active_hold_s = 0.08
                self._creep_pulse_until = now + 0.08
                self._creep_next_pulse_at = now + 0.08 + self.cfg.creep_rest_s
                self._creep_backing = True
            else:
                self._creep_backing = False
                base_s = (self.cfg.creep_pulse_near_s
                          if lidar_front < 0.35
                          else self.cfg.creep_pulse_s)
                pulse_s = min(
                    self.cfg.creep_pulse_max_s,
                    base_s + self._stuck_level
                    * self.cfg.creep_pulse_step_s,
                )
                if self._close_to_taught_mask(perception):
                    pulse_s = min(pulse_s, self.cfg.taught_close_pulse_max_s)
                self._creep_active_hold_s = pulse_s
                self._creep_pulse_until = now + pulse_s
                self._creep_next_pulse_at = (now + pulse_s
                                             + self.cfg.creep_rest_s)
        if (not self._creep_backing
                and self._close_to_taught_mask(perception)
                and self._creep_pulse_until is not None
                and self._creep_active_hold_s
                > self.cfg.taught_close_pulse_max_s):
            started = self._creep_pulse_until - self._creep_active_hold_s
            self._creep_active_hold_s = self.cfg.taught_close_pulse_max_s
            self._creep_pulse_until = min(
                self._creep_pulse_until,
                started + self._creep_active_hold_s)
            self._creep_next_pulse_at = min(
                self._creep_next_pulse_at,
                self._creep_pulse_until + self.cfg.creep_rest_s)
        pulsing = (self._creep_pulse_until is not None
                   and now < self._creep_pulse_until)
        if pulsing and self._creep_backing:
            v = -min(550, self.cfg.creep_pulse_mm_s)
            phase = "back"
        elif pulsing:
            v = min(550, self.cfg.creep_pulse_mm_s)
            phase = "pulse"
        else:
            v = 0
            phase = "rest"
        # 休息期必须 (0,0)，run_visual_pursuit 才会发 MSG_STOP 立即清空
        # ESP32 内部速度斜坡；保留角速度会变成持续转向，画面无法稳定。
        angular = decision.motion.angular_mrad_s if pulsing else 0
        return Decision(Motion(v, angular,
                               self.cfg.ttl_near_ms,
                               f"creep {phase} hold="
                               f"{self._creep_active_hold_s:.2f}s "
                               f"level={self._stuck_level} "
                               f"lidar={lidar_front:.2f}m"))

    # ---------------- 视觉丢失回退 ----------------

    def _reset_recovery(self, clear_history: bool = False) -> None:
        self._recovery_started_at = None
        self._recovery_pulse_until = None
        self._recovery_next_pulse_at = 0.0
        self._recovery_turn_sign = 0
        self._recovery_dual_frames = 0
        self._recovery_last_confirm_frame_id = -1
        self._recovery_from_blind = False
        self._recovery_pulses = 0
        self._recovery_swept_used = False
        self._recovery_swept_turn = 0
        self._recovery_swept_estimated_m = 0.0
        self._recovery_swept_pulses = 0
        self._recovery_swept_last_frame_id = -1
        self._turn_rollback_active = False
        self._turn_rollback_edge_x = None
        if clear_history:
            self._turn_rollback_baseline = None
            self._recent_dual_centers.clear()
            self._recent_motion.clear()
            self._recent_visual_approach_at = 0.0
            self._last_dual_visual_signature = None
            self._last_near_blind_at = 0.0
            self._last_history_frame_id = -1

    def _record_dual_history(self, perception: Perception,
                             now: float) -> None:
        """记录双目确认帧的平均局部 x，不重复记录同一帧。"""
        if (not perception.tracking or perception.bbox is None
                or not perception.tracking2 or perception.bbox2 is None):
            return
        if (perception.frame_id > 0
                and perception.frame_id == self._last_history_frame_id):
            return
        centers = []
        for bbox in (perception.bbox, perception.bbox2):
            cx = (bbox[0] + bbox[2]) / 2.0
            if cx >= self.frame_half_width:
                cx -= self.frame_half_width
            if self.cfg.forward_reference_enabled:
                # Keep the existing history storage convention, but centre
                # it on the calibrated body axis, not the optical image centre.
                cx=self.frame_half_width/2+self._center_third_x_error(bbox)
            centers.append(cx)
        timestamp = (perception.received_at
                     if perception.received_at > 0.0 else now)
        signature = self._stall_visual_signature(perception)
        if self._visual_approach_progress(
                self._last_dual_visual_signature, signature):
            self._recent_visual_approach_at = timestamp
        self._last_dual_visual_signature = signature
        if any(
            (box[1] + box[3]) / 2.0
            >= self.cfg.visual_blind_guard_row_px - 80.0
            and box[3] - box[1]
            >= self.cfg.visual_blind_guard_box_height_px - 40.0
            for box in (perception.bbox, perception.bbox2)
        ):
            self._last_near_blind_at = timestamp
        self._recent_dual_centers.append(
            (timestamp, sum(centers) / len(centers)))
        keep = max(2, int(self.cfg.recovery_history_frames))
        del self._recent_dual_centers[:-keep]
        if perception.frame_id > 0:
            self._last_history_frame_id = perception.frame_id

    def _single_view_recovery_sign(
            self, bbox: tuple[float, float, float, float]) -> int:
        """单目优先按车身参考线偏差转向；在线附近不猜测方向。

        Disabled calibration retains the legacy eye-label fallback only.
        """
        error = self._center_third_x_error(bbox)
        if error < -self.cfg.angle_deadband_px:
            return 1
        if error > self.cfg.angle_deadband_px:
            return -1
        if self.cfg.forward_reference_enabled:
            # A visible physical-left eye does NOT prove target is left of
            # the chassis axis with toe-in cameras. Do not invent a turn.
            return 0
        center_x = (bbox[0] + bbox[2]) / 2.0
        physical_left = (center_x < self.frame_half_width) == self.cfg.physical_left_on_left
        return 1 if physical_left else -1

    def _history_recovery_sign(self) -> int:
        """双目都丢失时，由丢失前数帧位置和速度预测方向。"""
        history = self._recent_dual_centers
        if not history:
            return 0
        last_t, last_x = history[-1]
        velocity = 0.0
        if len(history) >= 2:
            first_t, first_x = history[0]
            span = last_t - first_t
            if span > 0.02:
                velocity = (last_x - first_x) / span
        predicted_x = (last_x + velocity
                       * self.cfg.recovery_prediction_horizon_s)
        offset = predicted_x - self.frame_half_width / 2.0
        if abs(offset) <= self.cfg.recovery_direction_deadband_px:
            return 0
        # angular 正=左转；目标趋势向右则右转（负）。
        return 1 if offset < 0.0 else -1

    def _motion_rollback_sign(self, now: float) -> int:
        """刚才有明显带角速度的前进命令时，优先反向复现路径。"""
        recent = [(v, w) for t, v, w in self._recent_motion
                  if 0.0 <= now - t <= 2.0 and v > 0 and abs(w) >= 100]
        if not recent:
            return 0
        net_turn = sum(w for _, w in recent)
        if abs(net_turn) < 100:
            return 0
        return -1 if net_turn > 0 else 1

    def _turn_rollback_evidence(self, now: float) -> bool:
        """Only a fresh, large binocular sweep following our forward arc qualifies.

        This is not an arbitrary lost-target reverse permit. Both eyes must
        have tracked the same abrupt lateral sweep, and the last issued arc
        must have the matching image-motion sign. No rear lidar is consulted.
        """
        baseline = self._turn_rollback_baseline
        if (baseline is None or not self._recent_dual_centers
                or not self._link_ready or self._simulate_motion):
            return False
        max_age = self.cfg.recovery_turn_rollback_max_age_s
        last_t, last_x = self._recent_dual_centers[-1]
        if not (0 <= now-baseline[0] <= max_age
                and baseline[0] < last_t <= now
                and now-last_t <= self.cfg.recovery_swept_path_max_age_s):
            return False
        scale = self.frame_half_width / 1280.0
        displacement = last_x-baseline[1]
        if (abs(displacement) < self.cfg.recovery_turn_rollback_jump_px*scale
                or abs(last_x-self.frame_half_width/2)
                < self.cfg.recovery_turn_rollback_edge_px*scale):
            return False
        # Use the earliest recent arc, which caused the sweep; an opposite
        # correction selected AFTER the box jumped must not erase that cause.
        arcs = [(t,w) for t,v,w in self._recent_motion
                if v>0 and abs(w)>=100 and baseline[0] <= t <= last_t]
        return bool(arcs and displacement*arcs[0][1] > 0)

    def _turn_rollback_recovered(self) -> bool:
        if (self._turn_rollback_edge_x is None or
                not self._recent_dual_centers):
            return False
        mid = self.frame_half_width/2
        now_x = self._recent_dual_centers[-1][1]
        scale = self.frame_half_width/1280.0
        return (abs(now_x-mid) <= self.cfg.recovery_turn_rollback_edge_px*scale
                and abs(now_x-mid) <= abs(self._turn_rollback_edge_x-mid)
                - 35.0*scale)

    def _lidar_rear(self, scan: LaserScan | None) -> float | None:
        if not self.cfg.recovery_rear_sensor_usable:
            return None  # chassis-occluded sector is not environmental evidence
        if scan is None:
            return None
        value = sector_min_distance(
            scan, center_deg=180.0,
            width_deg=2.0 * self.cfg.recovery_rear_half_width_deg,
            min_distance_m=min(0.02, self.cfg.lidar_hard_floor_m),
            max_distance_m=self.cfg.lidar_max_range_m,
        )
        return float(value) if math.isfinite(value) else None

    def _swept_path_retreat_allowed(self, now: float) -> bool:
        """后向固定近距回波时，仅撤销刚发生且双目证实的前进。"""
        if (not self.cfg.recovery_swept_path_enabled
                or not self._link_ready or self._simulate_motion):
            return False
        if self._turn_rollback_active:
            return True  # evidence latched at the fresh binocular sweep
        max_age = self.cfg.recovery_swept_path_max_age_s
        if not (0.0 <= now - self._recent_visual_approach_at <= max_age):
            return False
        if not (self._recovery_from_blind or
                0.0 <= now - self._last_near_blind_at <= max_age):
            return False
        return any(v > 0 and 0.0 <= now - t <= max_age
                   and t <= self._recent_visual_approach_at
                   for t, v, _ in self._recent_motion)

    def _swept_path_retreat(self, now: float, source: str,
                           turn_sign: int, frame_id: int) -> Decision:
        """小步回退；每步后重新看画面，累计理论位移不超过预算。"""
        remaining_m = (self.cfg.recovery_swept_path_max_m
                       - self._recovery_swept_estimated_m)
        limit_s = math.sqrt(
            max(0.0, 2.0 * remaining_m * 1000.0)
            / self.cfg.recovery_swept_path_accel_mm_s2)
        pulse_s = min(self.cfg.recovery_swept_path_pulse_s, limit_s)
        if pulse_s < 0.15:
            return self._stop_at_blind_guard("原路短回退配置无有效时长，安全停车")
        self._recovery_swept_used = True
        self._recovery_swept_pulses += 1
        self._recovery_swept_estimated_m += (
            0.5 * self.cfg.recovery_swept_path_accel_mm_s2
            * pulse_s * pulse_s / 1000.0)
        self._recovery_pulses += 1
        self._recovery_pulse_until = now + pulse_s
        self._recovery_next_pulse_at = now + pulse_s + self.cfg.recovery_rest_s
        self._recovery_swept_last_frame_id = frame_id
        # A confirmed wrong-way arc needs the inverse turn from the FIRST
        # small rollback; ordinary near-blind retreat still starts straight.
        turn = (turn_sign * self.cfg.recovery_swept_path_turn_mrad_s
                if self._turn_rollback_active or self._recovery_swept_pulses > 1
                else 0)
        self._recovery_swept_turn = turn
        self._recovery_turn_sign = 1 if turn > 0 else -1 if turn < 0 else 0
        return Decision(Motion(
            -self.cfg.recovery_reverse_mm_s, turn, self.cfg.ttl_near_ms,
            f"recovery swept-path pulse "
            f"{self._recovery_swept_pulses}/"
            f"{self.cfg.recovery_swept_path_max_pulses} "
            f"{pulse_s:.2f}s budget="
            f"{self._recovery_swept_estimated_m:.3f}m "
            f"{'turn-rollback' if self._turn_rollback_active else source}"))

    def _recovery_decision(self, perception: Perception | None,
                           scan: LaserScan | None,
                           now: float) -> Decision:
        """每帧先检查当前视觉，再决定单目引导或历史趋势回退。"""
        visible = []
        if (perception is not None and perception.tracking
                and perception.bbox is not None):
            visible.append(perception.bbox)
        if (perception is not None and perception.tracking2
                and perception.bbox2 is not None):
            visible.append(perception.bbox2)

        if self._recovery_started_at is None:
            self._recovery_started_at = now
            self._recovery_pulse_until = None
            self._recovery_next_pulse_at = now + self.cfg.recovery_settle_s
            self._recovery_dual_frames = 0
            self._recovery_last_confirm_frame_id = -1
            self._recovery_pulses = 0
            # 重获后不能再与丢失前旧框做跳变比较。
            self._last_bbox = None

        elapsed = now - self._recovery_started_at
        if len(visible) == 1:
            turn_sign = self._single_view_recovery_sign(visible[0])
            source = "single-left" if (
                (visible[0][0] + visible[0][2]) / 2.0
                < self.frame_half_width) else "single-right"
        elif len(visible) == 2:
            angular, _ = self._steer_for_dual_bearing(visible[0], visible[1])
            turn_sign = -1 if angular < 0 else 1 if angular > 0 else 0
            source = "dual-near-blind"
        else:
            # 找回不要求精确抵消底盘轨迹：先信丢失前的目标画面
            # 趋势，只有视觉历史无法判断时才参考刚才的转向命令。
            turn_sign = self._history_recovery_sign()
            source = "history-both-lost" if turn_sign else "motion-both-lost"
            if not turn_sign:
                turn_sign = self._motion_rollback_sign(now)
        if self._turn_rollback_active and not turn_sign:
            turn_sign = self._history_recovery_sign()
        self._recovery_turn_sign = turn_sign

        if elapsed >= self.cfg.recovery_max_s:
            if self._turn_rollback_active:
                return self._stop_at_blind_guard(
                    "转向回溯找回超时，已停车；不得继续盲转或前进")
            if self._recovery_from_blind:
                return self._stop_at_blind_guard(
                    "近端盲区双目找回超时，已停车；不得继续前进或盲转")
            self._enter_searching()
            return Decision(Motion(
                0, 0, self.cfg.ttl_near_ms,
                f"recovery exhausted {source}, searching"))
        if elapsed < self.cfg.recovery_settle_s:
            return Decision(Motion(
                0, 0, self.cfg.ttl_near_ms,
                f"recovery visual check {source}"))

        scan_fresh = (scan is not None and
                      (scan.timestamp_s <= 0.0 or
                       0.0 <= now - scan.timestamp_s
                       <= self.cfg.recovery_scan_max_age_s))
        rear = self._lidar_rear(scan) if scan_fresh else None
        if self._recovery_swept_used:
            # 这条路径不能靠后向雷达估距，因此只允许刚才已开始的
            # 限时小步完成；每一步必须在停车后收到新画面才能继续。
            if (not scan_fresh or
                    (rear is not None and
                     self.cfg.recovery_self_echo_max_m < rear
                     < self.cfg.recovery_rear_clear_m)):
                return self._stop_at_blind_guard(
                    "短回退期间雷达扫描失效或后方出现非固定近距障碍，已停车")
            if (self._recovery_pulse_until is not None and
                    now < self._recovery_pulse_until):
                return Decision(Motion(
                    -self.cfg.recovery_reverse_mm_s,
                    self._recovery_swept_turn,
                    self.cfg.ttl_near_ms,
                    f"recovery swept-path active {source}"))
            if now < self._recovery_next_pulse_at:
                return Decision(Motion(
                    0, 0, self.cfg.ttl_near_ms,
                    f"recovery swept-path stop/observe {source}"))
            fresh_view = bool(
                perception is not None and
                (perception.frame_id > self._recovery_swept_last_frame_id
                 if perception.frame_id > 0 else
                 perception.received_at > self._recovery_pulse_until))
            if not fresh_view:
                return Decision(Motion(
                    0, 0, self.cfg.ttl_near_ms,
                    "recovery swept-path waiting new frame"))
            if (self._recovery_swept_pulses
                    >= self.cfg.recovery_swept_path_max_pulses or
                    self._recovery_swept_estimated_m
                    >= self.cfg.recovery_swept_path_max_m):
                return self._stop_at_blind_guard(
                    "小步回退预算已用尽，双目仍未安全找回；停车等待重新观察")
            return self._swept_path_retreat(
                now, source, turn_sign, perception.frame_id)
        if rear is None or rear < self.cfg.recovery_rear_clear_m:
            # 0.09m 的底盘固定回波不能作为可用后向量程，但刚刚前进
            # 扫过的几厘米路径可撤回一次。若回波落在可疑的真实障碍
            # 区间 (0.13..0.32m)，或整个雷达扫描失效，仍不后退。
            self_echo = (rear is None or
                         rear <= self.cfg.recovery_self_echo_max_m)
            if (scan_fresh and self_echo
                    and self._swept_path_retreat_allowed(now)):
                return self._swept_path_retreat(
                    now, source, turn_sign,
                    perception.frame_id if perception is not None else 0)
            reason = ("rear disabled/body occlusion"
                      if not self.cfg.recovery_rear_sensor_usable else
                      "rear unknown/stale" if rear is None
                      else f"rear blocked {rear:.2f}m")
            if self._recovery_from_blind:
                return self._stop_at_blind_guard(
                    ("近端找回已停车（后向雷达不参与判断）："
                     if not self.cfg.recovery_rear_sensor_usable else
                     f"近端找回已停车：后向量程 {reason}，") +
                    "原路短回退证据不足或预算已用尽")
            return Decision(Motion(
                0, 0, self.cfg.ttl_near_ms,
                f"recovery {reason} {source}"))

        if now >= self._recovery_next_pulse_at:
            if self._recovery_pulses >= self.cfg.recovery_max_reverse_pulses:
                if self._recovery_from_blind:
                    return self._stop_at_blind_guard(
                        "近端盲区已尝试有限次短回退，双目仍未安全找回；停车等待人工检查")
                self._enter_searching()
                return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                       "recovery reverse budget exhausted"))
            self._recovery_pulses += 1
            self._recovery_pulse_until = (
                now + self.cfg.recovery_reverse_pulse_s)
            self._recovery_next_pulse_at = (
                now + self.cfg.recovery_reverse_pulse_s
                + self.cfg.recovery_rest_s)
        pulsing = (self._recovery_pulse_until is not None
                   and now < self._recovery_pulse_until)
        if not pulsing:
            return Decision(Motion(
                0, 0, self.cfg.ttl_near_ms,
                f"recovery rest/check {source} sign={turn_sign}"))
        angular = turn_sign * self.cfg.recovery_turn_mrad_s
        return Decision(Motion(
            -self.cfg.recovery_reverse_mm_s, angular,
            self.cfg.ttl_near_ms,
            f"recovery reverse {source} sign={turn_sign}"))

    # ---------------- TRACKING ----------------

    def _stationary_recheck(self, now: float, *, near: bool,
                            reason: str) -> Decision | None:
        if (not self._has_identity or self._stationary_recheck_count
                >= self.cfg.lost_stationary_recheck_limit):
            return None
        self._stationary_recheck_count += 1
        self._stationary_recheck_pending = True
        self._stationary_recheck_near = near
        self._stationary_recheck_deadline = now + self.cfg.lost_stationary_recheck_timeout_s
        self._reset_visual_pulse()
        self._reset_align_turn()
        return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
            f"stationary target recheck {reason} "
            f"{self._stationary_recheck_count}/{self.cfg.lost_stationary_recheck_limit}"),
            reground_now=True)

    def _visual_handoff(self, reason: str) -> Decision:
        """Stop while both views exist; arm/model validation stays downstream."""
        self._reset_visual_pulse()
        self._reset_align_turn()
        self.state = State.STOPPED
        return Decision(Motion(0, 0, self.cfg.ttl_near_ms, reason),
            finished=True, result="alignment_pending",
            detail="双目仍可见，底盘停止；交给抓取区域检查和限步微调，不继续向盲区前进")

    def _tick_tracking(self, perception: Perception | None,
                       scan: LaserScan | None,
                       lidar_front: float | None, dt: float,
                       now: float) -> Decision:
        if self._stationary_recheck_pending:
            if now >= self._stationary_recheck_deadline:
                return self._stop_at_blind_guard("停车重新识别超时，已保持停止；未授权抓取")
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "stationary target recheck pending"))

        if self._near_recheck_handoff and now >= self._stationary_recheck_deadline:
            return self._stop_at_blind_guard("重新识别后双目未连续恢复，已停车")

        if self._near_recheck_handoff:
            fresh = bool(perception is not None
                and perception.tracking and perception.bbox is not None
                and perception.tracking2 and perception.bbox2 is not None
                and perception.status != "busy"
                and perception.received_at > self._reground_confirm_after
                and now - perception.received_at <= self.cfg.perception_max_age_s)
            new_frame = bool(perception is not None and (
                perception.frame_id != self._reground_confirm_frame_id
                if perception.frame_id > 0 else
                perception.received_at != self._reground_confirm_rx_at))
            if fresh and new_frame:
                self._reground_confirm_frames += 1
                self._reground_confirm_frame_id = perception.frame_id
                self._reground_confirm_rx_at = perception.received_at
            elif not fresh:
                self._reground_confirm_frames = 0
            if self._reground_confirm_frames >= self.cfg.reground_dual_confirm_frames:
                self._near_recheck_handoff = False
                self._record_dual_history(perception, now)
                return self._visual_handoff("near target reidentified; stop for grasp alignment")
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                f"reground dual confirm {self._reground_confirm_frames}/{self.cfg.reground_dual_confirm_frames}"))

        if perception is None:
            self._lost_since += dt
            if self._lost_since >= self.cfg.lost_stationary_recheck_s:
                recheck = self._stationary_recheck(now,
                    near=self._last_near_blind_at > 0, reason="no perception")
                if recheck is not None:
                    return recheck
                if self._has_identity:
                    return self._stop_at_blind_guard("无新感知且停车复检次数已用尽，已停车")
            return Decision(Motion(0, 0, self.cfg.ttl_far_ms, "no perception"))

        # 服务端正在执行 Qwen 识别（网页"重新识别"期间帧结果全是 busy）：
        # 这是识别忙，不是目标丢失，不能累计丢失时间，否则会误进旋转重搜。
        if perception.status == "busy":
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "server grounding"))

        # 不能把旧 bbox 当成 50Hz 新观测连续驱动车：3090/网络卡顿时
        # 0.45s 后先停，持续无新结果再进入既有旋转重搜。
        if (perception.received_at > 0.0
                and now - perception.received_at
                > self.cfg.perception_max_age_s):
            self._lost_since += dt
            age = now - perception.received_at
            if (self.cfg.recovery_enabled and self.cfg.use_taught_stop
                    and self.cfg.taught_require_both_views):
                # Stale network frames cannot justify a reverse/turn. First
                # request a new stationary observation, including near range.
                self._reset_visual_pulse()
                self._reset_align_turn()
                if self._lost_since >= self.cfg.lost_stationary_recheck_s:
                    recheck = self._stationary_recheck(now,
                        near=self._last_near_blind_at > 0, reason="stale perception")
                    if recheck is not None:
                        return recheck
                    return self._stop_at_blind_guard("感知帧过期且停车复检已用尽，禁止依据旧框回退")
                return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                       f"perception stale {age:.2f}s, stop"))
            if self._lost_since > self.cfg.lost_before_search_s:
                self._enter_searching()
                return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                       f"perception stale {age:.2f}s, "
                                       "searching"))
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   f"perception stale {age:.2f}s, stop"))

        primary_visible = bool(
            perception.tracking and perception.bbox is not None)
        secondary_visible = bool(
            perception.tracking2 and perception.bbox2 is not None)
        visible_count = int(primary_visible) + int(secondary_visible)
        dual_required = bool(
            self.cfg.use_taught_stop
            and self.cfg.taught_require_both_views)

        # The 16:13 run swept both boxes from the right of the body axis to
        # the far left immediately after a right arc. Stopping for Qwen alone
        # left the car motionless until its 20s timeout. Detect that specific
        # fresh, binocular, self-induced overshoot before issuing an opposite
        # forward arc, then undo at most 10cm along the just-used path.
        if (self.cfg.recovery_enabled and dual_required
                and visible_count == 2):
            self._record_dual_history(perception, now)
        if (self.cfg.recovery_enabled and dual_required
                and not self._turn_rollback_active
                and self._turn_rollback_evidence(now)):
            self._turn_rollback_active = True
            self._turn_rollback_edge_x = self._recent_dual_centers[-1][1]
            self._reset_visual_pulse()
            self._reset_align_turn()
            return self._recovery_decision(perception, scan, now)
        if (self._turn_rollback_active and
                (visible_count < 2 or self._recovery_swept_pulses == 0
                 or not self._turn_rollback_recovered())):
            return self._recovery_decision(perception, scan, now)

        # 上一帧已到画面近端、下一帧两眼同时出画时，仍按近端越界
        # 处理，不能误入远距离旋转搜索或要求后向雷达重新量程。
        if (self.cfg.recovery_enabled and dual_required
                and visible_count < 2
                and self._last_near_blind_at > 0.0
                and 0.0 <= now - self._last_near_blind_at
                <= self.cfg.recovery_swept_path_max_age_s):
            self._recovery_from_blind = True

        # One simple loss path, before all near-blind/reverse branches.
        # The previous near-blind branch bypassed Qwen recheck and exited.
        if (self.cfg.recovery_enabled and dual_required and visible_count < 2
                and self._has_identity and self._stationary_recheck_count
                < self.cfg.lost_stationary_recheck_limit):
            self._lost_since += dt
            self._reset_visual_pulse()
            self._reset_align_turn()
            if self._lost_since >= self.cfg.lost_stationary_recheck_s:
                near = self._recovery_from_blind or bool(
                    visible_count == 1 and self._near_blind_guard(
                        perception.bbox if primary_visible else perception.bbox2))
                return self._stationary_recheck(now, near=near, reason="before recovery")
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "lost target stop/settle before recheck"))

        # 单目已在画面底部且框明显变大时，不再执行回退/盲转搜索。
        # 它更可能是底盘越过了可观测终点，而不是远处目标离开视野。
        if (self.cfg.use_taught_stop and visible_count == 1
                and self._near_blind_guard(
                    perception.bbox if primary_visible else perception.bbox2)):
            if self.cfg.recovery_enabled and dual_required:
                self._recovery_from_blind = True
                return self._recovery_decision(perception, scan, now)
            return self._stop_at_blind_guard()

        # 每帧的优先级：先看当前视觉。单目有框就用该路
        # 引导回退；只有两路都没有才使用丢失前多帧历史。
        if (self.cfg.recovery_enabled and dual_required
                and visible_count < 2):
            self._lost_since += dt
            return self._recovery_decision(perception, scan, now)

        # 回退期间重新出现双目后，先停车连续确认多帧，
        # 避免一帧短暂误检就立即改为向前。
        if (self.cfg.recovery_enabled and dual_required
                and visible_count == 2
                and self._recovery_started_at is not None):
            if self._recovery_from_blind and self._near_blind_guard(
                    perception.bbox, perception.bbox2):
                return self._recovery_decision(perception, scan, now)
            if (perception.frame_id <= 0 or perception.frame_id !=
                    self._recovery_last_confirm_frame_id):
                self._recovery_dual_frames += 1
                self._recovery_last_confirm_frame_id = perception.frame_id
            self._record_dual_history(perception, now)
            if (self._recovery_dual_frames
                    < self.cfg.recovery_dual_confirm_frames):
                return Decision(Motion(
                    0, 0, self.cfg.ttl_near_ms,
                    f"recovery dual confirm "
                    f"{self._recovery_dual_frames}/"
                    f"{self.cfg.recovery_dual_confirm_frames}"))
            from_blind = self._recovery_from_blind
            self._reset_recovery(clear_history=False)
            self._lost_since = 0.0
            self._last_bbox = None
            if from_blind:
                self.state = State.STOPPED
                return Decision(
                    Motion(0, 0, self.cfg.ttl_near_ms,
                           "blind recovery dual restored"),
                    finished=True, result="alignment_pending",
                    detail="短回退后双目目标已在安全视野中重新确认；停车交给左右二维示教区复核")

        if perception.tracking and perception.bbox is not None:
            if (self.cfg.use_taught_stop
                    and self.cfg.taught_require_both_views
                    and (not perception.tracking2
                         or perception.bbox2 is None)):
                self._lost_since += dt
                if self._lost_since > self.cfg.lost_before_search_s:
                    self._enter_searching()
                    return Decision(Motion(
                        0, 0, self.cfg.ttl_near_ms,
                        "stereo lock incomplete, searching"))
                return Decision(Motion(
                    0, 0, self.cfg.ttl_near_ms,
                    f"waiting second view {self._lost_since:.1f}s"))
            # bbox 跳变检测（2026-08-17 安全修复）：相邻帧框中心位移
            # 过大说明锁已漂（服务端主路门控漏掉时车会对着漂移框
            # 全速冲撞，实测 0.68m 处锁漂、2s 后冲到 0.5m）。跳变帧
            # 不更新参考、立即停车，等下一帧重新确认。
            if self._last_bbox is not None:
                prev_cx = (self._last_bbox[0] + self._last_bbox[2]) / 2.0
                prev_cy = (self._last_bbox[1] + self._last_bbox[3]) / 2.0
                new_cx = (perception.bbox[0] + perception.bbox[2]) / 2.0
                new_cy = (perception.bbox[1] + perception.bbox[3]) / 2.0
                if (abs(new_cx - prev_cx) > self.cfg.bbox_jump_px
                        or abs(new_cy - prev_cy) > self.cfg.bbox_jump_px):
                    # 更新参考框：服务端锁稳定到新位置后下一帧不再
                    # 判跳，自动恢复追踪；持续真跳变则累加丢失时间
                    # 超阈值后进旋转重搜（2026-08-17 修复永久卡停）。
                    self._last_bbox = perception.bbox
                    self._last_px_t = None
                    self._drift_ema = 0.0
                    self._drift_frames = 0
                    self._lost_since += dt
                    if self._lost_since > self.cfg.lost_before_search_s:
                        self._enter_searching()
                        return Decision(Motion(
                            0, 0, self.cfg.ttl_near_ms,
                            "bbox jump, searching"))
                    return Decision(Motion(
                        0, 0, self.cfg.ttl_near_ms,
                        f"bbox jump {new_cx - prev_cx:.0f}/"
                        f"{new_cy - prev_cy:.0f}px, stop"))
                # 平滑漂移防护：单帧阈值被每帧小幅漂移骗过时的兜底。
                # 车自身转向会让目标像面横向移动（~w×f px/s），先扣
                # 掉上帧角速度的期望再判；只对服务端未外观确认的追踪
                # 源（csrt/kf）生效，yolo/tpl 是检测器确认过的强观测。
                bbox_changed = perception.bbox != self._last_bbox
                if bbox_changed:
                    px_dt = (max(0.02, now - self._last_px_t)
                             if self._last_px_t is not None else 0.04)
                    self._last_px_t = now
                    expected = (self._last_cmd_angular / 1000.0
                                * self.cfg.ego_focal_px)
                    residual = (new_cx - prev_cx) / px_dt - expected
                    self._drift_ema = (
                        0.7 * self._drift_ema + 0.3 * residual)
                    if abs(self._drift_ema) > 220.0 \
                            and perception.source in ("csrt", "kf"):
                        self._drift_frames += 1
                    else:
                        self._drift_frames = 0
                    if self._drift_frames >= 4:
                        self._drift_frames = 0
                        self._lost_since += dt
                        return Decision(Motion(
                            0, 0, self.cfg.ttl_near_ms,
                            f"smooth drift {self._drift_ema:.0f}px/s, stop"))
                else:
                    self._drift_frames = 0
            if perception.tracking2 and perception.bbox2 is not None:
                self._record_dual_history(perception, now)
            self._lost_since = 0.0
            self._last_bbox = perception.bbox
            taught_mode = (self.cfg.use_taught_stop
                           and perception.bbox2 is not None)
            if taught_mode:
                # 掩码定义停车位置；行驶航向按两路车身正前方投影线，
                # 不把各自光学中心/内八字视差误当成车身中线。
                angular, bearing_error = self._steer_for_dual_bearing(
                    perception.bbox, perception.bbox2)
                self._last_align_err = abs(bearing_error)
            else:
                angular = self._steer_for_bbox(perception.bbox)
                self._last_align_err = abs(
                    self._center_third_x_error(perception.bbox))
            # 近距转向增益降低（2026-08-17）：近距大目标框中心抖动
            # 绝对值大，全增益让转向来回猛甩（实测近距 w ±250 振荡）
            if lidar_front is not None \
                    and lidar_front < self.cfg.near_ttl_distance_m:
                angular = int(angular * self.cfg.angle_gain_near_ratio)
            if taught_mode:
                # 示教停车（优先）：两路目标中心都对齐掩码图示教位置
                # 才停车。
                taught_decision = self._track_to_taught(
                    perception.bbox, perception.bbox2, angular)
                if taught_decision.finished:
                    return taught_decision
                if self._near_handoff_rows(perception) and self._has_identity:
                    return self._visual_handoff('visual row staging stop before final long pulse')
                if self._near_blind_guard(
                        perception.bbox, perception.bbox2):
                    if self._has_identity and self.cfg.lost_stationary_recheck_limit > 0:
                        self._record_dual_history(perception, now)
                        return self._visual_handoff("dual target near blind edge; stop for grasp alignment")
                    if self.cfg.recovery_enabled and dual_required:
                        self._recovery_from_blind = True
                        return self._recovery_decision(perception, scan, now)
                    return self._stop_at_blind_guard()
                return taught_decision
            if self.cfg.use_visual_row_distance:
                # 视觉行测距模式：低矮地面目标深度与雷达都测不到，
                # 由目标中心在画面中的行位置决定速度与停车。
                return self._track_by_row(perception.bbox, angular)
            distance = self._distance(perception, lidar_front)

            if distance is None:
                return Decision(Motion(0, angular, self.cfg.ttl_far_ms,
                                       "no distance, steering only"))
            # 近端（< near_ttl_distance_m）转向限幅：离目标近时大幅摆头
            # 会让目标快速滑出视野，且原地转对停准没有帮助。
            if distance < self.cfg.near_ttl_distance_m:
                angular = int(max(-self.cfg.max_angular_near_mrad_s,
                                  min(self.cfg.max_angular_near_mrad_s,
                                      angular)))
            if distance <= self.cfg.stop_distance_m + self.cfg.stop_confirm_margin_m:
                self._stop_frames += 1
                if self._stop_frames >= self.cfg.stop_confirm_frames:
                    self.state = State.STOPPED
                    return Decision(
                        Motion(0, 0, self.cfg.ttl_near_ms, "stop reached"),
                        finished=True, result="success",
                        detail=f"目标距离 {distance:.2f}m ≤ {self.cfg.stop_distance_m}m，停车",
                    )
                # 近端确认期只做小幅最终对准（同上限幅），避免原地高速转动。
                return Decision(Motion(0, angular, self.cfg.ttl_near_ms,
                                       "near target, confirming stop"))
            self._stop_frames = 0
            linear = self._linear_for(distance)
            ttl = (self.cfg.ttl_near_ms if distance < self.cfg.near_ttl_distance_m
                   else self.cfg.ttl_far_ms)
            return Decision(Motion(linear, angular, ttl,
                                   f"approach d={distance:.2f}m"))

        # 目标暂时丢失：累计时间，超过阈值进入旋转重搜
        self._lost_since += dt
        if self._lost_since > self.cfg.lost_before_search_s:
            self._enter_searching()
            # 本 tick 不发运动命令，下一 tick 由 _tick_searching 拿到
            # 真实雷达扫描；若这里传 None 会误触时长退化计时，之后
            # ICP 失败时退化 deadline 早已过期，会误判本步已转到位。
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "target lost, searching"))
        # 短暂丢失通常发生在车体脉冲让画面运动模糊时。此时按旧框
        # 继续转向会进一步把目标甩出画面；先急停、等全画面 re-ID，
        # 超时后再交给 SEARCHING 主动旋转。
        if self._last_bbox is not None:
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   f"target lost {self._lost_since:.1f}s, "
                                   "settle/reacquire"))
        return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                               f"target lost {self._lost_since:.1f}s"))

    def _track_by_row(self, bbox: tuple[float, float, float, float],
                      angular: int) -> Decision:
        """视觉行测距：目标中心行越低越近。

        目标中心进入 visual_near_row_px 开始线性减速，
        到达 visual_stop_row_px 连续 stop_confirm_frames 帧停车。
        行阈值与摄像头俯仰/高度绑定，实车按网页画面微调配置即可。
        """
        center_y = (bbox[1] + bbox[3]) / 2.0
        if center_y >= self.cfg.visual_stop_row_px:
            self._stop_frames += 1
            if self._stop_frames >= self.cfg.stop_confirm_frames:
                self.state = State.STOPPED
                return Decision(
                    Motion(0, 0, self.cfg.ttl_near_ms, "stop reached"),
                    finished=True, result="success",
                    detail=f"目标中心行 {center_y:.0f} ≥ 停车行 "
                           f"{self.cfg.visual_stop_row_px:.0f}，停车",
                )
            # 停车确认期只做小幅最终对准
            angular = int(max(-self.cfg.max_angular_near_mrad_s,
                              min(self.cfg.max_angular_near_mrad_s,
                                  angular)))
            return Decision(Motion(0, angular, self.cfg.ttl_near_ms,
                                   f"near stop row y={center_y:.0f}"))
        self._stop_frames = 0
        near = center_y >= self.cfg.visual_near_row_px
        if near:
            angular = int(max(-self.cfg.max_angular_near_mrad_s,
                              min(self.cfg.max_angular_near_mrad_s,
                                  angular)))
            # 越接近停车行越慢：进入下带时全速，接近停车行降至最低速度
            span = self.cfg.visual_stop_row_px - self.cfg.visual_near_row_px
            ratio = max(0.0, min(1.0, (center_y - self.cfg.visual_near_row_px)
                                 / max(span, 1e-3)))
            linear = self.cfg.max_linear_mm_s - int(round(
                (self.cfg.max_linear_mm_s - self.cfg.min_linear_mm_s)
                * ratio))
            return Decision(Motion(linear, angular, self.cfg.ttl_near_ms,
                                   f"row approach y={center_y:.0f}"))
        return Decision(Motion(self.cfg.max_linear_mm_s, angular,
                               self.cfg.ttl_far_ms,
                               f"row approach y={center_y:.0f}"))

    def _taught_x_error(self, bbox: tuple[float, float, float, float]) -> float:
        center_x = (bbox[0] + bbox[2]) / 2.0
        if center_x >= self.frame_half_width:
            aim_x = self.frame_half_width + self.cfg.taught_right_cx_px
        else:
            aim_x = self.cfg.taught_left_cx_px
        return center_x - aim_x

    def _steer_for_taught(self,
                          bbox: tuple[float, float, float, float]) -> int:
        """示教模式转向：把目标中心 x 推向该路的示教 x（不是光学主点）。

        接近示教位姿的过程中目标中心会沿固定轨迹移动，横向对齐该路
        示教 x 后再前进，纵向由 _track_to_taught 的偏差控制停车。
        """
        error_px = self._taught_x_error(bbox)
        if abs(error_px) <= self.cfg.angle_deadband_px:
            return 0
        angular = -self.cfg.angle_gain_mrad_per_px * error_px
        return int(max(-self.cfg.max_angular_mrad_s,
                       min(self.cfg.max_angular_mrad_s, angular)))

    def _steer_for_dual_bearing(
            self, bbox: tuple[float, float, float, float],
            bbox2: tuple[float, float, float, float]) -> tuple[int, float]:
        """两路各自相对车身中线投影的误差，取共同航向分量。

        Disabled calibration retains the old centre-third rule for replay.
        """
        error1 = self._center_third_x_error(bbox)
        error2 = self._center_third_x_error(bbox2)
        if self.cfg.forward_reference_enabled:
            error_px=(error1+error2)/2
        elif error1 * error2 < 0.0:
            return 0, 0.0
        else:
            nonzero = [value for value in (error1, error2) if value != 0.0]
            error_px = sum(nonzero) / len(nonzero) if nonzero else 0.0
        self._last_bearing_errors=(error1,error2,error_px)
        if abs(error_px) <= self.cfg.angle_deadband_px:
            return 0, error_px
        angular = -self.cfg.angle_gain_mrad_per_px * error_px
        angular = int(max(-self.cfg.max_angular_mrad_s,
                          min(self.cfg.max_angular_mrad_s, angular)))
        return angular, error_px

    def _center_third_x_error(
            self, bbox: tuple[float, float, float, float]) -> float:
        """Steering reference error; name retained for legacy callers/tests."""
        center_x = (bbox[0] + bbox[2]) / 2.0
        camera='right' if center_x>=self.frame_half_width else 'left'
        if center_x >= self.frame_half_width:
            center_x -= self.frame_half_width
        if self.cfg.forward_reference_enabled:
            return reference_error(self.cfg,camera,
                (center_x,(bbox[1]+bbox[3])/2),self.frame_half_width,self.frame_height)
        left_edge = self.frame_half_width / 3.0
        right_edge = self.frame_half_width * 2.0 / 3.0
        if center_x < left_edge:
            return center_x - left_edge
        if center_x > right_edge:
            return center_x - right_edge
        return 0.0

    def _taught_error(self, bbox: tuple[float, float, float, float]) -> float:
        """该路目标中心到对应示教点的欧氏像素距离。"""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if cx >= self.frame_half_width:
            tx = self.frame_half_width + self.cfg.taught_right_cx_px
            ty = self.cfg.taught_right_cy_px
        else:
            tx = self.cfg.taught_left_cx_px
            ty = self.cfg.taught_left_cy_px
        return math.hypot(cx - tx, cy - ty)

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: tuple) -> bool:
        """Ray-casting point-in-polygon，边界由 margin 距离另行覆盖。"""
        inside = False
        j = len(polygon) - 1
        for i, (xi, yi) in enumerate(polygon):
            xj, yj = polygon[j]
            crosses = ((yi > y) != (yj > y))
            if crosses:
                edge_x = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < edge_x:
                    inside = not inside
            j = i
        return inside

    @staticmethod
    def _point_segment_distance(x: float, y: float,
                                ax: float, ay: float,
                                bx: float, by: float) -> float:
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return math.hypot(x - ax, y - ay)
        t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / denom))
        return math.hypot(x - (ax + t * dx), y - (ay + t * dy))

    def _taught_mask_distance(
            self, bbox: tuple[float, float, float, float]) -> float:
        """目标中心到对应白色掩码区域的距离；区域内（含 margin）为 0。"""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        right = cx >= self.frame_half_width
        if right:
            cx -= self.frame_half_width
            polygon = self.cfg.taught_right_mask_polygon_px
        else:
            polygon = self.cfg.taught_left_mask_polygon_px
        if self._point_in_polygon(cx, cy, polygon):
            return 0.0
        distances = []
        for i, (ax, ay) in enumerate(polygon):
            bx, by = polygon[(i + 1) % len(polygon)]
            distances.append(self._point_segment_distance(
                cx, cy, ax, ay, bx, by))
        return max(0.0, min(distances) - self.cfg.taught_mask_margin_px)

    def _taught_closer_shortfall(
            self, bbox: tuple[float, float, float, float]) -> float:
        """目标中心距离“比对应掩码质心更近”停车行还差多少像素。"""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        centroid_y = (self.cfg.taught_right_cy_px
                      if cx >= self.frame_half_width
                      else self.cfg.taught_left_cy_px)
        stop_y = centroid_y + self.cfg.taught_closer_than_centroid_px
        return max(0.0, stop_y - cy)

    def _parking_buffer(self) -> tuple[float,str]:
        """Reserve roughly one measured ordinary step, never a long last kick."""
        base=self.cfg.taught_early_stop_margin_px
        if base<=0:
            return 0.0,'disabled'
        fallback=max(base,self.cfg.taught_early_stop_uncertain_margin_px)
        sample=self._parking_step_sample
        if sample is None or self._last_tick is None:
            return fallback,'no_recent_step'
        at,change=sample
        if (not all(math.isfinite(v) for v in (at,change,self._last_tick)) or
                not 0<=self._last_tick-at<=self.cfg.taught_stop_feedback_max_age_s):
            return fallback,'stale_or_invalid_step'
        if change<6.0 or change>=self.cfg.taught_stop_normal_step_max_px:
            return fallback,'uncertain_or_large_step'
        return max(base,change),'recent_normal_visual_step'

    def _track_to_taught(self, bbox: tuple[float, float, float, float],
                         bbox2: tuple[float, float, float, float],
                         angular: int) -> Decision:
        """示教停车：两路中心都进入各自白色掩码区域 → 停车。

        偏差取两路的较大者；进入 slow_band 后按偏差线性减速，
        给横向对准留时间，避免冲过示教位姿。

        2026-08-17 两处修复（示教停车从未达成的直接原因）：
        - 速度斜坡方向反了：旧公式 err 越小 linear 反而越大（加速
          冲向示教点），修正为 err→tol 时降到 min_linear；
        - 首帧即停：近距目标像素速度 ~1000px/s，30px 容差窗口
          不到 1 帧就被穿过，旧 3 帧连续确认永远凑不满——停车
          条件物理上不可达，只能首帧命中即停。
        """
        mask_err = max(self._taught_mask_distance(bbox),
                       self._taught_mask_distance(bbox2))
        closer_err = max(self._taught_closer_shortfall(bbox),
                         self._taught_closer_shortfall(bbox2))
        err = max(mask_err, closer_err)
        # 最短可靠电机脉冲也可能让近距目标一次跨越 200px。等到中心
        # 完全进入掩码、再越过质心才停车，会在下一观测前直接冲出
        # 画面。缓冲区仅是安全的交接停车点，不是机械臂可抓取证明。
        # 编排切换到640x480，复核两眼各自二维示教区或25px外沿。
        stop_margin,buffer_source=self._parking_buffer()
        if mask_err <= stop_margin:
            self.state = State.STOPPED
            return Decision(
                Motion(0, 0, self.cfg.ttl_near_ms, "predictive stop"),
                finished=True, result="alignment_pending",
                detail=("双目目标已进入示教掩码提前停车缓冲区 "
                        f"{mask_err:.0f}px ≤ "
                        f"{stop_margin:.0f}px（{buffer_source}）；"
                        "停车后复核左右二维示教区或25px外沿"),
            )
        if mask_err <= 0.0 and closer_err <= 0.0:
            self.state = State.STOPPED
            return Decision(
                Motion(0, 0, self.cfg.ttl_near_ms, "stop reached"),
                finished=True, result="alignment_pending",
                detail=("左右目标中心均已进入对应掩码，且比掩码"
                        f"质心更近 "
                        f"{self.cfg.taught_closer_than_centroid_px:.0f}px；"
                        "停车后复核左右二维示教区或25px外沿"),
            )
        self._stop_frames = 0
        band = self.cfg.taught_slow_band_px
        if err <= band:
            angular = int(max(-self.cfg.max_angular_near_mrad_s,
                              min(self.cfg.max_angular_near_mrad_s,
                                  angular)))
            # 越接近掩码边缘越慢：band 处全速，边缘处降至最低速度
            ratio = max(0.0, min(1.0, err / max(band, 1e-3)))
            linear = int(round(
                self.cfg.min_linear_mm_s
                + (self.cfg.max_linear_mm_s - self.cfg.min_linear_mm_s)
                * ratio))
            return Decision(Motion(linear, angular, self.cfg.ttl_near_ms,
                                   f"taught approach err={err:.0f}px"))
        return Decision(Motion(self.cfg.max_linear_mm_s, angular,
                               self.cfg.ttl_far_ms,
                               f"taught approach err={err:.0f}px"))

    def _bbox_x_error(self, bbox: tuple[float, float, float, float]) -> float:
        center_x = (bbox[0] + bbox[2]) / 2.0
        # bbox 是整幅拼接原始照片坐标：左目瞄准点 = 左相机主点
        # cx_raw_left（标定 573.4），右目 = half + 右相机主点（653.8）；
        # 再加安装 yaw 补偿。注意瞄准点是各相机光学主点，不是 640。
        view_offset = (0.0 if center_x < self.frame_half_width
                       else float(self.frame_half_width))
        aim_x = (self.stereo.cx_raw_left_px if view_offset == 0.0
                 else self.frame_half_width + self.stereo.cx_raw_right_px) \
            + self.stereo.yaw_compensation_px
        return center_x - aim_x

    def _steer_for_bbox(self, bbox: tuple[float, float, float, float]) -> int:
        error_px = self._center_third_x_error(bbox)
        self._last_bearing_errors=(error_px,None,error_px)
        if abs(error_px) <= self.cfg.angle_deadband_px:
            return 0
        # 符号约定（car/drivers/drive_base.py 差速解算：angular 正 = 左转
        # CCW）。目标在画面右侧（error_px>0）需要右转，因此取负。
        angular = -self.cfg.angle_gain_mrad_per_px * error_px
        return int(max(-self.cfg.max_angular_mrad_s,
                       min(self.cfg.max_angular_mrad_s, angular)))

    def _distance(self, perception: Perception,
                  lidar_front: float | None) -> float | None:
        candidates = []
        if self.cfg.use_stereo_depth:
            # 大 bbox 常混入远景，有效视差占比偏低；0.15 即可接受，
            # 深度值本身取 bbox 内中位数视差，抗野值能力强。
            if perception.depth_m is not None and perception.depth_conf >= 0.15:
                candidates.append(perception.depth_m)
        if lidar_front is not None:
            candidates.append(lidar_front)
        return min(candidates) if candidates else None

    def _linear_for(self, distance: float) -> int:
        cfg = self.cfg
        if distance >= cfg.slow_start_distance_m:
            return cfg.max_linear_mm_s
        span = cfg.slow_start_distance_m - cfg.stop_distance_m
        ratio = max(0.0, (distance - cfg.stop_distance_m) / max(span, 1e-3))
        linear = cfg.min_linear_mm_s + int(round(
            (cfg.max_linear_mm_s - cfg.min_linear_mm_s) * ratio))
        return int(max(0, min(cfg.max_linear_mm_s, linear)))

    # ---------------- SEARCHING ----------------

    def _enter_searching(self) -> None:
        self.state = State.SEARCHING
        self._step_idx = 0
        self._awaiting_ground = False
        self._rotating = False
        self._stop_frames = 0
        self._search_pulse_until = self._search_next_pulse_at = 0.0
        self._search_total_on_s = 0.0
        # 重搜以"最后看见目标的方位"为中心：丢失瞬间目标相对车头
        # 的方位 ≈ 最后 bbox 中心相对瞄准点的偏角。先把车头回瞄
        # 目标方位，再交替外扩（known-target re-acquisition）。
        err_px = 0.0
        if self._last_bbox is not None:
            err_px = self._bbox_x_error(self._last_bbox)
        bearing_deg = math.degrees(math.atan(err_px
                                             / max(self.cfg.ego_focal_px,
                                                   1.0)))
        # angular 正=左转；目标在右（bearing>0）需右转 → 取负
        self._search_base_deg = max(
            -self.cfg.search_first_aim_back_max_deg,
            min(self.cfg.search_first_aim_back_max_deg, -bearing_deg))
        self._search_hold_until = None
        self._reset_step()

    def _reset_step(self) -> None:
        self._ref_points = None
        self._yaw_since_start = 0.0
        self._stall_deadline = None
        self._yaw_at_stall_start = 0.0
        self._kick = False
        self._step_checked_obstacle = False
        self._rotation_fallback_deadline = None
        self._fallback_start = 0.0
        self._icp_converged_once = False
        self._icp_fail_windows = 0
        self._search_stall_windows = 0
        self._reacquire_frames = 0
        self._step_deadline: float | None = None

    def _current_step(self) -> float:
        steps = self.cfg.search_rotation_steps_deg
        return self._search_base_deg + float(
            steps[min(self._step_idx, len(steps) - 1)])

    def _tick_searching(self, perception: Perception | None,
                        scan: LaserScan | None, now: float) -> Decision:
        if self._awaiting_ground:
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "reground pending"), reground_now=False)

        # 已有身份的搜索在每个旋转端点停车约 0.35s，给 3090 十余帧
        # 清晰画面做全局双目匹配；到期后继续下一角度，不调用 Qwen。
        if self._search_hold_until is not None:
            if now < self._search_hold_until:
                return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                       "identity re-acquire hold"))
            self._search_hold_until = None

        # 运动链路（BLE）未就绪时车不会转：旋转步按"时长估算"假转完
        # 只会白费 reground 并快速走完 7 步 GIVE_UP（2026-08-17 实测
        # 多轮：ESP32 恢复广播前的等待期任务已结束）。等待链路就绪
        # 再旋转。
        if not self._link_ready:
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "waiting motion link"))

        # 步骤耗尽（含障碍跳过路径）直接放弃：_current_step 会把越界索引
        # 钳到最后一个方向，不检查的话车会朝已跳过的方向继续转。
        if self._step_idx >= len(self.cfg.search_rotation_steps_deg):
            self.state = State.GIVE_UP
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "search exhausted"),
                            reground_now=True, finished=True,
                            result="give_up",
                            detail="旋转重搜已转满仍未找到，最后朝向"
                                   "自动 Qwen 复检一次")

        # 视觉先行找回：旋转中服务端模板/YOLO 提案追回连续确认目标 →
        # 立即停车提交 Qwen 复检（复检通过即恢复追踪），不必转完本步，
        # 避免转过头把刚找回的目标又转出视野。
        if perception is not None and perception.tracking:
            self._reacquire_frames += 1
        else:
            self._reacquire_frames = 0
        if self._reacquire_frames >= self.cfg.search_reacquire_confirm_frames:
            self._reacquire_frames = 0
            self._awaiting_ground = True
            self._rotating = False
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "visual re-acquire"),
                            reground_now=True)

        # 每步最大时长（2026-08-17 实测 ICP yaw 振荡让 step 4 卡
        # 15s+）：超过时限直接跳过本步，不依赖 yaw 判定（防死等）。
        if self._step_deadline is None:
            self._step_deadline = now + self.cfg.search_step_timeout_s
        elif now > self._step_deadline:
            self._advance_step("step timeout")
            return self._tick_searching(perception, scan, now)

        step = self._current_step()

        # 每步只做一次目标扇区障碍检查：正前方 ±21° 有障碍时跳过这一步。
        if not self._step_checked_obstacle:
            self._step_checked_obstacle = True
            if scan is not None:
                clearance = sector_min_distance(
                    scan, center_deg=0.0,
                    width_deg=2.0 * self.cfg.lidar_front_half_width_deg,
                    min_distance_m=self.cfg.lidar_min_range_m,
                    max_distance_m=self.cfg.lidar_max_range_m,
                )
                if (clearance is not None
                        and clearance < self.cfg.search_step_obstacle_clear_m):
                    self._advance_step("obstacle, skip")
                    return self._tick_searching(perception, scan, now)

        yaw = self._measure_yaw(scan, now)
        # 旋转卡死保护：车转不动时 ICP 可能一直"正常收敛"到 ~0，
        # 必须按连续无进展窗口数跳过本步（否则无限原地转）。
        if self._search_stall_windows >= self.cfg.search_stall_giveup_windows:
            self._advance_step("rotation stuck")
            return self._tick_searching(perception, scan, now)
        target = math.radians(step)
        tolerance = math.radians(self.cfg.search_turn_tolerance_deg)
        if abs(yaw) >= abs(target) - tolerance:
            if self._has_identity:
                self._advance_step("turn done, identity scan")
                self._search_hold_until = now + 0.35
                return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                       "identity re-acquire hold"))
            # 尚未成功识别过目标时没有外观锚点，只能请求 Qwen。
            self._awaiting_ground = True
            self._rotating = False
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms, "turn done"),
                            reground_now=True)

        speed = (self.cfg.search_turn_start_kick_mrad_s if self._kick
                 else self.cfg.search_turn_speed_mrad_s)
        if self.cfg.pulse_drive_enabled and now >= self._search_next_pulse_at:
            if self._search_total_on_s + self.cfg.search_pulse_s > self.cfg.search_total_on_time_s + 1e-9:
                self.state = State.GIVE_UP
                return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                    "search pulse budget exhausted"), finished=True, result="give_up",
                    detail="受限转向搜索仍未找回目标，已停车，禁止连续盲转")
            self._search_total_on_s += self.cfg.search_pulse_s
            self._search_pulse_until = now + self.cfg.search_pulse_s
            self._search_next_pulse_at = (self._search_pulse_until
                                        + self.cfg.search_pulse_rest_s)
        if self.cfg.pulse_drive_enabled and now >= self._search_pulse_until:
            return Decision(Motion(0, 0, self.cfg.ttl_near_ms,
                                   "search pulse rest/observe"))
        direction = 1 if target >= 0 else -1
        return Decision(Motion(0, direction * speed,
                               self.cfg.search_turn_ttl_ms,
                               f"search turn step {self._step_idx + 1}/"
                               f"{len(self.cfg.search_rotation_steps_deg)} "
                               f"yaw={math.degrees(yaw):.1f}°/{step:.0f}°"))

    def _advance_step(self, reason: str) -> None:
        self._step_idx += 1
        self._reset_step()

    def _time_fallback_yaw(self, now: float) -> float:
        """按指令角速度 × 效率按时长估算转角（效率未知，仅保证任务不卡死）。

        无雷达、dry-run 模拟（车不动 ICP 恒零）、或 ICP 持续不收敛
        （空旷/对称环境特征不足，实车模式下车会一直原地转）时使用。
        """
        if self._rotation_fallback_deadline is None:
            effective_rad_s = (self.cfg.search_turn_speed_mrad_s / 1000.0
                               * self.cfg.search_time_fallback_efficiency)
            duration = (abs(math.radians(self._current_step()))
                        / max(effective_rad_s, 1e-3))
            self._fallback_start = now
            self._rotation_fallback_deadline = now + duration
        if now < self._rotation_fallback_deadline:
            return self._yaw_since_start
        return math.radians(self._current_step())

    def _measure_yaw(self, scan: LaserScan | None, now: float) -> float:
        """用起始圈与当前圈的 ICP 匹配估计已转过角度。"""
        if scan is None or self._simulate_motion:
            return self._time_fallback_yaw(now)
        current = scan.points(min_distance_m=self.cfg.lidar_min_range_m,
                              max_distance_m=self.cfg.lidar_max_range_m)
        if self._ref_points is None:
            self._ref_points = current
            return 0.0
        initial = Pose2D(0.0, 0.0, self._yaw_since_start)
        match = self._icp.match(self._ref_points, current, initial=initial)
        if match.converged and math.isfinite(match.transform.yaw_rad):
            # 旋转中 ICP yaw 估计会来回跳（-23°→2°→-9°，2026-08-17
            # 实车实测）：直接用原始值会让卡死检测被"伪进展"骗过——
            # 每个窗口 yaw 都不同（|Δ|>1.15° 判有进展）但净位移为零，
            # 搜索步永远判不到位（step 4 卡 15s+）。EMA 滤掉振荡，
            # 只保留净进展。
            raw_yaw = float(match.transform.yaw_rad)
            self._yaw_since_start = (
                0.7 * self._yaw_since_start + 0.3 * raw_yaw)
            self._icp_converged_once = True
        # 卡死检测：一个窗口内航向无进展则启用起步增益；有进展则释放。
        if self._stall_deadline is None:
            self._stall_deadline = now + self.cfg.search_stall_s
            self._yaw_at_stall_start = self._yaw_since_start
        elif now > self._stall_deadline:
            progressed = abs(self._yaw_since_start - self._yaw_at_stall_start) >= 0.02
            self._kick = not progressed
            if not progressed:
                self._search_stall_windows += 1
            else:
                self._search_stall_windows = 0
            if not self._icp_converged_once:
                self._icp_fail_windows += 1
            self._stall_deadline = now + self.cfg.search_stall_s
            self._yaw_at_stall_start = self._yaw_since_start
        # ICP 从未收敛过且连续多个卡死窗口：起步增益救不了特征不足，
        # 改按时长估算，否则实车会无限原地旋转。
        if (not self._icp_converged_once
                and self._icp_fail_windows
                >= self.cfg.search_icp_fallback_windows):
            return self._time_fallback_yaw(now)
        return self._yaw_since_start

    # ---------------- 工具 ----------------

    def _lidar_front(self, scan: LaserScan | None) -> float | None:
        if scan is None:
            return None
        # 下限必须低于 lidar_hard_floor_m，否则"过近"的读数在进入
        # sector_min_distance 前就被 min_distance 过滤掉，硬停永远不触发。
        value = sector_min_distance(
            scan, center_deg=0.0,
            width_deg=2.0 * self.cfg.lidar_front_half_width_deg,
            min_distance_m=min(0.02, self.cfg.lidar_hard_floor_m),
            max_distance_m=self.cfg.lidar_max_range_m,
        )
        # sector_min_distance 扇区无有效点时返回 inf；必须视作"无数据"，
        # 否则 _distance 会把 inf 当有效距离，在深度与雷达都缺失时全速前进。
        return float(value) if math.isfinite(value) else None
