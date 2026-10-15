"""抓取/释放序列状态机。

完整流程:
  IDLE → QWEN_INIT → ALIGN → APPROACH → FINE_ALIGN → GRASP → LIFT → RETRACT → RELEASE → DONE

每一步由 VisualServoController 生成的 ServoCommand 驱动。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class Phase(Enum):
    IDLE = "idle"
    DEPLOY = "deploy"                     # 机械臂下降伸入摄像头视野
    CAMERA_SELECT = "camera_select"       # Qwen 比较左右相机后锁定视角
    QWEN_INIT = "qwen_init"               # 等待 Qwen 识别爪子 + 目标
    CLAW_INIT = "claw_init"               # Qwen 只识别了爪子，等待目标
    ALIGN = "align"                       # 水平 + 垂直对齐
    APPROACH = "approach"                  # 向前逼近
    FINE_ALIGN = "fine_align"             # 近距离精细对齐
    GRASP = "grasp"                       # 闭合爪子
    HOLD = "hold"                         # 保持抓取，等待稳定
    LIFT = "lift"                         # 抬起物体
    VERIFY = "verify"                     # 验证目标随夹爪共同移动
    RETRACT = "retract"                   # 后撤 arm
    CARRY = "carry"                       # 携带物体（可移动底盘）
    RELEASE = "release"                   # 张开爪子释放
    DONE = "done"
    ERROR = "error"


@dataclass
class SequenceStatus:
    """序列状态快照（用于 Web 展示）。"""
    phase: str = "idle"
    step: int = 0
    step_label: str = ""
    started_at: float = 0.0
    elapsed_s: float = 0.0
    claw_state: str = "unknown"          # open | closing | closed | opening
    target_name: str = ""
    last_action: str = ""
    last_reason: str = ""
    error: str = ""


class GraspSequence:
    """抓取任务序列状态机。

    视觉伺服控制循环:
      每帧: camera → tracker.update → servo.compute → arm_cmd
      每步: 执行 arm_cmd → 等待就位 → 下一帧 → 循环
    """

    def __init__(self, target_description: str = "") -> None:
        self.target_description = target_description
        self.phase = Phase.IDLE
        self._started_at: float = 0.0
        self._phase_started_at: float = 0.0
        self._last_action: str = ""
        self._last_reason: str = ""
        self._error: str = ""
        self._claw_state: str = "open"    # open | closing | closed | opening
        self._lift_completed: bool = False
        self._release_completed: bool = False

        # 各阶段尝试计数
        self._phase_attempts: int = 0
        self._max_phase_attempts: dict[str, int] = {
            "align": 60,
            "approach": 40,
            # 2 mm staged descent needs more closed-loop observations.
            "fine_align": 48,
        }

    # ── 状态查询 ──────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.phase not in (Phase.IDLE, Phase.DONE, Phase.ERROR)

    @property
    def is_finished(self) -> bool:
        return self.phase == Phase.DONE

    @property
    def is_error(self) -> bool:
        return self.phase == Phase.ERROR

    @property
    def started_at(self) -> float:
        return self._started_at

    # ── 状态推进 ──────────────────────────────────────────

    def start(self) -> None:
        """启动抓取序列: 先部署机械臂到视野。"""
        self.phase = Phase.DEPLOY
        self._started_at = time.time()
        self._phase_started_at = time.time()
        self._phase_attempts = 0
        self._error = ""

    def on_deploy_done(self) -> None:
        """机械臂完成固定部署，开始比较左右相机。"""
        self.phase = Phase.CAMERA_SELECT
        self._phase_started_at = time.time()
        self._phase_attempts = 0

    def on_qwen_done(self, claw_found: bool, target_found: bool,
                     target_name: str = "") -> None:
        """Qwen 识别完成回调。"""
        if claw_found and target_found:
            self.phase = Phase.ALIGN
            self.target_description = target_name or self.target_description
        elif claw_found and not target_found:
            self.phase = Phase.CLAW_INIT
        else:
            self._error = "Qwen 未找到爪子"
            self.phase = Phase.ERROR
        self._phase_started_at = time.time()
        self._phase_attempts = 0

    def on_target_found(self, target_name: str = "") -> None:
        """目标物被找到（在 CLAW_INIT 阶段）。"""
        self.target_description = target_name or self.target_description
        self.phase = Phase.ALIGN
        self._phase_started_at = time.time()
        self._phase_attempts = 0

    def update_from_servo(self, servo_cmd) -> str:
        """根据伺服命令更新序列状态。

        返回: 下一步动作字符串
          "continue"  — 继续当前阶段
          "advance"   — 阶段完成，进入下一阶段
          "error"     — 出错
        """
        action = servo_cmd.action
        state = servo_cmd.state
        self._last_action = action
        self._last_reason = servo_cmd.reason

        if action == "error":
            self._error = servo_cmd.reason
            self.phase = Phase.ERROR
            return "error"

        if action == "reground":
            # 跟踪丢失需要重识别
            self.phase = Phase.CAMERA_SELECT
            self._phase_started_at = time.time()
            return "continue"

        if action == "move":
            desired_phase = {
                "align_h": Phase.ALIGN,
                "approach": Phase.APPROACH,
                "fine_align": Phase.FINE_ALIGN,
            }.get(state)
            if desired_phase is not None and desired_phase != self.phase:
                self.phase = desired_phase
                self._phase_started_at = time.time()
                self._phase_attempts = 0

        if action == "grasp" and self.phase in (Phase.ALIGN, Phase.APPROACH, Phase.FINE_ALIGN):
            # 对齐完成 → 进入抓取
            self.phase = Phase.GRASP
            self._claw_state = "closing"
            self._phase_started_at = time.time()
            self._phase_attempts = 0
            return "advance"

        # 检查阶段内尝试次数
        phase_key = {
            Phase.ALIGN: "align",
            Phase.APPROACH: "approach",
            Phase.FINE_ALIGN: "fine_align",
        }.get(self.phase, "")

        if phase_key and action == "move":
            self._phase_attempts += 1
            max_attempts = self._max_phase_attempts.get(phase_key, 10)
            if self._phase_attempts >= max_attempts:
                self._error = (
                    "{} 阶段超出最大尝试次数 ({})".format(
                        self.phase.value, max_attempts,
                    )
                )
                self.phase = Phase.ERROR
                return "error"

        return "continue"

    def grasp_completed(self) -> None:
        """爪子闭合完成。"""
        self._claw_state = "closed"
        self.phase = Phase.HOLD
        self._phase_started_at = time.time()

    def hold_ready(self, hold_s: float = 1.0) -> bool:
        """保持抓取足够时间后可抬起。"""
        return (self.phase == Phase.HOLD
                and time.time() - self._phase_started_at > hold_s)

    def advance_to_lift(self) -> None:
        self.phase = Phase.LIFT
        self._phase_started_at = time.time()

    def advance_to_retract(self) -> None:
        self.phase = Phase.RETRACT
        self._phase_started_at = time.time()

    def advance_to_verify(self) -> None:
        self.phase = Phase.VERIFY
        self._phase_started_at = time.time()

    def retry_after_failed_grasp(self, reason: str = "") -> None:
        """Return a failed verified grasp to visual fine alignment."""
        self.phase = Phase.FINE_ALIGN
        self._phase_started_at = time.time()
        self._phase_attempts = 0
        self._claw_state = "open"
        self._last_action = "release"
        self._last_reason = reason or "grasp verification failed; retrying"
        self._error = ""

    def advance_to_carry(self) -> None:
        self.phase = Phase.CARRY
        self._phase_started_at = time.time()

    def advance_to_release(self) -> None:
        self.phase = Phase.RELEASE
        self._claw_state = "opening"
        self._phase_started_at = time.time()

    def release_completed(self) -> None:
        self._claw_state = "open"
        self._release_completed = True
        self.phase = Phase.DONE

    def abort(self, reason: str = "") -> None:
        """紧急中止。"""
        self._error = reason or "用户中止"
        self.phase = Phase.ERROR

    def reset(self) -> None:
        self.phase = Phase.IDLE
        self._started_at = 0.0
        self._phase_started_at = 0.0
        self._last_action = ""
        self._last_reason = ""
        self._error = ""
        self._claw_state = "open"
        self._phase_attempts = 0

    # ── 状态快照 ──────────────────────────────────────────

    def status(self) -> SequenceStatus:
        phase_labels = {
            Phase.IDLE: "空闲",
            Phase.DEPLOY: "部署机械臂到视野",
            Phase.CAMERA_SELECT: "Qwen 选择左右相机",
            Phase.QWEN_INIT: "Qwen 识别中",
            Phase.CLAW_INIT: "等待目标物识别",
            Phase.ALIGN: "对齐中",
            Phase.APPROACH: "逼近中",
            Phase.FINE_ALIGN: "精细对齐",
            Phase.GRASP: "闭合爪子",
            Phase.HOLD: "保持抓取",
            Phase.LIFT: "抬起物体",
            Phase.VERIFY: "验证抓取",
            Phase.RETRACT: "后撤",
            Phase.CARRY: "携带移动",
            Phase.RELEASE: "释放物体",
            Phase.DONE: "完成",
            Phase.ERROR: "错误",
        }
        elapsed = (time.time() - self._started_at) if self._started_at > 0 else 0.0
        return SequenceStatus(
            phase=self.phase.value,
            step=len([p for p in Phase if p.value <= self.phase.value]),
            step_label=phase_labels.get(self.phase, self.phase.value),
            started_at=self._started_at,
            elapsed_s=round(elapsed, 1),
            claw_state=self._claw_state,
            target_name=self.target_description,
            last_action=self._last_action,
            last_reason=self._last_reason,
            error=self._error,
        )
