"""双摄像头协调夹取对齐模块 v2（真机版）。

核心思想（用户提出的交替对齐 + 回溯，加上响应矩阵在线学习）：

1. 两个摄像头都看到夹爪与目标 → 每帧测量两相机的像素误差
2. 控制器输出一个控制轴的一步动作（yaw / up / forward）
3. 动作后测量两相机夹爪像素位移 → 在线学习 3控制轴×4图像轴 响应矩阵
4. 总误差发散 → 回溯上一步，并把该 (舵机,方向) 暂时封禁，换其他舵机
5. 超调 → 该轴自适应步长减半（不每次都用同一套参数），成功则缓慢恢复
6. 学习矩阵 + 各轴缩放持久化到 JSON，下次启动直接加载

机械臂模型（固件已抽象为笛卡尔控制轴）：
  yaw_deg    (joint0, 10us/deg)
  up_mm      (joints 1,2,3 组合)
  forward_mm (joints 1,2,3 组合)

响应矩阵 A (4x3)：
  [d_camL_x]   [a00 a01 a02] [Δyaw_deg]
  [d_camL_y] = [a10 a11 a12] [Δup_mm]
  [d_camR_x]   [a20 a21 a22] [Δforward_mm]
  [d_camR_y]   [a30 a31 a32]

误差向量 e = 目标中心 - 夹爪中心（已扣除悬停偏移），目标：min ||A·Δs - e||。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import time
from typing import Any

# 学习数据保存路径
LEARNED_RESPONSE_PATH = os.environ.get(
    "GRASP_LEARNED_PATH",
    "/home/gpu/grab_visual/learned_response.json",
)

AXIS_NAMES = ["camL_x", "camL_y", "camR_x", "camR_y"]
CONTROL_NAMES = ["yaw_deg", "up_mm", "forward_mm"]
N_CONTROL = 3
N_AXES = 4

# 每单位控制动作引起的像素位移合理范围（px/unit），超过视为噪声
MAX_PER_UNIT = 80.0
MIN_PER_UNIT = 0.03

# 各控制轴的合理响应上限（px/unit）：yaw 可较大，上下/前后物理上不会
# 造成巨大画面位移。超过该上限的样本通常是夹爪检测跳变，予以丢弃。
MAX_PER_UNIT_BY_CONTROL = {0: 40.0, 1: 15.0, 2: 15.0}


@dataclass
class AlignmentState:
    """单相机视角的对齐状态。dx/dy 已扣除悬停偏移：0 即期望位置。"""
    dx_px: float = 0.0        # 目标x - 夹爪x
    dy_px: float = 0.0        # (目标y - 夹爪y) - 悬停偏移
    aligned_x: bool = False
    aligned_y: bool = False
    x_tolerance_px: float = 40.0
    y_tolerance_px: float = 40.0
    visible: bool = False     # 该相机当前能否同时看到夹爪与目标
    claw_center: tuple[float, float] = (0.0, 0.0)
    target_center: tuple[float, float] = (0.0, 0.0)

    @property
    def error_magnitude(self) -> float:
        if not self.visible:
            return 1e9
        return math.hypot(self.dx_px, self.dy_px)

    @property
    def aligned(self) -> bool:
        return self.visible and self.aligned_x and self.aligned_y


class ResponseMatrixLearner:
    """在线学习 3控制轴→4图像轴 的响应矩阵。

    每次机械臂动作后，测量两相机中夹爪的实际像素位移，
    用指数移动平均更新矩阵（异常值剔除 + 噪声下限）。
    """

    def __init__(self, matrix: list[list[float]] | None = None,
                 scales: list[float] | None = None) -> None:
        # A[axis][control] = 每单位控制动作引起的夹爪像素位移
        self._matrix = self._normalize_matrix(matrix or [[0.0] * N_CONTROL for _ in range(N_AXES)])
        self._sample_count: list[list[int]] = [[0] * N_CONTROL for _ in range(N_AXES)]
        self._scales: list[float] = scales or [1.0] * N_CONTROL
        self._alpha = 0.25      # EMA 系数
        self._total_samples = 0

    @staticmethod
    def _normalize_matrix(matrix: list[list[float]]) -> list[list[float]]:
        """兼容旧版 4x4 数据：截取前 3 列。"""
        rows = []
        for row in matrix:
            values = [float(v) for v in row[:N_CONTROL]]
            while len(values) < N_CONTROL:
                values.append(0.0)
            rows.append(values)
        while len(rows) < N_AXES:
            rows.append([0.0] * N_CONTROL)
        return rows[:N_AXES]

    # ---- 学习 ----

    def observe(self, control_index: int, control_delta: float,
                responses: dict[str, float | None]) -> None:
        """记录一次动作的实际夹爪像素位移。

        Args:
            control_index: 0=yaw, 1=up, 2=forward
            control_delta: 实际执行的增量（度或mm，可为负）
            responses: {"camL_x": px, ...} 夹爪中心位移，None 表示该轴不可测
        """
        if abs(control_delta) < 1e-6:
            return
        learned_any = False
        control_limit = MAX_PER_UNIT_BY_CONTROL.get(control_index, MAX_PER_UNIT)
        for axis_index, axis in enumerate(AXIS_NAMES):
            px_delta = responses.get(axis)
            if px_delta is None:
                continue
            px_delta = float(px_delta)
            # 低于测量噪声下限的位移不可信（YOLO 框抖动 ~2-3px）
            if abs(px_delta) < 2.0:
                continue
            per_unit = px_delta / control_delta
            if abs(per_unit) > MAX_PER_UNIT:
                continue  # 异常值
            if abs(per_unit) > control_limit:
                continue  # 超出该控制轴的物理合理范围（检测跳变）
            if abs(per_unit) < MIN_PER_UNIT:
                continue  # 无响应
            old = self._matrix[axis_index][control_index]
            n = self._sample_count[axis_index][control_index]
            if n == 0:
                self._matrix[axis_index][control_index] = per_unit
            else:
                # 与旧值偏差过大时降低本次权重（防坏样本污染）
                deviation = abs(per_unit - old) / max(1.0, abs(old))
                alpha = self._alpha * (0.3 if deviation > 3.0 else 1.0)
                self._matrix[axis_index][control_index] = (
                    old * (1.0 - alpha) + per_unit * alpha
                )
            self._sample_count[axis_index][control_index] = n + 1
            learned_any = True
        if learned_any:
            self._total_samples += 1

    # ---- 查询 ----

    def response_for_control(self, control_index: int) -> list[float]:
        """返回某控制轴在4个图像轴上的响应向量。"""
        return [self._matrix[a][control_index] for a in range(N_AXES)]

    def sample_count_for_control(self, control_index: int) -> int:
        return max(self._sample_count[a][control_index] for a in range(N_AXES))

    def is_learned(self, minimum_samples: int = 1) -> bool:
        """每个控制轴是否都有足够的非零响应样本。"""
        for control in range(N_CONTROL):
            non_zero = sum(
                1 for a in range(N_AXES)
                if self._sample_count[a][control] > 0
                and abs(self._matrix[a][control]) > 1e-6
            )
            if non_zero < minimum_samples:
                return False
        return True

    def best_control_for_error(
        self,
        error_vector: list[float],
        weights: list[float],
        *,
        exclude_controls: set[int] | None = None,
    ) -> int:
        """选择最能减小加权总误差的控制轴。"""
        exclude = exclude_controls or set()
        current_total = self._weighted_norm(error_vector, weights)
        best_control, best_reduction = -1, 0.0
        for control in range(N_CONTROL):
            if control in exclude:
                continue
            response = self.response_for_control(control)
            response_mag = math.sqrt(sum(w * r * r for w, r in zip(weights, response)))
            if response_mag < 0.02:
                continue
            best_for_this = 0.0
            for unit in (1.0, -1.0):
                predicted = [
                    error_vector[a] - response[a] * unit for a in range(N_AXES)
                ]
                reduction = current_total - self._weighted_norm(predicted, weights)
                best_for_this = max(best_for_this, reduction)
            if best_for_this > best_reduction:
                best_reduction, best_control = best_for_this, control
        return best_control

    @staticmethod
    def _weighted_norm(vector: list[float], weights: list[float]) -> float:
        return math.sqrt(sum(w * v * v for w, v in zip(weights, vector)))

    # ---- 自适应轴缩放 ----

    def scale(self, control_index: int) -> float:
        return self._scales[control_index]

    def shrink_scale(self, control_index: int, factor: float = 0.5) -> None:
        self._scales[control_index] = max(0.25, self._scales[control_index] * factor)

    def recover_scale(self, control_index: int, factor: float = 1.1) -> None:
        self._scales[control_index] = min(1.0, self._scales[control_index] * factor)

    # ---- 持久化 ----

    def save(self, path: str = LEARNED_RESPONSE_PATH) -> None:
        try:
            data = {
                "matrix": self._matrix,
                "sample_counts": self._sample_count,
                "scales": self._scales,
                "total_samples": self._total_samples,
                "saved_at": time.time(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    @classmethod
    def load(cls, path: str = LEARNED_RESPONSE_PATH) -> "ResponseMatrixLearner":
        try:
            with open(path, "r") as f:
                data = json.load(f)
            learner = cls(
                matrix=data.get("matrix"),
                scales=data.get("scales"),
            )
            learner._sample_count = data.get("sample_counts", learner._sample_count)
            learner._total_samples = data.get("total_samples", 0)
            return learner
        except Exception:
            return cls()


@dataclass
class MoveStep:
    """一次机械臂动作的记录。"""
    control_index: int
    delta: float              # 实际执行增量（已量化为整数步长）
    before_error: float       # 动作前加权总误差
    before_err_vec: list[float]
    after_error: float | None = None
    after_err_vec: list[float] | None = None
    responses: dict[str, float | None] = field(default_factory=dict)
    backtrack: bool = False


class DualCameraAlignmentController:
    """双摄像头交替对齐控制器。

    流程：
      1. 读入两相机对齐状态（含可见性）
      2. 更新上一次动作的结果 → 学习矩阵 / 检测发散与超调
      3. 两相机都对齐且稳定 N 帧 → ready
      4. 发散 → 回溯 + 封禁该 (控制轴,方向)
      5. 否则选控制轴 → 加权最小二乘步长 → 量化 → 输出命令
    """

    def __init__(self, learner: ResponseMatrixLearner | None = None) -> None:
        self.learner = learner or ResponseMatrixLearner()
        self._moves: list[MoveStep] = []
        self._pending_move: MoveStep | None = None
        self._divergence_streak = 0
        self._banned: set[tuple[int, int]] = set()   # (control, sign)
        self._backtrack_count = 0
        self._stable_frames = 0
        self._max_moves = 30
        self._max_backtracks = 3
        self._best_error: float | None = None
        self._stall_moves = 0
        # 冷启动探测：按顺序试控制轴
        self._probe_order = [0, 1, 2]
        self._probe_index = 0
        # 量化残差（小数步长累积），避免小误差永远不动
        self._carry = [0.0] * N_CONTROL

    # ---- 主决策 ----

    def compute(
        self,
        cam_left: AlignmentState,
        cam_right: AlignmentState,
        *,
        servo_limits: list[float] | None = None,
        integer_steps: dict[int, tuple[float, float]] | None = None,
        divergence_band_px: float = 3.0,
        stable_frames: int = 2,
    ) -> dict[str, Any]:
        """根据两相机状态输出下一步动作。

        Args:
            servo_limits: 每控制轴每步最大幅度 [yaw_deg, up_mm, forward_mm]
            integer_steps: 每轴 (最小步长, 最大步长)，用于整数量化
            divergence_band_px: 超过此误差增幅视为发散
            stable_frames: 连续对齐帧数达到该值才 ready

        Returns:
            {"action": "move|backtrack|idle|ready|giveup",
             "servo_index": int,   # 0=yaw 1=up 2=forward
             "delta": float,       # 已量化、实际执行的增量
             "reason": str}
        """
        limits = servo_limits or [2.0, 2.0, 2.0]
        steps = integer_steps or {
            0: (1.0, limits[0]), 1: (1.0, limits[1]), 2: (1.0, limits[2]),
        }

        # 可见性权重：不可见相机的轴不参与误差
        weights = [
            1.0 if cam_left.visible else 0.0,
            1.0 if cam_left.visible else 0.0,
            1.0 if cam_right.visible else 0.0,
            1.0 if cam_right.visible else 0.0,
        ]
        err_vec = [
            cam_left.dx_px, cam_left.dy_px,
            cam_right.dx_px, cam_right.dy_px,
        ]
        total_error = self.learner._weighted_norm(err_vec, weights)

        # 更新上一次动作的结果
        if self._pending_move is not None:
            move = self._pending_move
            move.after_error = total_error
            move.after_err_vec = list(err_vec)
            self.learner.observe(move.control_index, move.delta, move.responses)
            if move.after_error > move.before_error + divergence_band_px:
                self._divergence_streak += 1
            else:
                self._divergence_streak = 0
                self._backtrack_count = 0
                self._banned.clear()  # 有进步 → 解封
                if self._best_error is None or total_error < self._best_error - 1.0:
                    self._best_error = total_error
                    self._stall_moves = 0
                else:
                    self._stall_moves += 1
            # 超调 → 该轴步长减半（自主调节）；成功 → 缓慢恢复
            if self._overshot(move):
                self.learner.shrink_scale(move.control_index)
            elif move.after_error <= move.before_error and not move.backtrack:
                self.learner.recover_scale(move.control_index)
            self._pending_move = None
            self._stable_frames = 0

        # 两相机都对齐？
        if cam_left.aligned and cam_right.aligned:
            self._stable_frames += 1
            if self._stable_frames >= max(1, stable_frames):
                return {"action": "ready", "servo_index": -1, "delta": 0.0,
                        "reason": "both cameras aligned"}
            return {"action": "idle", "servo_index": -1, "delta": 0.0,
                    "reason": f"confirming alignment ({self._stable_frames}/{stable_frames})"}
        self._stable_frames = 0

        # 发散 → 回溯
        if self._divergence_streak >= 2 and self._moves:
            if self._backtrack_count >= self._max_backtracks:
                return {"action": "giveup", "servo_index": -1, "delta": 0.0,
                        "reason": "diverging after repeated backtracking"}
            return self._backtrack(steps)

        # 移动次数限制
        if len(self._moves) >= self._max_moves:
            return {"action": "giveup", "servo_index": -1, "delta": 0.0,
                    "reason": f"max moves ({self._max_moves}) exceeded"}

        # 选择要对齐的相机（优先误差大的可见相机）
        if cam_left.error_magnitude >= cam_right.error_magnitude:
            primary, primary_name = cam_left, "left"
        else:
            primary, primary_name = cam_right, "right"

        # 停滞 → 强制探测样本最少的控制轴（探索，跳出局部极小）
        force_probe = self._stall_moves >= 4 and len(self._moves) >= 4

        # 选择控制轴
        exclude = {c for c, _sign in self._banned}
        control = self._select_control(primary, primary_name, err_vec, weights,
                                       exclude, force_probe)
        if control < 0:
            return {"action": "idle", "servo_index": -1, "delta": 0.0,
                    "reason": "no usable control axis"}

        # 计算方向与幅度
        delta = self._step_for_control(control, err_vec, weights,
                                       steps[control], primary)
        if delta == 0.0:
            # 预测已无有效步长但误差仍超容差 → 用启发式小步探测收集数据
            delta = self._probe_delta(control, primary, steps[control])
            if delta == 0.0:
                return {"action": "idle", "servo_index": -1, "delta": 0.0,
                        "reason": "predicted step below actuator resolution"}

        # 记录动作
        move = MoveStep(
            control_index=control,
            delta=delta,
            before_error=total_error,
            before_err_vec=list(err_vec),
        )
        self._moves.append(move)
        self._pending_move = move
        self._probe_index += 1

        return {
            "action": "move",
            "servo_index": control,
            "delta": delta,
            "reason": (
                f"align {primary_name} cam: {CONTROL_NAMES[control]} {delta:+.0f} "
                f"err=({primary.dx_px:.0f},{primary.dy_px:.0f})px "
                f"total={total_error:.0f}px scale={self.learner.scale(control):.2f} "
                f"learned={self.learner.is_learned(minimum_samples=1)}"
            ),
        }

    # ---- 控制轴选择 ----

    def _select_control(self, primary: AlignmentState, primary_name: str,
                        err_vec: list[float], weights: list[float],
                        exclude: set[int], force_probe: bool) -> int:
        # 优先使用学习到的响应
        if not force_probe and self.learner.is_learned(minimum_samples=1):
            control = self.learner.best_control_for_error(
                err_vec, weights, exclude_controls=exclude)
            if control >= 0:
                return control
            # 全部被排除 → 允许使用排除列表
            return self.learner.best_control_for_error(err_vec, weights)
        # 冷启动 / 停滞探测: 轮流试控制轴，跳过封禁的
        for offset in range(N_CONTROL):
            control = self._probe_order[(self._probe_index + offset) % N_CONTROL]
            if control not in exclude:
                return control
        return -1

    def _step_for_control(self, control: int, err_vec: list[float],
                          weights: list[float],
                          step_spec: tuple[float, float],
                          primary: AlignmentState) -> float:
        """加权最小二乘步长 + 轴缩放 + 整数量化（带残差累积）。"""
        response = self.learner.response_for_control(control)
        den = sum(w * r * r for w, r in zip(weights, response))
        step_min, step_max = step_spec
        if den > 1e-6:
            num = sum(w * e * r for w, e, r in zip(weights, err_vec, response))
            raw = num / den  # 使新误差与响应正交的最优步长
        else:
            raw = 0.0
        raw *= self.learner.scale(control)
        raw += self._carry[control]
        if abs(raw) < step_min * 0.5:
            self._carry[control] = raw
            return 0.0
        quantized = math.copysign(
            step_min * round(abs(raw) / step_min), raw)
        quantized = max(-step_max, min(step_max, quantized))
        self._carry[control] = raw - quantized
        return quantized

    def _probe_delta(self, control: int, primary: AlignmentState,
                     step_spec: tuple[float, float]) -> float:
        """冷启动探测方向与幅度。误差大时用更大步（否则 300px 误差
        需要几十步才走完）。符号启发式（猜错由发散+回溯纠正）。"""
        step_min, step_max = step_spec
        # 误差 <60px 用最小步，误差越大步长越大，封顶 step_max 的 80%
        magnitude = min(
            step_max * 0.8,
            max(step_min, step_min * (primary.error_magnitude / 60.0)),
        )
        if control == 0:      # yaw 主要影响水平
            sign = 1.0 if primary.dx_px > 0 else -1.0
        else:                 # up/forward 主要影响垂直
            sign = -1.0 if primary.dy_px > 0 else 1.0
        return math.copysign(magnitude, sign)

    # ---- 回溯 ----

    def _backtrack(self, step_specs: dict[int, tuple[float, float]]) -> dict[str, Any]:
        """回退上一步（反向一半幅度），并封禁该 (控制轴,方向)。"""
        if not self._moves:
            self._divergence_streak = 0
            return {"action": "idle", "servo_index": -1, "delta": 0.0,
                    "reason": "nothing to backtrack"}
        move = self._moves.pop()
        self._divergence_streak = 0
        self._backtrack_count += 1
        sign = 1 if move.delta > 0 else -1
        self._banned.add((move.control_index, sign))
        step_min, step_max = step_specs.get(move.control_index, (1.0, 2.0))
        delta = -math.copysign(
            max(step_min, min(step_max, abs(move.delta) * 0.5)), move.delta)
        # 回溯本身也是一次动作：记录它以便测响应、防连环发散
        reverse = MoveStep(
            control_index=move.control_index,
            delta=delta,
            before_error=move.after_error if move.after_error is not None else move.before_error,
            before_err_vec=move.after_err_vec or move.before_err_vec,
            backtrack=True,
        )
        self._moves.append(reverse)
        self._pending_move = reverse
        return {
            "action": "backtrack",
            "servo_index": reverse.control_index,
            "delta": delta,
            "reason": (f"backtrack {CONTROL_NAMES[move.control_index]} "
                       f"{delta:+.0f}; banned {move.control_index}{'+' if sign > 0 else '-'}"),
        }

    def _overshot(self, move: MoveStep) -> bool:
        """任一可见轴误差符号翻转且翻转后仍超容差 → 超调。"""
        if move.after_err_vec is None:
            return False
        for axis in range(N_AXES):
            before = move.before_err_vec[axis]
            after = move.after_err_vec[axis]
            if abs(before) < 1e-6 or abs(after) < 1e-6:
                continue
            if before * after < 0 and abs(after) > 6.0:
                return True
        return False

    # ---- 响应测量 ----

    def note_response(self, responses: dict[str, float | None]) -> None:
        """记录上一动作在两个相机中的实际夹爪像素位移。"""
        if self._pending_move is not None:
            self._pending_move.responses = dict(responses)

    def reset(self) -> None:
        """清空会话状态（保留已学习的矩阵与缩放）。"""
        self._moves.clear()
        self._pending_move = None
        self._divergence_streak = 0
        self._banned.clear()
        self._backtrack_count = 0
        self._stable_frames = 0
        self._best_error = None
        self._stall_moves = 0
        self._probe_index = 0
        self._carry = [0.0] * N_CONTROL

    def save_learning(self, path: str | None = None) -> None:
        self.learner.save(path or LEARNED_RESPONSE_PATH)

    def status(self) -> dict[str, Any]:
        return {
            "moves": len(self._moves),
            "divergence_streak": self._divergence_streak,
            "backtracks": self._backtrack_count,
            "banned": sorted(self._banned),
            "learned": self.learner.is_learned(minimum_samples=1),
            "matrix_samples": self.learner._total_samples,
            "scales": list(self.learner._scales),
        }
