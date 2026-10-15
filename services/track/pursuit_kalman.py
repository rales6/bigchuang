# -*- coding: utf-8 -*-
"""追踪链路卡尔曼滤波器（3090 端，纯 numpy，零新依赖）。

CV 模型 6 状态 x=[cx, cy, w, h, vx, vy]（中心/尺寸/像面速度）：
- predict：车体转动引起的像面位移作为控制输入 u=[ux, uy]（像素/秒）
  提前建模——转弯/旋转重搜时目标在画面里的大幅移动大部分是车自身
  运动引起的，不建模的话预测会显著滞后（文献：SAM2Plus / HiM2SAM
  等 2025 长时跟踪改进都把 KF 预测作为遮挡恢复的核心组件）；
- update：Mahalanobis 门控（位置 2 自由度 chi2 p=0.99）拒绝与预测
  不一致的远跳测量，防"随机跳框/认错"；强观测（模板高分/YOLO
  类别命中）可用 force=True 越过门控——门太严会把滤波器锁死在
  错误路径上（文献 JPDA/RobMOT 的教训）；
- predicted_box() 供搜索中心先验与 EMA 平滑使用。
"""

from __future__ import annotations

import math

import numpy as np

# 位置分量门控阈值：chi2 2 自由度 p=0.99 ≈ 9.21（约 3σ 椭圆）
GATE_POS_CHI2 = 9.21
# 相对加速度过程噪声（像素/秒²）：小车近距机动时目标像面速度变化快，
# 给足过程噪声保证门不误杀正确测量。
_ACCEL_NOISE_PX_S2 = 900.0
# 测量噪声：位置 10px、尺寸 12px 标准差（YOLO/CSRT/模板量级）
_R_POS = 100.0
_R_SIZE = 144.0
# 近距大目标位置噪声按尺寸放大的参考尺寸（像素）：框边长达到该值后
# 门控从"绝对像素"改为"相对尺寸"——CSRT/模板对 400px 大框的抖动
# ±15px 是合法噪声，固定 σ=10 会把正确测量当野值拒掉（近距丢锁）。
# 注意：平滑漂移（19px/帧）仍能穿过门控，靠服务端漂移看门狗拦截。
_R_POS_REF_SIZE = 150.0


class KalmanBoxTracker:
    """单路 bbox 的匀速卡尔曼滤波器。"""

    def __init__(self, box, now: float = 0.0) -> None:
        cx = (box.x1 + box.x2) / 2.0
        cy = (box.y1 + box.y2) / 2.0
        w = max(float(box.x2 - box.x1), 4.0)
        h = max(float(box.y2 - box.y1), 4.0)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([400.0, 400.0, 400.0, 400.0,
                          90000.0, 90000.0])
        self.R = np.diag([_R_POS, _R_POS, _R_SIZE, _R_SIZE])
        self.last_t = float(now)
        self.x_pred: np.ndarray = self.x.copy()
        self.P_pred: np.ndarray = self.P.copy()

    def predict(self, dt: float, u: tuple[float, float]) -> None:
        """匀速预测 + ego-motion 控制输入（像面速度 px/s）。"""
        dt = max(0.0, min(float(dt), 0.2))
        F = np.array([[1.0, 0.0, 0.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        self.x = F @ self.x
        self.x[0] += u[0] * dt
        self.x[1] += u[1] * dt
        q_p = (0.5 * _ACCEL_NOISE_PX_S2 * dt * dt) ** 2
        q_v = (_ACCEL_NOISE_PX_S2 * dt) ** 2
        # 尺寸过程噪声与当前尺寸成比例：近距逼近时目标在画面里快速
        # 变大（实测可达 ~40%/s），固定噪声会把尺度变化误杀为野值。
        q_s = (0.35 * max(abs(self.x[2]), 10.0) * dt) ** 2
        Q = np.diag([q_p, q_p, q_s, q_s, q_v, q_v])
        self.P = F @ self.P @ F.T + Q
        self.x_pred = self.x.copy()
        self.P_pred = self.P.copy()
        self.last_t += dt

    def update(self, measured_box, force: bool = False) -> tuple[bool, float]:
        """门控更新。返回 (是否采纳, 马氏距离)。"""
        z = np.array([(measured_box.x1 + measured_box.x2) / 2.0,
                      (measured_box.y1 + measured_box.y2) / 2.0,
                      measured_box.x2 - measured_box.x1,
                      measured_box.y2 - measured_box.y1],
                     dtype=np.float64)
        H = np.zeros((4, 6), dtype=np.float64)
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        v = z - H @ self.x
        # 近距大目标门控宽放：位置测量噪声按当前框尺寸放大（见
        # _R_POS_REF_SIZE），否则 CSRT/模板对 400px 大框 ±15px 的
        # 正常抖动会被当野值拒掉。
        size_scale = max(1.0, min(self.x[2], self.x[3]) / _R_POS_REF_SIZE)
        r_pos = _R_POS * size_scale * size_scale
        S = H @ self.P @ H.T + np.diag([r_pos, r_pos, _R_SIZE, _R_SIZE])
        s_pos = S[:2, :2]
        v_pos = v[:2]
        try:
            dist2 = float(v_pos @ np.linalg.inv(s_pos) @ v_pos)
        except np.linalg.LinAlgError:
            dist2 = float("inf")
        dist = math.sqrt(max(dist2, 0.0))
        if not force and dist2 > GATE_POS_CHI2:
            # 位置拒收（防漂移/跳框），但尺寸照收：近距目标快速变大
            # （逼近时 ~40%/s），KF 输出预测框期间尺寸若冻结在旧值，
            # 就会出现"框大小不跟本帧图片"（用户反馈的流畅性根因）。
            # 尺寸不是安全关键量（停车判据只看中心），漂移期间的
            # 错误尺寸由模板/YOLO 追回纠正。
            self._update_size(z)
            return False, dist
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False, dist
        self.x = self.x + K @ v
        self.P = (np.eye(6) - K @ H) @ self.P
        return True, dist

    def _update_size(self, z: np.ndarray) -> None:
        """只采纳尺寸分量（w,h）的测量更新，位置/速度保持不动。"""
        H_s = np.zeros((2, 6), dtype=np.float64)
        H_s[0, 2] = H_s[1, 3] = 1.0
        S_s = H_s @ self.P @ H_s.T + np.diag([_R_SIZE, _R_SIZE])
        try:
            K_s = self.P @ H_s.T @ np.linalg.inv(S_s)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K_s @ (z[2:] - H_s @ self.x)
        self.P = (np.eye(6) - K_s @ H_s) @ self.P
        # 输出快照同步：服务端拒收路径输出 predicted_box()，x_pred
        # 的尺寸不跟着走的话"框大小跟本帧"的修复在输出端不可见。
        self.x_pred[2:4] = self.x[2:4]

    def update_size(self, measured_box) -> None:
        """外部调用：只更新尺寸（服务端近距大框 CSRT 模板印证失败、
        回滚位置后调用，让尺寸仍跟随测量）。"""
        z = np.array([measured_box.x2 - measured_box.x1,
                      measured_box.y2 - measured_box.y1],
                     dtype=np.float64)
        H_s = np.zeros((2, 6), dtype=np.float64)
        H_s[0, 2] = H_s[1, 3] = 1.0
        S_s = H_s @ self.P @ H_s.T + np.diag([_R_SIZE, _R_SIZE])
        try:
            K_s = self.P @ H_s.T @ np.linalg.inv(S_s)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K_s @ (z - H_s @ self.x)
        self.P = (np.eye(6) - K_s @ H_s) @ self.P
        self.x_pred[2:4] = self.x[2:4]

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """KF 状态快照（x, P）。近距大框 CSRT 需模板印证后才采纳，
        印证失败要回滚本次 update，防止漂移测量污染预测。"""
        return self.x.copy(), self.P.copy()

    def restore(self, snap: tuple[np.ndarray, np.ndarray]) -> None:
        self.x, self.P = snap

    def predicted_box(self) -> tuple[float, float, float, float]:
        """预测的中心与尺寸（更新前快照，供 EMA 平滑）。"""
        cx, cy, w, h = (float(v) for v in self.x_pred[:4])
        return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

    @property
    def predicted_center(self) -> tuple[float, float]:
        return float(self.x_pred[0]), float(self.x_pred[1])
