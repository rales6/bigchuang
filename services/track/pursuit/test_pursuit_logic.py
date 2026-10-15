# -*- coding: utf-8 -*-
"""PursuitController 无头逻辑冒烟测试（不依赖 cv2/硬件）。"""
import math
import sys
from pathlib import Path

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

import numpy as np

from pursuit.pursuit_config import PursuitConfig, StereoConfig
from pursuit.camera_stream import frame_is_healthy
from pursuit.pursuit_controller import (Decision, Motion, Perception,
                                        PursuitController, State)
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping.geometry import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult

HALF = 1280
# 瞄准点是各相机原始照片主点（标定 K[0][2]：左 573.41 / 右 653.81）
AIM = StereoConfig().cx_raw_left_px
AIM_R = HALF + StereoConfig().cx_raw_right_px


def test_camera_frame_health_guard():
    normal = np.zeros((480, 640, 3), dtype=np.uint8)
    normal[:, :, 1] = np.arange(640, dtype=np.uint8)[None, :]
    assert frame_is_healthy(normal)
    corrupt = normal.copy()
    corrupt[120:, :, :] = 128
    assert not frame_is_healthy(corrupt)


def scan(angles_rad, distances_m):
    return LaserScan(np.asarray(angles_rad, dtype=np.float64),
                     np.asarray(distances_m, dtype=np.float64), 0.0)


def scan_front(distance_m):
    """一圈扫描：正前方 ±10° 给定距离，其余 NaN。"""
    angles = np.linspace(-math.pi, math.pi, 360)
    distances = np.full_like(angles, np.nan)
    mask = np.abs(angles) < math.radians(10)
    distances[mask] = distance_m
    return scan(angles, distances)


def scan_empty_front():
    """一圈扫描：全部 NaN（前方无有效点）。"""
    angles = np.linspace(-math.pi, math.pi, 360)
    return scan(angles, np.full_like(angles, np.nan))


def scan_with_rear(front_m=2.0, rear_m=0.2):
    """同时给出前/后扇区，用于验证回退防撞。"""
    angles = np.linspace(-math.pi, math.pi, 360)
    distances = np.full_like(angles, np.nan)
    distances[np.abs(angles) < math.radians(10)] = front_m
    rear_mask = np.abs(np.abs(angles) - math.pi) < math.radians(10)
    distances[rear_mask] = rear_m
    return scan(angles, distances)


def make(t0=1000.0):
    # 旧测距模式（距离驱动）用于既有回归；默认配置是行测距+示教停车。
    return PursuitController(PursuitConfig(instruction="测试",
                                           use_visual_row_distance=False,
                                           use_taught_stop=False,
                                           pulse_drive_enabled=False),
                             StereoConfig(), HALF), t0


def per(tracking=True, cx=HALF / 2, depth=1.0, conf=0.5):
    bbox = (cx - 40, 300, cx + 40, 400)
    return Perception(tracking=tracking, bbox=bbox, depth_m=depth, depth_conf=conf)


def main():
    test_camera_frame_health_guard()
    # 1) INIT → ground found → TRACKING
    c, t = make()
    assert c.state == State.INIT
    d = c.tick(None, None, t)
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s == 0
    c.on_ground_result(True, t)
    assert c.state == State.TRACKING

    # 2) 目标在右目主点位置 → 视线偏差 0，不转向；雷达 2.0 → 逼近
    #    （默认 use_stereo_depth=False：距离只信雷达）
    d = c.tick(per(cx=AIM_R), scan_front(2.0), t + 0.05)
    assert d.motion.angular_mrad_s == 0, d.motion
    assert d.motion.linear_mm_s > 0, d.motion

    # 3) 内八字相机按画面中间 1/3 判断航向：中间区内直行，
    #    只有越过左/右边界才左/右转。
    assert c._steer_for_bbox((600, 300, 680, 400)) == 0
    assert c._steer_for_bbox((900, 300, 980, 400)) < 0
    assert c._steer_for_bbox((300, 300, 380, 400)) > 0

    # 4) 安全回归：深度 None + 雷达前方无有效点 → 必须原地等待，绝不能全速前进
    c2, t2 = make()
    c2.on_ground_result(True, t2)
    d = c2.tick(Perception(tracking=True, bbox=(AIM - 40, 300, AIM + 40, 400),
                           depth_m=None, depth_conf=0.0),
                scan_empty_front(), t2 + 0.05)
    assert d.motion.linear_mm_s == 0, f"安全回归失败: {d.motion}"
    assert "no distance" in d.motion.reason, d.motion

    # 5) 深度 0.1 / 雷达前向 3.0 → 默认忽略双目深度，取雷达 3.0 逼近
    d = c2.tick(Perception(tracking=True, bbox=(AIM - 40, 300, AIM + 40, 400),
                           depth_m=0.1, depth_conf=0.9),
                scan_front(3.0), t2 + 0.10)
    assert "d=3.00m" in d.motion.reason, d.motion
    assert d.motion.linear_mm_s > 0

    # 5b) use_stereo_depth=True 时恢复取 min(深度, 雷达)：1.5 < 3.0 → 1.5
    c2b, t2b = make(1500.0)
    c2b = PursuitController(PursuitConfig(instruction="测试",
                                          use_stereo_depth=True,
                                          use_visual_row_distance=False,
                                          use_taught_stop=False),
                            StereoConfig(), HALF)
    c2b.on_ground_result(True, t2b)
    d = c2b.tick(Perception(tracking=True, bbox=(AIM - 40, 300, AIM + 40, 400),
                            depth_m=1.5, depth_conf=0.5),
                 scan_front(3.0), t2b + 0.05)
    assert "d=1.50m" in d.motion.reason, d.motion
    assert d.motion.linear_mm_s > 0

    # 6) 连续 3 帧 ≤ 0.2m(+0.03 容差) → STOPPED 成功
    for i in range(3):
        d = c2.tick(Perception(tracking=True, bbox=(AIM - 40, 300, AIM + 40, 400),
                               depth_m=0.21, depth_conf=0.5),
                    scan_front(0.21), t2 + 0.2 + i * 0.05)
    assert c2.state == State.STOPPED and d.finished and d.result == "success", d

    # 7) 丢失 1.8s → SEARCHING；已有身份时每个端点短暂停留给实时
    # 双目重识别，不在每一步重复阻塞 Qwen。
    c3, t3 = make()
    c3.on_ground_result(True, t3)
    now = t3
    for _ in range(12):
        now += 0.2
        d = c3.tick(Perception(tracking=False, bbox=None, depth_m=None),
                    None, now)
    assert c3.state == State.SEARCHING, c3.state
    d = c3.tick(None, None, now + 0.05)
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s != 0, d.motion
    # 推进到退化计时完成
    now += 5.0
    d = c3.tick(None, None, now)
    assert not d.reground_now and "identity re-acquire" in d.motion.reason, d

    # 8) 各端点均未实时找回 → 自动依次推进；转满才做最终 Qwen 兜底
    # 并进入 GIVE_UP。
    steps = PursuitConfig().search_rotation_steps_deg
    for _ in range(len(steps) + 2):
        now += 0.4  # 端点停留结束，开始下一步
        d = c3.tick(None, None, now)
        if c3.state == State.GIVE_UP:
            break
        now += 30.0  # 无雷达时长退化：足够完成任意剩余角度
        d = c3.tick(None, None, now)
        if c3.state == State.GIVE_UP:
            break
    assert c3.state == State.GIVE_UP, c3.state
    assert d.finished and d.result == "give_up", d

    # 9) 雷达硬停：前方 0.05m < 0.12m → GIVE_UP
    c4, t4 = make()
    c4.on_ground_result(True, t4)
    d = c4.tick(per(depth=1.0), scan_front(0.05), t4 + 0.05)
    assert c4.state == State.GIVE_UP and d.finished and d.result == "give_up", d

    # 10) 初始 ground 未找到 → 直接进入 SEARCHING
    c5, t5 = make()
    c5.on_ground_result(False, t5)
    assert c5.state == State.SEARCHING

    # 11) TRACKING 中收到测试网页 reground 未找到 → 进入 SEARCHING
    c6, t6 = make()
    c6.on_ground_result(True, t6)
    c6.on_ground_result(False, t6 + 0.1)
    assert c6.state == State.SEARCHING, c6.state

    # 12) TRACKING 中收到测试网页 reground 找到 → 保持 TRACKING（换目标）
    c7, t7 = make()
    c7.on_ground_result(True, t7)
    c7.on_ground_result(True, t7 + 0.1)
    assert c7.state == State.TRACKING, c7.state

    # 13) dry-run 模拟：车不动（静态扫描 ICP 恒 0），旋转步按时长完成
    c8, t8 = make(2000.0)
    c8._simulate_motion = True
    c8.on_ground_result(True, t8)
    for _ in range(12):
        t8 += 0.2
        c8.tick(Perception(tracking=False, bbox=None, depth_m=None),
                scan_front(3.0), t8)
    assert c8.state == State.SEARCHING, c8.state
    d = c8.tick(None, scan_front(3.0), t8 + 0.05)
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s != 0, d.motion
    t8 += 3.5   # 45° 步 @600mrad/s*0.5 效率 ≈ 2.6s，3.5s 足够转完
    d = c8.tick(None, scan_front(3.0), t8)
    assert not d.reground_now and "identity re-acquire" in d.motion.reason, d

    # 14) 短暂丢失：先完整停车等待全画面重捕获（不按旧框继续转）
    c9, t9 = make()
    c9.on_ground_result(True, t9)
    c9.tick(per(cx=AIM + 100, depth=1.0), None, t9 + 0.05)
    d = c9.tick(Perception(tracking=False, bbox=None, depth_m=None),
                None, t9 + 0.10)
    assert d.motion.linear_mm_s == 0, d.motion
    assert d.motion.angular_mrad_s == 0, d.motion
    assert "settle/reacquire" in d.motion.reason, d.motion
    assert c9.state == State.TRACKING, c9.state

    # 15) 服务端 busy（网页重新识别中）不算丢失：持续 2s 也不进 SEARCHING
    c10, t10 = make()
    c10.on_ground_result(True, t10)
    for _ in range(10):   # 2s > lost_before_search_s=1.2s
        t10 += 0.2
        d = c10.tick(Perception(tracking=False, bbox=None, depth_m=None,
                                status="busy"), None, t10)
    assert c10.state == State.TRACKING, c10.state
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s == 0, d.motion

    # 16) ICP 永不收敛（特征不足 < min_points=35）→ 若干卡死窗口后
    #     按时长退化，旋转步仍能完成并进入身份重识别停留（不无限转）
    def sparse_scan():
        angles = np.asarray([math.radians(a) for a in (-5, 0, 5)],
                            dtype=np.float64)
        return scan(angles, np.full_like(angles, 3.0))

    c11, t11 = make(3000.0)
    c11.on_ground_result(True, t11)
    for _ in range(12):
        t11 += 0.2
        c11.tick(Perception(tracking=False, bbox=None, depth_m=None),
                 sparse_scan(), t11)
    assert c11.state == State.SEARCHING, c11.state
    d = None
    for _ in range(50):   # 4 个卡死窗口(6s) + 退化时长(~1.05s) < 10s
        t11 += 0.2
        d = c11.tick(None, sparse_scan(), t11)
        if "identity re-acquire" in d.motion.reason:
            break
    assert not d.reground_now and "identity re-acquire" in d.motion.reason, d

    # 17) 近端（0.4m < 0.5m）目标越出中间 1/3 后小幅转向，
    #     且仍以温和长脉冲逼近。
    c12, t12 = make()
    c12.on_ground_result(True, t12)
    d = c12.tick(per(cx=900, depth=0.4), scan_front(0.4), t12 + 0.05)
    assert d.motion.linear_mm_s > 0, d.motion
    assert -80 <= d.motion.angular_mrad_s < 0, d.motion

    # 18) 每步前方都有障碍 → 全部跳过 → 直接 GIVE_UP（不再朝钳位后的
    #     最后方向继续转）
    c13, t13 = make(4000.0)
    c13.on_ground_result(False, t13)
    d = c13.tick(None, scan_front(0.20), t13 + 0.05)
    assert c13.state == State.GIVE_UP and d.finished and d.result == "give_up", d

    # 19) 视觉行测距（默认模式）：上方目标全速；进入下带减速；
    #     中心行 ≥ 停车行连续 3 帧 → STOPPED（不依赖深度/雷达）
    c14, t14 = make(5000.0)
    c14 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=True,
                                          use_taught_stop=False),
                            StereoConfig(), HALF)
    c14.on_ground_result(True, t14)
    cfg = PursuitConfig(use_visual_row_distance=True, use_taught_stop=False)
    d = None
    for _ in range(15):   # 冲量 0.4s + 减速斜坡 0.2s 后回落到全速
        t14 += 0.05
        d = c14.tick(per(cx=AIM, depth=None), scan_front(1.5), t14)
    assert "visual servo" in d.motion.reason, d.motion
    assert d.motion.linear_mm_s in (0, cfg.visual_pulse_mm_s), d.motion
    # 目标行 y 分步下移（每帧 ≤110px，模拟车逐步接近）
    near_motions = []
    for mid_y in (400, 480, 560):
        t14 += 0.05
        d = c14.tick(Perception(tracking=True,
                                bbox=(AIM - 40, mid_y, AIM + 40,
                                      mid_y + 100),
                                depth_m=None), scan_front(1.5), t14)
        near_motions.append(d.motion)
    assert all("visual servo" in m.reason for m in near_motions), near_motions
    for i in range(3):
        d = c14.tick(Perception(tracking=True,
                                bbox=(AIM - 40, 700, AIM + 40, 760),
                                depth_m=None), scan_front(1.5),
                     t14 + 0.2 + i * 0.05)
    assert c14.state == State.STOPPED and d.finished, d

    # 20) 示教停车：两路中心进入各自完整掩码（不再只认质心 ±30px）
    #     即 STOPPED；偏差大时步进逼近并转向示教 x。
    c15, t15 = make(6000.0)
    tcfg = PursuitConfig(instruction="测试", use_visual_row_distance=False,
                         use_taught_stop=True, recovery_rear_sensor_usable=True)
    c15 = PursuitController(tcfg, StereoConfig(), HALF)
    c15.on_ground_result(True, t15)
    lx, ly = tcfg.taught_left_cx_px, tcfg.taught_left_cy_px
    rx, ry = tcfg.taught_right_cx_px, tcfg.taught_right_cy_px
    # 偏离较大（>slow_band 之外）→ 起步冲量+减速斜坡后全速
    # （横向对齐示教 x，纵向偏差 425px > band）
    d = None
    for _ in range(15):
        t15 += 0.05
        d = c15.tick(Perception(tracking=True,
                                bbox=(lx - 30, 100, lx + 30, 140),
                                depth_m=None,
                                bbox2=(HALF + rx - 30, 100,
                                       HALF + rx + 30, 140),
                                tracking2=True),
                     scan_front(1.5), t15)
    assert "visual servo" in d.motion.reason, d.motion
    assert d.motion.linear_mm_s in (0, tcfg.visual_pulse_mm_s), d.motion
    # 目标逐步接近示教点（每帧 y ≤110px，防跳变保护误触）
    for step_y in (180, 280, 330):
        t15 += 0.05
        d = c15.tick(Perception(tracking=True,
                                bbox=(lx - 30, step_y, lx + 30,
                                      step_y + 40),
                                depth_m=None,
                                bbox2=(HALF + rx - 30, step_y,
                                       HALF + rx + 30, step_y + 40),
                                tracking2=True),
                     scan_front(1.5), t15)
    # 两路都在各自掩码上方、但已进入慢速带 → 脉冲/停稳视觉伺服
    d = c15.tick(Perception(tracking=True,
                            bbox=(lx - 30, 330, lx + 30, 370),
                            depth_m=None,
                            bbox2=(HALF + rx - 30, 220,
                                   HALF + rx + 30, 260),
                            tracking2=True),
                 scan_front(1.5), t15 + 0.08)
    assert "visual servo" in d.motion.reason, d.motion
    # 两路进入掩码时必须立即停车；新预测缓冲还会比这里更早触发。
    bbox_l_close = (620, 560, 680, 620)  # center=(650,590) > 545+30
    bbox_r_close = (HALF + 570, 440, HALF + 630, 500)  # (600,470) > 421+30
    d = c15.tick(Perception(tracking=True, bbox=bbox_l_close, depth_m=None,
                            bbox2=bbox_r_close, tracking2=True),
                 scan_front(1.5), t15 + 0.20)
    if not d.finished:  # 首帧大跳受安全门控；同位置下一新帧确认后停车
        d = c15.tick(Perception(tracking=True, bbox=bbox_l_close, depth_m=None,
                                bbox2=bbox_r_close, tracking2=True),
                     scan_front(1.5), t15 + 0.25)
    assert c15.state == State.STOPPED and d.finished, d
    # 实车 2026-09-20：上一观测仍距掩码约 118px；下一观测左目仅
    # 43px、右目已入掩码。若再发最短 0.8s 脉冲，中心会从 y≈394
    # 直接跳到 y≈640 并出画。应在 50px 缓冲区预测停车。
    c15_predict = PursuitController(tcfg, StereoConfig(), HALF)
    c15_predict.on_ground_result(True, t15)
    d = c15_predict.tick(
        Perception(
            tracking=True, bbox=(783.7, 340.1, 912.0, 448.1),
            depth_m=None,
            bbox2=(2000.2, 308.1, 2108.0, 432.9), tracking2=True),
        scan_front(1.5), t15 + 0.05)
    assert d.finished and d.result == "alignment_pending", d
    assert d.motion.linear_mm_s == 0, d.motion
    assert "提前停车缓冲区" in d.detail, d
    # 示教模式：两路都越出画面中间 1/3 的左边界，
    # 才做温和滚动左转脉冲。
    c15b, t15b = make(7000.0)
    c15b = PursuitController(tcfg, StereoConfig(), HALF)
    c15b.on_ground_result(True, t15b)
    d = c15b.tick(Perception(tracking=True,
                             bbox=(290, 280, 370, 320),
                             depth_m=None,
                             bbox2=(HALF + 310, 180,
                                    HALF + 390, 220),
                             tracking2=True),
                  scan_front(1.5), t15b + 0.05)
    assert d.motion.linear_mm_s == tcfg.align_arc_linear_mm_s, d.motion
    assert d.motion.angular_mrad_s == tcfg.align_turn_pulse_mrad_s, d.motion
    assert "align arc pulse" in d.motion.reason, d.motion
    d = c15b.tick(Perception(tracking=True,
                             bbox=(290, 280, 370, 320), depth_m=None,
                             bbox2=(HALF + 310, 180,
                                    HALF + 390, 220), tracking2=True),
                  scan_front(1.5), t15b + 0.90)
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s == 0, d.motion
    assert "align arc rest" in d.motion.reason, d.motion
    # 最新实车日志：左中心 611、右局部中心 770 都在各自
    # 画面的中间 1/3，内八字视差不得误触发左/右转。
    c15d = PursuitController(tcfg, StereoConfig(), HALF)
    c15d.on_ground_result(True, t15b)
    logged_per = Perception(
        tracking=True, bbox=(591, 158, 631, 198), depth_m=None,
        bbox2=(HALF + 750, 115, HALF + 790, 155), tracking2=True)
    d = c15d.tick(logged_per, scan_front(1.5), t15b + 0.05)
    assert d.motion.angular_mrad_s == 0, d.motion
    assert d.motion.linear_mm_s > 0, d.motion
    # 右目仅略过中间 1/3（约 44px）不能触发大幅弧线脉冲。
    c15m = PursuitController(tcfg, StereoConfig(), HALF)
    c15m.on_ground_result(True, t15b)
    d = c15m.tick(Perception(
        tracking=True, bbox=(790, 160, 850, 220), depth_m=None,
        bbox2=(HALF + 865, 150, HALF + 925, 210), tracking2=True),
        scan_front(1.5), t15b + 0.05)
    assert "align arc" not in d.motion.reason, d.motion
    # 两路分别向相反方向越界也是内八字正常视差，应直行。
    angular, error = c15d._steer_for_dual_bearing(
        (280, 100, 360, 180), (HALF + 920, 100, HALF + 1000, 180))
    assert angular == 0 and error == 0.0, (angular, error)
    # 示教任务缺任一路：先停车检查当前视觉，再根据仍可见
    # 的左目倒车左转找回右目；回退超时才进入旋转重搜。
    bbox_l = (470, 540, 530, 580)
    c15c = PursuitController(tcfg, StereoConfig(), HALF)
    c15c.on_ground_result(True, t15b)
    d = c15c.tick(Perception(tracking=True, bbox=bbox_l, depth_m=None),
                  scan_with_rear(1.5, 1.0), t15b + 0.05)
    assert d.motion.linear_mm_s == 0 \
        and "recovery visual check single-left" in d.motion.reason, d.motion
    d = c15c.tick(Perception(tracking=True, bbox=bbox_l, depth_m=None),
                  scan_with_rear(1.5, 1.0), t15b + 0.35)
    assert d.motion.linear_mm_s == -tcfg.recovery_reverse_mm_s, d.motion
    assert d.motion.angular_mrad_s > 0 and "single-left" in d.motion.reason
    d = c15c.tick(Perception(tracking=True, bbox=bbox_l, depth_m=None),
                  scan_with_rear(1.5, 1.0), t15b + 4.60)
    assert d.reground_now and d.motion.linear_mm_s == 0, d
    c15c.on_ground_result(False, t15b + 4.61)
    assert c15c.state == State.SEARCHING, c15c.state

    # 只有右目当前可见 → 当前视觉优先，倒车右转。
    c15e = PursuitController(tcfg, StereoConfig(), HALF)
    c15e.on_ground_result(True, t15b)
    right_only = Perception(
        tracking=False, bbox=None, depth_m=None,
        tracking2=True,
        bbox2=(HALF + 580, 200, HALF + 660, 280))
    c15e.tick(right_only, scan_with_rear(1.5, 1.0), t15b + 0.05)
    d = c15e.tick(right_only, scan_with_rear(1.5, 1.0), t15b + 0.35)
    assert d.motion.linear_mm_s == -tcfg.recovery_reverse_mm_s, d.motion
    assert d.motion.angular_mrad_s < 0 and "single-right" in d.motion.reason

    # 后方距离不足时即使单目有框也不得倒车。
    c15f = PursuitController(tcfg, StereoConfig(), HALF)
    c15f.on_ground_result(True, t15b)
    c15f.tick(right_only, scan_with_rear(), t15b + 0.05)
    d = c15f.tick(right_only, scan_with_rear(), t15b + 0.35)
    assert d.motion.linear_mm_s == 0 and "rear blocked" in d.motion.reason

    # 双目当前都丢失：才使用丢失前多帧向右的运动趋势。
    c15g = PursuitController(tcfg, StereoConfig(), HALF)
    c15g.on_ground_result(True, t15b)
    for idx, mean_x in enumerate((560.0, 620.0, 700.0)):
        hist = Perception(
            tracking=True,
            bbox=(mean_x - 80, 180, mean_x, 260), depth_m=None,
            tracking2=True,
            bbox2=(HALF + mean_x, 180,
                   HALF + mean_x + 80, 260),
            frame_id=idx + 1, received_at=t15b + idx * 0.10)
        c15g._record_dual_history(hist, t15b + idx * 0.10)
    both_lost = Perception(tracking=False, bbox=None, depth_m=None)
    c15g.tick(both_lost, scan_with_rear(1.5, 1.0), t15b + 0.30)
    d = c15g.tick(both_lost, scan_with_rear(1.5, 1.0), t15b + 0.60)
    assert d.motion.linear_mm_s == -tcfg.recovery_reverse_mm_s, d.motion
    assert d.motion.angular_mrad_s < 0 \
        and "history-both-lost" in d.motion.reason, d.motion

    # 当前单目框必须压过历史预测：即使历史向右，
    # 当前左目仍看得到就按左目引导向左找回。
    c15h = PursuitController(tcfg, StereoConfig(), HALF)
    c15h.on_ground_result(True, t15b)
    c15h._recent_dual_centers = [
        (t15b, 600.0), (t15b + 0.1, 760.0)]
    left_only = Perception(
        tracking=True, bbox=(560, 180, 640, 260), depth_m=None)
    c15h.tick(left_only, scan_with_rear(1.5, 1.0), t15b + 0.20)
    d = c15h.tick(left_only, scan_with_rear(1.5, 1.0), t15b + 0.50)
    assert d.motion.angular_mrad_s > 0 \
        and "single-left" in d.motion.reason, d.motion

    # 回退后双目需连续确认两帧：第一帧停住，第二帧才恢复逼近。
    restored = Perception(
        tracking=True, bbox=(560, 120, 640, 200), depth_m=None,
        tracking2=True,
        bbox2=(HALF + 620, 120, HALF + 700, 200))
    d = c15h.tick(restored, scan_with_rear(1.5, 1.0), t15b + 0.55)
    assert d.motion.linear_mm_s == 0 \
        and "recovery dual confirm 1/2" in d.motion.reason, d.motion
    d = c15h.tick(restored, scan_with_rear(1.5, 1.0), t15b + 0.60)
    assert c15h._recovery_started_at is None, d.motion
    assert "recovery dual confirm" not in d.motion.reason, d.motion

    # 21) 静止起步：先给 breakaway 冲量（破静摩擦），保持 hold 秒后
    #     回落到目标速度；停车后再次起步重新冲量
    c16, t16 = make(8000.0)
    c16.on_ground_result(True, t16)
    d = c16.tick(per(cx=AIM), scan_front(2.0), t16 + 0.05)
    assert d.motion.linear_mm_s == tcfg.linear_breakaway_mm_s, d.motion
    d = c16.tick(per(cx=AIM), scan_front(2.0), t16 + 0.10)
    assert d.motion.linear_mm_s == tcfg.linear_breakaway_mm_s, d.motion
    d = c16.tick(per(cx=AIM), scan_front(2.0), t16 + 0.50)
    # 高起步冲量结束后按减速度平滑回落，不瞬间从 480 跳到巡航240。
    assert (tcfg.max_linear_mm_s <= d.motion.linear_mm_s
            < tcfg.linear_breakaway_mm_s), d.motion
    c16.tick(None, None, t16 + 0.55)   # 无感知 → 停车，重置
    d = c16.tick(per(cx=AIM), scan_front(2.0), t16 + 0.60)
    assert d.motion.linear_mm_s == tcfg.linear_breakaway_mm_s, d.motion

    # 22) 雷达卡死检测：命令在走但扫描不动（静态相同扫描 ICP 零位移）
    #     → 连续窗口判定卡住 → 累计级别但不增加瞬时速度 → GIVE_UP
    #     注意 ICP 要求 ≥35 点，用密集全环扫描（scan_front 只有 20 点）
    def scan_ring(distance_m):
        # 距离加角度调制：完美对称圆环对 ICP 任意 yaw 等价，
        # 会误判为转动了；加调制后零位唯一。
        angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        distances = np.full_like(angles, distance_m) + 0.3 * np.sin(2 * angles)
        return scan(angles, distances)

    c17, t17 = make(9000.0)
    c17 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=False,
                                          use_taught_stop=False,
                                          stuck_window_s=0.2,
                                          stuck_escalate_windows=2,
                                          stuck_max_escalations=3,
                                          pulse_drive_enabled=False),
                            StereoConfig(), HALF)
    c17.on_ground_result(True, t17)
    stuck_cfg = PursuitConfig(instruction="测试",
                              use_visual_row_distance=False,
                              use_taught_stop=False,
                              stuck_window_s=0.2,
                              stuck_escalate_windows=2,
                              stuck_max_escalations=3)
    saw_stuck_level = False
    for _ in range(120):   # ~12s，足够 3 级提高 + 放弃
        t17 += 0.1
        d = c17.tick(per(cx=AIM, depth=2.0), scan_ring(2.0), t17)
        if d.finished:
            break
        if c17._stuck_level > 0:
            saw_stuck_level = True
        assert d.motion.linear_mm_s <= max(
            stuck_cfg.linear_breakaway_mm_s,
            stuck_cfg.visual_pulse_mm_s,
            stuck_cfg.creep_pulse_mm_s), d.motion
    assert saw_stuck_level, "应该累计过卡死级别"
    assert c17.state == State.GIVE_UP and d.finished and d.result == "give_up", \
        (c17.state, d)

    # 雷达缺帧/ICP 不收敛/厘米级噪声都不能把已积累的破静摩擦
    # 脉冲档位降回去；直行中单纯原地转向也不算前进。
    c17b, t17b = make(9500.0)
    c17b._stuck_level = 3
    ring = scan_ring(2.0)
    assert c17b._check_stalled(None, t17b) is None
    assert c17b._check_stalled(ring, t17b) is None

    class _FixedIcp:
        def __init__(self, result):
            self.result = result
        def match(self, _reference, _current):
            return self.result

    def check_match(distance, yaw=0.0, converged=True):
        c17b._stall_icp = _FixedIcp(MatchResult(
            Pose2D(x_m=distance, yaw_rad=yaw), 0.01, 100, converged, 0.8))
        return c17b._check_stalled(ring, c17b._stall_check_at + 0.01)

    assert check_match(0.0, converged=False) is None
    assert check_match(0.0) is True
    assert check_match(0.025) is None
    assert check_match(0.0, yaw=0.06) is None
    assert check_match(0.05) is False

    # 最新实车：雷达配准称“没动”时，两目方块却同时下移、变大。
    # 这证明至少视线正在逼近，必须撤销加长脉冲的判断。
    c17f, t17f = make(9550.0)
    p_before = Perception(
        tracking=True, bbox=(420, 220, 470, 270), depth_m=None,
        tracking2=True, bbox2=(1800, 170, 1850, 220))
    p_after = Perception(
        tracking=True, bbox=(420, 235, 476, 291), depth_m=None,
        tracking2=True, bbox2=(1800, 185, 1856, 241))
    assert c17f._check_stalled(ring, t17f, p_before) is None
    c17f._stall_icp = _FixedIcp(MatchResult(
        Pose2D(), 0.01, 100, True, 0.8))
    assert c17f._check_stalled(
        ring, c17f._stall_check_at + 0.01, p_after) is False

    # 2026-09-25 实车过冲前一帧：video0 实为物理右目，video2 实为
    # 物理左目。旧顺序把掩码算反，还差 54px，继续发了 1s 脉冲；
    # 正确拼接后双目都已入掩码，应立即停车，避免目标进入近端盲区。
    c17c = PursuitController(PursuitConfig(instruction="红色方块"),
                             StereoConfig(), HALF)
    raw_video0 = (288.9, 343.9, 462.3, 501.2)
    raw_video2 = (1827.5, 422.7, 1981.2, 590.4)
    assert max(c17c._taught_mask_distance(raw_video0),
               c17c._taught_mask_distance(raw_video2)) > \
        c17c.cfg.taught_early_stop_margin_px
    physical_left = (raw_video2[0] - HALF, raw_video2[1],
                     raw_video2[2] - HALF, raw_video2[3])
    physical_right = (raw_video0[0] + HALF, raw_video0[1],
                      raw_video0[2] + HALF, raw_video0[3])
    stopped = c17c._track_to_taught(physical_left, physical_right, 0)
    assert stopped.finished and stopped.result == "alignment_pending", stopped
    # 即使以后又误设左右顺序，也必须在下一次长脉冲前安全停下，
    # 不可把触发盲区兜底误报成“可抓取”。
    c17d = PursuitController(PursuitConfig(instruction="红色方块", recovery_rear_sensor_usable=True),
                             StereoConfig(), HALF)
    c17d.on_ground_result(True, 9600.0)
    blind = c17d.tick(Perception(
        tracking=True, bbox=raw_video0, depth_m=None,
        tracking2=True, bbox2=raw_video2), scan_front(0.88), 9600.1)
    assert not blind.finished and blind.motion.linear_mm_s == 0 \
        and "recovery visual check dual-near-blind" in blind.motion.reason, blind
    blocked = c17d.tick(Perception(
        tracking=True, bbox=raw_video0, depth_m=None,
        tracking2=True, bbox2=raw_video2),
        scan_with_rear(0.88, 0.09), 9600.4)
    assert blocked.finished and blocked.result == "give_up" \
        and blocked.motion.linear_mm_s == 0 \
        and "rear blocked" in blocked.detail, blocked
    c17e = PursuitController(PursuitConfig(instruction="红色方块", recovery_rear_sensor_usable=True),
                             StereoConfig(), HALF)
    c17e.on_ground_result(True, 9700.0)
    single = c17e.tick(Perception(
        tracking=True, bbox=(250.0, 520.0, 380.0, 680.0), depth_m=None),
        scan_front(0.86), 9700.1)
    assert not single.finished and single.motion.linear_mm_s == 0 \
        and "recovery visual check single-left" in single.motion.reason, single
    no_rear = c17e.tick(Perception(
        tracking=True, bbox=(250.0, 520.0, 380.0, 680.0), depth_m=None),
        scan_front(0.86), 9700.4)
    assert no_rear.finished and no_rear.result == "give_up" \
        and no_rear.motion.linear_mm_s == 0 \
        and "rear unknown/stale" in no_rear.detail, no_rear

    # 近端越界后，后方确实畅通才允许一次短回退；重新双目可见时
    # 停车交给抓取位姿复核，不立即再次前冲。
    c17h = PursuitController(PursuitConfig(instruction="红色方块", recovery_rear_sensor_usable=True),
                             StereoConfig(), HALF)
    c17h.on_ground_result(True, 9800.0)
    blind_per = Perception(
        tracking=True, bbox=raw_video0, depth_m=None,
        tracking2=True, bbox2=raw_video2, frame_id=10)
    d = c17h.tick(blind_per, scan_with_rear(0.88, 0.8), 9800.1)
    assert d.motion.linear_mm_s == 0 and not d.finished, d
    d = c17h.tick(blind_per, scan_with_rear(0.88, 0.8), 9800.4)
    assert d.motion.linear_mm_s == -c17h.cfg.recovery_reverse_mm_s \
        and "dual-near-blind" in d.motion.reason, d
    safely_visible = Perception(
        tracking=True, bbox=(430, 230, 540, 320), depth_m=None,
        tracking2=True, bbox2=(HALF + 400, 230, HALF + 510, 320),
        frame_id=11)
    d = c17h.tick(safely_visible, scan_with_rear(0.88, 0.8), 9800.45)
    assert d.motion.linear_mm_s == 0 \
        and "recovery dual confirm 1/2" in d.motion.reason, d
    d = c17h.tick(safely_visible, scan_with_rear(0.88, 0.8), 9800.50)
    assert d.motion.linear_mm_s == 0 \
        and "recovery dual confirm 1/2" in d.motion.reason, d
    safely_visible.frame_id = 12
    d = c17h.tick(safely_visible, scan_with_rear(0.88, 0.8), 9800.55)
    assert d.finished and d.result == "alignment_pending" \
        and d.motion.linear_mm_s == 0, d

    # 雷达扫描太旧与看不到后方都不可触发倒车；丢失时近期实际
    # 前进转向命令优先用于逆向回溯，过期后退回图像轨迹推断。
    c17i = PursuitController(PursuitConfig(instruction="红色方块", recovery_rear_sensor_usable=True),
                             StereoConfig(), HALF)
    c17i.on_ground_result(True, 9900.0)
    assert c17i._motion_rollback_sign(9900.3) == 0
    c17i._recent_motion.append((9900.1, 405, 320))
    assert c17i._motion_rollback_sign(9900.3) == -1
    assert c17i._motion_rollback_sign(9902.5) == 0
    c17i.tick(both_lost, scan_with_rear(0.9, 0.8), 9900.3)
    old_scan = scan_with_rear(0.9, 0.8)
    old_scan = LaserScan(old_scan.angles_rad, old_scan.distances_m, 9898.0)
    d = c17i.tick(both_lost, old_scan, 9900.6)
    assert d.motion.linear_mm_s == 0 \
        and "rear unknown/stale" in d.motion.reason, d

    # 后向 0.09m 固定近距回波：必须先有真实前进命令和双目接近趋势，
    # 才允许小步回退；两步之间停车等新帧，累计理论位移不超过 10cm。
    c17j = PursuitController(PursuitConfig(instruction="红色方块"),
                             StereoConfig(), HALF)
    c17j.on_ground_result(True, 10000.0)
    for frame_id, cy, now in (
            (1, 250.0, 10000.05),
            (2, 270.0, 10000.15),
            (3, 290.0, 10000.25)):
        p = Perception(
            tracking=True, bbox=(250, cy - 45, 350, cy + 45),
            depth_m=None, tracking2=True,
            bbox2=(HALF + 250, cy - 45, HALF + 350, cy + 45),
            frame_id=frame_id)
        c17j.tick(p, scan_with_rear(1.8, 0.09), now)
    assert c17j._recent_visual_approach_at == 10000.25
    assert any(v > 0 for _, v, _ in c17j._recent_motion)
    near = Perception(
        tracking=True, bbox=(240, 480, 360, 620), depth_m=None,
        tracking2=True,
        bbox2=(HALF + 240, 480, HALF + 360, 620), frame_id=4)
    d = c17j.tick(near, scan_with_rear(1.8, 0.09), 10000.35)
    assert d.motion.linear_mm_s == 0 and not d.finished, d
    d = c17j.tick(near, scan_with_rear(1.8, 0.09), 10000.45)
    assert d.motion.linear_mm_s == 0 and not d.finished, d
    d = c17j.tick(near, scan_with_rear(1.8, 0.09), 10000.75)
    assert d.motion.linear_mm_s == -405 and d.motion.angular_mrad_s == 0 \
        and "swept-path pulse 1/2" in d.motion.reason, d
    d = c17j.tick(near, scan_with_rear(1.8, 0.09), 10000.95)
    assert d.motion.linear_mm_s == -405, d
    d = c17j.tick(near, scan_with_rear(1.8, 0.09), 10001.18)
    assert d.motion.linear_mm_s == 0 \
        and "stop/observe" in d.motion.reason, d
    d = c17j.tick(near, scan_with_rear(1.8, 0.09), 10001.56)
    assert d.motion.linear_mm_s == 0 \
        and "waiting new frame" in d.motion.reason, d
    lost = Perception(tracking=False, bbox=None, depth_m=None, frame_id=5)
    d = c17j.tick(lost, scan_with_rear(1.8, 0.09), 10001.60)
    assert d.motion.linear_mm_s == -405 \
        and "swept-path pulse 2/2" in d.motion.reason, d
    assert c17j._recovery_swept_estimated_m <= 0.100001
    assert c17j._recovery_swept_pulses == 2
    d = c17j.tick(lost, scan_with_rear(1.8, 0.09), 10002.10)
    assert d.motion.linear_mm_s == 0, d
    d = c17j.tick(Perception(tracking=False, bbox=None, depth_m=None,
                             frame_id=6),
                  scan_with_rear(1.8, 0.09), 10002.55)
    assert d.finished and d.result == "give_up" \
        and d.motion.linear_mm_s == 0, d

    # 可疑的真实后障碍 (0.20m) 或过期扫描不能走短回退捷径。
    c17k = PursuitController(PursuitConfig(instruction="红色方块", recovery_rear_sensor_usable=True),
                             StereoConfig(), HALF)
    c17k.on_ground_result(True, 10100.0)
    c17k._recent_motion = [(10100.1, 405, 0)]
    c17k._recent_visual_approach_at = 10100.2
    c17k._recovery_from_blind = True
    c17k.tick(near, scan_with_rear(1.8, 0.2), 10100.25)
    d = c17k.tick(near, scan_with_rear(1.8, 0.2), 10100.55)
    assert d.finished and d.motion.linear_mm_s == 0, d

    # 上一帧接近近端、下一帧双目同时出画，也可在有近期前进证据时
    # 触发同一小步后退；若找回双目，应直接停车交接而非发第二步。
    c17l = PursuitController(PursuitConfig(instruction="红色方块"),
                             StereoConfig(), HALF)
    c17l.on_ground_result(True, 10200.0)
    c17l._recent_motion = [(10200.1, 405, 0)]
    c17l._recent_visual_approach_at = 10200.2
    c17l._last_near_blind_at = 10200.2
    lost_l = Perception(tracking=False, bbox=None, depth_m=None, frame_id=10)
    c17l.tick(lost_l, scan_with_rear(1.8, 0.09), 10200.25)
    d = c17l.tick(lost_l, scan_with_rear(1.8, 0.09), 10200.55)
    assert c17l._recovery_from_blind and d.motion.linear_mm_s == -405, d
    restored_l = Perception(
        tracking=True, bbox=(430, 230, 540, 320), depth_m=None,
        tracking2=True, bbox2=(HALF + 400, 230, HALF + 510, 320),
        frame_id=11)
    d = c17l.tick(restored_l, scan_with_rear(1.8, 0.09), 10200.65)
    assert d.motion.linear_mm_s == 0 and not d.finished, d
    restored_l.frame_id = 12
    d = c17l.tick(restored_l, scan_with_rear(1.8, 0.09), 10200.75)
    assert d.finished and d.result == "alignment_pending" \
        and c17l._recovery_swept_pulses == 0, d

    # 2026-09-25 最后一段原为 1.00s/405mm/s；掩码附近即使
    # stuck_level=2，也只能按 0.80s 上限发下一脉冲。
    c17g = PursuitController(PursuitConfig(instruction="红色方块"),
                             StereoConfig(), HALF)
    c17g._stuck_level = 2
    last_before_stop = Perception(
        tracking=True, bbox=(525.3, 259.9, 596.7, 332.4), depth_m=None,
        tracking2=True,
        bbox2=(1885.0, 204.3, 1954.6, 277.0))
    assert c17g._close_to_taught_mask(last_before_stop)
    c17g._visual_pulse_decision(
        Decision(Motion(200, 0, 400)), last_before_stop, 9800.0,
        near=False)
    assert c17g._visual_active_hold_s == 0.80
    c17g._creep_decision(
        Decision(Motion(200, 0, 400)), 0.90, 9900.0,
        last_before_stop)
    assert c17g._creep_active_hold_s == 0.80
    far_pair = Perception(
        tracking=True, bbox=(50, 10, 90, 50), depth_m=None,
        tracking2=True, bbox2=(1800, 10, 1840, 50))
    assert not c17g._close_to_taught_mask(far_pair)
    c17h = PursuitController(PursuitConfig(instruction="红色方块"),
                             StereoConfig(), HALF)
    c17h._stuck_level = 2
    c17h._visual_pulse_decision(
        Decision(Motion(200, 0, 400)), far_pair, 10000.0, near=False)
    assert c17h._visual_active_hold_s == 0.80  # lidar alone cannot escalate
    c17h._visual_feedback.level = 2
    c17h._reset_visual_pulse()
    c17h._visual_pulse_decision(
        Decision(Motion(200, 0, 400)), far_pair, 10000.0, near=False)
    assert c17h._visual_active_hold_s == 0.90
    c17h._visual_pulse_decision(
        Decision(Motion(200, 0, 400)), last_before_stop,
        10000.1, near=False)
    assert c17h._visual_active_hold_s == 0.85
    assert abs(c17h._visual_pulse_until - 10000.85) < 1e-6

    # 23) 横向偏差大 → 边前进边修正（不原地停）：前进速度按比例降低、
    #     转向保持（目标偏左就略微向左前进）；对准后恢复全速
    c18, t18 = make(10000.0)
    c18.on_ground_result(True, t18)
    d = c18.tick(per(cx=1000), scan_front(2.0), t18 + 0.05)
    assert d.motion.linear_mm_s > 0, d.motion   # 仍在前进（冲量起步）
    assert d.motion.angular_mrad_s != 0, d.motion
    d = c18.tick(per(cx=1000), scan_front(2.0), t18 + 0.50)
    assert "align" in d.motion.reason, d.motion
    assert 0 < d.motion.linear_mm_s < tcfg.linear_breakaway_mm_s, d.motion
    d = c18.tick(per(cx=AIM + 100), scan_front(2.0), t18 + 0.55)
    d = c18.tick(per(cx=AIM), scan_front(2.0), t18 + 0.60)
    assert d.motion.linear_mm_s > 0, d.motion

    # 24) SEARCHING 旋转卡死保护：车转不动（静态扫描 ICP 收敛 ~0）
    #     → 连续无进展窗口数达标后跳过本步 → 步尽 GIVE_UP（不再无限转）
    c19, t19 = make(11000.0)
    c19 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=False,
                                          use_taught_stop=False,
                                          search_stall_s=0.2,
                                          search_stall_giveup_windows=3),
                            StereoConfig(), HALF)
    c19.on_ground_result(True, t19)
    for _ in range(10):
        t19 += 0.2
        c19.tick(Perception(tracking=False, bbox=None, depth_m=None),
                 None, t19)
    assert c19.state == State.SEARCHING, c19.state
    d = None
    for _ in range(100):
        t19 += 0.1
        d = c19.tick(None, scan_ring(2.0), t19)
        if d.reground_now:
            c19.on_ground_result(False, t19)   # 模拟重新识别未找到
        if d.finished:
            break
    assert c19.state == State.GIVE_UP and d.finished and d.result == "give_up", \
        (c19.state, d)

    # 25) 任务结束后（GIVE_UP）网页重新识别找到目标 → 重启 TRACKING
    c20, t20 = make(12000.0)
    c20.on_ground_result(True, t20)
    d = c20.tick(per(depth=1.0), scan_front(0.05), t20 + 0.05)   # 硬停 → GIVE_UP
    assert c20.state == State.GIVE_UP, c20.state
    c20.on_ground_result(True, t20 + 0.1)
    assert c20.state == State.TRACKING, c20.state

    # 26) 丢失瞬间目标在右（最后 bbox 中心偏右 200px）→ 重搜序列以
    #     "最后看见目标的方位"为中心：base ≈ -atan(200/600)（右转），
    #     首步 = base + steps[0]
    c21, t21 = make(13000.0)
    c21.on_ground_result(True, t21)
    c21.tick(per(cx=AIM + 200, depth=1.0), scan_front(2.0), t21 + 0.05)
    for _ in range(10):
        t21 += 0.2
        c21.tick(Perception(tracking=False, bbox=None, depth_m=None),
                 None, t21)
        if c21.state == State.SEARCHING:
            break
    assert c21.state == State.SEARCHING, c21.state
    bearing = math.degrees(math.atan(200.0 / 600.0))
    assert abs(c21._search_base_deg - (-bearing)) < 0.5, c21._search_base_deg
    steps = PursuitConfig().search_rotation_steps_deg
    assert c21._current_step() == c21._search_base_deg + steps[0], \
        c21._current_step()

    # 27) 回瞄角钳位：目标极端偏右（err 3000px → 78.7°）→ base 钳到 -60°
    c22, t22 = make(14000.0)
    c22.on_ground_result(True, t22)
    c22.tick(Perception(tracking=True,
                        bbox=(AIM + 2960, 300, AIM + 3040, 400),
                        depth_m=1.0), scan_front(2.0), t22 + 0.05)
    for _ in range(10):
        t22 += 0.2
        c22.tick(Perception(tracking=False, bbox=None, depth_m=None),
                 None, t22)
        if c22.state == State.SEARCHING:
            break
    assert c22.state == State.SEARCHING, c22.state
    assert c22._search_base_deg == -60.0, c22._search_base_deg

    # 28) 搜索转满 → GIVE_UP 后首个 tick 自动触发一次 Qwen 复检
    #     （reground_now=True），之后不重复；复检找到 → 恢复 TRACKING
    c23, t23 = make(15000.0)
    c23.on_ground_result(False, t23)   # 初始未找到 → SEARCHING
    terminal_reground = False
    for _ in range(len(steps)):
        while True:
            t23 += 0.5
            d = c23.tick(None, None, t23)
            if d.reground_now or d.finished or c23.state == State.GIVE_UP:
                terminal_reground = terminal_reground or d.reground_now
                break
        if c23.state == State.GIVE_UP:
            break
        c23.on_ground_result(False, t23)   # 每一步都没找到
    assert c23.state == State.GIVE_UP, c23.state
    d = c23.tick(None, None, t23 + 0.05)
    assert d.finished and d.result == "give_up", d
    terminal_reground = terminal_reground or d.reground_now
    assert terminal_reground, d
    d = c23.tick(None, None, t23 + 0.10)
    assert not d.reground_now, d
    c23.on_ground_result(True, t23 + 0.15)
    assert c23.state == State.TRACKING, c23.state

    # 29) 视觉先行找回：旋转重搜中服务端追回连续确认 → 立即停车并
    #     触发 Qwen 复检，不必转完本步
    c24, t24 = make(16000.0)
    c24.on_ground_result(False, t24)   # 未找到 → SEARCHING
    d = None
    confirm_frames = c24.cfg.search_reacquire_confirm_frames
    for i in range(confirm_frames):
        t24 += 0.05
        d = c24.tick(Perception(tracking=True,
                                bbox=(AIM - 40, 300, AIM + 40, 400),
                                depth_m=None), None, t24)
        if i < confirm_frames - 1:
            assert not d.reground_now, (i, d)
    assert d.reground_now, d
    assert c24._awaiting_ground and not c24._rotating
    # 复检确认 → 恢复 TRACKING
    c24.on_ground_result(True, t24 + 0.05)
    assert c24.state == State.TRACKING, c24.state

    # 30) 雷达停车兜底（2026-08-17）：视觉锁漂/第二路丢失时行测距
    #     与示教停车都可能永远不触发——雷达 ≤ 0.2m+0.05 连续确认
    #     帧 → STOPPED（独立计数器，不受视觉停车确认干扰）。
    #     用行测距配置（目标行 y 300 远未到停车行，视觉不停车），
    #     只有雷达兜底能触发停车。
    c25 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=True,
                                          use_taught_stop=False),
                            StereoConfig(), HALF)
    t25 = 20000.0
    c25.on_ground_result(True, t25)
    d = None
    for i in range(3):
        t25 += 0.05
        d = c25.tick(per(cx=AIM, depth=None), scan_front(0.22), t25)
        if i < 2:
            assert not d.finished, (i, d)
    assert d.finished and d.result == "success", d
    assert c25.state == State.STOPPED, c25.state
    assert "雷达" in d.detail, d.detail
    # 雷达读数抖动（0.22 → 0.35 → 0.22 …）不会跨帧凑满确认
    # （2026-08-17 滞回：计数开始后 >0.30m 才重置，0.30 抖动不重置）
    c26 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=True,
                                          use_taught_stop=False),
                            StereoConfig(), HALF)
    t26 = 21000.0
    c26.on_ground_result(True, t26)
    for i in range(6):
        t26 += 0.05
        d = c26.tick(per(cx=AIM, depth=None),
                     scan_front(0.22 if i % 2 == 0 else 0.35), t26)
    assert not d.finished, d
    assert c26.state == State.TRACKING, c26.state

    # 31) 掩码区域停车 + 首帧即停：掩码外进入慢速步进；左右中心
    #     一旦都进入白区，首帧 STOPPED，不要求命中质心或连续 3 帧。
    c31 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=False,
                                          use_taught_stop=True),
                            StereoConfig(), HALF)
    t31 = 23000.0
    c31.on_ground_result(True, t31)
    cfg31 = PursuitConfig(instruction="测试", use_visual_row_distance=False,
                          use_taught_stop=True)
    lx, ly = cfg31.taught_left_cx_px, cfg31.taught_left_cy_px
    rx, ry = cfg31.taught_right_cx_px, cfg31.taught_right_cy_px

    def taught_per(l_cx, l_cy, r_cx, r_cy):
        return Perception(tracking=True,
                          bbox=(l_cx - 30, l_cy, l_cx + 30, l_cy + 40),
                          depth_m=None,
                          bbox2=(HALF + r_cx - 30, r_cy,
                                 HALF + r_cx + 30, r_cy + 40),
                          tracking2=True)

    # 预热：远离示教点（err≈405px > band 220）→ 全速
    for _ in range(15):
        t31 += 0.05
        c31.tick(taught_per(lx, 120, rx, 120), scan_front(1.5), t31)
    # 逐步接近（每帧 y 位移 <110px 防跳变保护误触）
    for step_y in (180, 280, 330):
        t31 += 0.05
        c31.tick(taught_per(lx, step_y, rx, step_y), scan_front(1.5), t31)
    # 两路在各自掩码上方的慢速带：短促前进，随后完全停稳测量。
    saw_pulse = saw_rest = False
    for _ in range(16):
        t31 += 0.05
        d = c31.tick(taught_per(lx, 330, rx, 220),
                     scan_front(1.5), t31)
        saw_pulse |= d.motion.linear_mm_s >= cfg31.min_linear_mm_s
        saw_rest |= d.motion.linear_mm_s == 0
    assert saw_pulse and saw_rest, d.motion
    # 进入掩码即停车，不再苛求越过质心；预测缓冲会更早一点停车。
    c31_stop = PursuitController(cfg31, StereoConfig(), HALF)
    c31_stop.on_ground_result(True, t31)
    t31 += 0.05
    d = c31_stop.tick(Perception(
        tracking=True, bbox=(470, 540, 530, 580), depth_m=None,
        bbox2=(HALF + 770, 420, HALF + 830, 460), tracking2=True),
        scan_front(1.5), t31)
    assert c31_stop.state == State.STOPPED, c31_stop.state
    assert d.finished and d.result == "alignment_pending", d

    # 32) 转向 EMA 平滑：目标越出中间 1/3 后仍不能猛甩，
    #     指令变化用 0.5/0.5 EMA 逐帧收敛。
    c32, t32 = make(24000.0)
    c32.on_ground_result(True, t32)
    d = c32.tick(per(cx=1000), scan_front(2.0), t32 + 0.05)
    assert d.motion.angular_mrad_s < 0, d.motion
    first_angular = d.motion.angular_mrad_s
    d = c32.tick(per(cx=1000), scan_front(2.0), t32 + 0.50)
    assert -140 < d.motion.angular_mrad_s < first_angular, d.motion
    # 逐帧回到中间 1/3，EMA 应平滑地回零。
    previous = d.motion.angular_mrad_s
    d = c32.tick(per(cx=900), scan_front(2.0), t32 + 0.55)
    assert previous < d.motion.angular_mrad_s < 0, d.motion
    previous = d.motion.angular_mrad_s
    d = c32.tick(per(cx=800), scan_front(2.0), t32 + 0.60)
    assert previous < d.motion.angular_mrad_s < 0, d.motion
    previous = d.motion.angular_mrad_s
    d = c32.tick(per(cx=700), scan_front(2.0), t32 + 0.65)
    assert previous < d.motion.angular_mrad_s <= 0, d.motion

    # 33) 搜索步超时（2026-08-17 实测：ICP yaw 振荡让 step 4 卡
    #     15s+，卡死检测被"伪进展"骗过）→ 每步超过时限直接跳过，
    #     不依赖 yaw 判定；静态扫描（ICP yaw 恒 0）下靠超时走完。
    c33 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=False,
                                          use_taught_stop=False,
                                          search_step_timeout_s=1.0),
                            StereoConfig(), HALF)
    t33 = 25000.0
    c33.on_ground_result(False, t33)   # 初始未找到 → SEARCHING
    steps = PursuitConfig().search_rotation_steps_deg
    d = None
    for _ in range(len(steps) + 3):
        while True:
            t33 += 0.5
            d = c33.tick(None, scan_ring(2.0), t33)
            if d.reground_now or d.finished:
                break
        if d.finished:
            break
        c33.on_ground_result(False, t33)   # 每步（含超时跳过）未找到
    assert c33.state == State.GIVE_UP, c33.state
    assert d.finished and d.result == "give_up", d

    # 34) 近距蠕动（2026-08-17 修复近距卡死）：<creep_zone 时脉冲
    #     冲量/休息交替（低速巡航 <160mm/s 推不动），脉冲速度带
    #     stuck 升级加成；0.25m 停车判定仍生效（不测试，见 #30）。
    c34 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=False,
                                          use_taught_stop=False,
                                          creep_zone_m=2.0,
                                          creep_pulse_s=0.3,
                                          creep_rest_s=0.5,
                                          stuck_escalate_windows=99),
                            StereoConfig(), HALF)
    t34 = 26000.0
    c34.on_ground_result(True, t34)
    per_c = per(cx=AIM, depth=None)
    d = c34.tick(per_c, scan_front(1.0), t34 + 0.05)
    assert "creep" in d.motion.reason and d.motion.linear_mm_s > 0, d.motion
    d = c34.tick(per_c, scan_front(1.0), t34 + 0.10)
    assert d.motion.linear_mm_s > 0, d.motion   # 脉冲期（0.3s）
    d = c34.tick(per_c, scan_front(1.0), t34 + 0.40)
    assert d.motion.linear_mm_s == 0 \
        and "rest" in d.motion.reason, d.motion   # 休息期
    d = c34.tick(per_c, scan_front(1.0), t34 + 0.90)
    assert d.motion.linear_mm_s > 0, d.motion    # 测试配置 0.3+0.5s 后下一脉冲
    # stuck 不再把已接近固件上限的速度继续抬高，而是延长脉冲时间。
    c34._stuck_level = 2
    d = c34.tick(per_c, scan_front(1.0), t34 + 0.95)
    cfg34 = PursuitConfig(instruction="测试", use_visual_row_distance=False,
                          use_taught_stop=False)
    assert d.motion.linear_mm_s == cfg34.creep_pulse_mm_s, d.motion

    # 34b) 雷达无位移等级必须只增加下一脉冲的持续时间，
    #      不再改变当前 330mm/s 脉冲速度。
    c34b, t34b = make(26500.0)
    c34b._stuck_level = 3
    base_decision = Decision(Motion(
        c34b.cfg.max_linear_mm_s, 0, c34b.cfg.ttl_near_ms, "adaptive test"))
    d = c34b._visual_pulse_decision(
        base_decision, per(cx=AIM), t34b + 0.05, near=True)
    expected_hold = c34b.cfg.visual_pulse_near_s
    assert abs(c34b._visual_active_hold_s - expected_hold) < 1e-6
    assert d.motion.linear_mm_s == c34b.cfg.visual_pulse_mm_s, d.motion
    assert f"hold={expected_hold:.2f}s visual_level=0" in d.motion.reason, d.motion

    # 35) 近距中间 1/3 外转向再减半，防止大目标框抖动。
    c35, t35 = make(27000.0)
    c35.on_ground_result(True, t35)
    d = c35.tick(per(cx=900, depth=0.4), scan_front(0.4), t35 + 0.05)
    assert -20 < d.motion.angular_mrad_s < 0, d.motion
    # 脉冲结束后的观测期必须完整急停，不能继续转向破坏画面稳定。
    d = c35.tick(per(cx=900, depth=0.4), scan_front(0.4), t35 + 0.90)
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s == 0, d.motion
    d = c35.tick(per(cx=900, depth=0.4), scan_front(0.4), t35 + 1.05)
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s == 0, d.motion
    # 下一脉冲再恢复小幅修正。
    d = c35.tick(per(cx=900, depth=0.4), scan_front(0.4), t35 + 1.35)
    assert d.motion.linear_mm_s > 0 and d.motion.angular_mrad_s < 0, d.motion

    # 36) BLE 未就绪跳过 stuck 卡死检测（2026-08-17 实测第八轮）：
    #     ESP32 恢复广播前的等待期（~30-40s）车不会动，若 stuck 检测
    #     照跑会在 36s 误判卡死 GIVE_UP（等 BLE 连上时任务已结束）。
    #     链路就绪后自动恢复检测。
    c36 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=False,
                                          use_taught_stop=False,
                                          stuck_window_s=0.2,
                                          stuck_escalate_windows=2,
                                          stuck_max_escalations=2,
                                          pulse_drive_enabled=False),
                            StereoConfig(), HALF)
    t36 = 28000.0
    c36.on_ground_result(True, t36)
    c36.set_link_ready(False)
    # 静态扫描 + 前进命令：链路未就绪 → 长时间不动也不 GIVE_UP
    d = None
    for _ in range(60):   # 12s >> stuck 升级链触发时长
        t36 += 0.2
        d = c36.tick(per(cx=AIM, depth=2.0), scan_ring(2.0), t36)
    assert c36.state == State.TRACKING, c36.state
    # 链路未就绪时搜索旋转也等待（不假转完白费 reground）
    c36.on_ground_result(False, t36)   # 目标丢失 → SEARCHING
    d = c36.tick(None, scan_ring(2.0), t36 + 0.1)
    assert c36.state == State.SEARCHING and "motion link" in d.motion.reason, \
        d.motion
    assert not d.reground_now, d
    # 链路就绪 → 恢复 TRACKING，stuck 检测恢复，很快 GIVE_UP
    c36.on_ground_result(True, t36)
    c36.set_link_ready(True)
    for _ in range(60):
        t36 += 0.2
        d = c36.tick(per(cx=AIM, depth=2.0), scan_ring(2.0), t36)
        if c36.state == State.GIVE_UP:
            break
    assert c36.state == State.GIVE_UP, c36.state
    assert d.finished and d.result == "give_up", d

    # dry-run 中运动命令只用于观察决策，车辆实际没有移动；静态雷达
    # 不能据此判定卡死，否则长时间视觉测试会被误杀为 GIVE_UP。
    c36_dry = PursuitController(PursuitConfig(
        instruction="测试", use_visual_row_distance=False,
        use_taught_stop=False, stuck_window_s=0.2,
        stuck_escalate_windows=2, stuck_max_escalations=2),
        StereoConfig(), HALF, simulate_motion=True)
    t36_dry = 28500.0
    c36_dry.on_ground_result(True, t36_dry)
    for _ in range(80):
        t36_dry += 0.2
        d = c36_dry.tick(per(cx=AIM, depth=2.0),
                         scan_ring(2.0), t36_dry)
    assert c36_dry.state == State.TRACKING, c36_dry.state
    assert not d.finished, d

    # 双目被动重识别在 GIVE_UP 后连续确认 3 帧 → 自动恢复；单帧不能
    # 唤醒，防止陈旧框误启动。
    c36_reid, t36_reid = make(28700.0)
    c36_reid.state = State.GIVE_UP
    reid_per = Perception(
        tracking=True, bbox=(AIM - 40, 300, AIM + 40, 400),
        depth_m=2.0, tracking2=True,
        bbox2=(AIM - 40, 302, AIM + 40, 402),
        reidentified=True)
    for i in range(2):
        d = c36_reid.tick(reid_per, scan_ring(2.0),
                          t36_reid + 0.1 * (i + 1))
        assert c36_reid.state == State.GIVE_UP, c36_reid.state
    d = c36_reid.tick(reid_per, scan_ring(2.0), t36_reid + 0.3)
    assert c36_reid.state == State.TRACKING, c36_reid.state
    assert not d.finished, d

    # SEARCHING 中可信双目重识别只需连续 2 帧即可直接恢复，不再额外
    # 调一次耗时 Qwen；普通 tracking 没有 reidentified 标志不能触发。
    c36_search_reid, t36_search_reid = make(28710.0)
    c36_search_reid.on_ground_result(False, t36_search_reid)
    d = c36_search_reid.tick(
        reid_per, scan_ring(2.0), t36_search_reid + 0.1)
    assert c36_search_reid.state == State.SEARCHING
    d = c36_search_reid.tick(
        reid_per, scan_ring(2.0), t36_search_reid + 0.2)
    assert c36_search_reid.state == State.TRACKING
    assert not d.reground_now, d

    # 37) 蠕动末段小步长 + 过冲后退（2026-08-17 用户要求"逐帧小步长
    #     前后移动到位"）：<0.35m 脉冲缩短到 0.2s（~4cm 步长）；
    #     过冲（雷达 <0.18m 且停车判定未触发）→ 后退脉冲修正
    c37 = PursuitController(PursuitConfig(instruction="测试",
                                          use_visual_row_distance=False,
                                          use_taught_stop=False,
                                          creep_zone_m=2.0,
                                          creep_pulse_s=0.3,
                                          creep_rest_s=0.5,
                                          stuck_escalate_windows=99),
                            StereoConfig(), HALF)
    t37 = 29000.0
    c37.on_ground_result(True, t37)
    per_c = per(cx=AIM, depth=None)
    d = c37.tick(per_c, scan_front(0.30), t37 + 0.05)
    assert d.motion.linear_mm_s > 0 and "pulse" in d.motion.reason, d.motion
    # 末段基础脉冲 0.80s：t+0.90 已结束 → 休息
    d = c37.tick(per_c, scan_front(0.30), t37 + 0.90)
    assert d.motion.linear_mm_s == 0, d.motion
    # 过冲后退（直接单元调用——0.15m 会先进入雷达停车确认期，
    # 蠕动被跳过，后退是停车判定失效时的保险分支）
    d = c37._creep_decision(Decision(Motion(0, 0, 400)), 0.15, t37 + 1.40)
    assert d.motion.linear_mm_s == -min(550, c37.cfg.creep_pulse_mm_s) \
        and "back" in d.motion.reason, d.motion
    d = c37._creep_decision(Decision(Motion(0, 0, 400)), 0.15, t37 + 1.55)
    assert d.motion.linear_mm_s == 0, d.motion   # 后退结束 → 休息

    # 38) 感知年龄门控：控制循环即使仍持有 tracking=True 的最后一框，
    #     超过 0.45s 也必须停车，不能等旧的 5s WS watchdog。
    c38, t38 = make(30000.0)
    c38.on_ground_result(True, t38)
    fresh = Perception(tracking=True,
                       bbox=(AIM - 40, 300, AIM + 40, 400),
                       depth_m=None, received_at=t38 + 0.05,
                       frame_interval_s=0.05, frame_id=1)
    d = c38.tick(fresh, scan_front(2.0), t38 + 0.05)
    assert d.motion.linear_mm_s > 0, d.motion
    d = c38.tick(fresh, scan_front(2.0), t38 + 0.60)
    assert d.motion.linear_mm_s == 0 \
        and "perception stale" in d.motion.reason, d.motion

    # 39) 极低感知帧率：45mm/帧上限若低于电机有效连续速度，自动
    #     改成脉冲-休息，而非输出一个硬件无法执行的 90mm/s。
    c39, t39 = make(31000.0)
    c39.on_ground_result(True, t39)
    low_fps = Perception(tracking=True,
                         bbox=(AIM - 40, 300, AIM + 40, 400),
                         depth_m=None, received_at=t39 + 0.05,
                         frame_interval_s=0.50, frame_id=1)
    d = c39.tick(low_fps, scan_front(2.0), t39 + 0.05)
    assert d.motion.linear_mm_s >= c39.cfg.min_linear_mm_s \
        and "visual servo pulse" in d.motion.reason, d.motion
    low_fps2 = Perception(tracking=True, bbox=low_fps.bbox, depth_m=None,
                          received_at=t39 + 0.90,
                          frame_interval_s=0.50, frame_id=2)
    d = c39.tick(low_fps2, scan_front(2.0), t39 + 0.90)
    assert d.motion.linear_mm_s == 0 \
        and "rest" in d.motion.reason, d.motion

    # 40) 丢框必须压过雷达蠕动：旧代码只看 lidar<1m，会把
    # “target lost”的零速决定重新覆盖成前进脉冲，导致越丢越冲。
    c40 = PursuitController(PursuitConfig(
        instruction="测试", use_visual_row_distance=False,
        use_taught_stop=False, pulse_drive_enabled=True),
        StereoConfig(), HALF)
    t40 = 32000.0
    c40.on_ground_result(True, t40)
    p40 = Perception(tracking=True,
                     bbox=(AIM - 40, 300, AIM + 40, 400),
                     depth_m=None, received_at=t40 + 0.05,
                     frame_interval_s=0.05, frame_id=1)
    d = c40.tick(p40, scan_front(0.80), t40 + 0.05)
    assert d.motion.linear_mm_s > 0 and "creep pulse" in d.motion.reason, d.motion
    d = c40.tick(Perception(tracking=False, bbox=None, depth_m=None,
                            received_at=t40 + 0.10, frame_id=2),
                 scan_front(0.80), t40 + 0.10)
    assert d.motion.linear_mm_s == 0 and d.motion.angular_mrad_s == 0, d.motion
    assert "settle/reacquire" in d.motion.reason, d.motion
    assert c40._creep_pulse_until is None

    # 卡住后提高的是脉冲时长，不是把 520 继续堆到固件上限；急停仍
    # 把单步限制在 visual_pulse_max_s 内。
    c40._stuck_level = 3  # lidar escalation must NOT lengthen visual pulses
    c40._reset_visual_pulse()
    base = Decision(Motion(200, 0, 400, "test"))
    pulse_per = Perception(tracking=True, bbox=p40.bbox, depth_m=None)
    d = c40._visual_pulse_decision(
        base, pulse_per, t40 + 1.0, near=True)
    expected_s = c40.cfg.visual_pulse_near_s
    assert d.motion.linear_mm_s == c40.cfg.visual_pulse_mm_s, d.motion
    assert abs(c40._visual_pulse_until - (t40 + 1.0) - expected_s) < 1e-6

    print("ALL_LOGIC_TESTS_PASSED")


if __name__ == "__main__":
    main()
