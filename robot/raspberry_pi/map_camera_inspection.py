"""Localize in a saved map, navigate to useful views, and take photographs.

The mission is intentionally read-only with respect to the map.  It stops on
uncertain localization, blocked paths, camera errors, or link errors.
"""

import argparse
from collections import deque
import csv
from dataclasses import replace
import glob
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageStat
from PIL import ImageFilter

from raspberry_pi.config import LidarConfig, SerialConfig
from raspberry_pi.esp32 import Esp32Client, LinkError
from raspberry_pi.known_map_navigation import (
    KnownOccupancyMap,
    ScanMapLocalizer,
    consistent_localization,
    order_targets,
)
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.mapping import (
    AdaptiveSpeedConfig,
    AdaptiveSpeedController,
    MotionCommand,
    Pose2D,
    SlamUpdate,
)
from raspberry_pi.mapping.geometry import normalize_angle
from raspberry_pi.mapping.scan_filter import sector_min_distance


def _panorama_result_message(result):
    if result.get("motion_complete", False):
        status = "开阔点全景巡检完成"
        suffix = ""
    else:
        status = "开阔点全景巡检已停车，任务未完成"
        reason = result.get("finish_reason", "unknown")
        description = {
            "odometry_unavailable_after_stopped_recovery": "雷达帧间里程计停车恢复失败",
            "no_safe_local_progress": "局部安全恢复未产生进展",
            "no_translation_progress": "局部恢复长时间没有位置进展（仅转向不算脱困）",
            "local_escape_turn_budget_exhausted": "前方仍受阻，原地小步脱困次数已达上限",
            "map_guard_no_local_progress": "旧地图保护下限次重新选向仍未取得进展，未覆盖区域已保留",
            "actuator_no_response": "有限起步和换向重试后仍无可信前进位移",
            "no_safe_exit": "持续没有雷达与只读地图共同允许的安全出口",
            "lidar_data_unavailable": "无法获得及时有效的雷达扫描",
            "requested_stop": "收到停止请求",
        }.get(reason, reason)
        suffix = " 退出原因：{}（{}）。".format(description, reason)
    return "{}：站点={}，照片={}，覆盖率={:.1f}%。{}".format(
        status, result["station_count"], result["photo_count"],
        result["coverage"]["coverage_ratio"] * 100.0, suffix,
    )


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="在 ble_scene_17 已知地图中定位、巡航并拍摄墙面和黑色区域"
    )
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--map", default="maps/ble_scene_17.pgm")
    parser.add_argument("--map-yaml", default=None)
    parser.add_argument(
        "--semantic-map",
        default=None,
        help="永久语义旁车文件；默认 maps/ble_scene_17_semantics.json",
    )
    parser.add_argument(
        "--ignore-verified-semantic-views",
        action="store_true",
        help="只用永久占据地图生成机位，不复用历史照片中的导航位姿",
    )
    parser.add_argument(
        "--skip-classified-semantic-components",
        action="store_true",
        help="跳过已有可信语义标签的内部黑色连通体，把本轮资源留给未识别区域",
    )
    parser.add_argument(
        "--online-qwen-url",
        default="",
        help="每次拍照后调用的3090 /landmarks接口；空字符串表示关闭",
    )
    parser.add_argument("--online-qwen-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--online-qwen-min-confidence", type=float, default=0.80)
    parser.add_argument("--online-qwen-confirmations", type=int, default=2)
    parser.add_argument("--online-qwen-max-component-attempts", type=int, default=3)
    parser.add_argument(
        "--defer-online-qwen",
        action="store_true",
        help="先完成全部拍摄并停车，再按照片顺序统一调用在线Qwen标注",
    )
    parser.add_argument(
        "--open-space-panorama",
        action="store_true",
        help=(
            "不生成固定拍摄点或全局路线；使用低频地图纠错、局部雷达"
            "方向选择和开阔位置原地全景拍摄"
        ),
    )
    parser.add_argument("--panorama-min-station-spacing", type=float, default=1.00)
    parser.add_argument("--panorama-ideal-station-spacing", type=float, default=1.60)
    parser.add_argument("--panorama-max-station-spacing", type=float, default=2.50)
    parser.add_argument("--panorama-min-stations", type=int, default=3)
    parser.add_argument("--panorama-max-stations", type=int, default=16)
    parser.add_argument("--panorama-angle-step-deg", type=float, default=60.0)
    parser.add_argument("--panorama-camera-fov-deg", type=float, default=82.0)
    parser.add_argument("--panorama-camera-range", type=float, default=4.0)
    parser.add_argument("--panorama-coverage-target", type=float, default=0.90)
    parser.add_argument("--panorama-open-min-clearance", type=float, default=0.48)
    parser.add_argument(
        "--panorama-open-direction-clearance", type=float, default=0.72
    )
    parser.add_argument(
        "--panorama-open-required-directions", type=int, default=3
    )
    parser.add_argument(
        "--panorama-map-correction-distance", type=float, default=1.20
    )
    parser.add_argument(
        "--panorama-map-correction-gain", type=float, default=0.20
    )
    parser.add_argument(
        "--panorama-map-correction-confirmations", type=int, default=2
    )
    parser.add_argument(
        "--panorama-map-correction-radius", type=float, default=0.65
    )
    parser.add_argument("--panorama-drive-speed", type=int, default=380)
    parser.add_argument("--panorama-slow-speed", type=int, default=240)
    parser.add_argument("--panorama-turn-speed", type=int, default=2600)
    parser.add_argument("--panorama-turn-max-speed", type=int, default=3000)
    parser.add_argument("--panorama-startup-window-seconds", type=float, default=.70)
    parser.add_argument("--panorama-progress-window-seconds", type=float, default=1.80)
    parser.add_argument("--panorama-max-scan-age-seconds", type=float, default=.50)
    parser.add_argument("--panorama-body-length", type=float, default=.24,
                        help="实车最大外廓前后长（米，包含突出部件）")
    parser.add_argument("--panorama-body-width", type=float, default=.22,
                        help="实车最大外廓左右宽（米，包含突出部件）")
    parser.add_argument("--panorama-footprint-margin", type=float, default=.05,
                        help="外廓测量与定位余量（米，至少0.04；不计入车体回波过滤）")
    parser.add_argument(
        "--panorama-status-interval-seconds", type=float, default=0.20
    )
    parser.add_argument(
        "--panorama-self-return-radius", type=float, default=0.0,
        help="独立的已确认车体回波遮罩半径（m，最多0.20）；0关闭，须按安装实测确认，不能按防撞余量扩大",
    )
    parser.add_argument(
        "--panorama-self-return-half-width", type=float, default=0.125,
        help="独立的已确认车体回波遮罩半宽（m，最多0.14）；仅过滤后半平面与半径/半宽交集，不由近似车身尺寸裁剪",
    )
    parser.add_argument(
        "--information-view-distance",
        type=float,
        default=0.0,
        help="优先从该最远距离拍摄同一地图成分；0关闭远景信息视点",
    )
    parser.add_argument("--output", default="inspections/ble_scene_17")
    parser.add_argument("--link", choices=("ble", "usb", "uart", "auto"), default="ble")
    parser.add_argument("--ble-name", default="ESP32-Robot-Car")
    parser.add_argument("--ble-address", default=None)
    parser.add_argument("--ble-connect-timeout", type=float, default=6.0)
    parser.add_argument("--ble-operation-timeout", type=float, default=0.8)
    parser.add_argument("--esp-port", default="/dev/serial0")
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--lidar-baud", type=int, default=230400)
    parser.add_argument("--counterclockwise", action="store_true")
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--no-motor-control", action="store_true")
    parser.add_argument("--lidar-offset-x", type=float, default=0.14)
    parser.add_argument("--lidar-offset-y", type=float, default=0.0)
    parser.add_argument("--clearance", type=float, default=0.30)
    parser.add_argument("--front-left-gain", type=float, default=1.15)
    parser.add_argument("--front-right-gain", type=float, default=1.00)
    parser.add_argument("--rear-left-gain", type=float, default=0.90)
    parser.add_argument("--rear-right-gain", type=float, default=1.00)
    parser.add_argument("--speed", type=int, default=300)
    parser.add_argument("--max-drive-speed", type=int, default=460)
    parser.add_argument("--drive-speed-step", type=int, default=40)
    parser.add_argument(
        "--start-kick-speed",
        type=int,
        default=400,
        help="Temporary straight-drive floor until lidar confirms real movement",
    )
    parser.add_argument("--turn-speed", type=int, default=2000)
    parser.add_argument("--slow-speed", type=int, default=280)
    parser.add_argument(
        "--navigation-fine-control-distance",
        type=float,
        default=0.85,
        help="进入最终拍摄位该距离后强制使用慢速和精细转向",
    )
    parser.add_argument(
        "--navigation-fine-turn-speed",
        type=int,
        default=1100,
        help="最终拍摄朝向的转速上限；不会使用自适应堵转加速",
    )
    parser.add_argument(
        "--navigation-fine-turn-min-speed",
        type=int,
        default=900,
        help="最终拍摄短脉冲的最低有效转速；必须高于底盘静摩擦死区",
    )
    parser.add_argument(
        "--navigation-precision-turn-ttl-ms",
        type=int,
        default=200,
        help="最终拍摄转向单脉冲时长上限；每个脉冲后必须重新读取雷达",
    )
    parser.add_argument(
        "--navigation-heading-seed-frames",
        type=int,
        default=3,
        help="进入最终拍摄朝向前停车平均的定位航向帧数",
    )
    parser.add_argument("--command-ttl-ms", type=int, default=700)
    parser.add_argument("--command-min-interval-ms", type=int, default=450)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.18)
    parser.add_argument(
        "--nearby-capture-distance",
        type=float,
        default=0.45,
        help="受阻时距计划拍摄位不超过该距离则停车转向并就近拍摄",
    )
    parser.add_argument("--heading-tolerance-deg", type=float, default=7.0)
    parser.add_argument("--stop-distance", type=float, default=0.42)
    parser.add_argument(
        "--emergency-front-distance",
        type=float,
        default=0.30,
        help="前方小于该距离时，停车并尝试经过双重确认的短距离倒车",
    )
    parser.add_argument(
        "--collision-reverse-distance",
        type=float,
        default=0.10,
        help="紧急脱离墙面时允许后退的总距离（米）",
    )
    parser.add_argument(
        "--collision-reverse-speed",
        type=int,
        default=180,
        help="紧急短距离后退速度（毫米/秒）",
    )
    parser.add_argument(
        "--collision-reverse-rear-clearance",
        type=float,
        default=0.55,
        help="允许紧急后退所需的雷达后方最小净空（米）",
    )
    parser.add_argument(
        "--collision-reverse-max-steps",
        type=int,
        default=4,
        help="紧急后退最多分成多少个停车复查脉冲",
    )
    parser.add_argument(
        "--collision-reverse-map-fallback",
        action="store_true",
        help=(
            "后方雷达数据不足时，仅在永久地图后退走廊明确安全的情况下"
            "允许低速短退"
        ),
    )
    parser.add_argument(
        "--collision-reverse-map-fallback-max-distance",
        type=float,
        default=0.08,
        help="缺少后方雷达时地图兜底后退的硬距离上限（米）",
    )
    parser.add_argument(
        "--localization-timeout-seconds", type=float, default=120.0
    )
    parser.add_argument(
        "--localization-window", type=int, default=5,
        help="定位接受率的滑动统计窗口，不代表原地旋转圈数",
    )
    parser.add_argument(
        "--localization-min-accepted", type=int, default=3
    )
    parser.add_argument(
        "--localization-acceptance-ratio", type=float, default=0.60
    )
    parser.add_argument("--max-localization-failures", type=int, default=5)
    parser.add_argument(
        "--relaxed-tracking-rmse", type=float, default=0.28,
        help="巡检途中允许继续关联照片的放宽定位RMSE；不用于初始定位",
    )
    parser.add_argument(
        "--relaxed-tracking-inlier-ratio", type=float, default=0.28,
        help="巡检途中允许继续关联照片的放宽定位内点率；不用于初始定位",
    )
    parser.add_argument("--localization-min-margin", type=float, default=0.025)
    parser.add_argument("--localization-max-rmse", type=float, default=0.12)
    parser.add_argument("--localization-min-inlier-ratio", type=float, default=0.60)
    parser.add_argument("--localization-min-range-agreement", type=float, default=0.55)
    parser.add_argument("--localization-max-range-error", type=float, default=0.24)
    parser.add_argument("--localization-min-range-coverage", type=float, default=0.40)
    parser.add_argument("--localization-fast-accept-margin", type=float, default=0.15)
    parser.add_argument("--localization-fast-min-inlier-ratio", type=float, default=0.70)
    parser.add_argument("--localization-fast-min-range-agreement", type=float, default=0.75)
    parser.add_argument("--localization-fast-max-range-error", type=float, default=0.20)
    parser.add_argument(
        "--localization-disambiguation-distance", type=float, default=0.26,
        help="Maximum safe translation used to separate symmetric radar pose candidates",
    )
    parser.add_argument("--relocalization-distance", type=float, default=0.12)
    parser.add_argument("--relocalization-speed", type=int, default=300)
    parser.add_argument("--relocalization-max-speed", type=int, default=460)
    parser.add_argument("--relocalization-speed-step", type=int, default=50)
    parser.add_argument("--relocalization-turn-speed", type=int, default=1500)
    parser.add_argument("--relocalization-max-turn-speed", type=int, default=2600)
    parser.add_argument("--relocalization-turn-step", type=int, default=300)
    parser.add_argument(
        "--relocalization-turn-stall-at-max-frames",
        type=int,
        default=2,
        help="最大自适应转向输出下连续多少个雷达静止帧后进入脱困",
    )
    parser.add_argument(
        "--relocalization-turn-escape-distance",
        type=float,
        default=0.05,
        help="原地转向堵转后的安全前冲目标距离（米）",
    )
    parser.add_argument(
        "--relocalization-turn-escape-max-steps",
        type=int,
        default=4,
        help="单次转向脱困最多使用的雷达运动步数",
    )
    parser.add_argument(
        "--relocalization-turn-escape-max-attempts",
        type=int,
        default=2,
        help="一次主动换位允许的转向脱困次数",
    )
    parser.add_argument(
        "--relocalization-min-scan-change",
        type=float,
        default=0.025,
        help="短动作前后雷达距离轮廓的最小中位变化，低于此值视为没有移动",
    )
    parser.add_argument(
        "--localization-required-moves", type=int, default=2,
        help="初始定位确认前必须完成并通过跨位置校验的实际位移次数",
    )
    parser.add_argument(
        "--localization-motion-min-distance", type=float, default=0.07,
        help="雷达帧间位移达到此值才算一次有效换位",
    )
    parser.add_argument(
        "--localization-motion-min-yaw-deg", type=float, default=8.0,
        help="A rotation this large also counts as an observable viewpoint change",
    )
    parser.add_argument(
        "--localization-motion-position-tolerance", type=float, default=0.24,
        help="地图位姿变化与雷达帧间位移允许的平移误差",
    )
    parser.add_argument(
        "--localization-motion-yaw-tolerance-deg", type=float, default=16.0,
        help="地图位姿变化与雷达帧间位移允许的航向误差",
    )
    parser.add_argument(
        "--temporary-obstacle-radius", type=float, default=0.12
    )
    parser.add_argument(
        "--temporary-obstacle-confirm-distance", type=float, default=0.18
    )
    parser.add_argument("--temporary-obstacle-min-hits", type=int, default=2)
    parser.add_argument("--standoff", type=float, default=0.65)
    parser.add_argument("--max-targets", type=int, default=32)
    parser.add_argument(
        "--min-target-spacing",
        type=float,
        default=0.0,
        help=(
            "拍摄视点之间的最小间距（米）；0 保留地图生成的全部关键点"
        ),
    )
    parser.add_argument(
        "--target-turn-cost-m-per-rad", type=float, default=0.35,
        help="目标排序中每弧度最终相机转向折算的路径代价（米）",
    )
    parser.add_argument(
        "--target-retry-min-displacement", type=float, default=0.25,
        help=(
            "受阻或朝向超时的目标再次参与最近邻排序前，车辆至少需"
            "离开失败位置的距离（米）；有其他目标时避免原地耗尽重试"
        ),
    )
    parser.add_argument(
        "--max-target-navigation-attempts",
        type=int,
        default=1,
        help="单个拍摄点遇到临时阻挡时最多尝试次数；1 表示不排队重试",
    )
    parser.add_argument(
        "--navigation-turn-stall-at-max-frames",
        type=int,
        default=8,
        help="巡检转向达到最大输出后连续多少帧无航向变化才触发脱困",
    )
    parser.add_argument(
        "--navigation-linear-stall-at-max-frames",
        type=int,
        default=6,
        help=(
            "巡检前进达到最大输出后连续多少个可信雷达帧无平移才"
            "停车并按雷达净空重新放宽启动脉冲；不会据此判定撞墙"
        ),
    )
    parser.add_argument(
        "--navigation-pure-turn-timeout-seconds",
        type=float,
        default=8.0,
        help="连续原地转向超过该时间时停止；已到拍摄位则就近拍照",
    )
    parser.add_argument(
        "--navigation-timeout-capture-max-heading-error-deg",
        type=float,
        default=25.0,
        help="最终转向超时后仍允许拍照的最大朝向误差；更大则稍后重试",
    )
    parser.add_argument(
        "--navigation-turn-refresh-window-seconds",
        type=float,
        default=1.0,
        help="纯转向命令允许后台连续保活的最大窗口；之后仍由TTL停车",
    )
    parser.add_argument(
        "--navigation-turn-escape-max-attempts",
        type=int,
        default=1,
        help="每个拍摄目标最多执行几次安全短前移转向脱困",
    )
    parser.add_argument(
        "--navigation-pose-stationary-yaw-limit-deg",
        type=float,
        default=18.0,
        help="停车帧允许的最大航向变化；超过时保留上一可信位姿",
    )
    parser.add_argument(
        "--navigation-pose-stationary-translation-limit",
        type=float,
        default=0.15,
        help="停车帧允许的最大位置变化（米）",
    )
    parser.add_argument(
        "--navigation-pose-direction-slack-deg",
        type=float,
        default=12.0,
        help="转向定位与指令方向相反时允许的角度噪声",
    )
    parser.add_argument(
        "--navigation-pose-max-correction-holds",
        type=int,
        default=8,
        help="连续异常定位最多停车保留可信位姿的帧数",
    )
    parser.add_argument(
        "--navigation-map-correction-interval-seconds",
        type=float,
        default=8.0,
        help="连续雷达里程计运行多久后才核验一次地图定位",
    )
    parser.add_argument(
        "--navigation-map-correction-distance",
        type=float,
        default=0.80,
        help="累计行驶多远后才核验一次地图定位（与时间条件任一满足）",
    )
    parser.add_argument(
        "--navigation-map-correction-gain",
        type=float,
        default=0.35,
        help="周期地图校准的融合比例，避免位姿瞬间跳变",
    )
    parser.add_argument(
        "--navigation-recovery-min-inlier-ratio",
        type=float,
        default=0.42,
        help="纠错重锚候选重新接管导航所需的最低内点率",
    )
    parser.add_argument(
        "--navigation-recovery-max-rmse",
        type=float,
        default=0.12,
        help="纠错重锚候选重新接管导航所允许的最大RMSE（米）",
    )
    parser.add_argument(
        "--navigation-recovery-confirmation-frames",
        type=int,
        default=2,
        help="纠错后至少连续多少个一致候选才能恢复导航",
    )
    parser.add_argument(
        "--alternate-view-max-shift",
        type=float,
        default=1.0,
        help="原拍摄位不可达时替代机位距原机位的最大距离（米）",
    )
    parser.add_argument(
        "--views-per-wall", type=int, default=12,
        help="每段外墙/大组件最多生成的拍摄位 (2026-08-16: 16 个太少, 6->12)",
    )
    parser.add_argument(
        "--views-per-region", type=int, default=8,
        help="每个内部黑色区域组件最多生成的拍摄位 (4->8)",
    )
    parser.add_argument(
        "--photos-per-target", type=int, default=2,
        help="Photo groups captured at each stopped annotation point",
    )
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--max-runtime-min", type=float, default=20.0)
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "rpicam", "libcamera", "opencv"),
        default="auto",
    )
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument(
        "--dual-usb-cameras",
        action="store_true",
        help="同时使用按USB物理端口稳定识别的两路UVC摄像头",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-timeout", type=float, default=12.0)
    parser.add_argument(
        "--free-cameras",
        action="store_true",
        help="摄像头被其他进程占用时先杀掉占用进程(fuser -k)再重试一次",
    )
    parser.add_argument(
        "--camera-test-only",
        action="store_true",
        help="只测试相机并保存一张图片，不连接ESP32/雷达、不允许运动",
    )
    parser.add_argument(
        "--camera-yaw-offset-deg",
        type=float,
        default=0.0,
        help="摄像头光轴相对车头的角度，向左为正；朝前安装填 0",
    )
    parser.add_argument(
        "--initial-pose",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "YAW_DEG"),
        default=None,
        help="可选定位提示，不是直接采用的真值；程序仍会与地图核验",
    )
    parser.add_argument(
        "--ignore-trajectory-pose",
        action="store_true",
        help="不采用同名 trajectory.csv 的起点位姿作为初始搜索提示",
    )
    return parser


class CameraCapture:
    def __init__(
        self, backend, device, width, height, timeout_s,
        dual_usb_cameras=False, free_cameras=False,
    ):
        self.backend = backend
        self.device = int(device)
        self.width = int(width)
        self.height = int(height)
        self.timeout_s = float(timeout_s)
        self.dual_usb_cameras = bool(dual_usb_cameras)
        self.free_cameras = bool(free_cameras)

    def capture(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        failures = []
        backends = self._candidate_backends()
        backend = None
        selected_device = None
        for candidate in backends:
            try:
                if path.exists():
                    path.unlink()
                capture_result = self._capture_backend(candidate, path)
                if isinstance(capture_result, dict):
                    selected_device = capture_result.get("camera_device")
                    extra_capture_info = capture_result
                else:
                    selected_device = capture_result
                    extra_capture_info = {}
                if not path.exists() or path.stat().st_size < 1000:
                    raise RuntimeError("没有生成有效图片")
                backend = candidate
                break
            except Exception as exc:
                failures.append("{}: {}".format(candidate, exc))
                if self.backend != "auto":
                    raise
        if backend is None:
            raise RuntimeError(
                "所有相机后端均失败：{}".format("；".join(failures))
            )
        if not path.exists() or path.stat().st_size < 1000:
            raise RuntimeError("相机没有生成有效图片：{}".format(path))
        with Image.open(path) as image:
            gray = image.convert("L")
            statistics = ImageStat.Stat(gray)
            brightness = float(statistics.mean[0])
            contrast = float(statistics.stddev[0])
            small = gray.copy()
            small.thumbnail((320, 240))
            pixels = np.asarray(small, dtype=np.uint8)
            dark_fraction = float(np.mean(pixels < 55))
            edge_pixels = np.asarray(
                small.filter(ImageFilter.FIND_EDGES), dtype=np.uint8
            )
            edge_fraction = float(np.mean(edge_pixels > 45))
        result = {
            "backend": backend,
            "camera_device": selected_device,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "dark_fraction": round(dark_fraction, 4),
            "edge_fraction": round(edge_fraction, 4),
            "possibly_blurred_or_flat": contrast < 18.0,
        }
        result.update(extra_capture_info)
        return result

    def _capture_backend(self, backend, path):
        if backend == "rpicam":
            command = [
                "rpicam-still", "--nopreview", "--immediate",
                "--width", str(self.width), "--height", str(self.height),
                "-o", str(path),
            ]
            self._run(command)
            return "rpicam"
        elif backend == "libcamera":
            command = [
                "libcamera-still", "--nopreview", "--immediate",
                "--width", str(self.width), "--height", str(self.height),
                "-o", str(path),
            ]
            self._run(command)
            return "libcamera"
        else:
            if self.dual_usb_cameras:
                return self._opencv_capture_dual(path)
            return self._opencv_capture(path)

    def _candidate_backends(self):
        if self.backend != "auto":
            return [self.backend]
        candidates = []
        # USB UVC cameras must use V4L2/OpenCV. Merely having the
        # rpicam-still executable installed does not mean a CSI/libcamera
        # camera is available; on a USB-only system it reports
        # "no cameras available".
        if glob.glob("/dev/video*"):
            candidates.append("opencv")
        if shutil.which("rpicam-still"):
            candidates.append("rpicam")
        if shutil.which("libcamera-still"):
            candidates.append("libcamera")
        if "opencv" not in candidates:
            candidates.append("opencv")
        return candidates

    def _run(self, command):
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError("相机命令失败：{}".format(message))

    def _opencv_capture(self, path):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "没有 rpicam-still/libcamera-still，且缺少 OpenCV；"
                "请运行 pip install -r requirements-pi.txt"
            ) from exc
        devices = [self.device]
        for device_path in sorted(glob.glob("/dev/video*")):
            suffix = device_path[len("/dev/video"):]
            if suffix.isdigit() and int(suffix) not in devices:
                devices.append(int(suffix))
        failures = []
        for device in devices:
            camera = self._open_opencv_device(
                device, "/dev/video{}".format(device)
            )
            try:
                if camera is None:
                    failures.append("/dev/video{} 无法打开".format(device))
                    continue
                frame = None
                for _index in range(8):
                    ok, candidate = camera.read()
                    if ok and candidate is not None and candidate.size:
                        frame = candidate
                if frame is None:
                    failures.append("/dev/video{} 无法取帧".format(device))
                    continue
                if not cv2.imwrite(str(path), frame):
                    failures.append("/dev/video{} 图片写入失败".format(device))
                    continue
                return "/dev/video{}".format(device)
            finally:
                camera.release()
        raise RuntimeError(
            "没有可取帧的USB摄像头节点：{}".format("；".join(failures))
        )

    @staticmethod
    def _stable_usb_capture_nodes():
        """Return one index-0 capture node per physical camera/USB port."""
        for pattern in (
            # Exclude Raspberry Pi codec/PISP nodes. Only USB host-controller
            # paths are physical UVC cameras. ``usb`` and ``usbv2`` aliases
            # are de-duplicated below by their resolved /dev/videoN target.
            "/dev/v4l/by-path/*xhci*usb*-video-index0",
            "/dev/v4l/by-id/*-video-index0",
        ):
            nodes = sorted(glob.glob(pattern))
            unique = []
            resolved = set()
            for node in nodes:
                try:
                    target = str(Path(node).resolve())
                except OSError:
                    target = node
                if target in resolved:
                    continue
                resolved.add(target)
                unique.append(node)
            if len(unique) >= 2:
                return unique
        # Linux UVC normally exposes image/metadata pairs. Even-numbered
        # nodes are only a last resort; stable by-path links are preferred.
        return sorted(glob.glob("/dev/video*"))

    @staticmethod
    def _opencv_device_index(stable_path):
        """Resolve a stable udev link to the integer required by Pi OpenCV."""
        try:
            resolved = Path(stable_path).resolve()
        except OSError:
            resolved = Path(stable_path)
        name = resolved.name
        if name.startswith("video") and name[5:].isdigit():
            return int(name[5:]), str(resolved)
        text = str(stable_path)
        if text.startswith("/dev/video") and text[10:].isdigit():
            return int(text[10:]), text
        raise RuntimeError(
            "稳定摄像头路径没有解析到 /dev/videoN：{} -> {}".format(
                stable_path, resolved
            )
        )

    def _open_opencv_device(self, device_index, resolved_device):
        """打开一路 V4L2 摄像头。

        --free-cameras 开启且打开失败、确有占用进程时, 杀掉占用进程
        (fuser -k) 并重试一次, 失败返回 None。
        """
        import cv2
        for attempt in range(2):
            camera = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            camera.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if camera.isOpened():
                return camera
            camera.release()
            if not self.free_cameras or attempt > 0:
                return None
            users = self._video_device_users(resolved_device)
            if not users:
                return None
            print(
                "释放摄像头占用进程: {} -> {}".format(resolved_device, users),
                flush=True,
            )
            # 除了 /dev/videoN 本身, 还要释放关联的 UVC 媒体控制器节点
            # (/dev/mediaN): pipewire/wireplumber 等会占住 media 节点导致
            # REQBUFS busy, 而 fuser 查 /dev/videoN 看不到它们 (2026-08-16 踩坑)。
            media_nodes = []
            text = str(resolved_device)
            if text.startswith("/dev/video") and text[10:].isdigit():
                for sysfs in glob.glob(
                    "/sys/class/video4linux/video{}/device/media*".format(
                        text[10:]
                    )
                ):
                    entry = sysfs.rsplit("/", 1)[-1]
                    if entry.startswith("media") and entry[5:].isdigit():
                        node = "/dev/" + entry
                        if glob.glob(node):
                            media_nodes.append(node)
            try:
                subprocess.run(
                    ["fuser", "-k", str(resolved_device)] + media_nodes,
                    capture_output=True,
                    timeout=10.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            time.sleep(1.5)
        return None

    def _opencv_capture_dual(self, path):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("双USB摄像头模式需要 OpenCV") from exc
        candidates = self._stable_usb_capture_nodes()
        cameras = []
        devices = []
        failures = []
        try:
            for device in candidates:
                try:
                    device_index, resolved_device = (
                        self._opencv_device_index(device)
                    )
                except RuntimeError as exc:
                    failures.append(str(exc))
                    continue
                # Raspberry Pi's packaged OpenCV accepts a V4L2 integer here,
                # but reports "can't be used to capture by name" for udev
                # symlink strings. Keep the stable path as identity and open
                # its current numeric target.
                camera = self._open_opencv_device(device_index, resolved_device)
                if camera is None:
                    users = self._video_device_users(resolved_device)
                    failures.append(
                        "{} -> {} 无法打开{}".format(
                            device,
                            resolved_device,
                            "（占用进程：{}）".format(users)
                            if users else
                            "（未发现占用进程；请检查video组权限或单路V4L2采集）",
                        )
                    )
                    continue
                cameras.append(camera)
                devices.append(str(device))
                if len(cameras) == 2:
                    break
            if len(cameras) < 2:
                raise RuntimeError(
                    "只找到 {} 路可用摄像头：{}".format(
                        len(cameras), "；".join(failures)
                    )
                )

            frames = [None, None]
            deadline = time.monotonic() + self.timeout_s
            while time.monotonic() < deadline:
                grabbed = [camera.grab() for camera in cameras]
                if not all(grabbed):
                    continue
                retrieved = [camera.retrieve() for camera in cameras]
                if all(
                    ok and frame is not None and frame.size
                    for ok, frame in retrieved
                ):
                    frames = [frame for _ok, frame in retrieved]
                    # Warm up exposure while always triggering both first.
                    if time.monotonic() + 0.35 >= deadline:
                        break
                    if all(
                        float(np.mean(frame)) > 4.0 for frame in frames
                    ):
                        break
            if any(frame is None for frame in frames):
                raise RuntimeError("两路摄像头未能在超时前同步取得图像")

            path = Path(path)
            second_path = path.with_name(
                path.stem + "_camera_2" + path.suffix
            )
            if not cv2.imwrite(str(path), frames[0]):
                raise RuntimeError("第一路摄像头图片写入失败")
            if not cv2.imwrite(str(second_path), frames[1]):
                raise RuntimeError("第二路摄像头图片写入失败")
            second_quality = self._image_quality(second_path)
            return {
                "camera_device": devices[0],
                "camera_count": 2,
                "camera_devices": devices,
                "camera_images": [path.name, second_path.name],
                "camera_2_image": second_path.name,
                "camera_2_device": devices[1],
                "camera_2_brightness": second_quality["brightness"],
                "camera_2_contrast": second_quality["contrast"],
                "camera_2_dark_fraction": second_quality["dark_fraction"],
                "camera_2_edge_fraction": second_quality["edge_fraction"],
                "camera_2_possibly_blurred_or_flat": second_quality[
                    "possibly_blurred_or_flat"
                ],
            }
        finally:
            for camera in cameras:
                camera.release()

    @staticmethod
    def _video_device_users(device_path):
        """Return process details holding a camera without changing state."""
        try:
            completed = subprocess.run(
                ["fuser", "-v", str(device_path)],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        text_value = " ".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        return " ".join(text_value.split())

    @staticmethod
    def _image_quality(path):
        with Image.open(path) as image:
            gray = image.convert("L")
            statistics = ImageStat.Stat(gray)
            brightness = float(statistics.mean[0])
            contrast = float(statistics.stddev[0])
            small = gray.copy()
            small.thumbnail((320, 240))
            pixels = np.asarray(small, dtype=np.uint8)
            dark_fraction = float(np.mean(pixels < 55))
            edge_pixels = np.asarray(
                small.filter(ImageFilter.FIND_EDGES), dtype=np.uint8
            )
            edge_fraction = float(np.mean(edge_pixels > 45))
        return {
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "dark_fraction": round(dark_fraction, 4),
            "edge_fraction": round(edge_fraction, 4),
            "possibly_blurred_or_flat": contrast < 18.0,
        }


class OnlineQwenComponentLabeler:
    """Call the shared 3090 model and promote only repeated bounded evidence."""

    _TYPE_LABELS = {
        "fixed_cabinet": "cabinet",
        "cabinet": "cabinet",
        "fixed_table": "table",
        "table": "table",
        "desk": "table",
        "chair": "chair",
        "stool": "stool",
        "fixed_trash_bin": "trash_bin",
        "trash_bin": "trash_bin",
        "box": "box",
        "fixed_structure": "fixed_structure",
        "door_frame": "door_frame",
        "wall_corner": "wall_corner",
        "column": "column",
        "cleaning_tool": "cleaning_tool",
        "broom": "broom",
        "dustpan": "dustpan",
    }
    _SPECIFIC_PARENT_LABELS = {
        "cabinet", "table", "chair", "stool", "trash_bin", "box",
        "cleaning_tool", "broom", "dustpan",
    }

    def __init__(
        self,
        url,
        timeout_seconds=90.0,
        min_confidence=0.80,
        confirmations=2,
    ):
        self.url = str(url or "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.min_confidence = float(min_confidence)
        self.confirmations = max(1, int(confirmations))
        self._candidates = {}

    @staticmethod
    def _jpeg_pair(left_path, right_path):
        with Image.open(left_path) as left_source:
            left = left_source.convert("RGB")
        with Image.open(right_path) as right_source:
            right = right_source.convert("RGB")
        height = min(left.height, right.height)
        if left.height != height:
            left.thumbnail((left.width, height))
        if right.height != height:
            right.thumbnail((right.width, height))
        canvas = Image.new("RGB", (left.width + right.width, height), "black")
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 82, 28), fill=(0, 0, 0))
        draw.text((6, 6), "LEFT", fill=(255, 255, 255))
        draw.rectangle((left.width, 0, left.width + 92, 28), fill=(0, 0, 0))
        draw.text((left.width + 6, 6), "RIGHT", fill=(255, 255, 255))
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=92)
        return output.getvalue()

    @staticmethod
    def _multipart(camera_bytes, context):
        boundary = "----codex-inspection-" + uuid.uuid4().hex
        chunks = []

        def add(name, value, filename=None, content_type=None):
            chunks.append(("--" + boundary + "\r\n").encode("ascii"))
            disposition = 'Content-Disposition: form-data; name="{}"'.format(name)
            if filename is not None:
                disposition += '; filename="{}"'.format(filename)
            chunks.append((disposition + "\r\n").encode("utf-8"))
            if content_type:
                chunks.append(("Content-Type: " + content_type + "\r\n").encode("ascii"))
            chunks.append(b"\r\n")
            chunks.append(value if isinstance(value, bytes) else str(value).encode("utf-8"))
            chunks.append(b"\r\n")

        add("camera_image", camera_bytes, "stereo.jpg", "image/jpeg")
        blank_lidar = io.BytesIO()
        Image.new("RGB", (320, 320), "white").save(blank_lidar, format="PNG")
        add("lidar_image", blank_lidar.getvalue(), "no_lidar.png", "image/png")
        add("context_json", json.dumps(context, ensure_ascii=False))
        chunks.append(("--" + boundary + "--\r\n").encode("ascii"))
        return b"".join(chunks), "multipart/form-data; boundary=" + boundary

    def _candidate(self, payload, target=None):
        candidates = []
        for item in payload.get("landmarks", ()):
            if not isinstance(item, dict):
                continue
            label = self._TYPE_LABELS.get(str(item.get("type") or "").lower())
            try:
                confidence = float(item.get("confidence", 0.0))
                bbox = [float(value) for value in item.get("camera_bbox", ())]
            except (TypeError, ValueError, OverflowError):
                continue
            if label is None or confidence < self.min_confidence or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height / 1_000_000.0
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            centred_in_one_camera = (
                120.0 <= center_x <= 380.0
                or 620.0 <= center_x <= 880.0
            )
            evidence_ok = (
                str(item.get("stereo_consistency")) == "both"
                and bool(item.get("ground_contact", False))
                and bool(item.get("map_memory_allowed", False))
                and not bool(item.get("behind_glass", False))
            )
            bounded_box = (
                0.006 <= area <= 0.35
                and 0.0 <= x1 < x2 <= 1000.0
                and 0.0 <= y1 < y2 <= 1000.0
                and center_y >= 120.0
                and centred_in_one_camera
            )
            # Qwen3-VL-2B sometimes classifies a low cabinet plinth correctly
            # and consistently but can only return the complete LEFT or RIGHT
            # camera half as its box. Permit that *classification* evidence
            # only for concrete parent objects at >=0.90 confidence. The usual
            # repeated-label confirmation still applies in ``analyze``.
            full_single_camera_half = (
                y1 <= 5.0
                and y2 >= 995.0
                and (
                    (x1 <= 5.0 and 495.0 <= x2 <= 505.0)
                    or (495.0 <= x1 <= 505.0 and x2 >= 995.0)
                )
            )
            specific_stereo_classification = (
                target is not None
                and target.kind == "interior_black_region"
                and label in self._SPECIFIC_PARENT_LABELS
                and confidence >= max(0.90, self.min_confidence)
                and full_single_camera_half
            )
            if not evidence_ok or not (
                bounded_box or specific_stereo_classification
            ):
                continue
            candidate_item = dict(item)
            if specific_stereo_classification and not bounded_box:
                candidate_item["evidence_mode"] = (
                    "specific_stereo_classification_without_tight_bbox"
                )
            candidates.append((confidence, label, candidate_item))
        return max(candidates, default=None, key=lambda item: item[0])

    def analyze(self, target, record, left_path, right_path):
        if not self.url:
            return {"enabled": False, "promoted": False}
        context = {
            "inspection_mode": "component_label",
            "camera_mode": "stereo_side_by_side",
            "target_id": target.target_id,
            "target_kind": target.kind,
            "target_component_cells": int(target.component_cells),
            "robot_pose": {
                "x_m": record.get("x_m"),
                "y_m": record.get("y_m"),
                "yaw_deg": record.get("yaw_deg"),
            },
            "allow_lidar_fallback": False,
        }
        try:
            camera_bytes = self._jpeg_pair(left_path, right_path)
            body, content_type = self._multipart(camera_bytes, context)
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={"Content-Type": content_type},
                method="POST",
            )
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            return {
                "enabled": True,
                "promoted": False,
                "error": str(exc),
            }
        candidate = self._candidate(payload, target)
        result = {
            "enabled": True,
            "promoted": False,
            "scene_summary": str(payload.get("scene_summary") or "")[:300],
            "observation_advice": payload.get("observation_advice", {}),
            "elapsed_s": payload.get("elapsed_s"),
        }
        if candidate is None:
            self._candidates.pop(target.target_id, None)
            result["rejection"] = "no_bounded_stereo_candidate"
            result["service_candidates"] = list(
                payload.get("landmarks", ())
            )[:3]
            return result
        confidence, label, raw_candidate = candidate
        history = self._candidates.setdefault(target.target_id, [])
        history.append(label)
        del history[:-self.confirmations]
        confirmed = (
            len(history) >= self.confirmations
            and len(set(history[-self.confirmations:])) == 1
        )
        result.update({
            "candidate_label": label,
            "candidate_confidence": round(confidence, 4),
            "confirmation_count": (
                len(history) if len(set(history)) == 1 else 1
            ),
            "required_confirmations": self.confirmations,
            "candidate": raw_candidate,
            "promoted": confirmed,
        })
        return result

    def analyze_panorama(self, record, left_path, right_path):
        """Observe the complete image without assigning a whole map component."""
        if not self.url:
            return {"enabled": False, "promoted": False}
        context = {
            "inspection_mode": "panorama_sector",
            "camera_mode": "stereo_side_by_side",
            "station_id": record.get("panorama_station_id"),
            "relative_yaw_deg": record.get("relative_panorama_yaw_deg"),
            "robot_pose": {
                "x_m": record.get("x_m"),
                "y_m": record.get("y_m"),
                "yaw_deg": record.get("yaw_deg"),
            },
            "pose_confidence": record.get("localization_confidence_score"),
            "allow_lidar_fallback": False,
            "assign_entire_black_component": False,
        }
        try:
            camera_bytes = self._jpeg_pair(left_path, right_path)
            body, content_type = self._multipart(camera_bytes, context)
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={"Content-Type": content_type},
                method="POST",
            )
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            return {"enabled": True, "promoted": False, "error": str(exc)}
        return {
            "enabled": True,
            "promoted": False,
            "semantic_scope": "photo_observations_only",
            "scene_summary": str(payload.get("scene_summary") or "")[:800],
            "landmarks": list(payload.get("landmarks") or ())[:12],
            "objects": list(payload.get("objects") or ())[:12],
            "observation_advice": payload.get("observation_advice", {}),
            "elapsed_s": payload.get("elapsed_s"),
        }


class InspectionReport:
    def __init__(self, output_directory, map_path):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.directory = Path(output_directory) / stamp
        self.map_path = str(map_path)
        self.records = []
        self.skipped_targets = []
        self.temporary_obstacle_events = []
        self.storage_recovery_events = []
        self._directory_initialized = False
        self._ensure_directory()

    def _ensure_directory(self):
        """Recreate a disappeared run directory without losing the mission.

        Field run 20260905_155436 successfully wrote its camera test and then
        found the whole timestamp directory missing during a later plan-map
        refresh.  Storage cleanup must not crash motion control or prevent the
        final manifest from explaining what happened.
        """
        existed = self.directory.is_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        if self._directory_initialized and not existed:
            event = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": "inspection_output_directory_disappeared",
                "directory": str(self.directory),
            }
            self.storage_recovery_events.append(event)
            print(
                "巡检结果目录曾被外部删除，已自动重建并继续；"
                "恢复事件会写入 manifest。",
                flush=True,
            )
        self._directory_initialized = True

    def add(self, record):
        self.records.append(record)
        self.save()

    def add_skipped_target(self, target, pose, reason, attempts):
        """Record a target that was not photographed without aborting the mission."""
        self.skipped_targets.append({
            "target_id": target.target_id,
            "kind": target.kind,
            "target_x_m": round(target.target_xy_m[0], 4),
            "target_y_m": round(target.target_xy_m[1], 4),
            "view_x_m": round(target.view_xy_m[0], 4),
            "view_y_m": round(target.view_xy_m[1], 4),
            "last_x_m": round(pose.x_m, 4),
            "last_y_m": round(pose.y_m, 4),
            "last_yaw_deg": round(math.degrees(pose.yaw_rad), 2),
            "reason": str(reason),
            "attempts": int(attempts),
        })
        self.save()

    def add_temporary_obstacle(self, known_map, pose, added_cells):
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "x_m": round(pose.x_m, 4),
            "y_m": round(pose.y_m, 4),
            "yaw_deg": round(math.degrees(pose.yaw_rad), 2),
            "added_cells": int(added_cells),
        }
        self.temporary_obstacle_events.append(event)
        self.save_temporary_map(known_map)
        self.save()

    def save_temporary_map(self, known_map):
        image = np.flipud(known_map.image).copy()
        rgb = np.repeat(image[..., None], 3, axis=2)
        temporary = np.flipud(known_map.temporary_occupied)
        rgb[temporary] = np.asarray((230, 35, 35), dtype=np.uint8)
        destination = self.directory / "temporary_obstacles.png"
        for attempt in range(2):
            self._ensure_directory()
            try:
                Image.fromarray(rgb).save(destination)
                break
            except FileNotFoundError:
                if attempt:
                    raise

    def save_plan(self, known_map, start_pose, ordered_targets,
                  live_pose=None, live_trail=None):
        """渲染巡检计划图。live_pose/live_trail 用于运行时实时刷新
        (2026-08-16 用户要求: 能跟踪定位点的变化)。"""
        image = Image.fromarray(np.flipud(known_map.image)).convert("RGB")
        draw = ImageDraw.Draw(image)

        def screen(xy):
            column, row = known_map.world_to_grid((xy,))[0]
            return int(column), int(known_map.height - 1 - row)

        current = (start_pose.x_m, start_pose.y_m)
        start_pixel = screen(current)
        draw.ellipse(
            (start_pixel[0] - 4, start_pixel[1] - 4,
             start_pixel[0] + 4, start_pixel[1] + 4),
            fill=(0, 180, 0),
        )
        for trail_pose in (live_trail or []):
            pixel = screen((trail_pose.x_m, trail_pose.y_m))
            draw.ellipse(
                (pixel[0] - 2, pixel[1] - 2,
                 pixel[0] + 2, pixel[1] + 2),
                fill=(120, 230, 120),
            )
        if live_pose is not None:
            pixel = screen((live_pose.x_m, live_pose.y_m))
            draw.ellipse(
                (pixel[0] - 6, pixel[1] - 6,
                 pixel[0] + 6, pixel[1] + 6),
                fill=(0, 255, 0),
                outline=(0, 0, 0),
                width=1,
            )
        for target, path in ordered_targets:
            pixels = [screen(point) for point in path]
            if pixels:
                draw.line([screen(current)] + pixels, fill=(30, 100, 230), width=2)
            view = screen(target.view_xy_m)
            observed = screen(target.target_xy_m)
            draw.line((view, observed), fill=(255, 150, 0), width=1)
            draw.ellipse(
                (view[0] - 3, view[1] - 3, view[0] + 3, view[1] + 3),
                fill=(20, 80, 255),
            )
            draw.rectangle(
                (observed[0] - 2, observed[1] - 2,
                 observed[0] + 2, observed[1] + 2),
                fill=(230, 30, 30),
            )
            current = target.view_xy_m
        destination = self.directory / "inspection_plan.png"
        for attempt in range(2):
            self._ensure_directory()
            try:
                image.save(destination)
                break
            except FileNotFoundError:
                if attempt:
                    raise

    def save(self):
        payload = {
            "map": self.map_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "captures": self.records,
            "skipped_targets": self.skipped_targets,
            "temporary_obstacle_events": self.temporary_obstacle_events,
            "storage_recovery_events": self.storage_recovery_events,
            "temporary_obstacles_are_session_only": True,
        }
        fields = (
            "target_id", "kind", "image", "x_m", "y_m", "yaw_deg",
            "target_x_m", "target_y_m", "brightness", "contrast",
            "dark_fraction", "edge_fraction",
            "possibly_blurred_or_flat", "inspection_subject", "review_question",
            "capture_mode", "capture_note", "photo_index", "photos_per_target",
            "camera_count", "camera_device", "camera_2_image",
            "camera_2_device", "camera_2_brightness", "camera_2_contrast",
            "camera_2_dark_fraction", "camera_2_edge_fraction",
            "camera_2_possibly_blurred_or_flat",
            "panorama_station_id", "relative_panorama_yaw_deg", "captured_at",
            "localization_confidence", "localization_confidence_score",
            "low_confidence_pose", "front_clearance_m",
            "new_coverage_samples", "coverage_ratio_after_capture",
        )
        for attempt in range(2):
            self._ensure_directory()
            csv_path = self.directory / "inspection_manifest.csv"
            csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
            json_path = self.directory / "inspection_manifest.json"
            json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
            try:
                with csv_temporary.open(
                    "w", newline="", encoding="utf-8-sig"
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for item in self.records:
                        writer.writerow({
                            field: item.get(field, "") for field in fields
                        })
                csv_temporary.replace(csv_path)
                json_temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                json_temporary.replace(json_path)
                break
            except FileNotFoundError:
                if attempt:
                    raise


class PermanentSemanticMap:
    """Persistent labels for permanent PGM geometry, keyed by map hash."""

    def __init__(self, path, map_path, map_hash, targets):
        self.path = Path(path)
        self.map_path = str(map_path)
        self.map_hash = str(map_hash)
        self.data = self._load()
        if self.data.get("map_sha256") != self.map_hash:
            self.data = {
                "format_version": 1,
                "map": self.map_path,
                "map_sha256": self.map_hash,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "entities": {},
            }
        entities = self.data.setdefault("entities", {})
        for target in targets:
            entity = entities.setdefault(target.target_id, {})
            component_id = target.target_id.rsplit("_", 1)[0]
            entity.update({
                "target_id": target.target_id,
                "component_id": component_id,
                "kind": target.kind,
                "target_x_m": round(target.target_xy_m[0], 4),
                "target_y_m": round(target.target_xy_m[1], 4),
                "component_cells": int(target.component_cells),
                "permanent": True,
            })
            entity.setdefault(
                "semantic_label",
                "wall_surface"
                if target.kind == "perimeter_wall"
                else "unclassified_permanent_obstacle",
            )
            entity.setdefault("observations", [])
        self.save()

    def _load(self):
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    def record_observation(self, target, record, image_path):
        entity = self.data["entities"][target.target_id]
        observation = {
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": str(Path(image_path).resolve()),
            "robot_x_m": record["x_m"],
            "robot_y_m": record["y_m"],
            "robot_yaw_deg": record["yaw_deg"],
            "brightness": record.get("brightness"),
            "contrast": record.get("contrast"),
        }
        observations = entity.setdefault("observations", [])
        observations.append(observation)
        # Bound history without losing the persistent semantic label.
        del observations[:-20]
        entity["last_observed_at"] = observation["captured_at"]
        entity["last_image"] = observation["image"]
        self.save()

    def record_online_qwen(self, target, inference, image_path):
        entity = self.data["entities"][target.target_id]
        history = entity.setdefault("online_qwen_history", [])
        item = {
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": str(Path(image_path).resolve()),
            **dict(inference),
        }
        history.append(item)
        del history[:-12]
        if inference.get("promoted"):
            component_id = entity.get("component_id") or (
                target.target_id.rsplit("_", 1)[0]
            )
            for component_entity in self.data["entities"].values():
                if component_entity.get("component_id") != component_id:
                    continue
                component_entity["semantic_label"] = str(
                    inference["candidate_label"]
                )
                component_entity["semantic_scope"] = (
                    "entire_connected_black_component"
                )
                component_entity["labelled_component_cell_count"] = int(
                    component_entity.get("component_cells")
                    or target.component_cells
                )
                component_entity["online_qwen"] = {
                    "promoted": True,
                    "confidence": inference.get("candidate_confidence"),
                    "confirmations": inference.get(
                        "required_confirmations"
                    ),
                    "source_target_id": target.target_id,
                    "source_image": item["image"],
                    "updated_at": item["captured_at"],
                }
        self.save()

    def target_classified(self, target_id):
        entity = self.data.get("entities", {}).get(target_id, {})
        label = str(entity.get("semantic_label") or "")
        return label not in {"", "unclassified_permanent_obstacle", "unknown"}

    def target_online_confirmed(self, target_id):
        """Return true only for labels promoted by the strict online workflow."""
        entity = self.data.get("entities", {}).get(target_id, {})
        online_qwen = entity.get("online_qwen") or {}
        return bool(online_qwen.get("promoted", False))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _skip_classified_interior_targets(targets, semantic_map):
    skipped_components = {
        target.target_id.rsplit("_", 1)[0]
        for target in targets
        if target.kind == "interior_black_region"
        and semantic_map.target_online_confirmed(target.target_id)
    }
    remaining = [
        target for target in targets
        if not (
            target.kind == "interior_black_region"
            and target.target_id.rsplit("_", 1)[0]
            in skipped_components
        )
    ]
    return remaining, skipped_components


def _verified_semantic_capture_pose(semantic_map, target):
    """Return the pose of the image that established a Qwen map label."""
    entity = semantic_map.data.get("entities", {}).get(target.target_id, {})
    annotation = entity.get("qwen_annotation") or {}
    if not annotation.get("promoted_to_semantic_label", False):
        return None
    if not any(
        bool(view.get("target_visible"))
        for view in annotation.get("views", ())
        if isinstance(view, dict)
    ):
        return None
    source_images = {
        Path(str(value)).name
        for value in annotation.get("source_images", ())
        if value
    }
    if not source_images:
        return None
    for observation in reversed(entity.get("observations", ())):
        if Path(str(observation.get("image", ""))).name not in source_images:
            continue
        try:
            pose = Pose2D(
                float(observation["robot_x_m"]),
                float(observation["robot_y_m"]),
                math.radians(float(observation["robot_yaw_deg"])),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        target_distance_m = math.hypot(
            target.target_xy_m[0] - pose.x_m,
            target.target_xy_m[1] - pose.y_m,
        )
        if 0.35 <= target_distance_m <= 1.50:
            return pose
    return None


def _apply_verified_semantic_views(targets, semantic_map):
    """Prefer proven photo poses and remove duplicate geometry targets."""
    verified = {}
    adjusted = []
    for target in targets:
        pose = _verified_semantic_capture_pose(semantic_map, target)
        if pose is not None:
            verified[target.target_id] = pose
            target = replace(
                target,
                view_xy_m=(pose.x_m, pose.y_m),
                yaw_rad=pose.yaw_rad,
            )
        adjusted.append(target)

    kept = []
    for target in sorted(
        adjusted,
        key=lambda item: item.target_id not in verified,
    ):
        duplicate = next((
            old for old in kept
            if math.hypot(
                old.target_xy_m[0] - target.target_xy_m[0],
                old.target_xy_m[1] - target.target_xy_m[1],
            ) <= 0.18
        ), None)
        if duplicate is None:
            kept.append(target)
            continue
        if (
            target.target_id in verified
            and duplicate.target_id not in verified
        ):
            kept.remove(duplicate)
            kept.append(target)
    order = {item.target_id: index for index, item in enumerate(targets)}
    kept.sort(key=lambda item: order.get(item.target_id, len(order)))
    return kept, verified


def _capture_reference_limit(
    target_id,
    reference_views,
    verified_capture_poses,
    configured_nearby_m,
):
    reference = reference_views[target_id]
    limit_m = float(configured_nearby_m)
    if target_id in verified_capture_poses:
        limit_m = min(limit_m, 0.32)
    return reference, limit_m


def _within_capture_reference(
    pose,
    target_id,
    reference_views,
    verified_capture_poses,
    configured_nearby_m,
):
    reference, limit_m = _capture_reference_limit(
        target_id,
        reference_views,
        verified_capture_poses,
        configured_nearby_m,
    )
    distance_m = math.hypot(
        reference[0] - pose.x_m,
        reference[1] - pose.y_m,
    )
    return distance_m <= limit_m, distance_m, limit_m


def _serial_config(args):
    return SerialConfig(
        port=args.esp_port,
        baudrate=230400,
        request_timeout_s=0.40,
        retries=1,
        link_mode=args.link,
        ble_device_name=args.ble_name,
        ble_address=args.ble_address,
        ble_connect_timeout_s=args.ble_connect_timeout,
        ble_operation_timeout_s=args.ble_operation_timeout,
        wheel_output_gains=(
            args.front_left_gain,
            args.front_right_gain,
            args.rear_left_gain,
            args.rear_right_gain,
        ),
    )


def _lidar_config(args):
    return LidarConfig(
        port=args.lidar_port,
        baudrate=args.lidar_baud,
        angle_offset_deg=args.angle_offset,
        clockwise=not args.counterclockwise,
        motor_control=not args.no_motor_control,
    )


def _build_speed_controller(args, *, navigation=False):
    """Build phase-specific motion control without weakening startup checks.

    Initial global localization keeps the controller's conservative scan
    quality limits.  Once that pose is confirmed, navigation deliberately
    accepts the same relaxed tracking envelope used by the mission.  This
    prevents a valid 0.10 m-RMSE corridor pose from resetting the turn-start
    boost on every frame and leaving the chassis buzzing at 1500 mrad/s.
    """
    config = {
        "minimum_turn_floor_mrad_s": args.turn_speed,
        "initial_turn_floor_mrad_s": args.turn_speed,
        "initial_turn_limit_mrad_s": args.relocalization_max_turn_speed,
        "turn_adjust_step_mrad_s": max(50, args.relocalization_turn_step),
        "linear_adjust_step_mm_s": max(20, args.drive_speed_step),
        "maximum_linear_boost_mm_s": max(
            0, args.max_drive_speed - min(args.speed, args.slow_speed)
        ),
        "minimum_applied_linear_speed_mm_s": min(
            args.speed, args.slow_speed
        ),
        "linear_start_kick_mm_s": min(
            args.max_drive_speed, args.start_kick_speed
        ),
        "linear_progress_confirmation_frames": 2,
        "preserve_start_kick_on_poor_initial_match": True,
    }
    if navigation:
        config.update({
            "poor_rmse_m": float(args.relaxed_tracking_rmse),
            "poor_inlier_ratio": float(
                args.relaxed_tracking_inlier_ratio
            ),
            # Field log 20260821_130545 showed that a nominal 500--700 ms
            # turn was refreshed for roughly one second.  With the ESP32's
            # 1.8 angular output gain that produced 60--124 degree jumps
            # between lidar frames.  Use separated breakaway pulses instead:
            # high torque crosses static friction, while the short TTL keeps
            # one scan inside the ICP/map tracker's angular capture range.
            "minimum_turn_ttl_ms": 200,
            "initial_turn_ttl_ms": 220,
            "maximum_turn_ttl_ms": 260,
            "turn_ttl_adjust_step_ms": 20,
            # End the breakaway kick after one measured translation frame;
            # a second lidar/matching interval would keep startup torque on
            # much longer than needed near an inspection target.
            "linear_progress_confirmation_frames": 1,
            "linear_start_kick_max_frames": 1,
        })
    return AdaptiveSpeedController(
        args.relocalization_max_turn_speed,
        args.max_drive_speed,
        AdaptiveSpeedConfig(**config),
    )


def _spaced_targets(targets, min_spacing_m=0.8):
    """Keep spatially distinct views while preserving interior subjects first."""
    kept = []
    # A wall view and an interior-object view often share almost the same
    # parking cell.  Keep the object-facing view in that cluster; otherwise
    # a wall-first greedy pass silently removes most meaningful objects.
    prioritized = sorted(
        enumerate(targets),
        key=lambda item: (
            item[1].kind != "interior_black_region",
            item[0],
        ),
    )
    for _index, target in prioritized:
        if all(
            math.hypot(
                target.view_xy_m[0] - other.view_xy_m[0],
                target.view_xy_m[1] - other.view_xy_m[1],
            ) >= min_spacing_m
            for other in kept
        ):
            kept.append(target)
    return kept


def _information_view_for_target(known_map, target, maximum_distance_m):
    """Move a component-facing camera view outward along clear map space.

    The target identity stays deterministic: a wide photo still points at the
    same black connected component.  When the ray is blocked by a corner, the
    original close view is retained for a detail photo.
    """
    maximum_distance_m = float(maximum_distance_m)
    target_xy = np.asarray(target.target_xy_m, dtype=np.float64)
    original_view = np.asarray(target.view_xy_m, dtype=np.float64)
    direction = original_view - target_xy
    original_distance_m = float(np.linalg.norm(direction))
    if maximum_distance_m <= original_distance_m + 0.25:
        return target, False
    direction /= max(original_distance_m, 1e-9)
    best_view = original_view
    for distance_m in np.arange(
        original_distance_m + known_map.resolution_m,
        maximum_distance_m + known_map.resolution_m * 0.5,
        known_map.resolution_m,
    ):
        candidate = target_xy + direction * distance_m
        column, row = known_map.world_to_grid((candidate,))[0]
        column, row = int(column), int(row)
        if (
            not known_map.in_bounds(column, row)
            or not known_map.traversable[row, column]
            or not _inspection_target_visible_from(
                known_map,
                (float(candidate[0]), float(candidate[1])),
                target.target_xy_m,
            )
        ):
            break
        best_view = candidate
    gain_m = float(np.linalg.norm(best_view - original_view))
    if gain_m < 0.35:
        return target, False
    view_xy = (float(best_view[0]), float(best_view[1]))
    return replace(
        target,
        view_xy_m=view_xy,
        yaw_rad=math.atan2(
            target.target_xy_m[1] - view_xy[1],
            target.target_xy_m[0] - view_xy[0],
        ),
    ), True


def _information_view_targets(known_map, targets, maximum_distance_m):
    adjusted = []
    metadata = {}
    for target in targets:
        original = target
        target, widened = _information_view_for_target(
            known_map, target, maximum_distance_m
        )
        adjusted.append(target)
        if widened:
            metadata[target.target_id] = {
                "original_view": original.view_xy_m,
                "wide_view": target.view_xy_m,
                "wide_distance_m": round(math.hypot(
                    target.target_xy_m[0] - target.view_xy_m[0],
                    target.target_xy_m[1] - target.view_xy_m[1],
                ), 4),
            }
    return adjusted, metadata


def _command_for_pose(
    args,
    pose,
    waypoint,
    final_yaw=None,
    force_forward_arc=False,
):
    dx = waypoint[0] - pose.x_m
    dy = waypoint[1] - pose.y_m
    distance = math.hypot(dx, dy)
    desired = math.atan2(dy, dx) if distance > args.waypoint_tolerance else final_yaw
    if desired is None:
        return 0, 0, True
    heading_error = normalize_angle(desired - pose.yaw_rad)
    heading_tolerance = math.radians(args.heading_tolerance_deg)
    if distance <= args.waypoint_tolerance:
        if abs(heading_error) <= heading_tolerance:
            return 0, 0, True
        turn = int(math.copysign(
                max(
                    args.navigation_fine_turn_min_speed,
                min(
                    args.navigation_fine_turn_speed,
                    abs(heading_error) * 900.0,
                ),
            ),
            heading_error,
        ))
        return 0, turn, False
    # The chassis often cannot break static friction during a pure pivot, but
    # starts reliably once both sides receive forward drive. Keep pure turns
    # for sharp corners where an arc could cut the inflated map corridor;
    # moderate errors are followed as a forward arc so the mission covers the
    # map instead of photographing every later target from one spot.
    if (
        abs(heading_error) > math.radians(50.0)
        and not force_forward_arc
    ):
        turn = int(math.copysign(
            max(650, min(args.turn_speed, abs(heading_error) * 1100.0)),
            heading_error,
        ))
        return 0, turn, False
    if force_forward_arc and abs(heading_error) > math.radians(50.0):
        speed = args.slow_speed
    else:
        speed = args.slow_speed if distance < 0.50 else args.speed
    angular = int(max(-700, min(700, heading_error * 1100.0)))
    return int(speed), angular, False


def _blocked_path_heading_command(args, pose, waypoint, tolerance_deg=10.0):
    """Return a stopped turn command after an arc was obstacle-blocked.

    A forced forward arc is a turn-stiction escape, not a license to keep
    pushing into the same corner. After any blocked forward command, align
    the chassis to the newly planned segment while stopped. ``None`` means
    alignment is close enough for normal path following to resume.
    """
    heading_error = normalize_angle(
        math.atan2(waypoint[1] - pose.y_m, waypoint[0] - pose.x_m)
        - pose.yaw_rad
    )
    if abs(heading_error) <= math.radians(float(tolerance_deg)):
        return None
    angular = int(math.copysign(
        min(
            args.relocalization_max_turn_speed,
            max(args.turn_speed, abs(heading_error) * 1100.0),
        ),
        heading_error,
    ))
    return 0, angular, False


def _forward_stop_envelope_m(
    linear_mm_s,
    ttl_ms,
    stop_distance_m,
    reserve_m=0.08,
):
    """Conservative lidar clearance for one pulse plus physical braking."""
    speed_m_s = abs(float(linear_mm_s)) / 1000.0
    duration_s = max(0.0, float(ttl_ms)) / 1000.0
    # Bound travel as if the chassis had already reached the requested speed.
    # This also covers a refresh arriving before the previous pulse fully
    # decelerated. ESP32 deceleration is 1000 mm/s².
    pulse_travel_m = speed_m_s * duration_s
    braking_m = speed_m_s * speed_m_s / (2.0 * 1.0)
    return (
        float(stop_distance_m)
        + float(reserve_m)
        + pulse_travel_m
        + braking_m
    )


def _navigation_command_ttl(
    applied_ttl_ms,
    linear_mm_s,
    front_clearance_m=None,
    stop_distance_m=0.42,
):
    """Choose forward pulse length from measured front lidar clearance.

    The former unconditional 350 ms cap kept resetting the ESP32's 650
    mm/s² linear ramp before the loaded chassis could start. In clear space
    that looked like a collision and caused repeated target replans/turns.
    Keep the short fallback when lidar support is unavailable, otherwise use
    the longest pulse whose travel and braking envelope fits the measured
    corridor.
    """
    requested_ms = max(1, int(applied_ttl_ms))
    if not linear_mm_s:
        return requested_ms
    try:
        clearance_m = float(front_clearance_m)
    except (TypeError, ValueError, OverflowError):
        clearance_m = math.nan
    if not math.isfinite(clearance_m):
        return min(requested_ms, 350)
    minimum_ms = min(requested_ms, 200)
    for candidate_ms in range(requested_ms, minimum_ms - 1, -25):
        if clearance_m >= _forward_stop_envelope_m(
            linear_mm_s,
            candidate_ms,
            stop_distance_m,
        ):
            return candidate_ms
    return minimum_ms


def _inspection_navigation_command_ttl(
    speed_controller,
    command,
    normal_ttl_ms,
    front_clearance_m,
    stop_distance_m,
):
    """Keep a steering correction from truncating a forward drive pulse.

    The adaptive controller deliberately gives angular commands a short turn
    TTL. That is appropriate for a pure spin, but waypoint driving almost
    always contains a small steering term. Mixed commands therefore use the
    lidar-bounded forward TTL; only zero-linear commands use the turn TTL.
    """
    if int(command.linear_mm_s) != 0:
        return _navigation_command_ttl(
            normal_ttl_ms,
            command.linear_mm_s,
            front_clearance_m,
            stop_distance_m,
        )
    return speed_controller.ttl_for(command, normal_ttl_ms)


def _obstacle_safe(
    scan,
    linear_mm_s,
    stop_distance_m,
    angular_mrad_s=None,
    full_scan_guard=False,
    command_ttl_ms=None,
):
    if linear_mm_s <= 0:
        return True
    speed_m_s = abs(linear_mm_s) / 1000.0
    if full_scan_guard:
        if command_ttl_ms is not None:
            required_m = _forward_stop_envelope_m(
                linear_mm_s,
                command_ttl_ms,
                stop_distance_m,
            )
        else:
            required_m = stop_distance_m + speed_m_s * 0.35 + 0.08
    else:
        # 初始主动定位只执行短动作并逐帧复查。保留此前实车接近成功
        # 版本的运动包络，避免把巡检连续运动的长延迟错误套到定位换位。
        required_m = stop_distance_m + speed_m_s * 0.4 + 0.15
    # External callers that do not provide angular velocity retain the
    # original 60-degree envelope. Known-map waypoint tracking provides it
    # explicitly: its map path already guarantees side clearance, so a
    # 42-degree forward corridor detects head-on obstacles without treating
    # both side walls of a narrow passage as one obstacle in front.
    width_deg = 60.0 if angular_mrad_s is None else 42.0
    return sector_min_distance(
        scan,
        center_deg=0.0,
        width_deg=width_deg,
        min_distance_m=0.08,
        max_distance_m=8.0,
    ) >= required_m


def _limit_navigation_linear_request(
    linear_mm_s,
    front_clearance_m,
    distance_to_target_view_m,
    final_waypoint,
    slow_speed_mm_s,
    fine_control_distance_m,
    motion_confirmed=False,
):
    """Apply approach limits before adaptive dead-zone compensation.

    The adaptive controller must see the same requested speed that is later
    sent to the chassis. Capping its applied output afterwards makes its
    feedback state believe that (for example) 360 mm/s was delivered while
    the ESP32 actually received only 170 mm/s. It then saturates its boost
    without ever producing enough torque to break static friction.

    A slow request can still receive the controller's short start kick while
    motion is unconfirmed. The speed-dependent obstacle envelope validates
    that real applied value, and after one confirmed movement frame the
    request remains at the configured slow approach speed.
    """
    try:
        linear_mm_s = int(linear_mm_s)
        slow_speed_mm_s = max(0, int(slow_speed_mm_s))
        front_clearance_m = float(front_clearance_m)
        distance_to_target_view_m = float(distance_to_target_view_m)
    except (TypeError, ValueError, OverflowError):
        return int(linear_mm_s)
    if linear_mm_s == 0:
        return 0
    near_front_object = (
        math.isfinite(front_clearance_m)
        and (
            front_clearance_m < 1.1
            or (front_clearance_m < 1.2 and bool(motion_confirmed))
        )
    )
    near_final_view = (
        bool(final_waypoint)
        and math.isfinite(distance_to_target_view_m)
        and distance_to_target_view_m <= float(fine_control_distance_m)
    )
    if not (near_front_object or near_final_view):
        return linear_mm_s
    capped = min(abs(linear_mm_s), slow_speed_mm_s)
    return capped if linear_mm_s > 0 else -capped


def _inspection_standoff_reached(
    pose,
    waypoint,
    front_clearance_m,
    waypoint_tolerance_m,
    stop_distance_m,
    nearby_capture_distance_m=0.30,
):
    """Accept a safe near-final stop when the mapped subject fills the lidar front.

    Inspection views intentionally sit close to a wall or black mapped object.
    The normal speed-dependent transit guard can therefore reject the subject
    itself before the center of the robot enters the exact planning cell. Only
    only the configured neighbourhood around that cell may use this rule, and
    measured clearance must still exceed the chassis stop distance.
    """
    try:
        front_clearance_m = float(front_clearance_m)
        distance_to_view_m = math.hypot(
            float(waypoint[0]) - float(pose.x_m),
            float(waypoint[1]) - float(pose.y_m),
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        math.isfinite(front_clearance_m)
        and front_clearance_m >= float(stop_distance_m)
        and distance_to_view_m <= max(
            float(waypoint_tolerance_m),
            float(nearby_capture_distance_m),
        )
    )


def _nearby_turn_timeout_capture_allowed(
    pose,
    target_view_xy,
    waypoint_index,
    waypoint_count,
    final_translation_complete,
    nearby_capture_distance_m,
):
    """Allow a photo after a bounded final turn, never at a transit waypoint."""
    if not (
        final_translation_complete
        or int(waypoint_index) == int(waypoint_count) - 1
    ):
        return False, math.inf
    distance_m = math.hypot(
        float(target_view_xy[0]) - float(pose.x_m),
        float(target_view_xy[1]) - float(pose.y_m),
    )
    return (
        math.isfinite(distance_m)
        and distance_m <= float(nearby_capture_distance_m),
        distance_m,
    )


def _navigation_turn_timeout_action(
    elapsed_s,
    timeout_s,
    nearby_final_capture_allowed,
    heading_error_rad=None,
    maximum_capture_error_deg=25.0,
):
    """Time out only a final nearby photo turn, never a transit turn."""
    if (
        nearby_final_capture_allowed
        and float(elapsed_s) >= float(timeout_s)
    ):
        if (
            heading_error_rad is not None
            and abs(math.degrees(float(heading_error_rad)))
            > float(maximum_capture_error_deg)
        ):
            return "replan"
        return "capture"
    return "continue"


def _capture_heading_within_limit(
    pose,
    final_yaw,
    measured_remaining_rad,
    maximum_error_deg,
):
    """Reject a nearby photo when the camera still faces the wrong subject."""
    if measured_remaining_rad is not None:
        error_rad = float(measured_remaining_rad)
    elif final_yaw is not None:
        error_rad = normalize_angle(float(final_yaw) - float(pose.yaw_rad))
    else:
        return False, math.inf
    error_deg = abs(math.degrees(error_rad))
    return error_deg <= float(maximum_error_deg), error_deg


def _stabilize_navigation_turn_pose(
    previous_pose,
    result,
    last_command,
    turn_anchor_pose,
):
    """Keep x/y fixed while the chassis was commanded to rotate in place."""
    pure_turn = (
        last_command is not None
        and int(last_command[0]) == 0
        and int(last_command[1]) != 0
    )
    if not pure_turn:
        return result, None, 0.0
    anchor = previous_pose if turn_anchor_pose is None else turn_anchor_pose
    suppressed_m = math.hypot(
        result.pose.x_m - anchor.x_m,
        result.pose.y_m - anchor.y_m,
    )
    stabilized = replace(
        result,
        pose=Pose2D(anchor.x_m, anchor.y_m, result.pose.yaw_rad),
    )
    return stabilized, anchor, suppressed_m


def _navigation_motion_pose(previous_pose, frame_motion):
    """Propagate one pose from scan-to-scan motion without snapping to map.

    The permanent map remains the long-term reference, but repeated walls and
    glass can move its best local score between neighbouring yaw basins during
    a turn.  Frame ICP is therefore the short-term steering signal.  Grossly
    non-finite or implausibly large matches are ignored by the caller.
    """
    if (
        frame_motion is None
        or not getattr(frame_motion, "converged", False)
        or not hasattr(frame_motion, "transform")
    ):
        return None
    transform = frame_motion.transform
    values = (
        transform.x_m,
        transform.y_m,
        transform.yaw_rad,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return None
    if math.hypot(transform.x_m, transform.y_m) >= 0.60:
        return None
    if abs(normalize_angle(transform.yaw_rad)) >= math.radians(70.0):
        return None
    return previous_pose.compose(transform)


def _periodic_map_correction_due(
    now_s,
    last_correction_s,
    distance_since_correction_m,
    interval_s,
    distance_m,
    force=False,
):
    """Schedule expensive/abrupt map anchoring by time OR travelled distance."""
    return bool(
        force
        or float(now_s) - float(last_correction_s) >= float(interval_s)
        or float(distance_since_correction_m) >= float(distance_m)
    )


def _prefer_displaced_retry_choices(
    choices, pose, retry_origins, minimum_displacement_m,
):
    """Do not spend every retry from one unchanged vehicle pose.

    Queue order alone did not defer a failed target because the next nearest-
    target pass selected it again. Prefer another target until the vehicle has
    moved away from the failure pose. If every choice is waiting, allow all of
    them so this policy cannot deadlock the mission.
    """
    ready = []
    for choice in choices:
        target_id = choice[2]
        origin = retry_origins.get(target_id)
        if origin is None or math.hypot(
            pose.x_m - origin[0], pose.y_m - origin[1]
        ) >= float(minimum_displacement_m):
            ready.append(choice)
    return ready or choices


def _inspection_route_cost(pose, target, path, turn_cost_m_per_rad):
    """Score one leg using both its departure and camera-arrival headings.

    The old score compared the *current* chassis yaw directly with the final
    camera yaw.  A route could therefore look cheap while its last segment
    approached the view from the wrong side, leaving a near-180 degree pivot
    immediately before the photo.  Use the actual first and last non-trivial
    path segments and add a strong penalty above a 90 degree arrival turn.
    """
    points = [(float(pose.x_m), float(pose.y_m))]
    points.extend((float(point[0]), float(point[1])) for point in path)
    segments = []
    length_m = 0.0
    for first, second in zip(points, points[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        distance_m = math.hypot(dx, dy)
        if distance_m <= 1e-6:
            continue
        length_m += distance_m
        if distance_m >= 0.04:
            segments.append(math.atan2(dy, dx))
    departure_yaw = segments[0] if segments else float(pose.yaw_rad)
    arrival_yaw = segments[-1] if segments else float(pose.yaw_rad)
    departure_turn_rad = abs(normalize_angle(
        departure_yaw - float(pose.yaw_rad)
    ))
    arrival_turn_rad = abs(normalize_angle(
        float(target.yaw_rad) - arrival_yaw
    ))
    turn_cost = max(0.0, float(turn_cost_m_per_rad))
    uturn_excess_rad = max(0.0, arrival_turn_rad - math.radians(90.0))
    score = (
        length_m
        + turn_cost * departure_turn_rad
        + 1.35 * turn_cost * arrival_turn_rad
        + 1.25 * uturn_excess_rad
    )
    return {
        "score": score,
        "length_m": length_m,
        "departure_turn_rad": departure_turn_rad,
        "arrival_turn_rad": arrival_turn_rad,
        "departure_yaw_rad": departure_yaw,
        "arrival_yaw_rad": arrival_yaw,
    }


def _inspection_path_for_target(
    known_map,
    pose,
    target,
    turn_cost_m_per_rad,
):
    """Plan a leg whose last segment is aligned with the camera subject.

    If the ordinary shortest path would arrive facing away from the subject,
    try a small staging point behind the photo pose and then drive into the
    view along the desired camera yaw.  This turns a last-second U-turn into a
    normal path approach while retaining the same map-validated photo pose.
    """
    start_xy = (float(pose.x_m), float(pose.y_m))
    direct_path = known_map.path(start_xy, target.view_xy_m)
    if not direct_path:
        return None
    direct_metrics = _inspection_route_cost(
        pose, target, direct_path, turn_cost_m_per_rad
    )
    best = {
        "path": direct_path,
        "metrics": direct_metrics,
        "staged_camera_approach": False,
    }
    if direct_metrics["arrival_turn_rad"] <= math.radians(55.0):
        return best
    view_x, view_y = target.view_xy_m
    for staging_distance_m in (0.45, 0.35, 0.25):
        staging_xy = (
            float(view_x) - staging_distance_m * math.cos(target.yaw_rad),
            float(view_y) - staging_distance_m * math.sin(target.yaw_rad),
        )
        column, row = known_map.world_to_grid((staging_xy,))[0]
        column, row = int(column), int(row)
        if (
            not known_map.in_bounds(column, row)
            or not known_map.traversable[row, column]
        ):
            continue
        first_path = known_map.path(start_xy, staging_xy)
        final_path = known_map.path(staging_xy, target.view_xy_m)
        if not first_path or not final_path:
            continue
        final_length_m = sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(final_path, final_path[1:])
        )
        if final_length_m > staging_distance_m + 0.18:
            continue
        combined_path = list(first_path) + list(final_path[1:])
        metrics = _inspection_route_cost(
            pose, target, combined_path, turn_cost_m_per_rad
        )
        if metrics["arrival_turn_rad"] > math.radians(45.0):
            continue
        if metrics["score"] < best["metrics"]["score"]:
            best = {
                "path": combined_path,
                "metrics": metrics,
                "staged_camera_approach": True,
                "staging_xy": staging_xy,
            }
    return best


def _final_heading_timeout_seconds(base_timeout_s, initial_error_rad):
    """Give a moving final turn time proportional to its measured angle."""
    base_timeout_s = max(2.0, float(base_timeout_s))
    if initial_error_rad is None:
        return base_timeout_s
    required_s = 3.0 + abs(math.degrees(float(initial_error_rad))) / 12.0
    return min(22.0, max(base_timeout_s, required_s))


def _blend_navigation_pose(predicted_pose, mapped_pose, gain):
    """Apply a bounded map anchor without a one-frame position/yaw snap."""
    gain = max(0.0, min(1.0, float(gain)))
    yaw_error = normalize_angle(mapped_pose.yaw_rad - predicted_pose.yaw_rad)
    return Pose2D(
        predicted_pose.x_m
        + (mapped_pose.x_m - predicted_pose.x_m) * gain,
        predicted_pose.y_m
        + (mapped_pose.y_m - predicted_pose.y_m) * gain,
        normalize_angle(predicted_pose.yaw_rad + yaw_error * gain),
    )


def _precision_heading_progress(
    angular_mrad_s,
    frame_motion,
    pulse_ttl_ms,
    elapsed_s,
    fallback_response_gain=0.75,
):
    """Measure one final-camera turn without trusting absolute map yaw.

    The first desired angle is computed from the already-localized pose.  Each
    following scan subtracts only credible scan-to-scan yaw.  An invalid ICP
    frame gets a deliberately conservative command-time estimate so repeated
    glass dropouts cannot cause unlimited full-circle turning.
    """
    angular_mrad_s = int(angular_mrad_s)
    if angular_mrad_s == 0:
        return 0.0, "stationary"
    direction = _direction(angular_mrad_s)
    active_s = min(
        max(0.05, float(elapsed_s)),
        max(0.05, float(pulse_ttl_ms) / 1000.0 + 0.08),
    )
    physical_limit_rad = (
        abs(angular_mrad_s) / 1000.0 * active_s * 1.55
        + math.radians(4.0)
    )
    if (
        frame_motion is not None
        and getattr(frame_motion, "converged", False)
        and math.isfinite(float(getattr(frame_motion, "rmse_m", math.inf)))
        and hasattr(frame_motion, "transform")
    ):
        transform = frame_motion.transform
        translation_m = math.hypot(transform.x_m, transform.y_m)
        yaw_rad = normalize_angle(transform.yaw_rad)
        same_direction = (
            abs(yaw_rad) <= math.radians(2.0)
            or _direction(yaw_rad) == direction
        )
        if (
            translation_m <= 0.10
            and same_direction
            and abs(yaw_rad) <= physical_limit_rad
        ):
            return yaw_rad, "scan_motion"
    # A command is not motion evidence: a wheel can stall or slip while the
    # controller accepts it. Never declare a camera turn complete unless the
    # lidar measured the chassis rotation.
    return 0.0, "unverified_no_progress"


def _circular_heading_seed(samples_rad):
    """Average stopped map-heading samples without a +/-pi discontinuity."""
    samples = [
        normalize_angle(float(value))
        for value in samples_rad
        if math.isfinite(float(value))
    ]
    if not samples:
        return None
    sine = sum(math.sin(value) for value in samples)
    cosine = sum(math.cos(value) for value in samples)
    if abs(sine) < 1e-9 and abs(cosine) < 1e-9:
        return samples[-1]
    return normalize_angle(math.atan2(sine, cosine))


def _navigation_pose_jump_reason(
    previous_pose,
    result,
    last_command,
    elapsed_s,
    turn_ttl_ms,
    args,
):
    """Reject only pose changes that contradict the bounded chassis command.

    Glass and repeated wall geometry can give a locally good map score in the
    wrong heading basin.  This guard deliberately does not inspect ordinary
    small drift or re-run global localization: it holds the last trusted pose
    for one stopped scan only when the proposed motion is physically
    inconsistent with the command that could have happened since that pose.
    """
    delta = previous_pose.inverse().compose(result.pose)
    translation_m = math.hypot(delta.x_m, delta.y_m)
    yaw_delta = normalize_angle(delta.yaw_rad)
    linear = int(last_command[0]) if last_command is not None else 0
    angular = int(last_command[1]) if last_command is not None else 0
    if linear == 0 and angular == 0:
        if translation_m > float(
            args.navigation_pose_stationary_translation_limit
        ):
            return "stationary_translation_jump"
        if abs(math.degrees(yaw_delta)) > float(
            args.navigation_pose_stationary_yaw_limit_deg
        ):
            return "stationary_yaw_jump"
        return ""

    direction_slack = math.radians(
        float(args.navigation_pose_direction_slack_deg)
    )
    if (
        linear == 0
        and angular != 0
        and abs(yaw_delta) > direction_slack
        and _direction(yaw_delta) != _direction(angular)
    ):
        return "turn_direction_contradiction"

    # A pure turn is sent as one deliberately expiring TTL pulse.  Include a
    # small coast allowance, but do not let a slow scan cycle justify an
    # arbitrary 60-90 degree map-matching jump.
    if linear != 0:
        # Straight/arc commands are kept alive while scan matching runs.
        active_s = min(2.0, max(0.05, float(elapsed_s)))
    else:
        active_s = min(
            max(0.05, float(elapsed_s)),
            max(0.05, float(turn_ttl_ms) / 1000.0 + 0.08),
        )
    yaw_limit = (
        math.radians(8.0)
        + abs(angular) / 1000.0 * active_s * 1.10
    )
    translation_limit = (
        0.08 + abs(linear) / 1000.0 * active_s * 1.40
    )
    if linear == 0:
        translation_limit = max(
            translation_limit,
            float(args.navigation_pose_stationary_translation_limit),
        )
        # The scan matcher searches only a bounded neighbouring heading.
        # A larger result is either an over-long motor pulse or a repeated-
        # wall alias; neither may become the next steering origin.  The new
        # 200--260 ms turn pulses keep genuine increments below this cap.
        yaw_limit = min(yaw_limit, math.radians(40.0))
    else:
        # Waypoint tracking only advances when its starting heading error is
        # already small.  A 45--90 degree yaw change during one forward frame
        # was seen repeatedly in the field log and is scan aliasing, not the
        # gentle steering command (normally below 0.65 rad/s).
        yaw_limit = min(yaw_limit, math.radians(28.0))
        translation_limit = min(translation_limit, 0.50)
    # Weak glass-affected matches are allowed when they stay near the
    # trajectory, but may not use the whole physical envelope for a large
    # correction.
    if (
        float(result.rmse_m) > 0.12
        or float(result.inlier_ratio) < 0.42
    ):
        yaw_limit = min(yaw_limit, math.radians(35.0))
        translation_limit = min(translation_limit, 0.22)
    if translation_m > translation_limit:
        return "translation_exceeds_command_envelope"
    if abs(yaw_delta) > yaw_limit:
        return "yaw_exceeds_command_envelope"
    return ""


def _navigation_recovery_candidate_usable(trusted_pose, result, args):
    """Accept a stopped re-anchor candidate only inside a strong local basin."""
    if (
        not math.isfinite(float(result.rmse_m))
        or float(result.rmse_m) > float(args.navigation_recovery_max_rmse)
        or float(result.inlier_ratio)
        < float(args.navigation_recovery_min_inlier_ratio)
    ):
        return False
    delta = trusted_pose.inverse().compose(result.pose)
    return (
        math.hypot(delta.x_m, delta.y_m)
        <= float(args.navigation_pose_stationary_translation_limit)
        and abs(math.degrees(delta.yaw_rad))
        <= float(args.navigation_pose_stationary_yaw_limit_deg)
    )


def _navigation_recovery_candidates_consistent(
    candidates,
    required_frames=2,
    position_tolerance_m=0.08,
    yaw_tolerance_deg=6.0,
):
    """Require several re-anchored scans to agree before motion resumes."""
    required_frames = max(1, int(required_frames))
    if len(candidates) < required_frames:
        return False
    recent = list(candidates)[-required_frames:]
    anchor = recent[-1].pose
    return all(
        math.hypot(
            item.pose.x_m - anchor.x_m,
            item.pose.y_m - anchor.y_m,
        ) <= float(position_tolerance_m)
        and abs(normalize_angle(
            item.pose.yaw_rad - anchor.yaw_rad
        )) <= math.radians(float(yaw_tolerance_deg))
        for item in recent
    )


def _navigation_recovery_candidate_bounded(trusted_pose, result, args):
    """Keep even glass-weakened candidates when their correction is small."""
    delta = trusted_pose.inverse().compose(result.pose)
    return (
        math.hypot(delta.x_m, delta.y_m)
        <= float(args.navigation_pose_stationary_translation_limit)
        and abs(math.degrees(delta.yaw_rad))
        <= float(args.navigation_pose_stationary_yaw_limit_deg)
    )


def _best_bounded_recovery_candidate(trusted_pose, candidates):
    """Prefer map support, then the smallest change from the trusted pose."""
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item.inlier_ratio),
            -float(item.rmse_m),
            -math.hypot(
                item.pose.x_m - trusted_pose.x_m,
                item.pose.y_m - trusted_pose.y_m,
            ),
            -abs(normalize_angle(
                item.pose.yaw_rad - trusted_pose.yaw_rad
            )),
        ),
    )


def _inspection_target_visible_from(known_map, view_xy, target_xy):
    """Check a camera ray while allowing its final cells to enter the target."""
    start = np.asarray(view_xy, dtype=np.float64)
    end = np.asarray(target_xy, dtype=np.float64)
    distance_m = float(np.linalg.norm(end - start))
    if not math.isfinite(distance_m) or distance_m < 0.45:
        return False
    sample_end_m = max(0.0, distance_m - max(0.18, known_map.resolution_m * 3.0))
    sample_count = max(2, int(math.ceil(sample_end_m / (
        known_map.resolution_m * 0.5
    ))))
    for travelled_m in np.linspace(0.0, sample_end_m, sample_count):
        point = start + (end - start) * (travelled_m / distance_m)
        column, row = known_map.world_to_grid((point,))[0]
        column, row = int(column), int(row)
        if (
            not known_map.in_bounds(column, row)
            or known_map.occupied[row, column]
            or known_map.unknown[row, column]
            or known_map.temporary_occupied[row, column]
        ):
            return False
    return True


def _alternate_inspection_target(
    known_map,
    pose,
    target,
    max_shift_m=1.0,
    excluded_views=(),
):
    """Find one reachable nearby view of the same mapped subject.

    This is not a photograph of a merely similar object elsewhere: the
    replacement must retain direct map line-of-sight to the original target
    cell and stay close to the originally planned camera position.
    """
    target_xy = np.asarray(target.target_xy_m, dtype=np.float64)
    original_view = np.asarray(target.view_xy_m, dtype=np.float64)
    original_vector = original_view - target_xy
    original_radius = float(np.linalg.norm(original_vector))
    if original_radius <= 0.0:
        return None
    original_angle = math.atan2(original_vector[1], original_vector[0])
    excluded_views = [
        np.asarray(view, dtype=np.float64) for view in excluded_views
    ]
    candidate_rows_columns = np.argwhere(known_map.traversable)
    geometric_candidates = []
    for row, column in candidate_rows_columns:
        candidate = known_map.grid_to_world(((int(column), int(row)),))[0]
        if any(
            float(np.linalg.norm(candidate - old_view)) < 0.16
            for old_view in excluded_views
        ):
            continue
        shift_m = float(np.linalg.norm(candidate - original_view))
        if shift_m < 0.18 or shift_m > float(max_shift_m):
            continue
        vector = candidate - target_xy
        radius_m = float(np.linalg.norm(vector))
        if not max(0.48, original_radius - 0.18) <= radius_m <= original_radius + 0.38:
            continue
        angle_error = abs(normalize_angle(
            math.atan2(vector[1], vector[0]) - original_angle
        ))
        if angle_error > math.radians(75.0):
            continue
        candidate_xy = (float(candidate[0]), float(candidate[1]))
        geometric_score = (
            0.80 * shift_m + 0.45 * angle_error
            + 0.50 * abs(radius_m - original_radius)
        )
        geometric_candidates.append((geometric_score, candidate_xy))
    # A* is intentionally limited to the best nearby geometrical candidates;
    # running it for every free map cell can otherwise pause a Raspberry Pi
    # for tens of seconds exactly when an obstacle has already been detected.
    ranked = []
    for geometric_score, candidate_xy in sorted(
        geometric_candidates, key=lambda item: item[0]
    )[:48]:
        if not _inspection_target_visible_from(
            known_map, candidate_xy, target.target_xy_m
        ):
            continue
        path = known_map.path((pose.x_m, pose.y_m), candidate_xy)
        if not path:
            continue
        path_length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(path, path[1:])
        )
        ranked.append((path_length + geometric_score, candidate_xy))
    if not ranked:
        return None
    _score, view_xy = min(ranked, key=lambda item: item[0])
    yaw_rad = math.atan2(
        target.target_xy_m[1] - view_xy[1],
        target.target_xy_m[0] - view_xy[0],
    )
    return replace(target, view_xy_m=view_xy, yaw_rad=yaw_rad)


def _first_trajectory_pose(map_path):
    path = Path(map_path)
    trajectory = path.with_name(path.stem + "_trajectory.csv")
    if not trajectory.exists():
        return None
    with trajectory.open("r", encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None:
        return None
    try:
        return Pose2D(
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["yaw_rad"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _direction(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def _start_motion_keepalive(client, owns_client):
    """Start the independent drive refresher for a directly-owned client.

    Map matching may take longer than the ESP32's 700 ms safety TTL.  Without
    this refresher, a waypoint command becomes a short motor pulse followed by
    a long stop.  Injected clients remain under their caller's lifecycle.
    """
    if not owns_client:
        return False
    start = getattr(client, "start", None)
    if start is None:
        return False
    start()
    return bool(getattr(client, "maintenance_running", True))


def _append_navigation_diagnostic(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _timed_twist(client, linear, angular, duration_s, ttl_ms):
    """Execute a bounded motion while refreshing a short safety TTL."""
    deadline = time.monotonic() + max(0.0, float(duration_s))
    client.set_twist(linear, angular, ttl_ms=ttl_ms)
    # Esp32Client's maintenance thread refreshes the acknowledged command
    # independently of slow lidar processing.  Do not create a second BLE
    # refresh stream when it is already running; duplicated writes used to
    # increase link contention and made motion visibly pulse.
    if getattr(client, "maintenance_running", False):
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return
    refresh = getattr(client, "refresh_twist", None)
    interval_s = min(0.35, max(0.10, ttl_ms / 1000.0 * 0.50))
    while time.monotonic() + interval_s < deadline:
        time.sleep(interval_s)
        if refresh is not None:
            refresh(linear, angular, ttl_ms=ttl_ms)
        else:
            client.set_twist(linear, angular, ttl_ms=ttl_ms)
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


def _scan_change_m(scan_a, scan_b, bin_size_deg=2.0):
    """Median range-profile change shared by two complete lidar scans."""
    bin_count = max(36, int(round(360.0 / bin_size_deg)))
    profiles = []
    for scan in (scan_a, scan_b):
        profile = np.full(bin_count, np.nan, dtype=np.float64)
        angles = np.mod(np.degrees(scan.angles_rad), 360.0)
        valid = (
            np.isfinite(angles)
            & np.isfinite(scan.distances_m)
            & (scan.distances_m >= 0.12)
            & (scan.distances_m <= 8.0)
        )
        indices = np.floor(angles[valid] / bin_size_deg).astype(int) % bin_count
        distances = scan.distances_m[valid]
        for index in np.unique(indices):
            profile[index] = float(np.median(distances[indices == index]))
        profiles.append(profile)
    shared = np.isfinite(profiles[0]) & np.isfinite(profiles[1])
    if np.count_nonzero(shared) < 30:
        return 0.0
    differences = np.abs(profiles[0][shared] - profiles[1][shared])
    # Ignore the noisiest 15% of rays (glass and moving people), while a real
    # chassis movement changes a broad part of the 360-degree profile.
    keep = max(1, int(math.floor(len(differences) * 0.85)))
    return float(np.median(np.sort(differences)[:keep]))


def _motion_update(match, duration_s):
    translation = math.hypot(match.transform.x_m, match.transform.y_m)
    duration_s = max(0.05, float(duration_s))
    return SlamUpdate(
        pose=match.transform,
        accepted=bool(match.converged and math.isfinite(match.rmse_m)),
        rmse_m=float(match.rmse_m),
        correspondences=int(match.correspondences),
        scan_points=max(1, int(match.correspondences)),
        inlier_ratio=float(match.inlier_ratio),
        translation_m=translation,
        rotation_rad=float(match.transform.yaw_rad),
        rejection_reason="" if match.converged else "scan_motion_match_failed",
        linear_speed_m_s=translation / duration_s,
        angular_speed_rad_s=abs(match.transform.yaw_rad) / duration_s,
    )


def _tracking_update(previous_pose, result, duration_s):
    """Translate a finite post-startup pose into motion feedback.

    Navigation has already committed to the strictly verified global map
    basin, so result quality fields are observational here and do not reject
    the measured chassis motion used for adaptive torque.
    """
    delta = previous_pose.inverse().compose(result.pose)
    translation = math.hypot(delta.x_m, delta.y_m)
    duration_s = max(0.05, float(duration_s))
    return SlamUpdate(
        pose=result.pose,
        accepted=True,
        rmse_m=float(result.rmse_m),
        correspondences=max(1, int(round(result.inlier_ratio * 100.0))),
        scan_points=100,
        inlier_ratio=float(result.inlier_ratio),
        translation_m=translation,
        rotation_rad=float(delta.yaw_rad),
        rejection_reason="",
        linear_speed_m_s=translation / duration_s,
        angular_speed_rad_s=abs(delta.yaw_rad) / duration_s,
    )


def _tracking_usable_for_photo_mission(result, args):
    """Relax tracking only after strict active startup localization passed."""
    return (
        result is not None
        and math.isfinite(result.rmse_m)
        and result.rmse_m <= args.relaxed_tracking_rmse
        and result.inlier_ratio >= args.relaxed_tracking_inlier_ratio
    )


def _tracking_pose_available(result):
    """Accept every finite tracking pose after strict startup localization.

    RMSE and inlier ratio remain in diagnostics, but they no longer veto a
    navigation pose or a photograph after the correct map basin is known.
    """
    if result is None or getattr(result, "pose", None) is None:
        return False
    pose = result.pose
    return all(math.isfinite(value) for value in (
        pose.x_m,
        pose.y_m,
        pose.yaw_rad,
    ))


def _is_turn_motion_stalled(command, match, speed_config):
    """Require credible lidar evidence that a requested pure turn did nothing."""
    if command.linear_mm_s != 0 or command.angular_mrad_s == 0:
        return False
    if (
        match is None
        or not match.converged
        or not math.isfinite(match.rmse_m)
    ):
        return False
    translation_m = math.hypot(
        match.transform.x_m,
        match.transform.y_m,
    )
    rotation_deg = abs(math.degrees(match.transform.yaw_rad))
    return (
        translation_m < speed_config.low_frame_translation_m
        and rotation_deg < speed_config.low_frame_rotation_deg
    )


def _turn_is_at_adaptive_max(command, controller_diagnostics):
    """Return true only when both the learned floor and applied turn hit max."""
    maximum = int(
        controller_diagnostics["maximum_turn_speed_mrad_s"]
    )
    tolerance = max(1, int(round(maximum * 0.02)))
    return (
        command.linear_mm_s == 0
        and abs(command.angular_mrad_s) >= maximum - tolerance
        and int(controller_diagnostics["turn_floor_mrad_s"])
        >= maximum - tolerance
    )


def _navigation_turn_stall_observed(
    previous_command,
    measured_yaw_delta_rad,
    current_command,
    controller_diagnostics,
    speed_config,
):
    """Return true for one credible no-yaw frame at maximum turn output."""
    previous_was_pure_turn = (
        previous_command is not None
        and int(previous_command[0]) == 0
        and int(previous_command[1]) != 0
    )
    return bool(
        previous_was_pure_turn
        and abs(float(measured_yaw_delta_rad))
        < math.radians(float(speed_config.low_frame_rotation_deg))
        and _turn_is_at_adaptive_max(
            current_command,
            controller_diagnostics,
        )
    )


def _navigation_linear_stall_observed(
    previous_command,
    motion_feedback,
    current_command,
    controller_diagnostics,
    maximum_linear_speed_mm_s,
    speed_config,
):
    """Detect a loaded/jammed chassis that max drive output cannot move.

    This deliberately uses scan-to-scan chassis motion rather than the
    periodically corrected global pose.  A glass-wall map correction can
    move the estimated pose by centimetres while the tyres are stationary;
    counting that as progress caused field run 20260915_142137 to issue the
    same 450 mm/s forward arc for minutes without entering any escape state.
    """
    previous_was_forward = (
        previous_command is not None
        and int(previous_command[0]) > 0
    )
    if (
        not previous_was_forward
        or motion_feedback is None
        or not bool(getattr(motion_feedback, "accepted", False))
        or int(current_command.linear_mm_s) <= 0
    ):
        return False
    maximum = max(1, int(maximum_linear_speed_mm_s))
    tolerance = max(2, int(round(maximum * 0.02)))
    maximum_boost = max(
        0, int(speed_config.maximum_linear_boost_mm_s)
    )
    boost_tolerance = max(2, int(round(maximum_boost * 0.03)))
    learned_at_ceiling = (
        maximum_boost > 0
        and int(controller_diagnostics["linear_adjustment_mm_s"])
        >= maximum_boost - boost_tolerance
    )
    return bool(
        (
            int(current_command.linear_mm_s) >= maximum - tolerance
            or learned_at_ceiling
        )
        and float(getattr(motion_feedback, "translation_m", math.inf))
        < float(speed_config.low_frame_translation_m)
    )


def _turn_escape_is_safe(scan, args):
    """Check only the explicitly supported forward escape direction."""
    angles = np.asarray(scan.angles_rad, dtype=np.float64)
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    relative = (angles + math.pi) % (2.0 * math.pi) - math.pi
    supported = (
        (np.abs(relative) <= math.radians(21.0))
        & np.isfinite(distances)
        & (distances >= 0.08)
        & (distances <= 8.0)
    )
    front_clearance_m = sector_min_distance(
        scan,
        center_deg=0.0,
        width_deg=42.0,
        min_distance_m=0.08,
        max_distance_m=8.0,
    )
    # Preserve the ordinary stop envelope after the short move and retain a
    # small scan/noise reserve. No reverse fallback is allowed here.
    required_clearance_m = max(
        0.50,
        float(args.stop_distance)
        + float(args.relocalization_turn_escape_distance)
        + 0.05,
    )
    return (
        int(np.count_nonzero(supported)) >= 3
        and front_clearance_m >= required_clearance_m,
        front_clearance_m,
        required_clearance_m,
    )


def _collision_reverse_is_safe(scan, pose, known_map, args):
    """Require robust rear lidar and permanent-map evidence before reversing.

    A single 8cm N10 reflection used to veto every escape even when the rest
    of the rear sector was open.  Use a low percentile plus a small hard-hit
    cluster: isolated speckles are ignored, while a real rear wall still
    occupies enough adjacent rays to block motion.
    """
    angles = np.asarray(scan.angles_rad, dtype=np.float64)
    distances = np.asarray(scan.distances_m, dtype=np.float64)
    relative_rear = (angles - math.pi + math.pi) % (2.0 * math.pi) - math.pi
    # N10 installations on this chassis repeatedly report a broad 8--10cm
    # rear return from the body/cable bundle.  It cannot be a usable external
    # obstacle distance for a reverse manoeuvre and previously cancelled all
    # escape pulses.  Ignore that self-shadow band, while a real cluster from
    # 14cm onward still vetoes reversing.
    supported = (
        (np.abs(relative_rear) <= math.radians(25.0))
        & np.isfinite(distances)
        & (distances >= 0.14)
        & (distances <= 8.0)
    )
    rear_values = distances[supported]
    rear_clearance_m = (
        float(np.quantile(rear_values, 0.15))
        if len(rear_values) >= 3 else math.inf
    )
    hard_rear_hits = int(np.count_nonzero(rear_values <= 0.20))
    required_clearance_m = max(
        float(args.collision_reverse_rear_clearance),
        float(args.collision_reverse_distance) + 0.18,
    )
    reverse_distance_m = float(args.collision_reverse_distance)
    if args.collision_reverse_map_fallback:
        reverse_distance_m = min(
            reverse_distance_m,
            float(args.collision_reverse_map_fallback_max_distance),
        )
    endpoint = (
        pose.x_m - reverse_distance_m * math.cos(pose.yaw_rad),
        pose.y_m - reverse_distance_m * math.sin(pose.yaw_rad),
    )
    map_clear = known_map.line_is_clear(
        (pose.x_m, pose.y_m),
        endpoint,
    )
    support_count = int(np.count_nonzero(supported))
    safe = (
        support_count >= 3
        and hard_rear_hits < 3
        and rear_clearance_m >= required_clearance_m
        and map_clear
    )
    reason = ""
    if support_count < 3:
        # Missing rear measurements are common behind the chassis/cable
        # bundle and are not themselves obstacle evidence. Only an explicit
        # <=8cm map fallback may proceed, and only when the clearance-inflated
        # map covers the complete reverse segment with no credible close hit.
        if (
            args.collision_reverse_map_fallback
            and hard_rear_hits == 0
            and map_clear
        ):
            safe = True
            reason = "map_only_rear_lidar_fallback"
        elif not map_clear:
            reason = "reverse_map_corridor_blocked"
        elif hard_rear_hits > 0:
            reason = "rear_sparse_close_return"
        else:
            reason = "rear_lidar_support_low"
    elif hard_rear_hits >= 3:
        reason = "rear_hard_obstacle_cluster"
    elif rear_clearance_m < required_clearance_m:
        reason = "rear_clearance_low"
    elif not map_clear:
        reason = "reverse_map_corridor_blocked"
    return safe, rear_clearance_m, required_clearance_m, endpoint, reason


def _execute_collision_reverse_escape(
    client,
    scan_iterator,
    initial_scan,
    pose,
    known_map,
    args,
    localizer=None,
):
    """Back away in stopped, rescanned pulses and verify physical motion."""
    scan = initial_scan
    target_distance_m = float(args.collision_reverse_distance)
    if args.collision_reverse_map_fallback:
        target_distance_m = min(
            target_distance_m,
            float(args.collision_reverse_map_fallback_max_distance),
        )
    speed_mm_s = abs(int(args.collision_reverse_speed))
    max_steps = int(args.collision_reverse_max_steps)
    pulse_distance_m = target_distance_m / max_steps
    # ESP32 linear acceleration is 650 mm/s². A 3 cm / 240 mm/s geometric
    # pulse lasts only 125 ms and expires exactly as the motor output reaches
    # its effective threshold. Give each stopped reverse pulse enough ramp
    # area to create measurable motion; lidar still checks every segment.
    pulse_duration_s = max(
        0.28,
        pulse_distance_m / (speed_mm_s / 1000.0),
    )
    success_distance_m = max(0.04, target_distance_m * 0.75)
    travelled_m = 0.0
    failure_reason = "reverse_step_limit"
    try:
        for step in range(max_steps):
            safe, rear_m, required_m, _endpoint, reason = (
                _collision_reverse_is_safe(
                    scan, pose, known_map, args
                )
            )
            if not safe:
                failure_reason = reason
                print(
                    "紧急后退取消/中止：后方净空={}，要求 {:.2f}m，"
                    "原因={}；车辆保持停车。".format(
                        "无有效数据" if not math.isfinite(rear_m)
                        else "{:.2f}m".format(rear_m),
                        required_m,
                        reason,
                    ),
                    flush=True,
                )
                break
            if reason == "map_only_rear_lidar_fallback" and step == 0:
                print(
                    "后方雷达数据不足，但永久地图确认后退走廊安全；"
                    "启用地图兜底，最多低速短退 {:.2f}m。".format(
                        target_distance_m
                    ),
                    flush=True,
                )
            ttl_ms = min(350, max(200, int(args.command_ttl_ms)))
            before_scan = scan
            _timed_twist(
                client,
                -speed_mm_s,
                0,
                pulse_duration_s,
                ttl_ms,
            )
            client.stop()
            scan = next(scan_iterator)
            measured_step_m = 0.0
            motion_verified = False
            if localizer is not None:
                reverse_match = localizer.relative_motion(
                    before_scan, scan
                )
                if (
                    reverse_match.converged
                    and math.isfinite(reverse_match.rmse_m)
                ):
                    measured_step_m = math.hypot(
                        reverse_match.transform.x_m,
                        reverse_match.transform.y_m,
                    )
                    motion_verified = (
                        measured_step_m
                        >= max(0.008, pulse_distance_m * 0.30)
                    )
            else:
                # Compatibility for callers without a scan matcher.  The
                # broad range-profile check rejects a buzzing/stationary
                # chassis without trusting one glass or multipath ray.
                measured_step_m = _scan_change_m(before_scan, scan)
                motion_verified = measured_step_m >= 0.012
            if not motion_verified:
                failure_reason = "reverse_no_measured_motion"
                print(
                    "紧急后退第 {}/{} 段未测得车身移动（{:.3f}m）；"
                    "不再把指令时长当作真实位移，车辆保持停车。".format(
                        step + 1, max_steps, measured_step_m
                    ),
                    flush=True,
                )
                break
            travelled_m = min(
                target_distance_m,
                travelled_m + min(pulse_distance_m, measured_step_m),
            )
            print(
                "紧急后退：第 {}/{} 段完成，累计约 {:.3f}m；"
                "停车后重新检查后方。".format(
                    step + 1, max_steps, travelled_m
                ),
                flush=True,
            )
            if travelled_m >= success_distance_m:
                return scan, True, travelled_m, ""
    finally:
        client.stop()
    return scan, False, travelled_m, failure_reason


def _turn_learning_step_budget(speed_controller):
    """Bound a turn cycle while allowing any configured step to reach max."""
    diagnostics = speed_controller.diagnostics()
    floor = int(diagnostics["turn_floor_mrad_s"])
    limit = int(diagnostics["turn_limit_mrad_s"])
    maximum = int(diagnostics["maximum_turn_speed_mrad_s"])
    events = int(diagnostics["turn_stall_events"])
    budget = 0
    while floor < maximum and budget < 1000:
        budget += max(
            1,
            int(speed_controller.config.stalled_frames_before_increase)
            - events,
        )
        multiplier = min(
            int(speed_controller.config.maximum_turn_stall_step_multiplier),
            events + 1,
        )
        increase = max(
            1,
            int(speed_controller.config.turn_adjust_step_mrad_s)
            * multiplier,
        )
        floor = min(limit, floor + increase)
        limit = min(maximum, limit + increase)
        events += 1
    return budget


def _execute_localization_turn_escape(
    client,
    localizer,
    speed_controller,
    initial_scan,
    args,
    deadline_s,
    measured_step,
    label="主动定位",
):
    """Use a short, lidar-measured forward move to break turn stiction."""
    safe, front_clearance_m, required_clearance_m = (
        _turn_escape_is_safe(initial_scan, args)
    )
    if not safe:
        client.stop()
        speed_controller.stopped()
        if math.isfinite(front_clearance_m):
            print(
                "{}脱困取消：前方净空仅 {:.2f}m，小于安全距离 "
                "{:.2f}m，禁止前冲。".format(
                    label,
                    front_clearance_m,
                    required_clearance_m,
                ),
                flush=True,
            )
        else:
            print(
                "{}脱困取消：雷达前方有效点不足，无法确认安全"
                "净空，禁止前冲。".format(label),
                flush=True,
            )
        return initial_scan, False, 0.0, "unsafe_forward_clearance"

    target_distance_m = float(
        args.relocalization_turn_escape_distance
    )
    print(
        "{}脱困：前方净空 {:.2f}m，执行 turn_escape，"
        "目标位移 {:.2f}m。".format(
            label,
            front_clearance_m,
            target_distance_m,
        ),
        flush=True,
    )
    baseline = initial_scan
    scan = initial_scan
    total_distance_m = 0.0
    failure_reason = "escape_step_limit"
    succeeded = False
    try:
        for _step in range(args.relocalization_turn_escape_max_steps):
            if time.monotonic() >= deadline_s:
                failure_reason = "localization_deadline"
                break
            safe, front_clearance_m, required_clearance_m = (
                _turn_escape_is_safe(scan, args)
            )
            if not safe:
                failure_reason = "forward_clearance_lost"
                print(
                    "{}脱困中止：前方净空降至 {:.2f}m，"
                    "安全距离 {:.2f}m。".format(
                        label,
                        front_clearance_m,
                        required_clearance_m,
                    ),
                    flush=True,
                )
                break
            _applied, _step_match, scan = measured_step(MotionCommand(
                args.relocalization_speed,
                0,
                "turn_escape",
                "moving forward to break stationary-turn stiction",
            ))
            total_match = localizer.relative_motion(baseline, scan)
            if total_match.converged and math.isfinite(total_match.rmse_m):
                total_distance_m = math.hypot(
                    total_match.transform.x_m,
                    total_match.transform.y_m,
                )
            print(
                "{}脱困进度：累计位移={:.3f}m / {:.3f}m。".format(
                    label,
                    total_distance_m,
                    target_distance_m,
                ),
                flush=True,
            )
            if total_distance_m >= target_distance_m:
                succeeded = True
                failure_reason = ""
                break
    finally:
        client.stop()
        speed_controller.stopped()

    if succeeded:
        print(
            "{}脱困完成：累计位移={:.3f}m，重新尝试原地转向。".format(
                label,
                total_distance_m
            ),
            flush=True,
        )
    else:
        print(
            "{}脱困失败：累计位移={:.3f}m，原因={}。".format(
                label,
                total_distance_m,
                failure_reason,
            ),
            flush=True,
        )
    return scan, succeeded, total_distance_m, failure_reason


def _execute_navigation_turn_escape(
    client,
    scan_iterator,
    localizer,
    speed_controller,
    initial_scan,
    args,
    deadline_s,
):
    """Run the shared 5 cm lidar-measured escape during photo navigation."""
    scan = initial_scan
    last_applied = None

    def measured_step(requested):
        nonlocal scan
        nonlocal last_applied
        applied = speed_controller.apply(requested)
        command = (applied.linear_mm_s, applied.angular_mrad_s)
        ttl_ms = speed_controller.ttl_for(applied, args.command_ttl_ms)
        if command == last_applied:
            refresh = getattr(client, "refresh_twist", None)
            if refresh is not None:
                refresh(*command, ttl_ms=ttl_ms)
            else:
                client.set_twist(*command, ttl_ms=ttl_ms)
        else:
            client.set_twist(*command, ttl_ms=ttl_ms)
        last_applied = command
        before = scan
        started_s = time.monotonic()
        scan = next(scan_iterator)
        duration_s = max(0.05, time.monotonic() - started_s)
        match = localizer.relative_motion(before, scan)
        speed_controller.observe(_motion_update(match, duration_s))
        return applied, match, scan

    return _execute_localization_turn_escape(
        client,
        localizer,
        speed_controller,
        scan,
        args,
        deadline_s,
        measured_step,
        label="巡检导航转向",
    )


def _execute_active_localization_turn(
    client,
    localizer,
    speed_controller,
    initial_scan,
    angular,
    reason,
    required_turn_rad,
    args,
    deadline_s,
    measured_step,
):
    """Bounded TURNING -> STALL_AT_MAX -> ESCAPE -> RETRY state loop."""
    scan = initial_scan

    # Compatibility for object_search_mission, which imports this active-move
    # helper but intentionally keeps its existing controller/mission defaults.
    if not hasattr(args, "relocalization_turn_escape_distance"):
        turn_baseline = scan
        for _attempt in range(10):
            _applied, _match, scan = measured_step(MotionCommand(
                0, angular, "active_localization_turn", reason
            ))
            total_turn = localizer.relative_motion(turn_baseline, scan)
            if (
                total_turn.converged
                and abs(total_turn.transform.yaw_rad)
                >= required_turn_rad - math.radians(3.0)
            ):
                break
            if time.monotonic() >= deadline_s:
                break
        return scan

    stall_frames_required = max(
        1, int(args.relocalization_turn_stall_at_max_frames)
    )
    maximum_escape_attempts = max(
        0, int(args.relocalization_turn_escape_max_attempts)
    )
    escape_attempts = 0
    turn_direction = 1 if angular >= 0 else -1

    while time.monotonic() < deadline_s:
        turn_step_limit = max(
            10,
            _turn_learning_step_budget(speed_controller)
            + stall_frames_required
            + 1,
        )
        turn_baseline = scan
        stalled_at_max_frames = 0
        reached_stall_at_max = False
        last_learning_floor = None
        print(
            "主动定位转向状态=TURNING，方向={}，脱困次数={}/{}。".format(
                "left" if turn_direction > 0 else "right",
                escape_attempts,
                maximum_escape_attempts,
            ),
            flush=True,
        )
        for _step in range(turn_step_limit):
            requested = MotionCommand(
                0,
                turn_direction * abs(int(angular)),
                "active_localization_turn",
                reason,
            )
            applied, match, scan = measured_step(requested)
            total_turn = localizer.relative_motion(turn_baseline, scan)
            if (
                total_turn.converged
                and abs(total_turn.transform.yaw_rad)
                >= required_turn_rad - math.radians(3.0)
            ):
                return scan

            diagnostics = speed_controller.diagnostics()
            motion_stalled = _is_turn_motion_stalled(
                applied, match, speed_controller.config
            )
            at_adaptive_max = _turn_is_at_adaptive_max(
                applied, diagnostics
            )
            if (
                motion_stalled
                and diagnostics["turn_stall_events"] > 0
                and not at_adaptive_max
                and diagnostics["turn_floor_mrad_s"]
                != last_learning_floor
            ):
                last_learning_floor = diagnostics[
                    "turn_floor_mrad_s"
                ]
                print(
                    "主动定位转向状态=TURN_STALL_LEARNING，"
                    "turn_floor={}mrad/s，turn_limit={}mrad/s。".format(
                        diagnostics["turn_floor_mrad_s"],
                        diagnostics["turn_limit_mrad_s"],
                    ),
                    flush=True,
                )
            if motion_stalled and at_adaptive_max:
                stalled_at_max_frames += 1
            else:
                stalled_at_max_frames = 0
            if stalled_at_max_frames >= stall_frames_required:
                reached_stall_at_max = True
                break
            if time.monotonic() >= deadline_s:
                break

        if not reached_stall_at_max:
            client.stop()
            speed_controller.stopped()
            print(
                "主动定位转向未在有限步数内取得可靠 yaw；已停车并退回"
                "静止定位，本次转向不再中止整圈巡检。",
                flush=True,
            )
            return scan

        diagnostics = speed_controller.diagnostics()
        print(
            "主动定位：原地转向达到最大自适应输出 "
            "{}mrad/s，连续 {} 帧实际 yaw < {:.1f}° 且位移 < "
            "{:.3f}m，状态=TURN_STALL_AT_MAX。".format(
                diagnostics["maximum_turn_speed_mrad_s"],
                stalled_at_max_frames,
                speed_controller.config.low_frame_rotation_deg,
                speed_controller.config.low_frame_translation_m,
            ),
            flush=True,
        )
        client.stop()
        speed_controller.stopped()
        if escape_attempts >= maximum_escape_attempts:
            print(
                "主动定位原地转向与 turn_escape 均未取得可靠运动；"
                "已停车并退回静止定位，本次失败不再中止整圈巡检。",
                flush=True,
            )
            return scan

        escape_attempts += 1
        scan, escaped, _distance, failure_reason = (
            _execute_localization_turn_escape(
                client,
                localizer,
                speed_controller,
                scan,
                args,
                deadline_s,
                measured_step,
            )
        )
        if escaped:
            print(
                "主动定位转向状态=RETRY_TURN，保持原转向方向。",
                flush=True,
            )
            continue

        turn_direction *= -1
        print(
            "主动定位转向状态=RETRY_TURN：turn_escape 失败（{}），"
            "改试相反转向方向；不会后退。".format(failure_reason),
            flush=True,
        )

    client.stop()
    speed_controller.stopped()
    print(
        "主动定位原地转向达到本轮截止时间；已停车并保留当前扫描，"
        "不会因此中止整圈巡检。",
        flush=True,
    )
    return scan


def _localization_motion_consistent(previous, current, match, args):
    if match is None or not match.converged or not math.isfinite(match.rmse_m):
        return False, math.inf, math.inf, 0.0
    measured = match.transform
    distance = math.hypot(measured.x_m, measured.y_m)
    expected = previous.pose.inverse().compose(current.pose)
    position_error = math.hypot(
        expected.x_m - measured.x_m,
        expected.y_m - measured.y_m,
    )
    yaw_error = abs(normalize_angle(expected.yaw_rad - measured.yaw_rad))
    observable_motion = (
        distance >= args.localization_motion_min_distance
        or abs(measured.yaw_rad) >= math.radians(
            args.localization_motion_min_yaw_deg
        )
    )
    valid = (
        observable_motion
        and match.rmse_m <= 0.10
        and match.inlier_ratio >= 0.55
        and position_error <= args.localization_motion_position_tolerance
        and yaw_error <= math.radians(args.localization_motion_yaw_tolerance_deg)
    )
    return valid, position_error, yaw_error, distance


def _execute_active_localization_move(
    client,
    scan_iterator,
    localizer,
    speed_controller,
    initial_scan,
    args,
    deadline_s,
    disambiguation_action=None,
):
    """Move continuously to a new viewpoint while learning motor dead zones."""
    scan = initial_scan
    baseline = initial_scan
    planned_turn_deg = 0.0
    planned_distance_m = args.relocalization_distance
    if disambiguation_action is not None:
        planned_turn_deg = float(disambiguation_action["turn_deg"])
        planned_distance_m = float(disambiguation_action["distance_m"])
        angular = (
            0 if abs(planned_turn_deg) < 5.0
            else int(math.copysign(args.relocalization_turn_speed, planned_turn_deg))
        )
        linear = args.relocalization_speed if angular == 0 else 0
        reason = "radar_candidate_disambiguation"
    else:
        linear, angular, reason = _active_localization_command(scan, args)
    print(
        "定位尚未完成，主动换观察位置：{}。".format(reason),
        flush=True,
    )

    last_applied = None

    def measured_step(requested):
        """Keep propulsion energized across scans instead of stop/start pulses."""
        nonlocal last_applied
        nonlocal scan
        applied = speed_controller.apply(requested)
        before = scan
        ttl_ms = speed_controller.ttl_for(applied, args.command_ttl_ms)
        command = (applied.linear_mm_s, applied.angular_mrad_s)
        if command == last_applied:
            refresh = getattr(client, "refresh_twist", None)
            if refresh is not None:
                refresh(*command, ttl_ms=ttl_ms)
            else:
                client.set_twist(*command, ttl_ms=ttl_ms)
        else:
            client.set_twist(*command, ttl_ms=ttl_ms)
        last_applied = command
        started_s = time.monotonic()
        scan = next(scan_iterator)
        duration_s = max(0.05, time.monotonic() - started_s)
        match = localizer.relative_motion(before, scan)
        speed_controller.observe(_motion_update(match, duration_s))
        diagnostics = speed_controller.diagnostics()
        print(
            "主动定位动作 state={} requested=({:+d},{:+d}) "
            "applied=({:+d},{:+d}) motion={:.3f}m/{:+.1f}deg "
            "turn_floor={} turn_limit={} turn_max={} "
            "turn_stall_events={} stalled_frames={} "
            "linear_adjust={:+d} ttl={}ms。".format(
                requested.state,
                requested.linear_mm_s,
                requested.angular_mrad_s,
                applied.linear_mm_s,
                applied.angular_mrad_s,
                math.hypot(match.transform.x_m, match.transform.y_m),
                math.degrees(match.transform.yaw_rad),
                diagnostics["turn_floor_mrad_s"],
                diagnostics["turn_limit_mrad_s"],
                diagnostics["maximum_turn_speed_mrad_s"],
                diagnostics["turn_stall_events"],
                diagnostics["stalled_frames"],
                diagnostics["linear_adjustment_mm_s"],
                ttl_ms,
            ),
            flush=True,
        )
        return applied, match, scan

    if linear == 0 and time.monotonic() < deadline_s:
        required_turn_rad = math.radians(
            max(10.0, min(70.0, abs(planned_turn_deg)))
            if disambiguation_action is not None else 10.0
        )
        scan = _execute_active_localization_turn(
            client,
            localizer,
            speed_controller,
            scan,
            angular,
            reason,
            required_turn_rad,
            args,
            deadline_s,
            measured_step,
        )

    # Keep sending a continuous straight command. Each scan supplies measured
    # motion feedback, so a stalled chassis rises 300 -> 340 -> 420 -> 460,
    # while useful movement immediately releases the temporary start boost.
    attempts = 0
    total_match = localizer.relative_motion(baseline, scan)
    while attempts < 18 and time.monotonic() < deadline_s:
        total_distance = math.hypot(
            total_match.transform.x_m, total_match.transform.y_m
        )
        total_yaw = abs(total_match.transform.yaw_rad)
        if (
            total_match.converged
            and (
                total_distance >= planned_distance_m
                or total_yaw >= math.radians(
                    args.localization_motion_min_yaw_deg
                )
            )
        ):
            break
        if not _obstacle_safe(
            scan, args.relocalization_speed, max(args.stop_distance, 0.50)
        ):
            _linear, turn, turn_reason = _active_localization_command(scan, args)
            if turn == 0:
                turn = args.relocalization_turn_speed
            scan = _execute_active_localization_turn(
                client,
                localizer,
                speed_controller,
                scan,
                turn,
                turn_reason,
                math.radians(10.0),
                args,
                deadline_s,
                measured_step,
            )
        else:
            measured_step(MotionCommand(
                args.relocalization_speed,
                0,
                "active_localization_advancing",
                "change lidar viewpoint",
            ))
        attempts += 1
        total_match = localizer.relative_motion(baseline, scan)
    client.stop()
    speed_controller.stopped()
    return scan, total_match


def _strict_radar_localization(result, scan, localizer, args):
    """Validate endpoint shape plus calibrated metric beam distances."""
    evidence = {
        "rmse_ok": False,
        "inliers_ok": False,
        "range_ok": False,
        "range_agreement": 0.0,
        "range_mean_error_m": math.inf,
        "range_coverage": 0.0,
    }
    if result is None or not math.isfinite(result.rmse_m):
        return False, evidence
    evidence["rmse_ok"] = result.rmse_m <= args.localization_max_rmse
    evidence["inliers_ok"] = result.inlier_ratio >= args.localization_min_inlier_ratio
    profile = localizer.range_profile(scan, maximum=72)
    metrics = localizer._range_consistency(result.pose, profile)
    if metrics is not None:
        agreement, mean_error, coverage = metrics
        evidence.update({
            "range_agreement": agreement,
            "range_mean_error_m": mean_error,
            "range_coverage": coverage,
        })
        evidence["range_ok"] = (
            agreement >= args.localization_min_range_agreement
            and mean_error <= args.localization_max_range_error
            and coverage >= args.localization_min_range_coverage
        )
    return bool(evidence["rmse_ok"] and evidence["inliers_ok"] and evidence["range_ok"]), evidence


def _fast_unique_radar_localization(result, evidence, args):
    """Allow fast startup only for a stable, uniquely strong radar solution."""
    return bool(
        result is not None
        and result.ambiguity_margin >= args.localization_fast_accept_margin
        and result.inlier_ratio >= args.localization_fast_min_inlier_ratio
        and evidence["range_agreement"] >= args.localization_fast_min_range_agreement
        and evidence["range_mean_error_m"] <= args.localization_fast_max_range_error
        and evidence["range_coverage"] >= args.localization_min_range_coverage
    )


def _radar_window_can_finish_without_motion(result, evidence, args):
    """A strict ready window is sufficient unless forced moves were opted in."""
    return bool(
        args.localization_required_moves == 0
        or _fast_unique_radar_localization(result, evidence, args)
    )


def _pose_matches_motion_prediction(result, predicted_pose, args):
    if result is None or predicted_pose is None:
        return False, math.inf, math.inf
    position_error = math.hypot(
        result.pose.x_m - predicted_pose.x_m,
        result.pose.y_m - predicted_pose.y_m,
    )
    yaw_error = abs(normalize_angle(result.pose.yaw_rad - predicted_pose.yaw_rad))
    return (
        position_error <= args.localization_motion_position_tolerance
        and yaw_error <= math.radians(args.localization_motion_yaw_tolerance_deg),
        position_error,
        yaw_error,
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _localization_acceptance(results, args):
    window = list(results)[-max(1, args.localization_window):]
    radar_valid = [
        item for item in window if item is not None and item.reliable
    ]
    # Single-scan ambiguity margins are noisy in repeated room geometry.  In
    # the field log, four strict metric matches agreed on one pose while their
    # runner-up score happened to be close; filtering by margin first reduced
    # a real 4/5 consensus to 1/5.  Cluster strict shape+distance results first
    # and use the dominant pose basin as the acceptance ratio.
    clusters = []
    for anchor in radar_valid:
        cluster = [
            item for item in radar_valid
            if math.hypot(
                item.pose.x_m - anchor.pose.x_m,
                item.pose.y_m - anchor.pose.y_m,
            ) <= 0.22
            and abs(normalize_angle(
                item.pose.yaw_rad - anchor.pose.yaw_rad
            )) <= math.radians(9.0)
        ]
        clusters.append(cluster)
    accepted = max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            float(np.median([item.score for item in cluster])),
        ),
        default=[],
    )
    ratio = len(accepted) / max(1, len(window))
    localization = consistent_localization(accepted)
    ready = (
        len(window) >= args.localization_min_accepted
        and len(accepted) >= args.localization_min_accepted
        and ratio >= args.localization_acceptance_ratio
        and localization is not None
    )
    return ready, localization, ratio, len(accepted), len(window)


def _active_localization_command(
    scan,
    args,
    linear_speed=None,
    turn_speed=None,
):
    """Choose a conservative short move toward the clearest lidar sector."""
    directions = (-70.0, -35.0, 0.0, 35.0, 70.0)
    clearances = [
        sector_min_distance(
            scan,
            center_deg=angle,
            width_deg=24.0,
            min_distance_m=0.08,
            max_distance_m=4.0,
        )
        for angle in directions
    ]
    best_index = int(np.argmax(clearances))
    angle = directions[best_index]
    clearance = clearances[best_index]
    linear_speed = int(
        args.relocalization_speed
        if linear_speed is None else linear_speed
    )
    turn_speed = int(
        args.relocalization_turn_speed
        if turn_speed is None else turn_speed
    )
    if clearance >= 0.75 and abs(angle) <= 20.0:
        return linear_speed, 0, "move_forward"
    if clearance >= 0.65:
        turn = int(math.copysign(turn_speed, angle))
        return 0, turn, "turn_toward_clear_space"
    # A blocked start is still allowed to rotate slowly to reveal a distinct
    # wall arrangement, but never translates into an unverified obstacle.
    turn_sign = 1 if clearances[-1] >= clearances[0] else -1
    return 0, turn_sign * turn_speed, "rotate_in_place"


def _temporary_obstacle_points(
    scan,
    pose,
    known_map,
    confirm_distance_m=0.18,
    front_only=False,
    lidar_offset_xy=(0.0, 0.0),
):
    points = scan.points(0.08, 8.0)
    if front_only and len(points):
        angles = np.arctan2(points[:, 1], points[:, 0])
        points = points[np.abs(angles) <= math.radians(32.0)]
    if not len(points):
        return np.empty((0, 2), dtype=np.float64)
    points = points + np.asarray(lidar_offset_xy, dtype=np.float64)
    world = pose.transform_points(points)
    cells = known_map.world_to_grid(world)
    columns, rows = cells[:, 0], cells[:, 1]
    inside = (
        (columns >= 0) & (columns < known_map.width)
        & (rows >= 0) & (rows < known_map.height)
    )
    unexpected = np.zeros(len(world), dtype=np.bool_)
    if np.any(inside):
        threshold_cells = max(
            1, int(math.ceil(confirm_distance_m / known_map.resolution_m))
        )
        unexpected[inside] = (
            known_map.free[rows[inside], columns[inside]]
            & (
                known_map.obstacle_distance_cells[
                    rows[inside], columns[inside]
                ] > threshold_cells
            )
        )
    return world[unexpected]


def _record_temporary_obstacle(scan, pose, known_map, args, front_only=True):
    points = _temporary_obstacle_points(
        scan,
        pose,
        known_map,
        confirm_distance_m=args.temporary_obstacle_confirm_distance,
        front_only=front_only,
        lidar_offset_xy=(args.lidar_offset_x, args.lidar_offset_y),
    )
    return known_map.mark_temporary_obstacles(
        points,
        radius_m=args.temporary_obstacle_radius,
        minimum_hits=args.temporary_obstacle_min_hits,
    )


def _remember_temporary_obstacle(
    first_scan,
    scan_iterator,
    pose,
    known_map,
    args,
):
    """Require repeated stopped evidence before changing session planning."""
    added = _record_temporary_obstacle(
        first_scan, pose, known_map, args, front_only=True
    )
    for _index in range(max(0, args.temporary_obstacle_min_hits - 1)):
        confirmation_scan = next(scan_iterator)
        added += _record_temporary_obstacle(
            confirmation_scan,
            pose,
            known_map,
            args,
            front_only=True,
        )
    return added


def run(args, *, client=None, lidar=None, camera=None):
    if args.enable_motion is False and not args.camera_test_only:
        raise RuntimeError("未给出 --enable-motion；为安全起见不会控制小车")
    if args.speed <= 0 or args.turn_speed <= 0:
        raise ValueError("速度参数必须为正数")
    if not 0.0 < args.localization_acceptance_ratio <= 1.0:
        raise ValueError("定位接受率必须在 (0, 1] 范围内")
    if args.localization_min_accepted < 1:
        raise ValueError("定位最少接受帧数必须大于零")
    if args.localization_window < args.localization_min_accepted:
        raise ValueError("定位滑动窗口不能小于最少接受帧数")
    if args.localization_required_moves < 0:
        raise ValueError("localization-required-moves cannot be negative")
    if not 0.0 < args.localization_min_inlier_ratio <= 1.0:
        raise ValueError("localization-min-inlier-ratio must be in (0, 1]")
    if not 0.0 < args.localization_min_range_agreement <= 1.0:
        raise ValueError("localization-min-range-agreement must be in (0, 1]")
    if not 0.0 < args.localization_min_range_coverage <= 1.0:
        raise ValueError("localization-min-range-coverage must be in (0, 1]")
    if not 0.0 < args.localization_fast_min_inlier_ratio <= 1.0:
        raise ValueError("localization-fast-min-inlier-ratio must be in (0, 1]")
    if not 0.0 < args.localization_fast_min_range_agreement <= 1.0:
        raise ValueError("localization-fast-min-range-agreement must be in (0, 1]")
    if args.localization_max_rmse <= 0.0 or args.localization_max_range_error <= 0.0:
        raise ValueError("strict radar error thresholds must be positive")
    if args.relocalization_max_speed < args.relocalization_speed:
        raise ValueError("主动定位最大直行速度不能小于起始速度")
    if args.relocalization_max_turn_speed < args.relocalization_turn_speed:
        raise ValueError("主动定位最大转向速度不能小于起始转向速度")
    if args.relocalization_turn_stall_at_max_frames < 1:
        raise ValueError(
            "relocalization-turn-stall-at-max-frames must be positive"
        )
    if not 0.03 <= args.relocalization_turn_escape_distance <= 0.08:
        raise ValueError(
            "relocalization-turn-escape-distance must be between 0.03 and 0.08 m"
        )
    if args.relocalization_turn_escape_max_steps < 1:
        raise ValueError(
            "relocalization-turn-escape-max-steps must be positive"
        )
    if args.relocalization_turn_escape_max_attempts < 1:
        raise ValueError(
            "relocalization-turn-escape-max-attempts must be positive"
        )
    if args.max_drive_speed < max(args.speed, args.slow_speed):
        raise ValueError("巡检最大直行速度不能小于基础直行速度")
    if args.start_kick_speed <= 0:
        raise ValueError("--start-kick-speed must be positive")
    if not 0.15 <= args.emergency_front_distance <= args.stop_distance:
        raise ValueError(
            "--emergency-front-distance must be between 0.15m and stop-distance"
        )
    if not 0.05 <= args.collision_reverse_distance <= 0.12:
        raise ValueError(
            "--collision-reverse-distance must be between 0.05m and 0.12m"
        )
    if not 80 <= args.collision_reverse_speed <= 260:
        raise ValueError(
            "--collision-reverse-speed must be between 80 and 260mm/s"
        )
    if args.collision_reverse_rear_clearance < 0.25:
        raise ValueError(
            "--collision-reverse-rear-clearance must be at least 0.25m"
        )
    if not 2 <= args.collision_reverse_max_steps <= 6:
        raise ValueError(
            "--collision-reverse-max-steps must be between 2 and 6"
        )
    if not 0.04 <= args.collision_reverse_map_fallback_max_distance <= 0.08:
        raise ValueError(
            "--collision-reverse-map-fallback-max-distance must be between "
            "0.04m and 0.08m"
        )
    if args.min_target_spacing < 0.0:
        raise ValueError("--min-target-spacing cannot be negative")
    if args.target_turn_cost_m_per_rad < 0.0:
        raise ValueError("--target-turn-cost-m-per-rad cannot be negative")
    if not 0.10 <= args.target_retry_min_displacement <= 1.0:
        raise ValueError(
            "--target-retry-min-displacement must be in [0.10, 1.0]"
        )
    if args.max_target_navigation_attempts < 1:
        raise ValueError("--max-target-navigation-attempts must be positive")
    if args.navigation_turn_stall_at_max_frames < 1:
        raise ValueError(
            "--navigation-turn-stall-at-max-frames must be positive"
        )
    if args.navigation_linear_stall_at_max_frames < 2:
        raise ValueError(
            "--navigation-linear-stall-at-max-frames must be at least 2"
        )
    if args.navigation_pure_turn_timeout_seconds < 2.0:
        raise ValueError(
            "--navigation-pure-turn-timeout-seconds must be at least 2 seconds"
        )
    if not 5.0 <= args.navigation_timeout_capture_max_heading_error_deg <= 60.0:
        raise ValueError(
            "--navigation-timeout-capture-max-heading-error-deg must be in [5, 60]"
        )
    if not 0.5 <= args.navigation_turn_refresh_window_seconds <= 1.5:
        raise ValueError(
            "--navigation-turn-refresh-window-seconds must be between 0.5 and 1.5"
        )
    if args.navigation_turn_escape_max_attempts < 0:
        raise ValueError(
            "--navigation-turn-escape-max-attempts cannot be negative"
        )
    if args.navigation_pose_stationary_yaw_limit_deg <= 0.0:
        raise ValueError(
            "--navigation-pose-stationary-yaw-limit-deg must be positive"
        )
    if args.navigation_pose_stationary_translation_limit <= 0.0:
        raise ValueError(
            "--navigation-pose-stationary-translation-limit must be positive"
        )
    if args.navigation_pose_direction_slack_deg < 0.0:
        raise ValueError(
            "--navigation-pose-direction-slack-deg cannot be negative"
        )
    if args.navigation_pose_max_correction_holds < 1:
        raise ValueError(
            "--navigation-pose-max-correction-holds must be positive"
        )
    if args.navigation_fine_control_distance <= args.waypoint_tolerance:
        raise ValueError(
            "--navigation-fine-control-distance must exceed waypoint tolerance"
        )
    if args.navigation_fine_turn_speed <= 0:
        raise ValueError("--navigation-fine-turn-speed must be positive")
    if not (
        550 <= args.navigation_fine_turn_min_speed
        <= args.navigation_fine_turn_speed
    ):
        raise ValueError(
            "--navigation-fine-turn-min-speed must be in [550, fine max]"
        )
    if not 180 <= args.navigation_precision_turn_ttl_ms <= 350:
        raise ValueError(
            "--navigation-precision-turn-ttl-ms must be in [180, 350]"
        )
    if not 2 <= args.navigation_heading_seed_frames <= 5:
        raise ValueError(
            "--navigation-heading-seed-frames must be between 2 and 5"
        )
    if args.navigation_map_correction_interval_seconds <= 0.0:
        raise ValueError(
            "--navigation-map-correction-interval-seconds must be positive"
        )
    if args.navigation_map_correction_distance <= 0.0:
        raise ValueError(
            "--navigation-map-correction-distance must be positive"
        )
    if not 0.0 < args.navigation_map_correction_gain <= 1.0:
        raise ValueError(
            "--navigation-map-correction-gain must be in (0, 1]"
        )
    if not 0.0 < args.navigation_recovery_min_inlier_ratio <= 1.0:
        raise ValueError(
            "--navigation-recovery-min-inlier-ratio must be in (0, 1]"
        )
    if args.navigation_recovery_max_rmse <= 0.0:
        raise ValueError("--navigation-recovery-max-rmse must be positive")
    if args.navigation_recovery_confirmation_frames < 1:
        raise ValueError(
            "--navigation-recovery-confirmation-frames must be positive"
        )
    if args.alternate_view_max_shift <= 0.0:
        raise ValueError("--alternate-view-max-shift must be positive")
    if args.nearby_capture_distance < args.waypoint_tolerance:
        raise ValueError(
            "--nearby-capture-distance cannot be less than waypoint tolerance"
        )
    if not 1 <= args.photos_per_target <= 4:
        raise ValueError("photos-per-target must be between 1 and 4")
    if args.online_qwen_timeout_seconds <= 0.0:
        raise ValueError("--online-qwen-timeout-seconds must be positive")
    if not 0.5 <= args.online_qwen_min_confidence <= 1.0:
        raise ValueError("--online-qwen-min-confidence must be in [0.5, 1]")
    if not 1 <= args.online_qwen_confirmations <= 4:
        raise ValueError("--online-qwen-confirmations must be between 1 and 4")
    if not 1 <= args.online_qwen_max_component_attempts <= 5:
        raise ValueError(
            "--online-qwen-max-component-attempts must be between 1 and 5"
        )
    if args.information_view_distance < 0.0:
        raise ValueError("--information-view-distance cannot be negative")
    if args.open_space_panorama:
        from raspberry_pi.open_space_panorama import PanoramaConfig
        panorama_config = PanoramaConfig.from_args(args)
        panorama_config.validate()
        args.clearance = panorama_config.map_clearance_m(args.clearance)
    known_map = KnownOccupancyMap.load(
        args.map, args.map_yaml, clearance_m=args.clearance
    )
    permanent_map_hashes = {
        str(Path(args.map)): _file_sha256(args.map),
        str(Path(args.map_yaml or Path(args.map).with_suffix(".yaml"))): (
            _file_sha256(args.map_yaml or Path(args.map).with_suffix(".yaml"))
        ),
    }
    localizer = ScanMapLocalizer(
        known_map,
        lidar_offset_x_m=args.lidar_offset_x,
        lidar_offset_y_m=args.lidar_offset_y,
    )
    if args.open_space_panorama:
        # No fixed viewpoints or permanent semantic entities are generated in
        # panorama mode. Its observations live in the run's own sidecar only.
        targets = []
        semantic_map = None
        mission_interior_component_ids = set()
        preclassified_components = set()
        verified_capture_poses = {}
        original_component_targets = {}
        information_views = {}
        capture_reference_views = {}
    else:
        targets = known_map.inspection_targets(
            standoff_m=args.standoff,
            maximum_targets=args.max_targets,
            views_per_wall=args.views_per_wall,
            views_per_region=args.views_per_region,
        )
        if args.min_target_spacing > 0.0:
            targets = _spaced_targets(
                targets, min_spacing_m=args.min_target_spacing
            )
        if not targets:
            raise RuntimeError("地图中没有可达的墙面或黑色区域观察点")
        semantic_path = (
            Path(args.semantic_map)
            if args.semantic_map
            else Path(args.map).with_name(Path(args.map).stem + "_semantics.json")
        )
        semantic_map = PermanentSemanticMap(
            semantic_path,
            args.map,
            permanent_map_hashes[str(Path(args.map))],
            targets,
        )
        mission_interior_component_ids = {
            target.target_id.rsplit("_", 1)[0]
            for target in targets
            if target.kind == "interior_black_region"
        }
        preclassified_components = set()
        if args.skip_classified_semantic_components:
            targets, preclassified_components = (
                _skip_classified_interior_targets(targets, semantic_map)
            )
            if preclassified_components:
                print(
                    "永久语义地图已有 {} 个内部黑色连通体完成分类，"
                    "本轮跳过={}。".format(
                        len(preclassified_components),
                        ",".join(sorted(preclassified_components)),
                    ),
                    flush=True,
                )
        if args.ignore_verified_semantic_views:
            verified_capture_poses = {}
        else:
            targets, verified_capture_poses = _apply_verified_semantic_views(
                targets, semantic_map
            )
        original_component_targets = {
            target.target_id: target for target in targets
        }
        if args.information_view_distance > 0.0:
            targets, information_views = _information_view_targets(
                known_map, targets, args.information_view_distance
            )
        else:
            information_views = {}
        capture_reference_views = {
            target.target_id: tuple(target.view_xy_m) for target in targets
        }
    camera = camera or CameraCapture(
        args.camera_backend,
        args.camera_device,
        args.camera_width,
        args.camera_height,
        args.camera_timeout,
        dual_usb_cameras=args.dual_usb_cameras,
        free_cameras=args.free_cameras,
    )
    report = InspectionReport(args.output, args.map)
    online_qwen = OnlineQwenComponentLabeler(
        args.online_qwen_url,
        timeout_seconds=args.online_qwen_timeout_seconds,
        min_confidence=args.online_qwen_min_confidence,
        confirmations=args.online_qwen_confirmations,
    )
    online_confirmed_components = set()
    deferred_online_annotations = []

    def annotate_capture_online(target, record, image_path):
        if not online_qwen.url:
            return False
        second_name = record.get("camera_2_image")
        if not second_name:
            inference = {
                "enabled": True,
                "promoted": False,
                "error": "dual_camera_pair_required",
            }
        else:
            print(
                "在线Qwen：车辆保持停车，正在分析目标 {} 的第 {} 张"
                "双摄照片……".format(
                    target.target_id,
                    record.get("photo_index", 1),
                ),
                flush=True,
            )
            inference = online_qwen.analyze(
                target,
                record,
                image_path,
                Path(image_path).with_name(str(second_name)),
            )
        record["online_qwen"] = inference
        semantic_map.record_online_qwen(target, inference, image_path)
        report.save()
        if inference.get("promoted"):
            online_confirmed_components.add(
                target.target_id.rsplit("_", 1)[0]
            )
            print(
                "在线Qwen确认：{} 连续 {} 次一致，已把对应黑色"
                "连通区域标为 {}（置信度 {:.2f}）。".format(
                    target.target_id,
                    inference.get("required_confirmations"),
                    inference.get("candidate_label"),
                    float(inference.get("candidate_confidence", 0.0)),
                ),
                flush=True,
            )
        else:
            print(
                "在线Qwen暂不写图：{}，原因={}，候选={}，确认={}/{}。"
                .format(
                    target.target_id,
                    inference.get("error")
                    or inference.get("rejection")
                    or "awaiting_repeat",
                    inference.get("candidate_label", "none"),
                    inference.get("confirmation_count", 0),
                    inference.get("required_confirmations", args.online_qwen_confirmations),
                ),
                flush=True,
            )
        return bool(inference.get("promoted"))
    if args.camera_test_only:
        camera_test_path = report.directory / "00_camera_test.jpg"
        camera_test_info = camera.capture(camera_test_path)
        print(
            "相机测试成功：{}，后端={}，设备={}，路数={}，亮度={:.1f}，对比度={:.1f}".format(
                camera_test_path,
                camera_test_info["backend"],
                camera_test_info.get("camera_device"),
                camera_test_info.get("camera_count", 1),
                camera_test_info["brightness"],
                camera_test_info["contrast"],
            )
        )
        return report.directory
    owns_client = client is None
    owns_lidar = lidar is None
    client = client or Esp32Client(_serial_config(args))
    lidar = lidar or N10LidarDriver(_lidar_config(args))
    stopping = False

    def request_stop(_signal_number, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    scan_iterator = iter(lidar.scans())
    started = time.monotonic()
    last_command = None
    last_command_sent_s = -math.inf
    try:
        _start_motion_keepalive(client, owns_client)
        client.heartbeat()
        client.stop()
        camera_test_path = report.directory / "00_camera_test.jpg"
        camera_test_info = camera.capture(camera_test_path)
        print(
            "相机自检通过：{}（{} 路，设备={}，亮度 {:.1f}，对比度 {:.1f}）。".format(
                camera_test_path,
                camera_test_info.get("camera_count", 1),
                camera_test_info.get("camera_devices", camera_test_info.get("camera_device")),
                camera_test_info["brightness"],
                camera_test_info["contrast"],
            )
        )
        print(
            "已加载只读地图：{}，候选拍摄位 {} 个（复用 {} 个已验证"
            "语义机位）。".format(
                args.map, len(targets), len(verified_capture_poses)
            )
        )
        print(
            "正在进行已知地图主动定位：先采集静止候选，随后会连续短距离移动，"
            "完成跨位置校验后才确认。",
            flush=True,
        )
        print(
            "标注模式定位只使用永久地图与完整雷达扫描的形状、绝对距离；"
            "相机照片仅在定位完成后采集，不参与初始定位。",
            flush=True,
        )
        hint = None
        if args.initial_pose:
            hint = Pose2D(
                args.initial_pose[0],
                args.initial_pose[1],
                math.radians(args.initial_pose[2]),
            )
            print("使用命令行初始位姿作为搜索提示，并仍由雷达地图核验。")
        elif not args.ignore_trajectory_pose:
            hint = _first_trajectory_pose(args.map)
            if hint is not None:
                print(
                    "采用建图轨迹起点作为搜索提示：({:+.2f},{:+.2f},{:+.1f}°)；"
                    "它不会被直接当成定位结果。".format(
                        hint.x_m, hint.y_m, math.degrees(hint.yaw_rad)
                    )
                )
        speed_controller = _build_speed_controller(args)
        localization_results = deque(maxlen=max(1, args.localization_window))
        localization_diagnostics_path = (
            report.directory / "localization_diagnostics.jsonl"
        )
        localization = None
        localization_started_s = time.monotonic()
        stationary_attempts = 0
        anchor_localization = None
        anchor_scan = None
        pending_motion_match = None
        validated_moves = 0
        while not stopping:
            elapsed = time.monotonic() - localization_started_s
            if elapsed >= args.localization_timeout_seconds:
                client.stop()
                raise RuntimeError(
                    "主动定位超过 {:.0f} 秒仍未达到接受率要求，车辆已停车。".format(
                        args.localization_timeout_seconds
                    )
                )
            scan = next(scan_iterator)
            # Keep a whole-map competing basin even when an initial hint is
            # supplied. A local-only lookup can confidently select the wrong
            # corner in a symmetric room.
            result = localizer.global_localize(scan, hint=hint)
            strict_ok, radar_evidence = _strict_radar_localization(
                result, scan, localizer, args
            )
            if (
                strict_ok
                and anchor_localization is not None
                and pending_motion_match is not None
                and pending_motion_match.converged
            ):
                predicted_pose = anchor_localization.pose.compose(
                    pending_motion_match.transform
                )
                motion_ok, position_error, yaw_error = (
                    _pose_matches_motion_prediction(result, predicted_pose, args)
                )
                if not motion_ok:
                    constrained = localizer.localize_near_position(
                        scan, predicted_pose, radius_m=0.45
                    )
                    constrained_ok, constrained_evidence = (
                        _strict_radar_localization(
                            constrained, scan, localizer, args
                        )
                    )
                    constrained_motion_ok, _, _ = _pose_matches_motion_prediction(
                        constrained, predicted_pose, args
                    )
                    if constrained_ok and constrained_motion_ok:
                        print(
                            "拒绝跨区域全局跳点 {:.2f}m/{:.1f}°；"
                            "采用实测位移附近的严格雷达候选。".format(
                                position_error, math.degrees(yaw_error)
                            ),
                            flush=True,
                        )
                        result = constrained
                        radar_evidence = constrained_evidence
                    else:
                        strict_ok = False
                        print(
                            "拒绝与实测位移不一致的定位跳点 {:.2f}m/{:.1f}°。".format(
                                position_error, math.degrees(yaw_error)
                            ),
                            flush=True,
                        )
            localization_results.append(result if strict_ok else None)
            stationary_attempts += 1
            ready, candidate, ratio, accepted, total = _localization_acceptance(
                localization_results, args
            )
            diagnostic = {
                "elapsed_s": round(elapsed, 3),
                "strict_radar_accepted": bool(strict_ok),
                "window_ready": bool(ready),
                "window_accepted": int(accepted),
                "window_total": int(total),
                "window_ratio": round(ratio, 4),
                "required_moves": int(args.localization_required_moves),
                "validated_moves": int(validated_moves),
                "radar_evidence": radar_evidence,
                "result": None if result is None else {
                    "x_m": result.pose.x_m,
                    "y_m": result.pose.y_m,
                    "yaw_deg": math.degrees(result.pose.yaw_rad),
                    "rmse_m": result.rmse_m,
                    "inlier_ratio": result.inlier_ratio,
                    "ambiguity_margin": result.ambiguity_margin,
                },
            }
            _append_navigation_diagnostic(
                localization_diagnostics_path, diagnostic
            )
            if result is None:
                detail = "无有效候选"
            else:
                detail = (
                    "pose=({:+.2f},{:+.2f},{:+.1f}°) rmse={:.3f} "
                    "inliers={:.0f}% margin={:.3f} range={:.0f}%/{:.3f}m coverage={:.0f}%{}"
                ).format(
                    result.pose.x_m,
                    result.pose.y_m,
                    math.degrees(result.pose.yaw_rad),
                    result.rmse_m,
                    result.inlier_ratio * 100.0,
                    result.ambiguity_margin,
                    radar_evidence["range_agreement"] * 100.0,
                    radar_evidence["range_mean_error_m"],
                    radar_evidence["range_coverage"] * 100.0,
                    "" if strict_ok else " [strict radar rejected]",
                )
            print(
                "主动定位：{}；窗口接受 {}/{}={:.0f}%（要求≥{:.0f}%）。".format(
                    detail,
                    accepted,
                    total,
                    ratio * 100.0,
                    args.localization_acceptance_ratio * 100.0,
                )
            )
            if ready:
                if anchor_localization is None:
                    if _radar_window_can_finish_without_motion(
                        candidate, radar_evidence, args
                    ):
                        localization = candidate
                        client.stop()
                        print(
                            "严格雷达定位窗口已经通过；"
                            "无需强制换位，已确认初始位置。",
                            flush=True,
                        )
                        break
                    anchor_localization = candidate
                    anchor_scan = scan
                    print(
                        "静止结果只作为候选点；必须实际换位并完成 {}/{} 次"
                        "跨位置校验后才确认定位。".format(
                            validated_moves, args.localization_required_moves
                        ),
                        flush=True,
                    )
                elif pending_motion_match is not None:
                    valid, position_error, yaw_error, moved_distance = (
                        _localization_motion_consistent(
                            anchor_localization,
                            candidate,
                            pending_motion_match,
                            args,
                        )
                    )
                    if valid:
                        validated_moves += 1
                        print(
                            "跨位置校验通过 {}/{}：真实位移 {:.3f}m，"
                            "位姿变化误差 {:.3f}m/{:.1f}°。".format(
                                validated_moves,
                                args.localization_required_moves,
                                moved_distance,
                                position_error,
                                math.degrees(yaw_error),
                            ),
                            flush=True,
                        )
                        anchor_localization = candidate
                        anchor_scan = scan
                    else:
                        validated_moves = 0
                        print(
                            "跨位置校验未通过：真实位移 {:.3f}m，位姿变化误差 "
                            "{:.3f}m/{:.1f}°；拒绝稳定但可能对称错误的位置。".format(
                                moved_distance,
                                position_error,
                                math.degrees(yaw_error),
                            ),
                            flush=True,
                        )
                        anchor_localization = candidate
                        anchor_scan = scan
                    pending_motion_match = None
                if validated_moves >= args.localization_required_moves:
                    localization = candidate
                    client.stop()
                    break
            if stationary_attempts < max(
                args.localization_min_accepted,
                args.localization_window,
            ):
                continue

            deadline_s = (
                localization_started_s + args.localization_timeout_seconds
            )
            basins = localizer.global_localization_candidates(scan, maximum=4)
            plausible_basins = (
                [] if not basins else [
                    item for item in basins
                    if basins[0].score - item.score <= max(
                        0.12, args.localization_min_margin * 2.0
                    )
                ]
            )
            disambiguation_action = localizer.disambiguating_motion(
                plausible_basins,
                maximum_distance_m=args.localization_disambiguation_distance,
            )
            if disambiguation_action is not None:
                print(
                    "雷达存在 {} 个相似候选点；选择转向 {:+.0f}°、前进 {:.2f}m，"
                    "预计候选距离轮廓分离度 {:.3f}m。".format(
                        disambiguation_action["candidate_count"],
                        disambiguation_action["turn_deg"],
                        disambiguation_action["distance_m"],
                        disambiguation_action["separation_score_m"],
                    ),
                    flush=True,
                )
            after_motion_scan, _move_match = (
                _execute_active_localization_move(
                    client,
                    scan_iterator,
                    localizer,
                    speed_controller,
                    scan,
                    args,
                    deadline_s,
                    disambiguation_action=disambiguation_action,
                )
            )
            pending_motion_match = localizer.relative_motion(
                anchor_scan if anchor_scan is not None else scan,
                after_motion_scan,
            )
            localization_results.clear()
            stationary_attempts = 0
            hint = (
                anchor_localization.pose.compose(
                    pending_motion_match.transform
                )
                if anchor_localization is not None
                and pending_motion_match.converged
                else None
            )
            continue

        if localization is None:
            raise RuntimeError("定位流程被中止，车辆已停车")
        pose = localization.pose
        if args.open_space_panorama:
            from raspberry_pi.open_space_panorama import (
                run_open_space_panorama,
            )
            print(
                "定位确认：({:+.2f},{:+.2f},{:+.1f}°)。切换到开阔点"
                "全景巡检：不生成固定目标点、不执行全局路径规划、"
                "不重新建图。".format(
                    pose.x_m, pose.y_m, math.degrees(pose.yaw_rad)
                ),
                flush=True,
            )
            scan_watchdog = getattr(lidar, 'set_scan_watchdog', None)
            previous_watchdog = scan_watchdog(args.panorama_max_scan_age_seconds) if scan_watchdog else None
            try:
                panorama_result = run_open_space_panorama(
                    args=args,
                    known_map=known_map,
                    localizer=localizer,
                    initial_pose=pose,
                    initial_scan=scan,
                    scan_iterator=scan_iterator,
                    client=client,
                    camera=camera,
                    report=report,
                    online_qwen=online_qwen,
                    should_stop=lambda: stopping,
                )
            finally:
                if scan_watchdog:
                    scan_watchdog(previous_watchdog)
            print(
                _panorama_result_message(panorama_result),
                flush=True,
            )
            return report.directory
        # Viewpoint selection must happen after localization. A target can be
        # visible from several sides; choosing one before the start pose was
        # known made the Pi commit to an arbitrary (and platform-dependent)
        # side, then nearest-target ordering could no longer choose the short
        # upper passage. Rebuild the same stable target IDs with viewpoints
        # selected by actual A* distance from this confirmed start.
        targets = known_map.inspection_targets(
            standoff_m=args.standoff,
            maximum_targets=args.max_targets,
            views_per_wall=args.views_per_wall,
            views_per_region=args.views_per_region,
            preferred_view_xy=(pose.x_m, pose.y_m),
        )
        if args.min_target_spacing > 0.0:
            targets = _spaced_targets(
                targets, min_spacing_m=args.min_target_spacing
            )
        targets, preclassified_components = (
            _skip_classified_interior_targets(targets, semantic_map)
            if args.skip_classified_semantic_components
            else (targets, set())
        )
        if args.ignore_verified_semantic_views:
            verified_capture_poses = {}
        else:
            targets, verified_capture_poses = _apply_verified_semantic_views(
                targets, semantic_map
            )
        original_component_targets = {
            target.target_id: target for target in targets
        }
        if args.information_view_distance > 0.0:
            targets, information_views = _information_view_targets(
                known_map, targets, args.information_view_distance
            )
        else:
            information_views = {}
        capture_reference_views = {
            target.target_id: tuple(target.view_xy_m) for target in targets
        }
        # Startup localization remains strict. Navigation now follows every
        # finite local tracking pose directly, so its motion controller uses
        # the same relaxed quality envelope only for torque adaptation.
        speed_controller = _build_speed_controller(args, navigation=True)
        ordered = order_targets(
            known_map, (pose.x_m, pose.y_m), targets,
            start_yaw_rad=pose.yaw_rad,
            turn_cost_m_per_rad=args.target_turn_cost_m_per_rad,
        )
        if not ordered:
            raise RuntimeError("当前位置无法通过地图白色区域到达任何拍摄位")
        report.save_plan(known_map, pose, ordered)
        navigation_diagnostics_path = (
            report.directory / "navigation_diagnostics.jsonl"
        )
        print(
            "定位确认：({:+.2f},{:+.2f},{:+.1f}°)，将巡检 {} 个拍摄位。".format(
                pose.x_m, pose.y_m, math.degrees(pose.yaw_rad), len(ordered)
            )
        )
        # Each leg is replanned from the current localized pose below.  The
        # preview's precomputed paths are only a visual mission overview.
        ordered_targets = [target for target, _path in ordered]
        failures = 0
        completed_targets = 0
        target_attempts = {}
        target_retry_origins = {}
        turn_escape_attempts = {}
        force_forward_arc_targets = set()
        blocked_path_heading_targets = set()
        alternate_views = {}
        detail_view_targets = set()
        visual_component_attempts = {}
        navigation_turn_stall_frames = 0
        navigation_linear_stall_frames = 0
        navigation_stationary_command_frames = 0
        navigation_pose_correction_holds = 0
        navigation_recovery_candidates = deque(
            maxlen=max(2, args.navigation_recovery_confirmation_frames)
        )
        last_pose_update_s = time.monotonic()
        last_map_correction_s = last_pose_update_s
        navigation_distance_since_correction_m = 0.0
        turn_anchor_pose = None
        last_status_print_s = -math.inf
        capture_mode = "exact"
        capture_note = ""
        pose_trail = deque(maxlen=40)  # 实时计划图轨迹
        last_plan_render_s = 0.0
        confirmed_start_pose = pose
        while ordered_targets and not stopping:
            choices = []
            for pending_index, pending_target in enumerate(ordered_targets):
                route_proposal = _inspection_path_for_target(
                    known_map,
                    pose,
                    pending_target,
                    args.target_turn_cost_m_per_rad,
                )
                if route_proposal is None:
                    continue
                pending_path = route_proposal["path"]
                route_metrics = route_proposal["metrics"]
                choices.append((
                    route_metrics["score"],
                    route_metrics["length_m"],
                    pending_target.target_id,
                    pending_index,
                    pending_path,
                    route_metrics["departure_turn_rad"],
                    route_metrics["arrival_turn_rad"],
                    route_proposal["staged_camera_approach"],
                ))
            if choices:
                choices = _prefer_displaced_retry_choices(
                    choices,
                    pose,
                    target_retry_origins,
                    args.target_retry_min_displacement,
                )
            if not choices:
                # A volatile obstacle layer is not trustworthy enough to
                # cancel the rest of an inspection.  Field run
                # 20260821_134140 projected slightly mislocalized wall scans
                # into 1073 temporary cells, disconnected the map, and then
                # bulk-skipped twelve targets.  Forget volatile evidence and
                # retry against the user's permanent PGM first.
                temporary_count = int(np.count_nonzero(
                    known_map.temporary_occupied
                ))
                if temporary_count:
                    known_map.clear_temporary_obstacles()
                    print(
                        "临时障碍层导致剩余目标无路径；已丢弃 {} 个不可靠"
                        "临时栅格，并按永久地图继续规划，不会批量取消拍摄点。"
                        .format(temporary_count),
                        flush=True,
                    )
                    continue

                # If even the permanent map has no strict route, keep the
                # failure explicit.  A photo taken at the current pose is not
                # evidence for a remote target and must never consume one of
                # the requested target slots or enter the semantic map.
                selected_index = min(
                    range(len(ordered_targets)),
                    key=lambda index: math.hypot(
                        ordered_targets[index].view_xy_m[0] - pose.x_m,
                        ordered_targets[index].view_xy_m[1] - pose.y_m,
                    ),
                )
                target = ordered_targets.pop(selected_index)
                attempt = target_attempts.get(target.target_id, 0) + 1
                target_attempts[target.target_id] = attempt
                report.add_skipped_target(
                    target,
                    pose,
                    "permanent_map_route_unavailable; no photo was taken "
                    "and the target slot remains uncovered",
                    attempt,
                )
                print(
                    "目标 {} 在永久地图上没有可用路径；不在远处代拍、"
                    "不计入完成数，已加入未覆盖清单并继续其他目标。"
                    .format(target.target_id),
                    flush=True,
                )
                continue
            else:
                (
                    _score,
                    _length,
                    _target_id,
                    selected_index,
                    path,
                    planned_departure_turn_rad,
                    planned_arrival_turn_rad,
                    staged_camera_approach,
                ) = min(choices)
            target = ordered_targets.pop(selected_index)
            target_attempts[target.target_id] = (
                target_attempts.get(target.target_id, 0) + 1
            )
            navigation_turn_stall_frames = 0
            navigation_linear_stall_frames = 0
            navigation_stationary_command_frames = 0
            print(
                "动态选点：{}，路径 {:.2f}m，出发转向 {:.0f}°，"
                "预计抵达后拍摄转向 {:.0f}°{}。".format(
                    target.target_id,
                    _length,
                    math.degrees(planned_departure_turn_rad),
                    math.degrees(planned_arrival_turn_rad),
                    (
                        "（使用机位前对准接近点）"
                        if staged_camera_approach else ""
                    ),
                ),
                flush=True,
            )
            if target.target_id in alternate_views:
                capture_mode = "alternate_view_same_subject"
                original_view = alternate_views[target.target_id][
                    "original_view"
                ]
                capture_note = (
                    "original view ({:.3f},{:.3f}) was unreachable; "
                    "used a nearby reachable view with permanent-map "
                    "line-of-sight to the same target"
                ).format(original_view[0], original_view[1])
            elif target.target_id in detail_view_targets:
                capture_mode = "component_detail_retry"
                capture_note = (
                    "wide view did not produce repeated bounded Qwen evidence; "
                    "returned to the original close component-facing view"
                )
            elif target.target_id in information_views:
                capture_mode = "wide_information_view"
                capture_note = (
                    "map line-of-sight allowed a {:.3f}m component-facing "
                    "overview; corners without this clearance retain their "
                    "original close view"
                ).format(
                    information_views[target.target_id]["wide_distance_m"]
                )
            else:
                capture_mode = "exact"
                capture_note = ""
            if stopping:
                break
            waypoints = path[1:] or [target.view_xy_m]
            replan_target = False
            target_ready_for_capture = False
            fine_map_correction_done = False
            for waypoint_index, waypoint in enumerate(waypoints):
                final_translation_complete = False
                pure_turn_started_s = None
                precision_heading_remaining_rad = None
                precision_heading_initial_error_rad = None
                precision_heading_last_source = "not_started"
                precision_heading_response_gain = 0.75
                precision_heading_seed_samples = []
                while not stopping:
                    if time.monotonic() - started >= args.max_runtime_min * 60.0:
                        stopping = True
                        break
                    scan = next(scan_iterator)
                    result = localizer.track(scan, pose)
                    if not _tracking_pose_available(result):
                        failures += 1
                        client.stop()
                        last_command = (0, 0)
                        speed_controller.stopped()
                        print(
                            "导航定位没有返回有限位姿 {}/{}，车辆已停车等待"
                            "下一帧；不按旧位姿盲走。".format(
                                failures,
                                args.max_localization_failures,
                            ),
                            flush=True,
                        )
                        if failures >= args.max_localization_failures:
                            raise RuntimeError(
                                "NAVIGATION_POSE_UNAVAILABLE: 连续定位帧没有"
                                "返回有限位姿"
                            )
                        continue
                    previous_navigation_pose = pose
                    update_now = time.monotonic()
                    frame_motion = getattr(
                        localizer, "last_track_motion", None
                    )
                    motion_pose = _navigation_motion_pose(
                        pose, frame_motion
                    )
                    motion_pose_reason = ""
                    if motion_pose is not None:
                        motion_pose_reason = _navigation_pose_jump_reason(
                            pose,
                            replace(result, pose=motion_pose),
                            last_command,
                            update_now - last_pose_update_s,
                            speed_controller.diagnostics()["turn_ttl_ms"],
                            args,
                        )
                        if motion_pose_reason:
                            motion_pose = None
                        else:
                            navigation_distance_since_correction_m += (
                                math.hypot(
                                    frame_motion.transform.x_m,
                                    frame_motion.transform.y_m,
                                )
                            )
                    predicted_pose = pose if motion_pose is None else motion_pose
                    final_control_zone = (
                        waypoint_index == len(waypoints) - 1
                        and math.hypot(
                            target.view_xy_m[0] - predicted_pose.x_m,
                            target.view_xy_m[1] - predicted_pose.y_m,
                        ) <= args.navigation_fine_control_distance
                    )
                    force_fine_correction = (
                        final_control_zone and not fine_map_correction_done
                    )
                    map_correction_due = _periodic_map_correction_due(
                        update_now,
                        last_map_correction_s,
                        navigation_distance_since_correction_m,
                        args.navigation_map_correction_interval_seconds,
                        args.navigation_map_correction_distance,
                        force=force_fine_correction,
                    )
                    navigation_pose_policy = "scan_motion_continuous"
                    if map_correction_due:
                        correction_reason = _navigation_pose_jump_reason(
                            predicted_pose,
                            result,
                            (0, 0),
                            0.0,
                            speed_controller.diagnostics()["turn_ttl_ms"],
                            args,
                        )
                        if correction_reason:
                            result = replace(result, pose=predicted_pose)
                            navigation_pose_policy = (
                                "periodic_map_correction_skipped"
                            )
                            print(
                                "周期地图校准跳过：{}；继续采用连续雷达"
                                "里程计，不停车。".format(correction_reason),
                                flush=True,
                            )
                        else:
                            corrected_pose = _blend_navigation_pose(
                                predicted_pose,
                                result.pose,
                                args.navigation_map_correction_gain,
                            )
                            correction_delta = predicted_pose.inverse().compose(
                                result.pose
                            )
                            result = replace(result, pose=corrected_pose)
                            navigation_pose_policy = (
                                "periodic_map_correction_blended"
                            )
                            print(
                                "周期地图校准：累计 {:.2f}m/{:.1f}s，"
                                "温和修正 {:.2f}m/{:+.1f}°（融合 {:.0f}%）。"
                                .format(
                                    navigation_distance_since_correction_m,
                                    update_now - last_map_correction_s,
                                    math.hypot(
                                        correction_delta.x_m,
                                        correction_delta.y_m,
                                    ),
                                    math.degrees(correction_delta.yaw_rad),
                                    args.navigation_map_correction_gain * 100.0,
                                ),
                                flush=True,
                            )
                        last_map_correction_s = update_now
                        navigation_distance_since_correction_m = 0.0
                        if force_fine_correction:
                            fine_map_correction_done = True
                    elif motion_pose is not None:
                        result = replace(result, pose=motion_pose)
                    else:
                        # Never replace a rejected scan-motion increment with
                        # an unchecked local-map result.  That was the direct
                        # cause of the 49--124 degree pose jumps in field run
                        # 20260821_130545: the command-envelope check rejected
                        # ICP, then this fallback immediately accepted a map
                        # candidate from the same wrong basin.  A bounded map
                        # candidate is still useful; an unbounded one simply
                        # holds the previous pose for this frame and lets the
                        # short motor TTL expire before the next scan.
                        fallback_reason = _navigation_pose_jump_reason(
                            pose,
                            result,
                            last_command,
                            update_now - last_pose_update_s,
                            speed_controller.diagnostics()["turn_ttl_ms"],
                            args,
                        )
                        if fallback_reason:
                            result = replace(result, pose=pose)
                            navigation_pose_policy = (
                                "command_bounded_hold:" + fallback_reason
                            )
                            reset_motion_prior = getattr(
                                localizer,
                                "reset_tracking_motion_prior",
                                None,
                            )
                            if reset_motion_prior is not None:
                                reset_motion_prior()
                        else:
                            navigation_pose_policy = "bounded_local_map_fallback"
                    if navigation_pose_correction_holds > 0:
                        recovery_bounded = (
                            _navigation_recovery_candidate_bounded(
                                pose, result, args
                            )
                        )
                        recovery_strong = (
                            _navigation_recovery_candidate_usable(
                                pose, result, args
                            )
                        )
                        if recovery_bounded:
                            navigation_recovery_candidates.append(result)
                        strong_recovery_ready = (
                            recovery_strong
                            and _navigation_recovery_candidates_consistent(
                                navigation_recovery_candidates,
                                args.navigation_recovery_confirmation_frames,
                            )
                        )
                        soft_recovery_ready = (
                            recovery_bounded
                            and _navigation_recovery_candidates_consistent(
                                navigation_recovery_candidates,
                                args.navigation_recovery_confirmation_frames,
                                position_tolerance_m=0.12,
                                yaw_tolerance_deg=8.0,
                            )
                        )
                        forced_soft_recovery = (
                            navigation_pose_correction_holds + 1
                            >= args.navigation_pose_max_correction_holds
                        )
                        if (
                            strong_recovery_ready
                            or soft_recovery_ready
                            or forced_soft_recovery
                        ):
                            selected_recovery = (
                                _best_bounded_recovery_candidate(
                                    pose,
                                    navigation_recovery_candidates,
                                )
                            )
                            if selected_recovery is not None:
                                result = selected_recovery
                                recovery_mode = (
                                    "strong_consensus"
                                    if strong_recovery_ready
                                    else "bounded_soft_consensus"
                                )
                            else:
                                # Do not let the protection mechanism abort a
                                # whole annotation mission.  Keep the trusted
                                # pose for this frame and resume slowly; the
                                # ordinary guard remains active on later scans.
                                result = replace(result, pose=pose)
                                recovery_mode = "trusted_pose_fallback"
                            print(
                                "定位纠错恢复：{}，采用有限小窗口结果并"
                                "继续巡检。".format(recovery_mode),
                                flush=True,
                            )
                            navigation_pose_correction_holds = 0
                            navigation_recovery_candidates.clear()
                        else:
                            navigation_pose_correction_holds += 1
                            client.stop()
                            speed_controller.stopped()
                            last_command = (0, 0)
                            last_command_sent_s = update_now
                            turn_anchor_pose = None
                            navigation_turn_stall_frames = 0
                            reset_motion_prior = getattr(
                                localizer,
                                "reset_tracking_motion_prior",
                                None,
                            )
                            if reset_motion_prior is not None:
                                reset_motion_prior()
                            elif hasattr(localizer, "_last_track_scan"):
                                localizer._last_track_scan = None
                            recovery_reason = (
                                "reanchor_candidate_quality_low"
                                if not recovery_strong
                                else "reanchor_waiting_for_consensus"
                            )
                            _append_navigation_diagnostic(
                                navigation_diagnostics_path,
                                {
                                    "timestamp_s": round(update_now, 3),
                                    "target_id": target.target_id,
                                    "waypoint_index": waypoint_index,
                                    "pose": {
                                        "x_m": pose.x_m,
                                        "y_m": pose.y_m,
                                        "yaw_deg": math.degrees(
                                            pose.yaw_rad
                                        ),
                                    },
                                    "rejected_pose": {
                                        "x_m": result.pose.x_m,
                                        "y_m": result.pose.y_m,
                                        "yaw_deg": math.degrees(
                                            result.pose.yaw_rad
                                        ),
                                    },
                                    "pose_policy": (
                                        "trusted_pose_recovery_wait"
                                    ),
                                    "pose_correction_reason": (
                                        recovery_reason
                                    ),
                                    "pose_correction_holds": (
                                        navigation_pose_correction_holds
                                    ),
                                    "recovery_consistent_candidates": len(
                                        navigation_recovery_candidates
                                    ),
                                    "tracking_motion_prior_reset": True,
                                    "rmse_m": float(result.rmse_m),
                                    "inlier_ratio": float(
                                        result.inlier_ratio
                                    ),
                                },
                            )
                            last_pose_update_s = update_now
                            print(
                                "定位纠错恢复：{}，已停车等待一致候选 "
                                "{}/{}（保护帧 {}/{}）。".format(
                                    recovery_reason,
                                    len(navigation_recovery_candidates),
                                    args.navigation_recovery_confirmation_frames,
                                    navigation_pose_correction_holds,
                                    args.navigation_pose_max_correction_holds,
                                ),
                                flush=True,
                            )
                            continue
                    # The former every-frame jump guard stopped the chassis
                    # and re-anchored repeatedly.  Pose plausibility is now
                    # checked above for each ICP increment, while permanent
                    # map correction runs only on the time/distance schedule.
                    jump_reason = ""
                    if jump_reason:
                        navigation_pose_correction_holds += 1
                        navigation_recovery_candidates.clear()
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        last_command_sent_s = update_now
                        turn_anchor_pose = None
                        navigation_turn_stall_frames = 0
                        reset_motion_prior = getattr(
                            localizer,
                            "reset_tracking_motion_prior",
                            None,
                        )
                        if reset_motion_prior is not None:
                            reset_motion_prior()
                        elif hasattr(localizer, "_last_track_scan"):
                            # Compatibility for injected/older localizers.
                            localizer._last_track_scan = None
                        _append_navigation_diagnostic(
                            navigation_diagnostics_path,
                            {
                                "timestamp_s": round(update_now, 3),
                                "target_id": target.target_id,
                                "waypoint_index": waypoint_index,
                                "pose": {
                                    "x_m": pose.x_m,
                                    "y_m": pose.y_m,
                                    "yaw_deg": math.degrees(pose.yaw_rad),
                                },
                                "rejected_pose": {
                                    "x_m": result.pose.x_m,
                                    "y_m": result.pose.y_m,
                                    "yaw_deg": math.degrees(
                                        result.pose.yaw_rad
                                    ),
                                },
                                "pose_policy": "trusted_pose_correction_hold",
                                "pose_correction_reason": jump_reason,
                                "pose_correction_holds": (
                                    navigation_pose_correction_holds
                                ),
                                "tracking_motion_prior_reset": True,
                                "rmse_m": float(result.rmse_m),
                                "inlier_ratio": float(result.inlier_ratio),
                            },
                        )
                        last_pose_update_s = update_now
                        print(
                            "定位纠错：候选位姿与车辆动作不一致 ({})，"
                            "已停车并保留上一可信位姿 {}/{}。".format(
                                jump_reason,
                                navigation_pose_correction_holds,
                                args.navigation_pose_max_correction_holds,
                            ),
                            flush=True,
                        )
                        continue
                    failures = 0
                    navigation_pose_correction_holds = 0
                    result, turn_anchor_pose, turn_xy_suppressed_m = (
                        _stabilize_navigation_turn_pose(
                            pose,
                            result,
                            last_command,
                            turn_anchor_pose,
                        )
                    )
                    frame_motion = getattr(
                        localizer, "last_track_motion", None
                    )
                    if (
                        last_command is not None
                        and (
                            int(last_command[0]) != 0
                            or int(last_command[1]) != 0
                        )
                        and frame_motion is not None
                        and getattr(frame_motion, "converged", False)
                    ):
                        # Map symmetry/glass can move the absolute pose while
                        # the chassis is stationary. Both turn and linear
                        # torque learning must use the scan-to-scan physical
                        # motion already computed inside track(). In the
                        # 20260915_142137 field run, periodic global-map
                        # correction otherwise hid a fully jammed 450 mm/s
                        # forward arc for several minutes.
                        measured_yaw_delta_rad = normalize_angle(
                            frame_motion.transform.yaw_rad
                        )
                        motion_feedback = _motion_update(
                            frame_motion,
                            update_now - last_pose_update_s,
                        )
                        if motion_pose_reason:
                            # Do not teach the adaptive controller that an
                            # impossible scan alias was real chassis motion.
                            motion_feedback = replace(
                                motion_feedback,
                                accepted=False,
                                rejection_reason=(
                                    "command_envelope:" + motion_pose_reason
                                ),
                            )
                            measured_yaw_delta_rad = 0.0
                    else:
                        measured_yaw_delta_rad = normalize_angle(
                            result.pose.yaw_rad
                            - previous_navigation_pose.yaw_rad
                        )
                        motion_feedback = _tracking_update(
                            pose,
                            result,
                            update_now - last_pose_update_s,
                        )
                        if motion_pose_reason:
                            motion_feedback = replace(
                                motion_feedback,
                                accepted=False,
                                rejection_reason=(
                                    "command_envelope:" + motion_pose_reason
                                ),
                            )
                    if (
                        precision_heading_remaining_rad is not None
                        and last_command is not None
                        and int(last_command[0]) == 0
                        and int(last_command[1]) != 0
                    ):
                        heading_progress_rad, heading_source = (
                            _precision_heading_progress(
                                int(last_command[1]),
                                frame_motion,
                                args.navigation_precision_turn_ttl_ms,
                                update_now - last_pose_update_s,
                                precision_heading_response_gain,
                            )
                        )
                        if (
                            heading_source == "scan_motion"
                            and abs(heading_progress_rad)
                            >= math.radians(1.0)
                        ):
                            nominal_yaw_rad = (
                                abs(int(last_command[1])) / 1000.0
                                * args.navigation_precision_turn_ttl_ms
                                / 1000.0
                            )
                            if nominal_yaw_rad > 1e-6:
                                observed_gain = max(
                                    0.35,
                                    min(
                                        1.25,
                                        abs(heading_progress_rad)
                                        / nominal_yaw_rad,
                                    ),
                                )
                                precision_heading_response_gain = (
                                    0.70 * precision_heading_response_gain
                                    + 0.30 * observed_gain
                                )
                        precision_heading_remaining_rad = normalize_angle(
                            precision_heading_remaining_rad
                            - heading_progress_rad
                        )
                        if heading_source == "unverified_no_progress":
                            # The pulse may have moved even though adjacent
                            # scan ICP was unusable. Continuing in the same
                            # direction would over-turn. Stop and rebuild the
                            # remaining angle from a few stationary map frames.
                            precision_heading_remaining_rad = None
                            precision_heading_initial_error_rad = None
                            precision_heading_seed_samples.clear()
                            precision_heading_last_source = (
                                "motion_unverified_reseed"
                            )
                        else:
                            precision_heading_last_source = heading_source
                    speed_controller.observe(motion_feedback)
                    last_pose_update_s = update_now
                    pose = result.pose
                    pose_trail.append(pose)
                    if update_now - last_plan_render_s >= 1.0:
                        # 实时刷新计划图: 绿色大点=当前定位, 浅绿轨迹=走过的位置
                        report.save_plan(
                            known_map, confirmed_start_pose, ordered,
                            live_pose=pose, live_trail=list(pose_trail),
                        )
                        last_plan_render_s = update_now
                    if (
                        not final_translation_complete
                        and waypoint_index == len(waypoints) - 1
                    ):
                        approximate_distance_m = math.hypot(
                            target.view_xy_m[0] - pose.x_m,
                            target.view_xy_m[1] - pose.y_m,
                        )
                        (
                            reference_capture_ok,
                            reference_capture_distance_m,
                            reference_capture_limit_m,
                        ) = _within_capture_reference(
                            pose,
                            target.target_id,
                            capture_reference_views,
                            verified_capture_poses,
                            args.nearby_capture_distance,
                        )
                        if (
                            approximate_distance_m
                            <= args.nearby_capture_distance
                            and reference_capture_ok
                        ):
                            # Inspection is evidence collection, not docking.
                            # Once the car is in the useful neighbourhood,
                            # stop translating and spend the remaining effort
                            # only on pointing the camera at the mapped subject.
                            final_translation_complete = True
                            capture_mode = (
                                "alternate_view_nearby_position"
                                if target.target_id in alternate_views
                                else "nearby_position"
                            )
                            nearby_note = (
                                "accepted approximate inspection position "
                                "{:.3f}m from the active view and {:.3f}m "
                                "from the reference view (limit {:.3f}m); "
                                "final control "
                                "prioritized camera heading"
                            ).format(
                                approximate_distance_m,
                                reference_capture_distance_m,
                                reference_capture_limit_m,
                            )
                            capture_note = "; ".join(
                                item for item in (
                                    capture_note, nearby_note
                                ) if item
                            )
                            client.stop()
                            speed_controller.stopped()
                            last_command = (0, 0)
                            last_command_sent_s = update_now
                            turn_anchor_pose = None
                            print(
                                "目标 {} 已到达近似拍摄区域（距计划视点 "
                                "{:.2f}m）；不再追求精确位置，只调整相机"
                                "朝向。".format(
                                    target.target_id,
                                    approximate_distance_m,
                                ),
                                flush=True,
                            )
                    verified_pose = verified_capture_poses.get(
                        target.target_id
                    )
                    use_verified_yaw = bool(
                        verified_pose is not None
                        and math.hypot(
                            pose.x_m - verified_pose.x_m,
                            pose.y_m - verified_pose.y_m,
                        ) <= 0.32
                    )
                    final_yaw = (
                        (
                            verified_pose.yaw_rad
                            if use_verified_yaw
                            else normalize_angle(
                                math.atan2(
                                    target.target_xy_m[1] - pose.y_m,
                                    target.target_xy_m[0] - pose.x_m,
                                )
                                - math.radians(args.camera_yaw_offset_deg)
                            )
                        )
                        if (
                            waypoint_index == len(waypoints) - 1
                            or final_translation_complete
                        )
                        else None
                    )
                    command_waypoint = (
                        (pose.x_m, pose.y_m)
                        if final_translation_complete else waypoint
                    )
                    precision_control = (
                        final_yaw is not None
                        and math.hypot(
                            command_waypoint[0] - pose.x_m,
                            command_waypoint[1] - pose.y_m,
                        ) <= args.waypoint_tolerance
                    )
                    if (
                        precision_control
                        and precision_heading_remaining_rad is None
                    ):
                        precision_heading_seed_samples.append(
                            normalize_angle(final_yaw - pose.yaw_rad)
                        )
                        if (
                            len(precision_heading_seed_samples)
                            >= args.navigation_heading_seed_frames
                        ):
                            precision_heading_remaining_rad = (
                                _circular_heading_seed(
                                    precision_heading_seed_samples
                                )
                            )
                            precision_heading_initial_error_rad = (
                                precision_heading_remaining_rad
                            )
                            precision_heading_last_source = (
                                "stopped_map_heading_average"
                            )
                    if precision_control:
                        heading_tolerance_rad = math.radians(
                            args.heading_tolerance_deg
                        )
                        if precision_heading_remaining_rad is None:
                            # One bad glass-wall frame must not define the
                            # camera direction for the whole final turn.  The
                            # car is already stopped here; collect only a few
                            # frames, then use relative scan yaw for motion.
                            linear, angular, reached = 0, 0, False
                        elif (
                            abs(precision_heading_remaining_rad)
                            <= heading_tolerance_rad
                        ):
                            linear, angular, reached = 0, 0, True
                        else:
                            angular = int(math.copysign(
                                max(
                                    args.navigation_fine_turn_min_speed,
                                    min(
                                        args.navigation_fine_turn_speed,
                                        abs(precision_heading_remaining_rad)
                                        * 900.0,
                                    ),
                                ),
                                precision_heading_remaining_rad,
                            ))
                            linear, reached = 0, False
                    else:
                        blocked_heading_command = None
                        if target.target_id in blocked_path_heading_targets:
                            blocked_heading_command = (
                                _blocked_path_heading_command(
                                    args, pose, command_waypoint
                                )
                            )
                            if blocked_heading_command is None:
                                blocked_path_heading_targets.discard(
                                    target.target_id
                                )
                        if blocked_heading_command is not None:
                            linear, angular, reached = blocked_heading_command
                        else:
                            linear, angular, reached = _command_for_pose(
                                args,
                                pose,
                                command_waypoint,
                                final_yaw=final_yaw,
                                force_forward_arc=(
                                    target.target_id
                                    in force_forward_arc_targets
                                ),
                            )
                    front_m = None
                    if linear != 0:
                        front_m = sector_min_distance(
                            scan, center_deg=0.0, width_deg=42.0,
                            min_distance_m=0.08, max_distance_m=8.0,
                        )
                        distance_to_target_view_m = math.hypot(
                            target.view_xy_m[0] - pose.x_m,
                            target.view_xy_m[1] - pose.y_m,
                        )
                        linear = _limit_navigation_linear_request(
                            linear,
                            front_m,
                            distance_to_target_view_m,
                            waypoint_index == len(waypoints) - 1,
                            args.slow_speed,
                            args.navigation_fine_control_distance,
                            speed_controller.diagnostics()[
                                "linear_motion_confirmed"
                            ],
                        )
                    requested = MotionCommand(
                        linear,
                        angular,
                        (
                            "advancing"
                            if linear
                            else (
                                "precision_turning"
                                if angular != 0 and precision_control
                                else "turning"
                            )
                        ),
                        "known-map waypoint tracking",
                        waypoint,
                    )
                    applied = speed_controller.apply(requested)
                    linear = applied.linear_mm_s
                    angular = applied.angular_mrad_s
                    if (
                        requested.linear_mm_s == 0
                        and requested.angular_mrad_s != 0
                    ):
                        capture_allowed, distance_to_view_m = (
                            _nearby_turn_timeout_capture_allowed(
                                pose,
                                capture_reference_views[target.target_id],
                                waypoint_index,
                                len(waypoints),
                                final_translation_complete,
                                args.nearby_capture_distance,
                            )
                        )
                        if capture_allowed:
                            if pure_turn_started_s is None:
                                pure_turn_started_s = time.monotonic()
                            pure_turn_elapsed_s = (
                                time.monotonic() - pure_turn_started_s
                            )
                        else:
                            # This is a normal path-heading turn.  Preserve
                            # the previously successful navigation behavior;
                            # the ordinary measured stall detector below is
                            # its only escape trigger.
                            pure_turn_started_s = None
                            pure_turn_elapsed_s = 0.0
                        capture_heading_error_rad = (
                            precision_heading_remaining_rad
                            if precision_heading_remaining_rad is not None
                            else (
                                None
                                if final_yaw is None
                                else normalize_angle(
                                    final_yaw - pose.yaw_rad
                                )
                            )
                        )
                        timeout_action = _navigation_turn_timeout_action(
                            pure_turn_elapsed_s,
                            _final_heading_timeout_seconds(
                                args.navigation_pure_turn_timeout_seconds,
                                precision_heading_initial_error_rad,
                            ),
                            capture_allowed,
                            heading_error_rad=capture_heading_error_rad,
                            maximum_capture_error_deg=(
                                args.navigation_timeout_capture_max_heading_error_deg
                            ),
                        )
                        if timeout_action == "replan":
                            client.stop()
                            speed_controller.stopped()
                            last_command = (0, 0)
                            last_command_sent_s = time.monotonic()
                            turn_anchor_pose = None
                            retry_key = "heading:" + target.target_id
                            heading_retry = turn_escape_attempts.get(
                                retry_key, 0
                            )
                            if (
                                target_attempts[target.target_id]
                                < args.max_target_navigation_attempts
                            ):
                                turn_escape_attempts[retry_key] = (
                                    heading_retry + 1
                                )
                                target_retry_origins[target.target_id] = (
                                    pose.x_m, pose.y_m
                                )
                                ordered_targets.append(target)
                                print(
                                    "目标 {} 最终朝向误差仍为 {:.1f}°，"
                                    "拒绝反向/侧向代拍；先继续巡检其他点，"
                                    "稍后从新位置重试（第 {}/{} 次）。".format(
                                        target.target_id,
                                        abs(math.degrees(
                                            capture_heading_error_rad
                                        )),
                                        target_attempts[target.target_id],
                                        args.max_target_navigation_attempts,
                                    ),
                                    flush=True,
                                )
                                replan_target = True
                            else:
                                report.add_skipped_target(
                                    target,
                                    pose,
                                    "final camera heading remained outside "
                                    "the allowed envelope after two attempts; "
                                    "no misdirected photo was taken",
                                    target_attempts[target.target_id],
                                )
                                print(
                                    "目标 {} 连续 {} 次最终朝向均未进入 {:.0f}°"
                                    "范围；不反向/侧向代拍、不计入完成数，"
                                    "已加入未覆盖清单。".format(
                                        target.target_id,
                                        args.max_target_navigation_attempts,
                                        args.navigation_timeout_capture_max_heading_error_deg,
                                    ),
                                    flush=True,
                                )
                                replan_target = True
                            break
                        if timeout_action == "capture":
                            client.stop()
                            speed_controller.stopped()
                            last_command = (0, 0)
                            last_command_sent_s = time.monotonic()
                            turn_anchor_pose = None
                            target_ready_for_capture = True
                            capture_mode = (
                                "alternate_view_nearby_turn_timeout"
                                if target.target_id in alternate_views
                                else "nearby_turn_timeout"
                            )
                            timeout_note = (
                                "vehicle was within {:.3f}m of the view; "
                                "final heading made no reliable progress "
                                "for {:.1f}s, so motion stopped and the "
                                "current nearby view was captured"
                            ).format(
                                distance_to_view_m,
                                pure_turn_elapsed_s,
                            )
                            capture_note = "; ".join(
                                item for item in (
                                    capture_note, timeout_note
                                ) if item
                            )
                            print(
                                "目标 {} 已在拍摄位附近（距视点 {:.2f}m），"
                                "最终转向 {:.1f}s 无可靠进展；已停车并按"
                                "当前位置直接拍照。".format(
                                    target.target_id,
                                    distance_to_view_m,
                                    pure_turn_elapsed_s,
                                ),
                                flush=True,
                            )
                            break
                    else:
                        pure_turn_started_s = None
                    # The controller now receives the already limited request,
                    # so its returned command exactly matches what is sent to
                    # the chassis.  Check clearance against that real output,
                    # including a bounded breakaway kick when one is needed.
                    obstacle_guard_linear = linear
                    speed_diagnostics = speed_controller.diagnostics()
                    command_ttl_ms = _inspection_navigation_command_ttl(
                        speed_controller,
                        applied,
                        args.command_ttl_ms,
                        front_m,
                        args.stop_distance,
                    )
                    if requested.state == "precision_turning":
                        command_ttl_ms = min(
                            command_ttl_ms,
                            args.navigation_precision_turn_ttl_ms,
                        )
                    if _navigation_turn_stall_observed(
                        last_command,
                        measured_yaw_delta_rad,
                        applied,
                        speed_diagnostics,
                        speed_controller.config,
                    ):
                        navigation_turn_stall_frames += 1
                    else:
                        navigation_turn_stall_frames = 0
                    if _navigation_linear_stall_observed(
                        last_command,
                        motion_feedback,
                        applied,
                        speed_diagnostics,
                        speed_controller.maximum_linear_speed_mm_s,
                        speed_controller.config,
                    ):
                        navigation_linear_stall_frames += 1
                    else:
                        navigation_linear_stall_frames = 0
                    if (
                        last_command is not None
                        and (
                            int(last_command[0]) != 0
                            or int(last_command[1]) != 0
                        )
                        and bool(getattr(motion_feedback, "accepted", False))
                        and float(motion_feedback.translation_m)
                        < speed_controller.config.low_frame_translation_m
                        and abs(float(motion_feedback.rotation_rad))
                        < math.radians(
                            speed_controller.config.low_frame_rotation_deg
                        )
                    ):
                        navigation_stationary_command_frames += 1
                    else:
                        navigation_stationary_command_frames = 0
                    _append_navigation_diagnostic(
                        navigation_diagnostics_path,
                        {
                            "timestamp_s": round(time.monotonic(), 3),
                            "target_id": target.target_id,
                            "waypoint_index": waypoint_index,
                            "pose": {
                                "x_m": pose.x_m,
                                "y_m": pose.y_m,
                                "yaw_deg": math.degrees(pose.yaw_rad),
                            },
                            "waypoint": [float(waypoint[0]), float(waypoint[1])],
                            "requested": [int(requested.linear_mm_s), int(requested.angular_mrad_s)],
                            "applied": [int(linear), int(angular)],
                            "reached": bool(reached),
                            "rmse_m": (
                                float(result.rmse_m)
                                if result is not None
                                else float("nan")
                            ),
                            "inlier_ratio": (
                                float(result.inlier_ratio)
                                if result is not None else float("nan")
                            ),
                            "pose_policy": (
                                navigation_pose_policy
                                + (
                                    "+turn_xy_anchored"
                                    if turn_anchor_pose is not None
                                    else ""
                                )
                            ),
                            "turn_xy_suppressed_m": float(
                                turn_xy_suppressed_m
                            ),
                            "precision_heading_remaining_deg": (
                                None
                                if precision_heading_remaining_rad is None
                                else math.degrees(
                                    precision_heading_remaining_rad
                                )
                            ),
                            "precision_heading_source": (
                                precision_heading_last_source
                            ),
                            "precision_heading_response_gain": float(
                                precision_heading_response_gain
                            ),
                            "front_clearance_m": (
                                float(front_m)
                                if front_m is not None else None
                            ),
                            "obstacle_sector_width_deg": 42.0,
                            "obstacle_guard_linear_mm_s": int(
                                obstacle_guard_linear
                            ),
                            "command_ttl_ms": int(command_ttl_ms),
                            "linear_adjustment_mm_s": int(
                                speed_diagnostics[
                                    "linear_adjustment_mm_s"
                                ]
                            ),
                            "turn_floor_mrad_s": int(
                                speed_diagnostics["turn_floor_mrad_s"]
                            ),
                            "turn_limit_mrad_s": int(
                                speed_diagnostics["turn_limit_mrad_s"]
                            ),
                            "turn_stall_events": int(
                                speed_diagnostics["turn_stall_events"]
                            ),
                            "turn_ttl_ms": int(
                                speed_diagnostics["turn_ttl_ms"]
                            ),
                            "navigation_turn_stall_frames": int(
                                navigation_turn_stall_frames
                            ),
                            "navigation_linear_stall_frames": int(
                                navigation_linear_stall_frames
                            ),
                            "navigation_stationary_command_frames": int(
                                navigation_stationary_command_frames
                            ),
                        },
                    )
                    obstacle_safe = _obstacle_safe(
                        scan,
                        obstacle_guard_linear,
                        args.stop_distance,
                        angular,
                        full_scan_guard=True,
                        command_ttl_ms=command_ttl_ms,
                    )
                    if (
                        not obstacle_safe
                        and linear > 0
                        and _inspection_standoff_reached(
                            pose,
                            target.view_xy_m,
                            front_m,
                            args.waypoint_tolerance,
                            args.stop_distance,
                            args.nearby_capture_distance,
                        )
                        and _within_capture_reference(
                            pose,
                            target.target_id,
                            capture_reference_views,
                            verified_capture_poses,
                            args.nearby_capture_distance,
                        )[0]
                    ):
                        distance_to_view_m = math.hypot(
                            target.view_xy_m[0] - pose.x_m,
                            target.view_xy_m[1] - pose.y_m,
                        )
                        final_translation_complete = True
                        capture_mode = (
                            "alternate_view_safe_standoff"
                            if target.target_id in alternate_views
                            else "safe_inspection_standoff"
                        )
                        standoff_note = (
                            "mapped subject entered the transit safety envelope; "
                            "stopped {:.3f}m before the planned view and turned "
                            "to the target using the current finite pose"
                        ).format(distance_to_view_m)
                        capture_note = "; ".join(
                            item for item in (capture_note, standoff_note)
                            if item
                        )
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        last_command_sent_s = time.monotonic()
                        print(
                            "目标 {} 已进入安全拍摄距离（距计划视点 {:.2f}m，"
                            "前向净空 {:.2f}m）；停止平移，按当前定位转向后直接拍照。"
                            .format(
                                target.target_id,
                                distance_to_view_m,
                                front_m,
                            ),
                            flush=True,
                        )
                        continue
                    if not obstacle_safe:
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        last_command_sent_s = time.monotonic()
                        if (
                            linear > 0
                            and front_m is not None
                            and front_m <= args.emergency_front_distance
                        ):
                            print(
                                "前方仅 {:.2f}m，已触发碰墙脱离保护；"
                                "先停车，再验证雷达后方和地图倒车走廊。".format(
                                    front_m
                                ),
                                flush=True,
                            )
                            reverse_scan, reversed_ok, reversed_m, reverse_reason = (
                                _execute_collision_reverse_escape(
                                    client,
                                    scan_iterator,
                                    scan,
                                    pose,
                                    known_map,
                                    args,
                                    localizer,
                                )
                            )
                            scan = reverse_scan
                            if reversed_ok:
                                predicted_pose = Pose2D(
                                    pose.x_m - reversed_m * math.cos(pose.yaw_rad),
                                    pose.y_m - reversed_m * math.sin(pose.yaw_rad),
                                    pose.yaw_rad,
                                )
                                reverse_result = localizer.track(
                                    scan, predicted_pose
                                )
                                if (
                                    _tracking_pose_available(reverse_result)
                                    and math.hypot(
                                        reverse_result.pose.x_m - predicted_pose.x_m,
                                        reverse_result.pose.y_m - predicted_pose.y_m,
                                    ) <= 0.20
                                    and abs(normalize_angle(
                                        reverse_result.pose.yaw_rad
                                        - predicted_pose.yaw_rad
                                    )) <= math.radians(20.0)
                                ):
                                    pose = reverse_result.pose
                                else:
                                    pose = predicted_pose
                                pose_trail.append(pose)
                                print(
                                    "紧急后退完成：已离开墙面约 {:.2f}m，"
                                    "将从停车位置重新规划。".format(reversed_m),
                                    flush=True,
                                )
                            else:
                                print(
                                    "未执行完整后退（已退约 {:.2f}m，原因={}）；"
                                    "车辆保持停车并改用其他拍摄位。".format(
                                        reversed_m, reverse_reason
                                    ),
                                    flush=True,
                                )
                        # A range mismatch is not semantic obstacle evidence.
                        # With even a small pose error, a real mapped wall is
                        # projected into free space and used to poison every
                        # later A* query. Obstacle scans therefore stop only
                        # the current motion pulse; they are never persisted
                        # into the mission-wide planning layer.
                        known_map.clear_temporary_obstacles()
                        # The next route attempt must first point into its
                        # newly planned segment while stopped. Reusing a
                        # forward-arc stiction escape here repeatedly drives
                        # the same corner and exhausts every alternate view.
                        force_forward_arc_targets.discard(target.target_id)
                        blocked_path_heading_targets.add(target.target_id)
                        print(
                            "前方受阻：仅停止当前动作，不把本帧雷达差异写入"
                            "全局临时地图。将就近换机位或记录覆盖照片。",
                            flush=True,
                        )
                        attempt = target_attempts[target.target_id]
                        alternate_state = alternate_views.get(
                            target.target_id
                        )
                        if attempt < args.max_target_navigation_attempts:
                            excluded_views = (
                                []
                                if alternate_state is None
                                else alternate_state["tried_views"]
                            )
                            replacement = _alternate_inspection_target(
                                known_map,
                                pose,
                                target,
                                args.alternate_view_max_shift,
                                excluded_views=excluded_views,
                            )
                        else:
                            replacement = None
                        if replacement is not None:
                            if alternate_state is None:
                                alternate_state = {
                                    "original_view": target.view_xy_m,
                                    "tried_views": [target.view_xy_m],
                                }
                                alternate_views[target.target_id] = (
                                    alternate_state
                                )
                            alternate_state["tried_views"].append(
                                replacement.view_xy_m
                            )
                            alternate_state["alternate_view"] = (
                                replacement.view_xy_m
                            )
                            # The replacement was map-validated to see the
                            # same subject.  Arrival checks must therefore use
                            # this active view, not the first blocked one.
                            capture_reference_views[target.target_id] = (
                                replacement.view_xy_m
                            )
                            target_retry_origins[target.target_id] = (
                                pose.x_m, pose.y_m
                            )
                            ordered_targets.append(replacement)
                            replan_target = True
                            print(
                                "目标 {} 原机位受阻，已找到同一标注物的"
                                "附近替代机位 ({:+.2f},{:+.2f})；稍后从"
                                "当前位置重新规划。".format(
                                    target.target_id,
                                    replacement.view_xy_m[0],
                                    replacement.view_xy_m[1],
                                ),
                                flush=True,
                            )
                        elif attempt >= args.max_target_navigation_attempts:
                            report.add_skipped_target(
                                target,
                                pose,
                                "all map-validated views remained blocked; "
                                "no remote proxy photo was taken",
                                attempt,
                            )
                            replan_target = True
                            print(
                                "目标 {} 的原机位和 {} 个同物体替代机位仍"
                                "受阻；不在当前位置代拍、不计入完成数，已加入"
                                "未覆盖清单并继续其他目标。".format(
                                    target.target_id,
                                    max(0, attempt - 1),
                                ),
                                flush=True,
                            )
                            break
                        else:
                            target_retry_origins[target.target_id] = (
                                pose.x_m, pose.y_m
                            )
                            ordered_targets.append(target)
                            replan_target = True
                            print(
                                "目标 {} 将在完成其他可达点后重新规划（第 {}/{} 次）。"
                                .format(
                                    target.target_id,
                                    attempt,
                                    args.max_target_navigation_attempts,
                                ),
                                flush=True,
                            )
                        break
                    if (
                        navigation_linear_stall_frames
                        >= args.navigation_linear_stall_at_max_frames
                    ):
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        last_command_sent_s = time.monotonic()
                        turn_anchor_pose = None
                        clear_restart_ttl_ms = _navigation_command_ttl(
                            args.command_ttl_ms,
                            args.start_kick_speed,
                            front_m,
                            args.stop_distance,
                        )
                        clear_restart_description = (
                            "未知"
                            if front_m is None
                            or not math.isfinite(float(front_m))
                            else "{:.2f}m".format(float(front_m))
                        )
                        print(
                            "巡检前进连续 {} 个可信雷达帧尚未建立足够"
                            "位移，但前方雷达净空 {}，未判定撞墙。"
                            "保持当前目标和路径方向不变，下次启动脉冲按"
                            "净空放宽至 {}ms，不执行后退或左右换向。".format(
                                navigation_linear_stall_frames,
                                clear_restart_description,
                                clear_restart_ttl_ms,
                            ),
                            flush=True,
                        )
                        navigation_linear_stall_frames = 0
                        navigation_stationary_command_frames = 0
                        # Stay inside the same target/waypoint loop. The next
                        # fresh lidar frame re-arms the one-shot start kick;
                        # only the ordinary radar obstacle branch above may
                        # replan or reverse this route.
                        continue
                    if (
                        navigation_turn_stall_frames
                        >= args.navigation_turn_stall_at_max_frames
                    ):
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        last_command_sent_s = time.monotonic()
                        turn_anchor_pose = None
                        escape_attempt = turn_escape_attempts.get(
                            target.target_id, 0
                        )
                        print(
                            "巡检导航转向达到最大输出，连续 {} 帧航向几乎"
                            "不变；车辆已停车。".format(
                                navigation_turn_stall_frames
                            ),
                            flush=True,
                        )
                        if (
                            escape_attempt
                            < args.navigation_turn_escape_max_attempts
                        ):
                            turn_escape_attempts[target.target_id] = (
                                escape_attempt + 1
                            )
                            escape_deadline_s = min(
                                started + args.max_runtime_min * 60.0,
                                time.monotonic() + 10.0,
                            )
                            escape_scan, escaped, _distance, escape_reason = (
                                _execute_navigation_turn_escape(
                                    client,
                                    scan_iterator,
                                    localizer,
                                    speed_controller,
                                    scan,
                                    args,
                                    escape_deadline_s,
                                )
                            )
                            scan = escape_scan
                            if escaped:
                                escape_pose_result = localizer.track(scan, pose)
                                if _tracking_pose_available(escape_pose_result):
                                    pose = escape_pose_result.pose
                                    pose_trail.append(pose)
                                last_pose_update_s = time.monotonic()
                                if precision_control:
                                    force_forward_arc_targets.discard(
                                        target.target_id
                                    )
                                else:
                                    force_forward_arc_targets.add(
                                        target.target_id
                                    )
                                target_retry_origins[target.target_id] = (
                                    pose.x_m, pose.y_m
                                )
                                ordered_targets.append(target)
                                replan_target = True
                                print(
                                    "目标 {} 已完成一次安全短移脱困，将从"
                                    "当前位置重新规划{}；该目标"
                                    "不再执行第二次原地转向脱困。"
                                    .format(
                                        target.target_id,
                                        (
                                            "并等待更合适的抵达朝向"
                                            if precision_control
                                            else "并改用前进弧线"
                                        ),
                                    ),
                                    flush=True,
                                )
                            else:
                                nearby_capture_ok, stalled_distance_m = (
                                    _nearby_turn_timeout_capture_allowed(
                                        pose,
                                        capture_reference_views[
                                            target.target_id
                                        ],
                                        waypoint_index,
                                        len(waypoints),
                                        final_translation_complete,
                                        args.nearby_capture_distance,
                                    )
                                )
                                reference_ok = _within_capture_reference(
                                    pose,
                                    target.target_id,
                                    capture_reference_views,
                                    verified_capture_poses,
                                    args.nearby_capture_distance,
                                )[0]
                                heading_ok, stalled_heading_error_deg = (
                                    _capture_heading_within_limit(
                                        pose,
                                        final_yaw,
                                        precision_heading_remaining_rad,
                                        args.navigation_timeout_capture_max_heading_error_deg,
                                    )
                                )
                                if (
                                    nearby_capture_ok
                                    and reference_ok
                                    and heading_ok
                                ):
                                    target_ready_for_capture = True
                                    replan_target = False
                                    capture_mode = (
                                        "nearby_turn_stalled_capture"
                                    )
                                    capture_note = "; ".join(
                                        item for item in (
                                            capture_note,
                                            "turn escape failed ({}), but "
                                            "the vehicle was already {:.3f}m "
                                            "from the verified view".format(
                                                escape_reason,
                                                stalled_distance_m,
                                            ),
                                        ) if item
                                    )
                                    print(
                                        "目标 {} 已在近似拍摄范围内，转向脱困"
                                        "失败后停车拍照。".format(
                                            target.target_id
                                        ),
                                        flush=True,
                                    )
                                else:
                                    replan_target = True
                                    if (
                                        target_attempts[target.target_id]
                                        >= args.max_target_navigation_attempts
                                    ):
                                        report.add_skipped_target(
                                            target,
                                            pose,
                                            "turn stalled outside a valid "
                                            "capture pose or heading; no photo "
                                            "was taken",
                                            target_attempts[target.target_id],
                                        )
                                        print(
                                            "目标 {} 转向脱困后仍不满足拍摄"
                                            "条件（距离 {:.2f}m，朝向误差 "
                                            "{:.1f}°）；不计入完成数。".format(
                                                target.target_id,
                                                stalled_distance_m,
                                                stalled_heading_error_deg,
                                            ),
                                            flush=True,
                                        )
                                    else:
                                        if precision_control:
                                            force_forward_arc_targets.discard(
                                                target.target_id
                                            )
                                        else:
                                            force_forward_arc_targets.add(
                                                target.target_id
                                            )
                                        target_retry_origins[target.target_id] = (
                                            pose.x_m, pose.y_m
                                        )
                                        ordered_targets.append(target)
                                        print(
                                            "目标 {} 尚未满足拍摄条件（距离 "
                                            "{:.2f}m，朝向误差 {:.1f}°）；"
                                            "重新规划{}。".format(
                                                target.target_id,
                                                stalled_distance_m,
                                                stalled_heading_error_deg,
                                                (
                                                    "并等待更合适的抵达朝向"
                                                    if precision_control
                                                    else "并改用前进弧线"
                                                ),
                                            ),
                                            flush=True,
                                        )
                        else:
                            nearby_capture_ok, stalled_distance_m = (
                                _nearby_turn_timeout_capture_allowed(
                                    pose,
                                    capture_reference_views[
                                        target.target_id
                                    ],
                                    waypoint_index,
                                    len(waypoints),
                                    final_translation_complete,
                                    args.nearby_capture_distance,
                                )
                            )
                            reference_ok = _within_capture_reference(
                                pose,
                                target.target_id,
                                capture_reference_views,
                                verified_capture_poses,
                                args.nearby_capture_distance,
                            )[0]
                            heading_ok, stalled_heading_error_deg = (
                                _capture_heading_within_limit(
                                    pose,
                                    final_yaw,
                                    precision_heading_remaining_rad,
                                    args.navigation_timeout_capture_max_heading_error_deg,
                                )
                            )
                            if (
                                nearby_capture_ok
                                and reference_ok
                                and heading_ok
                            ):
                                target_ready_for_capture = True
                                replan_target = False
                                capture_mode = "nearby_turn_stalled_capture"
                                capture_note = "; ".join(
                                    item for item in (
                                        capture_note,
                                        "turn stalled after one escape, but "
                                        "the verified view was already near",
                                    ) if item
                                )
                                print(
                                    "目标 {} 已在近似拍摄范围内；最终微调"
                                    "堵转，停车拍照并继续。".format(
                                        target.target_id
                                    ),
                                    flush=True,
                                )
                            else:
                                replan_target = True
                                if (
                                    target_attempts[target.target_id]
                                    >= args.max_target_navigation_attempts
                                ):
                                    report.add_skipped_target(
                                        target,
                                        pose,
                                        "turn stalled outside a valid capture "
                                        "pose or heading; no photo was taken",
                                        target_attempts[target.target_id],
                                    )
                                    print(
                                        "目标 {} 最终转向仍不满足拍摄条件"
                                        "（距离 {:.2f}m，朝向误差 {:.1f}°）；"
                                        "不拍照、不计入完成数。".format(
                                            target.target_id,
                                            stalled_distance_m,
                                            stalled_heading_error_deg,
                                        ),
                                        flush=True,
                                    )
                                else:
                                    if precision_control:
                                        force_forward_arc_targets.discard(
                                            target.target_id
                                        )
                                    else:
                                        force_forward_arc_targets.add(
                                            target.target_id
                                        )
                                    target_retry_origins[target.target_id] = (
                                        pose.x_m, pose.y_m
                                    )
                                    ordered_targets.append(target)
                                    print(
                                        "目标 {} 尚未满足拍摄条件（距离 "
                                        "{:.2f}m，朝向误差 {:.1f}°）；"
                                        "{}重试。".format(
                                            target.target_id,
                                            stalled_distance_m,
                                            stalled_heading_error_deg,
                                            (
                                                "等待更合适的抵达朝向"
                                                if precision_control
                                                else "改用前进弧线"
                                            ),
                                        ),
                                        flush=True,
                                    )
                        navigation_turn_stall_frames = 0
                        break
                    command = (linear, angular)
                    if reached:
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        last_command_sent_s = time.monotonic()
                        if final_translation_complete:
                            target_ready_for_capture = True
                        break
                    now = time.monotonic()
                    if now - last_status_print_s >= 1.0:
                        print(
                            "巡检导航：pose=({:+.2f},{:+.2f},{:+.1f}°) "
                            "cmd=({:+d},{:+d}) target=({:+.2f},{:+.2f}) "
                            "linear_adjust={:+d}。".format(
                                pose.x_m,
                                pose.y_m,
                                math.degrees(pose.yaw_rad),
                                linear,
                                angular,
                                waypoint[0],
                                waypoint[1],
                                speed_controller.linear_adjustment_mm_s,
                            ),
                            flush=True,
                        )
                        last_status_print_s = now
                    direction_changed = (
                        last_command is None
                        or _direction(linear) != _direction(last_command[0])
                        or _direction(angular) != _direction(last_command[1])
                    )
                    command_due = (
                        direction_changed
                        or now - last_command_sent_s
                        >= args.command_min_interval_ms / 1000.0
                    )
                    if command_due:
                        if command == last_command:
                            refresh = getattr(client, "refresh_twist", None)
                            if refresh is not None:
                                refresh(
                                    linear, angular, ttl_ms=command_ttl_ms
                                )
                            else:
                                client.set_twist(
                                    linear, angular, ttl_ms=command_ttl_ms
                                )
                        else:
                            client.set_twist(
                                linear, angular, ttl_ms=command_ttl_ms
                            )
                        if linear != 0:
                            # Require a fresh scan before another drive pulse.
                            allow_expiry = getattr(
                                client, "allow_twist_to_expire", None
                            )
                            if allow_expiry is not None:
                                allow_expiry()
                        elif angular != 0:
                            # Turns remain one pulse per fresh lidar frame.
                            allow_expiry = getattr(
                                client,
                                "allow_twist_to_expire",
                                None,
                            )
                            if allow_expiry is not None:
                                allow_expiry()
                        last_command = command
                        last_command_sent_s = now
                if replan_target:
                    break
                if target_ready_for_capture:
                    break
            if replan_target:
                continue
            if stopping:
                break
            time.sleep(args.settle_seconds)
            # The final waypoint command already required both position and
            # camera heading tolerance. Use that latest finite pose directly;
            # do not run a second RMSE/inlier decision before taking a photo.
            capture_index = completed_targets + 1
            image_name = "{:02d}_{}_{}_p01.jpg".format(
                capture_index, target.kind, target.target_id
            )
            image_path = report.directory / image_name
            camera_info = camera.capture(image_path)
            record = {
                "target_id": target.target_id,
                "kind": target.kind,
                "image": image_name,
                "photo_index": 1,
                "photos_per_target": args.photos_per_target,
                "x_m": round(pose.x_m, 4),
                "y_m": round(pose.y_m, 4),
                "yaw_deg": round(math.degrees(pose.yaw_rad), 2),
                "capture_heading_error_estimate_deg": (
                    None
                    if precision_heading_remaining_rad is None
                    else round(
                        math.degrees(precision_heading_remaining_rad),
                        2,
                    )
                ),
                "capture_heading_control": (
                    "relative_scan_turn_accumulator"
                    if precision_heading_initial_error_rad is not None
                    else "map_pose_heading"
                ),
                "target_x_m": round(target.target_xy_m[0], 4),
                "target_y_m": round(target.target_xy_m[1], 4),
                "inspection_subject": (
                    "wall_surface"
                    if target.kind == "perimeter_wall"
                    else "unknown_mapped_black_region"
                ),
                "review_question": (
                    "检查墙面材质、门窗/突出物、损伤或标志"
                    if target.kind == "perimeter_wall"
                    else "确认该地图黑色区域对应桌子、柜子或其他障碍物"
                ),
                "capture_mode": capture_mode,
                "capture_note": capture_note,
                **camera_info,
            }
            report.add(record)
            semantic_map.record_observation(target, record, image_path)
            if args.defer_online_qwen and online_qwen.url:
                deferred_online_annotations.append(
                    (target, record, image_path)
                )
                component_promoted = False
            else:
                component_promoted = annotate_capture_online(
                    target, record, image_path
                )
            for photo_index in range(2, args.photos_per_target + 1):
                time.sleep(min(0.6, max(0.15, args.settle_seconds * 0.25)))
                extra_name = "{:02d}_{}_{}_p{:02d}.jpg".format(
                    capture_index,
                    target.kind,
                    target.target_id,
                    photo_index,
                )
                extra_path = report.directory / extra_name
                extra_record = dict(record)
                extra_record.update(camera.capture(extra_path))
                extra_record["image"] = extra_name
                extra_record["photo_index"] = photo_index
                report.add(extra_record)
                semantic_map.record_observation(
                    target, extra_record, extra_path
                )
                if args.defer_online_qwen and online_qwen.url:
                    deferred_online_annotations.append(
                        (target, extra_record, extra_path)
                    )
                else:
                    component_promoted = (
                        annotate_capture_online(
                            target, extra_record, extra_path
                        )
                        or component_promoted
                    )
            completed_targets += 1
            target_retry_origins.pop(target.target_id, None)
            blocked_path_heading_targets.discard(target.target_id)
            if (
                online_qwen.url
                and not args.defer_online_qwen
                and target.kind == "interior_black_region"
                and not component_promoted
            ):
                visual_attempt = visual_component_attempts.get(
                    target.target_id, 0
                ) + 1
                visual_component_attempts[target.target_id] = visual_attempt
                if visual_attempt < args.online_qwen_max_component_attempts:
                    close_target = original_component_targets[
                        target.target_id
                    ]
                    detail_view_targets.add(target.target_id)
                    capture_reference_views[target.target_id] = (
                        close_target.view_xy_m
                    )
                    alternate_views.pop(target.target_id, None)
                    target_attempts[target.target_id] = 0
                    ordered_targets.append(close_target)
                    print(
                        "黑色区域 {} 的远景尚无连续可信标签；已追加"
                        "原始近景机位补拍（视觉尝试 {}/{}），不会用"
                        "猜测填图。".format(
                            target.target_id,
                            visual_attempt,
                            args.online_qwen_max_component_attempts,
                        ),
                        flush=True,
                    )
                else:
                    print(
                        "黑色区域 {} 已完成 {} 轮视觉观察但证据仍"
                        "不足；保留 unclassified，不扩散错误标签。"
                        .format(target.target_id, visual_attempt),
                        flush=True,
                    )
            # Volatile obstacle evidence is scoped to one immediate motion
            # decision.  Never let it accumulate across photographed targets
            # and silently disconnect later regions of the permanent map.
            known_map.clear_temporary_obstacles()
            print(
                "已拍摄 {}/{}：{} [{}] -> {}".format(
                    completed_targets, len(ordered), target.kind,
                    capture_mode, image_path
                )
            )
        if deferred_online_annotations:
            client.stop()
            speed_controller.stopped()
            print(
                "巡检运动已经结束并停车；开始统一标注 {} 张照片，"
                "标注期间不会再向底盘发送运动命令。".format(
                    len(deferred_online_annotations)
                ),
                flush=True,
            )
            for queued_target, queued_record, queued_path in (
                deferred_online_annotations
            ):
                annotate_capture_online(
                    queued_target, queued_record, queued_path
                )
        interior_component_ids = mission_interior_component_ids
        if online_qwen.url and interior_component_ids:
            confirmed_components = (
                online_confirmed_components | preclassified_components
            )
            unresolved_ids = sorted(
                interior_component_ids - confirmed_components
            )
            print(
                "在线语义覆盖：本轮已确认 {}/{} 个内部黑色连通区域；"
                "未确认={}。".format(
                    len(interior_component_ids) - len(unresolved_ids),
                    len(interior_component_ids),
                    ",".join(unresolved_ids) or "无",
                ),
                flush=True,
            )
        return report.directory
    finally:
        try:
            client.stop()
        except Exception as exc:
            print("最终停车告警：{}".format(exc))
        if owns_lidar:
            try:
                lidar.close()
            except Exception:
                pass
        if owns_client:
            try:
                client.close()
            except Exception:
                pass
        report.save()
        for permanent_path, expected_hash in permanent_map_hashes.items():
            actual_hash = _file_sha256(permanent_path)
            if actual_hash != expected_hash:
                print(
                    "严重告警：永久地图文件在巡检期间发生变化：{}".format(
                        permanent_path
                    )
                )
        print("巡检结果目录：{}".format(report.directory))


def main():
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
