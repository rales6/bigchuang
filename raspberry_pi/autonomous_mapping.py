"""基于 car1.0 已验证 BLE 控制链路的前向 160°自主建图。

程序先检查 BLE 心跳，再启动雷达。车辆先原地慢速扫描一圈，随后使用栅格前沿
探索可达未知区域。完整 360° 扫描只用于定位历史，前向限距扇区才写入地图。
每条运动命令均带短 TTL，不启动后台无限续期。
"""

import argparse
import math
import signal
import time

from raspberry_pi.config import LidarConfig, MappingConfig, SerialConfig
from raspberry_pi.esp32 import Esp32Client, LinkError
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.realtime_map import RealtimeMapServer
from raspberry_pi.mapping import (
    AdaptiveSpeedConfig,
    AdaptiveSpeedController,
    ExplorationConfig,
    FrontierExplorer,
    LidarSlam,
    filter_scan_sector,
)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="car1.0 BLE + 全帧定位/前向 160° N10 自主建图"
    )
    parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="必须显式给出才允许车辆运动",
    )
    parser.add_argument(
        "--link",
        choices=("ble", "uart", "auto"),
        default="ble",
        help="car1.0 默认使用已验证的 BLE 链路",
    )
    parser.add_argument("--ble-name", default="ESP32-Robot-Car")
    parser.add_argument("--ble-address", default=None)
    parser.add_argument(
        "--ble-connect-timeout",
        type=float,
        default=12.0,
        help="BLE scan and connection timeout in seconds",
    )
    parser.add_argument(
        "--ble-operation-timeout",
        type=float,
        default=1.5,
        help="maximum time for one BLE GATT write in seconds",
    )
    parser.add_argument(
        "--link-request-timeout",
        type=float,
        default=0.80,
        help="response timeout for one protocol request in seconds",
    )
    parser.add_argument(
        "--link-request-retries",
        type=int,
        default=3,
        help="number of protocol retries after a timeout",
    )
    parser.add_argument("--esp-port", default="/dev/serial0")
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--lidar-baud", type=int, default=230400)
    parser.add_argument("--counterclockwise", action="store_true")
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--no-motor-control", action="store_true")

    parser.add_argument(
        "--front-center",
        type=float,
        default=0.0,
        help="有效扇区中心相对车头角度，向左为正",
    )
    parser.add_argument(
        "--usable-fov",
        type=float,
        default=160.0,
        help="写地图和避障使用的前向视场角；定位始终使用完整扫描",
    )
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--map-size", type=float, default=20.0)
    parser.add_argument(
        "--mapping-max-distance",
        type=float,
        default=3.0,
        help="写入黑色障碍边界的最远距离；更远回波只清空到此距离，单位 m",
    )
    parser.add_argument(
        "--wall-gap-max",
        type=float,
        default=0.15,
        help="导出地图中允许拼接的黑色墙线最大短缺口，单位 m",
    )
    parser.add_argument(
        "--map-min-speed",
        type=float,
        default=0.025,
        help="低于该实际平移速度时只定位、不写地图，单位 m/s",
    )
    parser.add_argument(
        "--map-max-speed",
        type=float,
        default=0.30,
        help="高于该实际平移速度时只定位、不写地图，单位 m/s",
    )
    parser.add_argument(
        "--map-max-turn-rate",
        type=float,
        default=0.65,
        help="高于该实际角速度时只定位、不写地图，单位 rad/s",
    )
    parser.add_argument(
        "--turn-max-translation",
        type=float,
        default=0.08,
        help="雷达判定正在转向时允许的最大单帧平移，单位 m",
    )
    parser.add_argument(
        "--no-manhattan",
        action="store_true",
        help="关闭矩形房间横平竖直偏航校正",
    )
    parser.add_argument(
        "--min-obstacle-area",
        type=float,
        default=0.03,
        help="小于该连通面积的障碍点块不进入规划和地图，单位 m^2",
    )
    parser.add_argument("--output", default="maps/ble_auto_map")
    parser.add_argument("--save-every", type=int, default=40)
    parser.add_argument(
        "--live-map",
        action="store_true",
        help="启动树莓派实时 LidarSlam 地图监控网页",
    )
    parser.add_argument(
        "--live-map-bind",
        default="0.0.0.0",
        help="实时地图监听地址；0.0.0.0 允许局域网设备访问",
    )
    parser.add_argument(
        "--live-map-port",
        type=int,
        default=8766,
        help="实时地图 HTTP 端口",
    )
    parser.add_argument(
        "--live-map-refresh-hz",
        type=float,
        default=2.0,
        help="实时地图 PNG 刷新频率，建议 1..3 Hz",
    )

    parser.add_argument("--speed", type=int, default=250, help="巡航速度 mm/s")
    parser.add_argument(
        "--max-drive-speed",
        type=int,
        default=460,
        help="检测到直行堵转或执行脱困时允许提升到的速度上限 mm/s",
    )
    parser.add_argument(
        "--front-left-gain",
        type=float,
        default=1.15,
        help="树莓派下发的左前轮输出倍率",
    )
    parser.add_argument(
        "--front-right-gain",
        type=float,
        default=1.00,
        help="树莓派下发的右前轮输出倍率",
    )
    parser.add_argument(
        "--rear-left-gain",
        type=float,
        default=0.90,
        help="树莓派下发的左后轮输出倍率；空转时应适当降低",
    )
    parser.add_argument(
        "--rear-right-gain",
        type=float,
        default=1.00,
        help="树莓派下发的右后轮输出倍率",
    )
    parser.add_argument("--slow-speed", type=int, default=220)
    parser.add_argument(
        "--measured-speed-min",
        type=float,
        default=0.075,
        help="雷达定位测得速度低于此值时逐步增大驱动，单位 m/s",
    )
    parser.add_argument(
        "--measured-speed-max",
        type=float,
        default=0.22,
        help="雷达定位测得速度高于此值时逐步减小驱动，单位 m/s",
    )
    parser.add_argument(
        "--measured-turn-min",
        type=float,
        default=0.12,
        help="雷达定位测得角速度低于此值时逐步增大转向驱动，单位 rad/s",
    )
    parser.add_argument(
        "--measured-turn-max",
        type=float,
        default=0.85,
        help="雷达定位测得角速度高于此值时逐步减小转向驱动，单位 rad/s",
    )
    parser.add_argument("--turn-speed", type=int, default=2500, help="转速 mrad/s")
    parser.add_argument(
        "--turn-arc-speed",
        type=int,
        default=260,
        help="用小半径弧线代替长时间原地旋转时的前进速度",
    )
    parser.add_argument("--stop-distance", type=float, default=0.40)
    parser.add_argument("--slow-distance", type=float, default=0.75)
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.25,
        help="障碍膨胀半径，应不小于车体中心到最外缘的距离",
    )
    parser.add_argument("--robot-length", type=float, default=0.35)
    parser.add_argument("--robot-width", type=float, default=0.24)
    parser.add_argument("--lidar-offset-x", type=float, default=0.14)
    parser.add_argument("--lidar-offset-y", type=float, default=0.0)
    parser.add_argument("--safety-margin", type=float, default=0.04)
    parser.add_argument("--max-runtime-min", type=float, default=15.0)
    parser.add_argument("--max-rejected-scans", type=int, default=3)
    parser.add_argument(
        "--command-ttl-ms",
        type=int,
        default=450,
        help="雷达停止更新后 ESP32 自动停车的最长命令有效期",
    )
    parser.add_argument(
        "--command-min-interval-ms",
        type=int,
        default=300,
        help="相同方向运动命令的最短发送间隔，降低BLE拥塞",
    )
    parser.add_argument(
        "--link-reconnect-attempts",
        type=int,
        default=3,
        help="运动命令超时后的链路重连次数",
    )
    parser.add_argument(
        "--link-reconnect-delay",
        type=float,
        default=1.0,
        help="每次重连前等待秒数",
    )
    parser.add_argument(
        "--reconnect-discard-scans",
        type=int,
        default=2,
        help="重连后丢弃的积压雷达帧数",
    )
    parser.add_argument(
        "--reconnect-stabilize-scans",
        type=int,
        default=1,
        help="重连并停车后用于稳定和全局重定位的雷达帧数",
    )
    parser.add_argument(
        "--reconnect-map-hold-scans",
        type=int,
        default=8,
        help="重定位后只定位、不写地图的有效雷达帧数",
    )
    return parser


class ActiveRuntimeBudget:
    """Track active exploration time while excluding reconnect pauses."""

    def __init__(self, maximum_s, clock=None):
        self.maximum_s = max(0.0, float(maximum_s))
        self._clock = clock or time.monotonic
        self._started_s = self._clock()
        self._paused_total_s = 0.0
        self._pause_started_s = None

    def pause(self):
        if self._pause_started_s is None:
            self._pause_started_s = self._clock()

    def resume(self):
        if self._pause_started_s is None:
            return
        self._paused_total_s += self._clock() - self._pause_started_s
        self._pause_started_s = None

    @property
    def elapsed_s(self):
        now_s = self._clock()
        active_pause_s = (
            0.0
            if self._pause_started_s is None
            else now_s - self._pause_started_s
        )
        return max(
            0.0,
            now_s
            - self._started_s
            - self._paused_total_s
            - active_pause_s,
        )

    @property
    def paused_s(self):
        now_s = self._clock()
        active_pause_s = (
            0.0
            if self._pause_started_s is None
            else now_s - self._pause_started_s
        )
        return self._paused_total_s + active_pause_s

    def expired(self):
        return self.elapsed_s >= self.maximum_s


class MotionCommandScheduler:
    """限制BLE命令频率，同时让停车和运动方向变化立即生效。"""

    def __init__(self, minimum_interval_ms=300):
        self.minimum_interval_s = max(
            0.0, float(minimum_interval_ms) / 1000.0
        )
        self.last_command = None
        self.last_sent_s = None

    def should_send(self, command, ttl_ms, now_s=None):
        now_s = time.monotonic() if now_s is None else float(now_s)
        current = (
            int(command.linear_mm_s),
            int(command.angular_mrad_s),
            int(ttl_ms),
        )
        if self.last_command is None:
            return True
        previous = self.last_command
        if current[0] == 0 and current[1] == 0:
            return current != previous
        if _motion_direction(current[0]) != _motion_direction(previous[0]):
            return True
        if _motion_direction(current[1]) != _motion_direction(previous[1]):
            return True
        return (
            self.last_sent_s is None
            or now_s - self.last_sent_s >= self.minimum_interval_s
        )

    def mark_sent(self, command, ttl_ms, now_s=None):
        self.last_command = (
            int(command.linear_mm_s),
            int(command.angular_mrad_s),
            int(ttl_ms),
        )
        self.last_sent_s = (
            time.monotonic() if now_s is None else float(now_s)
        )

    def reset(self):
        self.last_command = None
        self.last_sent_s = None


def _motion_direction(value):
    return 0 if value == 0 else (1 if value > 0 else -1)


def reconnect_client(factory, old_client, attempts, delay_s):
    """关闭失效客户端并建立新链路；新连接先心跳再硬停车。"""
    try:
        old_client.close(send_cancel=False)
    except TypeError:
        old_client.close()
    except Exception:
        pass
    last_error = None
    for attempt in range(1, int(attempts) + 1):
        if delay_s > 0:
            time.sleep(float(delay_s))
        candidate = None
        try:
            candidate = factory()
            candidate.heartbeat()
            candidate.stop()
            status_note = ""
            query_status = getattr(candidate, "query_status", None)
            if query_status is not None:
                try:
                    status = query_status()
                    status_note = "，ESP32 uptime={:.1f}s".format(
                        status.get("uptime_ms", 0) / 1000.0
                    )
                except Exception:
                    status_note = ""
            print(
                "链路重连成功 {}/{}：当前链路={}{}".format(
                    attempt,
                    attempts,
                    candidate.active_transport,
                    status_note,
                )
            )
            return candidate
        except Exception as exc:
            last_error = exc
            print("链路重连失败 {}/{}：{}".format(
                attempt, attempts, exc
            ))
            if candidate is not None:
                try:
                    candidate.close(send_cancel=False)
                except TypeError:
                    candidate.close()
                except Exception:
                    pass
    raise RuntimeError(
        "链路连续重连失败 {} 次：{}".format(attempts, last_error)
    )


def validate_args(args, parser=None):
    errors = []
    if not args.enable_motion:
        errors.append("必须添加 --enable-motion 才允许自主行驶")
    if not 30.0 <= args.usable_fov <= 180.0:
        errors.append("--usable-fov 必须在 30..180 度之间")
    if not 50 <= args.speed <= 500:
        errors.append("--speed 必须在 50..500 mm/s 之间")
    if not args.speed <= args.max_drive_speed <= 500:
        errors.append(
            "--max-drive-speed 必须在 --speed..500 mm/s 之间"
        )
    wheel_gains = (
        args.front_left_gain,
        args.front_right_gain,
        args.rear_left_gain,
        args.rear_right_gain,
    )
    if any(value < 0.50 or value > 1.50 for value in wheel_gains):
        errors.append("四个轮输出倍率必须都在 0.50..1.50 之间")
    if not 30 <= args.slow_speed <= args.speed:
        errors.append("--slow-speed 必须在 30..speed mm/s 之间")
    if not 0.01 <= args.measured_speed_min < args.measured_speed_max <= 0.50:
        errors.append(
            "必须满足 0.01 <= --measured-speed-min "
            "< --measured-speed-max <= 0.50 m/s"
        )
    if not 0.02 <= args.measured_turn_min < args.measured_turn_max <= 3.0:
        errors.append(
            "必须满足 0.02 <= --measured-turn-min "
            "< --measured-turn-max <= 3.0 rad/s"
        )
    if not 300 <= args.turn_speed <= 3500:
        errors.append("--turn-speed 必须在 300..3500 mrad/s 之间")
    if not 80 <= args.turn_arc_speed <= 500:
        errors.append("--turn-arc-speed 必须在 80..500 mm/s 之间")
    if not 0.20 <= args.stop_distance < args.slow_distance:
        errors.append("必须满足 0.20 <= --stop-distance < --slow-distance")
    if not 0.15 <= args.clearance <= 1.0:
        errors.append("--clearance 必须在 0.15..1.0 米之间")
    if not 0.20 <= args.robot_length <= 1.0:
        errors.append("--robot-length must be in 0.20..1.0 m")
    if not 0.15 <= args.robot_width <= 1.0:
        errors.append("--robot-width must be in 0.15..1.0 m")
    if not -0.5 <= args.lidar_offset_x <= 0.5:
        errors.append("--lidar-offset-x must be in -0.5..0.5 m")
    if not -0.5 <= args.lidar_offset_y <= 0.5:
        errors.append("--lidar-offset-y must be in -0.5..0.5 m")
    if not 0.0 <= args.safety_margin <= 0.30:
        errors.append("--safety-margin must be in 0..0.30 m")
    if not 200 <= args.command_ttl_ms <= 700:
        errors.append("--command-ttl-ms 必须在 200..700 ms 之间")
    if not 100 <= args.command_min_interval_ms <= 500:
        errors.append("--command-min-interval-ms 必须在 100..500 ms 之间")
    if not 1 <= args.link_reconnect_attempts <= 10:
        errors.append("--link-reconnect-attempts 必须在 1..10 之间")
    if not 0.1 <= args.link_reconnect_delay <= 10.0:
        errors.append("--link-reconnect-delay 必须在 0.1..10 秒之间")
    if not 3.0 <= args.ble_connect_timeout <= 30.0:
        errors.append("--ble-connect-timeout 必须在 3..30 秒之间")
    if not 0.3 <= args.ble_operation_timeout <= 3.0:
        errors.append("--ble-operation-timeout 必须在 0.3..3 秒之间")
    if not 0.2 <= args.link_request_timeout <= 3.0:
        errors.append("--link-request-timeout 必须在 0.2..3 秒之间")
    if not 0 <= args.link_request_retries <= 8:
        errors.append("--link-request-retries 必须在 0..8 之间")
    if not 0 <= args.reconnect_discard_scans <= 10:
        errors.append("--reconnect-discard-scans 必须在 0..10 之间")
    if not 0 <= args.reconnect_stabilize_scans <= 20:
        errors.append("--reconnect-stabilize-scans 必须在 0..20 之间")
    if not 0 <= args.reconnect_map_hold_scans <= 50:
        errors.append("--reconnect-map-hold-scans 必须在 0..50 之间")
    if not 0.02 <= args.resolution <= 0.20:
        errors.append("--resolution 必须在 0.02..0.20 米之间")
    if not 3.0 <= args.map_size <= 100.0:
        errors.append("--map-size 必须在 3..100 米之间")
    if not 0.30 <= args.mapping_max_distance <= 8.0:
        errors.append("--mapping-max-distance 必须在 0.30..8.0 米之间")
    if not 0.0 <= args.wall_gap_max <= 0.50:
        errors.append("--wall-gap-max 必须在 0..0.50 米之间")
    if not 0.0 <= args.map_min_speed < args.map_max_speed:
        errors.append("必须满足 0 <= --map-min-speed < --map-max-speed")
    if not 0.05 <= args.map_max_speed <= 1.0:
        errors.append("--map-max-speed 必须在 0.05..1.0 m/s 之间")
    if not 0.1 <= args.map_max_turn_rate <= 5.0:
        errors.append("--map-max-turn-rate 必须在 0.1..5.0 rad/s 之间")
    if not 0.03 <= args.turn_max_translation <= 0.12:
        errors.append("--turn-max-translation 必须在 0.03..0.12 m 之间")
    if not 0.0 <= args.min_obstacle_area <= 0.50:
        errors.append("--min-obstacle-area 必须在 0..0.50 m^2 之间")
    if args.max_runtime_min <= 0:
        errors.append("--max-runtime-min 必须大于 0")
    if not 1 <= args.live_map_port <= 65535:
        errors.append("--live-map-port 必须在 1..65535 之间")
    if not 0.2 <= args.live_map_refresh_hz <= 10.0:
        errors.append("--live-map-refresh-hz 必须在 0.2..10.0 Hz 之间")
    if errors:
        message = "；".join(errors)
        if parser is not None:
            parser.error(message)
        raise ValueError(message)


def run(
    args,
    client=None,
    lidar=None,
    client_factory=None,
    live_map_server=None,
):
    """运行自主建图；可注入 client/lidar 以进行无硬件集成测试。"""
    validate_args(args)
    cells = int(round(args.map_size / args.resolution))
    lidar_config = LidarConfig(
        port=args.lidar_port,
        baudrate=args.lidar_baud,
        angle_offset_deg=args.angle_offset,
        clockwise=not args.counterclockwise,
        motor_control=not args.no_motor_control,
    )
    slam = LidarSlam(
        MappingConfig(
            resolution_m=args.resolution,
            width_cells=cells,
            height_cells=cells,
            lidar_offset_x_m=args.lidar_offset_x,
            lidar_offset_y_m=args.lidar_offset_y,
            mapping_max_distance_m=args.mapping_max_distance,
            render_wall_gap_max_m=args.wall_gap_max,
            map_min_linear_speed_m_s=args.map_min_speed,
            map_max_linear_speed_m_s=args.map_max_speed,
            map_max_angular_speed_rad_s=args.map_max_turn_rate,
            lidar_turn_max_translation_m=args.turn_max_translation,
            manhattan_enabled=not args.no_manhattan,
            min_obstacle_area_m2=args.min_obstacle_area,
        ),
        lidar_config,
    )
    explorer = FrontierExplorer(ExplorationConfig(
        cruise_speed_mm_s=args.speed,
        slow_speed_mm_s=args.slow_speed,
        turn_speed_mrad_s=args.turn_speed,
        turn_arc_speed_mm_s=args.turn_arc_speed,
        stop_distance_m=args.stop_distance,
        slow_distance_m=args.slow_distance,
        robot_clearance_m=args.clearance,
        robot_length_m=args.robot_length,
        robot_width_m=args.robot_width,
        lidar_offset_x_m=args.lidar_offset_x,
        lidar_offset_y_m=args.lidar_offset_y,
        safety_margin_m=args.safety_margin,
    ))
    speed_controller = AdaptiveSpeedController(
        args.turn_speed,
        maximum_linear_speed_mm_s=args.max_drive_speed,
        config=AdaptiveSpeedConfig(
            target_linear_speed_min_m_s=args.measured_speed_min,
            target_linear_speed_max_m_s=args.measured_speed_max,
            target_angular_speed_min_rad_s=args.measured_turn_min,
            target_angular_speed_max_rad_s=args.measured_turn_max,
        ),
    )
    serial_config = SerialConfig(
        port=args.esp_port,
        link_mode=args.link,
        ble_device_name=args.ble_name,
        ble_address=args.ble_address,
        request_timeout_s=args.link_request_timeout,
        retries=args.link_request_retries,
        ble_connect_timeout_s=args.ble_connect_timeout,
        ble_operation_timeout_s=args.ble_operation_timeout,
        wheel_output_gains=(
            args.front_left_gain,
            args.front_right_gain,
            args.rear_left_gain,
            args.rear_right_gain,
        ),
    )
    injected_client = client is not None
    client_factory = client_factory or (
        None if injected_client else lambda: Esp32Client(serial_config)
    )
    client = client or client_factory()
    command_scheduler = MotionCommandScheduler(
        args.command_min_interval_ms
    )

    stopping = False
    motion_started = False
    accepted_count = 0
    rejected_count = 0
    total_scan_count = 0
    total_rejected_count = 0
    relocalization_count = 0
    paths = None
    runtime_budget = None
    last_update = None
    last_command = None

    def publish_live_map(
        update,
        command=None,
        *,
        state=None,
        reason=None,
        running=True,
        force=False,
    ):
        nonlocal live_map_server
        if live_map_server is None or update is None:
            return
        command_status = {
            "state": (
                state
                or getattr(command, "state", None)
                or "localizing"
            ),
            "reason": (
                reason
                or getattr(command, "reason", None)
                or ""
            ),
            "linear_mm_s": int(
                getattr(command, "linear_mm_s", 0)
            ),
            "angular_mrad_s": int(
                getattr(command, "angular_mrad_s", 0)
            ),
            "target_xy_m": getattr(command, "target_xy_m", None),
        }
        status = {
            "running": bool(running),
            "scan_count": total_scan_count,
            "accepted_count": accepted_count,
            "rejected_count": total_rejected_count,
            "accepted_ratio": (
                accepted_count / max(1, total_scan_count)
            ),
            "accepted": bool(update.accepted),
            "pose": {
                "x_m": float(update.pose.x_m),
                "y_m": float(update.pose.y_m),
                "yaw_deg": math.degrees(update.pose.yaw_rad),
            },
            "rmse_m": (
                float(update.rmse_m)
                if math.isfinite(update.rmse_m)
                else None
            ),
            "correspondences": int(update.correspondences),
            "scan_points": int(update.scan_points),
            "inlier_ratio": float(update.inlier_ratio),
            "rejection_reason": update.rejection_reason,
            "map_status": update.map_status,
            "linear_speed_m_s": float(update.linear_speed_m_s),
            "angular_speed_rad_s": float(
                update.angular_speed_rad_s
            ),
            "command": command_status,
            "mapping": {
                "integrated_count": int(slam.map_integrated_count),
                "skip_counts": dict(slam.map_skip_counts),
                "global_relocalizations": int(
                    slam.global_relocalization_count
                ),
                "manhattan_corrections": int(
                    slam.manhattan_correction_count
                ),
            },
            "elapsed_s": (
                float(runtime_budget.elapsed_s)
                if runtime_budget is not None
                else 0.0
            ),
        }
        try:
            live_map_server.publish(slam, status, force=force)
        except Exception as exc:
            print("\n实时地图更新失败，车辆控制继续运行：{}".format(exc))
            try:
                live_map_server.stop()
            finally:
                live_map_server = None

    def request_stop(_signal_number, _frame):
        nonlocal stopping
        stopping = True

    if signal.getsignal(signal.SIGINT) is not None:
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    try:
        # 蓝牙和 ESP32 必须先通过无动作心跳，失败时不启动雷达和电机。
        try:
            client.heartbeat()
        except LinkError as exc:
            if client_factory is None:
                raise
            print(
                "首次 ESP32 心跳超时，正在重新建立链路：{}".format(
                    exc
                )
            )
            client = reconnect_client(
                client_factory,
                client,
                args.link_reconnect_attempts,
                args.link_reconnect_delay,
            )
        print("ESP32 通信通过：当前链路={}".format(client.active_transport))
        if args.live_map:
            if live_map_server is None:
                live_map_server = RealtimeMapServer(
                    bind=args.live_map_bind,
                    port=args.live_map_port,
                    refresh_hz=args.live_map_refresh_hz,
                )
            monitor_url = live_map_server.start()
            print(
                "实车算法地图监控已启动：{}"
                "（显示 LidarSlam 估计结果，不使用仿真真值）".format(
                    monitor_url
                )
            )
        lidar = lidar or N10LidarDriver(lidar_config)
        max_runtime_s = args.max_runtime_min * 60.0
        runtime_budget = ActiveRuntimeBudget(max_runtime_s)
        print(
            "自主建图已启动：定位使用完整扫描，地图只使用前方 {:.0f}°/"
            "{:.1f}m，巡航/脱困上限 {}/{} mm/s；"
            "Ctrl+C 会停车并保存。".format(
                args.usable_fov,
                args.mapping_max_distance,
                args.speed,
                args.max_drive_speed,
            )
        )

        scan_iterator = iter(lidar.scans())
        for raw_scan in scan_iterator:
            if stopping or runtime_budget.expired():
                break
            total_scan_count += 1
            mapping_scan = filter_scan_sector(
                raw_scan, args.front_center, args.usable_fov
            )
            update = slam.process(raw_scan, mapping_scan=mapping_scan)
            last_update = update
            speed_controller.observe(update)
            if not update.accepted:
                rejected_count += 1
                total_rejected_count += 1
                # 不等待连续失败：运动中的第一帧拒绝就立即停车。否则旧命令
                # 会在 TTL 内继续驱动车辆，使当前帧与参考帧越来越难匹配。
                if motion_started:
                    try:
                        client.stop()
                    except Exception as exc:
                        print("\n定位拒绝后的停车告警：{}".format(exc))
                    motion_started = False
                    speed_controller.stopped()
                    command_scheduler.reset()
                print(
                    "\n定位拒绝 {}/{}：{} rmse={:.3f}m "
                    "inliers={:.0f}% delta=({:.3f}m,{:.1f}deg)".format(
                        rejected_count,
                        args.max_rejected_scans,
                        update.rejection_reason or "unknown",
                        update.rmse_m,
                        update.inlier_ratio * 100.0,
                        update.translation_m,
                        math.degrees(update.rotation_rad),
                    )
                )
                publish_live_map(
                    update,
                    state="localization_rejected",
                    reason=update.rejection_reason or "定位匹配被拒绝",
                )
                if (
                    rejected_count >= 3
                    and relocalization_count < 3
                    and (
                        slam.relocalize(raw_scan)
                        or slam.reseed(raw_scan)
                    )
                ):
                    relocalization_count += 1
                    rejected_count = 0
                    slam.suspend_mapping(
                        args.reconnect_map_hold_scans
                    )
                    print(
                        "车辆已停车并恢复定位参考 ({}/3)；"
                        "随后 {} 帧只定位、不写地图。".format(
                            relocalization_count,
                            args.reconnect_map_hold_scans,
                        )
                    )
                    continue
                if rejected_count >= args.max_rejected_scans:
                    if motion_started:
                        client.stop()
                    raise RuntimeError(
                        "连续 {} 帧定位失败，已停车；请降低速度或改善雷达视野".format(
                            rejected_count
                        )
                    )
                continue

            rejected_count = 0
            relocalization_count = 0
            accepted_count += 1
            command = explorer.update(update.pose, mapping_scan, slam.grid)
            command = speed_controller.apply(command)
            last_command = command
            command_ttl_ms = speed_controller.ttl_for(
                command,
                args.command_ttl_ms,
            )
            command_sent = False
            if command_scheduler.should_send(command, command_ttl_ms):
                try:
                    client.set_twist(
                        command.linear_mm_s,
                        command.angular_mrad_s,
                        ttl_ms=command_ttl_ms,
                    )
                    command_scheduler.mark_sent(
                        command, command_ttl_ms
                    )
                    command_sent = True
                except LinkError as exc:
                    print(
                        "\n运动命令通信超时，旧命令将在TTL内失效：{}".format(
                            exc
                        )
                    )
                    publish_live_map(
                        update,
                        state="link_reconnecting",
                        reason=str(exc),
                    )
                    motion_started = False
                    speed_controller.stopped()
                    command_scheduler.reset()
                    if client_factory is None:
                        raise RuntimeError(
                            "注入的测试客户端没有提供重连工厂"
                        ) from exc
                    runtime_budget.pause()
                    try:
                        client = reconnect_client(
                            client_factory,
                            client,
                            args.link_reconnect_attempts,
                            args.link_reconnect_delay,
                        )
                        fresh_raw_scan = raw_scan
                        stable_scan_count = (
                            args.reconnect_discard_scans
                            + args.reconnect_stabilize_scans
                        )
                        for _index in range(stable_scan_count):
                            fresh_raw_scan = next(scan_iterator)
                        globally_relocalized = slam.relocalize(
                            fresh_raw_scan
                        )
                        if not globally_relocalized:
                            if not slam.reseed(fresh_raw_scan):
                                raise RuntimeError(
                                    "链路重连成功，但稳定扫描不足，"
                                    "无法恢复定位参考"
                                )
                            print(
                                "全局重定位证据不足，保持原位姿并重建局部参考；"
                                "恢复期禁止写地图。"
                            )
                        slam.suspend_mapping(
                            args.reconnect_map_hold_scans
                        )
                    finally:
                        runtime_budget.resume()
                    rejected_count = 0
                    relocalization_count = 0
                    print(
                        "重连停车稳定完成，{}；随后 {} 帧只定位、不写地图。"
                        .format(
                            (
                                "全局重定位成功"
                                if globally_relocalized
                                else "局部参考恢复"
                            ),
                            args.reconnect_map_hold_scans,
                        )
                    )
                    continue
            if command_sent:
                motion_started = (
                    command.linear_mm_s != 0
                    or command.angular_mrad_s != 0
                )
            target = "-"
            if command.target_xy_m is not None:
                target = "({:+.2f},{:+.2f})".format(*command.target_xy_m)
            safety_note = "-"
            if command.state in (
                "rotation_blocked",
                "turn_limit_blocked",
                "cautious_turn_probe",
                "turn_escape",
                "progress_escape",
                "progress_escape_turn",
            ):
                safety_note = command.reason
            publish_live_map(update, command)
            print(
                "\rscan={} pose=({:+.2f},{:+.2f},{:+.1f}deg) "
                "state={} cmd=({:+d},{:+d}) target={} points={} "
                "rmse={:.3f} inliers={:.0f}% turn={}/{} "
                "linear_adjust={:+d} ttl={} map={} "
                "v={:.2f} w={:.2f} tx={} safety={}    ".format(
                    accepted_count,
                    update.pose.x_m,
                    update.pose.y_m,
                    math.degrees(update.pose.yaw_rad),
                    command.state,
                    command.linear_mm_s,
                    command.angular_mrad_s,
                    target,
                    update.scan_points,
                    update.rmse_m,
                    update.inlier_ratio * 100.0,
                    speed_controller.turn_floor_mrad_s,
                    speed_controller.turn_limit_mrad_s,
                    speed_controller.linear_adjustment_mm_s,
                    command_ttl_ms,
                    update.map_status,
                    update.linear_speed_m_s,
                    update.angular_speed_rad_s,
                    "sent" if command_sent else "held",
                    safety_note,
                ),
                end="",
                flush=True,
            )
            if (
                args.save_every > 0
                and accepted_count % args.save_every == 0
            ):
                slam.save(args.output, rebuild=False)
            if command.finished:
                print("\n没有可达的未知区域，探索完成。")
                break
    finally:
        print("\n正在安全停车……")
        # 无条件发送硬停止；最后一条 (0, 0) 速度命令只会把速度降到零，
        # 旧底板的轮电机 PWM 通道仍可能保持使能并发出嗡鸣。
        try:
            client.stop()
        except Exception as exc:
            print("停车命令告警（短 TTL 仍会使旧命令失效）：{}".format(exc))
        if lidar is not None:
            lidar.close()
        client.close()
        if accepted_count:
            paths = slam.save(args.output)
            print("已保存：{}".format(", ".join(str(path) for path in paths)))
            print("建图门控统计：{}".format(slam.mapping_summary()))
            if runtime_budget is not None:
                print(
                    "运行计时：有效探索 {:.1f}s，重连暂停 {:.1f}s。"
                    .format(
                        runtime_budget.elapsed_s,
                        runtime_budget.paused_s,
                    )
                )
        else:
            print("没有有效扫描，未生成空白地图。")
        if last_update is not None:
            publish_live_map(
                last_update,
                last_command,
                state="finished",
                reason="建图程序已停止，最终地图已保存",
                running=False,
                force=True,
            )
        if live_map_server is not None:
            live_map_server.stop()
            print("实时地图监控已关闭；最终 PNG、PGM 和轨迹文件仍保留。")
    return paths


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    run(args)


if __name__ == "__main__":
    main()
