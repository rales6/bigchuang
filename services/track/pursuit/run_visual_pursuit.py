# -*- coding: utf-8 -*-
"""视觉追踪逼近任务主程序（树莓派端）。

流程：
1. 自动探测双 USB 摄像头并开始双路采集；
2. 启动 N10 雷达（可选，缺失时退化为无雷达模式）；
3. 连接 3090 pursuit 服务：首帧拍照提交 Qwen 识别（HTTP /ground），
   之后以 ~10fps 上传左右拼接帧（WebSocket），逐帧接收
   {track, bbox, depth}；
4. PursuitController 状态机输出运动命令，经 Esp32Client(BLE)
   下发（--dry-run 只打印不下发，实车必须先询问确认）。

用法（track 目录下，使用 dachuang/.venv）：
  python pursuit/run_visual_pursuit.py --instruction "追踪那个蓝色瓶子" --dry-run
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import queue
import signal
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import faulthandler

# 排障用：kill -USR1 <pid> 打印主循环当前栈（断连卡死现场取证）
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except (AttributeError, ValueError, OSError):
    pass  # Windows 无 SIGUSR1

import cv2
import numpy as np

TRACK_ROOT = Path(__file__).resolve().parents[1]
if str(TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACK_ROOT))

from raspberry_pi.config import LidarConfig, SerialConfig
from raspberry_pi.esp32.client import Esp32Client
from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.lidar.n10_driver import N10LidarDriver

from pursuit.pursuit_config import CameraConfig, PursuitConfig, StereoConfig
from pursuit.camera_stream import StereoCameraStream
from pursuit.camera_contract import model_devices as model_camera_devices
from pursuit.pursuit_controller import Perception, PursuitController
from pursuit.run_logging import RunLogBundle


class LatestScanHolder:
    """雷达线程与主循环之间传递最新一圈扫描。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scan: LaserScan | None = None
        self.error = ""

    def set(self, scan: LaserScan) -> None:
        with self._lock:
            self.scan = scan

    def get(self) -> LaserScan | None:
        with self._lock:
            return self.scan

    def set_error(self, message: str) -> None:
        with self._lock:
            self.error = message


def _lidar_thread(driver: N10LidarDriver, holder: LatestScanHolder,
                  stop_event: threading.Event) -> None:
    try:
        for scan in driver.scans():
            if stop_event.is_set():
                return
            holder.set(scan)
    except Exception as exc:
        holder.set_error(f"{type(exc).__name__}: {exc}")


class DryRunClient:
    """不驱动实车：打印将要下发的命令。"""

    def __init__(self) -> None:
        self.last_status = None

    def start(self) -> None:
        print("[dry-run] motion link disabled, commands are printed only")

    def set_twist(self, linear_mm_s, angular_mrad_s, ttl_ms=600):
        print(f"[dry-run] SET_TWIST v={linear_mm_s}mm/s "
              f"w={angular_mrad_s}mrad/s ttl={ttl_ms}ms")

    def refresh_twist(self, linear_mm_s, angular_mrad_s, ttl_ms=600):
        pass  # 与 set_twist 相同命令的保活，dry-run 不重复打印

    def stop(self):
        print("[dry-run] STOP")

    def close(self):
        print("[dry-run] link closed")


def geometry_for_camera_order(
        pursuit_cfg: PursuitConfig, stereo_cfg: StereoConfig,
        camera_cfg: CameraConfig) -> tuple[PursuitConfig, StereoConfig]:
    """Map taught physical-left/right geometry (video2/video0) to views.

    The old fixed-0/2 assumption contradicted the operator port identity and
    the model replay audit. Displaying 2/0 needs NO mask swap; the reverse
    order needs both geometry and principal points swapped.
    """
    from pursuit.camera_contract import geometry_needs_swap
    if not geometry_needs_swap(camera_cfg):
        return pursuit_cfg, stereo_cfg
    return replace(
        pursuit_cfg,
        physical_left_on_left=False,
        forward_left_line=pursuit_cfg.forward_right_line,
        forward_right_line=pursuit_cfg.forward_left_line,
        forward_left_line_1280=pursuit_cfg.forward_right_line_1280,
        forward_right_line_1280=pursuit_cfg.forward_left_line_1280,
        taught_left_cx_px=pursuit_cfg.taught_right_cx_px,
        taught_left_cy_px=pursuit_cfg.taught_right_cy_px,
        taught_right_cx_px=pursuit_cfg.taught_left_cx_px,
        taught_right_cy_px=pursuit_cfg.taught_left_cy_px,
        taught_left_mask_polygon_px=pursuit_cfg.taught_right_mask_polygon_px,
        taught_right_mask_polygon_px=pursuit_cfg.taught_left_mask_polygon_px,
    ), replace(
        stereo_cfg,
        cx_raw_left_px=stereo_cfg.cx_raw_right_px,
        cx_raw_right_px=stereo_cfg.cx_raw_left_px,
    )


class ServerClient:
    """与 3090 pursuit 服务的 HTTP + WebSocket 通信。"""

    def __init__(self, http_base: str, ws_url: str) -> None:
        self.http_base = http_base.rstrip("/")
        self.ws_url = ws_url
        self._ws = None
        self._connect_args: tuple | None = None
        self._last_result_at = 0.0
        # 2026-08-18 断连修复：send/recv 失败立即标记 dead，主循环不等
        # 5s 超时就重连（实测 3090 消失后旧逻辑卡在 send failed 循环，
        # 重连判断从未触发）
        self._dead = False

    def dead(self) -> bool:
        return self._ws is None or self._dead

    def mark_alive(self) -> None:
        self._dead = False

    def connect(self, instruction: str, frame_width: int,
                frame_height: int, upload_fps: float) -> None:
        from websocket import create_connection
        self._connect_args = (instruction, int(frame_width),
                              int(frame_height), float(upload_fps))
        self._ws = create_connection(self.ws_url, timeout=10)
        self._ws.send(json.dumps({
            "type": "pursuit_start",
            "instruction": instruction,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "upload_fps": float(upload_fps),
        }, ensure_ascii=False))
        reply = self._ws.recv()
        # websocket-client 的 timeout 同时作用于 recv 与 send。常态保留
        # 1s 发送超时；仅 recv_result 调用期间临时切到 20ms。旧逻辑
        # 常态 20ms 会把正常的小文本/帧发送误杀，造成周期性离线重连。
        self._ws.settimeout(1.0)
        self.mark_alive()
        print(f"[3090] {reply}")

    def reconnect(self) -> bool:
        """服务器重启/网络抖动后重连（3090 会话状态保留，追踪不丢）。"""
        self.close()
        if self._connect_args is None:
            return False
        try:
            self.connect(*self._connect_args)
            return True
        except Exception as exc:
            print(f"[ws] 重连失败: {exc}")
            return False

    def seconds_since_result(self, now: float) -> float:
        return now - self._last_result_at if self._last_result_at else 999.0

    def ground(self, left_jpeg: bytes, right_jpeg: bytes,
               instruction: str, timeout_s: float = 90.0) -> dict:
        """提交左右两帧给 Qwen 识别，返回 {found, bbox, target_name}。"""
        boundary = uuid.uuid4().hex
        body = bytearray()
        for name, filename, data in (
            ("left", "left.jpg", left_jpeg),
            ("right", "right.jpg", right_jpeg),
        ):
            body += (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{name}"; '
                     f'filename="{filename}"\r\n'
                     f"Content-Type: image/jpeg\r\n\r\n").encode()
            body += data
            body += b"\r\n"
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="instruction"\r\n\r\n'
                 f"{instruction}\r\n"
                 f"--{boundary}--\r\n").encode("utf-8")
        request = urllib.request.Request(
            f"{self.http_base}/ground",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def recv_result(self) -> dict | None:
        """非阻塞取一帧结果或测试网页的识别结果。"""
        if self._ws is None:
            return None
        try:
            self._ws.settimeout(0.02)
            raw = self._ws.recv()
        except Exception as exc:
            # 20ms 无数据超时是正常节拍，不算断连（2026-08-18 实测：
            # 误标记 dead 导致无限重连循环）；真正的连接错误才标记。
            if isinstance(exc, TimeoutError) or "timed out" in str(exc):
                return None
            print(f"[ws] recv dead: {type(exc).__name__}: {exc}")
            self._dead = True
            return None
        finally:
            if self._ws is not None:
                try:
                    self._ws.settimeout(1.0)
                except Exception:
                    pass
        if not raw:
            return None
        if isinstance(raw, (bytes, bytearray)):
            return None
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if message.get("type") not in ("frame_result", "ground_result"):
            return None
        if message.get("type") == "frame_result":
            self._last_result_at = time.monotonic()
        return message

    def send_text(self, obj: dict) -> None:
        """上行小文本消息（状态机状态等），失败标记 dead。"""
        if self._ws is None:
            return
        try:
            self._ws.settimeout(1.0)
            self._ws.send(json.dumps(obj, ensure_ascii=False))
        except Exception as exc:
            print(f"[ws] send_text dead: {type(exc).__name__}: {exc}")
            self._dead = True

    def send_frame(self, jpeg: bytes) -> bool:
        """上传一帧，成功返回 True。

        无背压（每帧都发，服务端 drain 循环丢弃陈旧帧只处理最新，
        队列积压保持在 ~2ms 量级——2026-08-15 的断连风暴已由
        服务端 drain + 本端超时兜底双保险解决）。发送超时 1s：
        TCP 反压（3090 忙于 /ground 等慢任务）不卡主循环 20ms 节拍，
        超时丢帧、下帧照发（latest-wins），不会死锁也不会内存堆积。
        """
        if self._ws is None:
            return False
        try:
            # 发送大帧使用 1s 超时；recv_result 会自行临时切到 20ms。
            self._ws.settimeout(1.0)
            self._ws.send_binary(jpeg)
            return True
        except Exception as exc:
            # 1s 发送超时是 TCP 反压（3090 忙），连接可能仍可用；
            # 真正的连接错误（对端消失/重置）才标记 dead。
            if isinstance(exc, TimeoutError) or "timed out" in str(exc):
                return False
            self._dead = True
            print(f"[ws] send failed: {exc}")
            return False
        finally:
            if self._ws is not None:
                try:
                    self._ws.settimeout(1.0)
                except Exception:
                    pass

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视觉追踪逼近（树莓派端）")
    parser.add_argument("--instruction", required=True,
                        help="目标描述，如：追踪那个蓝色瓶子")
    parser.add_argument("--server-ws", default=None, help="3090 WebSocket URL")
    parser.add_argument("--server-http", default=None, help="3090 HTTP 地址")
    parser.add_argument("--dry-run", action="store_true", default=None,
                        help="只打印运动命令，不下发 ESP32")
    parser.add_argument("--esp32", action="store_true",
                        help="实车模式：通过 BLE 下发 ESP32（默认 dry-run）")
    parser.add_argument("--stop-distance", type=float, default=None)
    parser.add_argument("--swap-cameras", dest="swap_cameras",
                        action="store_true", default=None,
                        help="交换自动探测到的左右摄像头（本车默认开启）")
    parser.add_argument("--no-swap-cameras", dest="swap_cameras",
                        action="store_false",
                        help="不交换左右摄像头（临时换线/排障用）")
    parser.add_argument("--left-camera", type=int, default=-1)
    parser.add_argument("--right-camera", type=int, default=-1)
    parser.add_argument("--upload-fps", type=float, default=None,
                        help="上传帧率上限（0=每帧上传，默认按相机帧率）")
    parser.add_argument("--yaw-compensation-px", type=float, default=None,
                        help="车头正前方在左目画面中的水平像素偏移"
                             "（覆盖 StereoConfig.yaw_compensation_px，"
                             "实车对准验证后固化到配置）")
    parser.add_argument("--log-dir", default="logs/pursuit",
                        help="每次任务的日志归档根目录（默认 logs/pursuit）")
    parser.add_argument("--log-csv", default=None,
                        help="兼容参数：另存遥测 CSV 到指定路径；默认写入本次任务目录")
    parser.add_argument("--no-lidar", action="store_true",
                        help="不启动雷达（无雷达退化模式）")
    parser.add_argument("--lidar-port", default=None,
                        help="雷达串口，默认取 raspberry_pi.config.LidarConfig")
    parser.add_argument("--exit-on-finish", action="store_true",
                        help="任务成功或放弃后退出；供网页任务编排使用。"
                             "未指定时保留原有持续推流行为")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # 实车模式必须显式 --esp32；默认 dry-run 不驱动任何电机。
    # 日志在初始化硬件之前建立，摄像头/BLE/3090 启动失败也能完整留痕。
    dry_run = bool(args.dry_run) or not args.esp32
    run_logs = RunLogBundle.create(
        args.log_dir,
        instruction=args.instruction,
        mode="dry_run" if dry_run else "esp32",
        argv=sys.argv,
        arguments=vars(args),
    )
    run_logs.install_console_tee()
    print(f"[log] 本次任务目录: {run_logs.run_dir.resolve()}")
    print(f"[log] 终端日志: {run_logs.console_path.resolve()}")

    camera_defaults = CameraConfig()
    camera_cfg = CameraConfig(
        left_device=args.left_camera, right_device=args.right_camera,
        swap_left_right=(camera_defaults.swap_left_right
                         if args.swap_cameras is None
                         else bool(args.swap_cameras)),
    )
    overrides: dict = {"instruction": args.instruction}
    if args.stop_distance is not None:
        overrides["stop_distance_m"] = args.stop_distance
    if args.upload_fps is not None:
        overrides["upload_fps"] = args.upload_fps
    pursuit_cfg = PursuitConfig(**overrides)
    if args.yaw_compensation_px is not None:
        stereo_cfg = StereoConfig(yaw_compensation_px=args.yaw_compensation_px)
    else:
        stereo_cfg = StereoConfig()
    pursuit_cfg, stereo_cfg = geometry_for_camera_order(
        pursuit_cfg, stereo_cfg, camera_cfg)

    print(f"[pursuit] instruction={args.instruction!r} dry_run={dry_run} "
          f"stop_distance={pursuit_cfg.stop_distance_m}m")

    # ---- 摄像头（分辨率/帧率自动探测，生效值见 stream.width/height）----
    stream = StereoCameraStream(camera_cfg, jpeg_quality=pursuit_cfg.jpeg_quality)
    stream.start()
    time.sleep(1.0)
    cam_width = stream.width
    cam_height = stream.height
    if cam_width != 1280:
        # 探测档位可能非 1280 宽：瞄准主点按比例缩放（3090 端 ego-motion
        # 焦距同样按 pursuit_start 的 frame_width 缩放）
        scale = cam_width / 1280.0
        stereo_cfg = replace(
            stereo_cfg,
            cx_raw_left_px=stereo_cfg.cx_raw_left_px * scale,
            cx_raw_right_px=stereo_cfg.cx_raw_right_px * scale,
        )
    if cam_width != 1280 or cam_height != 720:
        # 示教掩码来自 1280x720 原图；相机自动降档时同步缩放整个
        # 多边形，不能只缩放质心，否则网页能看到目标但永远停不准。
        sx, sy = cam_width / 1280.0, cam_height / 720.0
        mean_scale = (sx + sy) / 2.0

        def scaled_polygon(points):
            return tuple((x * sx, y * sy) for x, y in points)

        pursuit_cfg = replace(
            pursuit_cfg,
            visual_near_row_px=pursuit_cfg.visual_near_row_px * sy,
            visual_stop_row_px=pursuit_cfg.visual_stop_row_px * sy,
            visual_align_handoff_left_row_px=pursuit_cfg.visual_align_handoff_left_row_px * sy,
            visual_align_handoff_right_row_px=pursuit_cfg.visual_align_handoff_right_row_px * sy,
            taught_left_cx_px=pursuit_cfg.taught_left_cx_px * sx,
            taught_left_cy_px=pursuit_cfg.taught_left_cy_px * sy,
            taught_right_cx_px=pursuit_cfg.taught_right_cx_px * sx,
            taught_right_cy_px=pursuit_cfg.taught_right_cy_px * sy,
            taught_closer_than_centroid_px=(
                pursuit_cfg.taught_closer_than_centroid_px * sy),
            taught_left_mask_polygon_px=scaled_polygon(
                pursuit_cfg.taught_left_mask_polygon_px),
            taught_right_mask_polygon_px=scaled_polygon(
                pursuit_cfg.taught_right_mask_polygon_px),
            taught_mask_margin_px=pursuit_cfg.taught_mask_margin_px
            * mean_scale,
            taught_early_stop_margin_px=(
                pursuit_cfg.taught_early_stop_margin_px * mean_scale),
            taught_early_stop_uncertain_margin_px=(
                pursuit_cfg.taught_early_stop_uncertain_margin_px * mean_scale),
            taught_stop_normal_step_max_px=(
                pursuit_cfg.taught_stop_normal_step_max_px * mean_scale),
            taught_slow_band_px=pursuit_cfg.taught_slow_band_px
            * mean_scale,
            bbox_jump_px=pursuit_cfg.bbox_jump_px * mean_scale,
        )
    run_logs.update(
        status="hardware_ready",
        camera={"half_width": cam_width, "height": cam_height,
                "swap_left_right": camera_cfg.swap_left_right,
                "actual_devices": [stream.left_index, stream.right_index],
                "model_devices": list(model_camera_devices())},
        effective_pursuit_config=asdict(pursuit_cfg),
        effective_stereo_config=asdict(stereo_cfg),
        control_profile="visual_405_body_forward_line_v16_step_reserve",
    )
    print("[control] 1.0x 无额外增益；120~405mm/s 自适应脉冲；"
          "双目确认静止两步后才加时，每档 0.05s，远处最多 0.95s/近处 0.85s；"
          "双目按车身正前方参考线判断左右，近端保留凸包微调；"
          "丢失后单目优先/双丢多帧趋势倒车找回；"
          f"抓取停车预留参考上一步，正常至少 {pursuit_cfg.taught_early_stop_margin_px:.0f}px；"
          f"反馈不足/步幅过大用 {pursuit_cfg.taught_early_stop_uncertain_margin_px:.0f}px 保护")

    # ---- 雷达 ----
    scan_holder = LatestScanHolder()
    lidar = None
    lidar_stop = threading.Event()
    lidar_start_error = ""
    if not args.no_lidar:
        lidar_config = LidarConfig()
        if args.lidar_port:
            lidar_config = LidarConfig(port=args.lidar_port)
        try:
            lidar = N10LidarDriver(lidar_config)
            threading.Thread(
                target=_lidar_thread, args=(lidar, scan_holder, lidar_stop),
                name="n10-lidar", daemon=True,
            ).start()
            time.sleep(1.0)
            if scan_holder.get() is None and not scan_holder.error:
                print("[lidar] 等待首圈数据…（无数据时运行于退化模式）")
        except Exception as exc:
            # 雷达是增强传感器，不应因串口未插入让已启动的双摄线程异常退出。
            # 实车仍保留视觉停车和 ESP32 TTL 急停；日志明确标记降级原因。
            lidar_start_error = f"{type(exc).__name__}: {exc}"
            lidar = None
            print(f"[lidar] 启动失败，自动切换无雷达退化模式: "
                  f"{lidar_start_error}")
    else:
        print("[lidar] 已通过 --no-lidar 禁用，运行于无雷达退化模式")
    run_logs.update(lidar={
        "enabled": lidar is not None,
        "requested": not args.no_lidar,
        "port": args.lidar_port or getattr(LidarConfig(), "port", ""),
        "startup_error": lidar_start_error,
    })

    # ---- 控制器与运动链路 ----
    controller = PursuitController(pursuit_cfg, stereo_cfg, cam_width,
                                   simulate_motion=dry_run,frame_height=cam_height)
    client: DryRunClient | Esp32Client | None = None
    if dry_run:
        client = DryRunClient()
    else:
        # BLE 是构造时同步连接（6s 扫描超时抛异常）；ESP32 未上电时
        # 不阻塞视觉流程，主循环会每 15s 重试接入。
        try:
            client = Esp32Client(SerialConfig(
                link_mode=pursuit_cfg.link_mode,
                ble_device_name=pursuit_cfg.ble_device_name,
                ble_address=pursuit_cfg.ble_address,
                ble_connect_timeout_s=pursuit_cfg.ble_connect_timeout_s,
                wheel_output_gains=(pursuit_cfg.wheel_output_gain,) * 4,
            ))
        except Exception as exc:
            print(f"[link] BLE 初始化失败（ESP32 未上电？）: {exc}，"
                  "视觉部分照常运行，上电后自动接入")
    if client is not None:
        client.start()
        if not dry_run:
            print("[link] BLE 链路已建立")

    # ---- 3090 通信 ----
    ws_url = args.server_ws or pursuit_cfg.server_ws
    http_base = args.server_http or pursuit_cfg.server_http
    run_logs.update(server_ws=ws_url, server_http=http_base)
    server = ServerClient(http_base, ws_url)
    try:
        server.connect(args.instruction, 2 * cam_width,
                       cam_height, pursuit_cfg.upload_fps)
    except Exception as exc:
        print(f"[pursuit] 无法连接 3090 服务 {ws_url}: {exc}")
        if client is not None:
            client.close()
        stream.close()
        run_logs.finalize("startup_failed", error=f"3090 connect: {exc}")
        run_logs.close()
        return 1

    # ---- 首次 Qwen 识别 ----
    time.sleep(0.5)
    pair = stream.stereo_frame()
    if pair is None:
        print("[camera] 等待双路出帧失败")
        stream.close()
        server.close()
        if client is not None:
            client.close()
        run_logs.finalize("startup_failed", error="stereo frame timeout")
        run_logs.close()
        return 1
    half_width = cam_width
    left_jpeg = cv2.imencode(".jpg", pair[:, :half_width],
                             [int(cv2.IMWRITE_JPEG_QUALITY),
                              pursuit_cfg.jpeg_quality])[1].tobytes()
    right_jpeg = cv2.imencode(".jpg", pair[:, half_width:],
                              [int(cv2.IMWRITE_JPEG_QUALITY),
                               pursuit_cfg.jpeg_quality])[1].tobytes()
    print("[ground] 首次提交 Qwen 识别…")
    ground_found = False
    try:
        for attempt in range(2):
            result = server.ground(left_jpeg, right_jpeg, args.instruction,
                                   pursuit_cfg.ground_timeout_s)
            print(f"[ground] {json.dumps(result, ensure_ascii=False)}")
            ground_found = bool(result.get("found"))
            if ground_found:
                break
            if attempt == 0:
                print("[ground] 未找到，原画面重试一次…")
        controller.on_ground_result(ground_found, time.monotonic())
    except Exception as exc:
        print(f"[ground] 首次识别失败: {exc}，将按未找到进入旋转重搜")
        controller.on_ground_result(False, time.monotonic())

    # ---- 主循环 ----
    csv_path = run_logs.choose_telemetry_path(args.log_csv)
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "wall_time", "t_s", "state", "frame_id", "perception_age_ms",
        "perception_fps", "tracking", "status", "bbox_x1", "bbox_y1",
        "bbox_x2", "bbox_y2", "bbox_cx_px", "bbox_cy_px", "tracking2",
        "status2", "bbox2_x1", "bbox2_y1", "bbox2_x2", "bbox2_y2",
        "bbox_cx2_px", "bbox_cy2_px", "src", "depth_m", "depth_conf",
        "lidar_front_m", "v_mm_s", "w_mrad_s", "left_fps", "right_fps",
        "encode_fps", "encode_ms", "upload_fps", "upload_skip_fps",
        "send_ms", "detail",
    ])
    csv_file.flush()
    run_logs.update(status="running", telemetry_csv=str(csv_path.resolve()))
    # 上传帧率上限：0 = 每帧上传（相机帧率，网页显示帧率 = 相机帧率）
    upload_cap = max(0.0, pursuit_cfg.upload_fps)
    next_upload_at = 0.0
    last_upload_seq = -1
    uploaded = 0           # 每秒成功上传帧数
    send_fail = 0          # 每秒发送失败（超时丢帧）数
    send_ms_sum = 0.0      # 每秒发送耗时累计
    last_command = None
    last_sent_at = 0.0
    command_min_interval_s = 0.15
    last_log_at = time.monotonic()
    last_reconnect_at = 0.0
    next_link_attempt = 0.0
    link_fail_streak = 0
    last_state_sent_at = 0.0
    last_stats_at = 0.0
    stats_msg: dict | None = None   # 每秒统计快照（日志与 pi_state 共用）
    finished_notified = False
    perception: Perception | None = None
    last_perception_rx_at = 0.0
    perception_interval_ema = 0.0
    started = time.monotonic()
    # 当前生效的识别指令：测试网页"重新识别"会更新它，后续重搜都用新指令
    active_instruction = args.instruction
    final_result = ""
    final_detail = ""
    exit_code = 0
    interrupted = False
    run_error = ""
    last_lidar_error = ""
    # 搜索阶段 Qwen HTTP 请求放到守护线程；主循环继续上传相机帧和
    # 发送停车命令，避免一次 1~3s 复检把网页/感知帧率压到 3~4fps。
    reground_results: queue.Queue[tuple[int, dict | None, BaseException | None]] = \
        queue.Queue()
    reground_seq = 0
    reground_pending = False
    last_control_event = None

    def start_reground(left_data: bytes, right_data: bytes,
                       instruction: str) -> None:
        nonlocal reground_seq, reground_pending
        reground_seq += 1
        token = reground_seq
        reground_pending = True

        def worker() -> None:
            try:
                answer = server.ground(
                    left_data, right_data, instruction,
                    min(pursuit_cfg.ground_timeout_s,
                        pursuit_cfg.lost_stationary_recheck_timeout_s))
                reground_results.put((token, answer, None))
            except BaseException as exc:
                reground_results.put((token, None, exc))

        threading.Thread(target=worker, name="pursuit-reground",
                         daemon=True).start()

    print("[pursuit] 主循环启动，Ctrl+C 停止")
    try:
        while True:
            now = time.monotonic()

            # 非阻塞接收后台 Qwen 结果。网页手动重新识别会递增 token，
            # 因而较早发出的自动复检结果不会覆盖用户的新选择。
            try:
                while True:
                    token, result, error = reground_results.get_nowait()
                    if token != reground_seq:
                        continue
                    reground_pending = False
                    if error is not None:
                        print(f"[ground] 重新识别失败: {error}，按未找到处理")
                        controller.on_ground_result(False, time.monotonic())
                    else:
                        print(f"[ground] {json.dumps(result, ensure_ascii=False)}")
                        controller.on_ground_result(
                            bool((result or {}).get("found")),
                            time.monotonic())
                    perception = None
            except queue.Empty:
                pass

            # 收 3090 结果
            message = server.recv_result()
            if message:
                if message.get("type") == "ground_result":
                    # 测试网页 /reground 的识别结果，直接驱动状态机；
                    # 采纳网页指令，后续所有重搜都用新指令。
                    found = bool(message.get("found"))
                    if message.get("instruction"):
                        active_instruction = str(message["instruction"])
                    print(f"[web] 网页重新识别: found={found} "
                          f"target={message.get('target_name', '')!r} "
                          f"instruction={active_instruction!r}")
                    # 使尚未返回的自动 Qwen 结果失效，网页指令优先。
                    reground_seq += 1
                    reground_pending = False
                    controller.on_ground_result(found, time.monotonic())
                    perception = None
                else:
                    received_at = time.monotonic()
                    if last_perception_rx_at > 0.0:
                        interval = received_at - last_perception_rx_at
                        if interval >= 0.005:
                            interval = min(interval, 1.0)
                            perception_interval_ema = (
                                interval if perception_interval_ema <= 0.0
                                else 0.8 * perception_interval_ema
                                + 0.2 * interval)
                    last_perception_rx_at = received_at
                    track = message.get("track") or {}
                    bbox = tuple(float(v) for v in (track.get("bbox") or [])) \
                        if len(track.get("bbox") or []) == 4 else None
                    depth = message.get("depth_m")
                    track2 = message.get("track2") or {}
                    bbox2 = tuple(float(v) for v in (track2.get("bbox") or [])) \
                        if len(track2.get("bbox") or []) == 4 else None
                    perception = Perception(
                        tracking=bool(track.get("status") == "tracking"),
                        bbox=bbox,
                        depth_m=float(depth) if depth is not None else None,
                        depth_conf=float(message.get("depth_conf") or 0.0),
                        status=str(track.get("status") or ""),
                        tracking2=bool(track2.get("status") == "tracking"),
                        bbox2=bbox2,
                        status2=str(track2.get("status") or ""),
                        reidentified=bool(message.get("reidentified")),
                        source=str(track.get("source") or ""),
                        frame_id=int(message.get("frame_id") or 0),
                        received_at=received_at,
                        frame_interval_s=perception_interval_ema,
                    )

            # 断连处理：连接死亡（send/recv 失败）立即重连，不等 5s 超时；
            # 结果超时（连接活着但没结果）则感知置空 + 定期重连。
            if server.dead() or server.seconds_since_result(now) > 5.0:
                if server.dead():
                    if now - last_reconnect_at > 5.0:
                        print("[ws] 连接已死，尝试重连 3090 …")
                        if server.reconnect():
                            print("[ws] 重连成功")
                            server.mark_alive()
                        last_reconnect_at = now
                else:
                    if perception is not None:
                        perception = None
                        print("[ws] 超过 5s 未收到结果，感知置空")
                    if now - last_reconnect_at > 10.0:
                        print("[ws] 尝试重连 3090 …")
                        if server.reconnect():
                            print("[ws] 重连成功")
                            server.mark_alive()
                        last_reconnect_at = now

            decision = controller.tick(perception, scan_holder.get(), now)

            event_key = (controller.state.name, bool(decision.motion.linear_mm_s),
                bool(decision.motion.angular_mrad_s), bool(perception and perception.tracking),
                bool(perception and perception.tracking2), decision.reground_now, decision.finished)
            if event_key != last_control_event:
                run_logs.event("control_transition", elapsed_s=round(now-started, 3),
                    state=controller.state.name, tracking=bool(perception and perception.tracking),
                    tracking2=bool(perception and perception.tracking2),
                    frame_id=perception.frame_id if perception else None,
                    perception_age_s=round(now-perception.received_at,3) if perception and perception.received_at else None,
                    v=decision.motion.linear_mm_s, w=decision.motion.angular_mrad_s,
                    reason=decision.motion.reason, reground=decision.reground_now,
                    recheck_count=controller._stationary_recheck_count,
                    bearing_reference=('body_forward_line' if pursuit_cfg.forward_reference_enabled else 'centre_third'),
                    bearing_errors_px=controller._last_bearing_errors if perception and perception.tracking else None,
                    result=decision.result, detail=decision.detail)
                last_control_event = event_key

            if decision.reground_now:
                print(f"[ground] 已停车，拍照并后台复检：{decision.motion.reason}")
                frame = stream.stereo_frame()
                if frame is not None:
                    left_jpeg = cv2.imencode(
                        ".jpg", frame[:, :half_width],
                        [int(cv2.IMWRITE_JPEG_QUALITY),
                         pursuit_cfg.jpeg_quality])[1].tobytes()
                    right_jpeg = cv2.imencode(
                        ".jpg", frame[:, half_width:],
                        [int(cv2.IMWRITE_JPEG_QUALITY),
                         pursuit_cfg.jpeg_quality])[1].tobytes()
                    if not reground_pending:
                        start_reground(left_jpeg, right_jpeg,
                                       active_instruction)
                else:
                    print("[search] 双路出帧失败，按未找到处理")
                    controller.on_ground_result(False, time.monotonic())

            # 下发运动命令（相同命令走 refresh，变速/变向走 set）
            # BLE 未就绪（ESP32 未上电）时不崩溃：每 15s 重试接入，
            # 下发失败冷却 2s 再重试；连续 3 次失败说明链路已死，
            # 关闭并重建（否则在死对象上无限重试）。
            motion = decision.motion
            pair = (motion.linear_mm_s, motion.angular_mrad_s)
            if client is None:
                if now >= next_link_attempt:
                    next_link_attempt = now + 15.0
                    try:
                        client = Esp32Client(SerialConfig(
                            link_mode=pursuit_cfg.link_mode,
                            ble_device_name=pursuit_cfg.ble_device_name,
                            ble_address=pursuit_cfg.ble_address,
                            ble_connect_timeout_s=(
                                pursuit_cfg.ble_connect_timeout_s),
                            wheel_output_gains=(
                                pursuit_cfg.wheel_output_gain,) * 4,
                        ))
                        client.start()
                        print("[link] BLE 链路已建立")
                    except Exception as exc:
                        print(f"[link] BLE 重试失败（ESP32 未上电？）: {exc}")
                        client = None
            else:
                try:
                    if motion.linear_mm_s == 0 and motion.angular_mrad_s == 0:
                        if last_command != pair:
                            client.stop()
                            last_command, last_sent_at = pair, now
                    elif pair != last_command or now - last_sent_at >= command_min_interval_s:
                        if pair == last_command:
                            client.refresh_twist(*pair, ttl_ms=motion.ttl_ms)
                        else:
                            client.set_twist(*pair, ttl_ms=motion.ttl_ms)
                        last_command, last_sent_at = pair, now
                    link_fail_streak = 0
                except Exception as exc:
                    print(f"[link] 运动命令下发失败: {exc}")
                    last_command = None
                    last_sent_at = now + 2.0  # 冷却 2s 再重试
                    link_fail_streak += 1
                    if link_fail_streak >= 3:
                        print("[link] 连续下发失败，关闭链路等待重建")
                        try:
                            client.close()
                        except Exception:
                            pass
                        client = None
                        link_fail_streak = 0
            # 链路未就绪时车实际动不了：搜索转角改用时长估算，避免卡步；
            # stuck 卡死检测一并挂起（否则 ESP32 恢复广播前的等待期会被
            # 误判卡死 GIVE_UP——2026-08-17 实测）
            if not dry_run:
                controller.set_simulate_motion(client is None)
                controller.set_link_ready(client is not None)

            # 每秒统计一次（采集/编码/上传），日志与 pi_state 共用这份
            # 快照——CameraReader.stats() 读后清零，双调用会双清零。
            if now - last_stats_at >= 1.0:
                last_stats_at = now
                stats = stream.stats_all()
                stats_msg = {
                    "left_fps": stats["left_fps"],
                    "right_fps": stats["right_fps"],
                    "left_ms": round(stats["left_ms"], 1),
                    "right_ms": round(stats["right_ms"], 1),
                    "enc_fps": stats["enc_fps"],
                    "enc_ms": round(stats["enc_ms"], 1),
                    "upload_fps": round(uploaded, 1),
                    "skip_fps": round(send_fail, 1),
                    "send_ms": round(send_ms_sum / max(uploaded, 1), 1),
                }
                uploaded = send_fail = 0
                send_ms_sum = 0.0

            # 状态机状态推送（1Hz）→ 3090 网页实时显示状态/指令/停车消息/帧率
            if now - last_state_sent_at >= 1.0:
                last_state_sent_at = now
                server.send_text({
                    "type": "pi_state",
                    "state": controller.state.name,
                    "cmd": [motion.linear_mm_s, motion.angular_mrad_s],
                    "detail": motion.reason,
                    "result": decision.result,
                    **(stats_msg or {}),
                })

            # 上传拼接帧（相机帧率，最新帧即可；帧前先上报当前运动指令：
            # 同一 WS 流内保序，服务端逐帧用 ego-motion 补偿卡尔曼预测/
            # 模板搜索中心，转弯与旋转重搜时预测才不会滞后）。
            # 无背压：服务端 drain 丢弃陈旧帧只处理最新，显示帧率 = 相机
            # 帧率，感知率 = 3090 处理能力（两者解耦）。
            if (upload_cap <= 0.0 or now >= next_upload_at) \
                    and not server.dead():
                seq, jpeg = stream.latest_stitched()
                if jpeg is not None and seq != last_upload_seq:
                    server.send_text({
                        "type": "motion",
                        # dry-run 的决策命令不会真正驱动车辆；若仍上报给
                        # 3090，ego-motion 会凭空补偿并触发漂移看门狗，
                        # 表现为框闪烁/丢失。服务端只接收实际执行的运动。
                        "v": 0 if dry_run else int(motion.linear_mm_s),
                        "w": 0 if dry_run else int(motion.angular_mrad_s),
                    })
                    t0 = time.monotonic()
                    ok = server.send_frame(jpeg)
                    send_ms_sum += (time.monotonic() - t0) * 1000.0
                    if ok:
                        uploaded += 1
                    else:
                        send_fail += 1
                    last_upload_seq = seq
                    if upload_cap > 0.0:
                        next_upload_at = now + 1.0 / upload_cap

            # 日志
            if now - last_log_at >= 1.0 or decision.finished:
                tracking = perception.tracking if perception else False
                bbox_values: list[str] = [""] * 6
                bbox2_values: list[str] = [""] * 6
                bbox_cx = ""
                bbox_cy = ""
                bbox_cx2 = ""
                bbox_cy2 = ""
                depth = ""
                depth_conf = ""
                if perception and perception.bbox:
                    bbox_cx = f"{(perception.bbox[0] + perception.bbox[2]) / 2:.0f}"
                    bbox_cy = f"{(perception.bbox[1] + perception.bbox[3]) / 2:.0f}"
                    bbox_values = [
                        *(f"{value:.1f}" for value in perception.bbox),
                        bbox_cx, bbox_cy,
                    ]
                if perception and perception.bbox2:
                    bbox_cx2 = f"{(perception.bbox2[0] + perception.bbox2[2]) / 2:.0f}"
                    bbox_cy2 = f"{(perception.bbox2[1] + perception.bbox2[3]) / 2:.0f}"
                    bbox2_values = [
                        *(f"{value:.1f}" for value in perception.bbox2),
                        bbox_cx2, bbox_cy2,
                    ]
                if perception and perception.depth_m is not None:
                    depth = f"{perception.depth_m:.2f}"
                    depth_conf = f"{perception.depth_conf:.2f}"
                perception_age_ms = ""
                perception_fps = ""
                if perception and perception.received_at > 0.0:
                    perception_age_ms = f"{max(0.0, now - perception.received_at) * 1000:.1f}"
                if perception and perception.frame_interval_s > 0.0:
                    perception_fps = f"{1.0 / perception.frame_interval_s:.2f}"
                lidar_front = ""
                scan = scan_holder.get()
                if scan is not None:
                    lidar_front = f"{controller._lidar_front(scan) or 0:.2f}"
                state = controller.state.name
                st = stats_msg or {}
                print(f"[pursuit] t={now - started:6.1f}s {state:<9} "
                      f"track={tracking} bbox_cx={bbox_cx}/{bbox_cx2} d={depth} "
                      f"lidar={lidar_front} cmd=({motion.linear_mm_s},"
                      f"{motion.angular_mrad_s}) {motion.reason} "
                      f"body_line_err={controller._last_bearing_errors if tracking and pursuit_cfg.forward_reference_enabled else None} "
                      f"cam={st.get('left_fps')}/{st.get('right_fps')}fps "
                      f"enc={st.get('enc_fps')}fps {st.get('enc_ms')}ms "
                      f"up={st.get('upload_fps')}fps skip={st.get('skip_fps')}")
                if not dry_run and client is not None and client.last_status is not None:
                    # Diagnostics, not measured travel: legacy hardware may
                    # have no wheel feedback or battery instrumentation.
                    print("[esp32] " + json.dumps(client.last_status, ensure_ascii=False))
                csv_writer.writerow([
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    f"{now - started:.2f}", state,
                    perception.frame_id if perception else "",
                    perception_age_ms, perception_fps, int(tracking),
                    perception.status if perception else "",
                    *bbox_values,
                    int(perception.tracking2 if perception else False),
                    perception.status2 if perception else "",
                    *bbox2_values,
                    perception.source if perception else "",
                    depth, depth_conf, lidar_front,
                    motion.linear_mm_s, motion.angular_mrad_s,
                    st.get("left_fps", ""), st.get("right_fps", ""),
                    st.get("enc_fps", ""), st.get("enc_ms", ""),
                    st.get("upload_fps", ""), st.get("skip_fps", ""),
                    st.get("send_ms", ""), motion.reason,
                ])
                csv_file.flush()
                last_log_at = now
                if scan_holder.error and scan_holder.error != last_lidar_error:
                    print(f"[lidar] {scan_holder.error}")
                    last_lidar_error = scan_holder.error

            if decision.finished and not finished_notified:
                finished_notified = True
                final_result = decision.result
                final_detail = decision.detail
                print(f"[pursuit] 任务结束: result={decision.result} "
                      f"detail={decision.detail}")
                if args.exit_on_finish:
                    # alignment_pending is a successful *handoff*, not proof
                    # of graspability.  pursuit_session performs the exact
                    # hull check after it restores the grab preview.
                    exit_code = (0 if decision.result in {
                        "success", "alignment_pending"} else 2)
                    print(f"[pursuit] 编排模式结束进程: exit_code={exit_code}")
                    break
                # 独立调试模式保留持续推流，网页“重新识别”可再次启动追踪。
                print("[pursuit] 继续推流中——网页重新识别可再次启动追踪")

            time.sleep(0.02)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[pursuit] 用户中断")
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        print(f"[pursuit] 未处理异常: {run_error}", file=sys.stderr)
        raise
    finally:
        if client is not None:
            try:
                client.stop()
            except Exception:
                pass  # 链路已断时 stop 可能抛 LinkError，不应中断清理流程
            try:
                client.close()
            except Exception:
                pass  # close 内 cancel_all/串口关闭同样可能失败，清理阶段不抛出
        for cleanup in (server.close, stream.close, lidar_stop.set,
                        csv_file.close):
            try:
                cleanup()
            except Exception:
                pass  # 清理阶段任何一步失败都不应掩盖任务结果
        if lidar is not None:
            try:
                lidar.close()
            except Exception:
                pass
        status = "error" if run_error else (
            "interrupted" if interrupted else (final_result or "closed"))
        run_logs.finalize(
            status,
            duration_s=round(time.monotonic() - started, 3),
            final_state=controller.state.name,
            final_result=final_result,
            final_detail=final_detail,
            active_instruction=active_instruction,
            error=run_error,
        )
        print(f"[log] 日志已保存: {run_logs.run_dir.resolve()}")
        run_logs.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
