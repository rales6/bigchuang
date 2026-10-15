"""Localize in a saved map, navigate to useful views, and take photographs.

The mission is intentionally read-only with respect to the map.  It stops on
uncertain localization, blocked paths, camera errors, or link errors.
"""

import argparse
from collections import deque
import csv
import glob
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time

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
    parser.add_argument("--turn-speed", type=int, default=1500)
    parser.add_argument("--slow-speed", type=int, default=280)
    parser.add_argument("--command-ttl-ms", type=int, default=700)
    parser.add_argument("--command-min-interval-ms", type=int, default=450)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.18)
    parser.add_argument("--heading-tolerance-deg", type=float, default=7.0)
    parser.add_argument("--stop-distance", type=float, default=0.42)
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
    parser.add_argument("--relocalization-distance", type=float, default=0.12)
    parser.add_argument("--relocalization-speed", type=int, default=300)
    parser.add_argument("--relocalization-max-speed", type=int, default=460)
    parser.add_argument("--relocalization-speed-step", type=int, default=50)
    parser.add_argument("--relocalization-turn-speed", type=int, default=1500)
    parser.add_argument("--relocalization-max-turn-speed", type=int, default=2600)
    parser.add_argument("--relocalization-turn-step", type=int, default=300)
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
        dual_usb_cameras=False,
    ):
        self.backend = backend
        self.device = int(device)
        self.width = int(width)
        self.height = int(height)
        self.timeout_s = float(timeout_s)
        self.dual_usb_cameras = bool(dual_usb_cameras)

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
            camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
            try:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if not camera.isOpened():
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
                camera = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
                camera.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG"),
                )
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not camera.isOpened():
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
                    camera.release()
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


class InspectionReport:
    def __init__(self, output_directory, map_path):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.directory = Path(output_directory) / stamp
        self.directory.mkdir(parents=True, exist_ok=True)
        self.map_path = str(map_path)
        self.records = []
        self.temporary_obstacle_events = []

    def add(self, record):
        self.records.append(record)
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
        Image.fromarray(rgb).save(
            self.directory / "temporary_obstacles.png"
        )

    def save_plan(self, known_map, start_pose, ordered_targets):
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
        image.save(self.directory / "inspection_plan.png")

    def save(self):
        payload = {
            "map": self.map_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "captures": self.records,
            "temporary_obstacle_events": self.temporary_obstacle_events,
            "temporary_obstacles_are_session_only": True,
        }
        (self.directory / "inspection_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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
        )
        with (self.directory / "inspection_manifest.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in self.records:
                writer.writerow({field: item.get(field, "") for field in fields})


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
            entity.update({
                "target_id": target.target_id,
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

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


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


def _command_for_pose(args, pose, waypoint, final_yaw=None):
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
            max(650, min(args.turn_speed, abs(heading_error) * 1100.0)),
            heading_error,
        ))
        return 0, turn, False
    if abs(heading_error) > math.radians(18.0):
        turn = int(math.copysign(
            max(650, min(args.turn_speed, abs(heading_error) * 1100.0)),
            heading_error,
        ))
        return 0, turn, False
    speed = args.slow_speed if distance < 0.50 else args.speed
    angular = int(max(-650, min(650, heading_error * 900.0)))
    return int(speed), angular, False


def _obstacle_safe(scan, linear_mm_s, stop_distance_m):
    if linear_mm_s <= 0:
        return True
    return sector_min_distance(
        scan,
        center_deg=0.0,
        width_deg=42.0,
        min_distance_m=0.08,
        max_distance_m=8.0,
    ) >= stop_distance_m


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
    delta = previous_pose.inverse().compose(result.pose)
    translation = math.hypot(delta.x_m, delta.y_m)
    duration_s = max(0.05, float(duration_s))
    return SlamUpdate(
        pose=result.pose,
        accepted=bool(result.reliable),
        rmse_m=float(result.rmse_m),
        correspondences=max(1, int(round(result.inlier_ratio * 100.0))),
        scan_points=100,
        inlier_ratio=float(result.inlier_ratio),
        translation_m=translation,
        rotation_rad=float(delta.yaw_rad),
        rejection_reason="" if result.reliable else "map_tracking_unreliable",
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
):
    """Move continuously to a new viewpoint while learning motor dead zones."""
    scan = initial_scan
    baseline = initial_scan
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
        print(
            "主动定位动作 cmd=({:+d},{:+d})，实测位移={:.3f}m、转角={:.1f}°，"
            "自学习直行补偿={:+d}mm/s。".format(
                applied.linear_mm_s,
                applied.angular_mrad_s,
                math.hypot(match.transform.x_m, match.transform.y_m),
                math.degrees(match.transform.yaw_rad),
                speed_controller.linear_adjustment_mm_s,
            ),
            flush=True,
        )
        return applied, match

    if linear == 0 and time.monotonic() < deadline_s:
        turn_baseline = scan
        for _attempt in range(10):
            _applied, _match = measured_step(MotionCommand(
                0, angular, "active_localization_turn", reason
            ))
            total_turn = localizer.relative_motion(turn_baseline, scan)
            if (
                total_turn.converged
                and abs(total_turn.transform.yaw_rad) >= math.radians(10.0)
            ):
                break
            if time.monotonic() >= deadline_s:
                break

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
                total_distance >= args.relocalization_distance
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
            measured_step(MotionCommand(
                0, turn, "active_localization_turn", turn_reason
            ))
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


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _localization_acceptance(results, args):
    window = list(results)[-max(1, args.localization_window):]
    accepted = [
        item for item in window
        if item is not None
        and item.reliable
        and item.ambiguity_margin >= args.localization_min_margin
    ]
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
    if args.localization_required_moves < 1:
        raise ValueError("initial localization requires at least one real move")
    if args.relocalization_max_speed < args.relocalization_speed:
        raise ValueError("主动定位最大直行速度不能小于起始速度")
    if args.relocalization_max_turn_speed < args.relocalization_turn_speed:
        raise ValueError("主动定位最大转向速度不能小于起始转向速度")
    if args.max_drive_speed < max(args.speed, args.slow_speed):
        raise ValueError("巡检最大直行速度不能小于基础直行速度")
    if not 1 <= args.photos_per_target <= 4:
        raise ValueError("photos-per-target must be between 1 and 4")
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
    targets = known_map.inspection_targets(
        standoff_m=args.standoff,
        maximum_targets=args.max_targets,
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
    camera = camera or CameraCapture(
        args.camera_backend,
        args.camera_device,
        args.camera_width,
        args.camera_height,
        args.camera_timeout,
        dual_usb_cameras=args.dual_usb_cameras,
    )
    report = InspectionReport(args.output, args.map)
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
        print("已加载只读地图：{}，候选拍摄位 {} 个。".format(args.map, len(targets)))
        print(
            "正在进行已知地图主动定位：先采集静止候选，随后会连续短距离移动，"
            "完成跨位置校验后才确认。",
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
        speed_controller = AdaptiveSpeedController(
            args.relocalization_max_turn_speed,
            args.max_drive_speed,
            AdaptiveSpeedConfig(
                minimum_turn_floor_mrad_s=args.turn_speed,
                initial_turn_floor_mrad_s=args.turn_speed,
                initial_turn_limit_mrad_s=args.relocalization_max_turn_speed,
                turn_adjust_step_mrad_s=max(50, args.relocalization_turn_step),
                linear_adjust_step_mm_s=max(20, args.drive_speed_step),
                maximum_linear_boost_mm_s=max(
                    0, args.max_drive_speed - min(args.speed, args.slow_speed)
                ),
                minimum_applied_linear_speed_mm_s=min(
                    args.speed, args.slow_speed
                ),
            ),
        )
        localization_results = deque(maxlen=max(1, args.localization_window))
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
            result = (
                localizer.localize_near_position(scan, hint)
                if hint is not None
                else localizer.global_localize(scan, hint=None)
            )
            if hint is not None and (result is None or not result.reliable):
                result = localizer.global_localize(scan, hint=None)
            localization_results.append(result)
            stationary_attempts += 1
            ready, candidate, ratio, accepted, total = _localization_acceptance(
                localization_results, args
            )
            if result is None:
                detail = "无有效候选"
            else:
                detail = (
                    "pose=({:+.2f},{:+.2f},{:+.1f}°) rmse={:.3f} "
                    "inliers={:.0f}% margin={:.3f}"
                ).format(
                    result.pose.x_m,
                    result.pose.y_m,
                    math.degrees(result.pose.yaw_rad),
                    result.rmse_m,
                    result.inlier_ratio * 100.0,
                    result.ambiguity_margin,
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
            after_motion_scan, _move_match = (
                _execute_active_localization_move(
                    client,
                    scan_iterator,
                    localizer,
                    speed_controller,
                    scan,
                    args,
                    deadline_s,
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
        ordered = order_targets(known_map, (pose.x_m, pose.y_m), targets)
        if not ordered:
            raise RuntimeError("当前位置无法通过地图白色区域到达任何拍摄位")
        report.save_plan(known_map, pose, ordered)
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
        target_index = 0
        last_pose_update_s = time.monotonic()
        last_status_print_s = -math.inf
        capture_mode = "exact"
        capture_note = ""
        while ordered_targets and not stopping:
            choices = []
            for pending_index, pending_target in enumerate(ordered_targets):
                pending_path = known_map.path(
                    (pose.x_m, pose.y_m), pending_target.view_xy_m
                )
                if not pending_path:
                    continue
                length = sum(
                    math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(pending_path, pending_path[1:])
                )
                choices.append((length, pending_index, pending_path))
            no_path_fallback = not choices
            if no_path_fallback:
                selected_index = min(
                    range(len(ordered_targets)),
                    key=lambda index: math.hypot(
                        ordered_targets[index].view_xy_m[0] - pose.x_m,
                        ordered_targets[index].view_xy_m[1] - pose.y_m,
                    ),
                )
                path = []
                print(
                    "剩余目标没有严格地图路径；选择最近目标在当前位置拍摄，"
                    "不会因为粗糙地图而漏图。",
                    flush=True,
                )
            else:
                _length, selected_index, path = min(
                    choices, key=lambda item: item[0]
                )
            target = ordered_targets.pop(selected_index)
            target_index += 1
            capture_mode = "approximate" if no_path_fallback else "exact"
            capture_note = "no_strict_map_path" if no_path_fallback else ""
            if stopping:
                break
            waypoints = (
                [] if no_path_fallback else (path[1:] or [target.view_xy_m])
            )
            replan_target = False
            for waypoint_index, waypoint in enumerate(waypoints):
                while not stopping:
                    if time.monotonic() - started >= args.max_runtime_min * 60.0:
                        stopping = True
                        break
                    scan = next(scan_iterator)
                    result = localizer.track(scan, pose)
                    if not _tracking_usable_for_photo_mission(result, args):
                        recovery = localizer.localize_near_position(
                            scan, pose, radius_m=0.45
                        )
                        if _tracking_usable_for_photo_mission(recovery, args):
                            result = recovery
                            print(
                                "局部跟踪质量下降，已在最后可靠位置附近恢复定位。",
                                flush=True,
                            )
                    if not _tracking_usable_for_photo_mission(result, args):
                        failures += 1
                        client.stop()
                        last_command = (0, 0)
                        speed_controller.stopped()
                        print(
                            "定位暂时不可靠 {}/{}，车辆已停车。".format(
                                failures, args.max_localization_failures
                            )
                        )
                        if failures >= args.max_localization_failures:
                            capture_mode = "approximate"
                            capture_note = "tracking_lost_near_target"
                            print(
                                "连续定位质量下降；保留最后可靠位姿并在当前位置拍摄本目标，"
                                "随后继续剩余拍摄点。",
                                flush=True,
                            )
                            replan_target = False
                            break
                        continue
                    failures = 0
                    update_now = time.monotonic()
                    speed_controller.observe(_tracking_update(
                        pose,
                        result,
                        update_now - last_pose_update_s,
                    ))
                    last_pose_update_s = update_now
                    pose = result.pose
                    final_yaw = (
                        normalize_angle(
                            target.yaw_rad
                            - math.radians(args.camera_yaw_offset_deg)
                        )
                        if waypoint_index == len(waypoints) - 1
                        else None
                    )
                    linear, angular, reached = _command_for_pose(
                        args, pose, waypoint, final_yaw=final_yaw
                    )
                    requested = MotionCommand(
                        linear,
                        angular,
                        "advancing" if linear else "turning",
                        "known-map waypoint tracking",
                        waypoint,
                    )
                    applied = speed_controller.apply(requested)
                    linear = applied.linear_mm_s
                    angular = applied.angular_mrad_s
                    if not _obstacle_safe(scan, linear, args.stop_distance):
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        added_cells = _remember_temporary_obstacle(
                            scan,
                            scan_iterator,
                            pose,
                            known_map,
                            args,
                        )
                        if added_cells:
                            report.add_temporary_obstacle(
                                known_map, pose, added_cells
                            )
                            print(
                                "发现永久地图中没有的新障碍：已加入本次运行的临时层 "
                                "({} 个栅格)，正在绕行；不会改写PGM/YAML。".format(
                                    added_cells
                                )
                            )
                        else:
                            print("前方受阻，正在按当前临时地图重新规划。")
                        if added_cells:
                            ordered_targets.append(target)
                            replan_target = True
                        else:
                            capture_mode = "approximate"
                            capture_note = "blocked_near_target"
                            print(
                                "阻挡没有形成可确认的临时障碍；不再追求地图中的精确拍摄位，"
                                "将在当前安全位置拍摄该目标。"
                            )
                            replan_target = False
                        break
                    command = (linear, angular)
                    if reached:
                        client.stop()
                        speed_controller.stopped()
                        last_command = (0, 0)
                        last_command_sent_s = time.monotonic()
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
                                    linear, angular, ttl_ms=args.command_ttl_ms
                                )
                            else:
                                client.set_twist(
                                    linear, angular, ttl_ms=args.command_ttl_ms
                                )
                        else:
                            client.set_twist(
                                linear, angular, ttl_ms=args.command_ttl_ms
                            )
                        last_command = command
                        last_command_sent_s = now
                if replan_target or capture_mode == "approximate":
                    break
            if replan_target:
                continue
            if stopping:
                break
            time.sleep(args.settle_seconds)
            # Best-effort stopped reconfirmation. The mission prioritizes
            # collecting an image; a coarse map does not justify dropping it.
            for _index in range(2):
                photo_scan = next(scan_iterator)
                result = localizer.track(photo_scan, pose)
                if not _tracking_usable_for_photo_mission(result, args):
                    result = localizer.localize_near_position(
                        photo_scan, pose, radius_m=0.35
                    )
                if _tracking_usable_for_photo_mission(result, args):
                    pose = result.pose
                    continue
                capture_mode = "approximate"
                if not capture_note:
                    capture_note = "photo_pose_recheck_unreliable"
                print(
                    "拍照前定位复核质量偏低；使用最后可靠位姿关联照片，不丢弃该目标。",
                    flush=True,
                )
                break
            image_name = "{:02d}_{}_{}_p01.jpg".format(
                target_index, target.kind, target.target_id
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
            for photo_index in range(2, args.photos_per_target + 1):
                time.sleep(min(0.6, max(0.15, args.settle_seconds * 0.25)))
                extra_name = "{:02d}_{}_{}_p{:02d}.jpg".format(
                    target_index, target.kind, target.target_id, photo_index
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
            completed_targets += 1
            print(
                "已拍摄 {}/{}：{} [{}] -> {}".format(
                    completed_targets, len(ordered), target.kind,
                    capture_mode, image_path
                )
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
