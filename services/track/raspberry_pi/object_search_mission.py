from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
import uuid

from PIL import Image, ImageDraw
from PIL import ImageFilter, ImageStat
import numpy as np

from raspberry_pi.config import LidarConfig, SerialConfig
from raspberry_pi.esp32 import Esp32Client
from raspberry_pi.known_map_navigation import (
    KnownOccupancyMap,
    LocalizationResult,
    ScanMapLocalizer,
    consistent_localization,
)
from raspberry_pi.lidar import N10LidarDriver
from raspberry_pi.map_camera_inspection import (
    CameraCapture,
    _command_for_pose,
    _execute_active_localization_move,
    _localization_acceptance,
    _localization_motion_consistent,
    _obstacle_safe,
    _remember_temporary_obstacle,
    _motion_update,
    _tracking_usable_for_photo_mission,
)
from raspberry_pi.mapping import AdaptiveSpeedConfig, AdaptiveSpeedController, MotionCommand, Pose2D
from raspberry_pi.mapping.geometry import normalize_angle
from raspberry_pi.search_route_planner import build_search_route
from raspberry_pi.visual_place_localization import (
    VisualPlaceMatch,
    VisualPlaceRecognizer,
    detection_landmark_labels,
    rank_landmark_candidates,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Localize, plan, navigate and search with guarded 3090 Qwen/YOLO supervision"
    )
    parser.add_argument("--decision", required=True)
    parser.add_argument("--map", default="maps/ble_scene_17.pgm")
    parser.add_argument("--map-yaml", default="maps/ble_scene_17.yaml")
    parser.add_argument("--output", default="search_runs/ble_scene_17")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--3090-url", dest="gpu_url", default="http://192.168.55.33:8011")
    parser.add_argument("--link", choices=("ble", "usb", "uart", "auto"), default="ble")
    parser.add_argument("--ble-name", default="ESP32-Robot-Car")
    parser.add_argument("--ble-address", default="64:B7:08:69:9A:E6")
    parser.add_argument("--ble-connect-timeout", type=float, default=6.0)
    parser.add_argument("--ble-operation-timeout", type=float, default=0.8)
    parser.add_argument("--link-connect-attempts", type=int, default=8)
    parser.add_argument("--link-connect-delay", type=float, default=1.0)
    parser.add_argument("--esp-port", default="/dev/serial0")
    parser.add_argument("--lidar-port", default="/dev/ttyACM0")
    parser.add_argument("--lidar-baud", type=int, default=230400)
    parser.add_argument("--angle-offset", type=float, default=0.0)
    parser.add_argument("--counterclockwise", action="store_true")
    parser.add_argument("--no-motor-control", action="store_true")
    parser.add_argument("--lidar-offset-x", type=float, default=0.14)
    parser.add_argument("--lidar-offset-y", type=float, default=0.0)
    parser.add_argument("--clearance", type=float, default=0.30)
    parser.add_argument("--front-left-gain", type=float, default=1.15)
    parser.add_argument("--front-right-gain", type=float, default=1.00)
    parser.add_argument("--rear-left-gain", type=float, default=0.90)
    parser.add_argument("--rear-right-gain", type=float, default=1.00)
    # Intentionally conservative first real-car values.
    parser.add_argument("--speed", type=int, default=240)
    parser.add_argument("--slow-speed", type=int, default=230)
    parser.add_argument("--max-drive-speed", type=int, default=460)
    parser.add_argument("--drive-speed-step", type=int, default=60)
    parser.add_argument(
        "--start-kick-speed", type=int, default=400,
        help="Minimum straight command until lidar confirms the chassis has started",
    )
    parser.add_argument("--turn-speed", type=int, default=1250)
    parser.add_argument("--max-turn-speed", type=int, default=2600)
    parser.add_argument("--command-ttl-ms", type=int, default=650)
    parser.add_argument("--command-min-interval-ms", type=int, default=400)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.18)
    parser.add_argument("--heading-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--stop-distance", type=float, default=0.45)
    parser.add_argument("--localization-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--localization-window", type=int, default=5)
    parser.add_argument("--localization-min-accepted", type=int, default=3)
    parser.add_argument("--localization-acceptance-ratio", type=float, default=0.60)
    parser.add_argument("--localization-min-margin", type=float, default=0.025)
    parser.add_argument("--localization-required-moves", type=int, default=1)
    parser.add_argument("--localization-motion-min-distance", type=float, default=0.07)
    parser.add_argument("--localization-motion-min-yaw-deg", type=float, default=8.0)
    parser.add_argument("--localization-motion-position-tolerance", type=float, default=0.24)
    parser.add_argument("--localization-motion-yaw-tolerance-deg", type=float, default=16.0)
    parser.add_argument(
        "--visual-localization-reference-root",
        default="inspections/ble_scene_17",
        help="Latest inspection_manifest.json and its permanent reference photographs",
    )
    parser.add_argument("--visual-localization-min-score", type=float, default=0.30)
    parser.add_argument("--visual-localization-min-margin", type=float, default=0.055)
    parser.add_argument("--visual-localization-min-inliers", type=int, default=9)
    parser.add_argument(
        "--visual-localization-semantic-map",
        default="maps/ble_scene_17_semantics_qwen.json",
        help="Permanent landmark annotations used to disambiguate radar pose basins",
    )
    parser.add_argument("--visual-landmark-min-score", type=float, default=2.4)
    parser.add_argument("--visual-landmark-min-margin", type=float, default=1.25)
    parser.add_argument("--visual-landmark-max-distance", type=float, default=3.5)
    parser.add_argument("--visual-landmark-fov-deg", type=float, default=150.0)
    parser.add_argument("--visual-localization-max-position-error", type=float, default=1.20)
    parser.add_argument("--visual-localization-max-yaw-error-deg", type=float, default=95.0)
    parser.add_argument("--no-visual-localization", dest="visual_localization", action="store_false")
    parser.set_defaults(visual_localization=True)
    parser.add_argument("--relocalization-distance", type=float, default=0.12)
    parser.add_argument("--relocalization-speed", type=int, default=300)
    parser.add_argument("--relocalization-max-speed", type=int, default=460)
    parser.add_argument("--relocalization-turn-speed", type=int, default=1500)
    parser.add_argument("--relocalization-max-turn-speed", type=int, default=2600)
    parser.add_argument("--relocalization-turn-step", type=int, default=300)
    parser.add_argument("--max-localization-failures", type=int, default=4)
    parser.add_argument("--relaxed-tracking-rmse", type=float, default=0.24)
    parser.add_argument("--relaxed-tracking-inlier-ratio", type=float, default=0.34)
    parser.add_argument("--temporary-obstacle-radius", type=float, default=0.14)
    parser.add_argument("--temporary-obstacle-confirm-distance", type=float, default=0.18)
    parser.add_argument("--temporary-obstacle-min-hits", type=int, default=2)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--camera-backend", choices=("auto", "rpicam", "libcamera", "opencv"), default="opencv")
    parser.add_argument("--camera-device", type=int, default=0)
    parser.add_argument("--dual-usb-cameras", action="store_true", default=True)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-timeout", type=float, default=12.0)
    parser.add_argument("--camera-capture-attempts", type=int, default=2)
    parser.add_argument("--camera-retry-delay", type=float, default=0.5)
    parser.add_argument(
        "--camera-stream-base-url",
        default="http://127.0.0.1:8080",
        help="Reuse the existing dual-camera MJPEG service; empty uses direct V4L2 capture",
    )
    parser.add_argument("--max-runtime-min", type=float, default=15.0)
    parser.add_argument("--max-perception-failures", type=int, default=3)
    parser.add_argument("--max-replans", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument(
        "--navigation-no-progress-seconds",
        type=float,
        default=18.0,
        help="Stop and replan when a waypoint does not get closer for this long",
    )
    parser.add_argument(
        "--navigation-progress-distance",
        type=float,
        default=0.04,
        help="Distance reduction that resets the navigation no-progress timer",
    )
    parser.add_argument("--live-status", default="search_runs/live_status.json")
    parser.add_argument("--live-map-image", default="search_runs/live_map.png")
    parser.add_argument(
        "--real-test-max-distance",
        type=float,
        default=0.80,
        help="Single low-speed acceptance run distance ceiling; 0 disables the ceiling",
    )
    return parser


def _serial_config(args: argparse.Namespace) -> SerialConfig:
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


def _lidar_config(args: argparse.Namespace) -> LidarConfig:
    return LidarConfig(
        port=args.lidar_port,
        baudrate=args.lidar_baud,
        angle_offset_deg=args.angle_offset,
        clockwise=not args.counterclockwise,
        motor_control=not args.no_motor_control,
    )


def _create_client_with_retry(factory, attempts=8, delay_s=1.0):
    """Create a stopped chassis client without advancing mission time."""
    last_error = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        candidate = None
        try:
            candidate = factory()
            candidate.heartbeat()
            candidate.stop()
            print(f"底盘链路连接成功 {attempt}/{attempts}", flush=True)
            return candidate
        except Exception as exc:
            last_error = exc
            print(f"底盘链路连接失败 {attempt}/{attempts}：{exc}", flush=True)
            if candidate is not None:
                try:
                    candidate.close(send_cancel=False)
                except TypeError:
                    candidate.close()
                except Exception:
                    pass
            if attempt < int(attempts):
                time.sleep(max(0.2, float(delay_s)))
    raise RuntimeError(f"底盘链路连续连接失败 {attempts} 次：{last_error}")


class MissionLog:
    def __init__(self, root: str | Path, instruction: str) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.directory = Path(root) / stamp
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "mission.json"
        self.data: dict[str, Any] = {
            "schema": "object_search_mission_log.v1",
            "instruction": instruction,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "events": [],
            "result": "running",
            "hardware_motion_enabled": False,
        }

    def event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        item = {
            "index": len(self.data["events"]),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            **fields,
        }
        self.data["events"].append(item)
        self.save()
        return item

    def save(self) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


class SearchDashboardPublisher:
    """Atomically publish a known-map task overlay for the port-8766 server."""

    def __init__(self, known_map, image_path: str | Path, status_path: str | Path, decision):
        self.map = known_map
        self.image_path = Path(image_path)
        self.status_path = Path(status_path)
        self.decision = decision
        self.route: dict[str, Any] | None = None
        self.pose: Pose2D | None = None
        self.current_view_index: int | None = None
        self.visited: set[int] = set()
        self.found_pose: Pose2D | None = None
        self.state = "localizing"
        self.navigation: dict[str, Any] = {}
        self.reason = "正在使用已知地图定位"
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.publish()

    def _pixel(self, world_xy) -> tuple[int, int]:
        column, row = self.map.world_to_grid((world_xy,))[0]
        return int(column), int(self.map.height - 1 - row)

    def publish(self) -> None:
        image = Image.fromarray(self.map.image, mode="L").transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        ).convert("RGB")
        draw = ImageDraw.Draw(image)
        radius = max(3, int(round(0.12 / self.map.resolution_m)))
        anchor = (self.decision.get("anchor") or {}).get("map_xy")
        if anchor:
            x, y = self._pixel(anchor)
            draw.rectangle((x-radius, y-radius, x+radius, y+radius), outline=(255, 0, 210), width=3)
        if self.route:
            for index, view in enumerate(self.route.get("views") or []):
                points = [self._pixel(point) for point in view.get("path_xy_m") or []]
                if len(points) >= 2:
                    draw.line(points, fill=(0, 100, 255), width=2)
                x, y = self._pixel(view["view_xy_m"])
                colour = (0, 190, 90) if index in self.visited else (0, 100, 255)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=colour)
                draw.text((x+radius+1, y-radius), str(index + 1), fill=(20, 20, 20))
        if self.pose:
            x, y = self._pixel((self.pose.x_m, self.pose.y_m))
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=(255, 120, 0))
        if self.found_pose:
            x, y = self._pixel((self.found_pose.x_m, self.found_pose.y_m))
            size = radius + 3
            draw.line((x-size, y-size, x+size, y+size), fill=(255, 0, 0), width=3)
            draw.line((x-size, y+size, x+size, y-size), fill=(255, 0, 0), width=3)
        image_tmp = self.image_path.with_suffix(self.image_path.suffix + ".tmp")
        image.save(image_tmp, format="PNG")
        image_tmp.replace(self.image_path)
        status = {
            "running": self.state not in {"success", "failure", "stopped"},
            "accepted": self.pose is not None,
            "scan_count": 0,
            "pose": None if self.pose is None else {
                "x_m": self.pose.x_m, "y_m": self.pose.y_m,
                "yaw_deg": math.degrees(self.pose.yaw_rad),
            },
            "command": {"state": self.state, "reason": self.reason},
            "navigation": self.navigation,
            "map_status": "寻物：{} / 地标：{}".format(
                (self.decision.get("target") or {}).get("query") or "未知目标",
                (self.decision.get("anchor") or {}).get("name")
                or (self.decision.get("anchor") or {}).get("label")
                or "指定区域",
            ),
            "task": {
                "instruction": self.decision.get("instruction"),
                "target": self.decision.get("target"),
                "anchor": self.decision.get("anchor"),
                "current_view_index": self.current_view_index,
                "visited_view_indices": sorted(self.visited),
                "found_at_robot_pose": self.found_pose is not None,
            },
            "map": {
                "width_cells": self.map.width,
                "height_cells": self.map.height,
                "resolution_m": self.map.resolution_m,
                "observed_cells": int((self.map.image != 205).sum()),
                "occupied_cells": int(self.map.occupied.sum()),
            },
        }
        status_tmp = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        status_tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        status_tmp.replace(self.status_path)

    def update(
        self, *, pose=None, route=None, view_index=None, visited=None,
        state=None, reason=None, found=False, navigation=None,
    ):
        if pose is not None:
            self.pose = pose
        if route is not None:
            self.route = route
        if view_index is not None:
            self.current_view_index = int(view_index)
        if visited is not None:
            self.visited = set(visited)
        if state is not None:
            self.state = str(state)
        if reason is not None:
            self.reason = str(reason)
        if navigation is not None:
            self.navigation = dict(navigation)
        if found and self.pose is not None:
            self.found_pose = self.pose
        self.publish()


class GpuMissionClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"3090 request failed: {exc}") from exc
        if not isinstance(value, dict) or not value.get("ok", False):
            raise RuntimeError(str(value.get("error") if isinstance(value, dict) else value))
        return value

    def next(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(self._json("/mission/next", state)["directive"])

    def perceive(self, left_path: Path, right_path: Path, context: dict[str, Any]) -> dict[str, Any]:
        # urllib multipart keeps the Pi dependency-free.
        boundary = "----car-search-" + uuid.uuid4().hex
        chunks: list[bytes] = []

        def field(name: str, value: str) -> None:
            chunks.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"), b"\r\n",
            ])

        def file_field(name: str, path: Path) -> None:
            chunks.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode(),
                b"Content-Type: image/jpeg\r\n\r\n", path.read_bytes(), b"\r\n",
            ])

        file_field("left_image", left_path)
        file_field("right_image", right_path)
        field("observation_id", str(context.get("observation_id") or uuid.uuid4().hex))
        field("context_json", json.dumps(context, ensure_ascii=False))
        chunks.append(f"--{boundary}--\r\n".encode())
        request = urllib_request.Request(
            self.base_url + "/perceive",
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"3090 perception failed: {exc}") from exc
        if not isinstance(value, dict) or not value.get("ok", False):
            raise RuntimeError(str(value.get("error") if isinstance(value, dict) else value))
        return value

    def visual_localize(
        self,
        current_left: Path,
        current_right: Path,
        reference_left: Path,
        reference_right: Path,
        references: list[dict[str, Any]],
    ) -> dict[str, Any]:
        boundary = "----car-place-" + uuid.uuid4().hex
        chunks: list[bytes] = []

        def file_field(name: str, path: Path) -> None:
            chunks.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode(),
                b"Content-Type: image/jpeg\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            ])

        def field(name: str, value: str) -> None:
            chunks.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ])

        file_field("current_left", current_left)
        file_field("current_right", current_right)
        file_field("reference_left", reference_left)
        file_field("reference_right", reference_right)
        field("references_json", json.dumps(references, ensure_ascii=False))
        chunks.append(f"--{boundary}--\r\n".encode())
        request = urllib_request.Request(
            self.base_url + "/localize/visual",
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"3090 visual localization failed: {exc}") from exc
        if not isinstance(value, dict) or not value.get("ok", False):
            raise RuntimeError(str(value.get("error") if isinstance(value, dict) else value))
        return value

    def visual_similarity(
        self,
        current_left: Path,
        current_right: Path,
        reference_left: Path,
        reference_right: Path,
    ) -> dict[str, Any]:
        boundary = "----car-embedding-" + uuid.uuid4().hex
        chunks: list[bytes] = []
        for name, path in (
            ("current_left", current_left),
            ("current_right", current_right),
            ("reference_left", reference_left),
            ("reference_right", reference_right),
        ):
            chunks.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode(),
                b"Content-Type: image/jpeg\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            ])
        chunks.append(f"--{boundary}--\r\n".encode())
        request = urllib_request.Request(
            self.base_url + "/localize/embedding",
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"3090 visual embedding failed: {exc}") from exc
        if not isinstance(value, dict) or not value.get("ok", False):
            raise RuntimeError(str(value.get("error") if isinstance(value, dict) else value))
        return value


class MjpegCameraCapture:
    """Capture one fresh JPEG from each existing 8080 MJPEG stream."""

    def __init__(self, base_url: str, timeout_s: float = 12.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _frame(self, name: str) -> bytes:
        try:
            with urllib_request.urlopen(
                self.base_url + "/video/" + name, timeout=self.timeout_s
            ) as response:
                buffer = b""
                deadline = time.monotonic() + self.timeout_s
                while time.monotonic() < deadline:
                    chunk = response.read(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start >= 0 and end >= 0:
                        return buffer[start:end + 2]
                    if len(buffer) > 4 * 1024 * 1024:
                        buffer = buffer[-1024 * 1024:]
        except (OSError, urllib_error.URLError) as exc:
            raise RuntimeError(f"相机流 {name} 不可用：{exc}") from exc
        raise RuntimeError(f"相机流 {name} 未在超时内返回完整 JPEG")

    @staticmethod
    def _quality(path: Path) -> dict[str, Any]:
        with Image.open(path) as image:
            gray = image.convert("L")
            stats = ImageStat.Stat(gray)
            small = gray.copy()
            small.thumbnail((320, 240))
            pixels = np.asarray(small, dtype=np.uint8)
            edges = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
        return {
            "brightness": round(float(stats.mean[0]), 2),
            "contrast": round(float(stats.stddev[0]), 2),
            "dark_fraction": round(float(np.mean(pixels < 55)), 4),
            "edge_fraction": round(float(np.mean(edges > 45)), 4),
            "possibly_blurred_or_flat": float(stats.stddev[0]) < 18.0,
        }

    def capture(self, path: Path) -> dict[str, Any]:
        second_path = path.with_name(path.stem + "_camera_2" + path.suffix)
        path.write_bytes(self._frame("left"))
        second_path.write_bytes(self._frame("right"))
        info = {"backend": "mjpeg", "camera_count": 2,
                "camera_images": [path.name, second_path.name],
                "camera_2_image": second_path.name}
        info.update(self._quality(path))
        second = self._quality(second_path)
        info.update({f"camera_2_{key}": value for key, value in second.items()})
        return info


class ResilientCameraCapture:
    """Prefer the shared MJPEG stream and fall back to direct V4L2 capture.

    The camera stream is an optimization, not a mission dependency.  A reboot
    or crashed stream server must not strand a robot after it has already
    navigated to an observation point.
    """

    def __init__(self, primary, fallback, attempts=2, retry_delay_s=0.5):
        self.backends = [primary, fallback]
        self.active_index = 0
        self.attempts = max(1, int(attempts))
        self.retry_delay_s = max(0.0, float(retry_delay_s))

    @staticmethod
    def _remove_partial_images(path: Path) -> None:
        for candidate in (
            path,
            path.with_name(path.stem + "_camera_2" + path.suffix),
        ):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def capture(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        failures = []
        order = [self.active_index] + [
            index for index in range(len(self.backends))
            if index != self.active_index
        ]
        for attempt in range(self.attempts):
            for index in order:
                self._remove_partial_images(path)
                try:
                    info = dict(self.backends[index].capture(path))
                    self.active_index = index
                    info["camera_failover_used"] = index != 0
                    if failures:
                        info["camera_recovered_after"] = list(failures)
                    return info
                except Exception as exc:
                    failures.append(
                        "{}: {}".format(
                            type(self.backends[index]).__name__, exc
                        )
                    )
            if attempt + 1 < self.attempts and self.retry_delay_s > 0:
                time.sleep(self.retry_delay_s)
        raise RuntimeError(
            "相机流和直接 V4L2 拍照均不可用（已重试 {} 次）：{}".format(
                self.attempts, "；".join(failures)
            )
        )

def _normalise(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def _line_safe_with_margin(known_map, start_xy, end_xy, extra_margin_m=0.04) -> bool:
    """Check the planned center line and two parallel swept-envelope lines."""
    dx = float(end_xy[0]) - float(start_xy[0])
    dy = float(end_xy[1]) - float(start_xy[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return True
    nx, ny = -dy / length, dx / length
    for offset in (0.0, float(extra_margin_m), -float(extra_margin_m)):
        shifted_start = (float(start_xy[0]) + nx * offset, float(start_xy[1]) + ny * offset)
        shifted_end = (float(end_xy[0]) + nx * offset, float(end_xy[1]) + ny * offset)
        if not known_map.line_is_clear(shifted_start, shifted_end):
            return False
    return True


def _densify_path(known_map, path, spacing_m=0.18) -> list[tuple[float, float]]:
    """Keep short control segments so feedback cannot cut a long wall corner."""
    if not path:
        return []
    result = [tuple(float(value) for value in path[0])]
    for start, end in zip(path, path[1:]):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        count = max(1, int(math.ceil(math.hypot(dx, dy) / spacing_m)))
        for index in range(1, count + 1):
            fraction = index / count
            result.append((float(start[0]) + dx * fraction, float(start[1]) + dy * fraction))
    return result


def _strict_command_for_pose(args, pose, waypoint, tolerance_m=None):
    """Follow a waypoint without producing a turn-stop-turn limit cycle."""
    dx = float(waypoint[0]) - pose.x_m
    dy = float(waypoint[1]) - pose.y_m
    distance = math.hypot(dx, dy)
    effective_tolerance = (
        float(args.waypoint_tolerance)
        if tolerance_m is None
        else min(float(args.waypoint_tolerance), float(tolerance_m))
    )
    if distance <= effective_tolerance:
        return 0, 0, True
    desired = math.atan2(dy, dx)
    heading_error = math.atan2(math.sin(desired - pose.yaw_rad), math.cos(desired - pose.yaw_rad))
    pure_turn_threshold = math.radians(max(18.0, args.heading_tolerance_deg * 2.25))
    if abs(heading_error) > pure_turn_threshold:
        angular = int(math.copysign(
            max(650, min(args.turn_speed, abs(heading_error) * 1100.0)), heading_error
        ))
        return 0, angular, False
    speed = args.slow_speed if distance < 0.50 else args.speed
    if abs(heading_error) <= math.radians(args.heading_tolerance_deg):
        angular = 0
    else:
        angular = int(max(-450, min(450, heading_error * 950.0)))
    return int(speed), angular, False


def _scan_motion_prediction(pose, match):
    """Predict the global pose from a physically plausible adjacent scan pair."""
    if match is None or not match.converged or not math.isfinite(match.rmse_m):
        return pose, False
    translation = math.hypot(match.transform.x_m, match.transform.y_m)
    if (
        match.inlier_ratio < 0.42
        or match.rmse_m > 0.16
        or translation > 0.22
        or abs(match.transform.yaw_rad) > math.radians(24.0)
    ):
        return pose, False
    return pose.compose(match.transform), True


def _fuse_tracking_pose(predicted_pose, result, correction_gain=0.45):
    """Apply a bounded map correction to scan-odometry prediction."""
    if result is None:
        return predicted_pose
    correction = predicted_pose.inverse().compose(result.pose)
    distance = math.hypot(correction.x_m, correction.y_m)
    translation_scale = (
        min(1.0, 0.035 / max(distance, 1e-9)) * float(correction_gain)
    )
    yaw = max(
        -math.radians(2.0),
        min(math.radians(2.0), correction.yaw_rad * float(correction_gain)),
    )
    return predicted_pose.compose(Pose2D(
        correction.x_m * translation_scale,
        correction.y_m * translation_scale,
        normalize_angle(yaw),
    ))


def _result_with_pose(result, pose):
    return LocalizationResult(
        pose=pose,
        rmse_m=result.rmse_m,
        inlier_ratio=result.inlier_ratio,
        score=result.score,
        ambiguity_margin=result.ambiguity_margin,
    )


def _red_pixel_score(path: Path, detection: dict[str, Any]) -> float:
    from colorsys import rgb_to_hsv
    import numpy as np

    image = Image.open(path).convert("RGB")
    bbox = detection.get("bbox_1000") or []
    if not isinstance(bbox, list) or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    x1, x2 = sorted((int(x1 * image.width / 1000), int(x2 * image.width / 1000)))
    y1, y2 = sorted((int(y1 * image.height / 1000), int(y2 * image.height / 1000)))
    crop = np.asarray(image)[max(0, y1):min(image.height, y2), max(0, x1):min(image.width, x2)]
    if crop.size == 0:
        return 0.0
    pixels = crop.reshape(-1, 3).astype(float) / 255.0
    scores = []
    for red, green, blue in pixels:
        hue, saturation, value = rgb_to_hsv(red, green, blue)
        if value >= 0.18 and value <= 0.98 and saturation >= 0.28:
            scores.append(hue <= 0.06 or hue >= 0.94)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _complete_bbox_1000(
    bbox: Any, *, edge_margin: float = 15.0, minimum_size: float = 12.0
) -> tuple[bool, dict[str, float | bool]]:
    """Reject detections that are invalid, tiny, or likely cropped by an image edge."""
    evidence: dict[str, float | bool] = {
        "bbox_valid": False,
        "bbox_complete": False,
        "edge_margin_1000": edge_margin,
    }
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False, evidence
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return False, evidence
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return False, evidence
    width = x2 - x1
    height = y2 - y1
    valid = (
        0.0 <= x1 < x2 <= 1000.0
        and 0.0 <= y1 < y2 <= 1000.0
        and width >= minimum_size
        and height >= minimum_size
    )
    complete = bool(
        valid
        and x1 >= edge_margin
        and y1 >= edge_margin
        and x2 <= 1000.0 - edge_margin
        and y2 <= 1000.0 - edge_margin
    )
    evidence.update({
        "bbox_valid": valid,
        "bbox_complete": complete,
        "bbox_width_1000": round(width, 2),
        "bbox_height_1000": round(height, 2),
    })
    return complete, evidence


def _same_stereo_target(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Conservatively associate a complete left/right detection with one target."""
    if _normalise(left.get("class_name")) != _normalise(right.get("class_name")):
        return False
    left_appearance = _normalise(left.get("appearance_key"))
    right_appearance = _normalise(right.get("appearance_key"))
    if left_appearance and right_appearance and left_appearance != right_appearance:
        return False
    left_bbox = [float(value) for value in left["bbox_1000"][:4]]
    right_bbox = [float(value) for value in right["bbox_1000"][:4]]
    left_centre_y = (left_bbox[1] + left_bbox[3]) / 2.0
    right_centre_y = (right_bbox[1] + right_bbox[3]) / 2.0
    if abs(left_centre_y - right_centre_y) > 300.0:
        return False
    left_area = (left_bbox[2] - left_bbox[0]) * (left_bbox[3] - left_bbox[1])
    right_area = (right_bbox[2] - right_bbox[0]) * (right_bbox[3] - right_bbox[1])
    return min(left_area, right_area) / max(left_area, right_area) >= 0.15


def confirm_detections(
    detections: list[dict[str, Any]], target: dict[str, Any], image_paths: dict[str, Path]
) -> list[dict[str, Any]]:
    required_class = _normalise(target.get("class_name"))
    required_colour = _normalise(target.get("color"))
    detection_mode = _normalise(target.get("detection_mode"))
    generic = required_class in {"", "any yolo object", "any object"}
    left_path = image_paths.get("left")
    right_path = image_paths.get("right")
    distinct_camera_frames = bool(
        left_path is not None
        and right_path is not None
        and left_path.is_file()
        and right_path.is_file()
        and left_path.resolve() != right_path.resolve()
    )
    result = []
    for detection in detections:
        item = dict(detection)
        detected_class = _normalise(item.get("class_name"))
        class_ok = bool(detected_class) and (generic or detected_class == required_class)
        source = _normalise(item.get("source"))
        confidence = float(item.get("confidence") or 0.0)
        bbox = item.get("bbox_1000") or []
        bbox_complete, bbox_evidence = _complete_bbox_1000(bbox)
        bbox_ok = bool(bbox_evidence["bbox_valid"])
        if detection_mode == "qwen fallback":
            source_ok = "qwen" in source and confidence >= 0.55 and bbox_ok
        elif detection_mode in {"yolo primary", "hybrid generic"}:
            source_ok = confidence >= 0.20 and bbox_ok
        else:
            source_ok = True
        attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
        detected_colour = _normalise(attributes.get("color"))
        colour_score = 0.0
        path = image_paths.get(str(item.get("camera_id") or ""))
        if required_colour == "red" and path is not None:
            colour_score = _red_pixel_score(path, item)
            colour_ok = colour_score >= 0.30
        else:
            colour_ok = not required_colour or detected_colour == required_colour
        camera_id = _normalise(item.get("camera_id"))
        single_camera_match = bool(
            camera_id in {"left", "right"}
            and source_ok
            and class_ok
            and colour_ok
            and bbox_complete
        )
        # Mission success is assigned only after a compatible complete detection
        # from the other physical camera is found below.
        item["confirmed_match"] = False
        item["single_camera_match"] = single_camera_match
        item["complete_in_frame"] = bbox_complete
        item["dual_camera_confirmed"] = False
        item["search_evidence"] = {
            "detection_mode": detection_mode or "legacy",
            "camera_id": camera_id,
            "distinct_camera_frames": distinct_camera_frames,
            "source_ok": source_ok,
            "class_ok": class_ok,
            "colour_ok": colour_ok,
            "colour_score": round(colour_score, 4),
            **bbox_evidence,
        }
        result.append(item)

    if not distinct_camera_frames:
        return result

    left_candidates = [
        item for item in result
        if _normalise(item.get("camera_id")) == "left" and item["single_camera_match"]
    ]
    right_candidates = [
        item for item in result
        if _normalise(item.get("camera_id")) == "right" and item["single_camera_match"]
    ]
    pairs = []
    for left in left_candidates:
        for right in right_candidates:
            if _same_stereo_target(left, right):
                score = float(left.get("confidence") or 0.0) + float(right.get("confidence") or 0.0)
                pairs.append((score, left, right))
    if pairs:
        _score, left, right = max(pairs, key=lambda value: value[0])
        pair_id = "stereo_target_1"
        for item, peer in ((left, right), (right, left)):
            item["confirmed_match"] = True
            item["dual_camera_confirmed"] = True
            item["stereo_pair_id"] = pair_id
            item["search_evidence"]["peer_camera_id"] = _normalise(peer.get("camera_id"))
    return result


def _visual_localization_check(
    args,
    candidate,
    scan,
    localizer,
    camera,
    recognizer,
    gpu,
    directory: Path,
    attempt: int,
):
    """Disambiguate radar basins with permanent, semantically labelled objects."""
    image_path = directory / f"localization_visual_{attempt:02d}.jpg"
    camera_info = camera.capture(image_path)
    second_name = camera_info.get("camera_2_image")
    image_paths = [image_path]
    if second_name:
        second = image_path.with_name(str(second_name))
        if second.is_file():
            image_paths.append(second)
    # Pixel feature matching is retained only as diagnostics. Cross-session
    # calibration showed that repeated wall/floor texture can create strong
    # but false homographies, so it is never authoritative by itself.
    local_match = recognizer.match(image_paths)
    perception = None
    perception_error = ""
    radar_candidates = []
    landmark_ranking = []
    match = None
    authoritative_visual_match = False
    try:
        radar_candidates = localizer.global_localization_candidates(scan, maximum=4)
        if not radar_candidates:
            radar_candidates = [candidate]
        elif not any(
            math.hypot(item.pose.x_m - candidate.pose.x_m, item.pose.y_m - candidate.pose.y_m) < 0.35
            and abs(normalize_angle(item.pose.yaw_rad - candidate.pose.yaw_rad)) < math.radians(20.0)
            for item in radar_candidates
        ):
            radar_candidates.insert(0, candidate)
        perception = gpu.perceive(
            image_paths[0],
            image_paths[1] if len(image_paths) > 1 else image_paths[0],
            {
                "observation_id": f"visual-localization-{attempt:02d}",
                "purpose": "permanent_landmark_localization",
                "search_phase": "localization",
                "target": {},
            },
        )
        detected_labels = detection_landmark_labels(perception.get("detections") or [])
        landmark_ranking = rank_landmark_candidates(
            radar_candidates,
            recognizer.landmarks,
            detected_labels,
            maximum_distance_m=args.visual_landmark_max_distance,
            field_of_view_deg=args.visual_landmark_fov_deg,
            known_map=localizer.map,
        )
        if landmark_ranking:
            best = landmark_ranking[0]
            runner_score = landmark_ranking[1].score if len(landmark_ranking) > 1 else 0.0
            authoritative_visual_match = bool(
                best.score >= args.visual_landmark_min_score
                and best.score - runner_score >= args.visual_landmark_min_margin
                and best.matched_labels
            )
            if authoritative_visual_match:
                winner = radar_candidates[best.candidate_index]
                nearest_place = min(
                    recognizer.places,
                    key=lambda place: math.hypot(
                        place.pose.x_m - winner.pose.x_m,
                        place.pose.y_m - winner.pose.y_m,
                    ),
                )
                match = VisualPlaceMatch(
                    target_id="landmarks:" + "+".join(best.matched_labels),
                    pose=winner.pose,
                    score=best.score,
                    runner_up_score=runner_score,
                    margin=best.score - runner_score,
                    inliers=0 if local_match is None else local_match.inliers,
                    good_matches=0 if local_match is None else local_match.good_matches,
                    reference_images=tuple(str(path) for path in nearest_place.images),
                    manifest=str(nearest_place.manifest),
                    reliable=True,
                )
    except Exception as exc:
        perception_error = str(exc)

    diagnostics = {
        "capture": [str(path) for path in image_paths],
        "reference_count": len(recognizer.places),
        "camera_count": camera_info.get("camera_count", len(image_paths)),
        "match": None if match is None else match.to_dict(),
        "local_feature_match": None if local_match is None else local_match.to_dict(),
        "detected_landmarks": sorted(detection_landmark_labels(
            [] if perception is None else (perception.get("detections") or [])
        )),
        "landmark_candidates": [
            {
                "candidate_index": item.candidate_index,
                "score": round(item.score, 4),
                "matched_labels": list(item.matched_labels),
                "expected_labels": list(item.expected_labels),
            }
            for item in landmark_ranking
        ],
        "landmark_reliable": authoritative_visual_match,
        "perception_error": perception_error,
    }
    if not authoritative_visual_match or match is None:
        return "uncertain", candidate, match, diagnostics

    position_error = math.hypot(
        candidate.pose.x_m - match.pose.x_m,
        candidate.pose.y_m - match.pose.y_m,
    )
    yaw_error = abs(normalize_angle(candidate.pose.yaw_rad - match.pose.yaw_rad))
    diagnostics.update({
        "radar_visual_position_error_m": round(position_error, 4),
        "radar_visual_yaw_error_deg": round(math.degrees(yaw_error), 2),
    })
    if (
        position_error <= args.visual_localization_max_position_error
        and yaw_error <= math.radians(args.visual_localization_max_yaw_error_deg)
    ):
        return "confirmed", candidate, match, diagnostics

    # Fixed landmarks select a different radar basin. Re-run geometric fitting
    # around that basin; camera semantics never become a metric pose directly.
    guided = localizer.localize_near_position(
        scan,
        match.pose,
        radius_m=max(0.45, args.visual_localization_max_position_error),
    )
    if guided is not None and guided.reliable:
        guided_error = math.hypot(
            guided.pose.x_m - match.pose.x_m,
            guided.pose.y_m - match.pose.y_m,
        )
        guided_yaw_error = abs(normalize_angle(guided.pose.yaw_rad - match.pose.yaw_rad))
        diagnostics.update({
            "guided_position_error_m": round(guided_error, 4),
            "guided_yaw_error_deg": round(math.degrees(guided_yaw_error), 2),
            "guided_rmse_m": round(guided.rmse_m, 4),
            "guided_inlier_ratio": round(guided.inlier_ratio, 4),
        })
        if (
            guided_error <= args.visual_localization_max_position_error
            and guided_yaw_error <= math.radians(args.visual_localization_max_yaw_error_deg)
        ):
            return "corrected", guided, match, diagnostics
    return "conflict", candidate, match, diagnostics


def _localize(
    args,
    client,
    scan_iterator,
    localizer,
    speed_controller,
    stopping,
    *,
    camera=None,
    visual_recognizer=None,
    gpu=None,
    visual_directory: Path | None = None,
    event_callback=None,
    status_callback=None,
) -> tuple[Pose2D, Any]:
    results = deque(maxlen=max(1, args.localization_window))
    anchor = None
    anchor_scan = None
    pending_match = None
    validated_moves = 0
    attempts = 0
    deadline = time.monotonic() + args.localization_timeout_seconds
    latest_scan = None
    visual_attempts = 0
    forced_visual_hint = None
    while not stopping() and time.monotonic() < deadline:
        latest_scan = next(scan_iterator)
        if anchor is not None:
            result = localizer.localize_near_position(latest_scan, anchor.pose)
        elif forced_visual_hint is not None:
            result = localizer.localize_near_position(
                latest_scan,
                forced_visual_hint,
                radius_m=max(0.45, args.visual_localization_max_position_error),
            )
            if result is None or not result.reliable:
                result = localizer.global_localize(latest_scan, hint=None)
        else:
            result = localizer.global_localize(latest_scan, hint=None)
        results.append(result)
        attempts += 1
        ready, candidate, ratio, accepted, total = _localization_acceptance(results, args)
        print(f"定位候选：accepted={accepted}/{total} ({ratio:.0%})", flush=True)
        if ready:
            if anchor is None:
                anchor, anchor_scan = candidate, latest_scan
            elif pending_match is not None:
                valid, position_error, yaw_error, measured_distance = (
                    _localization_motion_consistent(
                        anchor, candidate, pending_match, args
                    )
                )
                if not valid:
                    print(
                        "跨位置校验未通过：位移 {:.3f}m、转角 {:.1f}°，"
                        "地图一致性误差 {:.3f}m/{:.1f}°。".format(
                            measured_distance,
                            math.degrees(abs(pending_match.transform.yaw_rad)),
                            position_error,
                            math.degrees(yaw_error),
                        ),
                        flush=True,
                    )
                validated_moves = validated_moves + 1 if valid else 0
                anchor, anchor_scan, pending_match = candidate, latest_scan, None
                if validated_moves >= args.localization_required_moves:
                    client.stop()
                    if (
                        camera is not None
                        and gpu is not None
                        and visual_recognizer is not None
                        and visual_recognizer.available
                    ):
                        visual_attempts += 1
                        if status_callback is not None:
                            status_callback(
                                "visual_localizing",
                                "雷达定位已稳定，正在用历史标注照片排除相似角落",
                            )
                        try:
                            visual_status, visual_candidate, visual_match, diagnostics = (
                                _visual_localization_check(
                                    args,
                                    candidate,
                                    latest_scan,
                                    localizer,
                                    camera,
                                    visual_recognizer,
                                    gpu,
                                    visual_directory or Path("search_runs"),
                                    visual_attempts,
                                )
                            )
                        except Exception as exc:
                            visual_status, visual_candidate, visual_match = "error", candidate, None
                            diagnostics = {"error": str(exc)}
                        if event_callback is not None:
                            event_callback(
                                "visual_localization_checked",
                                status=visual_status,
                                **diagnostics,
                            )
                        print(
                            "视觉地点校验：status={}{}".format(
                                visual_status,
                                "" if visual_match is None else (
                                    " target={} score={:.3f} margin={:.3f} inliers={}".format(
                                        visual_match.target_id,
                                        visual_match.score,
                                        visual_match.margin,
                                        visual_match.inliers,
                                    )
                                ),
                            ),
                            flush=True,
                        )
                        if visual_status in {"confirmed", "corrected"}:
                            if status_callback is not None:
                                status_callback(
                                    "localized",
                                    "视觉与雷达一致，定位已确认"
                                    if visual_status == "confirmed"
                                    else "视觉识别出相似角落，已约束雷达修正位置",
                                )
                            return visual_candidate.pose, latest_scan
                        radar_is_ambiguous = (
                            math.isfinite(candidate.ambiguity_margin)
                            and candidate.ambiguity_margin < max(
                                0.08, args.localization_min_margin * 2.0
                            )
                        )
                        if visual_status == "conflict" or (
                            visual_status == "uncertain" and radar_is_ambiguous
                        ):
                            forced_visual_hint = (
                                None if visual_match is None else visual_match.pose
                            )
                            validated_moves = 0
                            anchor = None
                            anchor_scan = None
                            results.clear()
                            attempts = 0
                            print(
                                "拒绝当前雷达角落；将使用视觉参考位置重新采样定位。",
                                flush=True,
                            )
                            continue
                    return candidate.pose, latest_scan
        if attempts < max(args.localization_window, args.localization_min_accepted):
            continue
        after_scan, _ = _execute_active_localization_move(
            client, scan_iterator, localizer, speed_controller, latest_scan, args, deadline
        )
        pending_match = localizer.relative_motion(anchor_scan or latest_scan, after_scan)
        results.clear()
        attempts = 0
    client.stop()
    raise RuntimeError("主动定位超时或被停止")


def run(args: argparse.Namespace, *, client=None, lidar=None, camera=None, gpu=None) -> Path:
    decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
    if decision.get("status") != "executable":
        raise RuntimeError("只有已通过可执行性审查的任务才能进入闭环执行")
    if not args.simulate and not args.enable_motion:
        raise RuntimeError("实车必须显式提供 --enable-motion；仿真请使用 --simulate")
    known_map = KnownOccupancyMap.load(args.map, args.map_yaml, clearance_m=args.clearance)
    report = MissionLog(args.output, str(decision.get("instruction") or ""))
    dashboard = SearchDashboardPublisher(
        known_map, args.live_map_image, args.live_status, decision
    )
    report.data["hardware_motion_enabled"] = bool(args.enable_motion and not args.simulate)
    report.save()

    if args.simulate:
        from raspberry_pi.object_search_simulation import simulate_mission

        result = simulate_mission(known_map, decision, report)
        report.data.update(result)
        report.save()
        return report.directory

    owns_client, owns_lidar = client is None, lidar is None
    try:
        if client is None:
            client = _create_client_with_retry(
                lambda: Esp32Client(_serial_config(args)),
                attempts=args.link_connect_attempts,
                delay_s=args.link_connect_delay,
            )
        lidar = lidar or N10LidarDriver(_lidar_config(args))
    except Exception as exc:
        report.data["result"] = "failure"
        report.data["summary"] = str(exc)
        report.event("startup_failed", error=str(exc))
        dashboard.update(state="failure", reason=str(exc))
        report.save()
        if client is not None and owns_client:
            try:
                client.close()
            except Exception:
                pass
        raise
    if camera is None:
        direct_camera = CameraCapture(
            args.camera_backend, args.camera_device, args.camera_width, args.camera_height,
            args.camera_timeout, dual_usb_cameras=args.dual_usb_cameras,
        )
        camera = (
            ResilientCameraCapture(
                MjpegCameraCapture(
                    args.camera_stream_base_url, args.camera_timeout
                ),
                direct_camera,
                attempts=args.camera_capture_attempts,
                retry_delay_s=args.camera_retry_delay,
            )
            if args.camera_stream_base_url
            else direct_camera
        )
    visual_recognizer = None
    if args.visual_localization:
        visual_recognizer = VisualPlaceRecognizer(
            args.visual_localization_reference_root,
            semantic_map=args.visual_localization_semantic_map,
            minimum_score=args.visual_localization_min_score,
            minimum_margin=args.visual_localization_min_margin,
            minimum_inliers=args.visual_localization_min_inliers,
        )
        if visual_recognizer.available:
            print(
                "视觉定位参考库已加载：{} 个地点；雷达确认前会与最新标注照片复核。".format(
                    len(visual_recognizer.places)
                ),
                flush=True,
            )
        else:
            print(
                "视觉定位参考库为空；本次只能使用雷达主动定位。",
                flush=True,
            )
    gpu = gpu or GpuMissionClient(args.gpu_url, args.request_timeout)
    stopping = False

    def stop_requested(_signal=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    scan_iterator = iter(lidar.scans())
    speed_controller = AdaptiveSpeedController(
        args.max_turn_speed,
        args.max_drive_speed,
        AdaptiveSpeedConfig(
            minimum_turn_floor_mrad_s=args.turn_speed,
            initial_turn_floor_mrad_s=args.turn_speed,
            initial_turn_limit_mrad_s=args.max_turn_speed,
            turn_adjust_step_mrad_s=args.relocalization_turn_step,
            # Active localization is a startup manoeuvre, so it needs to
            # cross the proven chassis dead zone quickly. Normal waypoint
            # cruising still requests only speed/slow-speed below.
            linear_adjust_step_mm_s=max(50, args.drive_speed_step),
            maximum_linear_boost_mm_s=max(0, args.max_drive_speed - args.relocalization_speed),
            minimum_applied_linear_speed_mm_s=min(args.speed, args.slow_speed),
            linear_start_kick_mm_s=min(args.max_drive_speed, args.start_kick_speed),
            linear_progress_confirmation_frames=2,
            preserve_start_kick_on_poor_initial_match=True,
            minimum_turn_ttl_ms=300,
            initial_turn_ttl_ms=350,
            maximum_turn_ttl_ms=450,
            turn_ttl_adjust_step_ms=25,
        ),
    )
    started = time.monotonic()
    last_command = None
    last_sent = -math.inf
    pose = None
    try:
        client.heartbeat()
        client.stop()
        camera_test_path = report.directory / "00_camera_preflight.jpg"
        camera_test_info = camera.capture(camera_test_path)
        report.event(
            "camera_preflight_ok",
            image=camera_test_path.name,
            backend=camera_test_info.get("backend"),
            camera_count=camera_test_info.get("camera_count", 1),
            failover_used=bool(
                camera_test_info.get("camera_failover_used", False)
            ),
        )
        # Start the independent TTL refresher before active localization.
        # A lidar scan/global match can take much longer than the 650 ms motor
        # TTL; starting this only after localization made every localization
        # maneuver expire between scans, so the motors audibly pulsed without
        # producing enough body motion to disambiguate similar corners.
        start_link_maintenance = getattr(client, "start", None)
        if start_link_maintenance is not None:
            start_link_maintenance()
        localizer = ScanMapLocalizer(
            known_map, lidar_offset_x_m=args.lidar_offset_x, lidar_offset_y_m=args.lidar_offset_y
        )
        pose, latest_scan = _localize(
            args,
            client,
            scan_iterator,
            localizer,
            speed_controller,
            lambda: stopping,
            camera=camera,
            visual_recognizer=visual_recognizer,
            gpu=gpu,
            visual_directory=report.directory,
            event_callback=report.event,
            status_callback=lambda state, reason: dashboard.update(
                state=state, reason=reason
            ),
        )
        report.event("localized", pose=[pose.x_m, pose.y_m, pose.yaw_rad])
        route = build_search_route(known_map, (pose.x_m, pose.y_m), decision["route_request"])
        dashboard.update(
            pose=pose, route=route, state="route_planned",
            reason="定位完成，准备前往目标附近",
        )
        views = route["views"]
        remaining = list(range(len(views)))
        current_index = None
        detections: list[dict[str, Any]] = []
        latest_event = report.event("route_planned", views=len(views))
        perception_failures = 0
        replans = 0
        visited_view_points: list[tuple[float, float]] = []

        while not stopping:
            if time.monotonic() - started > args.max_runtime_min * 60:
                latest_event = report.event("mission_timeout")
            state = {
                "instruction": decision.get("instruction"),
                "target": decision.get("target"),
                "anchor": decision.get("anchor"),
                "current_view_index": current_index,
                "total_views": len(views),
                "remaining_view_indices": remaining,
                "latest_event": latest_event,
                "detections": detections,
                "attempts": len([e for e in report.data["events"] if e["type"] == "observation_complete"]),
            }
            first_observation_pending = not any(
                event["type"] == "observation_complete" for event in report.data["events"]
            )
            try:
                # Do not ask Qwen to search or choose another view while the
                # robot is still approaching the requested map area.  The
                # first view is selected deterministically from the A* route;
                # Qwen supervision starts only after a stopped observation.
                if (
                    first_observation_pending
                    and latest_event["type"] in {
                        "navigation_no_progress",
                        "temporary_obstacle",
                        "path_invalid",
                    }
                ):
                    directive = {
                        "action": "replan",
                        "reason": "首次观察点尚未到达，按最新障碍/无进展证据重新规划",
                        "terminal": False,
                        "source": "deterministic_recovery",
                    }
                elif first_observation_pending:
                    directive = {
                        "action": "navigate_next_view",
                        "next_view_index": remaining[0],
                        "reason": "先到达目标区域附近并停车，再启动视觉搜索",
                        "terminal": False,
                        "source": "deterministic_approach",
                    }
                else:
                    directive = gpu.next(state)
            except Exception as exc:
                # Network/Qwen cannot authorize movement. Safe deterministic
                # continuation only applies if an already planned view exists.
                if latest_event["type"] in {"mission_timeout", "localization_failed"} or not remaining:
                    directive = {"action": "report_failure", "reason": str(exc), "terminal": True}
                elif latest_event["type"] in {
                    "temporary_obstacle", "path_invalid", "navigation_no_progress"
                }:
                    directive = {"action": "replan", "reason": str(exc), "terminal": False}
                else:
                    directive = {"action": "navigate_next_view", "next_view_index": remaining[0], "reason": str(exc), "terminal": False}
            report.event("supervisor_directive", directive=directive)
            action = directive.get("action")
            if action in {"report_success", "report_failure"}:
                client.stop()
                report.data["result"] = "success" if action == "report_success" else "failure"
                report.data["summary"] = str(directive.get("reason") or "")
                dashboard.update(
                    pose=pose,
                    state=report.data["result"],
                    reason=report.data["summary"],
                    found=action == "report_success",
                )
                break
            if action == "replan":
                replans += 1
                if replans > args.max_replans:
                    client.stop()
                    report.data["result"] = "failure"
                    report.data["summary"] = "重规划次数超过安全上限"
                    report.event("replan_limit_reached", replans=replans)
                    break
                route = build_search_route(known_map, (pose.x_m, pose.y_m), decision["route_request"])
                views = [
                    view for view in route["views"]
                    if all(
                        math.hypot(
                            float(view["view_xy_m"][0]) - old[0],
                            float(view["view_xy_m"][1]) - old[1],
                        ) >= 0.25
                        for old in visited_view_points
                    )
                ]
                remaining = list(range(len(views)))
                latest_event = report.event("route_replanned", views=len(views))
                dashboard.update(
                    pose=pose, route=route, state="route_replanned",
                    reason="临时障碍物导致路线重规划",
                )
                continue

            requested_index = int(directive.get("next_view_index", remaining[0]))
            if requested_index not in remaining:
                requested_index = remaining[0]
            current_index = requested_index
            view = views[current_index]
            dashboard.update(
                pose=pose, view_index=current_index, state="navigating",
                reason=f"前往目标附近观察点 {current_index + 1}",
            )
            goal = tuple(view["view_xy_m"])
            localized_distance = math.hypot(goal[0] - pose.x_m, goal[1] - pose.y_m)
            if (
                args.real_test_max_distance > 0
                and not any(event["type"] == "observation_complete" for event in report.data["events"])
                and localized_distance > args.real_test_max_distance
            ):
                client.stop()
                report.data["result"] = "preflight_only"
                report.data["summary"] = (
                    f"首个观察点距离 {localized_distance:.2f}m，超过低速验收上限 "
                    f"{args.real_test_max_distance:.2f}m；未移动，需人工确认后扩大上限"
                )
                report.event(
                    "real_test_distance_guard",
                    distance_m=round(localized_distance, 3),
                    limit_m=args.real_test_max_distance,
                )
                break
            path = known_map.path((pose.x_m, pose.y_m), goal)
            if not path:
                remaining.remove(current_index)
                latest_event = report.event("path_invalid", view_index=current_index)
                continue
            replan = False
            failures = 0
            last_pose_update = time.monotonic()
            previous_scan = latest_scan
            waypoints = _densify_path(known_map, path, spacing_m=0.18)[1:] or [goal]
            for waypoint in waypoints:
                best_waypoint_distance = math.hypot(
                    float(waypoint[0]) - pose.x_m,
                    float(waypoint[1]) - pose.y_m,
                )
                progress_deadline = time.monotonic() + args.navigation_no_progress_seconds
                physical_translation = 0.0
                physical_rotation = 0.0
                navigation_scans = 0
                while not stopping:
                    latest_scan = next(scan_iterator)
                    now = time.monotonic()
                    duration_s = max(0.05, now - last_pose_update)
                    motion = localizer.relative_motion(previous_scan, latest_scan)
                    speed_controller.observe(_motion_update(motion, duration_s))
                    predicted_pose, motion_usable = _scan_motion_prediction(pose, motion)
                    if motion_usable:
                        physical_translation += math.hypot(
                            motion.transform.x_m, motion.transform.y_m
                        )
                        physical_rotation += abs(motion.transform.yaw_rad)
                    result = localizer.track(latest_scan, predicted_pose)
                    if not _tracking_usable_for_photo_mission(result, args):
                        recovery = localizer.localize_near_position(
                            latest_scan, predicted_pose, radius_m=0.35
                        )
                        result = recovery if _tracking_usable_for_photo_mission(recovery, args) else None
                    if result is None:
                        failures += 1
                        client.stop()
                        speed_controller.stopped()
                        if failures >= args.max_localization_failures:
                            latest_event = report.event("localization_failed", view_index=current_index)
                            stopping = True
                            break
                        previous_scan = latest_scan
                        last_pose_update = now
                        continue
                    failures = 0
                    pose = _fuse_tracking_pose(predicted_pose, result)
                    result = _result_with_pose(result, pose)
                    last_pose_update = now
                    previous_scan = latest_scan
                    distance_to_waypoint = math.hypot(
                        float(waypoint[0]) - pose.x_m,
                        float(waypoint[1]) - pose.y_m,
                    )
                    if distance_to_waypoint <= (
                        best_waypoint_distance - args.navigation_progress_distance
                    ):
                        best_waypoint_distance = distance_to_waypoint
                        progress_deadline = now + args.navigation_no_progress_seconds
                    navigation_scans += 1
                    linear, angular, reached = _strict_command_for_pose(args, pose, waypoint)
                    command = speed_controller.apply(MotionCommand(
                        linear, angular, "advancing" if linear else "turning", "search waypoint", waypoint
                    ))
                    dashboard.update(
                        pose=pose, view_index=current_index, state="navigating",
                        navigation={
                            "waypoint_xy_m": [round(float(waypoint[0]), 3), round(float(waypoint[1]), 3)],
                            "distance_m": round(distance_to_waypoint, 3),
                            "best_distance_m": round(best_waypoint_distance, 3),
                            "linear_mm_s": command.linear_mm_s,
                            "angular_mrad_s": command.angular_mrad_s,
                            "motion_match_ok": motion_usable,
                            "motion_rmse_m": round(float(motion.rmse_m), 4),
                            "map_rmse_m": round(float(result.rmse_m), 4),
                            "map_inlier_ratio": round(float(result.inlier_ratio), 3),
                        },
                        reason=f"前往目标附近观察点 {current_index + 1}",
                    )
                    if not _obstacle_safe(latest_scan, command.linear_mm_s, args.stop_distance):
                        client.stop()
                        speed_controller.stopped()
                        added = _remember_temporary_obstacle(
                            latest_scan, scan_iterator, pose, known_map, args
                        )
                        latest_event = report.event(
                            "temporary_obstacle", view_index=current_index, added_cells=added
                        )
                        replan = True
                        break
                    if now >= progress_deadline and not reached:
                        client.stop()
                        speed_controller.stopped()
                        latest_event = report.event(
                            "navigation_no_progress",
                            view_index=current_index,
                            waypoint=[float(waypoint[0]), float(waypoint[1])],
                            distance_m=round(distance_to_waypoint, 3),
                            physical_translation_m=round(physical_translation, 3),
                            physical_rotation_deg=round(math.degrees(physical_rotation), 1),
                            scans=navigation_scans,
                            last_command={
                                "linear_mm_s": command.linear_mm_s,
                                "angular_mrad_s": command.angular_mrad_s,
                            },
                            linear_adjustment_mm_s=(
                                speed_controller.linear_adjustment_mm_s
                            ),
                        )
                        dashboard.update(
                            pose=pose,
                            state="route_replanned",
                            reason="航点距离长时间没有减小，停车并重新规划",
                        )
                        replan = True
                        break
                    if reached:
                        client.stop()
                        speed_controller.stopped()
                        break
                    command_pair = (command.linear_mm_s, command.angular_mrad_s)
                    pure_turn = command.linear_mm_s == 0 and command.angular_mrad_s != 0
                    if pure_turn:
                        client.set_twist(
                            *command_pair,
                            ttl_ms=speed_controller.ttl_for(
                                command, args.command_ttl_ms
                            ),
                        )
                        allow_expiry = getattr(client, "allow_twist_to_expire", None)
                        if allow_expiry is not None:
                            allow_expiry()
                        # The next scan must explicitly authorize another turn;
                        # it cannot be treated as an identical TTL refresh.
                        last_command, last_sent = None, now
                    elif command_pair != last_command or now - last_sent >= args.command_min_interval_ms / 1000.0:
                        if command_pair == last_command and getattr(client, "refresh_twist", None):
                            client.refresh_twist(*command_pair, ttl_ms=args.command_ttl_ms)
                        else:
                            client.set_twist(*command_pair, ttl_ms=args.command_ttl_ms)
                        last_command, last_sent = command_pair, now
                if replan or stopping:
                    break
            if replan:
                continue
            if stopping:
                continue
            client.stop()
            time.sleep(args.settle_seconds)
            image_path = report.directory / f"view_{current_index + 1:02d}.jpg"
            camera_info = camera.capture(image_path)
            second_name = camera_info.get("camera_2_image")
            right_path = image_path.with_name(second_name) if second_name else image_path
            context = {
                "observation_id": f"{report.directory.name}_view_{current_index}",
                "task_instruction": decision.get("instruction"),
                "target": decision.get("target"),
                "search_phase": "arrived_near_target",
                "anchor": decision.get("anchor"),
                "robot_pose": [round(pose.x_m, 3), round(pose.y_m, 3), round(pose.yaw_rad, 3)],
            }
            try:
                perception = gpu.perceive(image_path, right_path, context)
                detections = confirm_detections(
                    list(perception.get("detections") or []),
                    decision.get("target") or {},
                    {"left": image_path, "right": right_path},
                )
                perception_failures = 0
            except Exception as exc:
                perception_failures += 1
                detections = []
                if perception_failures >= args.max_perception_failures:
                    latest_event = report.event("perception_unavailable", error=str(exc))
                    continue
                latest_event = report.event("perception_retryable", error=str(exc))
                continue
            remaining.remove(current_index)
            visited_view_points.append((float(goal[0]), float(goal[1])))
            dual_camera_confirmed = any(
                bool(item.get("dual_camera_confirmed")) for item in detections
            )
            complete_cameras = sorted({
                _normalise(item.get("camera_id"))
                for item in detections
                if item.get("single_camera_match") and item.get("complete_in_frame")
            })
            confirmation_reason = (
                "双摄像头均完整捕捉到同一目标，允许结束任务"
                if dual_camera_confirmed
                else "双目完整确认尚未满足，继续前往下一观察点"
            )
            print(
                "目标观察：{}；完整目标摄像头={}。".format(
                    confirmation_reason,
                    ",".join(complete_cameras) if complete_cameras else "无",
                ),
                flush=True,
            )
            dashboard.update(
                pose=pose,
                visited=[index for index in range(len(views)) if index not in remaining],
                state="searching",
                reason=confirmation_reason,
            )
            latest_event = report.event(
                "observation_complete",
                view_index=current_index,
                image=image_path.name,
                detections=detections,
                dual_camera_confirmed=dual_camera_confirmed,
                complete_cameras=complete_cameras,
                confirmation_reason=confirmation_reason,
            )
        if stopping and report.data["result"] == "running":
            report.data["result"] = "stopped"
            report.data["summary"] = "用户停止或安全门触发"
            dashboard.update(pose=pose, state="stopped", reason=report.data["summary"])
        return report.directory
    except Exception as exc:
        if report.data.get("result") == "running":
            report.data["result"] = "error"
            report.data["summary"] = str(exc)
            report.event("mission_error", error=str(exc))
            dashboard.update(
                pose=pose,
                state="failure",
                reason=str(exc),
            )
        raise
    finally:
        if report.data.get("result") == "running":
            dashboard.update(
                pose=pose,
                state="failure",
                reason="任务异常结束；详细原因见本次 mission.json 或终端日志",
            )
        try:
            client.stop()
        except Exception:
            pass
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
        report.data["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report.save()
        print(f"寻物任务结果目录：{report.directory}", flush=True)


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
