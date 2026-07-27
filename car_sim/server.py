"""car1.0 本地二维仿真服务。

只使用 Python 标准库：负责提供网页、接收外部控制指令并保存浏览器回传的仿真状态。
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
MAX_BODY_BYTES = 1_000_000
KNOWN_COMMANDS = {
    "set_twist",
    "stop",
    "cancel_all",
    "goto",
    "set_pose",
    "reset_map",
    "select_task",
    "set_arm_joints",
    "arm_stop",
}
ARM_LIMITS_US = (
    (500, 2500),
    (800, 1700),
    (1500, 2200),
    (800, 1500),
    (900, 2100),
    (1100, 1600),
)


class SimulationBus:
    """线程安全的内存指令总线。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lidar_ready = threading.Condition(self._lock)
        self._next_id = 1
        self._next_scan_sequence = 1
        self._commands: list[dict] = []
        self._latest_lidar: dict | None = None
        self._scene: dict = {}
        self._state = {
            "online": False,
            "mode": "editing",
            "pose": {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            "linear_mm_s": 0,
            "angular_mrad_s": 0,
            "task": "mapping",
            "joint_positions": [1500, 1700, 2000, 1100, 1500, 1200],
            "arm_moving": False,
            "grasped_item_id": None,
            "camera": {"width": 640, "height": 360, "detections": []},
            "last_command_id": 0,
            "updated_at": None,
        }

    def push(self, command: dict) -> dict:
        normalized = validate_command(command)
        with self._lock:
            item = {
                **normalized,
                "id": self._next_id,
                "received_at": time.time(),
            }
            self._next_id += 1
            self._commands.append(item)
            self._commands = self._commands[-500:]
            return dict(item)

    def commands_after(self, command_id: int) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._commands if item["id"] > command_id]

    def update_state(self, state: dict) -> dict:
        if not isinstance(state, dict):
            raise ValueError("state 必须是 JSON 对象")
        with self._lock:
            allowed = {
                "online",
                "mode",
                "pose",
                "linear_mm_s",
                "angular_mrad_s",
                "last_command_id",
                "mapping_progress",
                "collision",
                "goal",
                "commanded_linear_mm_s",
                "commanded_angular_mrad_s",
                "physics",
                "task",
                "joint_positions",
                "arm_moving",
                "grasped_item_id",
                "camera",
            }
            for key in allowed:
                if key in state:
                    self._state[key] = state[key]
            self._state["online"] = True
            self._state["updated_at"] = time.time()
            return dict(self._state)

    def state(self) -> dict:
        with self._lock:
            result = dict(self._state)
        updated_at = result.get("updated_at")
        result["online"] = bool(updated_at and time.time() - updated_at < 2.0)
        return result

    def push_lidar(self, scan: dict) -> dict:
        normalized = validate_lidar_scan(scan)
        with self._lidar_ready:
            item = {
                **normalized,
                "sequence": self._next_scan_sequence,
                "received_at": time.time(),
            }
            self._next_scan_sequence += 1
            self._latest_lidar = item
            self._lidar_ready.notify_all()
            return dict(item)

    def lidar_after(self, sequence: int, timeout_s: float) -> dict | None:
        deadline = time.monotonic() + timeout_s
        with self._lidar_ready:
            while (
                self._latest_lidar is None
                or self._latest_lidar["sequence"] <= sequence
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lidar_ready.wait(remaining)
            return dict(self._latest_lidar)

    def update_scene(self, scene: dict) -> dict:
        normalized = validate_scene(scene)
        with self._lock:
            self._scene = normalized
            return dict(self._scene)

    def scene(self) -> dict:
        with self._lock:
            return dict(self._scene)


BUS = SimulationBus()


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数字")
    return value


def validate_command(command: object) -> dict:
    if not isinstance(command, dict):
        raise ValueError("指令必须是 JSON 对象")
    command_type = command.get("type")
    if command_type not in KNOWN_COMMANDS:
        raise ValueError(f"不支持的指令类型: {command_type!r}")

    result = {"type": command_type}
    if command_type == "set_twist":
        linear = finite_number(command.get("linear_mm_s"), "linear_mm_s")
        angular = finite_number(command.get("angular_mrad_s"), "angular_mrad_s")
        ttl = finite_number(command.get("ttl_ms", 600), "ttl_ms")
        if not -550 <= linear <= 550:
            raise ValueError("linear_mm_s 超出项目协议范围 -550..550")
        if not -3500 <= angular <= 3500:
            raise ValueError("angular_mrad_s 超出项目协议范围 -3500..3500")
        if not 50 <= ttl <= 2500:
            raise ValueError("ttl_ms 超出 50..2500")
        result.update(
            linear_mm_s=round(linear),
            angular_mrad_s=round(angular),
            ttl_ms=round(ttl),
        )
    elif command_type in {"goto", "set_pose"}:
        result["x_m"] = finite_number(command.get("x_m"), "x_m")
        result["y_m"] = finite_number(command.get("y_m"), "y_m")
        if command_type == "set_pose":
            result["yaw_rad"] = finite_number(command.get("yaw_rad", 0), "yaw_rad")
    elif command_type == "select_task":
        task = command.get("task")
        if task not in {"mapping", "pickup"}:
            raise ValueError("task 必须是 mapping 或 pickup")
        result["task"] = task
    elif command_type == "set_arm_joints":
        joints = command.get("joints")
        duration = finite_number(command.get("duration_ms", 800), "duration_ms")
        if not isinstance(joints, list) or not 1 <= len(joints) <= 6:
            raise ValueError("joints 必须包含 1..6 个 [关节编号, 脉宽] 项")
        if not 20 <= duration <= 10000:
            raise ValueError("duration_ms 超出 20..10000")
        normalized_joints = []
        seen = set()
        for index, joint in enumerate(joints):
            if not isinstance(joint, (list, tuple)) or len(joint) != 2:
                raise ValueError(f"joints[{index}] 必须是 [关节编号, 脉宽]")
            joint_id = int(finite_number(joint[0], f"joints[{index}].id"))
            pulse_us = round(finite_number(joint[1], f"joints[{index}].pulse_us"))
            if joint_id in seen or not 0 <= joint_id < 6:
                raise ValueError("关节编号必须唯一且位于 0..5")
            minimum, maximum = ARM_LIMITS_US[joint_id]
            if not minimum <= pulse_us <= maximum:
                raise ValueError(
                    f"关节 {joint_id} 脉宽超出项目范围 {minimum}..{maximum}"
                )
            seen.add(joint_id)
            normalized_joints.append([joint_id, pulse_us])
        result["joints"] = normalized_joints
        result["duration_ms"] = round(duration)

    source = command.get("source")
    if source is not None:
        result["source"] = str(source)[:80]
    return result


def validate_lidar_scan(scan: object) -> dict:
    if not isinstance(scan, dict):
        raise ValueError("雷达扫描必须是 JSON 对象")
    angles = scan.get("angles_rad")
    distances = scan.get("distances_m")
    if not isinstance(angles, list) or not isinstance(distances, list):
        raise ValueError("angles_rad 和 distances_m 必须是数组")
    if len(angles) != len(distances) or not 1 <= len(angles) <= 1000:
        raise ValueError("雷达角度与距离数组长度不一致或超限")
    normalized_angles = [
        finite_number(value, f"angles_rad[{index}]")
        for index, value in enumerate(angles)
    ]
    normalized_distances = [
        finite_number(value, f"distances_m[{index}]")
        for index, value in enumerate(distances)
    ]
    if any(distance < 0 or distance > 30 for distance in normalized_distances):
        raise ValueError("雷达距离超出 0..30 m")
    pose = scan.get("ground_truth_pose", {})
    if not isinstance(pose, dict):
        raise ValueError("ground_truth_pose 必须是对象")
    return {
        "timestamp_s": finite_number(scan.get("timestamp_s"), "timestamp_s"),
        "angles_rad": normalized_angles,
        "distances_m": normalized_distances,
        "ground_truth_pose": {
            "x_m": finite_number(pose.get("x_m", 0), "ground_truth_pose.x_m"),
            "y_m": finite_number(pose.get("y_m", 0), "ground_truth_pose.y_m"),
            "yaw_rad": finite_number(
                pose.get("yaw_rad", 0), "ground_truth_pose.yaw_rad"
            ),
        },
    }


def validate_scene(scene: object) -> dict:
    if not isinstance(scene, dict):
        raise ValueError("scene 必须是 JSON 对象")
    room = scene.get("room")
    obstacles = scene.get("obstacles")
    if not isinstance(room, dict) or not isinstance(obstacles, list):
        raise ValueError("scene 缺少 room 或 obstacles")
    width = finite_number(room.get("width"), "room.width")
    height = finite_number(room.get("height"), "room.height")
    if not 1 <= width <= 100 or not 1 <= height <= 100:
        raise ValueError("房间尺寸超出范围")
    if len(obstacles) > 1000:
        raise ValueError("障碍物数量超限")
    # 场景仅用于评测与诊断；保留有限 JSON 字段，不执行其中的任何内容。
    return json.loads(json.dumps(scene, ensure_ascii=False))


class SimulationHandler(BaseHTTPRequestHandler):
    server_version = "CarSim/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[car_sim] {self.address_string()} - {fmt % args}")

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> object:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("请求体为空或过大")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json({"ok": True, "service": "car_sim"})
            return
        if parsed.path == "/api/state":
            self._json(BUS.state())
            return
        if parsed.path == "/api/scene":
            self._json({"ok": True, "scene": BUS.scene()})
            return
        if parsed.path == "/api/lidar":
            query = parse_qs(parsed.query)
            try:
                after = max(0, int(query.get("after", ["0"])[0]))
                timeout_ms = min(
                    5000, max(0, int(query.get("timeout_ms", ["1000"])[0]))
                )
            except ValueError:
                self._json(
                    {"ok": False, "error": "after/timeout_ms 必须是整数"},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            scan = BUS.lidar_after(after, timeout_ms / 1000.0)
            self._json({"ok": True, "scan": scan})
            return
        if parsed.path == "/api/commands":
            query = parse_qs(parsed.query)
            try:
                after = max(0, int(query.get("after", ["0"])[0]))
            except ValueError:
                self._json({"ok": False, "error": "after 必须是整数"}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"ok": True, "commands": BUS.commands_after(after)})
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/commands":
                item = BUS.push(payload)
                self._json({"ok": True, "command": item}, HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/state":
                state = BUS.update_state(payload)
                self._json({"ok": True, "state": state})
                return
            if parsed.path == "/api/lidar":
                scan = BUS.push_lidar(payload)
                self._json(
                    {"ok": True, "sequence": scan["sequence"]},
                    HTTPStatus.ACCEPTED,
                )
                return
            if parsed.path == "/api/scene":
                scene = BUS.update_scene(payload)
                self._json({"ok": True, "scene": scene})
                return
            self._json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        candidate = (WEB_ROOT / relative).resolve()
        try:
            candidate.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if mime.startswith("text/") or mime in {"application/javascript", "application/json"}:
            mime += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 car1.0 二维仿真样品")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SimulationHandler)
    print(f"car_sim 已启动: http://{args.host}:{args.port}")
    print("外部指令接口: POST /api/commands")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 car_sim…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
