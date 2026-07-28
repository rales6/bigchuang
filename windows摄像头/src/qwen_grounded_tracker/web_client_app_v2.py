from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request as UrlRequest, urlopen

import cv2
import numpy as np
from websocket import WebSocketConnectionClosedException, create_connection

from qwen_grounded_tracker.camera.opencv_camera import OpenCVCamera


STATIC_DIR = Path(__file__).resolve().parent / "web_static"


def _source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _now_label() -> str:
    return time.strftime("%H:%M:%S")


def _now_full() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _health_url_from_ws_url(server_url: str) -> str | None:
    parsed = urlparse(server_url)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.hostname}{port}/health"
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        return None
    scheme = "https" if parsed.scheme == "wss" else "http"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{scheme}://{parsed.hostname}{port}/health"


def _http_base_url_from_server_url(server_url: str) -> str:
    parsed = urlparse(server_url)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.hostname}{port}"
    if parsed.scheme in {"ws", "wss"} and parsed.hostname:
        scheme = "https" if parsed.scheme == "wss" else "http"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{scheme}://{parsed.hostname}{port}"
    raise ValueError(f"Unsupported 3090 server URL: {server_url}")


def _server_port_from_ws_url(server_url: str) -> int | None:
    parsed = urlparse(server_url)
    return parsed.port


def _is_local_ws_url(server_url: str) -> bool:
    parsed = urlparse(server_url)
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _health_ok(health_url: str | None, timeout_seconds: float = 1.0) -> bool:
    if not health_url:
        return False
    try:
        with urlopen(health_url, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def _maybe_start_tunnel(args: argparse.Namespace) -> subprocess.Popen[Any] | None:
    if not args.auto_tunnel:
        return None
    if not _is_local_ws_url(args.server):
        print("[Tunnel] server is not localhost; auto tunnel skipped")
        return None

    health_url = _health_url_from_ws_url(args.server)
    if _health_ok(health_url):
        print(f"[Tunnel] already reachable: {health_url}")
        return None

    local_port = _server_port_from_ws_url(args.server) or args.tunnel_local_port
    script_path = Path(args.tunnel_script)
    if not script_path.is_absolute():
        script_path = Path.cwd() / script_path
    if not script_path.exists():
        print(f"[Tunnel] script not found; auto tunnel skipped: {script_path}")
        return None

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-LocalPort",
        str(local_port),
        "-RemotePort",
        str(args.tunnel_remote_port),
    ]
    if args.tunnel_ssh_host:
        command.extend(["-SshHost", str(args.tunnel_ssh_host)])
    if args.tunnel_ssh_user:
        command.extend(["-SshUser", str(args.tunnel_ssh_user)])
    if args.tunnel_ssh_port:
        command.extend(["-SshPort", str(args.tunnel_ssh_port)])
    if args.tunnel_identity_file:
        command.extend(["-IdentityFile", str(args.tunnel_identity_file)])

    print(f"[Tunnel] starting local tunnel on port {local_port}")
    process = subprocess.Popen(command, cwd=str(Path.cwd()))

    for _ in range(20):
        time.sleep(0.4)
        if _health_ok(health_url, timeout_seconds=0.5):
            print(f"[Tunnel] ready: {health_url}")
            return process
        if process.poll() is not None:
            print(f"[Tunnel] process exited early with code {process.returncode}")
            return process

    print("[Tunnel] started, but health check is not ready yet")
    return process


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conversations: list[dict[str, Any]] = []
        self.active_id = ""
        self.load()

    def load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}

        conversations = data.get("conversations", [])
        if isinstance(conversations, list):
            self.conversations = [
                item for item in conversations if isinstance(item, dict) and item.get("id")
            ]
        if not self.conversations:
            self.conversations = [self.make_conversation()]

        active_id = str(data.get("active_conversation_id") or self.conversations[0]["id"])
        if not any(item["id"] == active_id for item in self.conversations):
            active_id = self.conversations[0]["id"]
        self.active_id = active_id
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_conversation_id": self.active_id,
            "conversations": self.conversations,
        }
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def make_conversation(self, title: str = "New chat") -> dict[str, Any]:
        now = _now_full()
        return {
            "id": uuid.uuid4().hex,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

    def active(self) -> dict[str, Any]:
        for item in self.conversations:
            if item["id"] == self.active_id:
                return item
        self.active_id = self.conversations[0]["id"]
        return self.conversations[0]

    def messages(self) -> list[dict[str, str]]:
        return self.active().setdefault("messages", [])

    def append(self, role: str, content: str, kind: str = "message") -> None:
        conversation = self.active()
        messages = conversation.setdefault("messages", [])
        messages.append({"role": role, "content": content, "kind": kind, "time": _now_label()})
        conversation["messages"] = messages[-200:]
        conversation["updated_at"] = _now_full()
        if role == "user" and conversation.get("title") == "New chat":
            conversation["title"] = content[:28] + ("..." if len(content) > 28 else "")
        self.save()

    def new(self) -> str:
        conversation = self.make_conversation()
        self.conversations.insert(0, conversation)
        self.active_id = conversation["id"]
        self.save()
        return self.active_id

    def select(self, conversation_id: str) -> bool:
        if any(item["id"] == conversation_id for item in self.conversations):
            self.active_id = conversation_id
            self.save()
            return True
        return False

    def rename(self, conversation_id: str, title: str) -> bool:
        title = title.strip()
        if not title:
            return False
        for item in self.conversations:
            if item["id"] == conversation_id:
                item["title"] = title[:80]
                item["updated_at"] = _now_full()
                self.save()
                return True
        return False

    def delete(self, conversation_id: str) -> bool:
        original_count = len(self.conversations)
        self.conversations = [item for item in self.conversations if item["id"] != conversation_id]
        if len(self.conversations) == original_count:
            return False
        if not self.conversations:
            self.conversations = [self.make_conversation()]
        if not any(item["id"] == self.active_id for item in self.conversations):
            self.active_id = self.conversations[0]["id"]
        self.save()
        return True

    def summaries(self) -> list[dict[str, str]]:
        return [
            {
                "id": str(item["id"]),
                "title": str(item.get("title") or "New chat"),
                "updated_at": str(item.get("updated_at") or ""),
            }
            for item in self.conversations
        ]


class WindowsWebClientRuntime:
    """Browser console for local-camera or Raspberry-Pi-camera 3090 inference."""

    def __init__(
        self,
        server_url: str,
        camera_mode: str,
        camera_source: int | str,
        backend: str,
        width: int,
        height: int,
        fps: int,
        side_by_side: str,
        jpeg_quality: int,
        overlay_quality: int,
        overlay_every_n: int,
        remote_fps: float,
        remote_width: int,
        remote_height: int,
        video_url: str,
        initial_instruction: str,
    ) -> None:
        self.server_url = server_url
        self.http_base_url = _http_base_url_from_server_url(server_url)
        self.camera_mode = camera_mode
        self.video_url = video_url.strip()
        if not self.video_url and self.camera_mode == "remote":
            self.video_url = f"{self.http_base_url}/stream.mjpg"
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self.overlay_quality = max(30, min(95, int(overlay_quality)))
        self.overlay_every_n = max(1, int(overlay_every_n))
        self.remote_frame_interval = 1.0 / max(1.0, float(remote_fps))
        self.remote_width = max(0, int(remote_width))
        self.remote_height = max(0, int(remote_height))

        self.camera: OpenCVCamera | None = None
        if self.camera_mode == "local":
            self.camera = OpenCVCamera(
                source=camera_source,
                width=width,
                height=height,
                fps=fps,
                backend=backend,
                side_by_side=side_by_side,
            )

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.camera_thread: threading.Thread | None = None
        self.remote_thread: threading.Thread | None = None
        self.ws_receiver_thread: threading.Thread | None = None
        self.ws: Any | None = None
        self.ws_send_lock = threading.Lock()

        self.store = ConversationStore(Path.cwd() / "outputs" / "chat_history.json")
        self.runtime_status = "stopped"
        self.remote_status = "not connected"
        self.grounding_status = "waiting for instruction"
        self.current_instruction = initial_instruction.strip()
        self.pending_start_instruction = self.current_instruction or None
        self.control_queue: list[dict[str, Any]] = []
        self.manual_roi_queue: list[dict[str, Any]] = []

        self.latest_frame: np.ndarray | None = None
        self.latest_raw_jpeg: bytes | None = None
        self.latest_overlay_jpeg: bytes | None = None
        self.latest_overlay_b64 = ""
        self.latest_overlay_at = 0.0
        self.latest_image_version = 0
        self.latest_frame_size = {"width": 0, "height": 0}
        self.latest_response_at = time.monotonic()
        self.local_fps = 0.0
        self.remote_fps = 0.0
        self._local_frames = 0
        self._remote_frames = 0
        self._last_fps_at = time.monotonic()
        self.latest_response: dict[str, Any] = {
            "track": {
                "status": "waiting for remote server"
                if self.camera_mode == "local"
                else "waiting for Raspberry Pi camera",
                "bbox": None,
                "visible": False,
            },
            "guidance": {"direction": "STOP", "linear": 0.0, "angular": 0.0},
            "safety": {"blocked": True, "reasons": []},
            "boundary_mode": "-",
            "emergency_stop": False,
            "obstacles": {"status": "not run"},
            "lidar": {"status": "not run"},
            "robot_task": {
                "active": False,
                "task_type": "none",
                "phase": "idle",
                "phase_label": "No robot task",
                "completed": False,
            },
            "robot_command": {
                "mode": "simulated",
                "subsystem": "vision",
                "action": "idle",
                "reason": "No active task",
            },
            "target_queue": [],
        }

        if self.current_instruction:
            self.store.append("user", self.current_instruction, "initial")

    def start(self) -> None:
        if self.remote_thread and self.remote_thread.is_alive():
            return
        if self.camera_mode == "local":
            self.camera_thread = threading.Thread(
                target=self._camera_loop,
                name="camera-loop",
                daemon=True,
            )
            self.remote_thread = threading.Thread(
                target=self._remote_loop,
                name="remote-loop",
                daemon=True,
            )
            self.camera_thread.start()
        else:
            self.runtime_status = "remote camera mode"
            self.grounding_status = "waiting for Raspberry Pi camera"
            self.remote_thread = threading.Thread(
                target=self._remote_state_loop,
                name="remote-state-loop",
                daemon=True,
            )
        self.remote_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._close_ws()
        for thread in (self.camera_thread, self.remote_thread):
            if thread:
                thread.join(timeout=3.0)

    def _append_chat(self, role: str, content: str, kind: str = "message") -> None:
        self.store.append(role, content, kind)

    def submit_instruction(self, instruction: str) -> dict[str, Any]:
        instruction = instruction.strip()
        if not instruction:
            return {"ok": False, "message": "empty instruction"}
        with self.lock:
            self.current_instruction = instruction
            self.pending_start_instruction = instruction
            self._append_chat("user", instruction)
            if self.camera_mode == "remote":
                self.pending_start_instruction = None
                self._append_chat("assistant", "Sending this instruction to the 3090 shared camera session.", "status")
                return self._remote_post_control({"type": "set_instruction", "instruction": instruction})
            if self.ws is None:
                self._append_chat("assistant", "Queued. Connecting to 3090 with this instruction.", "status")
            else:
                self.control_queue.append({"type": "set_instruction", "instruction": instruction})
                self._append_chat("assistant", "Queued. Sending the new instruction to 3090.", "status")
        return {"ok": True}

    def new_chat(self) -> dict[str, Any]:
        with self.lock:
            conversation_id = self.store.new()
            self.current_instruction = ""
            self.pending_start_instruction = None
            if self.ws is not None:
                self.control_queue.append({"type": "reset"})
        return {"ok": True, "conversation_id": conversation_id}

    def select_chat(self, conversation_id: str) -> dict[str, Any]:
        with self.lock:
            ok = self.store.select(conversation_id)
        return {"ok": ok, "message": "" if ok else "conversation not found"}

    def rename_chat(self, conversation_id: str, title: str) -> dict[str, Any]:
        with self.lock:
            ok = self.store.rename(conversation_id, title)
        return {"ok": ok, "message": "" if ok else "rename failed"}

    def delete_chat(self, conversation_id: str) -> dict[str, Any]:
        with self.lock:
            was_active = self.store.active_id == conversation_id
            ok = self.store.delete(conversation_id)
            if ok and was_active:
                self.current_instruction = ""
                self.pending_start_instruction = None
                if self.ws is not None:
                    self.control_queue.append({"type": "reset"})
        return {"ok": ok, "message": "" if ok else "conversation not found"}

    def apply_manual_roi(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            width = self.latest_frame_size["width"]
            height = self.latest_frame_size["height"]
            if width <= 0 or height <= 0:
                return {"ok": False, "message": "camera frame is not ready"}
            try:
                x1 = float(payload["x1"]) * width
                y1 = float(payload["y1"]) * height
                x2 = float(payload["x2"]) * width
                y2 = float(payload["y2"]) * height
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "message": "invalid bbox"}
            x = min(x1, x2)
            y = min(y1, y2)
            box_width = abs(x2 - x1)
            box_height = abs(y2 - y1)
            if box_width <= 3 or box_height <= 3:
                return {"ok": False, "message": "bbox is too small"}
            if self.camera_mode == "remote":
                # 远程摄像头模式下，ROI 直接发给 3090 的共享会话，不经过本机 WebSocket。
                return self._remote_post_control(
                    {"type": "manual_roi", "bbox_xywh": [x, y, box_width, box_height]}
                )
            if self.ws is None and not self.pending_start_instruction:
                self.current_instruction = self.current_instruction or "manual ROI"
                self.pending_start_instruction = self.current_instruction
            self.manual_roi_queue.append(
                {"type": "manual_roi", "bbox_xywh": [x, y, box_width, box_height]}
            )
            self._append_chat("assistant", "Manual ROI queued for the 3090.", "status")
        return {"ok": True}

    def control(self, action: str) -> dict[str, Any]:
        with self.lock:
            if self.camera_mode == "remote" and action != "capture":
                if action == "reset":
                    self._append_chat("assistant", "Requested target reset on the 3090.", "status")
                    return self._remote_post_control({"type": "reset"})
                if action == "reground":
                    return self._remote_post_control({"type": "reground"})
                if action == "toggle_boundary":
                    return self._remote_post_control({"type": "toggle_boundary"})
                if action == "toggle_estop":
                    current = bool(self.latest_response.get("emergency_stop", False))
                    return self._remote_post_control(
                        {"type": "emergency_stop", "enabled": not current}
                    )
            if action == "reset":
                self.control_queue.append({"type": "reset"})
                self._append_chat("assistant", "Requested target reset on the 3090.", "status")
                return {"ok": True}
            if action == "reground":
                self.control_queue.append({"type": "reground"})
                return {"ok": True}
            if action == "toggle_boundary":
                self.control_queue.append({"type": "toggle_boundary"})
                return {"ok": True}
            if action == "toggle_estop":
                current = bool(self.latest_response.get("emergency_stop", False))
                self.control_queue.append({"type": "emergency_stop", "enabled": not current})
                return {"ok": True}
            if action == "capture":
                if self.camera_mode == "remote":
                    jpeg = None if self.latest_overlay_jpeg is None else bytes(self.latest_overlay_jpeg)
                    if jpeg is None:
                        return {"ok": False, "message": "no remote frame"}
                    output_dir = Path.cwd() / "outputs"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    filename = output_dir / f"web_remote_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    filename.write_bytes(jpeg)
                    return {"ok": True, "path": str(filename)}
                frame = None if self.latest_frame is None else self.latest_frame.copy()
            else:
                return {"ok": False, "message": f"unknown action: {action}"}
        if frame is None:
            return {"ok": False, "message": "no frame"}
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"web_client_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(filename), frame)
        return {"ok": True, "path": str(filename)}

    def status(self) -> dict[str, Any]:
        with self.lock:
            response = dict(self.latest_response)
            return {
                "runtime_status": self.runtime_status,
                "remote_status": self.remote_status,
                "grounding_status": self.grounding_status,
                "instruction": self.current_instruction,
                "server_url": self.server_url,
                "video_url": self.video_url,
                "camera_mode": self.camera_mode,
                "chat_store_path": str(self.store.path),
                "active_conversation_id": self.store.active_id,
                "conversations": self.store.summaries(),
                "frame": self.latest_frame_size,
                "local_fps": round(self.local_fps, 1),
                "remote_fps": round(self.remote_fps, 1),
                "track": response.get("track", {}),
                "predicted_track": response.get("predicted_track", response.get("track", {})),
                "paired_track": response.get("paired_track", {}),
                "grounded_tracks": response.get("grounded_tracks", []),
                "measured_track": response.get("measured_track", {}),
                "prediction": response.get("prediction", {}),
                "stereo": response.get("stereo", {}),
                "guidance": response.get("guidance", {}),
                "safety": response.get("safety", {}),
                "boundary_mode": response.get("boundary_mode", "-"),
                "emergency_stop": response.get("emergency_stop", False),
                "obstacles": response.get("obstacles", {}).get("status", "not run"),
                "lidar": response.get("lidar", {}).get("status", "not run"),
                "robot_task": response.get("robot_task", {}),
                "robot_command": response.get("robot_command", {}),
                "target_queue": response.get("target_queue", []),
                "chat": list(self.store.messages()),
            }

    def latest_image(self) -> bytes | None:
        with self.lock:
            if self.camera_mode == "remote":
                base = None if self.latest_overlay_jpeg is None else bytes(self.latest_overlay_jpeg)
            elif self.latest_overlay_jpeg is not None and time.monotonic() - self.latest_overlay_at < 0.6:
                base = bytes(self.latest_overlay_jpeg)
            else:
                base = None if self.latest_raw_jpeg is None else bytes(self.latest_raw_jpeg)
            response = dict(self.latest_response)
            response_at = self.latest_response_at
            quality = int(self.jpeg_quality)
        if base is None:
            return None
        if self.camera_mode == "remote":
            composed = self._compose_tracking_overlay(base, response, response_at, quality)
            return composed if composed is not None else base
        return base

    def latest_image_snapshot(self) -> tuple[int, bytes | None]:
        with self.lock:
            version = self.latest_image_version
        return version, self.latest_image()

    def stream_overlay_active(self) -> bool:
        with self.lock:
            track = self.latest_response.get("predicted_track") or self.latest_response.get("track") or {}
            return bool(isinstance(track, dict) and track.get("visible") and track.get("bbox"))

    @staticmethod
    def _remap_contour(
        contour: list[dict[str, Any]],
        old_box: dict[str, Any],
        new_box: dict[str, float],
    ) -> list[tuple[int, int]]:
        old_w = max(1.0, float(old_box.get("width") or (float(old_box["x2"]) - float(old_box["x1"]))))
        old_h = max(1.0, float(old_box.get("height") or (float(old_box["y2"]) - float(old_box["y1"]))))
        old_x = float(old_box["x1"])
        old_y = float(old_box["y1"])
        return [
            (
                int(round(new_box["x1"] + ((float(point["x"]) - old_x) / old_w) * new_box["width"])),
                int(round(new_box["y1"] + ((float(point["y"]) - old_y) / old_h) * new_box["height"])),
            )
            for point in contour
        ]

    @staticmethod
    def _predict_box_for_now(
        box: dict[str, Any],
        prediction: dict[str, Any],
        response_at: float,
    ) -> dict[str, float]:
        local_age = max(0.0, time.monotonic() - response_at)
        total_age = float(prediction.get("age_seconds") or 0.0) + local_age
        can_advance = bool(prediction.get("enabled", True)) and total_age <= 0.45
        dt = local_age if can_advance else 0.0
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
        width = max(3.0, float(box.get("width") or (x2 - x1)) + float(prediction.get("vw") or 0.0) * dt)
        height = max(3.0, float(box.get("height") or (y2 - y1)) + float(prediction.get("vh") or 0.0) * dt)
        cx = (x1 + x2) / 2.0 + float(prediction.get("vx") or 0.0) * dt
        cy = (y1 + y2) / 2.0 + float(prediction.get("vy") or 0.0) * dt
        return {
            "x1": cx - width / 2.0,
            "y1": cy - height / 2.0,
            "x2": cx + width / 2.0,
            "y2": cy + height / 2.0,
            "width": width,
            "height": height,
        }

    def _compose_tracking_overlay(
        self,
        jpeg: bytes,
        response: dict[str, Any],
        response_at: float,
        quality: int,
    ) -> bytes | None:
        track = response.get("predicted_track") or response.get("track") or {}
        if not isinstance(track, dict) or not track.get("visible") or not track.get("bbox"):
            return jpeg
        box = track.get("bbox")
        if not isinstance(box, dict):
            return jpeg
        array = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        height, width = frame.shape[:2]
        prediction = response.get("prediction") if isinstance(response.get("prediction"), dict) else {}
        try:
            predicted_box = self._predict_box_for_now(box, prediction, response_at)
            x1 = int(round(max(0.0, min(predicted_box["x1"], width - 1))))
            y1 = int(round(max(0.0, min(predicted_box["y1"], height - 1))))
            x2 = int(round(max(x1 + 1.0, min(predicted_box["x2"], width))))
            y2 = int(round(max(y1 + 1.0, min(predicted_box["y2"], height))))
        except (KeyError, TypeError, ValueError):
            return jpeg

        contour_points: list[tuple[int, int]] = []
        contour = track.get("contour")
        if isinstance(contour, list) and len(contour) >= 3:
            try:
                contour_points = self._remap_contour(contour, box, predicted_box)
            except (KeyError, TypeError, ValueError):
                contour_points = []

        if contour_points:
            pts = np.asarray(contour_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.drawContours(frame, [pts], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), max(45, min(95, quality))],
        )
        return encoded.tobytes() if ok else None

    def _close_ws(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _remote_json_request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.http_base_url}{path}"
        if payload is None:
            with urlopen(url, timeout=2.0) as response:
                return json.loads(response.read().decode("utf-8"))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = UrlRequest(
            url,
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urlopen(request, timeout=4.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _remote_post_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._remote_json_request("/control", payload)
        except Exception as exc:
            self.remote_status = f"control failed: {exc}"
            self.grounding_status = "waiting for 3090 control endpoint"
            self._append_chat("assistant", f"Failed to send control to 3090: {exc}", "error")
            return {"ok": False, "message": str(exc)}
        if response.get("type") == "error":
            message = str(response.get("message", "3090 control error"))
            self._append_chat("assistant", message, "error")
            return {"ok": False, "message": message}
        message = str(response.get("message", "control accepted"))
        self._append_chat("assistant", message, "status")
        return {"ok": True, "response": response}

    def _remote_state_loop(self) -> None:
        was_connected = False
        while not self.stop_event.is_set():
            try:
                state = self._remote_json_request("/state")
                camera = state.get("camera", {}) if isinstance(state.get("camera"), dict) else {}
                connected = bool(camera.get("connected", False))
                display_frame = None
                if not self.video_url:
                    display_frame = state.get("display_jpeg_b64") or state.get("overlay_jpeg_b64")
                with self.lock:
                    self.latest_response = state
                    self.latest_response_at = time.monotonic()
                    self.remote_status = (
                        "Raspberry Pi camera connected"
                        if connected
                        else "waiting for Raspberry Pi camera"
                    )
                    self.grounding_status = str(
                        state.get("grounding_status", self.grounding_status)
                    )
                    frame = camera.get("frame", {})
                    if isinstance(frame, dict):
                        self.latest_frame_size = {
                            "width": int(frame.get("width") or 0),
                            "height": int(frame.get("height") or 0),
                        }
                    self.local_fps = 0.0
                    self.remote_fps = float(camera.get("fps") or 0.0)
                    if display_frame and str(display_frame) != self.latest_overlay_b64:
                        self.latest_overlay_b64 = str(display_frame)
                        self.latest_overlay_jpeg = base64.b64decode(self.latest_overlay_b64)
                        self.latest_overlay_at = time.monotonic()
                        self.latest_image_version += 1
                    if connected and not was_connected:
                        self._append_chat("assistant", "Raspberry Pi camera stream is connected.", "status")
                    if not connected and was_connected:
                        self._append_chat("assistant", "Raspberry Pi camera stream is disconnected.", "error")
                was_connected = connected
            except Exception as exc:
                with self.lock:
                    self.runtime_status = "remote camera mode"
                    self.remote_status = f"state polling failed: {exc}"
                    self.grounding_status = "waiting for 3090 state endpoint"
                was_connected = False
            time.sleep(max(0.05, self.remote_frame_interval))

    def _encode_jpeg(self, frame: np.ndarray) -> bytes | None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        return encoded.tobytes() if ok else None

    def _resize_for_remote(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        target_width = self.remote_width
        target_height = self.remote_height
        if target_width <= 0 and target_height <= 0:
            return frame
        if target_width <= 0:
            scale = target_height / max(height, 1)
            target_width = int(round(width * scale))
        elif target_height <= 0:
            scale = target_width / max(width, 1)
            target_height = int(round(height * scale))
        target_width = max(16, target_width)
        target_height = max(16, target_height)
        if target_width == width and target_height == height:
            return frame
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _scale_actions_for_remote(
        self,
        actions: list[dict[str, Any]],
        source_shape: tuple[int, int, int],
        remote_shape: tuple[int, int, int],
    ) -> list[dict[str, Any]]:
        source_h, source_w = source_shape[:2]
        remote_h, remote_w = remote_shape[:2]
        if source_w == remote_w and source_h == remote_h:
            return actions
        scale_x = remote_w / max(source_w, 1)
        scale_y = remote_h / max(source_h, 1)
        scaled: list[dict[str, Any]] = []
        for action in actions:
            item = dict(action)
            if item.get("type") == "manual_roi":
                x, y, width, height = item.get("bbox_xywh", [0, 0, 0, 0])
                item["bbox_xywh"] = [
                    float(x) * scale_x,
                    float(y) * scale_y,
                    float(width) * scale_x,
                    float(height) * scale_y,
                ]
            scaled.append(item)
        return scaled

    def _camera_loop(self) -> None:
        if self.camera is None:
            return
        try:
            self.runtime_status = "opening camera"
            self.camera.open()
            self.runtime_status = "running"
            while not self.stop_event.is_set():
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.02)
                    continue
                jpeg = self._encode_jpeg(frame)
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_frame_size = {"width": frame.shape[1], "height": frame.shape[0]}
                    if jpeg is not None:
                        self.latest_raw_jpeg = jpeg
                        self.latest_image_version += 1
                    self._local_frames += 1
                    self._update_fps_locked()
        except Exception as exc:
            with self.lock:
                self.runtime_status = f"error: {exc!r}"
                self._append_chat("assistant", f"Windows camera error: {exc!r}", "error")
        finally:
            self.camera.close()

    def _update_fps_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_fps_at
        if elapsed >= 1.0:
            self.local_fps = self._local_frames / elapsed
            self.remote_fps = self._remote_frames / elapsed
            self._local_frames = 0
            self._remote_frames = 0
            self._last_fps_at = now

    def _connect_locked(self, instruction: str) -> bool:
        self.remote_status = f"connecting {self.server_url}"
        try:
            ws = create_connection(self.server_url, timeout=10)
            ws.send(
                json.dumps(
                    {
                        "type": "start",
                        "instruction": instruction,
                        "return_overlay": True,
                        "overlay_quality": self.overlay_quality,
                        "overlay_every_n": self.overlay_every_n,
                    },
                    ensure_ascii=False,
                )
            )
            ack = json.loads(ws.recv())
            if ack.get("type") == "error":
                raise RuntimeError(str(ack.get("message", "remote start failed")))
            self.ws = ws
            self.remote_status = "connected"
            self.grounding_status = "remote processor ready"
            self.ws_receiver_thread = threading.Thread(
                target=self._ws_receiver_loop,
                name="ws-receiver-loop",
                daemon=True,
            )
            self.ws_receiver_thread.start()
            self._append_chat("assistant", "3090 connected. Streaming camera frames.", "status")
            return True
        except Exception as exc:
            self._close_ws()
            self.remote_status = f"connect failed: {exc}"
            self.grounding_status = "waiting for 3090 connection"
            self._append_chat("assistant", f"Failed to connect to 3090: {exc}", "error")
            return False

    def _pop_remote_actions_locked(self) -> list[dict[str, Any]]:
        actions = self.control_queue + self.manual_roi_queue
        self.control_queue = []
        self.manual_roi_queue = []
        return actions

    def _ws_receiver_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                ws = self.ws
            if ws is None:
                return
            try:
                response = json.loads(ws.recv())
            except (WebSocketConnectionClosedException, OSError, RuntimeError) as exc:
                with self.lock:
                    if self.ws is ws:
                        self.remote_status = f"disconnected: {exc}"
                        self._append_chat("assistant", f"3090 disconnected: {exc}", "error")
                        self._close_ws()
                        self.pending_start_instruction = self.current_instruction or None
                return
            except json.JSONDecodeError as exc:
                with self.lock:
                    self.remote_status = f"bad 3090 json: {exc}"
                continue

            if response.get("type") == "ack":
                with self.lock:
                    self._append_chat("assistant", str(response.get("message", "ack")), "status")
                continue
            if response.get("type") == "error":
                with self.lock:
                    self.remote_status = f"remote error: {response.get('message')}"
                    self._append_chat("assistant", str(response.get("message", "remote error")), "error")
                continue

            overlay = response.get("overlay_jpeg_b64")
            with self.lock:
                if self.ws is not ws:
                    return
                self.latest_response = response
                self.latest_response_at = time.monotonic()
                self.grounding_status = str(response.get("grounding_status", self.grounding_status))
                if overlay and str(overlay) != self.latest_overlay_b64:
                    self.latest_overlay_b64 = str(overlay)
                    self.latest_overlay_jpeg = base64.b64decode(self.latest_overlay_b64)
                    self.latest_overlay_at = time.monotonic()
                    self.latest_image_version += 1
                self._remote_frames += 1
                self._update_fps_locked()

    def _remote_loop(self) -> None:
        while not self.stop_event.is_set():
            frame: np.ndarray | None = None
            actions: list[dict[str, Any]] = []
            with self.lock:
                if self.ws is None and self.pending_start_instruction:
                    instruction = self.pending_start_instruction
                    self.pending_start_instruction = None
                    self._connect_locked(instruction)
                if self.ws is not None:
                    actions = self._pop_remote_actions_locked()
                    frame = None if self.latest_frame is None else self.latest_frame.copy()

            if self.ws is None:
                time.sleep(0.05)
                continue

            try:
                if frame is None:
                    time.sleep(0.02)
                    continue
                remote_frame = self._resize_for_remote(frame)
                actions = self._scale_actions_for_remote(actions, frame.shape, remote_frame.shape)

                for action in actions:
                    with self.ws_send_lock:
                        self.ws.send(json.dumps(action, ensure_ascii=False))

                jpeg = self._encode_jpeg(remote_frame)
                if jpeg is None:
                    continue
                with self.ws_send_lock:
                    self.ws.send_binary(jpeg)
            except (WebSocketConnectionClosedException, OSError, RuntimeError) as exc:
                with self.lock:
                    self.remote_status = f"disconnected: {exc}"
                    self._append_chat("assistant", f"3090 disconnected: {exc}", "error")
                    self._close_ws()
                    self.pending_start_instruction = self.current_instruction or None
            except Exception as exc:
                with self.lock:
                    self.remote_status = f"error: {exc}"
                    self._append_chat("assistant", f"Remote communication error: {exc}", "error")
                    self._close_ws()
                    self.pending_start_instruction = self.current_instruction or None
            time.sleep(self.remote_frame_interval)


class WebHandler(BaseHTTPRequestHandler):
    runtime: WindowsWebClientRuntime

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "gpt_console_v3.html", "text/html; charset=utf-8")
            return
        if path == "/stream.mjpg":
            self._stream_mjpeg()
            return
        if path == "/api/status":
            self._send_json(self.runtime.status())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self._read_json()
        if path == "/api/message":
            self._send_json(self.runtime.submit_instruction(str(payload.get("message", ""))))
            return
        if path == "/api/control":
            self._send_json(self.runtime.control(str(payload.get("action", ""))))
            return
        if path == "/api/roi":
            self._send_json(self.runtime.apply_manual_roi(payload))
            return
        if path == "/api/chat/new":
            self._send_json(self.runtime.new_chat())
            return
        if path == "/api/chat/select":
            self._send_json(self.runtime.select_chat(str(payload.get("conversation_id", ""))))
            return
        if path == "/api/chat/rename":
            self._send_json(
                self.runtime.rename_chat(
                    str(payload.get("conversation_id", "")),
                    str(payload.get("title", "")),
                )
            )
            return
        if path == "/api/chat/delete":
            self._send_json(self.runtime.delete_chat(str(payload.get("conversation_id", ""))))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self) -> None:
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last_version = -1
        next_dynamic_frame_at = 0.0
        dynamic_interval = 1.0 / 30.0
        try:
            while not self.runtime.stop_event.is_set():
                dynamic_overlay = self.runtime.stream_overlay_active()
                if dynamic_overlay:
                    now = time.monotonic()
                    if now < next_dynamic_frame_at:
                        time.sleep(min(0.01, next_dynamic_frame_at - now))
                        continue
                    next_dynamic_frame_at = now + dynamic_interval
                version, frame = self.runtime.latest_image_snapshot()
                if frame is None:
                    time.sleep(0.01)
                    continue
                if not dynamic_overlay and version == last_version:
                    time.sleep(0.01)
                    continue
                last_version = version
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Cache-Control: no-store\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return


def run_web_console(args: argparse.Namespace) -> None:
    tunnel_process = _maybe_start_tunnel(args)
    runtime = WindowsWebClientRuntime(
        server_url=args.server,
        camera_mode=args.camera_mode,
        camera_source=_source(args.camera),
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        side_by_side=args.side_by_side,
        jpeg_quality=args.jpeg_quality,
        overlay_quality=args.overlay_quality,
        overlay_every_n=args.overlay_every_n,
        remote_fps=args.remote_fps,
        remote_width=args.remote_width,
        remote_height=args.remote_height,
        video_url=args.video_url,
        initial_instruction=args.command,
    )
    WebHandler.runtime = runtime
    runtime.start()
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(f"[Windows web console] open http://{args.host}:{args.port}")
    print(f"[Windows web console] chat store: {runtime.store.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Windows web console] stopping")
    finally:
        server.shutdown()
        runtime.stop()
        if tunnel_process is not None and tunnel_process.poll() is None:
            print("[Tunnel] stopping auto-started tunnel")
            tunnel_process.terminate()
            try:
                tunnel_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                tunnel_process.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows browser console for the 3090 tracker.")
    parser.add_argument("--server", required=True, help="WebSocket URL, e.g. ws://127.0.0.1:8001/ws")
    parser.add_argument(
        "--camera-mode",
        default="local",
        choices=["local", "remote"],
        help="local uses the Windows camera; remote watches frames uploaded by Raspberry Pi",
    )
    parser.add_argument("--command", default="", help="Optional initial target instruction")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or stream URL")
    parser.add_argument("--backend", default="dshow", help="OpenCV backend: any, dshow, msmf")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--side-by-side", default="auto", choices=["auto", "none", "left", "right"])
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--overlay-quality", type=int, default=65)
    parser.add_argument(
        "--overlay-every-n",
        type=int,
        default=2,
        help="Return annotated overlay from the 3090 once every N remote frames",
    )
    parser.add_argument(
        "--remote-fps",
        type=float,
        default=20.0,
        help="Local camera upload FPS, or /state polling FPS in Raspberry Pi camera mode",
    )
    parser.add_argument(
        "--remote-width",
        type=int,
        default=640,
        help="Resize frames sent to the 3090 to this width; 0 keeps original width",
    )
    parser.add_argument(
        "--remote-height",
        type=int,
        default=0,
        help="Resize frames sent to the 3090 to this height; 0 preserves aspect ratio",
    )
    parser.add_argument(
        "--video-url",
        default="",
        help="Direct MJPEG URL for remote camera mode; default is <3090-http-base>/stream.mjpg",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-auto-tunnel",
        dest="auto_tunnel",
        action="store_false",
        help="Do not auto-start the localhost SSH tunnel",
    )
    parser.set_defaults(auto_tunnel=True)
    parser.add_argument("--tunnel-script", default="scripts/start_3090_tunnel.ps1")
    parser.add_argument("--tunnel-local-port", type=int, default=8001)
    parser.add_argument("--tunnel-remote-port", type=int, default=8000)
    parser.add_argument("--tunnel-ssh-host", default="202.121.181.124")
    parser.add_argument("--tunnel-ssh-user", default="sjtu")
    parser.add_argument("--tunnel-ssh-port", type=int, default=2020)
    parser.add_argument("--tunnel-identity-file", default="F:/SSHKeys/sjtu_ed25519")
    args = parser.parse_args(argv)
    run_web_console(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
