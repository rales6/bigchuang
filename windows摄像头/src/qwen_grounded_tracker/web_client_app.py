from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


class WindowsWebClientRuntime:
    """Windows-only browser console for a remote 3090 tracker server."""

    def __init__(
        self,
        server_url: str,
        camera_source: int | str,
        backend: str,
        width: int,
        height: int,
        fps: int,
        side_by_side: str,
        jpeg_quality: int,
        overlay_quality: int,
        initial_instruction: str,
    ) -> None:
        self.server_url = server_url
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))
        self.overlay_quality = max(30, min(95, int(overlay_quality)))
        self.initial_instruction = initial_instruction.strip()

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
        self.thread: threading.Thread | None = None
        self.ws: Any | None = None

        self.runtime_status = "stopped"
        self.remote_status = "not connected"
        self.grounding_status = "waiting for instruction"
        self.current_instruction = self.initial_instruction
        self.pending_start_instruction = self.initial_instruction or None
        self.control_queue: list[dict[str, Any]] = []
        self.chat_history: list[dict[str, str]] = []
        self.latest_jpeg: bytes | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_size = {"width": 0, "height": 0}
        self.latest_response: dict[str, Any] = {
            "track": {"status": "waiting for remote server", "bbox": None, "visible": False},
            "guidance": {"direction": "STOP", "linear": 0.0, "angular": 0.0},
            "safety": {"blocked": True, "reasons": []},
            "boundary_mode": "-",
            "emergency_stop": False,
            "obstacles": {"status": "not run"},
            "lidar": {"status": "not run"},
        }

        if self.initial_instruction:
            self._append_chat("user", self.initial_instruction, "initial")

    def _append_chat(self, role: str, content: str, kind: str = "message") -> None:
        self.chat_history.append(
            {"role": role, "content": content, "kind": kind, "time": _now_label()}
        )
        self.chat_history = self.chat_history[-200:]

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, name="windows-web-client", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3.0)
        self._close_ws()

    def submit_instruction(self, instruction: str) -> dict[str, Any]:
        instruction = instruction.strip()
        if not instruction:
            return {"ok": False, "message": "empty instruction"}
        with self.lock:
            self.current_instruction = instruction
            self._append_chat("user", instruction)
            if self.ws is None:
                self.pending_start_instruction = instruction
                self._append_chat("assistant", "已准备连接 3090 并发送这条指令。", "status")
            else:
                self.control_queue.append({"type": "set_instruction", "instruction": instruction})
                self._append_chat("assistant", "已把新指令排队发送给 3090。", "status")
        return {"ok": True}

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
            if self.ws is None and not self.pending_start_instruction:
                self.current_instruction = self.current_instruction or "manual ROI"
                self.pending_start_instruction = self.current_instruction
            self.control_queue.append(
                {"type": "manual_roi", "bbox_xywh": [x, y, box_width, box_height]}
            )
            self._append_chat("assistant", "手动 ROI 已排队发送给 3090。", "status")
        return {"ok": True}

    def control(self, action: str) -> dict[str, Any]:
        with self.lock:
            if action == "reset":
                self.control_queue.append({"type": "reset"})
                self._append_chat("assistant", "已请求 3090 清除当前目标。", "status")
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
                if self.latest_frame is None:
                    return {"ok": False, "message": "no frame"}
                output_dir = Path.cwd() / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = output_dir / f"web_client_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(filename), self.latest_frame)
                return {"ok": True, "path": str(filename)}
            return {"ok": False, "message": f"unknown action: {action}"}

    def status(self) -> dict[str, Any]:
        with self.lock:
            response = dict(self.latest_response)
            return {
                "runtime_status": self.runtime_status,
                "remote_status": self.remote_status,
                "grounding_status": self.grounding_status,
                "instruction": self.current_instruction,
                "server_url": self.server_url,
                "frame": self.latest_frame_size,
                "track": response.get("track", {}),
                "guidance": response.get("guidance", {}),
                "safety": response.get("safety", {}),
                "boundary_mode": response.get("boundary_mode", "-"),
                "emergency_stop": response.get("emergency_stop", False),
                "obstacles": response.get("obstacles", {}).get("status", "not run"),
                "lidar": response.get("lidar", {}).get("status", "not run"),
                "chat": list(self.chat_history),
            }

    def latest_image(self) -> bytes | None:
        with self.lock:
            return None if self.latest_jpeg is None else bytes(self.latest_jpeg)

    def _close_ws(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

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
            self._append_chat("assistant", "3090 已连接，正在发送摄像头画面。", "status")
            return True
        except Exception as exc:
            self._close_ws()
            self.remote_status = f"connect failed: {exc}"
            self.grounding_status = "waiting for 3090 connection"
            self._append_chat("assistant", f"连接 3090 失败：{exc}", "error")
            return False

    def _send_controls_locked(self) -> None:
        if self.ws is None:
            return
        while self.control_queue:
            payload = self.control_queue.pop(0)
            self.ws.send(json.dumps(payload, ensure_ascii=False))
            response = json.loads(self.ws.recv())
            if response.get("type") == "ack":
                self._append_chat("assistant", str(response.get("message", "ack")), "status")
            elif response.get("type") == "error":
                self._append_chat("assistant", str(response.get("message", "remote error")), "error")

    def _encode_raw(self, frame: np.ndarray) -> bytes | None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        return encoded.tobytes() if ok else None

    def _loop(self) -> None:
        try:
            self.runtime_status = "opening camera"
            self.camera.open()
            self.runtime_status = "running"
            while not self.stop_event.is_set():
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.02)
                    continue

                raw_jpeg = self._encode_raw(frame)
                with self.lock:
                    self.latest_frame = frame.copy()
                    self.latest_frame_size = {"width": frame.shape[1], "height": frame.shape[0]}
                    if raw_jpeg is not None and self.ws is None:
                        self.latest_jpeg = raw_jpeg

                    if self.ws is None and self.pending_start_instruction:
                        instruction = self.pending_start_instruction
                        self.pending_start_instruction = None
                        self._connect_locked(instruction)

                    if self.ws is not None:
                        try:
                            self._send_controls_locked()
                            if raw_jpeg is not None:
                                self.ws.send_binary(raw_jpeg)
                                response = json.loads(self.ws.recv())
                                if response.get("type") == "error":
                                    self.remote_status = f"remote error: {response.get('message')}"
                                else:
                                    self.latest_response = response
                                    self.grounding_status = str(
                                        response.get("grounding_status", self.grounding_status)
                                    )
                                    overlay = response.get("overlay_jpeg_b64")
                                    if overlay:
                                        self.latest_jpeg = base64.b64decode(overlay)
                                    elif raw_jpeg is not None:
                                        self.latest_jpeg = raw_jpeg
                        except (WebSocketConnectionClosedException, OSError, RuntimeError) as exc:
                            self.remote_status = f"disconnected: {exc}"
                            self._append_chat("assistant", f"3090 连接断开：{exc}", "error")
                            self._close_ws()
                            self.pending_start_instruction = self.current_instruction or None
                        except Exception as exc:
                            self.remote_status = f"error: {exc}"
                            self._append_chat("assistant", f"远程通信出错：{exc}", "error")
                            self._close_ws()
                            self.pending_start_instruction = self.current_instruction or None
                time.sleep(0.001)
        except Exception as exc:
            with self.lock:
                self.runtime_status = f"error: {exc!r}"
                self._append_chat("assistant", f"Windows 客户端运行出错：{exc!r}", "error")
        finally:
            self._close_ws()
            self.camera.close()
            with self.lock:
                if not self.runtime_status.startswith("error:"):
                    self.runtime_status = "stopped"


class WebHandler(BaseHTTPRequestHandler):
    runtime: WindowsWebClientRuntime

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "console.html", "text/html; charset=utf-8")
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
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while not self.runtime.stop_event.is_set():
                frame = self.runtime.latest_image()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            return


def run_web_console(args: argparse.Namespace) -> None:
    runtime = WindowsWebClientRuntime(
        server_url=args.server,
        camera_source=_source(args.camera),
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        side_by_side=args.side_by_side,
        jpeg_quality=args.jpeg_quality,
        overlay_quality=args.overlay_quality,
        initial_instruction=args.command,
    )
    WebHandler.runtime = runtime
    runtime.start()
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    print(f"[Windows web console] open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Windows web console] stopping")
    finally:
        server.shutdown()
        runtime.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Windows browser console for the 3090 tracker.")
    parser.add_argument("--server", required=True, help="WebSocket URL, e.g. ws://3090-host:8000/ws")
    parser.add_argument("--command", default="", help="Optional initial target instruction")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or stream URL")
    parser.add_argument("--backend", default="dshow", help="OpenCV backend: any, dshow, msmf")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--side-by-side", default="auto", choices=["auto", "none", "left", "right"])
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--overlay-quality", type=int, default=75)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    run_web_console(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
