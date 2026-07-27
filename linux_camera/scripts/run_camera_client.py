from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np
from websocket import WebSocketConnectionClosedException, create_connection

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.camera.opencv_camera import OpenCVCamera
from qwen_grounded_tracker.robot_controller import RobotCommandDispatcher


def _source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _send_control(ws: Any, send_lock: threading.Lock, payload: dict[str, Any]) -> None:
    with send_lock:
        ws.send(json.dumps(payload, ensure_ascii=False))


def _decode_overlay(payload: dict[str, Any]) -> np.ndarray | None:
    encoded = payload.get("overlay_jpeg_b64")
    if not encoded:
        return None
    data = base64.b64decode(encoded)
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _resize_for_remote(frame: np.ndarray, remote_width: int, remote_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if remote_width <= 0 and remote_height <= 0:
        return frame
    if remote_width <= 0:
        scale = remote_height / max(height, 1)
        remote_width = int(round(width * scale))
    elif remote_height <= 0:
        scale = remote_width / max(width, 1)
        remote_height = int(round(height * scale))
    remote_width = max(16, remote_width)
    remote_height = max(16, remote_height)
    if remote_width == width and remote_height == height:
        return frame
    return cv2.resize(frame, (remote_width, remote_height), interpolation=cv2.INTER_AREA)


class RemoteReceiver(threading.Thread):
    """Receive 3090 responses without blocking camera uploads."""

    def __init__(self, ws: Any, robot: RobotCommandDispatcher, decode_overlay: bool) -> None:
        super().__init__(daemon=True)
        self.ws = ws
        self.robot = robot
        self.decode_overlay = decode_overlay
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.latest_response: dict[str, Any] = {}
        self.latest_overlay: np.ndarray | None = None
        self.responses = 0
        self.last_error = ""

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                response = json.loads(self.ws.recv())
            except (WebSocketConnectionClosedException, OSError, RuntimeError) as exc:
                self.last_error = str(exc)
                return
            except json.JSONDecodeError as exc:
                self.last_error = f"bad json from 3090: {exc}"
                continue

            if response.get("type") == "ack":
                print(f"[3090] {response}")
                continue
            if response.get("type") == "error":
                print(f"[3090 error] {response.get('message')}")
                continue

            self.robot.update_from_3090(response)
            overlay = _decode_overlay(response) if self.decode_overlay else None
            with self.lock:
                self.latest_response = response
                if overlay is not None:
                    self.latest_overlay = overlay
                self.responses += 1

    def snapshot(self) -> tuple[dict[str, Any], np.ndarray | None, int, str]:
        with self.lock:
            response = dict(self.latest_response)
            overlay = None if self.latest_overlay is None else self.latest_overlay.copy()
            responses = self.responses
            self.responses = 0
            return response, overlay, responses, self.last_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi camera uploader for the 3090 tracker server."
    )
    parser.add_argument(
        "--server",
        required=True,
        help="3090 camera WebSocket URL, e.g. ws://192.168.55.33:8000/camera_ws",
    )
    parser.add_argument(
        "--command",
        default="",
        help="Optional initial instruction; Windows web console can also send it later.",
    )
    parser.add_argument("--camera", default="0", help="OpenCV camera index or stream URL")
    parser.add_argument("--backend", default="v4l2", help="OpenCV backend: any, v4l2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="", help="Optional camera FOURCC, e.g. MJPG or YUYV")
    parser.add_argument("--camera-buffer", type=int, default=1, help="OpenCV camera buffer size")
    parser.add_argument(
        "--drop-stale-frames",
        type=int,
        default=0,
        help="Discard queued camera frames before each upload; keep 0 for async low-latency streaming",
    )
    parser.add_argument("--side-by-side", default="auto", choices=["auto", "none", "left", "right"])
    parser.add_argument("--remote-fps", type=float, default=10.0, help="Max upload FPS to 3090")
    parser.add_argument("--remote-width", type=int, default=640, help="Upload width; 0 keeps original")
    parser.add_argument("--remote-height", type=int, default=0, help="Upload height; 0 preserves aspect")
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--overlay-quality", type=int, default=60)
    parser.add_argument("--overlay-every-n", type=int, default=3)
    parser.add_argument("--no-overlay", action="store_true", help="Do not request annotated frames back")
    parser.add_argument("--window", action="store_true", help="Show a local debug preview on the Pi")
    parser.add_argument("--window-name", default="Raspberry Pi Camera -> 3090")
    parser.add_argument(
        "--robot-mode",
        default="dry-run",
        choices=["none", "dry-run", "serial", "esp32"],
        help="How to execute robot commands returned by 3090",
    )
    parser.add_argument("--robot-serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--robot-serial-baudrate", type=int, default=115200)
    parser.add_argument("--robot-command-rate", type=float, default=10.0)
    parser.add_argument("--robot-max-linear", type=float, default=0.18)
    parser.add_argument("--robot-max-angular", type=float, default=0.45)
    parser.add_argument(
        "--robot-debug-log",
        action="store_true",
        help="Print dry-run robot commands for debugging",
    )
    parser.add_argument("--esp32-link", default="ble", choices=["ble", "uart", "auto"])
    parser.add_argument("--esp32-uart-port", default="/dev/serial0")
    parser.add_argument("--esp32-uart-baudrate", type=int, default=230400)
    parser.add_argument("--esp32-ble-name", default="ESP32-Robot-Car")
    parser.add_argument("--esp32-ble-address", default="")
    parser.add_argument("--esp32-ttl-ms", type=int, default=600)
    parser.add_argument("--gripper-joint", type=int, default=5)
    parser.add_argument("--gripper-open-us", type=int, default=1200)
    parser.add_argument("--gripper-close-us", type=int, default=1550)
    parser.add_argument("--arm-duration-ms", type=int, default=800)
    args = parser.parse_args()

    camera = OpenCVCamera(
        source=_source(args.camera),
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        side_by_side=args.side_by_side,
        buffer_size=args.camera_buffer,
        fourcc=args.fourcc,
    )
    robot = RobotCommandDispatcher(
        mode=args.robot_mode,
        serial_port=args.robot_serial_port,
        serial_baudrate=args.robot_serial_baudrate,
        command_rate_hz=args.robot_command_rate,
        max_linear=args.robot_max_linear,
        max_angular=args.robot_max_angular,
        esp32_link=args.esp32_link,
        esp32_uart_port=args.esp32_uart_port,
        esp32_uart_baudrate=args.esp32_uart_baudrate,
        esp32_ble_name=args.esp32_ble_name,
        esp32_ble_address=args.esp32_ble_address,
        esp32_ttl_ms=args.esp32_ttl_ms,
        gripper_joint=args.gripper_joint,
        gripper_open_us=args.gripper_open_us,
        gripper_close_us=args.gripper_close_us,
        arm_duration_ms=args.arm_duration_ms,
        debug_log=args.robot_debug_log,
    )
    camera.open()
    robot.open()
    ws = create_connection(args.server, timeout=10)
    ws.send(
        json.dumps(
            {
                "type": "camera_start",
                "instruction": args.command,
                "return_overlay": not args.no_overlay,
                "overlay_quality": args.overlay_quality,
                "overlay_every_n": args.overlay_every_n,
            },
            ensure_ascii=False,
        )
    )
    print(f"[3090] {json.loads(ws.recv())}")
    print("[Pi camera] streaming frames. Press Ctrl+C to stop.")
    send_lock = threading.Lock()
    receiver = RemoteReceiver(ws, robot, decode_overlay=args.window)
    receiver.start()

    if args.window:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    emergency_stop = False
    last_log_at = 0.0
    next_send_at = 0.0
    sent_frames = 0
    sent_kb = 0
    read_ms_sum = 0.0
    resize_ms_sum = 0.0
    encode_ms_sum = 0.0
    send_ms_sum = 0.0
    quality = max(30, min(95, int(args.jpeg_quality)))
    remote_interval = 1.0 / max(1.0, float(args.remote_fps))
    last_raw_frame: np.ndarray | None = None

    try:
        while True:
            read_started = monotonic()
            frame = camera.read_latest(args.drop_stale_frames)
            read_ms_sum += (monotonic() - read_started) * 1000.0
            if frame is None:
                print("[Camera] frame missing")
                sleep(0.02)
                continue
            last_raw_frame = frame.copy()

            now = monotonic()
            if now < next_send_at:
                sleep(min(0.01, next_send_at - now))
                continue
            next_send_at = now + remote_interval

            # 树莓派先压缩/降采样再上传，减轻无线网络和 3090 解码压力。
            resize_started = monotonic()
            remote_frame = _resize_for_remote(frame, args.remote_width, args.remote_height)
            resize_ms_sum += (monotonic() - resize_started) * 1000.0
            encode_started = monotonic()
            ok, encoded = cv2.imencode(
                ".jpg",
                remote_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            encode_ms_sum += (monotonic() - encode_started) * 1000.0
            if not ok:
                print("[Pi camera] JPEG encode failed")
                continue

            jpeg_bytes = encoded.tobytes()
            send_started = monotonic()
            with send_lock:
                ws.send_binary(jpeg_bytes)
            send_ms_sum += (monotonic() - send_started) * 1000.0
            sent_frames += 1
            sent_kb += len(jpeg_bytes) // 1024

            if args.window:
                response, display, _, error = receiver.snapshot()
                if display is None:
                    display = remote_frame
                cv2.imshow(args.window_name, display)
                if error:
                    print(f"[3090 recv] {error}")

            now = monotonic()
            if now - last_log_at >= 1.0:
                elapsed = now - last_log_at
                response, _, received_frames, error = receiver.snapshot()
                guidance = response.get("guidance", {})
                track = response.get("track", {})
                camera_state = response.get("camera", {})
                if not isinstance(camera_state, dict):
                    camera_state = {}
                print(
                    f"[Remote] sent_fps={sent_frames / max(elapsed, 1e-3):.1f} "
                    f"recv_fps={received_frames / max(elapsed, 1e-3):.1f} "
                    f"server_fps={camera_state.get('fps', '-')} "
                    f"age_ms={camera_state.get('age_ms', '-')} "
                    f"track={track.get('status')} "
                    f"guidance={guidance.get('direction')} "
                    f"blocked={response.get('safety', {}).get('blocked')} "
                    f"avg_jpeg={sent_kb / max(sent_frames, 1):.0f}KB "
                    f"read={read_ms_sum / max(sent_frames, 1):.1f}ms "
                    f"resize={resize_ms_sum / max(sent_frames, 1):.1f}ms "
                    f"encode={encode_ms_sum / max(sent_frames, 1):.1f}ms "
                    f"send={send_ms_sum / max(sent_frames, 1):.1f}ms"
                )
                if error:
                    print(f"[3090 recv] {error}")
                sent_frames = 0
                sent_kb = 0
                read_ms_sum = 0.0
                resize_ms_sum = 0.0
                encode_ms_sum = 0.0
                send_ms_sum = 0.0
                last_log_at = now

            if not args.window:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                _send_control(ws, send_lock, {"type": "close"})
                break
            if key == ord("g"):
                _send_control(ws, send_lock, {"type": "reground"})
            elif key == ord("r"):
                _send_control(ws, send_lock, {"type": "reset"})
            elif key == ord("c"):
                _send_control(ws, send_lock, {"type": "toggle_boundary"})
            elif key == 32:
                emergency_stop = not emergency_stop
                _send_control(ws, send_lock, {"type": "emergency_stop", "enabled": emergency_stop})
            elif key == ord("i"):
                new_instruction = input("instruction> ").strip()
                if new_instruction:
                    _send_control(
                        ws,
                        send_lock,
                        {"type": "set_instruction", "instruction": new_instruction},
                    )
            elif key == ord("m") and last_raw_frame is not None:
                roi = cv2.selectROI(
                    "Manual target selection",
                    last_raw_frame,
                    fromCenter=False,
                    showCrosshair=True,
                )
                cv2.destroyWindow("Manual target selection")
                x, y, width, height = [int(v) for v in roi]
                if width > 1 and height > 1:
                    _send_control(
                        ws,
                        send_lock,
                        {"type": "manual_roi", "bbox_xywh": [x, y, width, height]},
                    )
    except KeyboardInterrupt:
        print("\n[Pi camera] stopping")
    finally:
        receiver.stop_event.set()
        camera.close()
        robot.close()
        ws.close()
        if args.window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
