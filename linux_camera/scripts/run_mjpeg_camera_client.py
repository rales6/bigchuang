from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from websocket import WebSocketConnectionClosedException, create_connection

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.robot_controller import RobotCommandDispatcher


SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


class RemoteReceiver(threading.Thread):
    """后台接收 3090 返回，避免阻塞摄像头 MJPEG 上传。"""

    def __init__(self, ws: Any, robot: RobotCommandDispatcher) -> None:
        super().__init__(daemon=True)
        self.ws = ws
        self.robot = robot
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.latest_response: dict[str, Any] = {}
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
            with self.lock:
                self.latest_response = response
                self.responses += 1

    def snapshot(self) -> tuple[dict[str, Any], int, str]:
        with self.lock:
            response = dict(self.latest_response)
            responses = self.responses
            self.responses = 0
            return response, responses, self.last_error


def _camera_device(camera: str) -> str:
    if camera.startswith("/dev/"):
        return camera
    try:
        return f"/dev/video{int(camera)}"
    except ValueError:
        return camera


def _ffmpeg_command(args: argparse.Namespace) -> list[str]:
    return [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel",
        args.ffmpeg_loglevel,
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        str(args.fps),
        "-i",
        _camera_device(args.camera),
        "-an",
        "-c:v",
        "copy",
        "-f",
        "mjpeg",
        "pipe:1",
    ]


def _iter_jpegs(stream: Any, chunk_size: int = 65536):
    buffer = bytearray()
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return
        buffer.extend(chunk)
        while True:
            start = buffer.find(SOI)
            if start < 0:
                if len(buffer) > chunk_size:
                    del buffer[:-2]
                break
            end = buffer.find(EOI, start + 2)
            if end < 0:
                if start > 0:
                    del buffer[:start]
                break
            frame = bytes(buffer[start : end + 2])
            del buffer[: end + 2]
            yield frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload camera MJPEG frames to the 3090 without OpenCV decode/re-encode."
    )
    parser.add_argument("--server", required=True, help="3090 camera WebSocket URL")
    parser.add_argument("--command", default="", help="Optional initial instruction")
    parser.add_argument("--camera", default="0", help="Camera index or /dev/videoX")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--remote-fps", type=float, default=30.0, help="Max upload FPS")
    parser.add_argument("--overlay-quality", type=int, default=60)
    parser.add_argument("--overlay-every-n", type=int, default=3)
    parser.add_argument("--no-overlay", action="store_true", help="Do not request annotated frames")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path")
    parser.add_argument("--ffmpeg-loglevel", default="warning")
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
    parser.add_argument("--robot-debug-log", action="store_true")
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
    robot.open()

    command = _ffmpeg_command(args)
    print("[MJPEG camera] " + " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
    )
    if process.stdout is None:
        raise RuntimeError("ffmpeg stdout is unavailable")

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

    receiver = RemoteReceiver(ws, robot)
    receiver.start()
    print("[MJPEG camera] streaming compressed JPEG frames. Press Ctrl+C to stop.")

    # remote-fps 高于或等于摄像头 FPS 时不做软件限帧，避免 ffmpeg 管道成批吐帧时被误判为“过早”而丢帧。
    remote_interval = 0.0 if float(args.remote_fps) >= float(args.fps) else 1.0 / max(1.0, float(args.remote_fps))
    next_send_at = 0.0
    last_log_at = monotonic()
    sent_frames = 0
    sent_kb = 0
    skipped_frames = 0
    send_ms_sum = 0.0

    try:
        for jpeg in _iter_jpegs(process.stdout):
            now = monotonic()
            if remote_interval > 0.0:
                if now < next_send_at:
                    skipped_frames += 1
                    continue
                next_send_at = now + remote_interval

            send_started = monotonic()
            ws.send_binary(jpeg)
            send_ms_sum += (monotonic() - send_started) * 1000.0
            sent_frames += 1
            sent_kb += len(jpeg) // 1024

            now = monotonic()
            if now - last_log_at >= 1.0:
                elapsed = now - last_log_at
                response, received_frames, error = receiver.snapshot()
                guidance = response.get("guidance", {})
                track = response.get("track", {})
                camera_state = response.get("camera", {})
                if not isinstance(camera_state, dict):
                    camera_state = {}
                print(
                    f"[MJPEG] sent_fps={sent_frames / max(elapsed, 1e-3):.1f} "
                    f"recv_fps={received_frames / max(elapsed, 1e-3):.1f} "
                    f"server_fps={camera_state.get('fps', '-')} "
                    f"track={track.get('status')} "
                    f"guidance={guidance.get('direction')} "
                    f"blocked={response.get('safety', {}).get('blocked')} "
                    f"avg_jpeg={sent_kb / max(sent_frames, 1):.0f}KB "
                    f"skipped={skipped_frames} "
                    f"send={send_ms_sum / max(sent_frames, 1):.1f}ms"
                )
                if error:
                    print(f"[3090 recv] {error}")
                sent_frames = 0
                sent_kb = 0
                skipped_frames = 0
                send_ms_sum = 0.0
                last_log_at = now
    except KeyboardInterrupt:
        print("\n[MJPEG camera] stopping")
    finally:
        receiver.stop_event.set()
        try:
            ws.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        try:
            ws.close()
        except Exception:
            pass
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
        robot.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
