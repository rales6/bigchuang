from __future__ import annotations

import argparse
import base64
import json
import threading
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np
from websocket import WebSocketConnectionClosedException, create_connection


def _source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _backend(name: str) -> int:
    return {"v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), "any": cv2.CAP_ANY}.get(
        name.lower(),
        cv2.CAP_ANY,
    )


def _open_camera(source: int | str, args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, _backend(args.backend))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.fourcc:
        code = args.fourcc.upper()[:4].ljust(4)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
    if args.camera_buffer > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, args.camera_buffer)
    return cap


def _read(cap: cv2.VideoCapture) -> np.ndarray | None:
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if width <= 0 and height <= 0:
        return frame
    if width <= 0:
        width = int(round(w * height / max(h, 1)))
    elif height <= 0:
        height = int(round(h * width / max(w, 1)))
    width = max(16, width)
    height = max(16, height)
    if width == w and height == h:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _decode_overlay(payload: dict[str, Any]) -> np.ndarray | None:
    encoded = payload.get("overlay_jpeg_b64")
    if not encoded:
        return None
    data = base64.b64decode(encoded)
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


class Receiver(threading.Thread):
    def __init__(self, ws: Any, decode_overlay: bool) -> None:
        super().__init__(daemon=True)
        self.ws = ws
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
    parser = argparse.ArgumentParser(description="Double-camera side-by-side uploader.")
    parser.add_argument("--server", required=True)
    parser.add_argument("--left-camera", required=True)
    parser.add_argument("--right-camera", required=True)
    parser.add_argument("--baseline-mm", type=float, required=True)
    parser.add_argument("--command", default="")
    parser.add_argument("--backend", default="v4l2", choices=["v4l2", "any"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="")
    parser.add_argument("--camera-buffer", type=int, default=1)
    parser.add_argument("--remote-fps", type=float, default=20.0)
    parser.add_argument("--remote-width", type=int, default=960)
    parser.add_argument("--remote-height", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=45)
    parser.add_argument("--overlay-quality", type=int, default=60)
    parser.add_argument("--overlay-every-n", type=int, default=3)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--window", action="store_true")
    args = parser.parse_args()

    left_source = _source(args.left_camera)
    right_source = _source(args.right_camera)
    left = _open_camera(left_source, args)
    right = _open_camera(right_source, args)
    ws = create_connection(args.server, timeout=10)
    ws.send(
        json.dumps(
            {
                "type": "camera_start",
                "instruction": args.command,
                "return_overlay": not args.no_overlay,
                "overlay_quality": args.overlay_quality,
                "overlay_every_n": args.overlay_every_n,
                "stereo": {
                    "layout": "side_by_side",
                    "left_camera": str(args.left_camera),
                    "right_camera": str(args.right_camera),
                    "baseline_mm": float(args.baseline_mm),
                    "left_half": "小车左侧摄像头",
                    "right_half": "小车右侧摄像头",
                },
            },
            ensure_ascii=False,
        )
    )
    print(f"[3090] {json.loads(ws.recv())}")
    receiver = Receiver(ws, decode_overlay=args.window)
    receiver.start()
    send_lock = threading.Lock()
    if args.window:
        cv2.namedWindow("double-camera", cv2.WINDOW_NORMAL)

    interval = 1.0 / max(1.0, args.remote_fps)
    next_send_at = 0.0
    quality = max(20, min(95, args.jpeg_quality))
    sent = 0
    sent_kb = 0
    read_left_ms = 0.0
    read_right_ms = 0.0
    encode_ms = 0.0
    send_ms = 0.0
    last_log_at = monotonic()

    try:
        while True:
            started = monotonic()
            left_frame = _read(left)
            read_left_ms += (monotonic() - started) * 1000.0
            started = monotonic()
            right_frame = _read(right)
            read_right_ms += (monotonic() - started) * 1000.0
            if left_frame is None or right_frame is None:
                print("[Camera] missing left or right frame")
                sleep(0.02)
                continue
            if left_frame.shape[:2] != right_frame.shape[:2]:
                right_frame = cv2.resize(
                    right_frame,
                    (left_frame.shape[1], left_frame.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            stitched = np.hstack([left_frame, right_frame])
            now = monotonic()
            if now < next_send_at:
                sleep(min(0.005, next_send_at - now))
                continue
            next_send_at = now + interval
            remote_frame = _resize(stitched, args.remote_width, args.remote_height)

            started = monotonic()
            ok, encoded = cv2.imencode(
                ".jpg",
                remote_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            encode_ms += (monotonic() - started) * 1000.0
            if not ok:
                continue
            jpeg = encoded.tobytes()
            started = monotonic()
            with send_lock:
                ws.send_binary(jpeg)
            send_ms += (monotonic() - started) * 1000.0
            sent += 1
            sent_kb += len(jpeg) // 1024

            if args.window:
                _, overlay, _, _ = receiver.snapshot()
                cv2.imshow("double-camera", overlay if overlay is not None else remote_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            now = monotonic()
            if now - last_log_at >= 1.0:
                elapsed = now - last_log_at
                response, _, received, error = receiver.snapshot()
                camera = response.get("camera", {}) if isinstance(response.get("camera"), dict) else {}
                print(
                    f"[DoubleCam] sent_fps={sent / max(elapsed, 1e-3):.1f} "
                    f"recv_fps={received / max(elapsed, 1e-3):.1f} "
                    f"server_fps={camera.get('fps', '-')} "
                    f"avg_jpeg={sent_kb / max(sent, 1):.0f}KB "
                    f"read_l={read_left_ms / max(sent, 1):.1f}ms "
                    f"read_r={read_right_ms / max(sent, 1):.1f}ms "
                    f"encode={encode_ms / max(sent, 1):.1f}ms "
                    f"send={send_ms / max(sent, 1):.1f}ms"
                )
                if error:
                    print(f"[3090 recv] {error}")
                sent = 0
                sent_kb = 0
                read_left_ms = 0.0
                read_right_ms = 0.0
                encode_ms = 0.0
                send_ms = 0.0
                last_log_at = now
    except KeyboardInterrupt:
        print("\n[DoubleCam] stopping")
    finally:
        receiver.stop_event.set()
        left.release()
        right.release()
        ws.close()
        if args.window:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

