from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import cv2
import numpy as np
from websocket import create_connection

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.camera.opencv_camera import OpenCVCamera


def _source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _send_control(ws: Any, payload: dict[str, Any]) -> dict[str, Any]:
    ws.send(json.dumps(payload, ensure_ascii=False))
    response = json.loads(ws.recv())
    print(f"[Server] {response}")
    return response


def _decode_overlay(payload: dict[str, Any]) -> np.ndarray | None:
    encoded = payload.get("overlay_jpeg_b64")
    if not encoded:
        return None
    data = base64.b64decode(encoded)
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a local Windows/Linux camera to the 3090 tracker server."
    )
    parser.add_argument("--server", required=True, help="WebSocket URL, e.g. ws://3090-host:8000/ws")
    parser.add_argument("--command", required=True, help="Natural-language target instruction")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or stream URL")
    parser.add_argument("--backend", default="any", help="OpenCV backend: any, dshow, msmf, v4l2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--side-by-side", default="auto", choices=["auto", "none", "left", "right"])
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--overlay-quality", type=int, default=75)
    parser.add_argument("--window-name", default="Qwen Grounded Tracker Remote Client")
    parser.add_argument("--no-overlay", action="store_true", help="Do not request annotated frames back")
    args = parser.parse_args()

    camera = OpenCVCamera(
        source=_source(args.camera),
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        side_by_side=args.side_by_side,
    )
    camera.open()
    ws = create_connection(args.server, timeout=10)
    ws.send(
        json.dumps(
            {
                "type": "start",
                "instruction": args.command,
                "return_overlay": not args.no_overlay,
                "overlay_quality": args.overlay_quality,
            },
            ensure_ascii=False,
        )
    )
    print(f"[Server] {json.loads(ws.recv())}")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    emergency_stop = False
    last_log_at = 0.0
    last_raw_frame: np.ndarray | None = None

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[Camera] frame missing")
                sleep(0.02)
                continue
            last_raw_frame = frame.copy()

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(95, args.jpeg_quality))],
            )
            if not ok:
                print("[Client] JPEG encode failed")
                continue

            ws.send_binary(encoded.tobytes())
            response = json.loads(ws.recv())
            if response.get("type") == "error":
                print(f"[Server error] {response.get('message')}")
                continue

            display = _decode_overlay(response)
            if display is None:
                display = frame
            cv2.imshow(args.window_name, display)

            now = monotonic()
            if now - last_log_at >= 1.0:
                guidance = response.get("guidance", {})
                track = response.get("track", {})
                print(
                    f"[Remote] track={track.get('status')} "
                    f"guidance={guidance.get('direction')} "
                    f"blocked={response.get('safety', {}).get('blocked')}"
                )
                last_log_at = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                _send_control(ws, {"type": "close"})
                break
            if key == ord("g"):
                _send_control(ws, {"type": "reground"})
            elif key == ord("r"):
                _send_control(ws, {"type": "reset"})
            elif key == ord("c"):
                _send_control(ws, {"type": "toggle_boundary"})
            elif key == 32:
                emergency_stop = not emergency_stop
                _send_control(ws, {"type": "emergency_stop", "enabled": emergency_stop})
            elif key == ord("i"):
                new_instruction = input("instruction> ").strip()
                if new_instruction:
                    _send_control(
                        ws,
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
                        {"type": "manual_roi", "bbox_xywh": [x, y, width, height]},
                    )
    finally:
        camera.close()
        ws.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
