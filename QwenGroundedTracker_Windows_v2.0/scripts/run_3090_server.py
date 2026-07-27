from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.config import load_config
from qwen_grounded_tracker.remote.processor import RemoteFrameProcessor


app = FastAPI(title="Qwen Grounded Tracker 3090 Server")


def _json_error(message: str) -> str:
    return json.dumps({"type": "error", "message": message}, ensure_ascii=False)


def _decode_jpeg(data: bytes) -> np.ndarray | None:
    array = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _encode_jpeg_b64(frame: np.ndarray, quality: int) -> str:
    quality = max(30, min(95, int(quality)))
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode overlay JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


async def _handle_control(
    websocket: WebSocket,
    processor: RemoteFrameProcessor,
    payload: dict[str, Any],
) -> bool:
    message_type = str(payload.get("type", ""))
    try:
        if message_type == "set_instruction":
            processor.set_instruction(str(payload.get("instruction", "")))
            await websocket.send_text(
                json.dumps({"type": "ack", "message": "instruction updated"}, ensure_ascii=False)
            )
        elif message_type == "reground":
            processor.request_reground()
            await websocket.send_text(
                json.dumps({"type": "ack", "message": "re-ground requested"}, ensure_ascii=False)
            )
        elif message_type == "reset":
            processor.reset()
            await websocket.send_text(
                json.dumps({"type": "ack", "message": "target reset"}, ensure_ascii=False)
            )
        elif message_type == "emergency_stop":
            processor.set_emergency_stop(bool(payload.get("enabled", True)))
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "ack",
                        "message": "emergency stop updated",
                        "enabled": processor.emergency_stop,
                    },
                    ensure_ascii=False,
                )
            )
        elif message_type == "toggle_boundary":
            mode = processor.toggle_boundary()
            await websocket.send_text(
                json.dumps({"type": "ack", "message": "boundary mode toggled", "mode": mode})
            )
        elif message_type == "manual_roi":
            x, y, width, height = payload.get("bbox_xywh", [0, 0, 0, 0])
            processor.set_manual_roi_xywh(float(x), float(y), float(width), float(height))
            await websocket.send_text(
                json.dumps({"type": "ack", "message": "manual ROI queued"})
            )
        elif message_type == "close":
            await websocket.send_text(json.dumps({"type": "ack", "message": "closing"}))
            return False
        else:
            await websocket.send_text(_json_error(f"Unknown control message type: {message_type}"))
    except Exception as exc:
        await websocket.send_text(_json_error(str(exc)))
    return True


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    config = websocket.app.state.tracker_config
    default_instruction = websocket.app.state.default_instruction

    processor: RemoteFrameProcessor | None = None
    try:
        first_message = await websocket.receive_text()
        try:
            hello = json.loads(first_message)
        except json.JSONDecodeError:
            await websocket.send_text(_json_error("First message must be JSON"))
            return
        if hello.get("type") != "start":
            await websocket.send_text(_json_error("First message must have type=start"))
            return

        instruction = str(hello.get("instruction") or default_instruction).strip()
        if not instruction:
            await websocket.send_text(_json_error("No instruction provided"))
            return

        return_overlay = bool(hello.get("return_overlay", True))
        overlay_quality = int(hello.get("overlay_quality", 75))
        processor = RemoteFrameProcessor(config, instruction)
        processor.start()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "ack",
                    "message": "remote processor ready",
                    "return_overlay": return_overlay,
                },
                ensure_ascii=False,
            )
        )

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_text(_json_error("Control message is not valid JSON"))
                    continue
                if not await _handle_control(websocket, processor, payload):
                    break
                continue

            data = message.get("bytes")
            if data is None:
                continue

            frame = _decode_jpeg(data)
            if frame is None:
                await websocket.send_text(_json_error("Could not decode JPEG frame"))
                continue

            result, overlay = processor.process_frame(frame, return_overlay=return_overlay)
            if overlay is not None:
                result["overlay_jpeg_b64"] = _encode_jpeg_b64(overlay, overlay_quality)
            await websocket.send_text(json.dumps(result, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    finally:
        if processor is not None:
            processor.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Linux/RTX 3090 WebSocket inference server."
    )
    parser.add_argument(
        "--config",
        default="configs/linux_3090_remote.yaml",
        help="Server-side YAML configuration path",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--command",
        default="",
        help="Default target instruction if the camera client does not send one",
    )
    args = parser.parse_args()

    config_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    app.state.tracker_config = load_config(config_path)
    app.state.default_instruction = args.command

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
