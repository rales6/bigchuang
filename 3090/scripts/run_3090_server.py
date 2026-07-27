from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.config import load_config
from qwen_grounded_tracker.remote.processor import RemoteFrameProcessor


app = FastAPI(title="Qwen Grounded Tracker 3090 Server")


class SharedRemoteSession:
    """树莓派上传帧、Windows 发控制时，共用这一份 3090 推理状态。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.processor_lock = threading.RLock()
        self.processor: RemoteFrameProcessor | None = None
        self.latest_result: dict[str, Any] = {
            "type": "result",
            "grounding_status": "waiting for Raspberry Pi camera",
            "track": {"visible": False, "status": "waiting for camera", "bbox": None},
            "guidance": {"direction": "STOP", "linear": 0.0, "angular": 0.0},
            "safety": {"blocked": True, "reasons": ["waiting for camera"]},
            "boundary_mode": "-",
            "emergency_stop": False,
        }
        self.latest_raw_jpeg_b64 = ""
        self.latest_raw_jpeg_bytes: bytes | None = None
        self.latest_raw_version = 0
        self.latest_overlay_jpeg_b64 = ""
        self.frame_size = {"width": 0, "height": 0}
        self.camera_connected = False
        self.last_frame_at = 0.0
        self.frame_count = 0
        self.processed_frame_count = 0
        self.camera_fps = 0.0
        self._fps_window_at = time.monotonic()
        self._fps_frames = 0
        self.pending_frame: np.ndarray | None = None
        self.worker_thread: threading.Thread | None = None
        self.pending_instruction = ""
        self.pending_emergency_stop = False
        self.return_overlay = True
        self.overlay_quality = 65
        self.overlay_every_n = 2

    def start_processor(self, config: dict[str, Any], instruction: str) -> None:
        with self.processor_lock:
            if self.processor is not None:
                self.processor.close()
            effective_instruction = instruction.strip() or self.pending_instruction.strip()
            safe_instruction = effective_instruction or "manual ROI"
            self.processor = RemoteFrameProcessor(config, safe_instruction)
            self.processor.start()
            if self.pending_emergency_stop:
                self.processor.set_emergency_stop(True)
            if not effective_instruction:
                self.processor.reset()
        with self.lock:
            self.pending_frame = None
            self.processed_frame_count = 0
            self.latest_result["grounding_status"] = (
                "camera connected; waiting for instruction"
                if not effective_instruction
                else "camera connected; waiting for first frame"
            )

    def close_processor(self) -> None:
        with self.processor_lock:
            if self.processor is not None:
                self.processor.close()
                self.processor = None
        with self.lock:
            self.camera_connected = False

    def handle_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        message_type = str(payload.get("type", ""))
        with self.lock:
            processor = self.processor
            if processor is None:
                if message_type == "set_instruction":
                    # Windows 可以先发送指令；树莓派摄像头连上后会用这条指令初始化。
                    self.pending_instruction = str(payload.get("instruction", "")).strip()
                    self.latest_result["grounding_status"] = (
                        "instruction queued; waiting for Raspberry Pi camera"
                    )
                    return {"type": "ack", "message": "instruction queued until camera connects"}
                if message_type == "emergency_stop":
                    self.pending_emergency_stop = bool(payload.get("enabled", True))
                    self.latest_result["emergency_stop"] = self.pending_emergency_stop
                    return {"type": "ack", "message": "emergency stop queued"}
                if message_type == "reset":
                    self.pending_instruction = ""
                    self.latest_result["grounding_status"] = "waiting for Raspberry Pi camera"
                    return {"type": "ack", "message": "pending target reset"}
                return {"type": "error", "message": "camera processor is not ready"}

        with self.processor_lock:
            if message_type == "set_instruction":
                instruction = str(payload.get("instruction", "")).strip()
                processor.set_instruction(instruction)
                with self.lock:
                    self.pending_instruction = instruction
                return {"type": "ack", "message": "instruction updated"}
            if message_type == "reground":
                processor.request_reground()
                return {"type": "ack", "message": "re-ground requested"}
            if message_type == "reset":
                processor.reset()
                return {"type": "ack", "message": "target reset"}
            if message_type == "emergency_stop":
                processor.set_emergency_stop(bool(payload.get("enabled", True)))
                with self.lock:
                    self.pending_emergency_stop = processor.emergency_stop
                return {
                    "type": "ack",
                    "message": "emergency stop updated",
                    "enabled": processor.emergency_stop,
                }
            if message_type == "toggle_boundary":
                mode = processor.toggle_boundary()
                return {"type": "ack", "message": "boundary mode toggled", "mode": mode}
            if message_type == "manual_roi":
                x, y, width, height = payload.get("bbox_xywh", [0, 0, 0, 0])
                processor.set_manual_roi_xywh(float(x), float(y), float(width), float(height))
                return {"type": "ack", "message": "manual ROI queued"}
        return {"type": "error", "message": f"Unknown control message type: {message_type}"}

    def process_camera_frame(self, frame: np.ndarray) -> dict[str, Any]:
        with self.lock:
            if self.processor is None:
                raise RuntimeError("camera processor is not ready")
            self.frame_count += 1
            self._fps_frames += 1
            now = time.monotonic()
            elapsed = now - self._fps_window_at
            if elapsed >= 1.0:
                self.camera_fps = self._fps_frames / elapsed
                self._fps_frames = 0
                self._fps_window_at = now

            should_overlay = self.return_overlay and self.frame_count % self.overlay_every_n == 0
            result, overlay = self.processor.process_frame(frame, return_overlay=should_overlay)
            if overlay is not None:
                result["overlay_jpeg_b64"] = _encode_jpeg_b64(overlay, self.overlay_quality)
                self.latest_overlay_jpeg_b64 = result["overlay_jpeg_b64"]
            elif self.latest_overlay_jpeg_b64:
                result["overlay_jpeg_b64"] = self.latest_overlay_jpeg_b64
            self.latest_result = result
            self.frame_size = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
            self.last_frame_at = time.time()
            self.camera_connected = True
            return dict(result)

    def submit_camera_frame(self, frame: np.ndarray, jpeg_bytes: bytes) -> dict[str, Any]:
        with self.lock:
            if self.processor is None:
                raise RuntimeError("camera processor is not ready")
            self.frame_count += 1
            self._fps_frames += 1
            now = time.monotonic()
            elapsed = now - self._fps_window_at
            if elapsed >= 1.0:
                self.camera_fps = self._fps_frames / elapsed
                self._fps_frames = 0
                self._fps_window_at = now

            self.latest_raw_jpeg_bytes = bytes(jpeg_bytes)
            self.latest_raw_version += 1
            self.frame_size = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
            self.last_frame_at = time.time()
            self.camera_connected = True
            # 只保留最新帧，后台处理慢时主动丢弃旧帧，降低控制延迟。
            self.pending_frame = frame
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.worker_thread = threading.Thread(
                    target=self._process_pending_frames,
                    name="camera-processing-worker",
                    daemon=True,
                )
                self.worker_thread.start()
            return self.state()

    def _process_pending_frames(self) -> None:
        while True:
            with self.lock:
                frame = self.pending_frame
                self.pending_frame = None
                processor = self.processor
            if frame is None or processor is None:
                return

            with self.processor_lock:
                self.processed_frame_count += 1
                should_overlay = (
                    self.return_overlay
                    and self.processed_frame_count % self.overlay_every_n == 0
                )
                result, overlay = processor.process_frame(frame, return_overlay=should_overlay)

            if overlay is not None:
                result["overlay_jpeg_b64"] = _encode_jpeg_b64(overlay, self.overlay_quality)
            with self.lock:
                if overlay is not None:
                    self.latest_overlay_jpeg_b64 = result["overlay_jpeg_b64"]
                self.latest_result = result

    def latest_raw_frame(self) -> tuple[int, bytes | None]:
        with self.lock:
            frame = None if self.latest_raw_jpeg_bytes is None else bytes(self.latest_raw_jpeg_bytes)
            return self.latest_raw_version, frame

    def state(self, include_image: bool = False) -> dict[str, Any]:
        with self.lock:
            result = dict(self.latest_result)
            processor = self.processor
            frame_size = dict(self.frame_size)
            if not include_image:
                result.pop("display_jpeg_b64", None)
                result.pop("overlay_jpeg_b64", None)
            if include_image and self.latest_raw_jpeg_bytes:
                result["display_jpeg_b64"] = base64.b64encode(self.latest_raw_jpeg_bytes).decode("ascii")
            if include_image and self.latest_overlay_jpeg_b64:
                result["overlay_jpeg_b64"] = self.latest_overlay_jpeg_b64
            result["camera"] = {
                "connected": self.camera_connected,
                "last_frame_at": self.last_frame_at,
                "fps": round(self.camera_fps, 1),
                "frame": dict(self.frame_size),
            }
        if processor is not None:
            try:
                width = int(frame_size.get("width") or 0)
                height = int(frame_size.get("height") or 0)
                if width > 0 and height > 0:
                    result.update(processor.current_prediction_payload(width, height))
            except Exception as exc:
                result["prediction"] = {
                    "enabled": False,
                    "active": False,
                    "age_seconds": 0.0,
                    "reason": f"prediction refresh failed: {exc}",
                }
        return result


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


def _mjpeg_stream(session: SharedRemoteSession):
    last_version = -1
    while True:
        version, frame = session.latest_raw_frame()
        if frame is None or version == last_version:
            time.sleep(0.005)
            continue
        last_version = version
        yield b"--frame\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield b"Cache-Control: no-store\r\n"
        yield f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
        yield frame
        yield b"\r\n"


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


@app.get("/state")
async def state(include_image: bool = False) -> JSONResponse:
    session: SharedRemoteSession = app.state.shared_session
    return JSONResponse(session.state(include_image=include_image))


@app.get("/stream.mjpg")
async def stream_mjpg() -> StreamingResponse:
    session: SharedRemoteSession = app.state.shared_session
    return StreamingResponse(
        _mjpeg_stream(session),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Age": "0",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/control")
async def control(request: Request) -> JSONResponse:
    session: SharedRemoteSession = app.state.shared_session
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"type": "error", "message": "control payload must be JSON object"})
    return JSONResponse(session.handle_control(payload))


@app.websocket("/camera_ws")
async def camera_websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    config = websocket.app.state.tracker_config
    default_instruction = websocket.app.state.default_instruction
    session: SharedRemoteSession = websocket.app.state.shared_session

    try:
        first_message = await websocket.receive_text()
        try:
            hello = json.loads(first_message)
        except json.JSONDecodeError:
            await websocket.send_text(_json_error("First message must be JSON"))
            return
        if hello.get("type") not in {"camera_start", "start"}:
            await websocket.send_text(_json_error("First message must have type=camera_start"))
            return

        instruction = str(hello.get("instruction") or default_instruction).strip()
        session.return_overlay = bool(hello.get("return_overlay", True))
        session.overlay_quality = int(hello.get("overlay_quality", 65))
        session.overlay_every_n = max(1, int(hello.get("overlay_every_n", 2)))
        session.start_processor(config, instruction)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "ack",
                    "message": "camera stream ready",
                    "return_overlay": session.return_overlay,
                    "overlay_every_n": session.overlay_every_n,
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
                if payload.get("type") == "close":
                    await websocket.send_text(json.dumps({"type": "ack", "message": "closing"}))
                    break
                await websocket.send_text(json.dumps(session.handle_control(payload), ensure_ascii=False))
                continue

            data = message.get("bytes")
            if data is None:
                continue
            frame = _decode_jpeg(data)
            if frame is None:
                await websocket.send_text(_json_error("Could not decode JPEG frame"))
                continue
            result = session.submit_camera_frame(frame, data)
            await websocket.send_text(json.dumps(result, ensure_ascii=False))
    except WebSocketDisconnect:
        pass
    finally:
        session.camera_connected = False


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
        overlay_every_n = max(1, int(hello.get("overlay_every_n", 1)))
        processor = RemoteFrameProcessor(config, instruction)
        processor.start()
        await websocket.send_text(
            json.dumps(
                {
                    "type": "ack",
                    "message": "remote processor ready",
                    "return_overlay": return_overlay,
                    "overlay_every_n": overlay_every_n,
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

            should_return_overlay = return_overlay and (
                (processor.frame_count + 1) % overlay_every_n == 0
            )
            result, overlay = processor.process_frame(frame, return_overlay=should_return_overlay)
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
    app.state.shared_session = SharedRemoteSession()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
