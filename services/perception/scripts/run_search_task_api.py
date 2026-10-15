from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.search_task_planner import (  # noqa: E402
    InstructionCoordinator,
    SearchTaskPlanner,
)
from qwen_grounded_tracker.semantic_task_planner import QwenTransformersBackend  # noqa: E402


app = FastAPI(title="Natural Language Object Search Planner")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "natural_language_object_search_planner",
        "hardware_commands_sent": False,
        "state": app.state.coordinator.snapshot(),
    }


@app.post("/command")
async def command(request: Request) -> JSONResponse:
    payload = await request.json()
    instruction = str(payload.get("instruction") or "").strip() if isinstance(payload, dict) else ""
    try:
        decision = app.state.coordinator.submit(instruction)
        return JSONResponse({"ok": True, "decision": decision})
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/state")
def state() -> dict[str, Any]:
    return {"ok": True, "state": app.state.coordinator.snapshot()}


@app.post("/task/complete")
async def complete(request: Request) -> JSONResponse:
    payload = await request.json()
    result = app.state.coordinator.complete(
        str(payload.get("task_id") or ""), str(payload.get("result") or "completed")
    )
    return JSONResponse(result, status_code=200 if result.get("ok") else 409)


@app.post("/idle/annotation/request")
async def idle_annotation(request: Request) -> JSONResponse:
    payload = await request.json()
    result = app.state.coordinator.request_idle_annotation(str(payload.get("dedup_key") or ""))
    return JSONResponse(result, status_code=200 if result.get("accepted") else 409)


@app.post("/idle/annotation/finish")
async def finish_idle_annotation(request: Request) -> JSONResponse:
    payload = await request.json()
    result = app.state.coordinator.finish_idle_annotation(
        str(payload.get("task_id") or ""), bool(payload.get("committed", False))
    )
    return JSONResponse(result, status_code=200 if result.get("ok") else 409)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-reviewed object-search planning API; no motor output.")
    parser.add_argument("--semantic-map", required=True)
    parser.add_argument("--yolo-model", default="models/yolo/yolo11n-seg.pt")
    parser.add_argument("--backend", choices=("qwen", "rules"), default="qwen")
    parser.add_argument("--model-path", default="models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()
    semantic = json.loads(Path(args.semantic_map).expanduser().read_text(encoding="utf-8"))
    from ultralytics import YOLO

    yolo = YOLO(str(Path(args.yolo_model).expanduser()))
    names = getattr(yolo, "names", {})
    classes = list(names.values()) if isinstance(names, dict) else list(names)
    backend = None
    if args.backend == "qwen":
        backend = QwenTransformersBackend(str(Path(args.model_path).expanduser()), device=args.device)
    app.state.coordinator = InstructionCoordinator(SearchTaskPlanner(classes, backend), semantic)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
