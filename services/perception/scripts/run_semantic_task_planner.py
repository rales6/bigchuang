from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qwen_grounded_tracker.semantic_task_planner import (
    QwenTransformersBackend,
    SemanticTaskPlanner,
    plan_from_dict,
)
from qwen_grounded_tracker.scene_change_memory import SceneChangeMemory
from qwen_grounded_tracker.task_execution import (
    HttpRobotBridge,
    PlanExecutionEngine,
    SimulationRobotBridge,
)


app = FastAPI(title="Semantic Robot Task Planner")


@app.get("/health")
async def health() -> JSONResponse:
    try:
        bridge = app.state.execution_engine.bridge.health()
    except Exception as exc:
        bridge = {"ok": False, "error": str(exc)}
    return JSONResponse(
        {
            "status": "ok",
            "backend": app.state.backend_name,
            "bridge": bridge,
            "execution": app.state.execution_engine.snapshot(),
            "suspicious_location_count": len(app.state.change_memory.locations),
        }
    )


@app.post("/plan")
async def plan(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        instruction = str(payload.get("instruction") or "").strip()
        semantic_map: dict[str, Any] | None = payload.get("semantic_map")
        if not isinstance(semantic_map, dict):
            semantic_map = app.state.default_map
        semantic_map = app.state.change_memory.overlay(semantic_map)
        result = app.state.planner.plan(instruction, semantic_map)
        app.state.last_plan = result
        return JSONResponse({"ok": True, "plan": result.to_dict()})
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.get("/scene-changes")
async def scene_changes() -> JSONResponse:
    return JSONResponse({"ok": True, "overlay": app.state.change_memory.snapshot()})


@app.post("/scene-changes/record")
async def record_scene_change(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        location = app.state.change_memory.record(
            change_type=str(payload.get("change_type") or ""),
            expected_item_id=payload.get("expected_item_id"),
            expected_query=str(payload.get("expected_query") or ""),
            map_xy=payload.get("map_xy"),
            grid_xy=payload.get("grid_xy"),
            observed_item_id=payload.get("observed_item_id"),
            confidence=float(payload.get("confidence") or 0.0),
            evidence=dict(payload.get("evidence") or {}),
        )
        return JSONResponse({"ok": True, "suspicious_location": location})
    except (ValueError, TypeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/execute/start")
async def execute_start(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        raw_plan = payload.get("plan")
        selected = plan_from_dict(raw_plan) if isinstance(raw_plan, dict) else app.state.last_plan
        if selected is None:
            raise ValueError("No plan supplied and no previously generated plan is available")
        engine: PlanExecutionEngine = app.state.execution_engine
        engine.load(selected)
        state = engine.start()
        auto_run = bool(payload.get("auto_run", False))
        if auto_run:
            if engine.bridge.hardware_enabled:
                raise ValueError("auto_run is forbidden in hardware mode; call /execute/tick explicitly")
            state = engine.run_until_terminal()
        return JSONResponse({"ok": True, "execution": state})
    except (ValueError, RuntimeError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/execute/tick")
async def execute_tick() -> JSONResponse:
    try:
        return JSONResponse({"ok": True, "execution": app.state.execution_engine.tick()})
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)


@app.get("/execute/state")
async def execute_state() -> JSONResponse:
    return JSONResponse({"ok": True, "execution": app.state.execution_engine.snapshot()})


@app.post("/execute/stop")
async def execute_stop(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    reason = str(payload.get("reason") or "operator_stop") if isinstance(payload, dict) else "operator_stop"
    try:
        state = app.state.execution_engine.cancel(reason)
        return JSONResponse({"ok": True, "execution": state})
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 3090 semantic task planning API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--backend", choices=("rules", "qwen"), default="qwen")
    parser.add_argument("--model-path", default="models/qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-map", default="")
    parser.add_argument(
        "--scene-change-memory",
        default="outputs/scene_change_overlay.json",
        help="independent suspicious-location overlay; never overwrites the semantic map",
    )
    parser.add_argument("--robot-url", default="", help="Raspberry Pi task gateway base URL")
    parser.add_argument(
        "--robot-timeout-s",
        type=float,
        default=30.0,
        help="Pi gateway timeout; manipulation verification may run Qwen on a stopped robot",
    )
    parser.add_argument(
        "--enable-hardware",
        action="store_true",
        help="allow the HTTP bridge to request hardware actions; requires ROBOT_TASK_HARDWARE_TOKEN",
    )
    args = parser.parse_args()

    backend = None
    if args.backend == "qwen":
        model_path = Path(args.model_path).expanduser()
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        backend = QwenTransformersBackend(str(model_path), device=args.device)
    app.state.backend_name = args.backend
    app.state.planner = SemanticTaskPlanner(backend)
    app.state.last_plan = None
    app.state.default_map = None
    if args.semantic_map:
        map_path = Path(args.semantic_map)
        if not map_path.is_absolute():
            map_path = ROOT / map_path
        app.state.default_map = json.loads(map_path.read_text(encoding="utf-8"))
    change_path = Path(args.scene_change_memory)
    if not change_path.is_absolute():
        change_path = ROOT / change_path
    app.state.change_memory = SceneChangeMemory(
        change_path,
        map_id=str((app.state.default_map or {}).get("map_id") or "unknown"),
    )

    if args.robot_url:
        safety_token = os.environ.get("ROBOT_TASK_HARDWARE_TOKEN", "")
        bridge = HttpRobotBridge(
            args.robot_url,
            hardware_enabled=args.enable_hardware,
            safety_token=safety_token,
            timeout_s=args.robot_timeout_s,
        )
    else:
        if args.enable_hardware:
            parser.error("--enable-hardware requires --robot-url")
        bridge = SimulationRobotBridge()
    app.state.execution_engine = PlanExecutionEngine(bridge)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
