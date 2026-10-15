"""Unified GPT-style web application hosted on the RTX 3090."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .history import HistoryStore
from .planner import AgentPlanner
from .robot_client import RobotClient


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
DATA = Path(os.environ.get("FINAL_WORK_DATA_DIR", str(ROOT / "data")))

app = FastAPI(title="Final Work Robot Agent", version="1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.state.history = HistoryStore(DATA / "history.sqlite3")
app.state.planner = AgentPlanner()
app.state.robot = RobotClient()
# DeepSeek Flash is strictly user-device managed.  Scrub any value that may
# have been written by an older build before this policy was introduced.
_legacy_credentials = app.state.history.get_setting("api_credentials", {})
if isinstance(_legacy_credentials, dict) and _legacy_credentials.get("deepseek_flash_api"):
    _legacy_credentials = dict(_legacy_credentials)
    _legacy_credentials.pop("deepseek_flash_api", None)
    app.state.history.set_setting("api_credentials", _legacy_credentials)
try:
    app.state.history.path.chmod(0o600)
except OSError:
    pass

PROVIDERS = {
    "local_qwen": {"label": "本地共享 Qwen", "requires_key": False},
    "qwen_api": {"label": "Qwen API", "requires_key": True},
    "deepseek_api": {"label": "DeepSeek API", "requires_key": True},
    "deepseek_flash_api": {
        "label": "DeepSeek V4.1 Flash API",
        "requires_key": True,
        "default_model": "deepseek-flash",
        "client_managed_key": True,
    },
    "gpt_api": {"label": "GPT API", "requires_key": True},
}
LANGUAGES = {"zh-CN", "en-US"}


def _camera_order_settings() -> dict[str, Any]:
    value = app.state.history.get_setting("camera_order", {})
    value = value if isinstance(value, dict) else {}
    return {
        "swapped": bool(value.get("swapped", False)),
        "source": str(value.get("source") or "default"),
        "confidence": float(value.get("confidence") or 0.0),
        "reason": str(value.get("reason") or ""),
        "updated_at": int(value.get("updated_at") or 0),
    }


def _camera_order_inference(left_jpeg: bytes, right_jpeg: bytes) -> dict[str, Any]:
    body = json.dumps({
        "left_image_b64": base64.b64encode(left_jpeg).decode("ascii"),
        "right_image_b64": base64.b64encode(right_jpeg).decode("ascii"),
    }).encode("utf-8")
    request = urlrequest.Request(
        os.environ.get("FINAL_WORK_QWEN_HUB_URL", "http://127.0.0.1:8010").rstrip("/") + "/agent/camera-order",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urlrequest.urlopen(request, timeout=70) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict) or not value.get("ok"):
        raise RuntimeError(str(value.get("error") if isinstance(value, dict) else value))
    return value


def _agent_preferences() -> dict[str, Any]:
    stored = app.state.history.get_setting("agent_preferences", {})
    stored = stored if isinstance(stored, dict) else {}
    provider = str(stored.get("provider") or "local_qwen")
    if provider not in PROVIDERS:
        provider = "local_qwen"
    language = str(stored.get("language") or "zh-CN")
    if language not in LANGUAGES:
        language = "zh-CN"
    models = stored.get("models") if isinstance(stored.get("models"), dict) else {}
    return {"provider": provider, "language": language, "models": models}


def _api_credentials() -> dict[str, str]:
    stored = app.state.history.get_setting("api_credentials", {})
    values = stored if isinstance(stored, dict) else {}
    env_keys = {
        "qwen_api": os.environ.get("FINAL_WORK_QWEN_API_KEY", ""),
        "deepseek_api": os.environ.get("FINAL_WORK_DEEPSEEK_API_KEY", ""),
        "gpt_api": os.environ.get("FINAL_WORK_GPT_API_KEY", ""),
    }
    return {
        name: "" if name == "deepseek_flash_api" else str(values.get(name) or env_keys.get(name) or "")
        for name in PROVIDERS if name != "local_qwen"
    }


def _public_settings() -> dict[str, Any]:
    preferences = _agent_preferences()
    credentials = _api_credentials()
    providers = []
    for name, metadata in PROVIDERS.items():
        providers.append({
            "id": name,
            **metadata,
            "configured": not metadata["requires_key"] or bool(credentials.get(name)),
            "model": str(preferences["models"].get(name) or metadata.get("default_model") or ""),
        })
    return {**preferences, "providers": providers}


def _clean_text(value: Any, name: str, limit: int = 1000) -> str:
    text = str(value or "").strip()
    if not text:
        raise HTTPException(400, f"{name}不能为空")
    if len(text) > limit:
        raise HTTPException(400, f"{name}不能超过 {limit} 个字符")
    return text


def _assistant_text(plan: dict[str, Any]) -> str:
    summary = str(plan.get("summary") or "计划已生成")
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    warnings = [str(item) for item in plan.get("warnings", []) if str(item)]
    lines = [summary, f"我生成了 {len(steps)} 个受控步骤。请先检查计划，再确认执行。"]
    if warnings:
        lines.append("注意：" + "；".join(warnings[:3]))
    if any(step.get("action") in {"patrol_search", "navigate_to", "return_home", "locate_destination"} for step in steps):
        lines.append("地图标注与导航已接入 car1.0 冻结接口；安全模式执行仿真，实车模式仍需逐步确认。")
    return "\n\n".join(lines)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    try:
        robot = await asyncio.to_thread(app.state.robot.request, "/health")
    except Exception as exc:
        robot = {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "service": "final_work_agent",
        "planner": _agent_preferences()["provider"],
        "robot": robot,
    }


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return {"ok": True, "settings": _public_settings()}


@app.patch("/api/settings")
async def update_settings(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "设置必须是 JSON 对象")
    preferences = _agent_preferences()
    provider = str(payload.get("provider") or preferences["provider"])
    language = str(payload.get("language") or preferences["language"])
    if provider not in PROVIDERS:
        raise HTTPException(400, "不支持的规划模型服务")
    if language not in LANGUAGES:
        raise HTTPException(400, "不支持的界面语言")
    models = dict(preferences["models"])
    if provider != "local_qwen" and "model" in payload:
        model = _clean_text(payload.get("model"), "模型 ID", 100)
        models[provider] = model
    app.state.history.set_setting(
        "agent_preferences",
        {"provider": provider, "language": language, "models": models},
    )
    credentials = _api_credentials()
    if provider == "deepseek_flash_api":
        # This provider's secret is browser-managed and must never be written
        # to the 3090 history database.
        credentials[provider] = ""
    elif bool(payload.get("clear_api_key")) and provider != "local_qwen":
        credentials[provider] = ""
    elif provider != "local_qwen" and str(payload.get("api_key") or "").strip():
        credentials[provider] = _clean_text(payload.get("api_key"), "API 密钥", 500)
    app.state.history.set_setting("api_credentials", credentials)
    return {"ok": True, "settings": _public_settings()}


@app.get("/api/conversations")
def conversations() -> dict[str, Any]:
    return {"ok": True, "conversations": app.state.history.list_conversations()}


@app.post("/api/conversations")
async def create_conversation(request: Request) -> dict[str, Any]:
    payload = await request.json()
    title = str(payload.get("title") or "新任务") if isinstance(payload, dict) else "新任务"
    return {"ok": True, "conversation": app.state.history.create_conversation(title)}


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        value = app.state.history.rename_conversation(conversation_id, str(payload.get("title") or ""))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "conversation": value}


@app.get("/api/conversations/{conversation_id}")
def conversation_detail(conversation_id: str) -> dict[str, Any]:
    conversation = app.state.history.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(404, "会话不存在")
    return {
        "ok": True,
        "conversation": conversation,
        "messages": app.state.history.list_messages(conversation_id),
        "tasks": app.state.history.list_tasks(conversation_id),
    }


@app.post("/api/chat")
async def chat(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "请求必须是 JSON 对象")
    conversation_id = str(payload.get("conversation_id") or "")
    if not conversation_id:
        conversation = app.state.history.create_conversation("新任务")
        conversation_id = str(conversation["id"])
    if app.state.history.get_conversation(conversation_id) is None:
        raise HTTPException(404, "会话不存在")
    instruction = _clean_text(payload.get("message"), "消息")
    user_message = app.state.history.add_message(conversation_id, "user", instruction)
    existing = app.state.history.list_messages(conversation_id)
    if len(existing) == 1:
        app.state.history.rename_conversation(conversation_id, instruction[:28])

    try:
        semantic_map = await asyncio.to_thread(app.state.robot.request, "/semantic-map")
    except Exception:
        semantic_map = {"objects": [], "landmarks": [], "_status": "robot_unavailable"}
    preferences = _agent_preferences()
    provider = preferences["provider"]
    if provider == "deepseek_flash_api":
        raise HTTPException(400, "DeepSeek Flash 必须从使用者设备直连，密钥不会发送到 3090")
    planner_config = {
        "provider": provider,
        "model": str(preferences["models"].get(provider) or ""),
        "api_key": _api_credentials().get(provider, ""),
    }
    plan = await asyncio.to_thread(
        app.state.planner.plan, instruction, semantic_map, planner_config
    )
    task = app.state.history.create_task(conversation_id, user_message["id"], instruction, plan)
    assistant = app.state.history.add_message(
        conversation_id,
        "assistant",
        _assistant_text(plan),
        {
            "task_id": task["id"],
            "plan_source": plan.get("source"),
            "planner_provider": provider,
        },
    )
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "user_message": user_message,
        "assistant_message": assistant,
        "task": task,
    }


@app.post("/api/planning/client-context")
async def client_planning_context(request: Request) -> dict[str, Any]:
    """Return a constrained prompt; this endpoint never accepts API secrets."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "请求必须是 JSON 对象")
    instruction = _clean_text(payload.get("message"), "消息")
    preferences = _agent_preferences()
    if preferences["provider"] != "deepseek_flash_api":
        raise HTTPException(409, "当前未选择 DeepSeek Flash")
    try:
        semantic_map = await asyncio.to_thread(app.state.robot.request, "/semantic-map")
    except Exception:
        semantic_map = {"objects": [], "landmarks": [], "_status": "robot_unavailable"}
    prompt = await asyncio.to_thread(app.state.planner.client_prompt, instruction, semantic_map)
    return {
        "ok": True,
        "prompt": prompt,
        "model": str(preferences["models"].get("deepseek_flash_api") or "deepseek-flash"),
    }


@app.post("/api/chat/client-plan")
async def client_chat_plan(request: Request) -> dict[str, Any]:
    """Compile model output created on the user device; no API key is accepted."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "请求必须是 JSON 对象")
    if "api_key" in payload or "client_api_key" in payload:
        raise HTTPException(400, "此接口禁止接收 API 密钥")
    preferences = _agent_preferences()
    if preferences["provider"] != "deepseek_flash_api":
        raise HTTPException(409, "当前未选择 DeepSeek Flash")
    conversation_id = str(payload.get("conversation_id") or "")
    if not conversation_id:
        conversation_id = str(app.state.history.create_conversation("新任务")["id"])
    if app.state.history.get_conversation(conversation_id) is None:
        raise HTTPException(404, "会话不存在")
    instruction = _clean_text(payload.get("message"), "消息")
    raw = _clean_text(payload.get("model_output"), "模型输出", 8000)
    try:
        plan = await asyncio.to_thread(
            app.state.planner.compile_client_output,
            instruction,
            raw,
            "deepseek_flash_api",
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"DeepSeek 输出无法通过计划校验：{exc}") from exc
    user_message = app.state.history.add_message(conversation_id, "user", instruction)
    existing = app.state.history.list_messages(conversation_id)
    if len(existing) == 1:
        app.state.history.rename_conversation(conversation_id, instruction[:28])
    task = app.state.history.create_task(conversation_id, user_message["id"], instruction, plan)
    assistant = app.state.history.add_message(
        conversation_id,
        "assistant",
        _assistant_text(plan),
        {"task_id": task["id"], "plan_source": plan.get("source"), "planner_provider": "deepseek_flash_api"},
    )
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "user_message": user_message,
        "assistant_message": assistant,
        "task": task,
    }


@app.get("/api/tasks/{task_id}")
def task_detail(task_id: str) -> dict[str, Any]:
    task = app.state.history.get_task(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    return {"ok": True, "task": task}


@app.post("/api/tasks/{task_id}/confirm")
def confirm_task(task_id: str) -> dict[str, Any]:
    task = app.state.history.get_task(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    if task["status"] not in {"draft", "paused", "blocked"}:
        raise HTTPException(409, f"任务状态不能确认：{task['status']}")
    execution = dict(task["execution"])
    events = list(execution.get("events", []))
    events.append({"at": int(time.time() * 1000), "type": "operator_confirmed"})
    execution["events"] = events
    return {"ok": True, "task": app.state.history.update_task(task_id, status="confirmed", execution=execution)}


@app.post("/api/tasks/{task_id}/steps/{step_id}/execute")
async def execute_step(task_id: str, step_id: str) -> dict[str, Any]:
    task = app.state.history.get_task(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    if task["status"] not in {"confirmed", "running", "blocked"}:
        raise HTTPException(409, "请先确认任务")
    steps = task["plan"].get("steps", [])
    step = next((item for item in steps if str(item.get("id")) == step_id), None)
    if step is None:
        raise HTTPException(404, "步骤不存在")
    execution = dict(task["execution"])
    step_states = dict(execution.get("steps", {}))
    dependencies = [str(item) for item in step.get("depends_on", [])]
    incomplete = [item for item in dependencies if step_states.get(item, {}).get("status") != "completed"]
    if incomplete:
        raise HTTPException(409, f"前置步骤尚未完成：{', '.join(incomplete)}")
    try:
        result = await asyncio.to_thread(app.state.robot.dispatch_step, step, task["instruction"])
    except urlerror.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"error": str(exc)}
        result = {
            "status": "blocked" if exc.code in {501, 503} else "failed",
            "detail": detail,
        }
    except Exception as exc:
        result = {"status": "failed", "detail": {"error": f"{type(exc).__name__}: {exc}"}}
    step_states[step_id] = {**result, "updated_at": int(time.time() * 1000)}
    execution["steps"] = step_states
    execution["active_step"] = step_id if result["status"] in {"running", "awaiting_verification"} else None
    events = list(execution.get("events", []))
    events.append({"at": int(time.time() * 1000), "type": "step_dispatch", "step_id": step_id, "status": result["status"]})
    execution["events"] = events
    statuses = {item.get("status") for item in step_states.values()}
    task_status = "blocked" if "blocked" in statuses else "running"
    if all(step_states.get(str(item.get("id")), {}).get("status") == "completed" for item in steps):
        task_status = "completed"
    updated = app.state.history.update_task(task_id, status=task_status, execution=execution)
    return {"ok": result["status"] not in {"failed", "blocked"}, "result": result, "task": updated}


@app.post("/api/tasks/{task_id}/steps/{step_id}/complete")
def complete_step(task_id: str, step_id: str) -> dict[str, Any]:
    """Operator acknowledgement for a dispatched/visually verified step."""
    task = app.state.history.get_task(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    if not any(str(item.get("id")) == step_id for item in task["plan"].get("steps", [])):
        raise HTTPException(404, "步骤不存在")
    execution = dict(task["execution"])
    states = dict(execution.get("steps", {}))
    states[step_id] = {"status": "completed", "updated_at": int(time.time() * 1000), "operator_ack": True}
    execution["steps"] = states
    execution["active_step"] = None
    status = "completed" if all(
        states.get(str(item.get("id")), {}).get("status") == "completed"
        for item in task["plan"].get("steps", [])
    ) else "running"
    return {"ok": True, "task": app.state.history.update_task(task_id, status=status, execution=execution)}


@app.post("/api/robot/emergency-stop")
async def emergency_stop() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(app.state.robot.request, "/actions/emergency-stop", {})
    except Exception as exc:
        raise HTTPException(502, f"急停请求失败：{exc}") from exc


@app.post("/api/robot/preview/start")
async def preview_start() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(app.state.robot.request, "/actions/preview/start", {})
    except Exception as exc:
        raise HTTPException(502, f"双目预览启动失败：{exc}") from exc


@app.post("/api/robot/preview/stop")
async def preview_stop() -> dict[str, Any]:
    try:
        return await asyncio.to_thread(app.state.robot.request, "/actions/preview/stop", {})
    except Exception as exc:
        raise HTTPException(502, f"双目预览停止失败：{exc}") from exc


@app.post("/api/robot/mapping/{action}")
async def mapping_control(action: str) -> dict[str, Any]:
    if action not in {"start", "stop"}:
        raise HTTPException(404, "建图动作不存在")
    try:
        return await asyncio.to_thread(
            app.state.robot.request, f"/actions/mapping/{action}", {}
        )
    except Exception as exc:
        raise HTTPException(502, f"建图{action}失败：{exc}") from exc


@app.get("/api/robot/state")
async def robot_state() -> JSONResponse:
    try:
        return JSONResponse(await asyncio.to_thread(app.state.robot.request, "/state"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.get("/api/robot/logs")
async def robot_logs() -> JSONResponse:
    try:
        return JSONResponse(await asyncio.to_thread(app.state.robot.request, "/logs/tail"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.get("/api/robot/map.png")
async def map_png() -> StreamingResponse:
    try:
        content_type, chunks = await asyncio.to_thread(app.state.robot.stream, "/map.png")
        return StreamingResponse(chunks, media_type=content_type, headers={"Cache-Control": "no-store"})
    except Exception as exc:
        raise HTTPException(503, f"地图暂不可用：{exc}") from exc


@app.get("/api/robot/map/layers")
async def map_layers() -> JSONResponse:
    try:
        return JSONResponse(await asyncio.to_thread(app.state.robot.request, "/map/layers"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc), "layers": []}, status_code=503)


@app.get("/api/robot/map/layer/{layer_id}.png")
async def map_layer_png(layer_id: str) -> StreamingResponse:
    if not re.fullmatch(r"[a-z0-9_]{1,64}", layer_id):
        raise HTTPException(404, "地图图层不存在")
    try:
        content_type, chunks = await asyncio.to_thread(
            app.state.robot.stream, f"/map/layer.png?id={layer_id}"
        )
        return StreamingResponse(chunks, media_type=content_type, headers={"Cache-Control": "no-store"})
    except Exception as exc:
        raise HTTPException(503, f"地图图层暂不可用：{exc}") from exc


@app.get("/api/robot/semantic-map")
async def semantic_map() -> JSONResponse:
    try:
        return JSONResponse(await asyncio.to_thread(app.state.robot.request, "/semantic-map"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.get("/api/robot/camera-order")
def camera_order() -> dict[str, Any]:
    return {"ok": True, "camera_order": _camera_order_settings()}


@app.post("/api/robot/camera-order/calibrate")
async def calibrate_camera_order() -> dict[str, Any]:
    try:
        try:
            await asyncio.to_thread(app.state.robot.request, "/actions/preview/start", {})
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code != 400 or "camera_preview" not in detail:
                raise
        await asyncio.sleep(1.5)
        left_jpeg, right_jpeg = await asyncio.gather(
            asyncio.to_thread(app.state.robot.capture_frame, "left"),
            asyncio.to_thread(app.state.robot.capture_frame, "right"),
        )
        inference = await asyncio.to_thread(_camera_order_inference, left_jpeg, right_jpeg)
        order = str(inference.get("order") or "uncertain")
        confidence = float(inference.get("confidence") or 0.0)
        applied = order in {"normal", "swapped"} and confidence >= 0.55
        if applied:
            value = {
                "swapped": order == "swapped",
                "source": str(inference.get("source") or "qwen"),
                "confidence": confidence,
                "reason": str(inference.get("reason") or "")[:300],
                "updated_at": int(time.time() * 1000),
            }
            app.state.history.set_setting("camera_order", value)
        else:
            value = _camera_order_settings()
        return {"ok": True, "applied": applied, "camera_order": value, "inference": inference}
    except Exception as exc:
        raise HTTPException(502, f"摄像头左右校准失败：{type(exc).__name__}: {exc}") from exc


@app.post("/api/robot/camera-order/swap")
def swap_camera_order() -> dict[str, Any]:
    current = _camera_order_settings()
    value = {
        "swapped": not current["swapped"],
        "source": "operator",
        "confidence": 1.0,
        "reason": "使用者手动交换左右画面",
        "updated_at": int(time.time() * 1000),
    }
    app.state.history.set_setting("camera_order", value)
    return {"ok": True, "camera_order": value}


@app.get("/api/robot/video/{camera}")
async def camera_stream(camera: str) -> StreamingResponse:
    if camera not in {"left", "right"}:
        raise HTTPException(404, "摄像头不存在")
    order = _camera_order_settings()
    source_camera = ({"left": "right", "right": "left"}[camera] if order["swapped"] else camera)
    try:
        content_type, chunks = await asyncio.to_thread(app.state.robot.stream, f"/video/{source_camera}")
        return StreamingResponse(chunks, media_type=content_type, headers={"Cache-Control": "no-store"})
    except Exception as exc:
        raise HTTPException(503, f"{camera} 摄像头暂不可用：{exc}") from exc


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="final_work 3090 web agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8030)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
