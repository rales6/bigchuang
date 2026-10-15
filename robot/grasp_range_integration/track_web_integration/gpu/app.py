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
from urllib import parse as urlparse
from urllib import request as urlrequest

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from .history import HistoryStore
from .planner import AgentPlanner
from .plan_adaptation import adapt_plan_to_live_view, normalize_web_plan
from .robot_client import RobotClient
from .task_recording import prepare_recording, start_recording, finish_recording, persist_finished_recording
from pursuit.camera_contract import preview_source
from pursuit.pursuit_config import PursuitConfig
from pursuit.forward_reference import reference_payload


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"
DATA = Path(os.environ.get("FINAL_WORK_DATA_DIR", str(ROOT / "data")))
PURSUIT_STREAM_URL = os.environ.get(
    "FINAL_WORK_PURSUIT_STREAM_URL", "http://127.0.0.1:8021"
).rstrip("/")

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
        # Installed inward-facing pair: video2 is physical left, video0 right.
        # Shared by display and pursuit dispatch; model training order is fixed.
        "swapped": bool(value.get("swapped", True)),
        "source": str(value.get("source") or "physical_mount"),
        "confidence": float(value.get("confidence") or 0.0),
        "reason": str(value.get("reason") or ""),
        "updated_at": int(value.get("updated_at") or 0),
    }


# Resolve the saved operator choice at dispatch time, not at server startup.
app.state.robot.camera_order_provider = _camera_order_settings


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
    lines = [summary, f"共 {len(steps)} 步。检查计划后确认一次，系统会自动推进；夹取实物仍需人工核验。"]
    if warnings:
        lines.append("注意：" + "；".join(warnings[:3]))
    if any(step.get("action") in {"patrol_search", "navigate_to", "return_home", "locate_destination"} for step in steps):
        lines.append("地图标注与导航已接入 car1.0；安全模式执行仿真，实车模式仅在确认计划后执行。")
    return "\n\n".join(lines)


async def _adapt_to_current_view(plan: dict[str, Any]) -> dict[str, Any]:
    plan = normalize_web_plan(plan)
    if plan.get("intent") not in {"pick", "pick_and_place", "search"}:
        return plan
    steps = plan.get("steps") or []
    # An explicit no-motion/current-view contract is already safe.  Its
    # readiness check at execution time must remain authoritative.
    if steps and steps[0].get("action") == "prepare_stationary_grasp":
        return plan
    target = ""
    for action in ("model_grasp", "align_for_grasp", "locate_object", "patrol_search"):
        step = next((item for item in steps if item.get("action") == action), None)
        if step:
            values = step.get("target") or {}
            target = str(values.get("target_query") or values.get("query") or "").strip()
            if target:
                break
    if not target:
        return plan
    try:
        probe = await asyncio.to_thread(app.state.robot.probe_current_target, target)
    except Exception as exc:
        probe = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    if probe.get("ble_ready") is False and plan.get("intent") in {"pick", "pick_and_place"}:
        plan["warnings"] = list(plan.get("warnings") or []) + [
            "当前机械臂 BLE 未连接；计划可以查看，但抓取前必须重新连上 ESP32"
        ]
    if probe.get("status") != "visible_both":
        plan["live_view_probe"] = {"status": probe.get("status", "unavailable"),
                                   "reason": str(probe.get("reason") or "")[:200],
                                   "ble_ready": probe.get("ble_ready")}
        return plan
    return adapt_plan_to_live_view(plan, probe)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/app-runtime.js")
def app_runtime() -> Response:
    """Serve the frontend from an uncached, build-specific URL."""
    return Response(
        (STATIC / "app.js").read_text(encoding='utf-8') +
        '\nimport("/static/empty-gripper.js?v=20261003-v1").catch(console.error);\n',
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


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
    plan = await _adapt_to_current_view(plan)
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


@app.post("/api/planning/client-event")
async def client_planning_event(request: Request) -> dict[str, Any]:
    """Record sanitized browser-side planner stages without accepting secrets."""
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(400, "请求必须是 JSON 对象")
    forbidden = {"api_key", "client_api_key", "authorization", "headers", "prompt", "model_output"}
    if forbidden & {str(key).lower() for key in payload}:
        raise HTTPException(400, "客户端诊断接口禁止接收密钥、请求头或模型原文")

    def safe(value: Any, limit: int) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()[:limit]
        # Defensive redaction in case a browser/network error unexpectedly
        # echoes an API token. Normal frontend events never contain it.
        return re.sub(r"(?i)\b(?:sk|key)-[a-z0-9_-]{8,}\b", "[REDACTED]", text)

    event = {
        "at": int(time.time() * 1000),
        "event": "client_planner",
        "stage": safe(payload.get("stage"), 64),
        "status": safe(payload.get("status"), 32),
        "model": safe(payload.get("model"), 100),
        "duration_ms": max(0, min(int(payload.get("duration_ms") or 0), 300_000)),
        "message": safe(payload.get("message"), 500),
    }
    print("CLIENT_PLANNER " + json.dumps(event, ensure_ascii=False), flush=True)
    return {"ok": True}


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
    plan = await _adapt_to_current_view(plan)
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


async def _sync_task_process(task: dict[str, Any]) -> dict[str, Any]:
    """Reflect a finished Pi action into the active web-plan step.

    Older builds stored accepted arm-queue actions as ``running`` forever even
    though those actions do not occupy the Pi managed-process slot.  Recover
    such persisted tasks here so merely refreshing the page unsticks them.
    """
    execution = dict(task.get("execution") or {})
    step_id = str(execution.get("active_step") or "")
    if not step_id:
        return task
    states = dict(execution.get("steps") or {})
    step_state = dict(states.get(step_id) or {})
    if step_state.get("status") != "running":
        return task
    step = next(
        (item for item in task.get("plan", {}).get("steps", []) if str(item.get("id")) == step_id),
        None,
    )
    if not step:
        return task
    action = str(step.get("action") or "")
    if action in {"model_grasp", "grasp", "reset_arm", "release"}:
        detail = dict(step_state.get("detail") or {})
        accepted = bool(detail.get("ok")) or any(
            marker in str(detail.get("message") or "")
            for marker in ("已入队", "queued", "后台开始执行")
        )
        if not accepted:
            return task
        now_ms = int(time.time() * 1000)
        detail["message"] = str(detail.get("message") or "机械臂动作已入队")
        detail["dispatch_completed"] = True
        detail["physical_result_requires_verification"] = action in {"model_grasp", "grasp"}
        states[step_id] = {
            "status": "completed",
            "detail": detail,
            "updated_at": now_ms,
            "recovered_from_running": True,
        }
        execution["steps"] = states
        execution["active_step"] = None
        events = list(execution.get("events", []))
        events.append({
            "at": now_ms,
            "type": "queued_arm_dispatch_recovered",
            "step_id": step_id,
            "action": action,
        })
        execution["events"] = events
        all_steps = task.get("plan", {}).get("steps", [])
        all_complete = all(
            states.get(str(item.get("id")), {}).get("status") == "completed"
            for item in all_steps
        )
        return app.state.history.update_task(
            task["id"],
            status="completed" if all_complete else "running",
            execution=execution,
        )
    managed_actions = {
        "patrol_search", "navigate_to", "return_home", "locate_destination",
        "locate_object", "align_for_grasp", "drive_relative",
    }
    if action not in managed_actions:
        return task
    dispatched_process = (step_state.get("detail") or {}).get("process") or {}
    dispatched_log = str(dispatched_process.get("log") or "")
    if not dispatched_log:
        return task
    try:
        robot = await asyncio.to_thread(app.state.robot.request, "/state")
    except Exception:
        return task
    managed = robot.get("managed") if isinstance(robot, dict) else None
    if not isinstance(managed, dict):
        return task
    same_process = str(managed.get("log") or "") == dispatched_log
    orphaned = (
        not managed.get("running")
        and not managed.get("log")
        and int(time.time() * 1000) - int(step_state.get("updated_at") or 0) > 5000
    )
    if not same_process and not orphaned:
        return task
    if same_process and (managed.get("running") or managed.get("returncode") is None):
        return task

    returncode = int(managed.get("returncode")) if same_process else 125
    terminal_status = "completed" if returncode == 0 else "failed"
    detail = dict(step_state.get("detail") or {})
    detail.update({
        "process": managed,
        "returncode": returncode,
        "message": (
            "底层动作已完成" if returncode == 0
            else str(
                managed.get("error")
                or ("桥接服务已重启，原执行进程不存在，请重试此步骤" if orphaned
                    else f"底层进程异常退出（{returncode}）")
            )
        ),
    })
    states[step_id] = {
        "status": terminal_status,
        "detail": detail,
        "updated_at": int(time.time() * 1000),
    }
    execution["steps"] = states
    execution["active_step"] = None
    events = list(execution.get("events", []))
    events.append({
        "at": int(time.time() * 1000),
        "type": "step_process_exit",
        "step_id": step_id,
        "status": terminal_status,
        "returncode": returncode,
    })
    execution["events"] = events
    all_steps = task.get("plan", {}).get("steps", [])
    all_complete = all(
        states.get(str(item.get("id")), {}).get("status") == "completed"
        for item in all_steps
    )
    task_status = "completed" if all_complete else ("blocked" if returncode else "running")
    return app.state.history.update_task(task["id"], status=task_status, execution=execution)


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str) -> dict[str, Any]:
    task = app.state.history.get_task(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    task = await _sync_task_process(task)
    try:
        if await asyncio.to_thread(finish_recording,app.state.robot,task):
            task=persist_finished_recording(app.state.history,task)
    except Exception:
        pass  # Keep recording on a transient disconnect; next poll retries.
    return {"ok": True, "task": task}


@app.post("/api/tasks/{task_id}/confirm")
def confirm_task(task_id: str) -> dict[str, Any]:
    task = app.state.history.get_task(task_id)
    if task is None:
        raise HTTPException(404, "任务不存在")
    if task["status"] not in {"draft", "paused", "blocked"}:
        raise HTTPException(409, f"任务状态不能确认：{task['status']}")
    execution = dict(task["execution"])
    prepare_recording(execution)
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
    existing = step_states.get(step_id, {})
    if existing.get("status") == "completed":
        return {"ok": True, "already_completed": True, "result": existing, "task": task}
    if existing.get("status") in {"running", "awaiting_verification"}:
        return {
            "ok": True,
            "already_running": True,
            "result": existing,
            "task": task,
        }
    dependencies = [str(item) for item in step.get("depends_on", [])]
    incomplete = [item for item in dependencies if step_states.get(item, {}).get("status") != "completed"]
    if incomplete:
        raise HTTPException(409, f"前置步骤尚未完成：{', '.join(incomplete)}")
    # Preserve the remote conversation-delete guard during retry dispatch.
    pending_execution = {**execution, 'active_step': step_id}
    app.state.history.update_task(task_id, status='running', execution=pending_execution)
    try:
        await asyncio.to_thread(start_recording,app.state.robot,execution,task['instruction'])
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
    task_status = "blocked" if statuses.intersection({'blocked', 'failed'}) else "running"
    if all(step_states.get(str(item.get("id")), {}).get("status") == "completed" for item in steps):
        task_status = "completed"
    updated = app.state.history.update_task(task_id, status=task_status, execution=execution)
    try:
        if await asyncio.to_thread(finish_recording,app.state.robot,updated):
            updated=persist_finished_recording(app.state.history,updated)
    except Exception:
        pass
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
    if (states.get(step_id) or {}).get("status") != "awaiting_verification":
        raise HTTPException(409, "只有等待人工核验的步骤才能手动确认，运行中的动作须等待真实结果")
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


@app.post('/api/robot/confirm-empty')
async def confirm_empty_gripper(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict) or payload.get('confirmed_empty') is not True:
        raise HTTPException(400, '请先确认夹爪已经没有物品')
    try:
        return await asyncio.to_thread(app.state.robot.request, '/skills/confirm-empty', payload)
    except Exception as exc:
        raise HTTPException(409, f'空夹爪状态复位失败：{exc}') from exc


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
async def robot_logs(name: str = "") -> JSONResponse:
    try:
        path = f"/logs/file?name={urlparse.quote(name)}" if name else "/logs/tail"
        return JSONResponse(await asyncio.to_thread(app.state.robot.request, path))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.get("/api/robot/logs/list")
async def robot_log_list() -> JSONResponse:
    try:
        return JSONResponse(await asyncio.to_thread(app.state.robot.request, "/logs/list"))
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc), "logs": []}, status_code=503)


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
    order=_camera_order_settings()
    reference=reference_payload(PursuitConfig())
    lines=reference['lines']
    reference['lines']={side:lines[preview_source(side,order['swapped'])]
                        for side in ('left','right')}
    return {"ok": True, "camera_order": order, "forward_reference": reference}


@app.post("/api/robot/camera-order/calibrate")
async def calibrate_camera_order() -> dict[str, Any]:
    try:
        state = await asyncio.to_thread(app.state.robot.request, "/state", timeout=3.0)
        managed = state.get("managed") or {}
        if managed.get("running") and managed.get("kind") != "camera_preview":
            raise HTTPException(409, "运动中不能校准或改变相机映射，请先停车")
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
        current = _camera_order_settings()
        # Visual geometry on this toe-in pair is ambiguous at some depths.
        # Never let a weaker automatic guess overwrite an operator mapping.
        applied = (current["source"] != "operator"
                   and order in {"normal", "swapped"}
                   and confidence >= 0.85)
        if applied:
            value = {
                "swapped": order == "normal",  # inference sees model-order 2/0 endpoints
                "source": str(inference.get("source") or "qwen"),
                "confidence": confidence,
                "reason": str(inference.get("reason") or "")[:300],
                "updated_at": int(time.time() * 1000),
            }
            app.state.history.set_setting("camera_order", value)
        else:
            value = current
        return {"ok": True, "applied": applied, "camera_order": value,
                "inference": inference,
                "note": ("手动确认的物理左右优先；AI 结果仅供参考"
                         if current["source"] == "operator" else "")}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"摄像头左右校准失败：{type(exc).__name__}: {exc}") from exc


@app.post("/api/robot/camera-order/swap")
def swap_camera_order() -> dict[str, Any]:
    state = app.state.robot.request("/state", timeout=3.0)
    managed = state.get("managed") or {}
    if managed.get("running") and managed.get("kind") != "camera_preview":
        raise HTTPException(409, "请先停止底盘任务再交换相机，不能在运动中改变左右映射")
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
    source_camera = preview_source(camera, order["swapped"])
    try:
        content_type, chunks = await asyncio.to_thread(app.state.robot.stream, f"/video/{source_camera}")
        return StreamingResponse(chunks, media_type=content_type, headers={"Cache-Control": "no-store"})
    except Exception as exc:
        raise HTTPException(503, f"{camera} 摄像头暂不可用：{exc}") from exc


@app.get("/api/robot/pursuit/stream.mjpg")
async def pursuit_camera_stream() -> StreamingResponse:
    """Show the pursuit-owned stereo stream without opening Pi cameras twice."""
    try:
        robot = await asyncio.to_thread(app.state.robot.request, "/state", timeout=3.0)
        managed = robot.get("managed") or {}
        if managed.get("kind") != "visual_pursuit" or not managed.get("running"):
            raise HTTPException(409, "视觉追踪未运行")
        response = await asyncio.to_thread(
            urlrequest.urlopen, f"{PURSUIT_STREAM_URL}/stream.mjpg", timeout=8.0
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"追踪双目画面暂不可用：{exc}") from exc

    def chunks():
        try:
            while block := response.read(16 * 1024):
                yield block
        finally:
            response.close()

    return StreamingResponse(chunks(), media_type=response.headers.get(
        "Content-Type", "multipart/x-mixed-replace"), headers={
            "Cache-Control": "no-store", "X-Accel-Buffering": "no",
        })


from .conversation_delete import install_conversation_delete
install_conversation_delete(app)


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

