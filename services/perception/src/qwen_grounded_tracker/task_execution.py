from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from qwen_grounded_tracker.semantic_task_planner import (
    NAVIGATION_ACTIONS,
    PlanStep,
    SemanticTaskPlan,
    validate_plan,
)


BRIDGE_STATES = {"accepted", "running", "succeeded", "failed", "blocked"}
TERMINAL_EXECUTION_STATES = {"completed", "failed", "blocked", "cancelled", "needs_replan"}
DEFAULT_STEP_TIMEOUTS_S = {
    "inspect_semantic_map": 10.0,
    "patrol_search": 180.0,
    "locate_object": 45.0,
    "navigate_to": 180.0,
    "align_for_grasp": 30.0,
    "grasp": 20.0,
    "verify_grasp": 15.0,
    "locate_destination": 45.0,
    "release": 20.0,
    "verify_completion": 20.0,
    "return_home": 180.0,
    "report_result": 10.0,
}


@dataclass
class BridgeResult:
    state: str
    message: str = ""
    evidence: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state not in BRIDGE_STATES:
            raise ValueError(f"Unsupported bridge state: {self.state}")
        if self.evidence is None:
            self.evidence = {}


class RobotBridge(Protocol):
    hardware_enabled: bool

    def health(self) -> dict[str, Any]: ...
    def dispatch(self, task_id: str, step: PlanStep) -> BridgeResult: ...
    def poll(self, task_id: str, step: PlanStep) -> BridgeResult: ...
    def emergency_stop(self, task_id: str, reason: str) -> BridgeResult: ...


def safety_contract_for(step: PlanStep) -> dict[str, Any]:
    navigation = step.action in NAVIGATION_ACTIONS
    manipulation = step.action in {
        "align_for_grasp",
        "grasp",
        "verify_grasp",
        "release",
        "verify_completion",
    }
    return {
        "schema": "robot_safety_contract.v1",
        "require_lidar_clearance": navigation,
        "require_camera_clearance": navigation,
        "require_localization": navigation,
        "require_chassis_stopped": manipulation,
        "require_target_in_workspace": step.action in {"align_for_grasp", "grasp", "release"},
        "stop_on_sensor_loss": True,
        "stop_on_person": True,
        "max_linear_m_s": 0.10 if navigation else 0.0,
        "max_angular_rad_s": 0.30 if navigation else 0.0,
    }


def command_payload(task_id: str, step: PlanStep, hardware_enabled: bool) -> dict[str, Any]:
    return {
        "schema": "robot_task_command.v1",
        "task_id": task_id,
        "step_id": step.id,
        "action": step.action,
        "description": step.description,
        "target": dict(step.target),
        "required_sensors": list(step.sensors),
        "success_condition": step.success_condition,
        "hardware_enabled": bool(hardware_enabled),
        "safety_contract": safety_contract_for(step),
    }


class SimulationRobotBridge:
    """Immediate deterministic bridge. It never opens a network or hardware link."""

    hardware_enabled = False

    def __init__(self, fail_action: str = "", blocked_action: str = "") -> None:
        self.fail_action = fail_action
        self.blocked_action = blocked_action
        self.commands: list[dict[str, Any]] = []
        self.stop_reasons: list[str] = []

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "ready": True,
            "mode": "simulation",
            "hardware_enabled": False,
        }

    def dispatch(self, task_id: str, step: PlanStep) -> BridgeResult:
        payload = command_payload(task_id, step, False)
        payload["hardware_command_sent"] = False
        self.commands.append(payload)
        if step.action == self.blocked_action:
            return BridgeResult("blocked", f"simulated blocked action: {step.action}")
        if step.action == self.fail_action:
            return BridgeResult("failed", f"simulated failed action: {step.action}")
        return BridgeResult(
            "succeeded",
            "simulation step accepted",
            {"hardware_command_sent": False, "action": step.action},
        )

    def poll(self, task_id: str, step: PlanStep) -> BridgeResult:
        del task_id, step
        return BridgeResult("succeeded", "simulation already completed")

    def emergency_stop(self, task_id: str, reason: str) -> BridgeResult:
        del task_id
        self.stop_reasons.append(reason)
        return BridgeResult("succeeded", "simulation stop recorded", {"hardware_command_sent": False})


class HttpRobotBridge:
    """Versioned HTTP bridge to a Raspberry Pi task gateway."""

    def __init__(
        self,
        base_url: str,
        *,
        hardware_enabled: bool = False,
        safety_token: str = "",
        timeout_s: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.hardware_enabled = bool(hardware_enabled)
        self.safety_token = safety_token.strip()
        self.timeout_s = max(0.2, float(timeout_s))
        if self.hardware_enabled and not self.safety_token:
            raise ValueError("Hardware mode requires a non-empty safety token")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.safety_token:
            headers["X-Robot-Safety-Token"] = self.safety_token
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Raspberry Pi gateway request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Raspberry Pi gateway returned non-object JSON")
        return data

    @staticmethod
    def _result(payload: dict[str, Any]) -> BridgeResult:
        state = str(payload.get("state") or ("succeeded" if payload.get("ok") else "failed"))
        return BridgeResult(
            state=state,
            message=str(payload.get("message") or payload.get("error") or ""),
            evidence=dict(payload.get("evidence") or {}),
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/api/task/v1/health")

    def dispatch(self, task_id: str, step: PlanStep) -> BridgeResult:
        payload = command_payload(task_id, step, self.hardware_enabled)
        return self._result(self._request("POST", "/api/task/v1/command", payload))

    def poll(self, task_id: str, step: PlanStep) -> BridgeResult:
        query = urllib.parse.urlencode({"task_id": task_id, "step_id": step.id})
        return self._result(self._request("GET", f"/api/task/v1/state?{query}"))

    def emergency_stop(self, task_id: str, reason: str) -> BridgeResult:
        return self._result(
            self._request(
                "POST",
                "/api/task/v1/stop",
                {"schema": "robot_task_stop.v1", "task_id": task_id, "reason": reason},
            )
        )


class PlanExecutionEngine:
    """Sequential, acknowledged execution with retries, timeouts and safe stop."""

    def __init__(self, bridge: RobotBridge) -> None:
        self.bridge = bridge
        self.lock = threading.RLock()
        self.plan: SemanticTaskPlan | None = None
        self.status = "idle"
        self.current_index = 0
        self.step_started_at = 0.0
        self.step_dispatched = False
        self.retry_counts: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []
        self.last_error = ""

    def load(self, plan: SemanticTaskPlan) -> None:
        validate_plan(plan)
        with self.lock:
            if self.status == "running":
                raise RuntimeError("Cannot replace a running task")
            self.plan = plan
            self.status = "ready"
            self.current_index = 0
            self.step_started_at = 0.0
            self.step_dispatched = False
            self.retry_counts = {}
            self.events = []
            self.last_error = ""

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.plan is None:
                raise RuntimeError("No task plan loaded")
            health = self.bridge.health()
            if not health.get("ok", False) or not health.get("ready", True):
                raise RuntimeError(f"Robot bridge is not ready: {health}")
            if bool(health.get("hardware_enabled", False)) != bool(self.bridge.hardware_enabled):
                raise RuntimeError("Robot bridge hardware mode does not match gateway")
            self.status = "running"
            self._event("task_started", {"bridge_health": health})
            return self.snapshot()

    def _event(self, kind: str, details: dict[str, Any]) -> None:
        self.events.append(
            {
                "time_s": time.time(),
                "kind": kind,
                "step_id": self.current_step.id if self.current_step else None,
                **details,
            }
        )
        self.events = self.events[-200:]

    @property
    def current_step(self) -> PlanStep | None:
        if self.plan is None or not (0 <= self.current_index < len(self.plan.steps)):
            return None
        return self.plan.steps[self.current_index]

    def _fail(self, step: PlanStep, result: BridgeResult) -> None:
        retries = self.retry_counts.get(step.id, 0)
        if step.on_failure == "retry" and retries < step.max_retries:
            self.retry_counts[step.id] = retries + 1
            self.step_dispatched = False
            self.step_started_at = 0.0
            self._event("step_retry", {"retry": retries + 1, "message": result.message})
            return
        reason = f"{step.id}/{step.action}: {result.message or result.state}"
        self.last_error = reason
        try:
            self.bridge.emergency_stop(self.plan.task_id if self.plan else "", reason)
        finally:
            if result.state == "blocked":
                self.status = "blocked"
            elif step.on_failure == "replan":
                self.status = "needs_replan"
            else:
                self.status = "failed"
            self._event("task_stopped", {"state": self.status, "reason": reason})

    def _handle_result(self, step: PlanStep, result: BridgeResult) -> None:
        self._event(
            "bridge_result",
            {"state": result.state, "message": result.message, "evidence": result.evidence},
        )
        if result.state in {"accepted", "running"}:
            return
        if result.state in {"failed", "blocked"}:
            self._fail(step, result)
            return
        self.current_index += 1
        self.step_dispatched = False
        self.step_started_at = 0.0
        if self.plan is not None and self.current_index >= len(self.plan.steps):
            self.status = "completed"
            self._event("task_completed", {})

    def tick(self) -> dict[str, Any]:
        with self.lock:
            if self.status != "running":
                return self.snapshot()
            step = self.current_step
            if step is None:
                self.status = "completed"
                return self.snapshot()
            now = time.monotonic()
            if self.step_dispatched:
                timeout_s = DEFAULT_STEP_TIMEOUTS_S.get(step.action, 60.0)
                if now - self.step_started_at > timeout_s:
                    self._fail(step, BridgeResult("failed", f"step timeout after {timeout_s:.1f}s"))
                    return self.snapshot()
                result = self.bridge.poll(self.plan.task_id, step)
            else:
                self.step_dispatched = True
                self.step_started_at = now
                self._event("step_dispatched", {"action": step.action})
                result = self.bridge.dispatch(self.plan.task_id, step)
            self._handle_result(step, result)
            return self.snapshot()

    def run_until_terminal(self, max_ticks: int = 200) -> dict[str, Any]:
        for _ in range(max(1, int(max_ticks))):
            state = self.tick()
            if state["status"] in TERMINAL_EXECUTION_STATES:
                return state
        raise RuntimeError("Execution did not reach a terminal state within max_ticks")

    def cancel(self, reason: str = "operator_cancel") -> dict[str, Any]:
        with self.lock:
            task_id = self.plan.task_id if self.plan else ""
            self.bridge.emergency_stop(task_id, reason)
            self.status = "cancelled"
            self.last_error = reason
            self._event("task_cancelled", {"reason": reason})
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        step = self.current_step
        return {
            "schema": "robot_task_execution.v1",
            "status": self.status,
            "task_id": self.plan.task_id if self.plan else None,
            "hardware_enabled": bool(self.bridge.hardware_enabled),
            "current_step_index": self.current_index,
            "total_steps": len(self.plan.steps) if self.plan else 0,
            "current_step": asdict(step) if step else None,
            "retry_counts": dict(self.retry_counts),
            "last_error": self.last_error,
            "events": list(self.events[-50:]),
        }
