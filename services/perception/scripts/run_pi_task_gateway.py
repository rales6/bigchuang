from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


ALLOWED_ACTIONS = {
    "inspect_semantic_map",
    "patrol_search",
    "locate_object",
    "navigate_to",
    "align_for_grasp",
    "grasp",
    "verify_grasp",
    "locate_destination",
    "release",
    "verify_completion",
    "return_home",
    "report_result",
}
NAVIGATION_ACTIONS = {"patrol_search", "navigate_to", "return_home"}
MANIPULATION_ACTIONS = {
    "align_for_grasp",
    "grasp",
    "verify_grasp",
    "release",
    "verify_completion",
}
ARM_MOTION_SEQUENCES = {
    "align_for_grasp": ["pregrasp"],
    "grasp": ["grasp", "lift", "carry"],
    "release": ["preplace", "place", "open", "stow"],
}
BASE_HARDWARE_CONNECTED_ACTIONS = {
    "inspect_semantic_map",
    "patrol_search",
    "locate_object",
    "navigate_to",
    "locate_destination",
    "report_result",
}
HARDWARE_CONNECTED_ACTIONS = BASE_HARDWARE_CONNECTED_ACTIONS | MANIPULATION_ACTIONS


QUERY_ALIAS_GROUPS = (
    ("簸箕", "dustpan"),
    ("垃圾桶", "trash bin", "trash_bin", "dustbin"),
    ("垃圾", "trash", "rubbish", "waste"),
    ("桌", "table", "desk"),
    ("柜", "cabinet", "cupboard"),
    ("瓶", "bottle"),
    ("椅", "chair"),
    ("墙角", "corner"),
    ("扫帚", "broom"),
    ("纸箱", "box", "carton"),
)


def _semantic_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, "")).lower()
        for key in ("id", "name", "label", "category", "type", "role", "visual_description")
    )


def _contains_alias(text: str, alias: str) -> bool:
    folded = alias.lower()
    if folded.isascii() and re.fullmatch(r"[a-z0-9_ -]+", folded):
        pattern = re.escape(folded).replace(r"\ ", r"[ _-]+")
        return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None
    return folded in text


def _query_aliases(query: str) -> tuple[str, ...]:
    folded = query.lower()
    aliases: list[str] = []
    container_requested = any(
        alias in folded
        for alias in ("簸箕", "dustpan", "垃圾桶", "trash bin", "trash_bin", "dustbin")
    )
    for group in QUERY_ALIAS_GROUPS:
        if group[0] == "垃圾" and container_requested:
            continue
        if any(alias in folded for alias in group):
            aliases.extend(group)
    return tuple(dict.fromkeys(aliases))


def _landmark_matches_query(item: dict[str, Any], query: str) -> bool:
    folded = query.lower().strip()
    if not folded:
        return True
    text = _semantic_text(item)
    aliases = _query_aliases(folded)
    if aliases:
        return any(_contains_alias(text, alias) for alias in aliases)
    return folded in text or text in folded


class PiTaskGateway:
    """Reference Raspberry Pi gateway. Hardware mode is deliberately capability-gated."""

    def __init__(self, map_url: str, hardware_enabled: bool, safety_token: str) -> None:
        self.map_url = map_url.rstrip("/")
        self.hardware_enabled = bool(hardware_enabled)
        self.safety_token = safety_token.strip()
        self.lock = threading.RLock()
        self.current: dict[str, Any] = {
            "state": "succeeded",
            "message": "idle",
            "task_id": "",
            "step_id": "",
            "action": "",
            "evidence": {"hardware_command_sent": False},
        }

    def _map_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = urllib.request.Request(self.map_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=max(0.2, float(timeout_s))) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"map service unavailable: {exc}") from exc
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {"raw_bytes": len(raw)}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _validate_command(payload: dict[str, Any]) -> None:
        if payload.get("schema") != "robot_task_command.v1":
            raise ValueError("unsupported command schema")
        action = str(payload.get("action") or "")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported action: {action}")
        if not payload.get("task_id") or not payload.get("step_id"):
            raise ValueError("task_id and step_id are required")
        safety = payload.get("safety_contract")
        if not isinstance(safety, dict) or safety.get("schema") != "robot_safety_contract.v1":
            raise ValueError("valid safety_contract is required")
        if action in NAVIGATION_ACTIONS:
            required = (
                safety.get("require_lidar_clearance"),
                safety.get("require_camera_clearance"),
                safety.get("require_localization"),
                safety.get("stop_on_sensor_loss"),
            )
            if not all(required):
                raise ValueError("navigation safety contract is incomplete")
        if action in MANIPULATION_ACTIONS:
            if not safety.get("require_chassis_stopped") or not safety.get("stop_on_sensor_loss"):
                raise ValueError("manipulation safety contract is incomplete")
            if action in {"align_for_grasp", "grasp", "release"} and not safety.get(
                "require_target_in_workspace"
            ):
                raise ValueError("manipulation workspace safety gate is required")
            required_sensors = {str(item).lower() for item in payload.get("required_sensors", [])}
            if "camera" not in required_sensors:
                raise ValueError("manipulation requires camera evidence")

    def health(self) -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        connected_actions = set(BASE_HARDWARE_CONNECTED_ACTIONS)
        if self.hardware_enabled:
            try:
                status = self._map_request("GET", "/api/status")
                evidence["map_service"] = "connected"
                evidence["lidar"] = status.get("lidar", {})
                arm = self._map_request("GET", "/api/arm/status")
                evidence["arm"] = arm
                if arm.get("ready", False):
                    connected_actions.update(MANIPULATION_ACTIONS)
            except Exception as exc:
                return {
                    "ok": False,
                    "ready": False,
                    "hardware_enabled": True,
                    "error": str(exc),
                }
        return {
            "ok": True,
            "ready": True,
            "service": "raspberry_pi_task_gateway",
            "hardware_enabled": self.hardware_enabled,
            "connected_hardware_actions": sorted(connected_actions),
            "missing_hardware_actions": sorted(ALLOWED_ACTIONS - connected_actions),
            "evidence": evidence,
        }

    def _stop_motion(self, reason: str) -> dict[str, Any]:
        errors: list[str] = []
        sent = False
        if self.hardware_enabled:
            for path in ("/api/patrol/stop", "/api/drive/stop", "/api/arm/stop"):
                try:
                    self._map_request("POST", path)
                    sent = True
                except Exception as exc:
                    errors.append(str(exc))
        with self.lock:
            self.current = {
                "state": "succeeded" if not errors else "failed",
                "message": reason,
                "task_id": self.current.get("task_id", ""),
                "step_id": self.current.get("step_id", ""),
                "action": "safe_stop",
                "evidence": {"hardware_command_sent": sent, "errors": errors},
            }
            return dict(self.current)

    def _arm_status(self) -> dict[str, Any]:
        status = self._map_request("GET", "/api/arm/status")
        if not status.get("ready", False):
            raise RuntimeError(
                str(status.get("error") or status.get("profile_errors") or "arm is not calibrated and ready")
            )
        if status.get("fault"):
            raise RuntimeError(f"arm fault: {status['fault']}")
        return status

    def _manipulation_observation(
        self,
        target: dict[str, Any],
        verification_mode: str,
    ) -> dict[str, Any]:
        observation = self._map_request(
            "POST",
            "/api/manipulation/observation",
            {"target": dict(target), "verification_mode": verification_mode},
            timeout_s=25.0,
        )
        if not observation.get("ok", False):
            raise RuntimeError(str(observation.get("error") or "manipulation observation failed"))
        return observation

    @staticmethod
    def _observation_failures(action: str, observation: dict[str, Any]) -> list[str]:
        if action in {"align_for_grasp", "grasp", "release"}:
            required = (
                "visible",
                "camera_fresh",
                "centered",
                "reachable",
                "lidar_clearance",
                "chassis_stopped",
            )
        elif action == "verify_grasp":
            required = ("camera_fresh", "object_in_gripper")
        elif action == "verify_completion":
            required = ("camera_fresh", "target_in_destination")
        else:
            required = ()
        return [name for name in required if not observation.get(name, False)]

    def _start_arm_sequence(self, state: dict[str, Any], poses: list[str]) -> None:
        if not poses:
            raise ValueError("arm pose sequence must not be empty")
        self._arm_status()
        response = self._map_request("POST", "/api/arm/pose", {"pose": poses[0]})
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error") or f"arm pose rejected: {poses[0]}"))
        state.update(
            state="running",
            message=f"arm pose sequence started: {poses[0]}",
            arm_sequence_remaining=list(poses[1:]),
            evidence={
                "hardware_command_sent": bool(response.get("hardware_command_sent", True)),
                "arm": response,
            },
        )

    def command(self, payload: dict[str, Any], supplied_token: str) -> dict[str, Any]:
        self._validate_command(payload)
        requested_hardware = bool(payload.get("hardware_enabled", False))
        if requested_hardware != self.hardware_enabled:
            return {"state": "blocked", "message": "hardware mode mismatch", "evidence": {}}
        if self.hardware_enabled and supplied_token != self.safety_token:
            self._stop_motion("invalid safety token")
            return {"state": "blocked", "message": "invalid safety token", "evidence": {}}

        action = str(payload["action"])
        state = {
            "state": "succeeded",
            "message": "dry-run step accepted",
            "task_id": str(payload["task_id"]),
            "step_id": str(payload["step_id"]),
            "action": action,
            "target": dict(payload.get("target") or {}),
            "evidence": {"hardware_command_sent": False, "mode": "dry_run"},
        }
        if not self.hardware_enabled:
            with self.lock:
                self.current = state
            return dict(state)

        if action not in HARDWARE_CONNECTED_ACTIONS:
            self._stop_motion(f"capability not connected: {action}")
            state.update(
                state="blocked",
                message=f"hardware adapter not implemented for {action}",
                evidence={"hardware_command_sent": False, "safe_stop_requested": True},
            )
        elif action == "inspect_semantic_map":
            status = self._map_request("GET", "/api/status")
            semantic = self._map_request("GET", "/api/semantic")
            state.update(
                message="map and localization snapshot acquired",
                evidence={
                    "hardware_command_sent": False,
                    "scan_count": status.get("scan_count"),
                    "landmarks": len(semantic.get("landmarks", [])),
                },
            )
        elif action == "patrol_search":
            self._map_request("POST", "/api/patrol/start")
            state.update(
                state="running",
                message="map patrol started; waiting for semantic target",
                evidence={"hardware_command_sent": True, "patrol_started": True},
            )
        elif action in {"locate_object", "locate_destination"}:
            semantic = self._map_request("GET", "/api/semantic")
            landmark = self._resolve_target(semantic, state["target"])
            if landmark is None:
                state.update(
                    state="failed",
                    message=f"semantic target not found: {state['target'].get('query', '')}",
                    evidence={"hardware_command_sent": False},
                )
            else:
                state.update(
                    message="semantic target resolved in map coordinates",
                    evidence={"hardware_command_sent": False, "landmark": landmark},
                )
        elif action == "navigate_to":
            semantic = self._map_request("GET", "/api/semantic")
            landmark = self._resolve_target(semantic, state["target"])
            map_xy = None if landmark is None else landmark.get("map_xy")
            if not isinstance(map_xy, list) or len(map_xy) < 2:
                state.update(
                    state="failed",
                    message="navigation target has no verified map_xy",
                    evidence={"hardware_command_sent": False},
                )
            else:
                goal_payload = {
                    "x_m": float(map_xy[0]),
                    "y_m": float(map_xy[1]),
                    "target_id": str(landmark.get("id") or ""),
                }
                response = self._map_request("POST", "/api/navigation/goal", goal_payload)
                if not response.get("ok", False):
                    state.update(
                        state="failed",
                        message=str(response.get("error") or "navigation goal rejected"),
                        evidence={"hardware_command_sent": False},
                    )
                else:
                    state.update(
                        state="running",
                        message="semantic map goal accepted by local navigator",
                        navigation_goal=goal_payload,
                        evidence={"hardware_command_sent": True, "landmark": landmark},
                    )
        elif action in MANIPULATION_ACTIONS:
            try:
                arm_status = self._arm_status()
                observation = self._manipulation_observation(state["target"], action)
                failures = self._observation_failures(action, observation)
                if failures:
                    self._stop_motion(
                        f"manipulation evidence missing for {action}: {', '.join(failures)}"
                    )
                    state.update(
                        state="blocked",
                        message=f"manipulation safety gates failed: {', '.join(failures)}",
                        evidence={
                            "hardware_command_sent": False,
                            "safe_stop_requested": True,
                            "arm": arm_status,
                            "observation": observation,
                            "failed_gates": failures,
                        },
                    )
                elif action in {"verify_grasp", "verify_completion"}:
                    state.update(
                        message=f"camera verification succeeded: {action}",
                        evidence={
                            "hardware_command_sent": False,
                            "arm": arm_status,
                            "observation": observation,
                        },
                    )
                else:
                    self._start_arm_sequence(state, ARM_MOTION_SEQUENCES[action])
                    state["evidence"]["observation"] = observation
            except Exception as exc:
                self._stop_motion(f"arm action failed: {action}: {exc}")
                state.update(
                    state="blocked",
                    message=str(exc),
                    evidence={"hardware_command_sent": False, "safe_stop_requested": True},
                )
        elif action == "report_result":
            stopped = self._stop_motion("task report and stop")
            state.update(
                state="succeeded" if stopped.get("state") == "succeeded" else "blocked",
                message="robot stopped; result ready"
                if stopped.get("state") == "succeeded"
                else "task result cannot be finalized because safe stop was not confirmed",
                evidence=stopped.get("evidence", {}),
            )
        with self.lock:
            self.current = state
        return dict(state)

    @staticmethod
    def _target_visible(semantic: dict[str, Any], target: dict[str, Any]) -> bool:
        return PiTaskGateway._resolve_target(semantic, target) is not None

    @staticmethod
    def _resolve_target(
        semantic: dict[str, Any],
        target: dict[str, Any],
    ) -> dict[str, Any] | None:
        landmark_id = str(target.get("landmark_id") or "")
        query = str(target.get("query") or "").lower()
        if not landmark_id and not query.strip():
            return None
        candidates: list[dict[str, Any]] = []
        for item in semantic.get("landmarks", []):
            if not isinstance(item, dict):
                continue
            if landmark_id and str(item.get("id")) == landmark_id:
                if _landmark_matches_query(item, query):
                    return dict(item)
                continue
            if _landmark_matches_query(item, query):
                candidates.append(item)
        if not candidates:
            return None
        return dict(
            max(
                candidates,
                key=lambda item: float(item.get("stability") or item.get("confidence") or 0.0),
            )
        )

    def state(self, task_id: str, step_id: str) -> dict[str, Any]:
        with self.lock:
            current = dict(self.current)
        if task_id != current.get("task_id") or step_id != current.get("step_id"):
            return {"state": "failed", "message": "unknown task or step", "evidence": {}}
        if self.hardware_enabled and current.get("action") == "patrol_search" and current.get("state") == "running":
            try:
                semantic = self._map_request("GET", "/api/semantic")
                if self._target_visible(semantic, current.get("target", {})):
                    self._map_request("POST", "/api/patrol/stop")
                    current.update(
                        state="succeeded",
                        message="semantic target found; patrol stopped",
                        evidence={"hardware_command_sent": True, "target_found": True},
                    )
                    with self.lock:
                        self.current = current
            except Exception as exc:
                self._stop_motion(f"patrol sensor failure: {exc}")
                current.update(state="failed", message=str(exc), evidence={"safe_stop_requested": True})
        elif self.hardware_enabled and current.get("action") == "navigate_to" and current.get("state") == "running":
            try:
                status = self._map_request("GET", "/api/status")
                navigation = status.get("control", {}).get("navigation", {})
                nav_state = str(navigation.get("state") or "")
                if nav_state == "succeeded":
                    current.update(
                        state="succeeded",
                        message="local navigator reached semantic target standoff",
                        evidence={"hardware_command_sent": True, "navigation": navigation},
                    )
                elif nav_state in {"blocked", "cancelled"}:
                    self._stop_motion(f"navigation {nav_state}: {navigation.get('reason', '')}")
                    current.update(
                        state="blocked" if nav_state == "blocked" else "failed",
                        message=str(navigation.get("reason") or nav_state),
                        evidence={"safe_stop_requested": True, "navigation": navigation},
                    )
                else:
                    current["evidence"] = {"hardware_command_sent": True, "navigation": navigation}
                with self.lock:
                    self.current = current
            except Exception as exc:
                self._stop_motion(f"navigation sensor failure: {exc}")
                current.update(state="failed", message=str(exc), evidence={"safe_stop_requested": True})
        elif (
            self.hardware_enabled
            and current.get("action") in ARM_MOTION_SEQUENCES
            and current.get("state") == "running"
        ):
            try:
                arm_status = self._arm_status()
                if arm_status.get("moving", False):
                    current["evidence"] = {
                        **dict(current.get("evidence") or {}),
                        "hardware_command_sent": True,
                        "arm": arm_status,
                    }
                else:
                    remaining = list(current.get("arm_sequence_remaining") or [])
                    if remaining:
                        next_pose = remaining.pop(0)
                        response = self._map_request("POST", "/api/arm/pose", {"pose": next_pose})
                        if not response.get("ok", False):
                            raise RuntimeError(
                                str(response.get("error") or f"arm pose rejected: {next_pose}")
                            )
                        current.update(
                            message=f"arm pose sequence advancing: {next_pose}",
                            arm_sequence_remaining=remaining,
                            evidence={
                                **dict(current.get("evidence") or {}),
                                "hardware_command_sent": True,
                                "arm": response,
                            },
                        )
                    else:
                        current.update(
                            state="succeeded",
                            message="arm pose sequence completed",
                            evidence={
                                **dict(current.get("evidence") or {}),
                                "hardware_command_sent": True,
                                "arm": arm_status,
                            },
                        )
                with self.lock:
                    self.current = current
            except Exception as exc:
                self._stop_motion(f"arm sequence failure: {exc}")
                current.update(
                    state="blocked",
                    message=str(exc),
                    evidence={"safe_stop_requested": True, "hardware_command_sent": False},
                )
        return current


def _handler(gateway: PiTaskGateway):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/task/v1/health":
                    self._json(200, gateway.health())
                elif parsed.path == "/api/task/v1/state":
                    query = parse_qs(parsed.query)
                    self._json(
                        200,
                        gateway.state(
                            str((query.get("task_id") or [""])[0]),
                            str((query.get("step_id") or [""])[0]),
                        ),
                    )
                else:
                    self._json(404, {"state": "failed", "message": "not found"})
            except Exception as exc:
                self._json(500, {"state": "failed", "message": str(exc)})

        def do_POST(self) -> None:
            try:
                payload = self._body()
                if self.path == "/api/task/v1/command":
                    result = gateway.command(payload, self.headers.get("X-Robot-Safety-Token", ""))
                    self._json(200, result)
                elif self.path == "/api/task/v1/stop":
                    self._json(200, gateway._stop_motion(str(payload.get("reason") or "remote stop")))
                else:
                    self._json(404, {"state": "failed", "message": "not found"})
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"state": "failed", "message": str(exc)})
            except Exception as exc:
                gateway._stop_motion(f"gateway exception: {exc}")
                self._json(500, {"state": "failed", "message": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[pi-task-gateway] {self.address_string()} {fmt % args}")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe Raspberry Pi task gateway reference service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--map-url", default="http://127.0.0.1:8766")
    parser.add_argument("--enable-hardware", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("ROBOT_TASK_HARDWARE_TOKEN", "").strip()
    if args.enable_hardware and not token:
        parser.error("--enable-hardware requires ROBOT_TASK_HARDWARE_TOKEN")
    gateway = PiTaskGateway(args.map_url, args.enable_hardware, token)
    server = ThreadingHTTPServer((args.host, args.port), _handler(gateway))
    print(
        f"[pi-task-gateway] http://{args.host}:{args.port} "
        f"hardware_enabled={args.enable_hardware} map={args.map_url}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        gateway._stop_motion("gateway shutdown")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
