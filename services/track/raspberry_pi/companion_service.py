"""HTTP bridge used by RobotCompanionAndroid.

The service exposes structured task planning and a validated configuration
surface.  It never accepts arbitrary shell commands from the phone: mapping,
inspection and object-search processes are assembled from fixed modules and
range-checked values only.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from raspberry_pi.bluetooth_recovery import recover_bluetooth


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "companion_service.json"
LIVE_STATUS_PATH = ROOT / "search_runs" / "live_status.json"
LIVE_MAP_PATH = ROOT / "search_runs" / "live_map.png"
DECISION_PATH = ROOT / "search_runs" / "latest_decision.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "mapping": {
        "usable_fov": 160.0,
        "mapping_max_distance": 4.0,
        "stationary_map_distance": 0.35,
        "stationary_map_angle": 20.0,
        "stationary_settle_scans": 2,
        "stationary_fusion_scans": 5,
        "stationary_min_support": 3,
        "stationary_range_tolerance": 0.10,
        "speed": 300,
        "max_drive_speed": 420,
        "slow_speed": 220,
        "turn_speed": 1500,
        "max_turn_speed": 2600,
        "stop_distance": 0.40,
        "clearance": 0.30,
        "map_size": 20.0,
        "max_runtime_min": 15.0,
        "output": "maps/ble_scene_17",
    },
    "annotation": {
        "map": "maps/ble_scene_17.pgm",
        "map_yaml": "maps/ble_scene_17.yaml",
        "semantic_map": "maps/ble_scene_17_semantics.json",
        "standoff": 0.65,
        "max_targets": 32,
        "photos_per_target": 2,
        "speed": 300,
        "max_drive_speed": 460,
        "slow_speed": 280,
        "turn_speed": 1500,
        "stop_distance": 0.42,
        "clearance": 0.30,
        "settle_seconds": 1.0,
        "camera_width": 1280,
        "camera_height": 720,
        "max_runtime_min": 30.0,
        "output": "inspections/ble_scene_17",
    },
    "bluetooth": {
        "device_name": "ESP32-Robot-Car",
        "device_address": "64:B7:08:69:9A:E6",
        "auto_recover": True,
        "failure_threshold": 3,
        "max_recovery_attempts": 2,
        "startup_grace_seconds": 45.0,
        "recovery_cooldown_seconds": 2.0,
    },
    "audio": {"volume": 0.80},
}


RANGES = {
    "mapping": {
        "usable_fov": (30.0, 180.0), "mapping_max_distance": (0.5, 8.0),
        "stationary_map_distance": (0.10, 1.0), "stationary_map_angle": (5.0, 90.0),
        "stationary_settle_scans": (0, 20), "stationary_fusion_scans": (1, 30),
        "stationary_min_support": (1, 20), "stationary_range_tolerance": (0.02, 0.50),
        "speed": (100, 500), "max_drive_speed": (100, 500), "slow_speed": (100, 500),
        "turn_speed": (500, 3000), "max_turn_speed": (500, 3500),
        "stop_distance": (0.20, 1.5), "clearance": (0.15, 0.80),
        "map_size": (5.0, 50.0), "max_runtime_min": (1.0, 120.0),
    },
    "annotation": {
        "standoff": (0.35, 1.5), "max_targets": (1, 64),
        "photos_per_target": (1, 4),
        "speed": (100, 500), "max_drive_speed": (100, 500), "slow_speed": (100, 500),
        "turn_speed": (500, 3000), "stop_distance": (0.20, 1.5),
        "clearance": (0.15, 0.80), "settle_seconds": (0.2, 10.0),
        "camera_width": (320, 3840), "camera_height": (240, 2160),
        "max_runtime_min": (1.0, 180.0),
    },
    "bluetooth": {
        "failure_threshold": (1, 20), "max_recovery_attempts": (0, 5),
        "startup_grace_seconds": (10.0, 180.0),
        "recovery_cooldown_seconds": (0.0, 30.0),
    },
    "audio": {"volume": (0.0, 1.0)},
}


ACTION_LABELS = {
    "localize": "在永久地图中定位小车",
    "plan_search_route": "规划前往目标区域的安全路线",
    "navigate_to_search_area": "沿航点前往目标附近",
    "capture_at_stop": "停车稳定后拍摄左右相机",
    "visual_detect_and_filter": "调用 YOLO/Qwen 并复核类别与颜色",
    "continue_nearby_search": "未发现时检查附近其他观察点",
    "stop_and_report": "停车并向用户汇报结果",
}

BLE_FAILURE_TOKENS = (
    "BLE 连接失败", "设备名称扫描超时", "GATT 连接与服务发现超时",
    "链路重连失败", "ESP32 未应答", "BLE GATT write exceeded",
    "BLE peripheral disconnected", "通信链路写入失败",
)
BLE_FATAL_TOKENS = (
    "BLE peripheral disconnected", "BLE GATT write exceeded",
    "通信链路写入失败",
)
BLE_SUCCESS_TOKENS = (
    "ESP32 通信通过", "底盘链路连接成功", "自主建图已启动",
    "已加载只读地图", "定位候选", "定位确认", "mission started",
)


def bluetooth_log_health(text: str) -> dict[str, Any]:
    return {
        "failures": sum(text.count(token) for token in BLE_FAILURE_TOKENS),
        "fatal": any(token in text for token in BLE_FATAL_TOKENS),
        "connected": any(token in text for token in BLE_SUCCESS_TOKENS),
    }


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _safe_relative_path(value: Any, prefixes: tuple[str, ...]) -> str:
    text = str(value or "").replace("\\", "/").strip().lstrip("/")
    candidate = (ROOT / text).resolve()
    allowed_roots = [(ROOT / prefix).resolve() for prefix in prefixes]
    if not any(candidate == base or base in candidate.parents for base in allowed_roots):
        raise ValueError(f"路径必须位于 {', '.join(prefixes)} 目录")
    return text


def validate_config(value: dict[str, Any]) -> dict[str, Any]:
    result = _deep_copy(DEFAULT_CONFIG)
    for section in result:
        incoming = value.get(section, {})
        if not isinstance(incoming, dict):
            raise ValueError(f"{section} 必须是对象")
        for key, raw in incoming.items():
            if key not in result[section]:
                raise ValueError(f"不支持的配置项：{section}.{key}")
            if key in RANGES.get(section, {}):
                low, high = RANGES[section][key]
                try:
                    number = float(raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{section}.{key} 必须是数字") from exc
                if not math.isfinite(number) or not low <= number <= high:
                    raise ValueError(f"{section}.{key} 必须在 {low}..{high} 之间")
                result[section][key] = int(number) if isinstance(result[section][key], int) else number
            elif section == "bluetooth":
                if key == "auto_recover":
                    if not isinstance(raw, bool):
                        raise ValueError("bluetooth.auto_recover 必须是布尔值")
                    result[section][key] = raw
                elif key == "device_name":
                    text = str(raw).strip()
                    if not text or len(text) > 80:
                        raise ValueError("蓝牙设备名称不能为空且不能超过 80 个字符")
                    result[section][key] = text
                elif key == "device_address":
                    text = str(raw).strip().upper()
                    parts = text.split(":")
                    if len(parts) != 6 or any(len(part) != 2 or any(c not in "0123456789ABCDEF" for c in part) for part in parts):
                        raise ValueError("蓝牙地址格式不正确")
                    result[section][key] = text
            else:
                prefixes = ("maps",) if section == "mapping" else ("maps", "inspections")
                result[section][key] = _safe_relative_path(raw, prefixes)
    if result["mapping"]["max_drive_speed"] < result["mapping"]["speed"]:
        raise ValueError("建图最大直行速度不能低于巡航速度")
    if result["annotation"]["max_drive_speed"] < result["annotation"]["speed"]:
        raise ValueError("标注最大直行速度不能低于巡航速度")
    if result["mapping"]["stationary_min_support"] > result["mapping"]["stationary_fusion_scans"]:
        raise ValueError("建图最小支持帧数不能大于融合帧数")
    return result


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return _deep_copy(DEFAULT_CONFIG)
    return validate_config(json.loads(path.read_text(encoding="utf-8")))


def save_config(value: dict[str, Any], path: Path = CONFIG_PATH) -> dict[str, Any]:
    checked = validate_config(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return checked


def _latest_mission() -> tuple[Path | None, dict[str, Any]]:
    paths = list((ROOT / "search_runs" / "ble_scene_17").glob("*/mission.json"))
    if not paths:
        return None, {}
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path, {}


def build_task_plan(decision: dict[str, Any], mission: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    events = list(mission.get("events") or [])
    event_types = {str(item.get("type") or "") for item in events if isinstance(item, dict)}
    live_state = str(live.get("command", {}).get("state") or "")
    result = (
        live_state
        if live_state in {"success", "failure", "stopped"}
        else str(mission.get("result") or "running")
    )
    steps = []
    raw_steps = decision.get("steps") if isinstance(decision.get("steps"), list) else []
    completion_events = {
        "localize": {"localized"},
        "plan_search_route": {"route_planned", "route_replanned"},
        "navigate_to_search_area": {"observation_complete"},
        "capture_at_stop": {"observation_complete"},
        "visual_detect_and_filter": {"observation_complete"},
        "continue_nearby_search": {"observation_complete"},
        "stop_and_report": set(),
    }
    active_action = {
        "planning": "plan_search_route", "qwen_planning": "plan_search_route",
        "localizing": "localize", "visual_localizing": "localize",
        "localized": "plan_search_route", "route_planned": "navigate_to_search_area",
        "navigating": "navigate_to_search_area", "searching": "visual_detect_and_filter",
        "capturing": "capture_at_stop", "detecting": "visual_detect_and_filter",
        "observing": "visual_detect_and_filter", "route_replanned": "plan_search_route",
    }.get(live_state, "")
    action_order = list(ACTION_LABELS)
    active_index = action_order.index(active_action) if active_action in action_order else -1
    for index, raw in enumerate(raw_steps):
        action = str(raw.get("action") or "")
        complete = bool(completion_events.get(action, set()) & event_types)
        if action in action_order and active_index > action_order.index(action):
            complete = True
        if action == "stop_and_report" and result in {"success", "failure", "stopped"}:
            complete = True
        status = "completed" if complete else ("active" if action == active_action else "pending")
        steps.append({
            "id": str(raw.get("id") or f"step_{index + 1:02d}"),
            "action": action,
            "title": ACTION_LABELS.get(action, action or "任务步骤"),
            "detail": str(raw.get("success") or ""),
            "status": status,
        })
    if result == "success":
        for step in steps:
            step["status"] = "completed"
    completed = sum(step["status"] == "completed" for step in steps)
    return {
        "instruction": str(decision.get("instruction") or mission.get("instruction") or ""),
        "status": result,
        "summary": str(mission.get("summary") or live.get("command", {}).get("reason") or ""),
        "target": decision.get("target") or {},
        "anchor": decision.get("anchor") or {},
        "steps": steps,
        "completed_steps": completed,
        "total_steps": len(steps),
        "map_url": "/task/map.png",
        "updated_at": 0,
    }


class ManagedProcess:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.kind = "idle"
        self.started_at = 0.0
        self.log_path: Path | None = None
        self._log_handle = None
        self.last_exit_code: int | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            return {
                "kind": self.kind,
                "running": self.process is not None,
                "pid": None if self.process is None else self.process.pid,
                "started_at": self.started_at,
                "log": None if self.log_path is None else str(self.log_path.relative_to(ROOT)),
                "last_exit_code": self.last_exit_code,
            }

    def _refresh(self) -> None:
        if self.process is not None and self.process.poll() is not None:
            self.last_exit_code = self.process.returncode
            self.process = None
            self.kind = "idle"
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def start(self, kind: str, command: list[str]) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            if self.process is not None:
                raise RuntimeError(f"{self.kind} 正在运行，请先停止当前服务")
            logs = ROOT / "service_runs"
            logs.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_path = logs / f"{stamp}_{kind}.log"
            self._log_handle = self.log_path.open("ab", buffering=0)
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdout=self._log_handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.kind = kind
            self.started_at = time.time()
            self.last_exit_code = None
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            process = self.process
            if process is not None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGINT)
                    else:
                        process.send_signal(signal.SIGINT)
                    process.wait(timeout=8)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    process.terminate()
                self.last_exit_code = process.poll()
                self.process = None
                self.kind = "idle"
                if self._log_handle is not None:
                    self._log_handle.close()
                    self._log_handle = None
            return self.snapshot()


def mapping_command(config: dict[str, Any]) -> list[str]:
    m = config["mapping"]
    bluetooth = config["bluetooth"]
    values = {
        "usable-fov": m["usable_fov"], "mapping-max-distance": m["mapping_max_distance"],
        "stationary-map-distance": m["stationary_map_distance"],
        "stationary-map-angle": m["stationary_map_angle"],
        "stationary-settle-scans": m["stationary_settle_scans"],
        "stationary-fusion-scans": m["stationary_fusion_scans"],
        "stationary-min-support": m["stationary_min_support"],
        "stationary-range-tolerance": m["stationary_range_tolerance"],
        "speed": m["speed"], "max-drive-speed": m["max_drive_speed"],
        "slow-speed": m["slow_speed"], "turn-speed": m["turn_speed"],
        "max-turn-speed": m["max_turn_speed"], "stop-distance": m["stop_distance"],
        "clearance": m["clearance"], "map-size": m["map_size"],
        "max-runtime-min": m["max_runtime_min"], "output": m["output"],
    }
    command = [sys.executable, "-u", "-m", "raspberry_pi.autonomous_mapping", "--enable-motion",
               "--link", "ble", "--ble-name", bluetooth["device_name"],
               "--ble-address", bluetooth["device_address"],
               "--lidar-port", "/dev/ttyACM0", "--wall-gap-max", "0", "--min-obstacle-area", "0.01",
               "--front-left-gain", "1.15", "--front-right-gain", "1.00",
               "--rear-left-gain", "0.90", "--rear-right-gain", "1.00",
               "--turn-arc-speed", "230", "--narrow-passage-width", "0.95", "--narrow-speed", "240",
               "--narrow-turn-speed", "1700", "--link-reconnect-rounds", "0", "--save-every", "0", "--live-map"]
    for key, value in values.items():
        command.extend((f"--{key}", str(value)))
    return command


def annotation_command(config: dict[str, Any]) -> list[str]:
    a = config["annotation"]
    bluetooth = config["bluetooth"]
    values = {
        "map": a["map"], "map-yaml": a["map_yaml"], "semantic-map": a["semantic_map"],
        "output": a["output"], "clearance": a["clearance"], "speed": a["speed"],
        "max-drive-speed": a["max_drive_speed"], "slow-speed": a["slow_speed"],
        "turn-speed": a["turn_speed"], "stop-distance": a["stop_distance"],
        "standoff": a["standoff"], "max-targets": a["max_targets"],
        "photos-per-target": a["photos_per_target"],
        "settle-seconds": a["settle_seconds"], "camera-width": a["camera_width"],
        "camera-height": a["camera_height"], "max-runtime-min": a["max_runtime_min"],
    }
    command = [sys.executable, "-u", "-m", "raspberry_pi.map_camera_inspection", "--enable-motion",
               "--link", "ble", "--ble-name", bluetooth["device_name"],
               "--ble-address", bluetooth["device_address"],
               "--lidar-port", "/dev/ttyACM0", "--lidar-offset-x", "0.14",
               "--front-left-gain", "1.15", "--front-right-gain", "1.00",
               "--rear-left-gain", "0.90", "--rear-right-gain", "1.00",
               "--drive-speed-step", "40", "--command-min-interval-ms", "450",
               "--localization-timeout-seconds", "120", "--localization-required-moves", "2",
               "--camera-backend", "opencv", "--dual-usb-cameras"]
    for key, value in values.items():
        command.extend((f"--{key}", str(value)))
    return command


class CompanionService:
    def __init__(self, config_path: Path = CONFIG_PATH) -> None:
        self.config_path = config_path
        self.config = load_config(config_path)
        self.runner = ManagedProcess()
        self.emergency_stop = False
        self.reply = ""
        self.reply_id = ""
        self._bluetooth_lock = threading.RLock()
        self._bluetooth_generation = 0
        self._bluetooth_status: dict[str, Any] = {
            "running": False,
            "last_ok": None,
            "last_message": "尚未执行蓝牙恢复",
            "last_reason": "",
            "last_started_at": 0.0,
            "last_finished_at": 0.0,
            "attempt": 0,
        }

    def bluetooth_status(self) -> dict[str, Any]:
        with self._bluetooth_lock:
            return dict(self._bluetooth_status)

    def recover_bluetooth(self, reason="manual", restart=None) -> dict[str, Any]:
        """Recover asynchronously so the Android HTTP request never times out."""
        with self._bluetooth_lock:
            if self._bluetooth_status["running"]:
                return self.bluetooth_status()
            self._bluetooth_generation += 1
            generation = self._bluetooth_generation
            self._bluetooth_status.update({
                "running": True,
                "last_reason": str(reason),
                "last_started_at": time.time(),
                "last_message": "正在安全停止运动任务并重置蓝牙",
            })

        def worker():
            try:
                self.runner.stop()
                bluetooth = self.config["bluetooth"]
                result = recover_bluetooth(bluetooth["device_address"])
                with self._bluetooth_lock:
                    self._bluetooth_status.update({
                        "running": False,
                        "last_ok": result.ok,
                        "last_message": result.message,
                        "last_finished_at": time.time(),
                        "output": result.output,
                    })
                if result.ok and restart is not None:
                    delay = float(bluetooth["recovery_cooldown_seconds"])
                    if delay:
                        time.sleep(delay)
                    with self._bluetooth_lock:
                        current = generation == self._bluetooth_generation
                    if current:
                        self.start_managed_service(
                            restart[0], restart[1], monitor=True,
                            recovery_restart=True,
                        )
            except Exception as exc:  # keep the bridge alive and report to phone
                with self._bluetooth_lock:
                    self._bluetooth_status.update({
                        "running": False,
                        "last_ok": False,
                        "last_message": f"蓝牙恢复异常：{exc}",
                        "last_finished_at": time.time(),
                    })

        threading.Thread(target=worker, name="bluetooth-recovery", daemon=True).start()
        return self.bluetooth_status()

    def start_managed_service(
        self, kind, command, monitor=True, recovery_restart=False
    ):
        if not recovery_restart:
            with self._bluetooth_lock:
                self._bluetooth_status["attempt"] = 0
        result = self.runner.start(kind, command)
        if monitor and self.config["bluetooth"]["auto_recover"]:
            with self._bluetooth_lock:
                self._bluetooth_generation += 1
                generation = self._bluetooth_generation
            threading.Thread(
                target=self._watch_bluetooth_process,
                args=(generation, kind, list(command), result.get("log")),
                name=f"ble-watch-{kind}",
                daemon=True,
            ).start()
        return result

    def _watch_bluetooth_process(self, generation, kind, command, relative_log):
        """Watch BLE for the complete task, not only its connection phase."""
        bluetooth = self.config["bluetooth"]
        log_path = ROOT / relative_log if relative_log else None
        started = time.monotonic()
        connected_reported = False
        while True:
            time.sleep(1.0)
            with self._bluetooth_lock:
                if generation != self._bluetooth_generation:
                    return
            text = ""
            try:
                if log_path and log_path.exists():
                    text = log_path.read_text(encoding="utf-8", errors="replace")[-64000:]
            except OSError:
                pass
            health = bluetooth_log_health(text)
            process = self.runner.snapshot()
            if health["connected"] and not connected_reported:
                connected_reported = True
                with self._bluetooth_lock:
                    self._bluetooth_status.update({
                        "last_ok": True,
                        "last_message": "BLE 已连接，正在持续监控运行期链路",
                    })
            startup_expired = (
                time.monotonic() - started
                >= float(bluetooth["startup_grace_seconds"])
            )
            should_recover = (
                health["fatal"]
                or health["failures"] >= int(bluetooth["failure_threshold"])
                or (not process["running"] and health["failures"] > 0)
            )
            if should_recover:
                attempt = int(self.bluetooth_status().get("attempt", 0)) + 1
                if attempt > int(bluetooth["max_recovery_attempts"]):
                    with self._bluetooth_lock:
                        self._bluetooth_status.update({
                            "last_ok": False,
                            "last_message": "BLE 自动恢复次数已用完，请检查 ESP32 供电",
                            "attempt": attempt - 1,
                        })
                    return
                with self._bluetooth_lock:
                    self._bluetooth_status["attempt"] = attempt
                phase = "运行中" if connected_reported else "启动阶段"
                self.recover_bluetooth(
                    reason=f"{kind} {phase}检测到 BLE 断连",
                    restart=(kind, command),
                )
                return
            if not process["running"]:
                return
            if startup_expired and not connected_reported:
                # The process can be busy with Qwen planning before opening
                # BLE. Keep watching it; lack of a success line is not itself
                # proof of a broken controller.
                continue

    def _watch_bluetooth_startup(self, generation, kind, command, relative_log):
        bluetooth = self.config["bluetooth"]
        deadline = time.monotonic() + float(bluetooth["startup_grace_seconds"])
        failure_tokens = (
            "BLE 连接失败", "设备名称扫描超时", "GATT 连接与服务发现超时",
            "链路重连失败", "ESP32 未应答", "BLE GATT write exceeded",
        )
        success_tokens = (
            "ESP32 通信通过", "自主建图已启动", "已加载只读地图",
            "定位确认", "mission started", "任务开始",
        )
        log_path = ROOT / relative_log if relative_log else None
        while time.monotonic() < deadline:
            time.sleep(1.0)
            with self._bluetooth_lock:
                if generation != self._bluetooth_generation:
                    return
            text = ""
            try:
                if log_path and log_path.exists():
                    text = log_path.read_text(encoding="utf-8", errors="replace")[-32000:]
            except OSError:
                pass
            if any(token in text for token in success_tokens):
                with self._bluetooth_lock:
                    self._bluetooth_status.update({
                        "last_ok": True,
                        "last_message": "BLE 已连接，运动服务启动正常",
                        "attempt": 0,
                    })
                return
            failures = sum(text.count(token) for token in failure_tokens)
            process = self.runner.snapshot()
            exited_with_ble_error = not process["running"] and failures > 0
            if failures >= int(bluetooth["failure_threshold"]) or exited_with_ble_error:
                attempt = int(self.bluetooth_status().get("attempt", 0)) + 1
                if attempt > int(bluetooth["max_recovery_attempts"]):
                    with self._bluetooth_lock:
                        self._bluetooth_status.update({
                            "last_ok": False,
                            "last_message": "蓝牙自动恢复次数已用完，请检查 ESP32 供电",
                            "attempt": attempt - 1,
                        })
                    return
                with self._bluetooth_lock:
                    self._bluetooth_status["attempt"] = attempt
                self.recover_bluetooth(
                    reason=f"{kind} 启动阶段检测到 {failures} 次 BLE 异常",
                    restart=(kind, command),
                )
                return

    def state(self) -> dict[str, Any]:
        try:
            live = json.loads(LIVE_STATUS_PATH.read_text(encoding="utf-8")) if LIVE_STATUS_PATH.exists() else {}
        except (OSError, json.JSONDecodeError):
            live = {}
        try:
            decision = json.loads(DECISION_PATH.read_text(encoding="utf-8")) if DECISION_PATH.exists() else {}
        except (OSError, json.JSONDecodeError):
            decision = {}
        mission_path, mission = _latest_mission()
        process = self.runner.snapshot()
        # A newly submitted search must not inherit the success/failure state of
        # the previous run while Qwen is still producing its decision file.
        try:
            mission_is_stale = mission_path is None or mission_path.stat().st_mtime < process["started_at"]
        except OSError:
            mission_is_stale = True
        try:
            live_is_stale = not LIVE_STATUS_PATH.exists() or LIVE_STATUS_PATH.stat().st_mtime < process["started_at"]
        except OSError:
            live_is_stale = True
        new_search_running = process["running"] and process["kind"] == "object_search"
        if new_search_running and mission_is_stale:
            mission = {}
        if new_search_running and live_is_stale:
            live = {
                "command": {"state": "planning", "reason": "Qwen 正在理解指令并生成任务步骤"},
                "task": {"instruction": str(decision.get("instruction") or "")},
            }
        plan = build_task_plan(decision, mission, live)
        try:
            plan["updated_at"] = LIVE_MAP_PATH.stat().st_mtime_ns // 1_000_000
        except OSError:
            plan["updated_at"] = 0
        phase = str(live.get("command", {}).get("state") or process["kind"] or "idle")
        active = bool(process["running"] or live.get("running", False))
        completed = str(plan.get("status")) == "success"
        task_type = process["kind"]
        if task_type == "idle" and live.get("running", False):
            task_type = "object_search_external"
        return {
            "frame_index": int(time.time() * 10),
            "camera": {"connected": True},
            "grounding_status": plan.get("summary") or ("任务服务运行中" if active else "等待任务"),
            "obstacles": {"status": "雷达保护已启用" if active else "未运行"},
            "lidar": {"status": "运行中" if active else "待机"},
            "emergency_stop": self.emergency_stop,
            "robot_command": live.get("command") or {"action": phase, "reason": plan.get("summary", "")},
            "robot_task": {
                "active": active, "task_type": task_type, "phase": phase,
                "phase_label": plan.get("summary") or phase, "completed": completed,
                "queue_total": plan.get("total_steps", 0),
            },
            "target_queue": [
                {"target_name": step["title"], "status": "done" if step["status"] == "completed" else step["status"],
                 "active": step["status"] == "active"}
                for step in plan.get("steps", [])
            ],
            "task_plan": plan,
            "services": {"process": process, "bluetooth": self.bluetooth_status()},
            "assistant_message": self.reply,
            "reply_id": self.reply_id,
        }

    def set_config(self, value: dict[str, Any]) -> dict[str, Any]:
        self.config = save_config(value, self.config_path)
        return self.config

    def start_search(self, instruction: str) -> dict[str, Any]:
        if not instruction.strip():
            raise ValueError("任务指令不能为空")
        if self.runner.snapshot()["running"]:
            self.runner.stop()
        self.emergency_stop = False
        DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
        planning_decision = {
            "instruction": instruction.strip(),
            "status": "planning",
            "target": {},
            "anchor": {},
            "steps": [
                {"id": f"step_{index:02d}", "action": action, "success": ACTION_LABELS[action]}
                for index, action in enumerate(ACTION_LABELS, start=1)
            ],
        }
        DECISION_PATH.write_text(
            json.dumps(planning_decision, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = self.start_managed_service(
            "object_search", [str(ROOT / "natural_search.sh"), instruction.strip(), "real"]
        )
        self.reply = "已接收指令，正在分析任务并生成路线"
        self.reply_id = f"task-{int(time.time() * 1000)}"
        return result


def _handler(service: CompanionService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RobotCompanion/1.0"

        def _json(self, value: Any, status: int = 200) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > 1024 * 1024:
                raise ValueError("请求过大")
            raw = self.rfile.read(length) if length else b"{}"
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON 根节点必须是对象")
            return value

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/health":
                self._json({"status": "ok", "service": "robot_companion"})
            elif path == "/state":
                self._json(service.state())
            elif path == "/config":
                self._json({"ok": True, "config": service.config})
            elif path == "/bluetooth/status":
                self._json({"ok": True, "bluetooth": service.bluetooth_status()})
            elif path == "/task/plan":
                self._json({"ok": True, "task_plan": service.state()["task_plan"]})
            elif path == "/task/map.png":
                self._file(LIVE_MAP_PATH)
            else:
                self._json({"type": "error", "message": "接口不存在"}, 404)

        def _file(self, path: Path) -> None:
            if not path.exists():
                self._json({"type": "error", "message": "任务地图尚未生成"}, 404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                payload = self._body()
                if path == "/control":
                    kind = str(payload.get("type") or "")
                    if kind == "set_instruction":
                        value = service.start_search(str(payload.get("instruction") or ""))
                        self._json({"type": "ack", "message": "任务规划已开始", "process": value})
                    elif kind in {"reset", "emergency_stop"}:
                        enabled = bool(payload.get("enabled", True)) if kind == "emergency_stop" else True
                        if enabled:
                            service.runner.stop()
                        service.emergency_stop = enabled
                        self._json({"type": "ack", "message": "小车已停车" if enabled else "急停已解除"})
                    else:
                        raise ValueError("不支持的控制类型")
                elif path == "/config":
                    self._json({"ok": True, "config": service.set_config(payload.get("config", payload))})
                elif path == "/services/mapping/start":
                    self._json({"ok": True, "process": service.start_managed_service("mapping", mapping_command(service.config))})
                elif path == "/services/annotation/start":
                    self._json({"ok": True, "process": service.start_managed_service("annotation", annotation_command(service.config))})
                elif path == "/services/stop":
                    self._json({"ok": True, "process": service.runner.stop()})
                elif path == "/bluetooth/recover":
                    self._json({
                        "ok": True,
                        "message": "蓝牙恢复已开始，手机桥接服务会保持在线",
                        "bluetooth": service.recover_bluetooth("phone_manual"),
                    })
                else:
                    self._json({"type": "error", "message": "接口不存在"}, 404)
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                self._json({"type": "error", "message": str(exc)}, 400)

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[companion] " + fmt % args, flush=True)

    return Handler


def run(host: str, port: int, config_path: Path = CONFIG_PATH) -> None:
    service = CompanionService(config_path)
    server = ThreadingHTTPServer((host, int(port)), _handler(service))
    print(f"Robot Companion API: http://{host}:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        service.runner.stop()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RobotCompanionAndroid bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.host, args.port, args.config)


if __name__ == "__main__":
    main()
