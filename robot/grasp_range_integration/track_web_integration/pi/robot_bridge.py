"""Safe Raspberry Pi HTTP bridge for the final_work web agent.

Only fixed, validated actions are exposed.  The bridge never accepts shell
commands from the network.  Existing mapping, pursuit and grasp programs stay
the owners of hardware behaviour; this process coordinates their lifecycle and
proxies map/camera telemetry to the 3090 web service.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
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
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import parse_qs, urlparse

from .map_navigation_adapter import MapNavigationAdapter


TRACK_ROOT = Path(os.environ.get("FINAL_WORK_TRACK_ROOT", "/home/dachuang/dachuang/track"))
PYTHON = Path(os.environ.get("FINAL_WORK_PI_PYTHON", "/home/dachuang/dachuang/.venv/bin/python"))
GRAB_ROOT = Path(os.environ.get("FINAL_WORK_GRAB_ROOT", "/home/dachuang/grab_visual"))
LEGACY_MAP_ROOT = Path(os.environ.get("FINAL_WORK_LEGACY_MAP_ROOT", "/home/dachuang/dachuang/car1.0"))
STATIC_MAP_PATH = Path(os.environ.get(
    "FINAL_WORK_STATIC_MAP_PATH",
    str(LEGACY_MAP_ROOT / "maps" / "ble_scene_17.pgm"),
))
STATIC_MAP_RESOLUTION_M = float(os.environ.get("FINAL_WORK_STATIC_MAP_RESOLUTION_M", "0.05"))
STATIC_MAP_ORIGIN = tuple(float(value) for value in os.environ.get(
    "FINAL_WORK_STATIC_MAP_ORIGIN", "-10,-10,0"
).split(","))
LOG_ROOT = Path(os.environ.get("FINAL_WORK_PI_LOG_DIR", str(TRACK_ROOT / "final_work" / "logs")))
COMPANION_URL = os.environ.get("FINAL_WORK_COMPANION_URL", "http://127.0.0.1:8770").rstrip("/")
MAP_URL = os.environ.get("FINAL_WORK_MAP_URL", "http://127.0.0.1:8766").rstrip("/")
GRAB_WEB_URL = os.environ.get("FINAL_WORK_GRAB_WEB_URL", "http://127.0.0.1:8080").rstrip("/")
PURSUIT_HTTP = os.environ.get("FINAL_WORK_PURSUIT_HTTP", "http://192.168.55.33:8021").rstrip("/")
PURSUIT_WS = os.environ.get("FINAL_WORK_PURSUIT_WS", "ws://192.168.55.33:8021/pursuit_ws")
HARDWARE_ENABLED = os.environ.get("FINAL_WORK_ENABLE_HARDWARE", "0") == "1"
HARDWARE_TOKEN = os.environ.get("FINAL_WORK_HARDWARE_TOKEN", "")


def _align_teaching_hull(target: str, *, real: bool,
                         cancel_event: threading.Event | None = None,
                         deadline: float | None = None) -> dict[str, Any]:
    """Fresh binocular check; move only short, radar-guarded pulses if needed.

    Import lazily so map/status requests never initialise motion dependencies.
    The arm endpoint checks a *new* frame again before enqueuing its sequence.
    """
    from pursuit.hull_micro_adjust import align, verify_model_camera_order
    from .pursuit_recovery import align_with_loss_recovery
    from .pursuit_session import _prepare_loss_recovery
    from .motion_session import _event

    verify_model_camera_order()
    cancelled = cancel_event if cancel_event is not None else threading.Event()
    deadline = min(deadline, time.monotonic() + 180.0) if deadline is not None else time.monotonic() + 180.0
    return align_with_loss_recovery(target, real=real, cancelled=cancelled,
        align=lambda: align(target, real=real, grab_url=GRAB_WEB_URL,
                            cancel_event=cancelled,
                            deadline_s=min(120.0, max(1.0, deadline-time.monotonic()))).__dict__,
        prepare=lambda: _prepare_loss_recovery(cancelled, GRAB_WEB_URL), emit=_event,
        deadline=deadline)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_text(value: Any, *, name: str, limit: int = 300) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name}不能为空")
    if len(text) > limit or any(ord(char) < 32 and char not in "\t\n" for char in text):
        raise ValueError(f"{name}过长或包含控制字符")
    return text


def _pursuit_command(instruction: str, *, dry_run: bool,
                     swap_cameras: bool | None = None,
                     grasp: bool = False) -> list[str]:
    """Build the visual-pursuit command with an explicit motion mode.

    ``run_visual_pursuit`` intentionally defaults to dry-run unless
    ``--esp32`` is present.  Omitting both flags therefore looks healthy in
    the UI and produces motion decisions, but never writes them to the wheels.
    """
    command = [
        str(PYTHON), "-u", "-m", "final_work.pi.pursuit_session",
        *(["--grasp"] if grasp else []), "--",
        "--instruction", instruction,
        "--server-http", PURSUIT_HTTP,
        "--server-ws", PURSUIT_WS,
        # Agent-owned runs must terminate when pursuit reaches success/give-up.
        # Otherwise the camera preview loop keeps the managed process alive and
        # the web task remains in "running" forever.
        "--exit-on-finish",
    ]
    command.append("--dry-run" if dry_run else "--esp32")
    # Pin acquisition order instead of depending on auto-probe enumeration.
    from pursuit.camera_contract import model_devices
    model_pair=model_devices(require_ports=sys.platform == 'linux')
    command.extend(["--left-camera", str(model_pair[1]), "--right-camera", str(model_pair[0])])
    if swap_cameras is not None:
        if not isinstance(swap_cameras, bool):
            raise ValueError("相机交换配置必须是布尔值")
        command.append("--swap-cameras" if swap_cameras else "--no-swap-cameras")
    return command


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 3.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body = json.loads(raw)
            detail = body.get("error") or body.get("detail") or raw
        except (json.JSONDecodeError, AttributeError):
            detail = raw or str(exc)
        raise RuntimeError(f"上游服务 {exc.code}: {detail}") from exc
    return value if isinstance(value, dict) else {"value": value}


def _phase1_handoff_from_log(log_path: Path, *, expected_instruction: str,
                             expected_task_id: str | None,
                             car1_root: Path) -> dict[str, Any]:
    """Bind the next pursuit to this completed map-search process, never an old run."""
    marker = "寻物交接验证通过："
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    announcements = [line.split(marker, 1)[1] for line in lines if marker in line]
    if len(announcements) != 1:
        raise RuntimeError("本轮地图寻物缺少唯一的已验证交接记录")
    announcement = json.loads(announcements[0])
    output_root = (car1_root / "search_runs/ble_scene_17").resolve(strict=True)
    mission_dir = Path(str(announcement["mission_dir"])).resolve(strict=True)
    if mission_dir.parent != output_root:
        raise RuntimeError("寻物交接目录不属于本轮任务输出根目录")
    handoff = json.loads((mission_dir / "phase1_verified_handoff.json").read_text(
        encoding="utf-8"))
    if (handoff.get("mission_dir") != str(mission_dir)
            or handoff.get("instruction") != expected_instruction
            or handoff.get("decision_task_id") != expected_task_id
            or (handoff.get("pursuit_entry") or {}).get("ready") is not True
            or handoff.get("chassis_mission_exited") is not True):
        raise RuntimeError("寻物交接与本轮指令或后续追踪入口不一致")
    for photo in (handoff.get("images") or {}).values():
        if not Path(str(photo)).resolve(strict=True).is_relative_to(mission_dir):
            raise RuntimeError("寻物交接照片越界")
    if set((handoff.get("images") or {})) != {"left", "right"}:
        raise RuntimeError("寻物交接缺少双摄原图")
    return handoff


def _capture_handoff_frames(handoff: dict[str, Any], pursuit_log: str,
                            evidence_dir: Path, baseline_digest: str | None) -> None:
    """Passively archive early pursuit frames; never issues a motion command."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "phase1_pursuit_evidence.v1",
        "phase1_handoff": handoff,
        "pursuit_log": pursuit_log,
        "pursuit_frames": [],
        "note": "追踪照片为服务画面，仅供排障；第二阶段仍须使用新帧重新识别，不能以旧框驱动车辆。",
    }
    seen = {baseline_digest} if baseline_digest else set()
    deadline = time.monotonic() + 35.0
    last_error = ""
    while len(report["pursuit_frames"]) < 3 and time.monotonic() < deadline:
        try:
            request = urlrequest.Request(f"{PURSUIT_HTTP}/frame.jpg", method="GET")
            with urlrequest.urlopen(request, timeout=2.0) as response:
                frame = response.read(5_000_000)
            if not (frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")):
                raise ValueError("追踪画面不是完整 JPEG")
            digest = hashlib.sha256(frame).hexdigest()
            if digest not in seen:
                seen.add(digest)
                try:
                    status = _http_json(f"{PURSUIT_HTTP}/status", timeout=1.0)
                except Exception as exc:
                    status = {"status_unavailable": str(exc)[:200]}
                photo = evidence_dir / f"pursuit_{len(report['pursuit_frames']) + 1:02d}.jpg"
                photo.write_bytes(frame)
                report["pursuit_frames"].append({
                    "path": str(photo), "captured_at_ms": _now_ms(),
                    "sha256": digest, "pursuit_status": status,
                })
        except Exception as exc:
            last_error = str(exc)[:300]
        time.sleep(0.6)
    if last_error and not report["pursuit_frames"]:
        report["capture_error"] = last_error
    grounding_lines: list[str] = []
    while True:
        try:
            lines = Path(pursuit_log).read_text(encoding="utf-8", errors="replace").splitlines()
            grounding_lines = [line[-500:] for line in lines if any(
                marker in line.lower() for marker in (
                    "/ground", '"found"', "found:", "目标未通过"))][-30:]
        except OSError:
            pass
        if grounding_lines or time.monotonic() >= deadline:
            break
        time.sleep(0.7)
    report["grounding_log_lines"] = grounding_lines
    (evidence_dir / "handoff_evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ManagedProcess:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._kind = "idle"
        self._started_at = 0.0
        self._log_path: Path | None = None
        self._log_handle = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            result = {
                "kind": self._kind if running else "idle",
                "running": running,
                "pid": self._process.pid if running and self._process else None,
                "returncode": None if running or self._process is None else self._process.returncode,
                "started_at": self._started_at,
                "log": str(self._log_path) if self._log_path else None,
            }
            if running:
                message = self._live_status_message()
                if message:
                    result["status_message"] = message
            if not running and result["returncode"] not in {None, 0}:
                result["error"] = self._diagnostic()
            return result

    def _live_status_message(self) -> str:
        """Return the newest status owned by the active managed process.

        The companion service can retain an old mapping/BLE failure after a
        visual-pursuit session starts.  Exposing the managed process' own log
        prevents the web UI from presenting that stale error as current.
        """
        lines = [line.strip() for line in self.tail(12000).splitlines()
                 if line.strip()]
        for line in reversed(lines):
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {}
                if event.get("event") == "arc_retreat_started":
                    return "转向侧空间不足：已停车，正在执行一次约 7cm 的限幅短回退"
                if event.get("event") == "short_reverse_started":
                    return (f"限幅后退 {event.get('target_m', 0)}m："
                            "单次短脉冲执行中，后向雷达不用于净空判断")
                if event.get("event") == "short_reverse_result":
                    return (f"短后退已停车：请求 {event.get('target_m', 0)}m，"
                            f"雷达实测 {event.get('measured_m', 0)}m")
                if event.get("event") == "arc_retreat_result":
                    if event.get("recovered"):
                        return (f"短回退已确认 {event.get('measured_m', 0)}m，"
                                f"转向净空 {event.get('clearance_m', 0)}m；继续画弧")
                    return (f"短回退后仍未满足继续条件：位移 {event.get('measured_m', 0)}m，"
                            f"净空 {event.get('clearance_m', 0)}m；停车")
                if event.get("event") == "arc_turn_progress":
                    direction = "左转" if event.get("direction") == "left" else "右转"
                    return (f"前进画弧{direction}：{event.get('angle_deg', 0)}°/"
                            f"{event.get('target_deg', 0)}°，前进 {event.get('travel_m', 0)}m，"
                            f"净空 {event.get('clearance_m', 0)}m")[:500]
                if event.get("event") == "arc_segment_settled":
                    return (f"第 {event.get('segment')} 段已停车复测："
                            f"累计 {event.get('angle_deg', 0)}°，"
                            f"停车期间补计 {event.get('extra_deg', 0)}°")
                if event.get("event") == "arc_predictive_stop":
                    return (f"第 {event.get('segment')} 段提前停车："
                            f"当前 {event.get('angle_deg', 0)}°，"
                            f"预计下一帧会越过 {event.get('segment_goal_deg', 0)}°")
                if event.get("event") == "arc_segment_started":
                    return f"前进画弧第 {event.get('segment')} 段，正复测雷达与实际转角"
            if line.startswith(("[link]", "[pursuit] t=", "[pursuit] 任务结束", "[phase]")):
                return line[:500]
        return ""

    def _diagnostic(self) -> str:
        text = self.tail(16000)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            if line.startswith("[phase] 任务停止:"):
                return line.split(":", 1)[1].strip()[:500]
            if '"event": "pursuit_session_failed"' in line:
                try:
                    return str(json.loads(line).get("reason") or line)[:500]
                except json.JSONDecodeError:
                    pass
            if line.startswith("[pursuit] 任务结束:"):
                return line[:500]
            if '"event": "pursuit_hull_handoff"' in line:
                try:
                    event = json.loads(line)
                    if not event.get("ok"):
                        return str(event.get("reason") or line)[:500]
                except json.JSONDecodeError:
                    pass
            if line.startswith("RuntimeError:") or line.startswith("ValueError:"):
                return line.split(":", 1)[1].strip()
            if '"event": "ble_connect_failed"' in line:
                try:
                    return f"ESP32 蓝牙连接失败：{json.loads(line).get('error', '未知错误')}"
                except json.JSONDecodeError:
                    pass
        return lines[-1][-500:] if lines else "底层进程异常退出，日志中没有错误文本"

    def start(
        self,
        kind: str,
        command: list[str],
        cwd: Path,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            current = self.snapshot()
            if current["running"]:
                raise RuntimeError(f"已有任务正在运行：{current['kind']}")
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._log_path = LOG_ROOT / f"{stamp}_{kind}.log"
            self._log_handle = self._log_path.open("a", encoding="utf-8")
            self._process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=__import__(__package__+'.camera_proxy',fromlist=['camera_environment']).camera_environment(
                    {**os.environ, "PYTHONUNBUFFERED": "1", **(extra_env or {})}),
            )
            self._kind = kind
            self._started_at = time.time()
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=2)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            return self.snapshot()

    def tail(self, max_bytes: int = 24000) -> str:
        with self._lock:
            path = self._log_path
        if path is None or not path.exists():
            return ""
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")

    def logs(self, limit: int = 30) -> list[dict[str, Any]]:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        active = self.snapshot().get("log")
        files = sorted(LOG_ROOT.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)
        return [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "modified_at": int(path.stat().st_mtime * 1000),
                "active": str(path) == active,
            }
            for path in files[:max(1, min(int(limit), 100))]
        ]

    def read_log(self, name: str, max_bytes: int = 200000) -> str:
        safe_name = Path(str(name)).name
        if safe_name != str(name) or not safe_name.endswith(".log"):
            raise ValueError("日志文件名无效")
        path = LOG_ROOT / safe_name
        if not path.is_file():
            raise ValueError("日志不存在")
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")


class RobotBridge:
    def __init__(self) -> None:
        self.process = ManagedProcess()
        self.map_navigation = MapNavigationAdapter()
        self.started_at = time.time()
        self._grasp_lock = threading.Lock()
        self._grasp_cancel = threading.Event()
        self._phase1_pending: dict[str, Any] | None = None
        self._phase1_verified_target: str | None = None
        from .robot_skills import RobotSkills
        self.skills = RobotSkills(self)
        from .task_video import TaskVideoManager
        self.video = TaskVideoManager(TRACK_ROOT, PYTHON)

    def start_recording(self,payload,headers):
        self._authorise_hardware(headers)
        session_id=_safe_text(payload.get('session_id'),name='录像会话',limit=160)
        current=self.video.status()
        if current.get('recording') and current.get('session_id') == session_id:
            return {"ok": True, **current}
        managed=self.process.snapshot()
        if self._grasp_lock.locked() or (managed.get('running') and managed.get('kind') != 'camera_preview'):
            raise RuntimeError('任务正在运行，拒绝重启并清理录像')
        arm=self._probe(GRAB_WEB_URL+'/api/arm-teach/exec')
        if arm.get('connected') and (arm.get('data',{}).get('status') or {}).get('active'):
            raise RuntimeError('机械臂正在运行，拒绝重启录像')
        if current.get('recording'):
            self.video.finish(current['session_id'],'superseded_by_new_idle_task')
        if managed.get('running'):self.process.stop()
        return {"ok": True, **self.video.start(session_id,str(payload.get('instruction') or ''))}

    @staticmethod
    def _probe(url: str) -> dict[str, Any]:
        try:
            return {"connected": True, "data": _http_json(url, timeout=1.2)}
        except Exception as exc:
            return {"connected": False, "error": str(exc)}

    def state(self) -> dict[str, Any]:
        semantic = self.semantic_map()
        map_probe = self._probe(f"{MAP_URL}/api/status")
        live_navigation = self.map_navigation.live_status()
        if not map_probe["connected"] and live_navigation:
            map_probe = {"connected": True, "source": "car1_live_status", "data": live_navigation}
        return {
            "service": "final_work_pi_bridge",
            "time_ms": _now_ms(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "hardware_enabled": HARDWARE_ENABLED,
            "managed": self.process.snapshot(),
            "companion": self._probe(f"{COMPANION_URL}/state"),
            "map": map_probe,
            "static_map": {
                "source_path": str(STATIC_MAP_PATH),
                "resolution_m": STATIC_MAP_RESOLUTION_M,
                "origin": STATIC_MAP_ORIGIN,
            },
            "semantic": {
                "status": semantic.get("_status", "available"),
                "source_path": semantic.get("_source_path"),
                "entity_count": semantic.get("_entity_count", 0),
                "release_available": semantic.get("_release_available", False),
                "warning": semantic.get("_warning", ""),
            },
            "navigation": live_navigation,
            "grasp": self._probe(f"{GRAB_WEB_URL}/api/status"),
            "robot_skills": self.skills.snapshot(),
            "task_recording": self.video.status(),
            "interfaces": {
                "observation_viewpoint_selection": "connected_car1_search_route",
                "text_map_navigation": "connected_car1_object_search",
                "visual_pursuit": "connected",
                "visual_grasp": "connected",
            },
        }

    def _authorise_hardware(self, headers: Any) -> None:
        if not HARDWARE_ENABLED:
            raise PermissionError("桥接服务处于安全模式；未启用实车动作")
        supplied = str(headers.get("X-Robot-Safety-Token") or "")
        if not HARDWARE_TOKEN or supplied != HARDWARE_TOKEN:
            raise PermissionError("实车动作缺少有效安全令牌")

    def start_preview(self) -> dict[str, Any]:
        managed = self.process.snapshot()
        if managed.get('running') and managed.get('kind') != 'camera_preview':
            raise RuntimeError('实车任务仍在运行，不能重建相机预览')
        # A bridge restart also closes its broker. An HTTP-live preview can
        # therefore have zero frames; recreate the owner, without recording.
        self.video.ensure_camera_owner()
        if self._probe(f"{GRAB_WEB_URL}/api/status")["connected"]:
            return {"ok": True, "already_running": True}
        value = self.process.start("camera_preview", ["bash", str(GRAB_ROOT / "start_web.sh")], GRAB_ROOT,
                                   extra_env={"LEFT_CAM":"2", "RIGHT_CAM":"0"})
        return {"ok": True, "process": value}

    def probe_current_view(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Read-only target check for planning; never sends a chassis/arm command."""
        target = _safe_text(payload.get("target"), name="视觉目标")
        managed = self.process.snapshot()
        if managed.get("running") and managed.get("kind") not in {"idle", "camera_preview"}:
            return {"status": "unavailable", "reason": "robot_busy"}
        status = self._probe(f"{GRAB_WEB_URL}/api/status")
        if not status["connected"]:
            try:
                self.start_preview()  # passive cameras only; no wheel/arm action
            except Exception as exc:
                return {"status": "unavailable", "reason": f"camera_preview_start_failed: {exc}"[:300]}
        deadline = time.monotonic() + 9.0
        while time.monotonic() < deadline:
            status = self._probe(f"{GRAB_WEB_URL}/api/status")
            data = status.get("data") or {}
            cameras = data.get("cameras") or {}
            if status["connected"] and all(
                (cameras.get(side) or {}).get("frames", 0) > 0
                and 0 <= float((cameras.get(side) or {}).get("age_s", 99)) < 1.0
                for side in ("left", "right")
            ):
                break
            time.sleep(0.25)
        else:
            return {"status": "unavailable", "reason": "binocular_preview_not_ready",
                    "observed_at_ms": _now_ms()}
        ble_ready = data.get("ble") is True
        from pursuit.hull_micro_adjust import TargetNotVisible, _observe, in_grasp_region

        try:
            observation = _observe(target, grab_url=GRAB_WEB_URL,
                                   timeout=25.0, require_ble=False)
        except TargetNotVisible:
            return {"status": "not_visible_both", "ble_ready": ble_ready,
                    "observed_at_ms": _now_ms()}
        except Exception as exc:
            # A model timeout or stale camera is not evidence that the object
            # is absent.  Keep the original plan and expose the diagnostic.
            return {"status": "unavailable", "reason": str(exc)[:300], "ble_ready": ble_ready,
                    "observed_at_ms": _now_ms()}
        return {
            "status": "visible_both",
            "inside_grasp_region": bool(in_grasp_region(observation)),
            "ble_ready": ble_ready,
            "grasp_margin_px": observation.get("grasp_margin_px"),
            "observed_at_ms": _now_ms(),
        }

    def start_pursuit(self, payload: dict[str, Any], headers: Any) -> dict[str, Any]:
        if self._grasp_lock.locked():
            raise RuntimeError("正在进行凸包微调或模型夹取，不能同时启动视觉追踪")
        instruction = _safe_text(payload.get("target") or payload.get("instruction"), name="目标描述")
        dry_run = bool(payload.get("dry_run", not HARDWARE_ENABLED))
        if not dry_run:
            self._authorise_hardware(headers)
        order = payload.get("camera_order")
        swap = None
        if order is not None:
            if not isinstance(order, dict) or not isinstance(order.get("swapped"), bool):
                raise ValueError("camera_order.swapped 必须是布尔值")
            swap = order["swapped"]
        grasp = payload.get("grasp", False)
        if not isinstance(grasp, bool):
            raise ValueError("grasp 必须是布尔值")
        current = self.process.snapshot()
        if current.get("running") and current.get("kind") != "camera_preview":
            raise RuntimeError("已有底盘任务执行中，请先停止，不能覆盖它")
        pending = getattr(self, "_phase1_pending", None)
        handoff = None
        baseline_digest = None
        if pending is not None:
            original_search_finished = (
                current.get("returncode") == 0
                and current.get("log") == pending["log"])
            preview_replaced_finished_search = (
                current.get("running") is True
                and current.get("kind") == "camera_preview")
            if not (original_search_finished or preview_replaced_finished_search):
                raise RuntimeError("第一阶段寻物尚未完成本轮交接，拒绝启动视觉追踪")
            handoff = _phase1_handoff_from_log(
                Path(pending["log"]),
                expected_instruction=pending["instruction"],
                expected_task_id=pending["task_id"],
                car1_root=self.map_navigation.car1_root,
            )
            # Follow the target confirmed in this exact search run, rather
            # than a second free-text target supplied by a later web step.
            verified_query = str((handoff.get("target") or {}).get("query") or "").strip()
            if not verified_query:
                raise RuntimeError("本轮寻物交接缺少目标描述，拒绝启动后续追踪")
            instruction = verified_query
            try:
                request = urlrequest.Request(f"{PURSUIT_HTTP}/frame.jpg", method="GET")
                with urlrequest.urlopen(request, timeout=1.0) as response:
                    baseline_digest = hashlib.sha256(response.read(5_000_000)).hexdigest()
            except Exception:
                pass  # Evidence capture is passive; an unavailable old frame is not a motion gate.
        command = _pursuit_command(instruction, dry_run=dry_run,
                                   swap_cameras=swap, grasp=grasp)
        self.process.stop()
        value = self.process.start("visual_pursuit", command, TRACK_ROOT)
        result = {"ok": True, "dry_run": dry_run, "grasp_requested": grasp,
                  "camera_order": order, "process": value}
        self._phase1_verified_target = (instruction if handoff is not None and not grasp else None)
        self._phase1_pending = None
        if handoff is not None:
            evidence_dir = LOG_ROOT / "handoff_evidence" / Path(value["log"]).stem
            try:
                threading.Thread(
                    target=_capture_handoff_frames,
                    args=(handoff, value["log"], evidence_dir, baseline_digest),
                    name="phase1_pursuit_evidence", daemon=True,
                ).start()
                result["handoff_evidence"] = str(evidence_dir / "handoff_evidence.json")
            except RuntimeError as exc:
                result["handoff_evidence_error"] = str(exc)[:200]
        return result

    def mapping(self, start: bool) -> dict[str, Any]:
        endpoint = "/services/mapping/start" if start else "/services/stop"
        return _http_json(f"{COMPANION_URL}{endpoint}", {}, timeout=8.0)

    def drive_relative(self, payload: dict[str, Any], headers: Any) -> dict[str, Any]:
        if self._grasp_lock.locked():
            raise RuntimeError("正在进行凸包微调或模型夹取，不能同时移动底盘")
        direction = str(payload.get("direction") or "")
        if direction not in {"forward", "backward", "left", "right"}:
            raise ValueError("运动方向无效")
        distance_m = float(payload.get("distance_m") or (0.10 if direction == "backward" else 0.30))
        angle_deg = float(payload.get("angle_deg") or 30.0)
        if direction in {"forward", "backward"} and not 0.05 <= distance_m <= 1.50:
            raise ValueError("单次直行距离必须在 0.05..1.50m")
        if direction == "backward" and distance_m > 0.10:
            raise ValueError("后向雷达被车身遮挡；单次后退最多 0.10m，请人工确认后方净空")
        if direction in {"left", "right"} and not 5.0 <= angle_deg <= 180.0:
            raise ValueError("单次转向角度必须在 5..180°")
        real = HARDWARE_ENABLED
        if real:
            self._authorise_hardware(headers)
        current = self.process.snapshot()
        if current["running"] and current["kind"] == "camera_preview":
            self.process.stop()
        command = [
            str(PYTHON), "-u", "-m", "final_work.pi.motion_session", "--",
            "--direction", direction,
            "--distance-m", str(distance_m),
            "--angle-deg", str(angle_deg),
        ]
        if not real:
            command.append("--simulate")
        value = self.process.start("bounded_motion", command, TRACK_ROOT)
        return {"ok": True, "mode": "real" if real else "simulate", "process": value}

    def grasp(self, action: str, payload: dict[str, Any], headers: Any) -> dict[str, Any]:
        if action == "model":
            if not self._grasp_lock.acquire(blocking=False):
                raise RuntimeError("已有凸包微调或模型夹取任务在执行")
            try:
                self._grasp_cancel.clear()
                return self._grasp_impl(action, payload, headers)
            finally:
                self._grasp_lock.release()
        if action != "stop" and self._grasp_lock.locked():
            raise RuntimeError("模型夹取进行中，不能插入其他机械臂动作")
        return self._grasp_impl(action, payload, headers)

    def _grasp_impl(self, action: str, payload: dict[str, Any], headers: Any) -> dict[str, Any]:
        if action in {"start", "model", "release", "home"}:
            self._authorise_hardware(headers)
        if action in {"start", "model", "home"}:
            target = ""
            if action != "home":
                target = _safe_text(payload.get("target") or payload.get("instruction"), name="抓取目标")
            if action == "model" and getattr(self, "_phase1_verified_target", None):
                target = self._phase1_verified_target
            if action == "model":
                self.skills.assert_empty()
            if not self._probe(f"{GRAB_WEB_URL}/api/status")["connected"]:
                self.start_preview()
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    if self._probe(f"{GRAB_WEB_URL}/api/status")["connected"]:
                        break
                    time.sleep(0.25)
                else:
                    raise RuntimeError("双目抓取客户端未能启动")
            if action == "home":
                if payload.get('confirmed_empty') is True:
                    self.skills.confirm_empty(payload, headers)
                self.skills.assert_empty()
                result = _http_json(f"{GRAB_WEB_URL}/api/arm-teach/home", {}, timeout=8.0)
                if result.get("ok"):
                    self.skills.cleared("explicit_home")
                return result
            if action == "model":
                managed = self.process.snapshot()
                if managed.get("running") and managed.get("kind") not in {"idle", "camera_preview"}:
                    raise RuntimeError("底盘仍在执行任务，必须先停车才能夹取")
                # Preview may be up before BLE and both cameras are ready.
                # Never enqueue an arm sequence based on stale/one-eye frames.
                ready_deadline = time.monotonic() + 15.0
                while True:
                    probe = self._probe(f"{GRAB_WEB_URL}/api/status")
                    data = probe.get("data") if probe.get("connected") else None
                    data = data if isinstance(data, dict) else {}
                    cameras = data.get("cameras") or {}
                    fresh = all(
                        isinstance(cameras.get(side), dict)
                        and cameras[side].get("frames", 0) > 0
                        and 0 <= float(cameras[side].get("age_s", 99)) < 1.5
                        for side in ("left", "right")
                    )
                    if data.get("ble") is True and fresh:
                        break
                    if time.monotonic() >= ready_deadline:
                        raise RuntimeError("夹取前双目或 ESP32 BLE 未就绪；已保持机械臂停止")
                    time.sleep(0.25)
                prior_execution = _http_json(
                    f"{GRAB_WEB_URL}/api/arm-teach/exec", timeout=3.0,
                ).get("status") or {}
                if prior_execution.get("active"):
                    raise RuntimeError("已有机械臂任务仍在执行，不能重复夹取")
                alignment = _align_teaching_hull(
                    target, real=payload.get("auto_align") is True,
                    cancel_event=self._grasp_cancel)
                if not alignment.get("ok"):
                    raise RuntimeError(
                        "夹取前未能进入双目四维示教凸包："
                        f"{alignment.get('reason') or '原因未知'}；"
                        f"日志：{alignment.get('log_path') or '无'}"
                    )
                # The micro-align process owns its own short-lived BLE link.
                # Wait until grab_visual's arm link reconnects before enqueue.
                arm_link_deadline = time.monotonic() + 15.0
                while True:
                    probe = self._probe(f"{GRAB_WEB_URL}/api/status")
                    data = probe.get("data") if probe.get("connected") else None
                    data = data if isinstance(data, dict) else {}
                    if data.get("ble") is True:
                        break
                    if time.monotonic() >= arm_link_deadline:
                        raise RuntimeError("微调后机械臂 BLE 未恢复，禁止夹取")
                    time.sleep(0.25)
                if self._grasp_cancel.is_set():
                    raise RuntimeError("收到急停，禁止启动模型夹取")
                from .grasp_execution import execute_model_grasp
                from .pursuit_session import _prepare_loss_recovery
                from .motion_session import _event
                try:
                    result = execute_model_grasp(target, request=_http_json,
                        cancel_event=self._grasp_cancel, base_url=GRAB_WEB_URL,
                        prior_execution=prior_execution, emit=_event,
                        realign=(lambda deadline: _align_teaching_hull(target, real=True,
                            cancel_event=self._grasp_cancel, deadline=deadline))
                            if payload.get('auto_align') is True else None,
                        prepare_recovery=lambda: _prepare_loss_recovery(self._grasp_cancel, GRAB_WEB_URL))
                finally:
                    self._phase1_verified_target = None
                if result.get('ok'):
                    result['alignment'] = result.get('hull_recheck_alignment') or alignment
                    result['carry'] = self.skills.record_pick(target, result)
                return result
        if action == "start":
            return _http_json(f"{GRAB_WEB_URL}/api/grasp/start", {"target": target}, timeout=8.0)
        if action == "release":
            result = _http_json(f"{GRAB_WEB_URL}/api/grasp/release", {}, timeout=8.0)
            if result.get("ok"):
                self.skills.cleared("explicit_release")
            return result
        if action == "stop":
            return _http_json(f"{GRAB_WEB_URL}/api/grasp/stop", {}, timeout=4.0)
        raise ValueError("不支持的抓取动作")

    def emergency_stop(self) -> dict[str, Any]:
        self._grasp_cancel.set()
        self._phase1_verified_target = None
        result: dict[str, Any] = {"process": self.process.stop(), "errors": []}
        for name, url, payload in (
            ("companion", f"{COMPANION_URL}/control", {"type": "emergency_stop", "enabled": True}),
            ("arm_teach", f"{GRAB_WEB_URL}/api/arm-teach/stop", {}),
            ("grasp", f"{GRAB_WEB_URL}/api/grasp/stop", {}),
            ("arm", f"{GRAB_WEB_URL}/api/arm/stop", {}),
        ):
            try:
                result[name] = _http_json(url, payload, timeout=2.5)
            except Exception as exc:
                result["errors"].append(f"{name}: {exc}")
        self.skills.invalidate("emergency_stop")
        result["ok"] = not result["errors"]
        return result

    def semantic_map(self) -> dict[str, Any]:
        return self.map_navigation.semantic_map()

    def map_mission(self, payload: dict[str, Any], headers: Any, *, purpose: str) -> dict[str, Any]:
        if self._grasp_lock.locked():
            raise RuntimeError("正在进行凸包微调或模型夹取，不能同时启动地图任务")
        target_payload = payload.get("target")
        target_query = target_payload.get("query") if isinstance(target_payload, dict) else ""
        query = _safe_text(
            payload.get("instruction") or payload.get("query") or target_query,
            name="地图导航指令",
        )
        requested_real = payload.get("real", HARDWARE_ENABLED)
        if not isinstance(requested_real, bool):
            raise ValueError("real 必须是布尔值")
        real = requested_real
        phase1_pick = bool(
            real and purpose == "observation_search"
            and isinstance(target_payload, dict)
            and target_payload.get("integration_workflow") == "phase1_deepseek_pick_v1"
            and target_payload.get("requested_intent") == "pick"
        )
        current = self.process.snapshot()
        if current.get("running") and current.get("kind") != "camera_preview":
            raise RuntimeError("已有底盘任务执行中，不能重新规划并覆盖它")
        self._phase1_pending = None
        self._phase1_verified_target = None
        if real:
            self._authorise_hardware(headers)
        if phase1_pick:
            query = str(target_payload.get("pickup_instruction") or query)
            decision, decision_path = self.map_navigation.prepare_phase1_pick(query)
        else:
            decision = self.map_navigation.request_decision(query)
        status = str(decision.get("status") or "rejected")
        if status != "executable":
            return {
                "ok": False,
                "status": status,
                "purpose": purpose,
                "decision": decision,
                "message": decision.get("clarification_question") or decision.get("reason") or "任务不可执行",
            }
        if phase1_pick:
            command, extra_env = self.map_navigation.phase1_pick_command(decision_path)
        else:
            decision_path = self.map_navigation.save_decision(decision)
            command, extra_env = self.map_navigation.mission_command(decision_path, real=real)
        process = self.process.start(
            "map_navigation_search",
            command,
            self.map_navigation.car1_root,
            extra_env=extra_env,
        )
        self._phase1_pending = ({
            "instruction": decision["instruction"],
            "task_id": decision.get("task_id"),
            "log": process["log"],
        } if phase1_pick else None)
        return {
            "ok": True,
            "status": "running",
            "purpose": purpose,
            "mode": "real" if real else "simulate",
            "integration_workflow": "phase1_deepseek_pick_v1" if phase1_pick else "legacy_map_mission",
            "decision": decision,
            "decision_path": str(decision_path),
            "process": process,
        }

    def saved_map_png(self) -> tuple[bytes, str] | None:
        default_layer = next(
            (item for item in self.map_navigation.layers()["layers"] if item.get("is_default")),
            None,
        )
        candidates = [
            Path(default_layer["source_path"]) if default_layer else STATIC_MAP_PATH,
            # User-designated unannotated map. Keep it ahead of stale live-map
            # snapshots so the web UI always displays the intended scene.
            STATIC_MAP_PATH,
            TRACK_ROOT / "search_runs" / "live_map.png",
            TRACK_ROOT / "maps" / "ble_scene_17.png",
            TRACK_ROOT / "maps" / "ble_scene_17.pgm",
            TRACK_ROOT / "maps" / "ble_auto_map.png",
            TRACK_ROOT / "maps" / "ble_auto_map.pgm",
            LEGACY_MAP_ROOT / "maps" / "ble_scene_17.png",
            LEGACY_MAP_ROOT / "search_runs" / "live_map.png",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".png":
                    return path.read_bytes(), str(path)
                from PIL import Image
                output = io.BytesIO()
                with Image.open(path) as image:
                    image.save(output, format="PNG")
                return output.getvalue(), str(path)
            except OSError:
                continue
        return None

    @staticmethod
    def _image_bytes(path: Path) -> tuple[bytes, str]:
        if path.suffix.lower() == ".png":
            return path.read_bytes(), "image/png"
        from PIL import Image
        output = io.BytesIO()
        with Image.open(path) as image:
            image.save(output, format="PNG")
        return output.getvalue(), "image/png"


def _handler(bridge: RobotBridge):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FinalWorkPiBridge/1.0"

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
            value = json.loads(self.rfile.read(length).decode("utf-8") if length else "{}")
            if not isinstance(value, dict):
                raise ValueError("JSON 根节点必须是对象")
            return value

        def _proxy(self, url: str) -> None:
            started = False
            try:
                with urlrequest.urlopen(url, timeout=10) as response:
                    self.send_response(response.status)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "application/octet-stream"))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    started = True
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            return
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                if not started and not self.wfile.closed:
                    self._json({"ok": False, "error": str(exc)}, 503)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/health":
                self._json({"ok": True, "service": "final_work_pi_bridge", "hardware_enabled": HARDWARE_ENABLED})
            elif path == "/state":
                self._json(bridge.state())
            elif path == "/skills":
                self._json({"ok": True, "schema": "robot_skills.v1", "skills": [
                    "go_to_region", "find_and_approach", "model_grasp", "carry_to_region", "place_on_floor"],
                    "placement_surfaces": ["floor_beside_region"],
                    "physical_outcome_verified_by_servo_completion": False})
            elif path == "/map/status":
                try:
                    self._json(_http_json(f"{MAP_URL}/api/status", timeout=2.0))
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, 503)
            elif path == "/map.png":
                try:
                    with urlrequest.urlopen(f"{MAP_URL}/api/map.png", timeout=2.0) as response:
                        body = response.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception:
                    saved = bridge.saved_map_png()
                    if saved is None:
                        self._json({"ok": False, "error": "实时或保存地图均不可用"}, 503)
                    else:
                        body, source = saved
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("X-Map-Source", source)
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
            elif path == "/semantic-map":
                self._json(bridge.semantic_map())
            elif path == "/map/layers":
                self._json(bridge.map_navigation.layers())
            elif path == "/map/layer.png":
                layer_id = (parse_qs(parsed.query).get("id") or [""])[0]
                image_path = bridge.map_navigation.layer_path(layer_id)
                if image_path is None:
                    self._json({"ok": False, "error": "地图图层不存在"}, 404)
                else:
                    try:
                        body, content_type = bridge._image_bytes(image_path)
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.send_header("X-Map-Source", str(image_path))
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    except OSError as exc:
                        self._json({"ok": False, "error": str(exc)}, 503)
            elif path in {"/video/left", "/video/right"}:
                camera = path.rsplit("/", 1)[-1]
                self._proxy(f"{GRAB_WEB_URL}/video/{camera}")
            elif path == "/logs/tail":
                self._json({"ok": True, "process": bridge.process.snapshot(), "text": bridge.process.tail()})
            elif path == "/logs/list":
                self._json({"ok": True, "logs": bridge.process.logs()})
            elif path == "/logs/file":
                name = (parse_qs(parsed.query).get("name") or [""])[0]
                try:
                    self._json({"ok": True, "name": name, "text": bridge.process.read_log(name)})
                except ValueError as exc:
                    self._json({"ok": False, "error": str(exc)}, 404)
            else:
                self._json({"ok": False, "error": "接口不存在"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                payload = self._body()
                if path == '/recording/start':
                    self._json(bridge.start_recording(payload,self.headers))
                elif path == '/recording/finish':
                    bridge._authorise_hardware(self.headers)
                    self._json({"ok": True, **bridge.video.finish(
                        str(payload.get('session_id') or ''),str(payload.get('reason') or 'task_finished'))})
                elif path == "/actions/preview/start":
                    self._json(bridge.start_preview())
                elif path == "/actions/preview/stop":
                    self._json({"ok": True, "process": bridge.process.stop()})
                elif path == "/perception/current-view":
                    self._json(bridge.probe_current_view(payload))
                elif path == "/actions/mapping/start":
                    bridge._authorise_hardware(self.headers)
                    self._json(bridge.mapping(True))
                elif path == "/actions/mapping/stop":
                    self._json(bridge.mapping(False))
                elif path == "/actions/pursuit/start":
                    self._json(bridge.start_pursuit(payload, self.headers))
                elif path == "/actions/drive-relative":
                    self._json(bridge.drive_relative(payload, self.headers))
                elif path == "/actions/pursuit/stop":
                    self._json({"ok": True, "process": bridge.process.stop()})
                elif path.startswith("/actions/grasp/"):
                    self._json(bridge.grasp(path.rsplit("/", 1)[-1], payload, self.headers))
                elif path == "/interfaces/skill-plan":
                    from .skill_planning import plan_contract
                    self._json(plan_contract(payload, bridge.map_navigation))
                elif path == "/skills/resolve-destination":
                    bridge._authorise_hardware(self.headers)
                    self._json(bridge.skills.resolve(payload))
                elif path == "/skills/navigate-region":
                    self._json(bridge.skills.navigate(payload, self.headers))
                elif path == "/skills/check-carry":
                    self._json(bridge.skills.check_carry(_http_json, GRAB_WEB_URL))
                elif path == "/skills/confirm-empty":
                    self._json(bridge.skills.confirm_empty(payload, self.headers))
                elif path == "/skills/place-on-floor":
                    self._json(bridge.skills.place(payload, self.headers,
                                                  request=_http_json, base_url=GRAB_WEB_URL))
                elif path == "/actions/emergency-stop":
                    result=bridge.emergency_stop()
                    current=bridge.video.status()
                    if current.get('recording'):bridge.video.finish(current['session_id'],'emergency_stop')
                    self._json(result)
                elif path == "/interfaces/observation-plan":
                    self._json(bridge.map_mission(payload, self.headers, purpose="observation_search"))
                elif path == "/interfaces/navigation-plan":
                    self._json(bridge.map_mission(payload, self.headers, purpose="semantic_navigation"))
                else:
                    self._json({"ok": False, "error": "接口不存在"}, 404)
            except PermissionError as exc:
                self._json({"ok": False, "error": str(exc), "code": "hardware_locked"}, 403)
            except (ValueError, RuntimeError, json.JSONDecodeError, urlerror.URLError) as exc:
                self._json({"ok": False, "error": str(exc)}, 400)

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[final-work-pi] " + fmt % args, flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="final_work Raspberry Pi robot bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    bridge = RobotBridge()
    server = ThreadingHTTPServer((args.host, args.port), _handler(bridge))
    print(f"final_work Pi bridge: http://{args.host}:{args.port}", flush=True)
    print(f"hardware enabled: {HARDWARE_ENABLED}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        bridge.process.stop()
        bridge.video.close()
        server.server_close()


if __name__ == "__main__":
    main()
