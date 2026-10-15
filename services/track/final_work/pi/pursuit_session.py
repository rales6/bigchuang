"""Run visual pursuit with exclusive camera/BLE ownership.

The grasp preview normally keeps both USB cameras open. Pursuit needs those
same devices, so this wrapper releases the preview first and restores it after
the pursuit process exits, succeeds, or is interrupted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import json
import csv
import math
from datetime import datetime

from .motion_session import _event, _restore_preview, _stop_preview
from .pursuit_recovery import (LossRecoveryBudget, identified_loss_run,
                               alignment_target_lost, align_with_loss_recovery)
from .camera_proxy import camera_environment


TRACK_ROOT = Path(os.environ.get("FINAL_WORK_TRACK_ROOT", "/home/dachuang/dachuang/track"))


def _target_from_args(args: list[str]) -> str:
    for index, value in enumerate(args):
        if value == "--instruction" and index + 1 < len(args):
            return args[index + 1].strip()
        if value.startswith("--instruction="):
            return value.split("=", 1)[1].strip()
    return ""


def _wait_preview_ready(timeout_s: float = 25.0, cancelled=None) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cancelled is not None and cancelled.is_set():
            return False
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/api/status",
                                        timeout=3.0) as response:
                status = json.load(response)
            cameras = status.get("cameras") or {}
            if status.get("ble") and all(
                    (cameras.get(side) or {}).get("frames", 0) > 0
                    and 0 <= float((cameras.get(side) or {}).get("age_s", 99)) < 1.0
                    for side in ("left", "right")):
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _prepare_loss_recovery(cancelled, base_url="http://127.0.0.1:8080"):
    """Restore ownership, confirm stopping; do not reset the arm or map task."""
    if cancelled.is_set():
        raise RuntimeError("任务已取消，未启动新追踪")
    _restore_preview()
    if not _wait_preview_ready(cancelled=cancelled):
        raise RuntimeError("重新识别前双目/BLE 尚未恢复，未启动新追踪")
    from pursuit.hull_micro_adjust import _get_json, _post_json
    state = _get_json(base_url + "/api/arm-teach/exec", timeout=3.0).get("status")
    if not isinstance(state, dict) or state.get("active") is not False:
        raise RuntimeError("机械臂执行状态未确认空闲，禁止重启靠近")
    stopped = _post_json(base_url + "/api/arm-teach/stop", {}, timeout=3.0)
    if stopped.get("ok") is not True or stopped.get("result") != "OK":
        raise RuntimeError("重新识别前停止请求未确认，不重复启动")
    if cancelled.is_set():
        raise RuntimeError("任务已取消，未启动新追踪")


def _recent_pursuit_retreat(target: str, child_pid: int):
    """This run's recent approach permits bounded approximate-path retreat.

    A saved old run, a bare request to reverse, or a spinning/one-eye approach
    cannot grant the hull stage permission to omit rear sensing.
    """
    from pursuit.hull_micro_adjust import SweptRetreatBudget
    from pursuit.pursuit_config import PursuitConfig
    config=PursuitConfig()

    try:
        log_root = (TRACK_ROOT / "logs" / "pursuit").resolve()
        run_dir = Path((log_root / "LATEST.txt").read_text().strip()).resolve()
        if not run_dir.is_relative_to(log_root):
            return None
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        age = time.time() - datetime.fromisoformat(metadata["ended_at"]).timestamp()
        if (metadata.get("pid") != child_pid or metadata.get("mode") != "esp32"
                or metadata.get("active_instruction") != target
                or metadata.get("final_result") != "alignment_pending"
                or metadata.get("error") or not 0 <= age <= 10.0):
            return None
        with (run_dir / "telemetry.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 2:
            return None
        end_t = float(rows[-1]["t_s"])
        recent = [row for row in rows if 0 <= end_t - float(row["t_s"]) <= 3.0]
        if len(recent) < 2:
            return None
        true_values = {"true", "1"}
        heading_proxy=0.0
        previous_t=float(recent[0]['t_s'])
        previous_w=0.0
        for row in recent:
            v, w = float(row["v_mm_s"]), float(row["w_mrad_s"])
            if (row["tracking"].lower() not in true_values
                    or row["tracking2"].lower() not in true_values
                    or row["state"] not in {"TRACKING", "STOPPED"}
                    or not math.isfinite(v + w) or abs(w) > config.retreat_recent_max_angular_mrad_s
                    or v < 0 or v==0 and w!=0):
                return None
            row_t=float(row['t_s'])
            dt=row_t-previous_t
            if not math.isfinite(row_t) or dt<0:
                return None
            heading_proxy+=abs(previous_w)*dt/1000.0
            previous_t=row_t
            previous_w=w
        if heading_proxy>config.retreat_recent_max_heading_rad:
            return None
        if not any(float(row["v_mm_s"]) > 0 for row in recent):
            return None
        first, last = recent[0], recent[-1]
        # Both center rows must show approach, not just an issued motor command.
        for key in ("bbox_cy_px", "bbox_cy2_px"):
            start, end = float(first[key]), float(last[key])
            if not math.isfinite(start + end) or end - start < 3.0:
                return None
        return SweptRetreatBudget(time.monotonic() + 150.0,
                                 str(run_dir)+'; approximate_recent_approach',
                                 heading_change_rad=heading_proxy)
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _finish_at_exact_hull(target: str, cancelled: threading.Event,
                          retreat_budget=None, *, return_result=False, deadline=None):
    """Mask stop hands off to BOTH independent 2D teaching regions/shells."""
    from pursuit.hull_micro_adjust import align

    _restore_preview()
    if cancelled.is_set() or not _wait_preview_ready(cancelled=cancelled):
        _event("pursuit_hull_handoff_failed", reason="preview_or_ble_not_ready")
        return {"ok": False, "reason": "preview_or_ble_not_ready"} if return_result else 2
    deadline = min(deadline, time.monotonic() + 180.0) if deadline is not None else time.monotonic() + 180.0
    result = align_with_loss_recovery(target, real=True, cancelled=cancelled,
        align=lambda: align(target, real=True, cancel_event=cancelled,
            retreat_budget=retreat_budget,
            deadline_s=min(120.0, max(1.0, deadline-time.monotonic()))).__dict__,
        prepare=lambda: _prepare_loss_recovery(cancelled), emit=_event,
        retry_target_loss=False, deadline=deadline)
    _event("pursuit_hull_handoff", **result)
    return result if return_result else (0 if result.get('ok') else 2)


def _grasp_after_alignment(target: str, cancelled: threading.Event) -> int:
    """Explicit combined mission only; endpoint freshly rechecks the model."""
    from pursuit.hull_micro_adjust import GRAB_URL, _get_json, _post_json
    if cancelled.is_set() or not _wait_preview_ready(cancelled=cancelled):
        raise RuntimeError("抓取前双目/BLE 未就绪或任务已取消")
    from .grasp_execution import execute_model_grasp
    print("[phase] 启动模型夹取：再次识别并检查抓取区域", flush=True)
    def request(url, payload=None, timeout=3.0):
        return (_get_json(url, timeout=timeout) if payload is None else
                _post_json(url, payload, timeout=timeout))
    def emit(event, **fields):
        state = fields.get('status') or {}
        print(f"[phase] 机械臂 {state.get('phase', event)}: {state.get('message', fields.get('error', ''))}", flush=True)
        _event(event, **fields)
    result = execute_model_grasp(target, request=request, cancel_event=cancelled,
                                 base_url=GRAB_URL, emit=emit,
                                 realign=lambda deadline: _finish_at_exact_hull(
                                     target, cancelled, return_result=True, deadline=deadline),
                                 prepare_recovery=lambda: _prepare_loss_recovery(cancelled, GRAB_URL))
    if not result.get('ok'):
        raise RuntimeError(result.get('error') or '机械臂拒绝启动')
    _event('grasp_sequence_complete', target=target, **result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="独占双目摄像头的视觉追踪会话")
    parser.add_argument("--grasp", action="store_true",
                        help="仅实车：对齐通过后执行机械臂夹取；默认只对齐")
    parser.add_argument("pursuit_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pursuit_args = list(args.pursuit_args)
    if pursuit_args[:1] == ["--"]:
        pursuit_args.pop(0)

    child: subprocess.Popen | None = None
    was_running = False
    cancelled = threading.Event()

    def terminate(_signum, _frame):
        cancelled.set()
        if child is not None and child.poll() is None:
            child.terminate()
        else:
            try:
                req = urllib.request.Request(
                    "http://127.0.0.1:8080/api/arm-teach/stop",
                    data=b"{}", method="POST",
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=2.0):
                    pass
            except Exception:
                pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    _event("pursuit_session_start", argv=pursuit_args, grasp_requested=args.grasp)
    try:
        was_running = _stop_preview()
        command = [
            sys.executable,
            "-u",
            str(TRACK_ROOT / "pursuit" / "run_visual_pursuit.py"),
            *pursuit_args,
        ]
        target = _target_from_args(pursuit_args)
        real = "--esp32" in pursuit_args and not any(
            value.split("=", 1)[0] == "--dry-run" for value in pursuit_args)
        recovery = LossRecoveryBudget()
        while not cancelled.is_set():
            launched_at = time.time()
            child = subprocess.Popen(command, cwd=str(TRACK_ROOT), env=camera_environment())
            returncode = child.wait()
            _event("pursuit_attempt_finished", returncode=returncode, pid=child.pid)
            if cancelled.is_set():
                return 130
            loss = (identified_loss_run(TRACK_ROOT, target=target, child_pid=child.pid,
                                       launched_at=launched_at, returncode=returncode)
                    if real else None)
            if returncode == 0 and real:
                if not target:
                    _event("pursuit_hull_handoff_failed", reason="missing_target")
                    return 2
                retreat_budget = _recent_pursuit_retreat(target, child.pid)
                _event("pursuit_retreat_handoff", allowed=retreat_budget is not None,
                       source=retreat_budget.source if retreat_budget else "",
                       approximate_path=True,
                       heading_command_proxy_rad=retreat_budget.heading_change_rad if retreat_budget else None)
                result = _finish_at_exact_hull(target, cancelled, retreat_budget, return_result=True)
                if cancelled.is_set():
                    return 130
                if result.get("ok"):
                    _event("pursuit_session_finished", returncode=0, restarts=recovery.restarts)
                    return _grasp_after_alignment(target, cancelled) if args.grasp else 0
                returncode = 2
                if alignment_target_lost(result):
                    loss = {"reason": result["reason"], "stage": "hull"}
            if cancelled.is_set():
                return 130
            if not loss or not recovery.reserve(cancelled):
                if loss:
                    _event("pursuit_loss_recovery_exhausted", target=target,
                           restarts=recovery.restarts, **loss)
                _event("pursuit_session_finished", returncode=returncode, restarts=recovery.restarts)
                return returncode
            _event("pursuit_loss_recovery", target=target, restart=recovery.restarts,
                   resume_from_current_position=True, **loss)
            print("[phase] 目标丢失：自动重新识别，从当前位置继续靠近（不重走导航）", flush=True)
            _prepare_loss_recovery(cancelled)
            if cancelled.is_set():
                return 130
            if loss.get('requires_fresh_lidar'):
                from .pursuit_scan_recovery import reacquire_scan
                port = None
                for i, value in enumerate(pursuit_args):
                    if value == '--lidar-port' and i+1 < len(pursuit_args):
                        port = pursuit_args[i+1]
                    elif value.startswith('--lidar-port='):
                        port = value.split('=', 1)[1]
                reacquire_scan(cancelled, port=port, emit=_event)
            _stop_preview()
            if cancelled.is_set():
                return 130
        return 130
    except KeyboardInterrupt:
        _event("pursuit_session_interrupted")
        return 130
    except Exception as exc:
        if cancelled.is_set():
            _event("pursuit_session_interrupted")
            return 130
        _event("pursuit_session_failed", reason=str(exc))
        print(f"[phase] 任务停止: {exc}", flush=True)
        return 2
    finally:
        # This wrapper is owned exclusively by the web bridge.  The bridge
        # may terminate its managed preview before this process starts, so
        # ``was_running`` can be false even though the UI expects preview and
        # grasp control to return after pursuit.  Restore it on every exit.
        _restore_preview()


if __name__ == "__main__":
    raise SystemExit(main())

