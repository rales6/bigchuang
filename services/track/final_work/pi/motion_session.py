"""Run one bounded motion while temporarily handing BLE ownership to it.

The stereo/grasp web process normally owns the only ESP32 BLE connection.
Direct chassis motion therefore pauses that process, runs the lidar-guarded
motion, and restores the camera service afterwards.  All lifecycle events are
written to the same per-action log as the motion process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib import request as urlrequest


GRAB_ROOT = Path(os.environ.get("FINAL_WORK_GRAB_ROOT", "/home/dachuang/grab_visual"))
GRAB_STATUS_URL = os.environ.get("FINAL_WORK_GRAB_STATUS_URL", "http://127.0.0.1:8080/api/status")
LOG_ROOT = Path(os.environ.get(
    "FINAL_WORK_PI_LOG_DIR", "/home/dachuang/dachuang/track/final_work/logs"
))


def _event(name: str, **fields) -> None:
    print(json.dumps({"event": name, "at": time.time(), **fields}, ensure_ascii=False), flush=True)


def _web_running() -> bool:
    try:
        with urlrequest.urlopen(GRAB_STATUS_URL, timeout=1.2) as response:
            return response.status == 200
    except Exception:
        return False


def _web_pids() -> list[int]:
    target = str(GRAB_ROOT / "web_app.py")
    result: list[int] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, PermissionError):
            continue
        if target in command:
            result.append(int(entry.name))
    return result


def _disconnect_ble_owner() -> None:
    """Release a stale BlueZ GATT owner and wait until it is observable."""
    address = os.environ.get("FINAL_WORK_BLE_ADDRESS", "64:B7:08:69:9A:E6")
    try:
        disconnected = subprocess.run(
            ["bluetoothctl", "disconnect", address],
            capture_output=True,
            text=True,
            timeout=6.0,
            check=False,
        )
        _event(
            "ble_owner_released", returncode=disconnected.returncode,
            output=((disconnected.stdout or "") + (disconnected.stderr or ""))[-800:],
        )
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            status = subprocess.run(
                ["bluetoothctl", "info", address],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            if "Connected: yes" not in (status.stdout or ""):
                break
            time.sleep(0.20)
    except Exception as exc:
        _event("ble_owner_release_warning", error=str(exc))
    # BlueZ needs a brief interval to publish the disconnect before a new
    # process can resolve the same peripheral reliably.
    time.sleep(1.5)


def _stop_preview() -> bool:
    was_running = _web_running() or bool(_web_pids())
    if not was_running:
        _event("preview_not_running")
    else:
        try:
            req = urlrequest.Request(
                "http://127.0.0.1:8080/api/grasp/stop",
                data=b"{}",
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urlrequest.urlopen(req, timeout=2.0):
                pass
        except Exception as exc:
            _event("preview_grasp_stop_warning", error=str(exc))
        pids = _web_pids()
        _event("preview_release_begin", pids=pids)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 6.0
        while _web_pids() and time.monotonic() < deadline:
            time.sleep(0.10)
        for pid in _web_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # The bridge may already have terminated its managed preview process
    # before this wrapper starts.  In that case ``was_running`` is false, but
    # BlueZ can still retain the old GATT connection for several seconds.
    # A connected BLE peripheral does not advertise, so the new Bleak client
    # sees repeated discovery timeouts.  Always release the fixed ESP32
    # address, even when no preview PID remains.
    _disconnect_ble_owner()
    _event("preview_released", remaining_pids=_web_pids())
    return was_running


def _restore_preview() -> None:
    if _web_running():
        _event("preview_already_restored")
        return
    # Esp32Client.close() may return before BlueZ publishes the disconnect.
    # Without this explicit release the preview can open both cameras yet
    # report ble:false until it is manually restarted.
    _disconnect_ble_owner()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    preview_log = (LOG_ROOT / "camera_preview.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            ["bash", str(GRAB_ROOT / "start_web.sh")],
            cwd=str(GRAB_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=preview_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=__import__(__package__+'.camera_proxy',fromlist=['camera_environment']).camera_environment(
                {**os.environ, "PYTHONUNBUFFERED": "1", "LEFT_CAM": "2", "RIGHT_CAM": "0"}),
        )
    finally:
        preview_log.close()
    deadline = time.monotonic() + 18.0
    while time.monotonic() < deadline:
        if _web_running():
            _event("preview_restored", pid=process.pid)
            return
        if process.poll() is not None:
            _event("preview_restore_failed", returncode=process.returncode)
            return
        time.sleep(0.25)
    _event("preview_restore_timeout", pid=process.pid)


def main() -> int:
    parser = argparse.ArgumentParser(description="独占 BLE 的受限底盘移动会话")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("motion_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    motion_args = list(args.motion_args)
    if motion_args[:1] == ["--"]:
        motion_args.pop(0)
    was_running = False
    child: subprocess.Popen | None = None

    def terminate(_signum, _frame):
        if child is not None and child.poll() is None:
            child.terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    _event("motion_session_start", probe_only=args.probe_only, argv=motion_args)
    try:
        was_running = _stop_preview()
        command = [sys.executable, "-u", "-m", "final_work.pi.bounded_motion"]
        if args.probe_only:
            command.append("--probe-only")
        command.extend(motion_args)
        from .camera_proxy import camera_environment
        child = subprocess.Popen(command,env=camera_environment())
        returncode = child.wait()
        _event("motion_session_finished", returncode=returncode)
        return returncode
    except KeyboardInterrupt:
        _event("motion_session_interrupted")
        return 130
    finally:
        if was_running:
            _restore_preview()


if __name__ == "__main__":
    raise SystemExit(main())

