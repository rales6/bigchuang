"""One-shot BlueZ recovery for phase-one startup discovery failures.

This adapter runs only before the chassis client is created. It does not
issue motion commands and never changes the protected BLE transport/client.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time


ESP32_ADDRESS = "64:B7:08:69:9A:E6"
DISCOVERY_ERRORS = (
    "已知地址扫描超时",
    "设备名称扫描超时",
    "未发现 ESP32-Robot-Car",
    "org.bluez.Error.NotReady",
)
MOTION_PROCESS_MARKERS = (
    "run_visual_pursuit.py",
    "hull_micro_adjust.py",
    "bounded_motion.py",
    "probe_bounded_turn.py",
    "probe_live_rotation.py",
    "object_search_mission.py",
    "relationship_find_only_entry.py",
    "grasp_execution.py",
)


def is_discovery_failure(error: BaseException) -> bool:
    return any(marker in str(error) for marker in DISCOVERY_ERRORS)


def _check_no_other_motion_process() -> None:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=6, check=True,
    )
    own_pid = os.getpid()
    conflicts = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == own_pid:
            continue
        if any(marker in parts[1] for marker in MOTION_PROCESS_MARKERS):
            conflicts.append(parts[0])
    if conflicts:
        raise RuntimeError(
            "检测到其他实车运动/夹取进程，拒绝重启蓝牙；PID=" + ",".join(conflicts)
        )


def _run(command: list[str], *, timeout: float, required: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("蓝牙恢复命令未完成：{}；{}".format(" ".join(command), exc)) from exc
    output = (result.stdout + " " + result.stderr).strip()
    if required and (result.returncode != 0 or "Failed to set power" in output):
        raise RuntimeError("蓝牙恢复命令失败：{}；{}".format(" ".join(command), output[-300:]))
    return result


def reset_bluetooth_for_missing_esp32() -> None:
    """Execute the operator's recovery sequence once, while the car is stopped."""
    _check_no_other_motion_process()
    print("BLE 已知地址扫描失败：底盘尚未连接；开始一次 BlueZ 断开/重启/上电恢复。", flush=True)
    # The disconnect is intentionally best effort, matching `2>/dev/null || true`.
    _run(["bluetoothctl", "disconnect", ESP32_ADDRESS], timeout=10, required=False)
    _run(["sudo", "-n", "systemctl", "restart", "bluetooth"], timeout=25)
    rfkill = (shutil.which("rfkill") or
              next((str(path) for path in (Path("/usr/sbin/rfkill"), Path("/sbin/rfkill"))
                    if path.is_file()), None))
    if rfkill:
        _run(["sudo", "-n", rfkill, "unblock", "bluetooth"], timeout=10)
    else:
        # Some images omit rfkill entirely. BlueZ restart and `power on` still
        # run; report the missing optional command plainly.
        print("BLE 恢复提示：系统未安装 rfkill，跳过 unblock；继续等待并执行 bluetoothctl power on。",
              flush=True)
    time.sleep(3)
    _run(["bluetoothctl", "power", "on"], timeout=10)
    print("BLE 恢复命令已完成；重新连接 ESP32 并确认心跳/STOP。", flush=True)


def install_ble_startup_recovery(mission) -> None:
    """Retry only discovery failures; preserve every other startup failure."""
    original = mission._create_client_with_retry

    def connect_with_recovery(factory, attempts=8, delay_s=1.0):
        total = max(1, int(attempts))
        initial = min(2, total)
        try:
            return original(factory, attempts=initial, delay_s=delay_s)
        except RuntimeError as exc:
            if is_discovery_failure(exc):
                reset_bluetooth_for_missing_esp32()
                print("BLE 恢复后重新尝试底盘连接。", flush=True)
            else:
                print("底盘连接失败不是 BLE 扫描超时；不重启蓝牙。", flush=True)
            remaining = max(1, total - initial)
            return original(factory, attempts=remaining, delay_s=delay_s)

    mission._create_client_with_retry = connect_with_recovery
