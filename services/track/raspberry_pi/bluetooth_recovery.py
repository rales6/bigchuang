"""Safe BlueZ recovery used by the phone companion service.

The bridge process is deliberately never selected by a broad ``pkill``.  The
caller first stops its managed motion process, then this module resets only the
Bluetooth controller through a small, validated privileged helper.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Callable


_MAC = re.compile(r"^(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


@dataclass
class BluetoothRecoveryResult:
    ok: bool
    message: str
    address: str
    elapsed_seconds: float
    output: str = ""

    def to_dict(self):
        return asdict(self)


def recover_bluetooth(
    address: str,
    *,
    helper: str = "/usr/local/sbin/robot-bluetooth-reset",
    timeout_seconds: float = 25.0,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> BluetoothRecoveryResult:
    """Reset BlueZ using a fixed helper; no shell text from the phone is run."""
    address = str(address).strip().upper()
    if not _MAC.fullmatch(address):
        raise ValueError("蓝牙地址格式不正确")
    started = time.monotonic()
    command = ["sudo", "-n", helper, address]
    try:
        completed = run(
            command,
            capture_output=True,
            text=True,
            timeout=max(5.0, float(timeout_seconds)),
            check=False,
        )
        output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )[-4000:]
        ok = completed.returncode == 0
        message = "蓝牙控制器已重置并重新上电" if ok else "系统级恢复不可用"
        if not ok:
            # Safe fallback requires no persistent root grant. It resolves
            # stale BlueZ connections while keeping the phone bridge alive.
            fallback_output = []
            fallback_ok = True
            for fallback_command in (
                ["bluetoothctl", "disconnect", address],
                ["bluetoothctl", "power", "off"],
                ["bluetoothctl", "power", "on"],
            ):
                if fallback_command[-1] == "on":
                    time.sleep(1.5)
                fallback = run(
                    fallback_command,
                    capture_output=True,
                    text=True,
                    timeout=8.0,
                    check=False,
                )
                fallback_output.append((fallback.stdout or "") + (fallback.stderr or ""))
                # Disconnect may return non-zero when already disconnected.
                if fallback_command[1] == "power" and fallback.returncode != 0:
                    fallback_ok = False
            ok = fallback_ok
            output = (output + "\n" + "\n".join(fallback_output))[-4000:]
            message = (
                "已执行无需 root 的 BLE 断开与控制器重新上电"
                if ok else "蓝牙恢复脚本执行失败"
            )
        return BluetoothRecoveryResult(
            ok, message, address, round(time.monotonic() - started, 2), output
        )
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            str(part or "").strip() for part in (exc.stdout, exc.stderr) if part
        )[-4000:]
        return BluetoothRecoveryResult(
            False,
            "蓝牙恢复超时",
            address,
            round(time.monotonic() - started, 2),
            output,
        )
