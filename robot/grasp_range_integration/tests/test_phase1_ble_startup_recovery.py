"""Offline tests: startup BLE repair never starts chassis motion."""

from __future__ import annotations

import subprocess
import os
import types
import unittest
from unittest.mock import patch

from grasp_range_integration.phase1 import ble_startup_recovery as recovery


class BleStartupRecoveryTests(unittest.TestCase):
    def test_discovery_failure_runs_operator_sequence_once_then_retries(self):
        calls = []

        def original(_factory, *, attempts, delay_s):
            calls.append((attempts, delay_s))
            if len(calls) == 1:
                raise RuntimeError("底盘链路连续连接失败 2 次：BLE 连接失败: 已知地址扫描超时")
            return "stopped_client"

        mission = types.SimpleNamespace(_create_client_with_retry=original)
        with patch.object(recovery, "reset_bluetooth_for_missing_esp32") as reset:
            recovery.install_ble_startup_recovery(mission)
            self.assertEqual(mission._create_client_with_retry(lambda: None, 8, 1.0),
                             "stopped_client")
        self.assertEqual(calls, [(2, 1.0), (6, 1.0)])
        reset.assert_called_once_with()

    def test_other_link_error_never_restarts_bluetooth(self):
        calls = []

        def original(_factory, *, attempts, delay_s):
            calls.append(attempts)
            if len(calls) == 1:
                raise RuntimeError("ESP32 未应答：心跳超时")
            return "stopped_client"

        mission = types.SimpleNamespace(_create_client_with_retry=original)
        with patch.object(recovery, "reset_bluetooth_for_missing_esp32") as reset:
            recovery.install_ble_startup_recovery(mission)
            self.assertEqual(mission._create_client_with_retry(lambda: None), "stopped_client")
        self.assertEqual(calls, [2, 6])
        reset.assert_not_called()

    def test_reset_command_order_and_missing_rfkill_fallback(self):
        commands = []

        def fake_run(command, **_kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.object(recovery, "_check_no_other_motion_process"), \
                patch.object(recovery.subprocess, "run", fake_run), \
                patch.object(recovery.shutil, "which", return_value=None), \
                patch.object(recovery.Path, "is_file", return_value=False), \
                patch.object(recovery.time, "sleep") as sleep:
            recovery.reset_bluetooth_for_missing_esp32()
        self.assertEqual(commands, [
            ["bluetoothctl", "disconnect", recovery.ESP32_ADDRESS],
            ["sudo", "-n", "systemctl", "restart", "bluetooth"],
            ["bluetoothctl", "power", "on"],
        ])
        sleep.assert_called_once_with(3)

    def test_reset_includes_rfkill_when_available(self):
        commands = []

        def fake_run(command, **_kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.object(recovery, "_check_no_other_motion_process"), \
                patch.object(recovery.subprocess, "run", fake_run), \
                patch.object(recovery.shutil, "which", return_value="/usr/sbin/rfkill"), \
                patch.object(recovery.time, "sleep"):
            recovery.reset_bluetooth_for_missing_esp32()
        self.assertEqual(commands[2], ["sudo", "-n", "/usr/sbin/rfkill",
                                       "unblock", "bluetooth"])

    def test_rfkill_absolute_path_used_when_not_in_user_path(self):
        commands = []

        def fake_run(command, **_kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with patch.object(recovery, "_check_no_other_motion_process"), \
                patch.object(recovery.subprocess, "run", fake_run), \
                patch.object(recovery.shutil, "which", return_value=None), \
                patch.object(recovery.Path, "is_file",
                             lambda path: path.as_posix() == "/usr/sbin/rfkill"), \
                patch.object(recovery.time, "sleep"):
            recovery.reset_bluetooth_for_missing_esp32()
        self.assertEqual(commands[2], ["sudo", "-n", str(recovery.Path("/usr/sbin/rfkill")),
                                       "unblock", "bluetooth"])

    def test_parallel_motion_process_blocks_reset(self):
        output = (f"{os.getpid()} python relationship_find_only_entry.py\n"
                  "8765 python run_visual_pursuit.py\n")
        with patch.object(recovery.subprocess, "run", return_value=
                          subprocess.CompletedProcess(["ps"], 0, stdout=output, stderr="")):
            with self.assertRaisesRegex(RuntimeError, "其他实车运动"):
                recovery._check_no_other_motion_process()

    def test_bluetooth_restart_failure_stops_before_power_on(self):
        commands = []

        def fake_run(command, **_kwargs):
            commands.append(command)
            code = 1 if command[:4] == ["sudo", "-n", "systemctl", "restart"] else 0
            return subprocess.CompletedProcess(command, code, stdout="", stderr="failed")

        with patch.object(recovery, "_check_no_other_motion_process"), \
                patch.object(recovery.subprocess, "run", fake_run):
            with self.assertRaisesRegex(RuntimeError, "蓝牙恢复命令失败"):
                recovery.reset_bluetooth_for_missing_esp32()
        self.assertEqual(len(commands), 2)


if __name__ == "__main__":
    unittest.main()
