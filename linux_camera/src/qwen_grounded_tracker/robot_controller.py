from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol


def _clip(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def _rounded(value: float) -> float:
    return round(float(value), 4)


class RobotOutput(Protocol):
    def open(self) -> None: ...

    def send(self, payload: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullRobotOutput:
    def open(self) -> None:
        pass

    def send(self, payload: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class DryRunRobotOutput:
    """安全占位输出：默认静默，不接真实电机。"""

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._last_payload_key = ""

    def open(self) -> None:
        if self.verbose:
            print("[Robot dry-run] enabled; commands will be printed only.")

    def send(self, payload: dict[str, Any]) -> None:
        if not self.verbose:
            return
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if key == self._last_payload_key:
            return
        self._last_payload_key = key
        print(f"[Robot dry-run] {json.dumps(payload, ensure_ascii=False)}")

    def close(self) -> None:
        if self.verbose:
            print("[Robot dry-run] closed.")


class SerialRobotOutput:
    """串口输出：给 Arduino/STM32/下位机发送一行 JSON。"""

    def __init__(self, port: str, baudrate: int) -> None:
        self.port = port
        self.baudrate = baudrate
        self.serial: Any | None = None

    def open(self) -> None:
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("serial mode requires pyserial: pip install pyserial") from exc
        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
        print(f"[Robot serial] opened {self.port} @ {self.baudrate}")

    def send(self, payload: dict[str, Any]) -> None:
        if self.serial is None:
            raise RuntimeError("serial output is not open")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.serial.write(line.encode("utf-8"))

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()
            self.serial = None


class Esp32RobotOutput:
    """ESP32 output using the raspberry_pi.esp32.Esp32Client protocol."""

    def __init__(
        self,
        link_mode: str,
        uart_port: str,
        uart_baudrate: int,
        ble_name: str,
        ble_address: str,
        ttl_ms: int,
        gripper_joint: int,
        gripper_open_us: int,
        gripper_close_us: int,
        arm_duration_ms: int,
    ) -> None:
        self.link_mode = link_mode
        self.uart_port = uart_port
        self.uart_baudrate = int(uart_baudrate)
        self.ble_name = ble_name
        self.ble_address = ble_address
        self.ttl_ms = int(ttl_ms)
        self.gripper_joint = int(gripper_joint)
        self.gripper_open_us = int(gripper_open_us)
        self.gripper_close_us = int(gripper_close_us)
        self.arm_duration_ms = int(arm_duration_ms)
        self.client: Any | None = None
        self._last_arm_action = ""

    def open(self) -> None:
        try:
            from raspberry_pi.config import SerialConfig  # type: ignore[import-not-found]
            from raspberry_pi.esp32 import Esp32Client  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "ESP32 mode requires raspberry_pi.config.SerialConfig and raspberry_pi.esp32.Esp32Client"
            ) from exc

        config_kwargs: dict[str, Any] = {
            "link_mode": self.link_mode,
            "port": self.uart_port,
            "baudrate": self.uart_baudrate,
            "ble_device_name": self.ble_name,
        }
        if self.ble_address:
            config_kwargs["ble_address"] = self.ble_address

        candidates: list[dict[str, Any]] = [config_kwargs]
        if self.ble_address:
            address_kwargs = dict(config_kwargs)
            address_kwargs.pop("ble_address", None)
            address_kwargs["ble_device_address"] = self.ble_address
            candidates.append(address_kwargs)
        if self.link_mode == "ble":
            ble_kwargs: dict[str, Any] = {
                "link_mode": "ble",
                "ble_device_name": self.ble_name,
            }
            if self.ble_address:
                candidates.append({**ble_kwargs, "ble_address": self.ble_address})
                candidates.append({**ble_kwargs, "ble_device_address": self.ble_address})
            candidates.append(ble_kwargs)
        elif self.link_mode == "uart":
            candidates.append(
                {
                    "link_mode": "uart",
                    "port": self.uart_port,
                    "baudrate": self.uart_baudrate,
                }
            )

        config = None
        last_type_error: TypeError | None = None
        for candidate in candidates:
            try:
                config = SerialConfig(**candidate)
                break
            except TypeError as exc:
                last_type_error = exc
        if config is None:
            raise RuntimeError(f"Could not create SerialConfig: {last_type_error}")

        self.client = Esp32Client(config)
        self.client.heartbeat()
        start = getattr(self.client, "start", None)
        if callable(start):
            start()
        print(f"[ESP32] connected through {self.link_mode}")

    def send(self, payload: dict[str, Any]) -> None:
        if self.client is None:
            raise RuntimeError("ESP32 output is not open")

        chassis = payload.get("chassis", {})
        arm = payload.get("arm", {})
        if not isinstance(chassis, dict):
            chassis = {}
        if not isinstance(arm, dict):
            arm = {}

        linear_m_s = float(chassis.get("linear", 0.0))
        angular_rad_s = float(chassis.get("angular", 0.0))
        linear_mm_s = int(round(linear_m_s * 1000.0))
        angular_mrad_s = int(round(angular_rad_s * 1000.0))
        if linear_mm_s == 0 and angular_mrad_s == 0:
            self.client.stop()
        else:
            self.client.set_twist(linear_mm_s, angular_mrad_s, ttl_ms=self.ttl_ms)

        action = str(arm.get("action", "hold"))
        if action == self._last_arm_action:
            return
        self._last_arm_action = action

        if action == "close_gripper":
            self.client.set_arm_joints(
                [(self.gripper_joint, self.gripper_close_us)],
                duration_ms=self.arm_duration_ms,
            )
        elif action == "open_gripper":
            self.client.set_arm_joints(
                [(self.gripper_joint, self.gripper_open_us)],
                duration_ms=self.arm_duration_ms,
            )
        elif action in {"hold", "stop"}:
            arm_stop = getattr(self.client, "arm_stop", None)
            if callable(arm_stop):
                arm_stop()

    def close(self) -> None:
        if self.client is None:
            return
        try:
            self.client.cancel_all()
        finally:
            self.client.close()
            self.client = None


@dataclass
class RobotCommandDispatcher:
    """把 3090 返回的视觉/任务结果转换成树莓派侧小车控制命令。

    默认策略很保守：
    - safety.blocked、急停、Qwen 等待期间，一律输出 STOP。
    - 只有 robot_command.subsystem == chassis 时才给底盘速度。
    - arm/gripper 动作会先让底盘速度归零，再发机械臂动作。
    """

    mode: str = "dry-run"
    serial_port: str = "/dev/ttyUSB0"
    serial_baudrate: int = 115200
    command_rate_hz: float = 10.0
    max_linear: float = 0.18
    max_angular: float = 0.45
    esp32_link: str = "ble"
    esp32_uart_port: str = "/dev/serial0"
    esp32_uart_baudrate: int = 230400
    esp32_ble_name: str = "ESP32-Robot-Car"
    esp32_ble_address: str = ""
    esp32_ttl_ms: int = 600
    gripper_joint: int = 5
    gripper_open_us: int = 1200
    gripper_close_us: int = 1550
    arm_duration_ms: int = 800
    debug_log: bool = False

    def __post_init__(self) -> None:
        self.output: RobotOutput
        if self.mode == "none":
            self.output = NullRobotOutput()
        elif self.mode == "serial":
            self.output = SerialRobotOutput(self.serial_port, self.serial_baudrate)
        elif self.mode == "esp32":
            self.output = Esp32RobotOutput(
                link_mode=self.esp32_link,
                uart_port=self.esp32_uart_port,
                uart_baudrate=self.esp32_uart_baudrate,
                ble_name=self.esp32_ble_name,
                ble_address=self.esp32_ble_address,
                ttl_ms=self.esp32_ttl_ms,
                gripper_joint=self.gripper_joint,
                gripper_open_us=self.gripper_open_us,
                gripper_close_us=self.gripper_close_us,
                arm_duration_ms=self.arm_duration_ms,
            )
        elif self.mode == "dry-run":
            self.output = DryRunRobotOutput(verbose=self.debug_log)
        else:
            raise ValueError(f"unknown robot mode: {self.mode}")
        self._last_sent_at = 0.0
        self._last_payload_key = ""

    def open(self) -> None:
        self.output.open()

    def close(self) -> None:
        # 退出程序时主动停车，避免最后一条运动命令残留在下位机。
        self.output.send(self.stop_payload("client closing"))
        self.output.close()

    def update_from_3090(self, response: dict[str, Any]) -> None:
        if self.mode == "none":
            return

        now = time.monotonic()
        min_interval = 1.0 / max(1.0, float(self.command_rate_hz))
        if now - self._last_sent_at < min_interval:
            return

        payload = self.payload_from_3090(response)
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if key == self._last_payload_key and now - self._last_sent_at < 1.0:
            return

        self.output.send(payload)
        self._last_payload_key = key
        self._last_sent_at = now

    def payload_from_3090(self, response: dict[str, Any]) -> dict[str, Any]:
        robot_command = response.get("robot_command", {})
        guidance = response.get("guidance", {})
        safety = response.get("safety", {})
        robot_task = response.get("robot_task", {})

        if not isinstance(robot_command, dict):
            robot_command = {}
        if not isinstance(guidance, dict):
            guidance = {}
        if not isinstance(safety, dict):
            safety = {}
        if not isinstance(robot_task, dict):
            robot_task = {}

        blocked = bool(safety.get("blocked", True))
        emergency_stop = bool(response.get("emergency_stop", False))
        subsystem = str(robot_command.get("subsystem") or "vision")
        action = str(robot_command.get("action") or "idle")
        reason = str(robot_command.get("reason") or guidance.get("reason") or "")

        if blocked or emergency_stop:
            return self.stop_payload(reason or "safety blocked", response)

        if subsystem == "chassis":
            linear = _clip(float(robot_command.get("linear", guidance.get("linear", 0.0))), self.max_linear)
            angular = _clip(float(robot_command.get("angular", guidance.get("angular", 0.0))), self.max_angular)
            return self.base_payload(
                response=response,
                chassis={"enabled": True, "linear": _rounded(linear), "angular": _rounded(angular)},
                arm={"action": "hold"},
                reason=reason,
            )

        if subsystem in {"arm", "gripper"}:
            return self.base_payload(
                response=response,
                chassis={"enabled": True, "linear": 0.0, "angular": 0.0},
                arm={"action": action},
                reason=reason,
            )

        if subsystem == "all" or action in {"stop", "idle"}:
            return self.stop_payload(reason or "stop requested", response)

        # vision/unknown 阶段只允许停车等待，例如正在 Qwen 框选目标或切换阶段。
        return self.stop_payload(reason or f"waiting for {subsystem}/{action}", response)

    def base_payload(
        self,
        response: dict[str, Any],
        chassis: dict[str, Any],
        arm: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        robot_command = response.get("robot_command", {})
        robot_task = response.get("robot_task", {})
        guidance = response.get("guidance", {})
        if not isinstance(robot_command, dict):
            robot_command = {}
        if not isinstance(robot_task, dict):
            robot_task = {}
        if not isinstance(guidance, dict):
            guidance = {}
        linear = float(chassis.get("linear", 0.0)) if isinstance(chassis, dict) else 0.0
        angular = float(chassis.get("angular", 0.0)) if isinstance(chassis, dict) else 0.0
        chassis = dict(chassis)
        chassis["linear_mm_s"] = int(round(linear * 1000.0))
        chassis["angular_mrad_s"] = int(round(angular * 1000.0))
        chassis["ttl_ms"] = self.esp32_ttl_ms
        return {
            "type": "robot_control",
            "time": round(time.time(), 3),
            "chassis": chassis,
            "arm": arm,
            "task": {
                "type": robot_task.get("task_type", "none"),
                "phase": robot_task.get("phase", "idle"),
                "completed": bool(robot_task.get("completed", False)),
            },
            "source": {
                "subsystem": robot_command.get("subsystem", "vision"),
                "action": robot_command.get("action", "idle"),
                "direction": guidance.get("direction", "STOP"),
                "reason": reason,
            },
        }

    def stop_payload(self, reason: str, response: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.base_payload(
            response=response or {},
            chassis={"enabled": True, "linear": 0.0, "angular": 0.0},
            arm={"action": "hold"},
            reason=reason,
        )
