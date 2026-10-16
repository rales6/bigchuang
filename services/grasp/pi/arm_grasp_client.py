"""树莓派侧客户端 — 接收 3090 arm_cmd 并通过 BLE 控制机械臂。

连接到 3090 的 /ws/grasp WebSocket:
  1. 发送 start 消息 + target_description
  2. 循环: 拍照 → 发送 JPEG 帧 → 接收 arm_cmd → 执行机械臂命令
  3. 用户 Ctrl+C 发送 stop 消息

用法:
  python -m grab_target.arm_grasp_client \
    --server ws://192.0.2.20:8020/ws/grasp \
    --target "红色的瓶子" \
    --camera-id 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

NUS_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
DEFAULT_BLE_ADDRESS = "AA:BB:CC:DD:EE:01"

try:
    import websockets
except ImportError:
    websockets = None


class ArmGraspClient:
    """通过 BLE ASCII 命令控制机械臂的客户端。"""

    def __init__(
        self,
        server_url: str,
        target_description: str = "",
        camera_id: int = 0,
        camera_width: int = 640,
        camera_height: int = 480,
        jpeg_quality: int = 70,
        ble_address: str = DEFAULT_BLE_ADDRESS,
        esp32_client=None,  # 可注入 ESP32 客户端用于测试
    ) -> None:
        self.server_url = server_url
        self.target_description = target_description
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.jpeg_quality = jpeg_quality
        self.ble_address = ble_address
        self._esp32 = esp32_client
        self._ble_client = None

        self.running = False
        self._last_arm_cmd: dict = {}
        self._frame_count: int = 0

    async def run(self) -> None:
        """主循环。"""
        if websockets is None:
            raise ImportError(
                "websockets 库未安装。请运行: pip install websockets"
            )

        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_id)
        # 请求摄像头硬件输出 MJPEG，避免默认未压缩 YUYV 占满 USB 带宽。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return

        print(f"连接到 {self.server_url}")
        print(f"目标: {self.target_description or '(未指定)'}")

        self.running = True

        try:
            if self._esp32 is None:
                from bleak import BleakClient
                self._ble_client = BleakClient(self.ble_address, timeout=10.0)
                await self._ble_client.connect()
                if not self._ble_client.is_connected:
                    raise ConnectionError(f"BLE connect failed: {self.ble_address}")

            async with websockets.connect(
                self.server_url,
                max_size=10 * 1024 * 1024,  # 10MB
            ) as ws:
                # 发送启动消息
                start_msg = {
                    "type": "start",
                    "target_description": self.target_description,
                }
                await ws.send(json.dumps(start_msg))
                ack = json.loads(await ws.recv())
                print(f"3090 响应: {ack.get('message', ack)}")

                # 帧循环
                while self.running:
                    # 拍照
                    ok, frame = cap.read()
                    if not ok:
                        print("摄像头读取失败")
                        await asyncio.sleep(0.1)
                        continue

                    self._frame_count += 1

                    # 编码 JPEG
                    ok, jpeg = cv2.imencode(
                        ".jpg", frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if not ok:
                        continue

                    # 发送
                    await ws.send(jpeg.tobytes())

                    # 接收 arm_cmd
                    raw = await ws.recv()
                    try:
                        response = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"无效 JSON 响应: {raw[:100]}")
                        continue

                    arm_cmd = response.get("arm_cmd", {})
                    phase = response.get("phase", "")
                    step_label = response.get("step_label", "")

                    # 打印状态
                    if arm_cmd.get("action") != self._last_arm_cmd.get("action"):
                        print(f"\n[{phase}] {step_label}: {arm_cmd.get('reason', '')}")

                    # 执行机械臂命令
                    if arm_cmd.get("action") in ("move", "grasp", "release"):
                        await self._execute_arm_cmd(arm_cmd)

                    # 完成/错误 → 退出
                    if arm_cmd.get("action") in ("done", "error"):
                        print(f"\n任务结束: {arm_cmd.get('reason', '')}")
                        break

                    # reground → 等待 Qwen 重识别
                    if arm_cmd.get("action") == "reground":
                        print("等待 Qwen 重识别...")

                    # 紧急停止
                    if arm_cmd.get("emergency_stop"):
                        print("⚠ 紧急停止!")
                        await self._execute_arm_stop()
                        break

                    self._last_arm_cmd = arm_cmd

        except asyncio.CancelledError:
            print("\n用户取消")
        except Exception as exc:
            print(f"连接错误: {exc}")
        finally:
            cap.release()
            try:
                await self._execute_arm_stop()
            except Exception as exc:
                print(f"停止命令发送失败: {exc}")
            if self._ble_client is not None:
                try:
                    await self._ble_client.disconnect()
                except Exception:
                    pass
                self._ble_client = None
            print("摄像头已关闭，机械臂已停止")

    async def _execute_arm_cmd(self, cmd: dict) -> None:
        """发送机械臂命令到 ESP32。"""
        action = cmd.get("action", "idle")

        if action == "idle":
            return

        if action == "move":
            forward = int(cmd.get("forward_mm", 0))
            up = int(cmd.get("up_mm", 0))
            left = int(cmd.get("left_deg", 0))
            claw = cmd.get("claw_open")

            # 构造 ASCII 命令: #ARMF{forward}U{up}L{left}[C{claw}]!
            parts = []
            # Match the installed firmware/web UI physical mapping.
            # D/U = forward/back; B/F = up/down.
            if forward >= 0:
                parts.append(f"D{forward}")
            else:
                parts.append(f"U{abs(forward)}")
            if up >= 0:
                parts.append(f"B{up}")
            else:
                parts.append(f"F{abs(up)}")
            if left >= 0:
                parts.append(f"L{left}")
            else:
                parts.append(f"R{abs(left)}")
            if claw is not None and claw >= 0:
                parts.append(f"C{int(claw)}")

            ascii_cmd = f"#ARM{''.join(parts)}!"
            await self._send_arm_ascii(ascii_cmd)

        elif action == "grasp":
            claw = int(cmd.get("claw_open", 10))
            ascii_cmd = f"#ARMC{claw}!"
            await self._send_arm_ascii(ascii_cmd)

        elif action == "release":
            claw = int(cmd.get("claw_open", 95))
            ascii_cmd = f"#ARMC{claw}!"
            await self._send_arm_ascii(ascii_cmd)

        # 等待机械臂运动完成
        duration_ms = int(cmd.get("duration_ms", 800))
        await asyncio.sleep(duration_ms / 1000.0)

    async def _execute_arm_stop(self) -> None:
        """停止机械臂。"""
        await self._send_arm_ascii("#STOP!")

    async def _send_arm_ascii(self, ascii_cmd: str) -> None:
        """通过 BLE 发送 ASCII 命令到 ESP32。

        如果注入了 esp32_client (测试用)，直接调用其方法。
        否则通过子进程调用现有 Pi 侧命令接口。
        """
        if self._esp32 is not None:
            # 注入模式: 直接调用客户端方法
            if hasattr(self._esp32, "send_ascii"):
                self._esp32.send_ascii(ascii_cmd)
            else:
                print(f"[TEST] 将发送: {ascii_cmd}")
            return

        if self._ble_client is None or not self._ble_client.is_connected:
            raise ConnectionError("BLE is not connected; refusing to simulate arm command")
        payload = (ascii_cmd.rstrip() + "\n").encode()
        await self._ble_client.write_gatt_char(NUS_RX, payload, response=True)
        print(f"[ARM] {ascii_cmd}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arm Grasp Client (Pi Side)")
    parser.add_argument("--server", default="ws://192.0.2.20:8020/ws/grasp")
    parser.add_argument("--target", default="", help="target description for Qwen")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--ble-address", default=DEFAULT_BLE_ADDRESS)
    args = parser.parse_args()

    client = ArmGraspClient(
        server_url=args.server,
        target_description=args.target,
        camera_id=args.camera_id,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        jpeg_quality=args.jpeg_quality,
        ble_address=args.ble_address,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n用户中断")
    except ImportError as exc:
        print(f"缺少依赖: {exc}")
        print("请运行: pip install websockets opencv-python")


if __name__ == "__main__":
    main()
