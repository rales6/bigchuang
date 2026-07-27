"""用 Bleak 将 Nordic UART Service 包装为同步 serial-like 字节流。"""

import asyncio
import threading
from collections import deque
from concurrent.futures import TimeoutError as FutureTimeoutError


NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Pi -> ESP32
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # ESP32 -> Pi
_DEVICE_ADDRESS_CACHE = {}


class BleSerialTransport:
    def __init__(
        self,
        device_name,
        address=None,
        connect_timeout_s=8.0,
        operation_timeout_s=1.5,
    ):
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError as exc:
            raise RuntimeError(
                "BLE 备用链路需要 bleak，请安装 requirements-pi.txt"
            ) from exc

        self.name = "ble"
        self._BleakClient = BleakClient
        self._BleakScanner = BleakScanner
        self.device_name = device_name
        self.address = address
        self.connect_timeout_s = connect_timeout_s
        self.operation_timeout_s = operation_timeout_s
        self._rx = deque()
        self._rx_size = 0
        self._rx_lock = threading.Lock()
        self._ready = threading.Event()
        self._startup_error = None
        self._loop = asyncio.new_event_loop()
        self._client = None
        self._disconnect_error = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(connect_timeout_s + 1.0):
            raise TimeoutError("BLE 连接线程启动超时")
        if self._startup_error:
            raise RuntimeError("BLE 连接失败: {}".format(self._startup_error))

    @property
    def in_waiting(self):
        with self._rx_lock:
            return self._rx_size

    def _thread_main(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _connect(self):
        target = self.address or _DEVICE_ADDRESS_CACHE.get(
            self.device_name
        )
        if not target:
            device = await self._BleakScanner.find_device_by_name(
                self.device_name, timeout=self.connect_timeout_s
            )
            if device is None:
                raise TimeoutError("未发现 {}".format(self.device_name))
            target = device
            resolved_address = getattr(device, "address", None)
            if resolved_address:
                _DEVICE_ADDRESS_CACHE[
                    self.device_name
                ] = resolved_address
        self._client = self._BleakClient(
            target,
            disconnected_callback=self._on_disconnect,
        )
        await self._client.connect()
        await self._client.start_notify(NUS_TX_UUID, self._on_notification)
        # BlueZ may complete start_notify slightly before the ESP32
        # notification path is ready, losing the very first heartbeat reply.
        await asyncio.sleep(0.20)

    def _on_disconnect(self, _client):
        self._disconnect_error = ConnectionError(
            "BLE peripheral disconnected"
        )

    def _on_notification(self, _sender, data):
        block = bytes(data)
        with self._rx_lock:
            self._rx.append(block)
            self._rx_size += len(block)

    def read(self, count=1):
        with self._rx_lock:
            remaining = max(0, int(count))
            output = bytearray()
            while remaining and self._rx:
                block = self._rx.popleft()
                take = min(remaining, len(block))
                output.extend(block[:take])
                self._rx_size -= take
                remaining -= take
                if take < len(block):
                    self._rx.appendleft(block[take:])
            return bytes(output)

    def write(self, data):
        if not self._client or not self._client.is_connected:
            if self._disconnect_error is not None:
                raise self._disconnect_error
            raise ConnectionError("BLE 未连接")
        future = asyncio.run_coroutine_threadsafe(self._write_chunks(bytes(data)), self._loop)
        try:
            future.result(timeout=self.operation_timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                "BLE GATT write exceeded {:.1f}s".format(
                    self.operation_timeout_s
                )
            ) from exc
        return len(data)

    async def _write_chunks(self, data):
        # NUS already has an application-level CRC, ACK and idempotent retry.
        # Write-without-response avoids two extra ATT round trips for the
        # 27-byte balanced motion frame. A tiny gap protects the ESP32 RX FIFO.
        offsets = tuple(range(0, len(data), 20))
        for index, offset in enumerate(offsets):
            await self._client.write_gatt_char(
                NUS_RX_UUID,
                data[offset:offset + 20],
                response=False,
            )
            if index + 1 < len(offsets):
                await asyncio.sleep(0.004)

    def close(self):
        if self._client and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._client.disconnect(), self._loop)
            try:
                future.result(timeout=2.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
