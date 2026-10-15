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
        connect_timeout_s=6.0,
        operation_timeout_s=0.8,
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
        # Discovery, GATT connection/service discovery and notification setup
        # are separate BlueZ operations.  A single connect_timeout deadline
        # around all three cancels healthy (but slightly slow) Raspberry Pi
        # connections before service discovery can finish.
        self._startup_timeout_s = max(
            0.5,
            4.0 * float(connect_timeout_s) + 1.0,
        )
        self._startup_phase = "启动"
        self._rx = deque()
        self._rx_size = 0
        self._rx_lock = threading.Lock()
        self._ready = threading.Event()
        self._thread_done = threading.Event()
        self._closing = threading.Event()
        self._startup_error = None
        self._loop = asyncio.new_event_loop()
        self._connect_task = None
        self._client = None
        self._disconnect_error = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="ble-nus-{}".format(device_name),
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(self._startup_timeout_s + 0.5):
            self._stop_thread()
            raise TimeoutError(
                "BLE 连接线程在 {:.1f}s 内未结束，当前阶段：{}".format(
                    self._startup_timeout_s,
                    self._startup_phase,
                )
            )
        if self._startup_error:
            self._stop_thread()
            error_text = str(self._startup_error).strip()
            if not error_text:
                error_text = type(self._startup_error).__name__
            raise RuntimeError("BLE 连接失败: {}".format(error_text))

    @property
    def in_waiting(self):
        with self._rx_lock:
            return self._rx_size

    def _thread_main(self):
        asyncio.set_event_loop(self._loop)
        try:
            # Keep an outer safety deadline for a genuinely stuck BlueZ call,
            # but allow each normal startup phase its configured timeout.
            self._connect_task = self._loop.create_task(
                asyncio.wait_for(
                    self._connect(),
                    timeout=self._startup_timeout_s,
                )
            )
            self._loop.run_until_complete(self._connect_task)
            if self._closing.is_set():
                return
            self._ready.set()
            self._loop.run_forever()
        except asyncio.TimeoutError:
            if not self._closing.is_set():
                error = self._connect_task.exception()
                detail = str(error).strip() if error is not None else ""
                self._startup_error = TimeoutError(
                    detail or "{}超时（BLE 安全总上限 {:.1f}s）".format(
                        self._startup_phase, self._startup_timeout_s
                    )
                )
        except asyncio.CancelledError:
            if not self._closing.is_set():
                self._startup_error = TimeoutError(
                    "{}被取消".format(self._startup_phase)
                )
        except Exception as exc:
            if not self._closing.is_set():
                self._startup_error = exc
        finally:
            task = self._connect_task
            if task is not None and not task.done():
                task.cancel()
                try:
                    self._loop.run_until_complete(task)
                except BaseException:
                    pass
            try:
                self._loop.run_until_complete(self._disconnect_client())
            except Exception:
                pass
            self._ready.set()
            try:
                self._loop.close()
            except Exception:
                pass
            self._thread_done.set()

    async def _connect(self):
        known_address = self.address or _DEVICE_ADDRESS_CACHE.get(
            self.device_name
        )
        target = None
        if known_address:
            # Always resolve a fresh BLEDevice instead of handing Bleak a raw
            # address.  On BlueZ a raw address triggers hidden discovery from
            # inside connect(); after a dropout that frequently ends in
            # "failed to discover services, device disconnected".
            find_by_address = getattr(
                self._BleakScanner,
                "find_device_by_address",
                None,
            )
            if find_by_address is not None:
                target = await self._await_startup_phase(
                    find_by_address(
                        known_address,
                        timeout=self.connect_timeout_s,
                    ),
                    "已知地址扫描",
                    self.connect_timeout_s,
                )
            else:
                target = known_address
        if target is None:
            device = await self._await_startup_phase(
                self._BleakScanner.find_device_by_name(
                    self.device_name,
                    timeout=self.connect_timeout_s,
                ),
                "设备名称扫描",
                self.connect_timeout_s,
            )
            if device is None:
                raise TimeoutError("未发现 {}".format(self.device_name))
            target = device
            resolved_address = getattr(device, "address", None)
            if resolved_address:
                _DEVICE_ADDRESS_CACHE[
                    self.device_name
                ] = resolved_address
        try:
            self._client = self._BleakClient(
                target,
                disconnected_callback=self._on_disconnect,
                timeout=self.connect_timeout_s,
            )
        except TypeError:
            # Keep compatibility with older Bleak releases that do not expose
            # a constructor timeout keyword.
            self._client = self._BleakClient(
                target,
                disconnected_callback=self._on_disconnect,
            )
        await self._await_startup_phase(
            self._client.connect(),
            "GATT 连接与服务发现",
            self.connect_timeout_s,
        )
        notify_timeout_s = max(
            2.0,
            min(
                self.connect_timeout_s,
                self.operation_timeout_s * 4.0,
            ),
        )
        await self._await_startup_phase(
            self._client.start_notify(
                NUS_TX_UUID,
                self._on_notification,
            ),
            "NUS 通知订阅",
            notify_timeout_s,
        )
        # BlueZ may complete start_notify slightly before the ESP32
        # notification path is ready, losing the very first heartbeat reply.
        await asyncio.sleep(0.20)
        self._startup_phase = "已连接"

    async def _await_startup_phase(self, awaitable, phase, timeout_s):
        self._startup_phase = phase
        try:
            return await asyncio.wait_for(
                awaitable,
                timeout=max(0.05, float(timeout_s)),
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                "{}超时（单阶段上限 {:.1f}s）".format(
                    phase,
                    timeout_s,
                )
            ) from exc

    def _on_disconnect(self, _client):
        self._disconnect_error = ConnectionError(
            "BLE peripheral disconnected"
        )
        # End this transport's event-loop thread immediately.  Reconnect uses
        # a new, clean BleakClient after this thread has fully released BlueZ.
        task = self._connect_task
        if (
            not self._closing.is_set()
            and task is not None
            and task.done()
            and self._loop.is_running()
        ):
            self._loop.call_soon_threadsafe(self._loop.stop)

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
        if self._loop.is_closed() or not self._loop.is_running():
            if self._disconnect_error is not None:
                raise self._disconnect_error
            raise ConnectionError("BLE event loop is not running")
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
        except Exception as exc:
            future.cancel()
            self._disconnect_error = ConnectionError(
                "BLE GATT write failed: {}".format(exc)
            )
            raise self._disconnect_error from exc
        return len(data)

    async def _write_chunks(self, data):
        # ACK-required commands also use an ATT confirmed write. Identical
        # motion/TTL refreshes have ACK_REQUIRED clear and retain the faster
        # write-without-response path.
        ack_required = (
            len(data) >= 4
            and data[:2] == b"\xA5\x5A"
            and bool(data[3] & 0x01)
        )
        offsets = tuple(range(0, len(data), 20))
        for index, offset in enumerate(offsets):
            await self._client.write_gatt_char(
                NUS_RX_UUID,
                data[offset:offset + 20],
                response=ack_required,
            )
            if index + 1 < len(offsets):
                # The balanced four-wheel command spans two ATT packets.
                # Eight milliseconds stays far below the command interval
                # while giving the MicroPython IRQ buffer time to drain.
                await asyncio.sleep(0.008)

    async def _disconnect_client(self):
        client = self._client
        if client is None:
            return
        try:
            # A cancelled BlueZ service-discovery call can report
            # is_connected=False while the controller still owns a half-open
            # ACL/GATT link. Always request disconnect before the next retry.
            await asyncio.wait_for(
                client.disconnect(),
                timeout=max(0.5, self.operation_timeout_s),
            )
        except Exception:
            pass

    def _stop_thread(self):
        self._closing.set()
        loop = self._loop
        if not loop.is_closed() and self._thread.is_alive():
            def request_stop():
                task = self._connect_task
                if task is not None and not task.done():
                    task.cancel()
                else:
                    loop.stop()

            try:
                loop.call_soon_threadsafe(request_stop)
            except RuntimeError:
                pass
        if (
            self._thread.is_alive()
            and self._thread is not threading.current_thread()
        ):
            self._thread.join(timeout=2.5)

    def close(self):
        self._stop_thread()
