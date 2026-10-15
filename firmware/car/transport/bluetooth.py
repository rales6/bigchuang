"""ESP32 MicroPython 的 BLE Nordic UART Service 字节流适配器。"""

try:
    import bluetooth
except ImportError:  # 允许在电脑上导入并运行单元测试
    bluetooth = None


_IRQ_CENTRAL_CONNECT = 1
_IRQ_CENTRAL_DISCONNECT = 2
_IRQ_GATTS_WRITE = 3
_FLAG_WRITE_NO_RESPONSE = 0x0004
_FLAG_WRITE = 0x0008
_FLAG_NOTIFY = 0x0010

_NUS_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
_NUS_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # Pi -> ESP32
_NUS_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # ESP32 -> Pi


class BLEUARTTransport:
    """提供与 ``UARTTransport`` 相同的 read/write/deinit 接口。"""

    def __init__(self, name="ESP32-Robot-Car", rx_buffer=1024, ble=None):
        if ble is None:
            if bluetooth is None:
                raise RuntimeError("MicroPython bluetooth module is unavailable")
            ble = bluetooth.BLE()
        self.name = "ble"
        self.device_name = name
        self.ble = ble
        self._rx = bytearray()
        self._connections = set()
        self._pending_notifications = []
        self.notify_errors = 0
        self.notify_drops = 0
        self.ble.active(True)
        self.ble.config(gap_name=name)

        uuid = bluetooth.UUID if bluetooth is not None else str
        service = (
            uuid(_NUS_SERVICE_UUID),
            (
                (uuid(_NUS_RX_UUID), _FLAG_WRITE | _FLAG_WRITE_NO_RESPONSE),
                (uuid(_NUS_TX_UUID), _FLAG_NOTIFY),
            ),
        )
        ((self._rx_handle, self._tx_handle),) = self.ble.gatts_register_services(
            (service,)
        )
        self.ble.gatts_set_buffer(self._rx_handle, rx_buffer, True)
        self.ble.irq(self._irq)
        self._advertise()

    def _irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            conn_handle, _addr_type, _addr = data
            self._connections.add(conn_handle)
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn_handle, _addr_type, _addr = data
            self._connections.discard(conn_handle)
            self._pending_notifications = [
                item for item in self._pending_notifications
                if item[0] != conn_handle
            ]
            self._advertise()
        elif event == _IRQ_GATTS_WRITE:
            _conn_handle, value_handle = data
            if value_handle == self._rx_handle:
                value = self.ble.gatts_read(self._rx_handle)
                if value:
                    self._rx.extend(value)

    def _advertise(self):
        name = self.device_name.encode("utf-8")[:26]
        payload = bytes((2, 0x01, 0x06, len(name) + 1, 0x09)) + name
        self.ble.gap_advertise(100_000, adv_data=payload)

    def read(self):
        self._flush_notification()
        if not self._rx:
            return b""
        # Swap the live IRQ buffer before converting it. If a BLE write IRQ
        # runs between these assignments it appends either to ``data`` or to
        # the new buffer, so bytes can no longer be silently discarded.
        data = self._rx
        self._rx = bytearray()
        return bytes(data)

    def write(self, data):
        # 20 字节兼容默认 ATT MTU；协议解析器天然支持拆包。
        chunks = [
            bytes(data[offset:offset + 20])
            for offset in range(0, len(data), 20)
        ]
        for conn_handle in tuple(self._connections):
            connection_has_backlog = any(
                item[0] == conn_handle
                for item in self._pending_notifications
            )
            for index, chunk in enumerate(chunks):
                if connection_has_backlog:
                    self._queue_notifications(
                        conn_handle,
                        chunks[index:],
                    )
                    break
                try:
                    self.ble.gatts_notify(
                        conn_handle,
                        self._tx_handle,
                        chunk,
                    )
                except OSError:
                    # MicroPython can report transient queue pressure while
                    # the controller remains connected. Retry later from the
                    # 5 ms main loop instead of silently dropping the reply.
                    self.notify_errors += 1
                    self._queue_notifications(
                        conn_handle,
                        chunks[index:],
                    )
                    break
        return len(data)

    def _queue_notifications(self, conn_handle, chunks):
        for chunk in chunks:
            if len(self._pending_notifications) >= 64:
                self.notify_drops += 1
                break
            self._pending_notifications.append(
                (conn_handle, bytes(chunk), 0)
            )

    def _flush_notification(self):
        if not self._pending_notifications:
            return
        conn_handle, chunk, attempts = self._pending_notifications[0]
        if conn_handle not in self._connections:
            self._pending_notifications.pop(0)
            return
        try:
            self.ble.gatts_notify(
                conn_handle,
                self._tx_handle,
                chunk,
            )
            self._pending_notifications.pop(0)
        except OSError:
            self.notify_errors += 1
            attempts += 1
            if attempts >= 40:
                self.notify_drops += 1
                self._pending_notifications.pop(0)
            else:
                self._pending_notifications[0] = (
                    conn_handle,
                    chunk,
                    attempts,
                )

    def deinit(self):
        self.ble.gap_advertise(None)
        self.ble.active(False)
