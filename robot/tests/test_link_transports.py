import os
import sys
import unittest
import asyncio
import threading
import types
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "car"))

from car.protocol.frame import (  # noqa: E402
    ADDR_ESP32, ADDR_RASPBERRY_PI, FLAG_ACK_REQUIRED, FLAG_RESPONSE,
    FrameParser, encode_frame,
)
from car.protocol.messages import MSG_ACK, MSG_HEARTBEAT, encode_ack  # noqa: E402
from raspberry_pi.config import SerialConfig  # noqa: E402
from raspberry_pi.esp32 import Esp32Client  # noqa: E402
from raspberry_pi.transport.failover import FailoverTransport  # noqa: E402
from raspberry_pi.transport.ble import BleSerialTransport  # noqa: E402
from raspberry_pi.transport.usb import UsbSerialTransport  # noqa: E402
from transport.multiplex import MultiplexTransport  # noqa: E402


class ByteTransport:
    def __init__(self, name, rx=b""):
        self.name = name
        self.rx = bytearray(rx)
        self.tx = bytearray()

    @property
    def in_waiting(self):
        return len(self.rx)

    def read(self, count=None):
        if count is None:
            count = len(self.rx)
        data = bytes(self.rx[:count])
        del self.rx[:count]
        return data

    def write(self, data):
        self.tx.extend(data)
        return len(data)


class RespondingTransport(ByteTransport):
    def __init__(self, name):
        super().__init__(name)
        self.parser = FrameParser()

    def write(self, data):
        for request in self.parser.feed(data):
            self.rx.extend(encode_frame(
                MSG_ACK, request.seq, encode_ack(request.msg_type),
                src=ADDR_ESP32, dst=ADDR_RASPBERRY_PI, flags=FLAG_RESPONSE,
            ))
        return len(data)


class LinkTransportTest(unittest.TestCase):
    def test_usb_serial_waits_for_boot_and_clears_startup_text(self):
        class FakeSerialPort:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.dtr = True
                self.rts = True
                self.input_resets = 0
                self.output_resets = 0
                self.rx = bytearray(b"reply")
                self.tx = bytearray()
                self.closed = False

            @property
            def in_waiting(self):
                return len(self.rx)

            def reset_input_buffer(self):
                self.input_resets += 1
                self.rx.clear()

            def reset_output_buffer(self):
                self.output_resets += 1

            def read(self, count):
                data = bytes(self.rx[:count])
                del self.rx[:count]
                return data

            def write(self, data):
                self.tx.extend(data)
                return len(data)

            def close(self):
                self.closed = True

        fake_port = FakeSerialPort()
        fake_serial = types.SimpleNamespace(Serial=lambda *_args, **_kwargs: fake_port)
        with (
            mock.patch.dict(sys.modules, {"serial": fake_serial}),
            mock.patch("raspberry_pi.transport.usb.time.sleep") as sleep,
        ):
            transport = UsbSerialTransport(
                "/dev/ttyUSB0",
                boot_settle_s=1.5,
            )

        sleep.assert_called_once_with(1.5)
        self.assertFalse(fake_port.dtr)
        self.assertFalse(fake_port.rts)
        self.assertEqual(1, fake_port.input_resets)
        self.assertEqual(1, fake_port.output_resets)
        self.assertEqual("usb", transport.name)
        self.assertEqual(3, transport.write(b"cmd"))
        self.assertEqual(b"cmd", bytes(fake_port.tx))
        transport.close()
        self.assertTrue(fake_port.closed)

    def test_failed_ble_startup_does_not_leak_event_loop_thread(self):
        class SlowClient:
            def __init__(self, *_args, **_kwargs):
                self.is_connected = False

            async def connect(self):
                await asyncio.sleep(10.0)

            async def start_notify(self, *_args):
                pass

            async def disconnect(self):
                self.is_connected = False

        class FakeScanner:
            @staticmethod
            async def find_device_by_name(_name, timeout):
                return object()

        fake_bleak = types.SimpleNamespace(
            BleakClient=SlowClient,
            BleakScanner=FakeScanner,
        )
        before = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("ble-nus-")
        }
        with mock.patch.dict(sys.modules, {"bleak": fake_bleak}):
            with self.assertRaisesRegex(
                RuntimeError,
                r"GATT 连接与服务发现超时.*单阶段上限 0.1s",
            ):
                BleSerialTransport(
                    "timeout-test",
                    connect_timeout_s=0.05,
                    operation_timeout_s=0.05,
                )
        after = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("ble-nus-")
        }
        self.assertEqual(before, after)

    def test_disconnect_during_service_discovery_cleans_up_task(self):
        class DisconnectingClient:
            instances = []

            def __init__(
                self,
                *_args,
                disconnected_callback=None,
                **_kwargs
            ):
                self.is_connected = False
                self.callback = disconnected_callback
                self.__class__.instances.append(self)

            async def connect(self):
                self.is_connected = True
                self.callback(self)

            async def start_notify(self, *_args):
                self.is_connected = False
                raise ConnectionError("disconnected during discovery")

            async def disconnect(self):
                self.is_connected = False

        class FakeScanner:
            @staticmethod
            async def find_device_by_name(_name, timeout):
                return object()

        fake_bleak = types.SimpleNamespace(
            BleakClient=DisconnectingClient,
            BleakScanner=FakeScanner,
        )
        before = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("ble-nus-")
        }
        with mock.patch.dict(sys.modules, {"bleak": fake_bleak}):
            with self.assertRaisesRegex(
                RuntimeError,
                "disconnected during discovery",
            ):
                BleSerialTransport(
                    "disconnect-test",
                    connect_timeout_s=0.2,
                    operation_timeout_s=0.05,
                )
        after = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("ble-nus-")
        }
        self.assertEqual(before, after)

    def test_ble_nus_uses_fast_unacknowledged_gatt_chunks(self):
        class FakeGattClient:
            def __init__(self):
                self.writes = []

            async def write_gatt_char(
                self, characteristic, data, response
            ):
                self.writes.append((
                    characteristic,
                    bytes(data),
                    response,
                ))

        transport = BleSerialTransport.__new__(BleSerialTransport)
        transport._client = FakeGattClient()

        asyncio.run(transport._write_chunks(b"x" * 27))

        self.assertEqual(
            [20, 7],
            [len(item[1]) for item in transport._client.writes],
        )
        self.assertTrue(all(
            item[2] is False
            for item in transport._client.writes
        ))

    def test_ble_nus_confirms_acknowledged_request_at_att_layer(self):
        class FakeGattClient:
            def __init__(self):
                self.writes = []

            async def write_gatt_char(
                self, characteristic, data, response
            ):
                self.writes.append((
                    characteristic,
                    bytes(data),
                    response,
                ))

        transport = BleSerialTransport.__new__(BleSerialTransport)
        transport._client = FakeGattClient()
        packet = encode_frame(
            MSG_HEARTBEAT,
            7,
            src=ADDR_RASPBERRY_PI,
            dst=ADDR_ESP32,
            flags=FLAG_ACK_REQUIRED,
        )

        asyncio.run(transport._write_chunks(packet))

        self.assertEqual(1, len(transport._client.writes))
        self.assertTrue(transport._client.writes[0][2])

    def test_uart_has_read_priority_and_response_returns_to_source(self):
        uart = ByteTransport("uart", b"u")
        ble = ByteTransport("ble", b"b")
        mux = MultiplexTransport((uart, ble))
        self.assertEqual(b"u", mux.read())
        mux.write(b"1")
        self.assertEqual(b"1", bytes(uart.tx))
        self.assertEqual(b"b", mux.read())
        mux.write(b"2")
        self.assertEqual(b"2", bytes(ble.tx))

    def test_client_switches_from_silent_uart_to_ble(self):
        uart = ByteTransport("uart")
        ble = RespondingTransport("ble")
        link = FailoverTransport(lambda: uart, lambda: ble)
        client = Esp32Client(
            SerialConfig(request_timeout_s=0.001, retries=1), link
        )
        response = client.heartbeat()
        self.assertEqual(MSG_ACK, response.msg_type)
        self.assertEqual("ble", client.active_transport)


if __name__ == "__main__":
    unittest.main()
