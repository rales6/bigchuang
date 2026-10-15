import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

from transport.bluetooth import (  # noqa: E402
    BLEUARTTransport, _IRQ_CENTRAL_CONNECT, _IRQ_GATTS_WRITE,
)


class FakeBLE:
    def __init__(self):
        self.values = {}
        self.notifications = []

    def active(self, value):
        self.enabled = value

    def config(self, **kwargs):
        self.configuration = kwargs

    def gatts_register_services(self, _services):
        return ((10, 11),)

    def gatts_set_buffer(self, *_args):
        pass

    def irq(self, handler):
        self.handler = handler

    def gap_advertise(self, *args, **kwargs):
        self.advertisement = (args, kwargs)

    def gatts_read(self, handle):
        return self.values.get(handle, b"")

    def gatts_notify(self, connection, handle, data):
        self.notifications.append((connection, handle, bytes(data)))


class Esp32BleTransportTest(unittest.TestCase):
    def test_receive_and_fragmented_notify(self):
        radio = FakeBLE()
        transport = BLEUARTTransport(ble=radio)
        radio.handler(_IRQ_CENTRAL_CONNECT, (7, 0, b"addr"))
        radio.values[10] = b"command"
        radio.handler(_IRQ_GATTS_WRITE, (7, 10))
        self.assertEqual(b"command", transport.read())

        transport.write(b"x" * 45)
        self.assertEqual([20, 20, 5], [len(item[2]) for item in radio.notifications])

    def test_notification_failure_does_not_crash_control_transport(self):
        class FailingNotifyBLE(FakeBLE):
            def gatts_notify(self, connection, handle, data):
                raise OSError("connection disappeared")

        radio = FailingNotifyBLE()
        transport = BLEUARTTransport(ble=radio)
        radio.handler(_IRQ_CENTRAL_CONNECT, (7, 0, b"addr"))

        written = transport.write(b"ack")

        self.assertEqual(3, written)
        self.assertEqual(1, transport.notify_errors)

    def test_transient_notification_failure_is_retried_from_main_loop(self):
        class FailOnceBLE(FakeBLE):
            def __init__(self):
                super().__init__()
                self.failures_left = 1

            def gatts_notify(self, connection, handle, data):
                if self.failures_left:
                    self.failures_left -= 1
                    raise OSError("temporary notify queue pressure")
                super().gatts_notify(connection, handle, data)

        radio = FailOnceBLE()
        transport = BLEUARTTransport(ble=radio)
        radio.handler(_IRQ_CENTRAL_CONNECT, (7, 0, b"addr"))

        transport.write(b"ack")
        self.assertEqual([], radio.notifications)
        transport.read()

        self.assertEqual([(7, 11, b"ack")], radio.notifications)
        self.assertEqual(1, transport.notify_errors)
        self.assertEqual(0, transport.notify_drops)


if __name__ == "__main__":
    unittest.main()
