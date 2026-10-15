import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "car"))

from drivers.actuator_bus import ActuatorBus  # noqa: E402
from protocol.frame import (  # noqa: E402
    ADDR_BASEBOARD,
    ADDR_ESP32,
    FLAG_RESPONSE,
    FrameParser,
    encode_frame,
)
from protocol.messages import (  # noqa: E402
    MSG_ACK,
    MSG_ACTUATOR_ARM,
    MSG_ACTUATOR_STOP,
    encode_ack,
)


class FakeTransport:
    def __init__(self):
        self.rx = b""
        self.writes = []

    def read(self):
        data, self.rx = self.rx, b""
        return data

    def write(self, data):
        self.writes.append(data)
        return len(data)


class ActuatorBusTest(unittest.TestCase):
    def test_only_one_reliable_request_is_in_flight(self):
        transport = FakeTransport()
        bus = ActuatorBus(transport, (ADDR_BASEBOARD,), guard_ms=0)
        now = bus._last_tx_ms + 1
        bus.send_arm(b"arm-one")
        bus.send_arm(b"arm-two")
        bus.poll(now)
        self.assertEqual(1, len(transport.writes))
        bus.poll(now + 1)
        self.assertEqual(1, len(transport.writes))

        request = FrameParser().feed(transport.writes[0])[0]
        transport.rx = encode_frame(
            MSG_ACK, request.seq, encode_ack(MSG_ACTUATOR_ARM),
            src=ADDR_BASEBOARD, dst=ADDR_ESP32, flags=FLAG_RESPONSE,
        )
        bus.poll(now + 2)
        self.assertEqual(2, len(transport.writes))

    def test_new_arm_command_inserts_stop_first(self):
        transport = FakeTransport()
        bus = ActuatorBus(transport, (ADDR_BASEBOARD,), guard_ms=0)
        bus.preempt_arm(b"new-arm")
        bus.poll(bus._last_tx_ms + 1)
        frame = FrameParser().feed(transport.writes[0])[0]
        self.assertEqual(MSG_ACTUATOR_STOP, frame.msg_type)

    def test_wrong_source_is_rejected(self):
        transport = FakeTransport()
        bus = ActuatorBus(transport, (ADDR_BASEBOARD,), guard_ms=0)
        bus.send_arm(b"arm")
        now = bus._last_tx_ms + 1
        bus.poll(now)
        request = FrameParser().feed(transport.writes[0])[0]
        transport.rx = encode_frame(
            MSG_ACK, request.seq, b"", src=0x10, dst=ADDR_ESP32,
            flags=FLAG_RESPONSE,
        )
        bus.poll(now + 1)
        self.assertEqual(1, bus.unexpected_count)
        self.assertIsNotNone(bus._pending)


if __name__ == "__main__":
    unittest.main()

