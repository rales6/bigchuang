"""Reliable USB-serial transport for the ESP32 Type-C connection."""

import time


def resolve_usb_port(requested_port):
    """Resolve ``auto`` to the most likely ESP32 USB serial device."""
    if requested_port and str(requested_port).lower() != "auto":
        return str(requested_port)
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError("缺少 pyserial，请安装 requirements-pi.txt") from exc

    candidates = []
    keywords = (
        "espressif",
        "cp210",
        "ch340",
        "ch910",
        "silicon labs",
        "usb jtag",
        "usb serial",
    )
    for info in list_ports.comports():
        device = str(getattr(info, "device", ""))
        identity = " ".join(str(value or "") for value in (
            getattr(info, "manufacturer", ""),
            getattr(info, "product", ""),
            getattr(info, "description", ""),
            getattr(info, "hwid", ""),
        )).lower()
        score = sum(20 for keyword in keywords if keyword in identity)
        if "ttyusb" in device.lower():
            score += 10
        elif "ttyacm" in device.lower():
            score += 2
        if device:
            candidates.append((score, device))
    if not candidates:
        raise RuntimeError(
            "未发现 USB 串口；请检查数据线并执行 ls -l /dev/serial/by-id/"
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


class UsbSerialTransport:
    """Small serial-like wrapper with ESP32 boot settling and clean buffers."""

    name = "usb"

    def __init__(
        self,
        port="auto",
        baudrate=230400,
        timeout_s=0.03,
        boot_settle_s=1.5,
    ):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError(
                "USB 串口链路需要 pyserial，请安装 requirements-pi.txt"
            ) from exc
        self.port = resolve_usb_port(port)
        self.serial = serial.Serial(
            self.port,
            int(baudrate),
            timeout=float(timeout_s),
            write_timeout=max(0.3, float(timeout_s) * 10.0),
            rtscts=False,
            dsrdtr=False,
        )
        # Avoid holding the ESP32 in an auto-program/reset state. Some bridge
        # chips still reset once when opened, so wait for main.py before the
        # first heartbeat and discard ROM/MicroPython startup text.
        try:
            self.serial.dtr = False
            self.serial.rts = False
        except Exception:
            pass
        time.sleep(max(0.0, float(boot_settle_s)))
        reset_input = getattr(self.serial, "reset_input_buffer", None)
        if reset_input is not None:
            reset_input()
        reset_output = getattr(self.serial, "reset_output_buffer", None)
        if reset_output is not None:
            reset_output()

    @property
    def in_waiting(self):
        return self.serial.in_waiting

    def read(self, count=1):
        return self.serial.read(count)

    def write(self, data):
        return self.serial.write(data)

    def close(self):
        return self.serial.close()
