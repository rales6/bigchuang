"""非阻塞 machine.UART 字节流适配。"""


class UARTTransport:
    def __init__(self, uart, name, read_size=256):
        self.uart = uart
        self.name = name
        self.read_size = read_size

    def read(self):
        count = self.uart.any()
        if not count:
            return b""
        return self.uart.read(min(count, self.read_size)) or b""

    def write(self, data):
        return self.uart.write(data)

    def deinit(self):
        self.uart.deinit()

