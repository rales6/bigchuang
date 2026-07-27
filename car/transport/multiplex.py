"""多个字节流之间的非阻塞选择器。

列表顺序就是优先级。每次读到数据后，响应会写回产生数据的那条链路。
"""


class MultiplexTransport:
    def __init__(self, transports, name="pi-link"):
        if not transports:
            raise ValueError("at least one transport is required")
        self.transports = tuple(transports)
        self.name = name
        self.active = self.transports[0]

    @property
    def active_name(self):
        return getattr(self.active, "name", self.active.__class__.__name__)

    def read(self):
        for transport in self.transports:
            data = transport.read()
            if data:
                self.active = transport
                return data
        return b""

    def write(self, data):
        return self.active.write(data)

    def deinit(self):
        for transport in self.transports:
            deinit = getattr(transport, "deinit", None)
            if deinit:
                deinit()
