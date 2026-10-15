"""主链路失败后延迟创建备用链路。"""


class FailoverTransport:
    def __init__(self, primary_factory, fallback_factory):
        self._fallback_factory = fallback_factory
        self._primary = None
        self._fallback = None
        self.active = None
        self.name = "uninitialized"
        try:
            self._primary = primary_factory()
            self.active = self._primary
            self.name = "uart"
        except Exception:
            self._activate_fallback()

    @property
    def in_waiting(self):
        return getattr(self.active, "in_waiting", 0)

    def read(self, count=1):
        return self.active.read(count)

    def write(self, data):
        return self.active.write(data)

    def switch_to_fallback(self):
        if self.active is self._fallback:
            return False
        self._activate_fallback()
        return True

    def _activate_fallback(self):
        if self._fallback is None:
            self._fallback = self._fallback_factory()
        self.active = self._fallback
        self.name = "ble"

    def close(self):
        for transport in (self._primary, self._fallback):
            if transport is not None:
                close = getattr(transport, "close", None)
                if close:
                    close()
