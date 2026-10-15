"""树莓派侧可替换的 UART/BLE 字节流。"""

from .factory import create_transport
from .failover import FailoverTransport

__all__ = ["create_transport", "FailoverTransport"]
