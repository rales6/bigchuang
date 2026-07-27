"""根据集中配置创建树莓派通信链路。"""

from .ble import BleSerialTransport
from .failover import FailoverTransport


def _uart(config):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("缺少 pyserial，请安装 requirements-pi.txt") from exc
    return serial.Serial(config.port, config.baudrate, timeout=config.timeout_s)


def _ble(config):
    return BleSerialTransport(
        config.ble_device_name,
        address=config.ble_address,
        connect_timeout_s=config.ble_connect_timeout_s,
        operation_timeout_s=config.ble_operation_timeout_s,
    )


def create_transport(config):
    mode = config.link_mode.lower()
    if mode == "uart":
        return _uart(config)
    if mode == "ble":
        return _ble(config)
    if mode == "auto":
        return FailoverTransport(lambda: _uart(config), lambda: _ble(config))
    raise ValueError("link_mode must be uart, ble or auto")
