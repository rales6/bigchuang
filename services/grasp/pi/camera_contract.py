"""Raw USB ports 0/2; taught physical-left/right input 2/0.

Toe-in disparity changes sign with depth. This configuration uses the
operator port identification and the 2026-09-26 model-coordinate replay.
"""
import os
from pathlib import Path
import sys

RAW_DEVICES = (0, 2)
MODEL_DEVICES = (2, 0)
MODEL_PORTS = (
    '/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:2:1.0-video-index0',
    '/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0',
)


def model_devices(*, require_ports=False):
    """Resolve fixed USB ports, so a reboot can change videoN harmlessly."""
    if sys.platform != 'linux': return MODEL_DEVICES
    paths=(Path(os.environ.get('FINAL_WORK_LEFT_CAMERA_PORT',MODEL_PORTS[0])),
           Path(os.environ.get('FINAL_WORK_RIGHT_CAMERA_PORT',MODEL_PORTS[1])))
    if not all(path.exists() for path in paths):
        if require_ports:
            raise RuntimeError('已绑定的双目 USB 路径缺失，请恢复原接口；禁止按枚举编号猜测左右')
        return MODEL_DEVICES  # logical geometry on GPU/non-camera test hosts
    names=tuple(path.resolve().name for path in paths)
    if any(not name.startswith('video') or not name[5:].isdigit() for name in names):
        raise RuntimeError('相机 USB 路径没有解析到 videoN')
    devices=tuple(int(name[5:]) for name in names)
    if devices[0] == devices[1]: raise RuntimeError('左右相机不能绑定同一个设备')
    return devices


def acquisition_devices(config):
    raw=model_devices()[::-1]
    left = config.left_device if config.left_device >= 0 else raw[0]
    right = config.right_device if config.right_device >= 0 else raw[1]
    pair = (left, right)
    return pair[::-1] if config.swap_left_right else pair


def geometry_needs_swap(config):
    pair = acquisition_devices(config)
    expected=model_devices(require_ports=sys.platform == 'linux')
    if pair == expected: return False
    if pair == expected[::-1]: return True
    raise ValueError(f'相机编号 {pair} 未与当前示教模型绑定，禁止猜测掩码顺序')


def preview_source(camera, raw_order_swapped):
    """Preview endpoints now use MODEL_DEVICES, not RAW_DEVICES."""
    if camera not in ('left','right'): raise ValueError('invalid camera')
    return camera if raw_order_swapped else {'left':'right','right':'left'}[camera]
