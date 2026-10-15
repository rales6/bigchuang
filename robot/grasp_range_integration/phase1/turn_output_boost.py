"""Transient motor-output boost via the existing balanced-twist interface.

Firmware wheel output gains are limited to 0.50..1.50. Preserve relative
wheel balance, restore normal encoding immediately, and never apply to drive.
This changes traction output, not the firmware's acceleration configuration.
"""

from dataclasses import replace
import math


def boosted_gains(gains, stalled_scans):
    values = tuple(float(value) for value in gains)
    if len(values) != 4 or not all(math.isfinite(v) and .5 <= v <= 1.5 for v in values):
        raise ValueError("Invalid normal wheel output gains")
    stage = min(2, max(0, int(stalled_scans)-2)//2)
    requested = (1.20, 1.35, 1.50)[stage] if stalled_scans >= 2 else 1.
    multiplier = min(requested, 1.5/max(values))
    return tuple(v*multiplier for v in values), multiplier


def send_boosted_turn(client, send, speed, ttl, deadline, stalled_scans, emit):
    if not 0 < abs(speed) <= 3500:
        raise ValueError("Turn speed outside existing firmware limits")
    normal = client.config
    gains, multiplier = boosted_gains(normal.wheel_output_gains, stalled_scans)
    client.config = replace(normal, wheel_output_gains=gains)
    try:
        result = send(0, speed, ttl, refresh_deadline_s=deadline)
        emit("turn_motor_output_boost", angular_mrad_s=speed, ttl_ms=ttl,
             wheel_output_gains=list(gains), output_multiplier=multiplier,
             straight_drive_unchanged=True)
        print("起转加力已下发：命令={:+d}mrad/s，轮输出倍率={:.2f}，脉冲={}ms；直行配置不变。".format(
            speed, multiplier, ttl), flush=True)
        return result
    finally:
        client.config = normal
