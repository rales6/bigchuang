"""ESP32 六路 PWM 机械臂驱动。

舵机直接连接 GPIO，不经过电机 UART 底板。关节目标在主循环中线性插值，避免
Timer 中断和阻塞 sleep；新目标从当前插值位置开始，可在运动中途平滑抢占。
"""

from core.timebase import ticks_diff, ticks_ms


class PWMArmController:
    def __init__(self, pwm_channels, cfg):
        if len(pwm_channels) != 6:
            raise ValueError("six PWM channels are required")
        self.channels = tuple(pwm_channels)
        self.cfg = cfg
        self.targets = list(cfg.ARM_HOME_US)
        self.positions = list(cfg.ARM_HOME_US)
        self._starts = list(cfg.ARM_HOME_US)
        self._started_ms = [ticks_ms()] * 6
        self._durations_ms = [0] * 6
        self.moving = False
        self.write_count = 0
        for joint_id, position in enumerate(self.positions):
            self._write_us(joint_id, position)

    def set_joints(self, joint_commands):
        if not joint_commands or len(joint_commands) > 6:
            raise ValueError("joint count must be in range 1..6")
        now = ticks_ms()
        self.update(now)
        seen = set()
        for joint_id, pulse_us, duration_ms in joint_commands:
            if joint_id in seen or not 0 <= joint_id < 6:
                raise ValueError("invalid or duplicate joint id")
            minimum, maximum = self.cfg.ARM_LIMITS_US[joint_id]
            if not minimum <= pulse_us <= maximum:
                raise ValueError("joint {} target exceeds limit".format(joint_id))
            if not 20 <= duration_ms <= 10000:
                raise ValueError("arm duration must be in range 20..10000")
            seen.add(joint_id)
            self._starts[joint_id] = self.positions[joint_id]
            self.targets[joint_id] = pulse_us
            self._started_ms[joint_id] = now
            self._durations_ms[joint_id] = duration_ms
        self.moving = True

    def update(self, now=None):
        now = ticks_ms() if now is None else now
        any_moving = False
        for joint_id in range(6):
            duration = self._durations_ms[joint_id]
            if duration <= 0:
                continue
            elapsed = max(0, ticks_diff(now, self._started_ms[joint_id]))
            if elapsed >= duration:
                position = self.targets[joint_id]
                self._durations_ms[joint_id] = 0
            else:
                start = self._starts[joint_id]
                target = self.targets[joint_id]
                position = int(start + (target - start) * elapsed / duration)
                any_moving = True
            if position != self.positions[joint_id]:
                self.positions[joint_id] = position
                self._write_us(joint_id, position)
        self.moving = any_moving

    def home(self, duration_ms=None):
        duration = duration_ms or self.cfg.ARM_DEFAULT_DURATION_MS
        self.set_joints([
            (joint_id, position, duration)
            for joint_id, position in enumerate(self.cfg.ARM_HOME_US)
        ])

    def stop(self, now=None):
        now = ticks_ms() if now is None else now
        self.update(now)
        for joint_id in range(6):
            self.targets[joint_id] = self.positions[joint_id]
            self._durations_ms[joint_id] = 0
        self.moving = False

    def set_feedback(self, positions):
        # PWM 舵机没有位置回传；状态中的 positions 表示当前输出指令位置。
        pass

    def deinit(self):
        for channel in self.channels:
            if hasattr(channel, "deinit"):
                channel.deinit()

    def _write_us(self, joint_id, pulse_us):
        channel = self.channels[joint_id]
        if hasattr(channel, "duty_ns"):
            channel.duty_ns(int(pulse_us) * 1000)
        elif hasattr(channel, "duty_u16"):
            channel.duty_u16(int(pulse_us * 65535 // 20000))
        else:
            # 兼容旧 ESP32 MicroPython 的 10-bit duty API。
            channel.duty(int(pulse_us * 1024 // 20000))
        self.write_count += 1


def create_pwm_arm(cfg):
    from machine import PWM, Pin
    channels = []
    for pin_number in cfg.ARM_PWM_PINS:
        pin = Pin(pin_number, Pin.OUT)
        try:
            channel = PWM(pin, freq=50)
        except TypeError:
            channel = PWM(pin)
            channel.freq(50)
        channels.append(channel)
    return PWMArmController(channels, cfg)
