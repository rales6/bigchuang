"""六自由度机械臂命令与状态管理。"""

from core.timebase import ticks_diff, ticks_ms
from drivers.arm_kinematics import ArmKinematics
from protocol.messages import encode_arm_joints


class ArmController:
    def __init__(self, bus, cfg):
        self.bus = bus
        self.cfg = cfg
        self.targets = list(cfg.ARM_HOME_US)
        self.positions = list(cfg.ARM_HOME_US)
        self._starts = list(cfg.ARM_HOME_US)
        self._started_ms = [ticks_ms()] * 6
        self._durations_ms = [0] * 6
        self.moving = False
        self.kinematics = ArmKinematics()
        self.reach_mm = getattr(cfg, "ARM_INITIAL_REACH_MM", 250)
        self.height_mm = getattr(cfg, "ARM_INITIAL_HEIGHT_MM", 80)
        self.yaw_deg = getattr(cfg, "ARM_INITIAL_YAW_DEG", 0)

    def set_joints(self, joint_commands):
        """joint_commands: [(id, pulse_us, duration_ms), ...]。"""
        if not joint_commands:
            raise ValueError("at least one joint is required")
        if len(joint_commands) > 6:
            raise ValueError("at most six joints are allowed")

        now = ticks_ms()
        self.update(now)
        # 只有旧动作确实仍在运行时才发送 #255PDST! 抢占。位置指令正常完成后，
        # 舵机会自行保持目标位置；此时重复发送全局停止可能让部分旧底板进入
        # 暂停状态，影响紧随其后的多关节组合动作。
        must_preempt = self.moving
        normalized = []
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
            normalized.append((joint_id, pulse_us, duration_ms))

        # 消息格式允许每个关节有独立时间；底板必须同步启动同一帧内的关节。
        payload = bytearray((len(normalized),))
        for joint_id, pulse_us, duration_ms in normalized:
            encoded = encode_arm_joints([(joint_id, pulse_us)], duration_ms)
            payload.extend(encoded[1:])
        if must_preempt:
            self.bus.preempt_arm(bytes(payload))
        else:
            self.bus.send_arm(bytes(payload))
        self.moving = True

    def home(self, duration_ms=None):
        duration = duration_ms or self.cfg.ARM_DEFAULT_DURATION_MS
        self.reach_mm = getattr(self.cfg, "ARM_INITIAL_REACH_MM", 250)
        self.height_mm = getattr(self.cfg, "ARM_INITIAL_HEIGHT_MM", 80)
        self.yaw_deg = getattr(self.cfg, "ARM_INITIAL_YAW_DEG", 0)
        self.set_joints([
            (joint_id, position, duration)
            for joint_id, position in enumerate(self.cfg.ARM_HOME_US)
        ])

    def move_cartesian_delta(self, forward_mm=0, up_mm=0, left_deg=0,
                             claw_open=None, duration_ms=None):
        if getattr(self.cfg, "ARM_ASCII_CONTROL_MODE", "calibrated_delta") == "calibrated_delta":
            return self._move_calibrated_delta(
                forward_mm, up_mm, left_deg, claw_open, duration_ms
            )
        duration = duration_ms or self._cartesian_duration(
            forward_mm, up_mm, left_deg
        )
        reach = _clamp(
            self.reach_mm + int(forward_mm),
            getattr(self.cfg, "ARM_MIN_REACH_MM", 120),
            getattr(self.cfg, "ARM_MAX_REACH_MM", 330),
        )
        # The original arm firmware defines "up" as servo1/3 increasing and
        # servo2 decreasing.  In this IK coordinate system that corresponds to
        # a smaller z target on the real robot, so the sign is configurable.
        height_delta = int(up_mm) * getattr(self.cfg, "ARM_UP_AXIS_SIGN", -1)
        height = _clamp(
            self.height_mm + height_delta,
            getattr(self.cfg, "ARM_MIN_HEIGHT_MM", 25),
            getattr(self.cfg, "ARM_MAX_HEIGHT_MM", 180),
        )
        yaw = _clamp(
            self.yaw_deg + int(left_deg),
            -getattr(self.cfg, "ARM_MAX_YAW_DEG", 80),
            getattr(self.cfg, "ARM_MAX_YAW_DEG", 80),
        )
        joints = self.kinematics.from_reach_yaw(reach, yaw, height, duration)
        if joints is None:
            raise ValueError("arm target is outside kinematic workspace")
        if claw_open is not None:
            joints.append((5, self._claw_pulse(claw_open), duration))
        self.set_joints(joints)
        self.reach_mm = reach
        self.height_mm = height
        self.yaw_deg = yaw
        return duration

    def _move_calibrated_delta(self, forward_mm=0, up_mm=0, left_deg=0,
                               claw_open=None, duration_ms=None):
        duration = duration_ms or self._cartesian_duration(
            forward_mm, up_mm, left_deg
        )
        self.update()
        pulses = list(self.positions)
        commands = {}

        if int(left_deg):
            pulses[0] += int(left_deg) * getattr(self.cfg, "ARM_YAW_US_PER_DEG", 10)
            commands[0] = pulses[0]

        if int(forward_mm):
            scale = int(forward_mm)
            deltas = getattr(self.cfg, "ARM_FORWARD_JOINT_US_PER_MM", (-2, 3, -2))
            for offset, joint_id in enumerate((1, 2, 3)):
                pulses[joint_id] += scale * deltas[offset]
                commands[joint_id] = pulses[joint_id]

        if int(up_mm):
            # Old manual control lifted the arm by moving joint1/2 down in pulse
            # and joint3 up in pulse.  Keep that physical calibration here.
            scale = int(up_mm)
            deltas = getattr(self.cfg, "ARM_UP_JOINT_US_PER_MM", (-4, -10, 8))
            for offset, joint_id in enumerate((1, 2, 3)):
                pulses[joint_id] += scale * deltas[offset]
                commands[joint_id] = pulses[joint_id]

        if claw_open is not None:
            commands[5] = self._claw_pulse(claw_open)

        if not commands:
            return 0

        joint_commands = []
        for joint_id in sorted(commands):
            minimum, maximum = self.cfg.ARM_LIMITS_US[joint_id]
            pulse = int(_clamp(commands[joint_id], minimum, maximum))
            joint_commands.append((joint_id, pulse, duration))
        self.set_joints(joint_commands)
        return duration

    def _cartesian_duration(self, forward_mm, up_mm, left_deg):
        linear = max(abs(int(forward_mm)), abs(int(up_mm)))
        angular_equiv = abs(int(left_deg)) * 3
        distance = max(linear, angular_equiv)
        return int(_clamp(700 + distance * 12, 900, 3200))

    def _claw_pulse(self, open_value):
        open_value = _clamp(
            int(open_value), 0, getattr(self.cfg, "ARM_CLAW_OPEN_VALUE_MAX", 100)
        )
        open_us = getattr(self.cfg, "ARM_CLAW_OPEN_US", 1200)
        closed_us = getattr(self.cfg, "ARM_CLAW_CLOSED_US", 1600)
        span = closed_us - open_us
        pulse = closed_us - span * open_value / getattr(
            self.cfg, "ARM_CLAW_OPEN_VALUE_MAX", 100
        )
        return int(_clamp(pulse, self.cfg.ARM_LIMITS_US[5][0],
                          self.cfg.ARM_LIMITS_US[5][1]))

    def stop(self):
        self.update()
        self.bus.stop(mask=0x02)
        for joint_id in range(6):
            self.targets[joint_id] = self.positions[joint_id]
            self._durations_ms[joint_id] = 0
        self.moving = False

    def update(self, now=None):
        """无位置反馈时，根据已发送的运动时间估算当前指令位置。"""
        now = ticks_ms() if now is None else now
        any_moving = False
        for joint_id in range(6):
            duration = self._durations_ms[joint_id]
            if duration <= 0:
                continue
            elapsed = max(0, ticks_diff(now, self._started_ms[joint_id]))
            if elapsed >= duration:
                self.positions[joint_id] = self.targets[joint_id]
                self._durations_ms[joint_id] = 0
            else:
                start = self._starts[joint_id]
                target = self.targets[joint_id]
                self.positions[joint_id] = int(
                    start + (target - start) * elapsed / duration
                )
                any_moving = True
        self.moving = any_moving

    def set_feedback(self, positions):
        self.positions = list(positions)
        self._durations_ms = [0] * 6
        self.moving = any(
            abs(self.targets[index] - self.positions[index]) > 8
            for index in range(6)
        )


def _clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)
