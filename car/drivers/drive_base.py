"""四轮差速底盘的平滑速度控制。

树莓派下发线速度和角速度，ESP32 完成差速运动学、曲率保持限幅、加减速斜坡
以及可选的轮速反馈 PI 修正。这样网络/模型命令的速度跳变不会直接冲击电机。
"""

from core.timebase import ticks_add, ticks_diff, ticks_ms


class DriveBase:
    def __init__(self, bus, cfg, calibration=None):
        if calibration is None:
            from drivers.drive_calibration import DriveCalibration
            calibration = DriveCalibration()
        self.bus = bus
        self.cfg = cfg
        self.calibration = calibration
        self.motor_output_gains = (1.0, 1.0, 1.0, 1.0)
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.command_deadline = ticks_ms()
        self.last_update = ticks_ms()
        self.last_feedback = None
        self.feedback = (0, 0, 0, 0)
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.outputs = (0, 0, 0, 0)
        self._last_sent_outputs = None
        self._last_output_sent_ms = self.last_update
        self.closed_loop = False

    def set_twist(
        self,
        linear_mm_s,
        angular_mrad_s,
        ttl_ms,
        now=None,
        motor_output_gains=None,
    ):
        now = ticks_ms() if now is None else now
        if abs(linear_mm_s) > self.cfg.MAX_LINEAR_MM_S:
            raise ValueError("linear speed exceeds limit")
        if abs(angular_mrad_s) > self.cfg.MAX_ANGULAR_MRAD_S:
            raise ValueError("angular speed exceeds limit")
        if not 50 <= ttl_ms <= self.cfg.COMMAND_MAX_TTL_MS:
            raise ValueError("ttl must be in range 50..{}".format(
                self.cfg.COMMAND_MAX_TTL_MS
            ))
        gains = (
            (1.0, 1.0, 1.0, 1.0)
            if motor_output_gains is None
            else tuple(float(value) for value in motor_output_gains)
        )
        if len(gains) != 4 or any(
            value < 0.50 or value > 1.50 for value in gains
        ):
            raise ValueError(
                "motor output gains must contain four values in 0.50..1.50"
            )
        self.motor_output_gains = gains
        self.target_linear = float(linear_mm_s)
        self.target_angular = float(angular_mrad_s)
        self.command_deadline = ticks_add(now, ttl_ms)

    def set_feedback(self, wheels, now=None):
        # 将各电机原始方向统一为“小车向前为正”，与目标轮速处于同一坐标系。
        self.feedback = tuple(
            wheels[index] * self.cfg.MOTOR_DIRECTIONS[index]
            for index in range(4)
        )
        self.last_feedback = ticks_ms() if now is None else now

    def update(self, now=None):
        now = ticks_ms() if now is None else now
        elapsed_ms = ticks_diff(now, self.last_update)
        if elapsed_ms < self.cfg.CONTROL_PERIOD_MS:
            return
        self.last_update = now
        dt = min(max(elapsed_ms / 1000.0, 0.001), 0.1)

        if ticks_diff(now, self.command_deadline) >= 0:
            self.target_linear = 0.0
            self.target_angular = 0.0

        linear_rate = (self.cfg.DECEL_MM_S2
                       if abs(self.target_linear) < abs(self.current_linear)
                       else self.cfg.ACCEL_MM_S2)
        self.current_linear = _approach(
            self.current_linear, self.target_linear, linear_rate * dt
        )
        self.current_angular = _approach(
            self.current_angular, self.target_angular,
            self.cfg.ANGULAR_ACCEL_MRAD_S2 * dt,
        )

        left_mm_s, right_mm_s = self._wheel_targets()
        left_output, right_output = self._speed_outputs(left_mm_s, right_mm_s, dt, now)
        self.outputs = self._motor_outputs(left_output, right_output)
        # 静止时不重复刷四个 P1500 帧；运动中仅在输出变化或保活周期到达时发送。
        if (self.outputs != self._last_sent_outputs or
                (self.moving and ticks_diff(now, self._last_output_sent_ms) >=
                 self.cfg.MOTOR_REFRESH_MS)):
            self.bus.send_drive(self.outputs)
            self._last_sent_outputs = self.outputs
            self._last_output_sent_ms = now

    def stop(self, emergency=False):
        self.target_linear = 0.0
        self.target_angular = 0.0
        self.command_deadline = ticks_ms()
        if emergency:
            self.current_linear = 0.0
            self.current_angular = 0.0
            self.outputs = (0, 0, 0, 0)
            self._last_sent_outputs = self.outputs
            self.integral_left = 0.0
            self.integral_right = 0.0
            self.bus.stop(mask=0x01)

    @property
    def moving(self):
        return any(abs(value) > 1 for value in self.outputs)

    def _wheel_targets(self):
        # omega 使用 mrad/s；转换为 rad/s 后套用差速底盘运动学。
        # 四轮底盘原地转向时需要额外克服轮胎侧向摩擦；使用独立增益，
        # 避免改变已经验证过的直行速度标定。
        angular_gain = getattr(self.cfg, "ANGULAR_OUTPUT_GAIN", 1.0)
        turn_mm_s = (
            (self.current_angular / 1000.0)
            * self.cfg.TRACK_WIDTH_MM
            / 2.0
            * angular_gain
        )
        left = self.current_linear - turn_mm_s
        right = self.current_linear + turn_mm_s
        peak = max(abs(left), abs(right))
        if peak > self.cfg.MAX_WHEEL_MM_S:
            scale = self.cfg.MAX_WHEEL_MM_S / peak
            left *= scale
            right *= scale
        return left, right

    def _speed_outputs(self, left_target, right_target, dt, now):
        feedback_fresh = (
            self.last_feedback is not None and
            ticks_diff(now, self.last_feedback) <= self.cfg.FEEDBACK_STALE_MS
        )
        self.closed_loop = feedback_fresh
        average_speed = (abs(left_target) + abs(right_target)) / 2.0
        trim = self.calibration.trim_for_speed(average_speed)
        left = (
            left_target
            * self.cfg.MOTOR_UNITS_PER_MM_S
            * (1.0 - trim)
        )
        right = (
            right_target
            * self.cfg.MOTOR_UNITS_PER_MM_S
            * (1.0 + trim)
        )

        if feedback_fresh:
            measured_left = (self.feedback[0] + self.feedback[2]) / 2.0
            measured_right = (self.feedback[1] + self.feedback[3]) / 2.0
            error_left = left_target - measured_left
            error_right = right_target - measured_right
            limit = self.cfg.WHEEL_INTEGRAL_LIMIT
            self.integral_left = _clamp(
                self.integral_left + error_left * dt, -limit, limit
            )
            self.integral_right = _clamp(
                self.integral_right + error_right * dt, -limit, limit
            )
            left += self.cfg.WHEEL_KP * error_left + self.cfg.WHEEL_KI * self.integral_left
            right += self.cfg.WHEEL_KP * error_right + self.cfg.WHEEL_KI * self.integral_right
        else:
            self.integral_left = 0.0
            self.integral_right = 0.0

        left = self._effective_output(left, left_target)
        right = self._effective_output(right, right_target)
        return left, right

    def _motor_outputs(self, left_output, right_output):
        """Apply independent traction balance after left/right calibration."""
        base_outputs = (
            left_output,
            right_output,
            left_output,
            right_output,
        )
        directions = self.cfg.MOTOR_DIRECTIONS
        outputs = []
        for index in range(4):
            value = (
                base_outputs[index]
                * self.motor_output_gains[index]
            )
            if 0 < abs(value) < self.cfg.MIN_EFFECTIVE_MOTOR_UNITS:
                value = (
                    self.cfg.MIN_EFFECTIVE_MOTOR_UNITS
                    if value > 0
                    else -self.cfg.MIN_EFFECTIVE_MOTOR_UNITS
                )
            value = _clamp(
                value,
                -self.cfg.MAX_MOTOR_UNITS,
                self.cfg.MAX_MOTOR_UNITS,
            )
            outputs.append(int(round(value)) * directions[index])
        return tuple(outputs)

    def set_calibration(self, trim_intercept, trim_slope_per_mm_s):
        if self.moving:
            raise ValueError("drive must be stopped before calibration update")
        self.calibration.set(trim_intercept, trim_slope_per_mm_s)

    def reset_calibration(self):
        if self.moving:
            raise ValueError("drive must be stopped before calibration reset")
        self.calibration.reset()

    def calibration_values(self):
        return self.calibration.values()

    def _effective_output(self, output, target):
        if abs(target) < 0.5:
            return 0
        minimum = self.cfg.MIN_EFFECTIVE_MOTOR_UNITS
        if 0 < abs(output) < minimum:
            output = minimum if output > 0 else -minimum
        return int(_clamp(output, -self.cfg.MAX_MOTOR_UNITS, self.cfg.MAX_MOTOR_UNITS))


def _approach(current, target, step):
    if current < target:
        return min(current + step, target)
    if current > target:
        return max(current - step, target)
    return target


def _clamp(value, minimum, maximum):
    return min(max(value, minimum), maximum)
