"""根据雷达测得的真实运动和匹配质量双向调节速度。"""

from dataclasses import dataclass
import math

from .explorer import MotionCommand


@dataclass(frozen=True)
class AdaptiveSpeedConfig:
    minimum_turn_floor_mrad_s: int = 1500
    initial_turn_floor_mrad_s: int = 1900
    initial_turn_limit_mrad_s: int = 2200
    turn_adjust_step_mrad_s: int = 100
    linear_adjust_step_mm_s: int = 40
    maximum_linear_stall_step_multiplier: int = 3
    maximum_linear_boost_mm_s: int = 160
    reconnect_linear_boost_floor_mm_s: int = 120
    maximum_linear_reduction_mm_s: int = 80
    minimum_applied_linear_speed_mm_s: int = 220
    maximum_steering_correction_mrad_s: int = 700
    stalled_frames_before_increase: int = 3
    low_frame_translation_m: float = 0.008
    high_frame_translation_m: float = 0.075
    low_frame_rotation_deg: float = 1.5
    high_frame_rotation_deg: float = 10.0
    target_linear_speed_min_m_s: float = 0.075
    target_linear_speed_max_m_s: float = 0.22
    target_angular_speed_min_rad_s: float = 0.12
    target_angular_speed_max_rad_s: float = 0.85
    poor_rmse_m: float = 0.09
    poor_inlier_ratio: float = 0.65
    # BLE空闲往返可达约145ms；转向期间为计算和无线抖动保留足够余量。
    minimum_turn_ttl_ms: int = 450
    initial_turn_ttl_ms: int = 500
    maximum_turn_ttl_ms: int = 700
    turn_ttl_adjust_step_ms: int = 50


class AdaptiveSpeedController:
    """学习底盘静摩擦死区，并在运动过快或定位变差时回退。"""

    def __init__(
        self,
        maximum_turn_speed_mrad_s,
        maximum_linear_speed_mm_s=500,
        config=None,
    ):
        self.config = config or AdaptiveSpeedConfig()
        self.maximum_turn_speed_mrad_s = int(
            maximum_turn_speed_mrad_s
        )
        self.maximum_linear_speed_mm_s = int(
            maximum_linear_speed_mm_s
        )
        self.turn_limit_mrad_s = min(
            self.maximum_turn_speed_mrad_s,
            self.config.initial_turn_limit_mrad_s,
        )
        self.turn_floor_mrad_s = min(
            self.turn_limit_mrad_s,
            self.config.initial_turn_floor_mrad_s,
        )
        self.linear_adjustment_mm_s = 0
        self.turn_ttl_ms = self.config.initial_turn_ttl_ms
        self.last_command = None
        self.last_requested_command = None
        self._stalled_frames = 0
        self._linear_stall_events = 0

    def observe(self, update):
        """使用上一条实际命令产生的运动结果调节下一条命令。"""
        previous = self.last_command
        if previous is None:
            return

        moving_linear = previous.linear_mm_s != 0
        turning = previous.angular_mrad_s != 0
        pure_turn = (
            previous.linear_mm_s == 0
            and turning
        )
        if not moving_linear and not pure_turn:
            self._stalled_frames = 0
            return

        if not update.accepted:
            if update.rejection_reason != "too_few_filtered_points":
                self._decrease(turning, moving_linear)
            return

        quality_poor = (
            update.rmse_m > self.config.poor_rmse_m
            or update.inlier_ratio < self.config.poor_inlier_ratio
        )
        if pure_turn:
            rotation_deg = math.degrees(abs(update.rotation_rad))
            measured_angular_speed = abs(
                getattr(update, "angular_speed_rad_s", 0.0)
            )
            angular_too_fast = (
                measured_angular_speed > 0.0
                and measured_angular_speed
                > self.config.target_angular_speed_max_rad_s
            )
            angular_stalled = (
                measured_angular_speed > 0.0
                and measured_angular_speed
                < self.config.target_angular_speed_min_rad_s
            )
            if (
                quality_poor
                or angular_too_fast
                or (
                    measured_angular_speed <= 0.0
                    and rotation_deg
                    > self.config.high_frame_rotation_deg
                )
            ):
                self._decrease(True, False)
            elif (
                angular_stalled
                or (
                    measured_angular_speed <= 0.0
                    and rotation_deg
                    < self.config.low_frame_rotation_deg
                )
            ):
                self._observe_stall(turning=True)
            else:
                self._stalled_frames = 0
            return

        translation_m = abs(update.translation_m)
        rotation_deg = math.degrees(abs(update.rotation_rad))
        measured_linear_speed = abs(
            getattr(update, "linear_speed_m_s", 0.0)
        )
        measured_angular_speed = abs(
            getattr(update, "angular_speed_rad_s", 0.0)
        )
        linear_too_fast = (
            measured_linear_speed > 0.0
            and measured_linear_speed
            > self.config.target_linear_speed_max_m_s
        )
        linear_stalled = (
            measured_linear_speed > 0.0
            and measured_linear_speed
            < self.config.target_linear_speed_min_m_s
        )
        if (
            quality_poor
            or linear_too_fast
            or (
                measured_linear_speed <= 0.0
                and translation_m
                > self.config.high_frame_translation_m
            )
            or (
                turning
                and (
                    (
                        measured_angular_speed > 0.0
                        and measured_angular_speed
                        > self.config.target_angular_speed_max_rad_s
                    )
                    or (
                        measured_angular_speed <= 0.0
                        and rotation_deg
                        > self.config.high_frame_rotation_deg
                    )
                )
            )
        ):
            self._decrease(turning, True)
        elif (
            linear_stalled
            or (
                measured_linear_speed <= 0.0
                and translation_m
                < self.config.low_frame_translation_m
            )
        ):
            self._observe_stall(turning=False)
        else:
            self._stalled_frames = 0
            self._linear_stall_events = 0

    def apply(self, command):
        """应用学习到的死区补偿和当前安全上限。"""
        linear = command.linear_mm_s
        angular = command.angular_mrad_s
        if linear != 0:
            requested_magnitude = abs(linear)
            adjustment = (
                0
                if command.state == "turn_escape"
                else self.linear_adjustment_mm_s
            )
            if angular != 0 and command.state != "advancing":
                # The explorer has already checked the swept envelope for
                # this exact short-arc speed. Increasing it changes the turn
                # radius and can drive a corner of the vehicle into the very
                # obstacle the arc was meant to avoid. A learned reduction is
                # still allowed; a boost is not.
                adjustment = min(0, adjustment)
            magnitude = min(
                self.maximum_linear_speed_mm_s,
                max(
                    self.config.minimum_applied_linear_speed_mm_s,
                    requested_magnitude + adjustment,
                ),
            )
            linear = magnitude if linear > 0 else -magnitude
        if angular != 0:
            requested_magnitude = abs(angular)
            if command.state == "advancing":
                # Steering while translating is a gentle path correction, not
                # a request to overcome the stationary-turn dead zone.
                magnitude = min(
                    requested_magnitude,
                    self.config.maximum_steering_correction_mrad_s,
                    self.turn_limit_mrad_s,
                )
            elif command.state == "cautious_turn_probe":
                magnitude = min(
                    self.turn_limit_mrad_s,
                    requested_magnitude,
                )
            else:
                magnitude = min(
                    self.turn_limit_mrad_s,
                    max(requested_magnitude, self.turn_floor_mrad_s),
                )
            angular = magnitude if angular > 0 else -magnitude

        applied = MotionCommand(
            int(linear),
            int(angular),
            command.state,
            command.reason,
            command.target_xy_m,
            command.finished,
        )
        self.last_requested_command = command
        self.last_command = applied
        return applied

    def ttl_for(self, command, normal_ttl_ms):
        if command.angular_mrad_s != 0:
            return min(int(normal_ttl_ms), self.turn_ttl_ms)
        return int(normal_ttl_ms)

    def stopped(self):
        """Record a stop without erasing learned starting torque.

        Planned stationary mapping and BLE reconnects are not evidence that
        the learned boost was excessive. Actual excessive motion is handled
        by ``observe`` and ``_decrease`` before this method is called.
        """
        self.last_command = MotionCommand(0, 0, "stopped", "safety stop")
        self.last_requested_command = self.last_command
        self._stalled_frames = 0

    def note_link_interruption(self):
        """Prepare a stronger restart after BLE interrupted linear motion.

        A disconnect does not prove that the chassis is physically stalled,
        but it does cut the usable acceleration window short.  Keep the
        learned correction and raise it to a bounded restart floor so the
        first command after reconnect can cross the static-friction dead
        zone instead of repeating the same weak ramp.
        """
        previous = self.last_command
        if previous is None or previous.linear_mm_s == 0:
            self.stopped()
            return False
        reconnect_floor = min(
            self.config.maximum_linear_boost_mm_s,
            self.config.reconnect_linear_boost_floor_mm_s
            + max(0, self._linear_stall_events - 1)
            * self.config.linear_adjust_step_mm_s,
        )
        self.linear_adjustment_mm_s = max(
            self.linear_adjustment_mm_s,
            reconnect_floor,
        )
        # A subsequent confirmed stall should use the fastest one-frame
        # increase path rather than waiting through the initial three frames.
        self._linear_stall_events = max(
            self._linear_stall_events,
            2,
        )
        self.stopped()
        return True

    def _observe_stall(self, turning):
        self._stalled_frames += 1
        required_frames = self.config.stalled_frames_before_increase
        if not turning:
            # React sooner after each confirmed failure so even a short BLE
            # healthy window can reach the learned starting torque.
            required_frames = max(
                1,
                required_frames - self._linear_stall_events,
            )
        if (
            self._stalled_frames
            < required_frames
        ):
            return
        if turning:
            self.turn_floor_mrad_s = min(
                self.turn_limit_mrad_s,
                self.turn_floor_mrad_s
                + self.config.turn_adjust_step_mrad_s,
            )
            self.turn_limit_mrad_s = min(
                self.maximum_turn_speed_mrad_s,
                self.turn_limit_mrad_s
                + self.config.turn_adjust_step_mrad_s,
            )
            self.turn_ttl_ms = min(
                self.config.maximum_turn_ttl_ms,
                self.turn_ttl_ms
                + self.config.turn_ttl_adjust_step_ms,
            )
        else:
            multiplier = min(
                self.config.maximum_linear_stall_step_multiplier,
                self._linear_stall_events + 1,
            )
            self.linear_adjustment_mm_s = min(
                self.config.maximum_linear_boost_mm_s,
                self.linear_adjustment_mm_s
                + self.config.linear_adjust_step_mm_s * multiplier,
            )
            self._linear_stall_events += 1
        self._stalled_frames = 0

    def _decrease(self, turning, moving_linear):
        if turning:
            self.turn_limit_mrad_s = max(
                self.config.minimum_turn_floor_mrad_s,
                self.turn_limit_mrad_s
                - self.config.turn_adjust_step_mrad_s,
            )
            self.turn_floor_mrad_s = max(
                self.config.minimum_turn_floor_mrad_s,
                min(
                    self.turn_limit_mrad_s,
                    self.turn_floor_mrad_s
                    - self.config.turn_adjust_step_mrad_s,
                ),
            )
            self.turn_ttl_ms = max(
                self.config.minimum_turn_ttl_ms,
                self.turn_ttl_ms
                - self.config.turn_ttl_adjust_step_ms,
            )
        if moving_linear:
            self.linear_adjustment_mm_s = max(
                -self.config.maximum_linear_reduction_mm_s,
                self.linear_adjustment_mm_s
                - self.config.linear_adjust_step_mm_s,
            )
            self._linear_stall_events = 0
        self._stalled_frames = 0
