"""根据雷达测得的真实运动和匹配质量双向调节速度。"""

from dataclasses import dataclass
import math

from .explorer import MotionCommand


def _sign(value):
    return 0 if value == 0 else (1 if value > 0 else -1)


@dataclass(frozen=True)
class AdaptiveSpeedConfig:
    minimum_turn_floor_mrad_s: int = 1500
    # Normal turns honour --turn-speed. A higher floor is learned only as a
    # short static-friction kick after lidar confirms that the body is stuck.
    initial_turn_floor_mrad_s: int = 1500
    initial_turn_limit_mrad_s: int = 2200
    turn_adjust_step_mrad_s: int = 100
    maximum_turn_stall_step_multiplier: int = 2
    linear_adjust_step_mm_s: int = 40
    maximum_linear_stall_step_multiplier: int = 3
    # A 240 mm/s narrow-passage request must still be able to reach the
    # configured 460 mm/s chassis start ceiling after confirmed zero motion.
    # The boost is used only until lidar confirms movement; see
    # ``_linear_motion_confirmed`` in ``apply``.
    maximum_linear_boost_mm_s: int = 220
    reconnect_linear_boost_floor_mm_s: int = 120
    maximum_linear_reduction_mm_s: int = 80
    minimum_applied_linear_speed_mm_s: int = 220
    # Optional one-shot static-friction floor. Zero preserves legacy mapping
    # behaviour; object search configures the measured chassis start floor.
    linear_start_kick_mm_s: int = 0
    linear_progress_confirmation_frames: int = 1
    preserve_start_kick_on_poor_initial_match: bool = False
    maximum_steering_correction_mrad_s: int = 700
    maximum_cautious_turn_boost_mrad_s: int = 400
    maximum_narrow_passage_linear_speed_mm_s: int = 280
    maximum_narrow_passage_turn_mrad_s: int = 2000
    maximum_narrow_passage_stall_turn_mrad_s: int = 2600
    maximum_narrow_passage_steering_mrad_s: int = 450
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
        self._linear_motion_confirmed = False
        self._linear_progress_frames = 0
        self._turn_stall_events = 0
        self._narrow_turn_stall_boost_mrad_s = 0

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
                self._release_turn_start_boost()
                self._decrease(True, False)
            elif (
                angular_stalled
                or (
                    measured_angular_speed <= 0.0
                    and rotation_deg
                    < self.config.low_frame_rotation_deg
                )
            ):
                self._observe_stall(
                    turning=True,
                    narrow_passage=previous.state.startswith(
                        "narrow_passage_"
                    ),
                )
            else:
                # The extra output is a static-friction kick, not a new
                # cruising turn speed.  Once lidar confirms useful yaw,
                # immediately return to the configured 1500/1700 region so the
                # car cannot overshoot, reverse its correction, and remain in
                # a turn-only loop instead of advancing.
                self._release_turn_start_boost()
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
        angular_stalled = (
            measured_angular_speed > 0.0
            and measured_angular_speed
            < self.config.target_angular_speed_min_rad_s
        )
        # Treat the two axes of an avoidance/initial-scan arc independently.
        # The chassis may translate while static friction keeps yaw at zero.
        # Previously that case fell through to the generic success branch,
        # which reset the learned turn kick on every frame.  Keep learning
        # turn starting torque without increasing the already-moving linear
        # axis.  Gentle steering corrections during normal advance remain
        # excluded because their deliberately small yaw rate is not a stall.
        arc_turn_stalled = (
            turning
            and previous.state not in (
                "advancing",
                "narrow_passage_advancing",
            )
            and not quality_poor
            and (
                angular_stalled
                or (
                    measured_angular_speed <= 0.0
                    and rotation_deg
                    < self.config.low_frame_rotation_deg
                )
            )
        )
        if arc_turn_stalled:
            self._observe_stall(
                turning=True,
                narrow_passage=previous.state.startswith(
                    "narrow_passage_"
                ),
            )
            return
        # Startup scans and obstacle-avoidance arcs carry linear and angular
        # motion together.  Previously a completely stuck arc only learned a
        # positive *linear* correction. ``apply`` intentionally suppresses
        # that correction on tight arcs to preserve their checked radius, so
        # the exact same ineffective turn could be sent forever.  Raise the
        # turn floor instead when both measured components show no useful
        # chassis motion; this adds starting torque without changing the arc's
        # linear speed or swept-envelope assumption.
        linear_excess = (
            linear_too_fast
            or (
                measured_linear_speed <= 0.0
                and translation_m > self.config.high_frame_translation_m
            )
        )
        angular_excess = (
            turning
            and (
                (
                    measured_angular_speed > 0.0
                    and measured_angular_speed
                    > self.config.target_angular_speed_max_rad_s
                )
                or (
                    measured_angular_speed <= 0.0
                    and rotation_deg > self.config.high_frame_rotation_deg
                )
            )
        )
        if quality_poor:
            if (
                self.config.preserve_start_kick_on_poor_initial_match
                and moving_linear
                and not self._linear_motion_confirmed
            ):
                # A noisy millimetre-scale scan match is not evidence that the
                # starting kick was excessive. Reducing torque here changed a
                # 400 mm/s breakaway command into 230 mm/s while the motors
                # merely buzzed. Keep the kick and wait for credible motion.
                self._linear_progress_frames = 0
                self._stalled_frames = 0
                if turning:
                    self._release_turn_start_boost()
                return
            if turning:
                self._release_turn_start_boost()
            self._decrease(turning, True)
        elif linear_excess or angular_excess:
            # Linear and angular feedback are independent. The former
            # combined branch reduced straight-drive torque when only an
            # avoidance arc was rotating too fast, so the next straight
            # command fell back into the motor dead zone.
            if angular_excess:
                self._release_turn_start_boost()
            if moving_linear and translation_m >= self.config.low_frame_translation_m:
                # The kick has done its job. Even when the first movement is
                # faster than the cruise target, release the start floor and
                # let the normal reduction govern the following command.
                self._record_linear_progress()
            self._decrease(angular_excess, linear_excess)
        elif (
            turning
            and previous.state not in (
                "advancing",
                "narrow_passage_advancing",
            )
            and (
                measured_angular_speed
                >= self.config.target_angular_speed_min_rad_s
                or rotation_deg >= self.config.low_frame_rotation_deg
            )
        ):
            # A tight, clearance-checked arc may physically pivot without
            # translating because positive linear boost is intentionally not
            # applied to it. Useful yaw means the manoeuvre is progressing;
            # release its turn kick and do not poison the later straight
            # command with a futile linear-stall adjustment.
            self._release_turn_start_boost()
        elif (
            linear_stalled
            or (
                measured_linear_speed <= 0.0
                and translation_m
                < self.config.low_frame_translation_m
            )
        ):
            self._linear_motion_confirmed = False
            self._linear_progress_frames = 0
            self._observe_stall(turning=False)
        else:
            self._stalled_frames = 0
            self._linear_stall_events = 0
            if moving_linear:
                self._record_linear_progress()
            if turning:
                self._release_turn_start_boost()

    def apply(self, command):
        """应用学习到的死区补偿和当前安全上限。"""
        linear = command.linear_mm_s
        angular = command.angular_mrad_s
        narrow_passage = command.state.startswith("narrow_passage_")
        if linear != 0:
            requested_magnitude = abs(linear)
            previous_linear = (
                0 if self.last_command is None
                else int(self.last_command.linear_mm_s)
            )
            if (
                previous_linear == 0
                or _sign(previous_linear) != _sign(linear)
            ):
                self._linear_motion_confirmed = False
                self._linear_progress_frames = 0
            # turn_escape is a straight, envelope-checked command.  It must
            # retain the learned starting-torque boost; otherwise a chassis
            # whose static-friction threshold is above slow_speed alternates
            # forever between a cautious turn probe and a 220 mm/s command
            # that never moves the car.
            adjustment = self.linear_adjustment_mm_s
            if self._linear_motion_confirmed:
                # Retain the learned static-friction value for the next stop,
                # but do not turn it into a permanent cruising-speed boost.
                adjustment = min(0, adjustment)
            if angular != 0 and command.state not in (
                "advancing",
                "narrow_passage_advancing",
            ):
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
            if not self._linear_motion_confirmed and self.config.linear_start_kick_mm_s > 0:
                magnitude = max(
                    magnitude,
                    min(
                        self.maximum_linear_speed_mm_s,
                        self.config.linear_start_kick_mm_s,
                    ),
                )
            if narrow_passage:
                magnitude = min(
                    magnitude,
                    self.config.maximum_narrow_passage_linear_speed_mm_s,
                )
            linear = magnitude if linear > 0 else -magnitude
        if angular != 0:
            requested_magnitude = abs(angular)
            if command.state in (
                "advancing",
                "narrow_passage_advancing",
            ):
                # Steering while translating is a gentle path correction, not
                # a request to overcome the stationary-turn dead zone.
                magnitude = min(
                    requested_magnitude,
                    (
                        self.config.maximum_narrow_passage_steering_mrad_s
                        if narrow_passage
                        else self.config.maximum_steering_correction_mrad_s
                    ),
                    self.turn_limit_mrad_s,
                )
            elif command.state == "cautious_turn_probe":
                # Keep the collision-confirmation probe below the learned
                # ordinary turn floor, but do not leave it at 1500 forever
                # when the chassis has already demonstrated that this command
                # produces virtually zero yaw.  The bounded 400 mrad/s lift
                # reaches the proven 1900 starting region without using the
                # full escape/normal-turn torque next to a possible obstacle.
                magnitude = min(
                    self.turn_limit_mrad_s,
                    max(
                        requested_magnitude,
                        min(
                            self.turn_floor_mrad_s,
                            requested_magnitude
                            + self.config
                            .maximum_cautious_turn_boost_mrad_s,
                        ),
                    ),
                )
            else:
                magnitude = min(
                    self.turn_limit_mrad_s,
                    max(requested_magnitude, self.turn_floor_mrad_s),
                )
                if narrow_passage:
                    # ``maximum_narrow_passage_turn_mrad_s`` is the normal
                    # close-wall limit, not a permanent motor dead-zone cap.
                    # Only lidar-confirmed body stall opens bounded headroom;
                    # otherwise a configured 1700 command could never benefit
                    # from the learned 1900+ turn floor and the car would sit
                    # beside the obstacle indefinitely.
                    narrow_stall_limit = min(
                        self.maximum_turn_speed_mrad_s,
                        self.config
                        .maximum_narrow_passage_stall_turn_mrad_s,
                        self.config.maximum_narrow_passage_turn_mrad_s
                        + self._narrow_turn_stall_boost_mrad_s,
                    )
                    magnitude = min(
                        magnitude,
                        narrow_stall_limit,
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
        """Record a stop without erasing learned linear starting torque.

        Planned stationary mapping and BLE reconnects are not evidence that
        the learned linear boost was excessive. Turn boost is transient and
        must not survive a planned stop; otherwise the next avoidance action
        begins with the previous action's maximum kick.
        """
        self.last_command = MotionCommand(0, 0, "stopped", "safety stop")
        self.last_requested_command = self.last_command
        self._linear_motion_confirmed = False
        self._linear_progress_frames = 0
        self._release_turn_start_boost()

    def _record_linear_progress(self):
        """Require the configured credible-frame count before dropping the kick."""
        required = max(1, int(self.config.linear_progress_confirmation_frames))
        self._linear_progress_frames = min(
            required, self._linear_progress_frames + 1
        )
        self._linear_motion_confirmed = self._linear_progress_frames >= required

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

    def _observe_stall(self, turning, narrow_passage=False):
        self._stalled_frames += 1
        required_frames = self.config.stalled_frames_before_increase
        if turning:
            # A repeated avoidance failure must reach useful starting torque
            # within the short safe command window. Subsequent confirmed
            # stalls therefore need fewer frames, while every increase remains
            # bounded by the user-provided global maximum.
            required_frames = max(
                1,
                required_frames - self._turn_stall_events,
            )
        else:
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
            multiplier = min(
                self.config.maximum_turn_stall_step_multiplier,
                self._turn_stall_events + 1,
            )
            increase = self.config.turn_adjust_step_mrad_s * multiplier
            self.turn_floor_mrad_s = min(
                self.turn_limit_mrad_s,
                self.turn_floor_mrad_s + increase,
            )
            self.turn_limit_mrad_s = min(
                self.maximum_turn_speed_mrad_s,
                self.turn_limit_mrad_s + increase,
            )
            if narrow_passage:
                maximum_boost = max(
                    0,
                    min(
                        self.maximum_turn_speed_mrad_s,
                        self.config
                        .maximum_narrow_passage_stall_turn_mrad_s,
                    )
                    - self.config.maximum_narrow_passage_turn_mrad_s,
                )
                self._narrow_turn_stall_boost_mrad_s = min(
                    maximum_boost,
                    self._narrow_turn_stall_boost_mrad_s + increase,
                )
            self._turn_stall_events += 1
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

    def _release_turn_start_boost(self):
        """Drop transient starting torque after real yaw begins.

        Static friction can require a stronger command to start the chassis,
        while substantially less torque is needed to keep it moving. Keeping
        the learned floor after startup caused avoidance overshoot and stopped
        the planner from ever reaching its straight-ahead heading tolerance.
        """
        self.turn_limit_mrad_s = min(
            self.maximum_turn_speed_mrad_s,
            self.config.initial_turn_limit_mrad_s,
        )
        self.turn_floor_mrad_s = min(
            self.turn_limit_mrad_s,
            self.config.initial_turn_floor_mrad_s,
        )
        self.turn_ttl_ms = self.config.initial_turn_ttl_ms
        self._narrow_turn_stall_boost_mrad_s = 0
        self._turn_stall_events = 0
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
            self._narrow_turn_stall_boost_mrad_s = max(
                0,
                self._narrow_turn_stall_boost_mrad_s
                - self.config.turn_adjust_step_mrad_s,
            )
            self._turn_stall_events = 0
        if moving_linear:
            self.linear_adjustment_mm_s = max(
                -self.config.maximum_linear_reduction_mm_s,
                self.linear_adjustment_mm_s
                - self.config.linear_adjust_step_mm_s,
            )
            self._linear_stall_events = 0
        self._stalled_frames = 0
