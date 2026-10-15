"""Closed-loop dead-zone stress test for the production speed controller.

Unlike the exploration-only simulator, this model feeds the motion produced
by every applied command back through ``AdaptiveSpeedController.observe``.
It distinguishes the higher torque needed to start a stationary chassis from
the lower torque needed to keep it moving, which is the failure mode seen on
the real four-wheel car.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math

from raspberry_pi.mapping import Pose2D, SlamUpdate
from raspberry_pi.mapping.adaptive_speed import (
    AdaptiveSpeedConfig,
    AdaptiveSpeedController,
)
from raspberry_pi.mapping.explorer import MotionCommand


@dataclass(frozen=True)
class DeadZoneChassisConfig:
    linear_start_mm_s: int = 440
    linear_hold_mm_s: int = 220
    angular_start_mrad_s: int = 2500
    angular_hold_mrad_s: int = 1600
    response: float = 0.65


class DeadZoneChassis:
    """Small deterministic chassis model with static/kinetic hysteresis."""

    def __init__(self, config=None):
        self.config = config or DeadZoneChassisConfig()
        self.pose = Pose2D()
        self.linear_speed_m_s = 0.0
        self.angular_speed_rad_s = 0.0
        self._linear_running = False
        self._angular_running = False

    @staticmethod
    def _direction(value):
        return 1.0 if value > 0 else -1.0 if value < 0 else 0.0

    def _axis_target(self, command, running, start, hold, scale):
        magnitude = abs(command)
        if magnitude == 0:
            return 0.0, False
        threshold = hold if running else start
        if magnitude < threshold:
            return 0.0, False
        return self._direction(command) * magnitude * scale, True

    def step(self, command, dt_s=0.10):
        target_linear, self._linear_running = self._axis_target(
            command.linear_mm_s,
            self._linear_running,
            self.config.linear_start_mm_s,
            self.config.linear_hold_mm_s,
            0.00032,
        )
        target_angular, self._angular_running = self._axis_target(
            command.angular_mrad_s,
            self._angular_running,
            self.config.angular_start_mrad_s,
            self.config.angular_hold_mrad_s,
            0.00030,
        )
        response = min(1.0, max(0.0, self.config.response))
        self.linear_speed_m_s += response * (
            target_linear - self.linear_speed_m_s
        )
        self.angular_speed_rad_s += response * (
            target_angular - self.angular_speed_rad_s
        )
        if not self._linear_running and abs(self.linear_speed_m_s) < 0.005:
            self.linear_speed_m_s = 0.0
        if not self._angular_running and abs(self.angular_speed_rad_s) < 0.01:
            self.angular_speed_rad_s = 0.0

        delta_yaw = self.angular_speed_rad_s * dt_s
        delta_distance = self.linear_speed_m_s * dt_s
        middle_yaw = self.pose.yaw_rad + delta_yaw * 0.5
        self.pose = Pose2D(
            self.pose.x_m + math.cos(middle_yaw) * delta_distance,
            self.pose.y_m + math.sin(middle_yaw) * delta_distance,
            (self.pose.yaw_rad + delta_yaw + math.pi) % math.tau - math.pi,
        )
        return SlamUpdate(
            self.pose,
            True,
            0.02,
            100,
            120,
            inlier_ratio=0.92,
            translation_m=abs(delta_distance),
            rotation_rad=delta_yaw,
            linear_speed_m_s=abs(self.linear_speed_m_s),
            angular_speed_rad_s=abs(self.angular_speed_rad_s),
        )


def _new_controller():
    return AdaptiveSpeedController(
        maximum_turn_speed_mrad_s=2600,
        maximum_linear_speed_mm_s=460,
        config=AdaptiveSpeedConfig(
            target_linear_speed_min_m_s=0.075,
            target_linear_speed_max_m_s=0.22,
            target_angular_speed_min_rad_s=0.12,
            target_angular_speed_max_rad_s=0.70,
            maximum_narrow_passage_linear_speed_mm_s=460,
            maximum_narrow_passage_turn_mrad_s=1700,
            maximum_narrow_passage_stall_turn_mrad_s=2600,
        ),
    )


def run_cycle(linear_start_mm_s=440, angular_start_mrad_s=2500,
              dt_s=0.10, maximum_steps=180):
    """Run a narrow avoidance turn followed by a straight restart."""
    controller = _new_controller()
    chassis = DeadZoneChassis(DeadZoneChassisConfig(
        linear_start_mm_s=int(linear_start_mm_s),
        angular_start_mrad_s=int(angular_start_mrad_s),
    ))
    turn_request = MotionCommand(
        0,
        1700,
        "narrow_passage_turning",
        "closed-loop avoidance test",
    )
    forward_request = MotionCommand(
        300,
        0,
        "narrow_passage_advancing",
        "closed-loop restart test",
    )
    phase = "turn"
    turn_yaw = 0.0
    forward_distance = 0.0
    first_turn_motion_step = None
    first_forward_motion_step = None
    applied_turns = []
    applied_linear = []

    for step in range(int(maximum_steps)):
        requested = turn_request if phase == "turn" else forward_request
        applied = controller.apply(requested)
        update = chassis.step(applied, dt_s)
        controller.observe(update)
        if phase == "turn":
            applied_turns.append(abs(applied.angular_mrad_s))
            turn_yaw += abs(update.rotation_rad)
            if (
                first_turn_motion_step is None
                and update.angular_speed_rad_s >= 0.12
            ):
                first_turn_motion_step = step
            if turn_yaw >= math.radians(35.0):
                phase = "forward"
                chassis.linear_speed_m_s = 0.0
                chassis._linear_running = False
                controller.stopped()
        else:
            applied_linear.append(abs(applied.linear_mm_s))
            forward_distance += abs(update.translation_m)
            if (
                first_forward_motion_step is None
                and update.linear_speed_m_s >= 0.075
            ):
                first_forward_motion_step = step
            if forward_distance >= 0.60:
                break

    high_turn_frames = sum(value > 2200 for value in applied_turns)
    return {
        "passed": (
            phase == "forward"
            and forward_distance >= 0.60
            and first_turn_motion_step is not None
            and first_forward_motion_step is not None
            and high_turn_frames <= 2
        ),
        "steps": step + 1,
        "simulated_s": round((step + 1) * dt_s, 3),
        "linear_start_mm_s": int(linear_start_mm_s),
        "angular_start_mrad_s": int(angular_start_mrad_s),
        "turn_start_s": (
            None if first_turn_motion_step is None
            else round(first_turn_motion_step * dt_s, 3)
        ),
        "forward_start_s": (
            None if first_forward_motion_step is None
            else round(first_forward_motion_step * dt_s, 3)
        ),
        "turn_yaw_deg": round(math.degrees(turn_yaw), 2),
        "forward_distance_m": round(forward_distance, 3),
        "maximum_turn_command_mrad_s": max(applied_turns, default=0),
        "high_turn_frames": high_turn_frames,
        "maximum_linear_command_mm_s": max(applied_linear, default=0),
    }


def run_matrix():
    cases = [
        run_cycle(linear_start, angular_start)
        for linear_start in (380, 420, 440, 460)
        for angular_start in (2100, 2300, 2500, 2600)
    ]
    return {
        "passed": all(case["passed"] for case in cases),
        "case_count": len(cases),
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description="闭环底盘死区压力仿真")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--linear-start", type=int, default=440)
    parser.add_argument("--angular-start", type=int, default=2500)
    args = parser.parse_args()
    result = (
        run_matrix()
        if args.matrix
        else run_cycle(args.linear_start, args.angular_start)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
