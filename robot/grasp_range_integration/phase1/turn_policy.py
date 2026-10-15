"""Bounded, convergent phase-one turns; protected control stays intact."""

from __future__ import annotations

from dataclasses import replace
import math
import time

LAST_REQUEST = (0, 0)
TELEMETRY_CALLBACK = None
TURN_START_REQUEST = None


def emit(kind, **fields):
    if TELEMETRY_CALLBACK is not None:
        TELEMETRY_CALLBACK(kind, **fields)


def bounded_turn_ttl_ms(error_rad: float, angular_mrad_s: int, original_ttl_ms: int,
                        *, stalled_scans: int = 0, measured_rate_rad_s: float = 0.0,
                        breakaway: bool = False, calibrated_start: bool = False) -> int:
    """Escalate large-angle start pulses, never relax the angle/firmware bounds.

    Two reliable stalled observations per stage permit 140 -> 250 -> 350 ->
    450 -> 500 ms. Nominal/fast measured rate bounds the possible sweep; no
    unmeasured slow-rate assumption can authorize a long small-angle pulse.
    """
    rate = max(0.1, abs(angular_mrad_s) / 1000, measured_rate_rad_s * 1.25)
    error_deg = abs(math.degrees(error_rad))
    angle_budget = max(math.radians(1), abs(error_rad) - math.radians(2))
    cap = (140, 250, 350, 450, 500)[min(4, max(0, stalled_scans) // 2)]
    if error_deg <= 35:
        cap = 70
    if int(original_ttl_ms) < 50:
        raise ValueError("ESP32 twist TTL must be at least 50 ms")
    if breakaway and calibrated_start:
        from grasp_range_integration.phase1.turn_calibration import calibrated_start_ttl_ms
        calibrated = calibrated_start_ttl_ms(error_rad, angular_mrad_s, stalled_scans,
                                            original_ttl_ms, measured_rate_rad_s)
        if calibrated is not None:
            return calibrated
    if breakaway and stalled_scans >= 2 and error_deg > 10:
        # Independently confirmed non-motion may extend the start pulse, not
        # an already-moving precision turn. Commanded/fast measured rate is
        # still a hard bound: no invented motor start delay or low-rate guess.
        stages = (70, 100, 120, 140, 160) if error_deg <= 35 else (140, 250, 350, 450, 500)
        cap = stages[min(4, stalled_scans // 2)]
        angle_limit_ms = int(min(angle_budget * .9, math.radians(20)) / rate * 1000)
        if angle_limit_ms < 50:
            raise RuntimeError("起转命令/反馈速率与剩余角度预算不兼容：不足最小50ms；应切换精细加力，不扩大脉冲")
        return min(int(original_ttl_ms), max(50, min(cap, angle_limit_ms)))
    angle_limit_ms = int(angle_budget / rate * (900 if error_deg > 35 else 700))
    return min(int(original_ttl_ms), max(50, min(cap, angle_limit_ms)))


def breakaway_speed_mrad_s(error_rad: float, stalled_scans: int) -> int:
    stage = min(2, max(0, int(stalled_scans)-2)//2)
    # Existing firmware rejects >3500mrad/s; raise within its public limit.
    return int(math.copysign((3000, 3250, 3500)[stage], error_rad))


def start_recovery_eligible(key, error):
    """The same recovery classification governs budget AND stall timeout.

    Arrival heading is a real turn, not automatically a tiny camera trim.
    Match the directional calibrated-start boundary, not the old >10deg
    boundary (10.75deg/3000 requested only 45ms and crashed at firmware 50ms).
    """
    return bool(key and key[0] in {"waypoint", "view"} and error is not None
                and math.isfinite(error)
                and abs(error) >= math.radians(14 if error > 0 else 12))


class TurnConvergence:
    """Track angular convergence independently of waypoint distance."""
    def __init__(self, *, corrective_reversals=False):
        self.corrective_reversals = corrective_reversals
        self.nonshrinking_crossings = 0
        self.last_crossing_error = math.inf
        self.key = None
        self.previous = None
        self.initial = self.total = 0.0
        self.crossings = self.worsening = 0
        self.started = 0.0
        self.last_sample = None
        self.active_seconds = 0.0

    def check(self, key, error, *, now=None, active=True):
        now = time.monotonic() if now is None else now
        if key != self.key:
            self.key, self.previous = key, error
            self.initial, self.total = abs(error), 0.0
            self.crossings = self.worsening = 0
            self.nonshrinking_crossings = 0
            self.last_crossing_error = math.inf
            self.started = now
            self.last_sample, self.active_seconds = now, 0.0
            return False
        elapsed = max(0.0, now - self.last_sample)
        self.last_sample = now
        if not active:
            # Map fusion can move the requested heading while the motors are
            # held at zero. Rebase it, never call this executed rotation.
            self.previous = error
            self.worsening = 0
            emit("turn_convergence_paused", yaw_error_deg=round(math.degrees(error), 2),
                 active_seconds=round(self.active_seconds, 3))
            return False
        self.active_seconds += elapsed
        old = self.previous
        self.previous = error
        delta = math.atan2(math.sin(error - old), math.cos(error - old))
        self.total += abs(delta)
        crossed = old * error < 0 and abs(delta) > math.radians(2)
        self.crossings += int(crossed)
        if crossed:
            shrinking = abs(error) < self.last_crossing_error-math.radians(.5)
            self.nonshrinking_crossings = 0 if shrinking else self.nonshrinking_crossings+1
            self.last_crossing_error = abs(error)
        self.worsening = self.worsening + 1 if abs(error) > abs(old) + math.radians(2) else 0
        emit("turn_convergence", yaw_error_deg=round(math.degrees(error), 2),
             cumulative_rotation_deg=round(math.degrees(self.total), 2), crossings=self.crossings)
        oscillation = self.nonshrinking_crossings if self.corrective_reversals else self.crossings
        if (oscillation >= 3 or self.worsening >= 3
                or (not self.corrective_reversals and self.total > self.initial + math.radians(45))
                or self.active_seconds > (90 if self.corrective_reversals else 45)):
            raise RuntimeError("原地转向未收敛或反复超调；退出动作，请检查运动日志后重试")
        return crossed  # A zero-command frame brakes before reversing.


def install_turn_policy(mission, *, dynamic_feedback=False) -> None:
    global TURN_START_REQUEST
    TURN_START_REQUEST = None
    original_command = mission._strict_command_for_pose
    cls = mission.AdaptiveSpeedController
    original_ttl, original_observe, original_stopped = cls.ttl_for, cls.observe, cls.stopped
    original_apply = getattr(cls, "apply", None)
    latest_error = {"rad": None}
    monitor = TurnConvergence(corrective_reversals=dynamic_feedback)
    from grasp_range_integration.phase1.adaptive_turn import AdaptiveTurnResponse
    adaptive = AdaptiveTurnResponse(emit, clock=lambda: time.monotonic()) if dynamic_feedback else None
    evidence = {"active": False}
    from grasp_range_integration.phase1.motion_experience import AngularProgress
    progress = mission._phase1_angular_progress = AngularProgress()
    sent_pulse = {"value": None}
    mission._phase1_turn_pulse_sent = lambda fields: sent_pulse.update(value=dict(fields))

    def hold_turn():
        evidence["active"] = False
        if monitor.key is not None and latest_error["rad"] is not None:
            monitor.check(monitor.key, latest_error["rad"], active=False)

    mission._phase1_hold_turn_convergence = hold_turn

    def observe(self, update):
        previous = self.last_command
        original_observe(self, update)
        reliable = (bool(getattr(update, "accepted", False))
                    and math.isfinite(float(getattr(update, "rmse_m", math.inf)))
                    and float(update.rmse_m) <= 0.09
                    and float(getattr(update, "inlier_ratio", 0)) >= 0.65)
        rotation = float(getattr(update, "rotation_rad", 0))
        pulse, sent_pulse["value"] = sent_pulse["value"], None
        experience = getattr(mission, "_phase1_turn_experience", None)
        if experience is not None and previous is not None and previous.linear_mm_s == 0:
            experience.record(pulse, rotation, float(getattr(update, "translation_m", math.inf)), reliable)
        if previous is not None and previous.linear_mm_s == 0 and previous.angular_mrad_s != 0:
            progress.feedback(rotation, latest_error["rad"], reliable)
        # last_command is the controller's applied command, not the planner's
        # request (which the outer clearance guard may have replaced by zero).
        evidence["active"] = bool(previous is not None and previous.linear_mm_s == 0
                                  and previous.angular_mrad_s != 0 and reliable
                                  and abs(math.degrees(rotation)) >= .25)
        emit("measured_motion", reliable=reliable, rotation_deg=round(math.degrees(rotation), 3),
             translation_m=round(float(getattr(update, "translation_m", 0)), 4))
        if previous is not None and previous.linear_mm_s == 0 and previous.angular_mrad_s == 0:
            # A short clearance hold is not a new turn. Freeze, do not erase
            # two proven failed starts. It never counts as stall evidence.
            return
        if previous is None or previous.linear_mm_s != 0 or not reliable:
            self._phase1_turn_stalled_scans = 0
            self._phase1_turn_capped_stalls = 0
            return
        if adaptive is not None:
            actual_speed = (pulse or {}).get('angular_mrad_s', previous.angular_mrad_s)
            actual_ttl = (pulse or {}).get('ttl_ms', getattr(self, '_phase1_previous_ttl_ms', 70))
            adaptive.feedback(actual_speed, actual_ttl, rotation,
                              float(getattr(update,'translation_m',math.inf)),
                              reliable, latest_error['rad'])
            row = adaptive.directions[1 if actual_speed > 0 else -1]
            self._phase1_turn_stalled_scans = row.low_progress
            self._phase1_turn_capped_stalls = 0
            return  # Live mode never uses the legacy max-rate / 8-or-20 exit.
        stalled = int(getattr(self, "_phase1_turn_stalled_scans", 0))
        if abs(math.degrees(rotation)) < .25:
            self._phase1_turn_stalled_scans = stalled + 1
            last_ttl = getattr(self, "_phase1_previous_ttl_ms", 0)
            maximum = getattr(self, "_phase1_turn_max_ttl_ms", math.inf)
            at_cap = last_ttl >= maximum - 1
            self._phase1_turn_capped_stalls = (
                int(getattr(self, "_phase1_turn_capped_stalls", 0)) + 1 if at_cap else 0)
            emit("turn_stall_recovery", stalled_scans=self._phase1_turn_stalled_scans,
                 previous_ttl_ms=last_ttl, maximum_permitted_ttl_ms=maximum if math.isfinite(maximum) else None,
                 capped_failures=self._phase1_turn_capped_stalls, nonmotion_threshold_deg=.25)
            if (experience is not None and getattr(self, "_phase1_experience_reused", False)
                    and self._phase1_turn_stalled_scans >= 3):
                direction = "left" if previous.angular_mrad_s > 0 else "right"
                experience.disabled.add(direction)
                emit("turn_experience_disabled", direction=direction, reason="three_actual_stalled_attempts")
        else:
            self._phase1_turn_stalled_scans = 0
            self._phase1_turn_capped_stalls = 0
            ttl = max(0.05, getattr(self, "_phase1_previous_ttl_ms", 140) / 1000)
            # Retain the fastest reliable response seen in this run. A slow
            # high-friction patch must not erase evidence from a slick patch
            # and then authorize an excessive pulse on re-entering it.
            observed_rate = abs(rotation) / ttl
            self._phase1_measured_rate = max(float(getattr(self, "_phase1_measured_rate", 0)), observed_rate)
            emit("turn_breakaway_motion_detected", rotation_deg=math.degrees(rotation),
                 measured_rate_rad_s=self._phase1_measured_rate,
                 latest_measured_rate_rad_s=observed_rate,
                 next_mode="angle_bounded_precision", duration_escalation_reset=True)
        error = latest_error["rad"]
        fine_turn = error is None or abs(math.degrees(error)) <= 35
        start_recovery = start_recovery_eligible(monitor.key, error)
        precision_recovery = fine_turn and not start_recovery
        exhausted = int(getattr(self, "_phase1_turn_capped_stalls", 0)) >= 3
        # Two attempts at each output tier: unboosted, 1.20, 1.35, 1.50.
        # Previously six failures ended precision alignment BEFORE the final
        # output tier could ever be tried. No extra time/angle is authorized.
        if self._phase1_turn_stalled_scans >= (8 if precision_recovery else 20) and (fine_turn or exhausted):
            raise RuntimeError("原地转向恢复未成功：已达到剩余角度/命令允许的脉冲上限且重复尝试仍无可靠转动；已停车")

    def stopped(self):
        global TURN_START_REQUEST
        TURN_START_REQUEST = None
        sent_pulse["value"] = None
        hold_turn()
        self._phase1_turn_stalled_scans = 0
        self._phase1_turn_capped_stalls = 0
        return original_stopped(self)

    def command_for_pose(args, pose, waypoint, tolerance_m=None):
        global LAST_REQUEST
        from grasp_range_integration.phase1 import visibility_route_planner
        # A live target-bearing segment has priority over the old station's
        # requested yaw, even when its endpoint happens to coincide with it.
        view = {} if getattr(args,'_phase1_visual_guidance',False) else visibility_route_planner.ACTIVE_VIEW or {}
        stop_rule = str((view.get("visibility_zone") or {}).get("stop_rule", ""))
        station_shift = stop_rule == "locked_target_station_shift"
        # A 12 cm deliberate viewpoint move must not be swallowed by the
        # normal 18 cm route tolerance. Same guarded/adaptive drive executor.
        if station_shift:
            tolerance_m = .04
        command = original_command(args, pose, waypoint, tolerance_m)
        mission._phase1_arrival_distance_m = None
        linear, angular, reached = command
        xy, yaw = view.get("view_xy_m"), view.get("target_yaw_rad")
        if station_shift and linear > 0:
            mission._phase1_arrival_distance_m = math.hypot(waypoint[0]-pose.x_m, waypoint[1]-pose.y_m)
        recovery = stop_rule.startswith("single_camera") or stop_rule == "local_view_sweep"
        if recovery and isinstance(xy, (list, tuple)) and len(xy) >= 2:
            if math.hypot(pose.x_m - float(xy[0]), pose.y_m - float(xy[1])) > args.waypoint_tolerance:
                raise RuntimeError("原站位复查时车体移位超限；不自动前进追物，请复核定位和净空")
        at_station = (isinstance(xy, (list, tuple)) and len(xy) >= 2 and yaw is not None
                      and math.hypot(waypoint[0] - float(xy[0]), waypoint[1] - float(xy[1])) <= 0.03
                      and math.hypot(pose.x_m - float(xy[0]), pose.y_m - float(xy[1])) <=
                          (.04 if station_shift else args.waypoint_tolerance))
        # Even if A* produced tiny intermediate waypoints after map drift,
        # single-camera recovery may never issue a translation command.
        at_station = at_station or recovery and yaw is not None
        route = visibility_route_planner.LATEST_ROUTE or {}
        if at_station and not recovery and route.get("mode") == "semantic_visibility_regions":
            from grasp_range_integration.phase1.arrival_visibility import arrival_action
            action = arrival_action(route, [pose.x_m,pose.y_m,pose.yaw_rad], view,
                                    visibility_route_planner.task_region_view_assessment)
            if action["action"] == "diagnostic_capture":
                LAST_REQUEST = (0,0)
                latest_error["rad"] = None
                monitor.key = None
                emit("arrival_borderline_capture", task_region_view=action["assessment"],
                     task_success_allowed=False)
                return 0,0,True
            if action["action"] == "adjust_on_route":
                command = original_command(args, pose, waypoint, .04)
                linear, angular, reached = command
                at_station = False
                mission._phase1_arrival_distance_m = action["distance_m"] if linear > 0 else None
                emit("arrival_visibility_position_adjustment", remaining_m=action["distance_m"],
                     task_region_view=action["assessment"], waypoint_xy_m=list(waypoint),
                     object_distance_used=False, tighter_arrival_tolerance_m=.04)
        if at_station:
            error = math.atan2(math.sin(float(yaw) - pose.yaw_rad), math.cos(float(yaw) - pose.yaw_rad))
            tolerance = 2.0 if recovery else max(5.0, min(10.0, args.heading_tolerance_deg))
            # A single-direction semantic station requests a usable view,
            # not perfect yaw. Do not stall in an uncalibrated tiny trim if
            # the actual pose already meets range/FOV/coverage requirements.
            # Never skip a multi-direction sweep or single-camera alignment.
            if (not recovery and abs(error) <= math.radians(15)
                    and route.get("mode") == "semantic_visibility_regions"
                    and int((view.get("visibility_zone") or {}).get(
                        "sweep_count", view.get("sweep_count") or 1)) == 1):
                assessment = visibility_route_planner.task_region_view_assessment(
                    route, [pose.x_m, pose.y_m, pose.yaw_rad], view=view)
                if assessment.get("eligible"):
                    LAST_REQUEST = (0, 0)
                    latest_error["rad"] = None
                    monitor.key = None
                    emit("view_heading_accepted_by_coverage", yaw_error_deg=math.degrees(error),
                         task_region_view=assessment, camera_confirmation_required=True)
                    return 0, 0, True
            if abs(error) <= math.radians(tolerance):
                LAST_REQUEST = (0, 0)
                latest_error["rad"] = None
                monitor.key = None
                return 0, 0, True
            linear, angular, reached = 0, int(math.copysign(min(args.turn_speed, max(650, abs(error) * 1100)), error)), False
            key = ("view", round(float(xy[0]), 3), round(float(xy[1]), 3), round(float(yaw), 3))
        elif linear == 0 and angular != 0:
            bearing = math.atan2(waypoint[1] - pose.y_m, waypoint[0] - pose.x_m)
            error = math.atan2(math.sin(bearing - pose.yaw_rad), math.cos(bearing - pose.yaw_rad))
            key = ("waypoint", round(float(waypoint[0]), 3), round(float(waypoint[1]), 3))
        else:
            latest_error["rad"] = None
            monitor.key = None
            LAST_REQUEST = (linear, angular)
            return command
        latest_error["rad"] = error
        progress.heading(key, error)
        if monitor.key != key:
            if adaptive is not None:
                adaptive.correction_scale = 1.
                for response in adaptive.directions.values():
                    response.low_progress = 0
                    response.output_stage = 0
                    response.last_progress_s = time.monotonic()
            print("转向用途：{}；剩余角度={:.1f}°。".format(
                "目标区域附近调整观察方向" if key[0] == "view" else "对齐前往目标区域的行驶路线（不是在起点寻物）",
                math.degrees(error)), flush=True)
        active = evidence["active"]
        evidence["active"] = False  # Each feedback interval is consumed once.
        if monitor.check(key, error, active=active):
            if adaptive is not None:
                adaptive.reverse_correction()
            LAST_REQUEST = (0, 0)
            return 0, 0, False
        LAST_REQUEST = (linear, angular)
        return linear, angular, reached

    def apply(self, command):
        # A late freshness hold is not an executed 400mm/s start kick or a
        # stalled turn. Reset the applied-command model if the output is held.
        mission._phase1_motion_hold = self.stopped
        error = latest_error["rad"]
        if command.linear_mm_s == 0 and command.angular_mrad_s and error is not None:
            stalled = int(getattr(self, "_phase1_turn_stalled_scans", 0))
            edge_precision = (10 < abs(math.degrees(error)) and
                              not start_recovery_eligible(monitor.key,error))
            if abs(math.degrees(error)) <= 35 and (stalled < 2 or edge_precision):
                # Existing precision branch does not force every small turn
                # to the learned 2600 floor. Escalation needs fresh stall evidence.
                magnitude = min(1800, max(1200, abs(command.angular_mrad_s)))
                command = replace(command, angular_mrad_s=int(math.copysign(magnitude, command.angular_mrad_s)),
                                  state="precision_turning")
        applied = original_apply(self, command)
        if getattr(mission, "_phase1_navigation_estimate", None) and abs(applied.linear_mm_s) > 400:
            applied = replace(applied, linear_mm_s=int(math.copysign(400, applied.linear_mm_s)))
            self.last_command = applied  # Feedback model must reflect actual limited output.
        emit("command", linear_mm_s=applied.linear_mm_s, angular_mrad_s=applied.angular_mrad_s,
             yaw_error_deg=round(math.degrees(error), 2) if error is not None else None)
        return applied

    def ttl_for(self, command, normal_ttl_ms):
        global TURN_START_REQUEST
        TURN_START_REQUEST = None
        original = original_ttl(self, command, normal_ttl_ms)
        if command.linear_mm_s != 0 or command.angular_mrad_s == 0:
            return original
        error = latest_error["rad"]
        if error is None or not math.isfinite(error):
            return min(original, 70)
        if adaptive is not None:
            experience = getattr(mission, '_phase1_turn_experience', None)
            learned = experience.ready(error) if experience is not None else None
            speed, ttl, output_stage, reused = adaptive.plan(
                error, command.angular_mrad_s, original, learned=learned)
            actual_stalls = adaptive.directions[1 if error > 0 else -1].low_progress
            # Explicit parent decision; helper must NOT re-cap this to the
            # old fastest-rate or nominal-speed angle product a second time.
            TURN_START_REQUEST = dict(mode='adaptive_feedback_turn', adaptive_start=True,
                error_rad=error, angular_mrad_s=command.angular_mrad_s,
                pulse_speed_mrad_s=speed, maximum_ttl_ms=ttl,
                stalled_scans=output_stage, measured_rate_rad_s=0., calibrated_start=False)
            TURN_START_REQUEST['actual_stalled_scans'] = actual_stalls
            self._phase1_previous_ttl_ms = ttl
            self._phase1_turn_max_ttl_ms = ttl
            self._phase1_experience_reused = reused
            emit('turn_pulse', ttl_ms=ttl, angular_mrad_s=command.angular_mrad_s,
                 pulse_speed_mrad_s=speed, yaw_error_deg=math.degrees(error),
                 stalled_scans=actual_stalls, output_stage=output_stage, dynamic_feedback=True,
                 turn_purpose=monitor.key[0] if monitor.key else None)
            return ttl
        if getattr(self, "_phase1_turn_key", None) != monitor.key:
            self._phase1_turn_key = monitor.key
            self._phase1_turn_stalled_scans = 0
            self._phase1_turn_capped_stalls = 0
        stalled = int(getattr(self, "_phase1_turn_stalled_scans", 0))
        experience = getattr(mission, "_phase1_turn_experience", None)
        learned = experience.ready(error) if experience is not None else None
        minimum = 14 if error > 0 else 12
        reuse = bool(learned and start_recovery_eligible(monitor.key, error)
                     and abs(math.degrees(error)) >= minimum and original >= 180)
        self._phase1_experience_reused = reuse
        effective_stalls = max(6, stalled) if reuse else stalled
        start_assist = (stalled >= 2 or reuse) and start_recovery_eligible(monitor.key, error)
        if start_assist:
            try:
                bounded_turn_ttl_ms(error, breakaway_speed_mrad_s(error,8), original,
                    stalled_scans=8, measured_rate_rad_s=float(getattr(self,"_phase1_measured_rate",0)),
                    breakaway=True, calibrated_start=True)
            except RuntimeError:
                start_assist = False
                emit("turn_start_budget_fallback", remaining_deg=math.degrees(error),
                     mode="precision_output_boost", high_speed_pulse_sent=False)
        precision_assist = stalled >= 2 and not start_assist and abs(math.degrees(error)) > 2
        rate = float(getattr(self, "_phase1_measured_rate", 0))
        if reuse:
            rate = max(rate, learned["fastest_rate_rad_s"])
            emit("turn_experience_reused", direction="left" if error > 0 else "right",
                 successes=learned["successes"], fastest_rate_rad_s=rate,
                 actual_stalled_scans=stalled, angle_and_clearance_limits_unchanged=True)
        pulse_speed = breakaway_speed_mrad_s(error, effective_stalls) if start_assist else command.angular_mrad_s
        ttl = bounded_turn_ttl_ms(error, pulse_speed, original, stalled_scans=effective_stalls,
                                  measured_rate_rad_s=rate, breakaway=start_assist,
                                  calibrated_start=start_assist)
        if start_assist:
            TURN_START_REQUEST = {"error_rad": error, "angular_mrad_s": command.angular_mrad_s,
                                  "maximum_ttl_ms": ttl, "stalled_scans": effective_stalls,
                                  "measured_rate_rad_s": rate, "calibrated_start": True}
        elif precision_assist:
            # Do not send this through the >10-degree breakaway helper or
            # force 3500mrad/s onto a 6-degree camera-alignment request.
            TURN_START_REQUEST = {"mode": "precision_output_boost", "error_rad": error,
                                  "angular_mrad_s": command.angular_mrad_s,
                                  "maximum_ttl_ms": ttl, "stalled_scans": stalled,
                                  "measured_rate_rad_s": rate}
        self._phase1_previous_ttl_ms = ttl
        self._phase1_turn_max_ttl_ms = bounded_turn_ttl_ms(
            error, breakaway_speed_mrad_s(error, 8) if start_assist else command.angular_mrad_s,
            original, stalled_scans=8, measured_rate_rad_s=rate, breakaway=start_assist,
            calibrated_start=start_assist)
        emit("turn_pulse", ttl_ms=ttl, angular_mrad_s=command.angular_mrad_s,
             yaw_error_deg=round(math.degrees(error), 2), stalled_scans=stalled,
             maximum_permitted_ttl_ms=self._phase1_turn_max_ttl_ms,
             pulse_speed_mrad_s=pulse_speed, breakaway=start_assist,
             precision_output_boost=precision_assist,
             turn_purpose=monitor.key[0] if monitor.key else None)
        return ttl

    mission._strict_command_for_pose = command_for_pose
    cls.observe, cls.stopped, cls.ttl_for = observe, stopped, ttl_for
    if original_apply is not None:
        cls.apply = apply
