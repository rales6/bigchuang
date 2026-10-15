"""Phase-one live scan pump and ACK-to-scan fence; no protected file edits.

N10 timestamps are host decode times, not device acquisition times. Continuous
draining plus rejection of implausibly fast buffered revolutions is required;
timestamp comparisons alone cannot turn an old buffered scan into a fresh one.
"""

from dataclasses import dataclass, replace
import math
import threading
import time
import numpy as np

from raspberry_pi.lidar.base import LaserScan


class FreshScanWaitTimeout(TimeoutError):
    """This monitor's wait expired; not a closed or failed lidar source."""


@dataclass(frozen=True)
class FreshScan(LaserScan):
    acquisition_start_s: float = 0.
    received_s: float = 0.
    sequence: int = 0


class MotionFeedbackFence:
    def __init__(self):
        self.pending = None
        self.generation = 0

    def acknowledge(self, sent, ack, ttl_ms, angular):
        self.generation += 1
        self.pending = {"generation": self.generation, "sent_s": sent, "ack_s": ack,
                        "ttl_ms": ttl_ms, "angular_mrad_s": angular,
                        "after_s": ack + ttl_ms / 1000. + .05}
        return dict(self.pending)

    def cancel(self):
        self.pending = None

    def not_before(self):
        return self.pending["after_s"] if self.pending else 0.

    def assess(self, before, after):
        pending = self.pending
        if pending is None:
            return None
        self.pending = None  # Exactly one motion observation per ACKed pulse.
        baseline = getattr(before, "received_s", None)
        start = getattr(after, "acquisition_start_s", None)
        end = getattr(after, "received_s", None)
        valid = (baseline is not None and start is not None and end is not None
                 and 0 <= pending["sent_s"] - baseline <= 1.5
                 and start >= pending["after_s"] and end >= start
                 and time.monotonic() - end <= .5)
        return {**pending, "valid": valid, "before_received_s": baseline,
                "after_start_s": start, "after_received_s": end,
                "reason": "post_pulse_fresh_scan" if valid else "stale_or_unpaired_turn_scan"}


class FreshScanStream:
    """One reader continuously drains the driver; only newest usable frame kept."""
    def __init__(self, source, driver, fence, *, timeout_s=3., emit=lambda *a, **kw: None):
        self.source, self.driver, self.fence = iter(source), driver, fence
        self.timeout_s, self.emit = timeout_s, emit
        self.condition = threading.Condition()
        self.closed = False
        self.latest = None
        self.error = None
        self.delivered = self.sequence = self.discarded_burst = 0
        self.thread = threading.Thread(target=self._pump, daemon=True, name="phase1-lidar-drain")
        self.thread.start()

    def _pump(self):
        previous_paced_start = None
        try:
            while not self.closed:
                started = time.monotonic()
                scan = next(self.source)
                received = time.monotonic()
                hz = float(getattr(self.driver, "last_scan_frequency_hz", 10.) or 10.)
                hz = max(2., min(30., hz)) if math.isfinite(hz) else 10.
                # Buffered old revolutions are decoded in ~6 ms despite a
                # ~100 ms live period. Reject the entire burst until reads
                # again span at least half a reported revolution period.
                paced = .5 / hz <= received - started <= max(.5, 2. / hz)
                if not paced or len(scan.angles_rad) < 80:
                    previous_paced_start = None
                    with self.condition:
                        self.discarded_burst += 1
                    continue
                # The first paced frame after backlog may straddle old/new
                # packets. Require another paced revolution and use the prior
                # read's start as a conservative host acquisition lower bound.
                acquisition_start = previous_paced_start
                previous_paced_start = started
                if acquisition_start is None:
                    with self.condition:
                        self.discarded_burst += 1
                    continue
                with self.condition:
                    self.sequence += 1
                    self.latest = FreshScan(scan.angles_rad, scan.distances_m, received,
                                            None, acquisition_start, received, self.sequence)
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def __iter__(self):
        return self

    def latest_after(self, sequence=None, timeout_s=.30):
        """Recovery monitor reads without consuming the navigation baseline."""
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                now = time.monotonic()
                scan = self.latest
                if (scan is not None and (sequence is None or scan.sequence > sequence)
                        and now-scan.received_s <= .25):
                    return scan
                if self.closed:
                    raise RuntimeError("实时雷达流已关闭；不授权新运动")
                if self.error is not None:
                    raise RuntimeError("实时雷达读取失败；不授权新运动") from self.error
                if now >= deadline:
                    raise FreshScanWaitTimeout("本次监测等待未收到新雷达帧；不续租，转后重新取帧")
                self.condition.wait(min(.03, deadline-now))

    def __next__(self):
        deadline = time.monotonic() + self.timeout_s
        with self.condition:
            while True:
                now = time.monotonic()
                scan = self.latest
                if (scan is not None and scan.sequence > self.delivered
                        and scan.acquisition_start_s >= self.fence.not_before()
                        and now - scan.received_s <= .5):
                    skipped = scan.sequence - self.delivered - 1
                    self.delivered = scan.sequence
                    break
                if self.closed:
                    raise StopIteration
                if self.error is not None:
                    raise RuntimeError("实时雷达读取失败；未获得新扫描") from self.error
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError("3秒内未获得脉冲结束后的完整新雷达扫描；不授权新运动")
                self.condition.wait(min(remaining, .1))
        self.emit("fresh_scan_delivered", sequence=scan.sequence, skipped_frames=skipped,
                  discarded_buffered_scans=self.discarded_burst,
                  acquisition_start_s=scan.acquisition_start_s, received_s=scan.received_s,
                  age_s=round(time.monotonic()-scan.received_s, 4))
        return scan

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.driver.close()
        self.thread.join(timeout=2.)


def install_fresh_motion_feedback(mission):
    from grasp_range_integration.phase1.turn_policy import emit as telemetry_emit

    def emit(kind, **fields):
        if getattr(mission, '_phase1_recovery_mode', None) == 'midrun' and kind.startswith('startup_'):
            kind = 'midrun_'+kind[len('startup_'):]
        if kind == "turn_motor_output_boost":
            mission._phase1_latest_turn_pulse = dict(fields, recorded_s=time.monotonic())
            callback = getattr(mission, "_phase1_turn_pulse_sent", None)
            if callback is not None:
                callback(fields)
        telemetry_emit(kind, **fields)

    fence = MotionFeedbackFence()
    original_driver, original_client = mission.N10LidarDriver, mission.Esp32Client
    original_relative = mission.ScanMapLocalizer.relative_motion
    live = {"stream": None, "localizer": None, "client": None}

    class LiveDriver:
        def __init__(self, *args, **kwargs):
            self.driver = original_driver(*args, **kwargs)
            self.stream = None

        def scans(self):
            if self.stream is None:
                self.stream = FreshScanStream(self.driver.scans(), self.driver, fence, emit=emit)
                live["stream"] = self.stream
            return self.stream

        def close(self):
            if self.stream is not None:
                self.stream.close()
            else:
                self.driver.close()

        def __getattr__(self, name):
            return getattr(self.driver, name)

    class AckClient(original_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live["client"] = self

        def startup_recovery_pulse(self, baseline, prior, known_map, linear, angular, ttl_ms,
                                   *, operator_clear=False, localizer=None, drive_budget_ms=1800,
                                   stopping=lambda: False, recovery_phase='startup'):
            """Separate, envelope-checked localization action; no guessed pose.

            Normal navigation requires a strong map pose and cannot authorize
            startup recovery. This path instead checks the whole user start
            envelope plus fresh raw ranges, without changing normal guards.
            """
            from grasp_range_integration.phase1.startup_active_recovery import assess_recovery_action, propagate_prior
            from grasp_range_integration.phase1.turn_output_boost import send_boosted_turn
            from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence
            phase_label = '途中' if recovery_phase == 'midrun' else '启动'
            if (fence.pending is not None or getattr(mission, '_phase1_motion_fault', None)
                    or live['stream'] is None or localizer is None or not 50 <= ttl_ms <= 400
                    or recovery_phase not in ('startup','midrun')
                    or (recovery_phase == 'midrun' and getattr(mission,'_phase1_recovery_mode',None) != 'midrun')
                    or (linear, angular) not in ((0, -3500), (0, 3500), (400, 0), (460, 0))):
                super().stop()
                raise RuntimeError(f'{phase_label}恢复动作/反馈契约不满足；未授权运动')
            assessment = assess_recovery_action(baseline, known_map, prior, linear, angular, ttl_ms,
                operator_clear=operator_clear, directional_translation=recovery_phase == 'midrun')
            emit('startup_pulse_authorization', **assessment)
            if not assessment['safe']:
                super().stop()
                # Veto this unsent action, not the entire mission. Contract,
                # STOP ACK and post-send transport failures still propagate.
                emit('startup_pulse_deferred', reason=assessment['reason'],
                     command_sent=False, terminal=False,
                     next_action='fresh_scan_reselect', ttl_ms=ttl_ms)
                return dict(status='deferred', command_sent=False, assessment=assessment)
            if linear:
                from types import SimpleNamespace
                from grasp_range_integration.phase1.startup_drive import run_startup_drive
                raw_client = super()
                generation = [None]

                def start_drive():
                    # Same wheel balance as successful normal navigation.
                    # No background thread may renew this command.
                    with self._request_lock:
                        with self._state_lock:
                            generation[0] = self._drive_request_generation+1
                        raw_client.set_bounded_twist(linear, 0, ttl_ms,
                            refresh_deadline_s=time.monotonic()-1.)

                def renew_drive(scan, expires, deadline):
                    # Encode and resend only this ACKed unchanged command.
                    # Unlike refresh_twist this never arms unlimited keepalive.
                    with self._request_lock:
                        with self._state_lock:
                            allowed = self._drive_request_generation == generation[0]
                            now = time.monotonic()
                            if (not allowed or stopping() or getattr(mission, '_phase1_motion_fault', None)
                                    or now >= expires or now+ttl_ms/1000. > deadline
                                    or not 0 <= now-scan.received_s <= .25):
                                return False
                            kind, payload = self._encode_twist_request(linear, 0, ttl_ms)
                            raw_client.send_without_response(kind, payload)
                            return True

                def measure_drive(before, current):
                    fit = original_relative(localizer, before, current)
                    t = fit.transform
                    evidence = turn_range_evidence(before, current, t,
                        lidar_offset=localizer.lidar_offset, maximum_range=localizer.max_distance_m)
                    reliable = (all(math.isfinite(v) for v in
                        (t.x_m, t.y_m, t.yaw_rad, fit.rmse_m, fit.inlier_ratio))
                        and fit.converged and fit.rmse_m <= .09 and fit.inlier_ratio >= .65
                        and abs(t.yaw_rad) <= math.radians(24) and evidence['accepted'])
                    return SimpleNamespace(transform=t, reliable=reliable, raw_range=evidence)

                return run_startup_drive(baseline, ttl_ms=ttl_ms, budget_ms=drive_budget_ms,
                    start=start_drive, renew=renew_drive, read_scan=live['stream'].latest_after,
                    measure=measure_drive,
                    assess=lambda scan, t: assess_recovery_action(scan, known_map,
                        propagate_prior(prior, t), linear, 0, ttl_ms, operator_clear=operator_clear,
                        directional_translation=recovery_phase == 'midrun'),
                    stop=raw_client.stop, cancelled=stopping, emit=emit,
                    clock=lambda: time.monotonic(),
                    short_probe=recovery_phase == 'midrun' and ttl_ms < 400)
            sent = time.monotonic()
            try:
                if angular:
                    send_boosted_turn(self, super().set_bounded_twist, angular, ttl_ms, sent-1., 6,
                                     lambda kind, **kw: emit('startup_'+kind, **kw))
                else:
                    super().set_bounded_twist(linear, 0, ttl_ms, refresh_deadline_s=sent-1.)
                sequence = baseline.sequence
                while time.monotonic() < sent+ttl_ms/1000.:
                    if stopping():
                        break
                    try:
                        current = live['stream'].latest_after(sequence, .25)
                    except FreshScanWaitTimeout:
                        # End this bounded pulse, then let recover_startup
                        # acquire and verify a complete stopped scan. Never
                        # renew or guess motion from this missing frame.
                        emit('startup_pulse_feedback_wait', terminal=False,
                             next_action='post_stop_fresh_scan', renewal_allowed=False)
                        break
                    sequence = current.sequence
                    fit = original_relative(localizer, baseline, current)
                    t = fit.transform
                    values = (t.x_m, t.y_m, t.yaw_rad, fit.rmse_m, fit.inlier_ratio)
                    evidence = turn_range_evidence(baseline, current, t,
                        lidar_offset=localizer.lidar_offset, maximum_range=localizer.max_distance_m)
                    usable = (all(math.isfinite(v) for v in values) and fit.converged
                              and fit.rmse_m <= .09 and fit.inlier_ratio >= .65 and evidence['accepted'])
                    emit('startup_pulse_live_motion', reliable=usable, dx_m=t.x_m, dy_m=t.y_m,
                         rotation_deg=math.degrees(t.yaw_rad), raw_range=evidence)
                    if not usable:
                        break
                    moved_prior = propagate_prior(prior, t)
                    check = assess_recovery_action(current, known_map, moved_prior, linear, angular, ttl_ms,
                                                    operator_clear=operator_clear,
                                                    directional_translation=recovery_phase == 'midrun')
                    emit('startup_pulse_live_clearance', **check)
                    if not check['safe']:
                        break
                    if ((angular and (abs(t.yaw_rad) >= math.radians(12)
                                      or t.yaw_rad*angular < -math.radians(.5)*abs(angular)))
                            or (linear and math.hypot(t.x_m, t.y_m) >= .10)):
                        emit('startup_pulse_live_brake', reason='measured_local_step_complete')
                        break
            finally:
                # No refresh or automatic repeat. The caller must establish
                # new motion evidence and search bounds after this STOP ACK.
                super().stop()
                emit('startup_pulse_stopped', ttl_ms=ttl_ms, renewal_allowed=False)

        def set_twist(self, linear_mm_s, angular_mrad_s, ttl_ms=600):
            if ((linear_mm_s or angular_mrad_s)
                    and (getattr(mission,'_phase1_turn_relocalization_match',None) is not None
                         or getattr(mission,'_phase1_turn_relocalization_pending',False))):
                super().stop()
                emit('motion_command_deferred',
                     reason='turn_relocalization_pending',
                     linear_mm_s=linear_mm_s,angular_mrad_s=angular_mrad_s,
                     motion_command_allowed=False)
                return None
            estimate = getattr(mission, "_phase1_navigation_estimate", None)
            if estimate and linear_mm_s:
                from grasp_range_integration.phase1.motion_interlock import (
                    estimate_command_horizon, estimate_defer_reason)
                # Preserve the established 400mm/s breakaway ceiling, but
                # never allow a full acceleration/long renewable lease while
                # the pose is only bounded odometry. Final glass/near-obstacle
                # authorization below sees the ACTUAL limited command.
                linear_mm_s = max(-400, min(400, linear_mm_s))
                speed_m_s = abs(linear_mm_s)/1000.
                remaining_s = estimate_command_horizon(estimate, time.monotonic(), speed_m_s)
                ttl_ms = min(250, ttl_ms, int(remaining_s*1000.))
                if ttl_ms < 50:
                    reason = estimate_defer_reason(estimate, time.monotonic(), speed_m_s)
                    # A slow map calculation can age just this scan. Wait
                    # for the next real frame, don't latch a whole recovery
                    # episode or reset/invent the odometry confidence.
                    estimate["command_budget_exhausted"] = reason == 'estimated_navigation_budget_exhausted'
                    super().stop()
                    hold = getattr(mission, "_phase1_motion_hold", None)
                    if hold is not None:
                        hold()
                    emit("motion_command_deferred", reason=reason,
                         command_sent=False)
                    return None
                emit("estimated_navigation_command_limited", linear_mm_s=linear_mm_s,
                     ttl_ms=ttl_ms, uncertainty_m=estimate["uncertainty_m"])
            distance = getattr(mission,"_phase1_arrival_distance_m",None)
            if linear_mm_s > 0 and distance is not None:
                from grasp_range_integration.phase1.arrival_visibility import linear_arrival_ttl
                ttl_ms = linear_arrival_ttl(distance,linear_mm_s,ttl_ms)
                emit("arrival_linear_lease_budget", distance_m=distance,linear_mm_s=linear_mm_s,
                     ttl_ms=ttl_ms,renewal_allowed=False,command_sent=False)
            if (linear_mm_s or angular_mrad_s) and fence.pending is not None:
                mission._phase1_motion_fault = "turn_feedback_not_consumed"
            elif fence.pending is not None:
                # A zero command is allowed to stop, never to erase the need
                # to establish the post-turn pose before future movement.
                mission._phase1_motion_fault = "turn_interrupted_without_feedback"
            if (linear_mm_s or angular_mrad_s) and getattr(mission, "_phase1_motion_fault", None):
                super().stop()
                raise RuntimeError("运动反馈故障已锁定；禁止沿旧位姿继续运动：" + mission._phase1_motion_fault)
            authorize = getattr(mission, "_phase1_authorize_motion", None)
            if (linear_mm_s or angular_mrad_s) and authorize is not None:
                try:
                    ready = authorize(linear_mm_s, angular_mrad_s, ttl_ms)
                except Exception:
                    super().stop()
                    raise
                if ready is False:
                    super().stop()
                    hold = getattr(mission, "_phase1_motion_hold", None)
                    if hold is not None:
                        hold()
                    emit("motion_command_deferred", linear_mm_s=linear_mm_s,
                         angular_mrad_s=angular_mrad_s, reason="freshness_or_clearance_recheck",
                         command_sent=False)
                    return None
            if linear_mm_s != 0 or angular_mrad_s == 0:
                fence.cancel()
                if linear_mm_s and (getattr(mission, "_phase1_verified_glass_required", False)
                                    or distance is not None):
                    sent = time.monotonic()
                    response = super().set_bounded_twist(linear_mm_s, angular_mrad_s, ttl_ms,
                        refresh_deadline_s=sent-1.)
                    if distance is not None:
                        emit("arrival_linear_command_ack",linear_mm_s=linear_mm_s,
                             ttl_ms=ttl_ms,renewal_allowed=False,physical_motion_pending=True)
                    return response
                return super().set_twist(linear_mm_s, angular_mrad_s, ttl_ms)
            sent = time.monotonic()
            recovery_started = False
            try:
                from grasp_range_integration.phase1 import turn_policy
                request = turn_policy.TURN_START_REQUEST
                assess = getattr(mission, "_phase1_rotation_assess", None)
                precision = request is not None and request.get("mode") == "precision_output_boost"
                if (request is not None and request.get('adaptive_start') and
                        (request['angular_mrad_s'] != angular_mrad_s or live['stream'] is None
                         or live['localizer'] is None or assess is None)):
                    raise RuntimeError('自适应转向缺少实时雷达/净空或命令不一致；不授权')
                if precision and request["angular_mrad_s"] != angular_mrad_s:
                    raise RuntimeError("小角度加力请求与实际命令不一致；不授权")
                if precision and (live["stream"] is None or live["localizer"] is None or assess is None):
                    raise RuntimeError("小角度加力缺少实时雷达/净空服务；不授权")
                if (request is not None and request["angular_mrad_s"] == angular_mrad_s
                        and live["stream"] is not None and live["localizer"] is not None
                        and assess is not None and not precision):
                    from grasp_range_integration.phase1.turn_start_recovery import run_turn_start_recovery
                    from grasp_range_integration.phase1.turn_output_boost import send_boosted_turn
                    localizer = live["localizer"]

                    def measure(before, after):
                        from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence
                        match = original_relative(localizer, before, after)
                        evidence = turn_range_evidence(before, after, match.transform,
                            lidar_offset=localizer.lidar_offset, maximum_range=localizer.max_distance_m)
                        emit("turn_range_feedback", **evidence)
                        return match if evidence["accepted"] else replace(match, converged=False)

                    print("转向恢复：根据当前反馈逐步加力加时；单次不续租，实时检查净空和转角，转后新扫描确认。", flush=True)
                    recovery_started = True
                    run_turn_start_recovery(
                        error_rad=request["error_rad"], read_scan=live["stream"].latest_after,
                        maximum_ttl_ms=min(ttl_ms, request.get("maximum_ttl_ms", ttl_ms)),
                        stalled_scans=request.get("stalled_scans", 2),
                        measured_rate_rad_s=request.get("measured_rate_rad_s", 0.),
                        calibrated_start=request.get("calibrated_start", False),
                        adaptive_start=request.get('adaptive_start',False),
                        pulse_speed_mrad_s=request.get('pulse_speed_mrad_s'),
                        actual_stalled_scans=request.get('actual_stalled_scans'),
                        measure=measure, assess=assess,
                        send=lambda speed, ttl, deadline: send_boosted_turn(self,
                            super(AckClient, self).set_bounded_twist,
                            speed, ttl, deadline, request.get("stalled_scans", 2), emit),
                        renew=super().allow_twist_to_refresh_for,
                        stop=super().stop, emit=emit)
                    finished = time.monotonic()
                    pending = fence.acknowledge(sent, finished, 0, angular_mrad_s)
                    emit("turn_start_feedback_fence", **pending, completed_s=finished)
                    return None
                # An already-expired refresh deadline prohibits background
                # refresh from ever being armed, including after a late ACK.
                if precision:
                    from grasp_range_integration.phase1.turn_output_boost import send_boosted_turn
                    cap = int(request["maximum_ttl_ms"])
                    if ttl_ms > cap or not 50 <= ttl_ms <= 70:
                        raise RuntimeError("小角度加力不得扩大短脉冲预算；不授权")
                    assessment = assess(live["stream"].latest_after(None, .25))
                    emit("precision_turn_clearance", **assessment)
                    if not assessment["safe"]:
                        raise RuntimeError("小角度加力净空未通过："+str(assessment.get("reason")))
                    value = send_boosted_turn(self, super(AckClient, self).set_bounded_twist,
                        angular_mrad_s, ttl_ms, sent, request["stalled_scans"], emit)
                    emit("precision_turn_output_boost", remaining_deg=math.degrees(request["error_rad"]),
                         angular_mrad_s=angular_mrad_s, ttl_ms=ttl_ms,
                         stalled_scans=request["stalled_scans"], linear_mm_s=0)
                else:
                    value = super().set_bounded_twist(0, angular_mrad_s, ttl_ms,
                                                     refresh_deadline_s=sent)
                self.allow_twist_to_expire()
            except Exception as exc:
                fence.cancel()
                super().stop()
                emit("turn_start_failed" if recovery_started else "turn_command_ack_failed",
                     angular_mrad_s=angular_mrad_s, ttl_ms=ttl_ms,
                     error_type=type(exc).__name__, error=str(exc))
                raise
            ack = time.monotonic()
            pending = fence.acknowledge(sent, ack, ttl_ms, angular_mrad_s)
            emit("turn_command_ack", **pending, ack_latency_s=round(ack-sent, 4))
            return value

        def stop(self):
            if fence.pending is not None:
                mission._phase1_motion_fault = "turn_interrupted_without_feedback"
            fence.cancel()
            return super().stop()

        def refresh_twist(self, linear_mm_s, angular_mrad_s, ttl_ms=600):
            # The protected mission uses this path for unchanged commands.
            # It must not bypass the same glass/fault gate or re-arm unlimited
            # background drive refresh after a slow localization calculation.
            return self.set_twist(linear_mm_s, angular_mrad_s, ttl_ms)

    def relative_motion(self, before, after, *args, **kwargs):
        live["localizer"] = self
        from grasp_range_integration.phase1.localization_handoff import recovery_baseline
        before = recovery_baseline(mission, self, before, after)
        assessment = fence.assess(before, after)
        # Identical raw scan pair is often fitted again inside track(). Only
        # that exact immutable evidence is reused, never a future/other pair.
        fingerprint = tuple(hash(np.asarray(v).tobytes()) for scan in (before,after)
                            for v in (scan.angles_rad,scan.distances_m))
        cached = getattr(self,"_phase1_same_pair_motion",None)
        hit = (not args and not kwargs and cached is not None
               and cached[0] is before and cached[1] is after and cached[2] == fingerprint)
        match = cached[3] if hit else original_relative(self, before, after, *args, **kwargs)
        if hit:
            emit("scan_pair_fit_reused", same_raw_pair=True,
                 after_sequence=getattr(after,"sequence",None), feedback_gate_rechecked=True)
        if assessment is not None:
            mission._phase1_turn_feedback_recovery_status = None
            mission._phase1_turn_feedback_uncertainty_m = 0.
            emit("turn_feedback_window", **assessment,
                 measured_rotation_deg=math.degrees(match.transform.yaw_rad),
                 measured_dx_m=match.transform.x_m, measured_dy_m=match.transform.y_m,
                 measured_translation_m=math.hypot(match.transform.x_m, match.transform.y_m),
                 rmse_m=match.rmse_m, inlier_ratio=match.inlier_ratio,
                 converged=match.converged)
            from grasp_range_integration.phase1.motion_interlock import validate_turn_feedback
            fault = validate_turn_feedback(assessment, match)
            if (fault == 'turn_motion_outside_prediction_limit'
                    and live['stream'] is not None and live['client'] is not None):
                from grasp_range_integration.phase1.turn_feedback_recovery import recover_large_turn
                def large_turn_event(kind, **fields):
                    if kind == 'turn_feedback_recovery':
                        mission._phase1_turn_feedback_recovery_status = fields.get('status')
                    emit(kind, **fields)
                try:
                    recovered = recover_large_turn(before, after, match, assessment, self,
                        intent=getattr(mission, '_phase1_latest_turn_pulse', None),
                        relative=lambda left,right: original_relative(self,left,right),
                        read_scan=live['stream'].latest_after,
                        stop=live['client'].stop,
                        anchor=getattr(mission,'_phase1_turn_pose_anchor',None),
                        emit=large_turn_event)
                except Exception as exc:
                    large_turn_event('turn_feedback_recovery',status='large_turn_recheck_error',
                                     error_type=type(exc).__name__,error=str(exc))
                    recovered = None
                if recovered is not None:
                    match = recovered
                    fault = None
                    mission._phase1_verified_large_turn_match = match
                    emit('turn_feedback_recovered',
                         recovery=getattr(mission,
                             '_phase1_turn_feedback_recovery_status',None),
                         rmse_m=match.rmse_m, inlier_ratio=match.inlier_ratio,
                         rotation_deg=math.degrees(match.transform.yaw_rad),
                         translation_m=math.hypot(match.transform.x_m,
                                                  match.transform.y_m),
                         full_motion_verified=True)
            if (fault == 'unreliable_turn_feedback' and live['stream'] is not None
                    and live['client'] is not None):
                from grasp_range_integration.phase1.turn_feedback_recovery import recover_ambiguous_turn
                def recovery_event(kind, **fields):
                    if kind == 'turn_feedback_recovery':
                        mission._phase1_turn_feedback_recovery_status = fields.get('status')
                        mission._phase1_turn_feedback_uncertainty_m = fields.get(
                            'extra_position_uncertainty_m', 0.)
                    emit(kind, **fields)
                try:
                    recovered = recover_ambiguous_turn(before, after, match, assessment, self,
                        relative=lambda left,right: original_relative(self,left,right),
                        read_scan=live['stream'].latest_after,
                        stop=live['client'].stop,
                        anchor=getattr(mission,'_phase1_turn_pose_anchor',None), emit=recovery_event)
                except Exception as exc:
                    recovery_event('turn_feedback_recovery',status='stop_or_feedback_error',
                                   error_type=type(exc).__name__,error=str(exc))
                    recovered = None
                if recovered is not None:
                    match = recovered
                    bounded = getattr(mission,'_phase1_turn_feedback_recovery_status',None) in (
                        'bounded_relative_odometry','second_scan_bounded_relative_odometry')
                    fault = None if bounded else validate_turn_feedback(assessment, match)
                    if bounded:
                        mission._phase1_bounded_turn_match = (match,
                            mission._phase1_turn_feedback_uncertainty_m)
                    emit('turn_feedback_recovered',rmse_m=match.rmse_m,
                         inlier_ratio=match.inlier_ratio,
                         rotation_deg=math.degrees(match.transform.yaw_rad),
                         full_motion_verified=True, bounded_odometry=bounded,
                         map_anchor_required=False if bounded else None)
            anchor = getattr(mission, '_phase1_turn_pose_anchor', None)
            if fault:
                # Quality/scan-pair rejection is a recoverable measurement
                # state even when the preceding pose was bounded odometry.
                # Requiring an *exact strong map anchor* here used to bypass
                # all recovery and terminate run 221417. Keep the last pose
                # as a prior, carry the measured transform as a search hint,
                # and reacquire evidence through the normal recovery path.
                if live['client'] is not None:
                    live['client'].stop()
                mission._phase1_turn_relocalization_hint = match
                match = replace(match, converged=False)
                mission._phase1_turn_relocalization_match = match
                mission._phase1_turn_relocalization_reason = fault
                emit('turn_relocalization_requested', reason=fault,
                     independent_recheck=getattr(mission,
                         '_phase1_turn_feedback_recovery_status',None),
                     last_strong_map_pose=([anchor[1].x_m,anchor[1].y_m,
                                           anchor[1].yaw_rad] if anchor is not None else None),
                     exact_map_anchor=bool(anchor is not None and anchor[0] is before),
                     terminal=False, next_action='fresh_scan_and_bounded_localization',
                     motion_command_allowed=False)
                fault = None
            if match is not getattr(mission,'_phase1_turn_relocalization_match',None):
                mission._phase1_post_turn_match = match
            mission._phase1_latest_turn_pulse = None
        if not args and not kwargs:
            self._phase1_same_pair_motion = (before,after,fingerprint,match)
        return match

    mission.N10LidarDriver, mission.Esp32Client = LiveDriver, AckClient
    mission.ScanMapLocalizer.relative_motion = relative_motion

    def refresh_rotation_evidence(reference):
        """Stop first, then get a scan acquired after STOP, not after map fit.

        This does not replace the mission odometry baseline. Check only tiny
        motion so the older map pose may conservatively support this frame.
        A moved vehicle must wait for the next normal map track instead.
        """
        if any(live[name] is None for name in ("client", "stream", "localizer")):
            return None
        live["client"].stop()
        stopped = time.monotonic()
        deadline, sequence = stopped + 1., getattr(reference, "sequence", -1)
        while time.monotonic() < deadline:
            current = live["stream"].latest_after(sequence, timeout_s=.3)
            sequence = current.sequence
            if current.acquisition_start_s >= stopped + .05:
                break
        else:
            raise TimeoutError("停车后未取得新旋转净空扫描")
        match = original_relative(live["localizer"], reference, current)
        from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence
        evidence = turn_range_evidence(reference, current, match.transform,
            lidar_offset=live["localizer"].lidar_offset,
            maximum_range=live["localizer"].max_distance_m)
        small = bool(match.converged and math.isfinite(match.rmse_m) and match.rmse_m <= .09
                     and match.inlier_ratio >= .65 and evidence["accepted"]
                     and math.hypot(match.transform.x_m, match.transform.y_m) <= .03
                     and abs(match.transform.yaw_rad) <= math.radians(2))
        emit("rotation_fresh_clearance_scan", sequence=current.sequence,
             reference_age_s=time.monotonic()-reference.received_s,
             scan_age_s=time.monotonic()-current.received_s, stationary_pose_supported=small,
             rotation_deg=math.degrees(match.transform.yaw_rad),
             translation_m=math.hypot(match.transform.x_m, match.transform.y_m))
        return current, small

    mission._phase1_refresh_rotation_evidence = refresh_rotation_evidence
    return fence
