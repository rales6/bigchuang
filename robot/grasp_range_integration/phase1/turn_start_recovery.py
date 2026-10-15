"""One angle-budgeted breakaway pulse; independent stopped feedback is mandatory."""

import math
import time

from grasp_range_integration.phase1.turn_policy import bounded_turn_ttl_ms, breakaway_speed_mrad_s


def run_turn_start_recovery(*, error_rad, read_scan, measure, assess, send,
                            renew, stop, emit, clock=time.monotonic,
                            maximum_ttl_ms=None, stalled_scans=0, measured_rate_rad_s=0.,
                            calibrated_start=False, adaptive_start=False, pulse_speed_mrad_s=None,
                            actual_stalled_scans=None):
    """Never replace a small parent pulse with a renewable 350 ms lease.

    Torque escalation uses OUTER, independent post-stop observations, not
    near-zero frames acquired before delayed motor breakaway. Live mode uses
    measure only to BRAKE early, never to renew or increase this pulse.
    The legacy calibrated probe still measures only after STOP.
    The caller must fence and validate final stopped motion before retrying.
    """
    if not math.isfinite(error_rad) or abs(error_rad) <= math.radians(2 if adaptive_start else 10):
        raise ValueError("Turn error is below the active recovery threshold")
    parent_cap = (70 if abs(error_rad) <= math.radians(35) else 140) if maximum_ttl_ms is None else int(maximum_ttl_ms)
    if not 50 <= parent_cap <= 650:
        raise ValueError("Invalid parent pulse budget")
    started = clock()
    reported_stalls = stalled_scans if actual_stalled_scans is None else actual_stalled_scans
    reason = "single_pulse_complete"
    try:
        anchor = read_scan(None, .25)
        assessment = assess(anchor)
        emit("turn_start_clearance", sequence=anchor.sequence, **assessment)
        if not assessment["safe"]:
            reason = ("near_obstacle_recheck" if assessment["reason"] == "exterior_near_obstacle"
                      else "clearance_recheck_needed")
            return reason
        if not 0 <= clock() - anchor.received_s <= .30:
            raise RuntimeError("转向雷达已过时；不授权脉冲")
        speed = breakaway_speed_mrad_s(error_rad, stalled_scans)
        # Even a large remaining angle cannot authorize a nominal sweep
        # beyond the protected predictor's 24-degree single-update range.
        # 20 degrees leaves predictor headroom; this is NOT a measured
        # motor calibration. Real overrun still triggers the hard interlock.
        feedback_cap = max(50, int(math.radians(20)/(abs(speed)/1000.)*1000))
        from grasp_range_integration.phase1.turn_calibration import calibrated_start_ttl_ms, source_run_for
        calibrated = (calibrated_start_ttl_ms(error_rad, speed, stalled_scans,
                       max(parent_cap, 180), measured_rate_rad_s) if calibrated_start else None)
        if calibrated is not None:
            # The empirical stopped-angle measurement includes acceleration,
            # expiry/deceleration and STOP ACK. Do not reapply target-rate*T
            # here: that was the second cap silently undoing the fix.
            feedback_cap = calibrated
        if adaptive_start:
            if (pulse_speed_mrad_s is None or not 0 < abs(pulse_speed_mrad_s) <= 3500
                    or pulse_speed_mrad_s*error_rad <= 0 or parent_cap > 250):
                raise ValueError('Invalid live-feedback turn output')
            speed, ttl = int(pulse_speed_mrad_s), parent_cap
        elif calibrated is not None:
            ttl = min(parent_cap, calibrated)
        else:
            ttl = bounded_turn_ttl_ms(error_rad, speed, min(parent_cap, feedback_cap),
                stalled_scans=stalled_scans, breakaway=True,
                measured_rate_rad_s=measured_rate_rad_s)
        sent = clock()
        # Expired BEFORE send: a late ACK must not arm background renewal.
        send(speed, ttl, sent-1.)
        emit("turn_start_lease", angular_mrad_s=speed, ttl_ms=ttl,
             parent_maximum_ttl_ms=parent_cap, remaining_deg=math.degrees(error_rad),
             absolute_refresh_deadline_s=sent-1., mode="single_nonrenewable_pulse",
             stalled_frames=reported_stalls, output_stage=stalled_scans,
             nominal_sweep_deg=math.degrees(abs(speed)*ttl/1e6),
             measured_rate_rad_s=measured_rate_rad_s)
        if calibrated is not None:
            emit("turn_calibrated_start", source_run=source_run_for(speed), ttl_ms=ttl,
                 direction="left" if speed > 0 else "right", empirical_estimate_not_angle_guarantee=True,
                 post_stop_feedback_required=True)
        print("起转恢复：连续{}次未起转，命令={:+d}mrad/s，实际脉冲={}ms，剩余角度={:.1f}°；单次不续租，转后雷达复核。".format(
            reported_stalls, speed, ttl, math.degrees(error_rad)), flush=True)
        sequence = anchor.sequence
        while clock() < sent + ttl/1000.:
            current = read_scan(sequence, .25)
            sequence = current.sequence
            assessment = assess(current)
            emit("turn_start_clearance", sequence=sequence, **assessment)
            if not assessment["safe"]:
                reason = "clearance_recheck_needed"
                break
            if adaptive_start:
                fit = measure(anchor, current)
                if (not fit.converged or not math.isfinite(fit.rmse_m)
                        or fit.rmse_m > .09 or fit.inlier_ratio < .65):
                    reason = 'live_motion_uncertain'
                    break
                angle = fit.transform.yaw_rad
                shift = math.hypot(fit.transform.x_m, fit.transform.y_m)
                emit('turn_live_progress', rotation_deg=math.degrees(angle), translation_m=shift,
                     reliable=True, post_stop_feedback_still_required=True)
                if (angle*error_rad < -math.radians(.5)*abs(error_rad)
                        or abs(angle) >= min(math.radians(8),abs(error_rad))):
                    reason = 'live_feedback_brake'
                    break
    except Exception as exc:
        reason = "failed:"+type(exc).__name__
        emit("turn_start_failure", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stop()
        emit("turn_start_complete", reason=reason, elapsed_s=clock()-started,
             actual_rotation_deg=None, rotation_scope="post_stop_feedback_required",
             post_stop_rotation_pending=True)
    return reason
