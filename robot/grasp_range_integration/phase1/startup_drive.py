"""Continuous breakaway from individually checked firmware leases, not keepalive.

400 mm/s for ~1.4 s worked in run 203859 after initially tiny scan motion.
Allow that settling time while checking EVERY fresh scan. Never renew from a
timer, a stale scan, an unacknowledged command or beyond this action's deadline.
"""
import math
import time


def run_startup_drive(baseline, *, ttl_ms, budget_ms, start, renew, read_scan,
                      measure, assess, stop, cancelled=lambda: False,
                      emit=lambda *a, **kw: None, clock=time.monotonic,
                      short_probe=False):
    if (short_probe and not (150 <= ttl_ms <= 250 and 600 <= budget_ms <= 1000)
            or not short_probe and (ttl_ms != 400 or not 1800 <= budget_ms <= 2400)):
        raise ValueError('invalid startup drive lease/breakaway budget')
    sent = clock()
    deadline = sent+budget_ms/1000.
    expires = sent+ttl_ms/1000.
    sequence = baseline.sequence
    reason, first_progress = 'breakaway_window_complete', None
    renewals = 0
    try:
        if cancelled():
            reason = 'cancelled'
            return dict(reason=reason, elapsed_s=0., renewals=0)
        start()
        while clock() < deadline:
            if cancelled():
                reason = 'cancelled'
                break
            from grasp_range_integration.phase1.fresh_motion_feedback import FreshScanWaitTimeout
            try:
                current = read_scan(sequence, min(.25, max(.001, expires-clock())))
            except FreshScanWaitTimeout:
                # A 150-250ms probe can end before the next full revolution.
                # STOP this pulse and return to fresh post-stop measurement,
                # rather than aborting the outer recovery mission. Real
                # source/transport failures still propagate through finally.
                reason = 'fresh_scan_wait_expired'
                emit('startup_pulse_feedback_wait', terminal=False,
                     next_action='post_stop_fresh_scan', renewal_allowed=False)
                break
            if clock() >= expires:
                # A monitoring/ACK delay must not silently restart an expired
                # command or manufacture a continuous-success sample.
                reason = 'lease_expired_before_fresh_feedback'
                break
            sequence = current.sequence
            if not 0 <= clock()-getattr(current, 'received_s', -math.inf) <= .25:
                reason = 'stale_drive_scan'
                break
            if getattr(current, 'acquisition_start_s', -math.inf) < sent:
                continue  # Do not renew on a frame straddling the first send.
            fit = measure(baseline, current)
            t = fit.transform
            emit('startup_pulse_live_motion', reliable=fit.reliable, dx_m=t.x_m,
                 dy_m=t.y_m, rotation_deg=math.degrees(t.yaw_rad), raw_range=fit.raw_range)
            if not fit.reliable:
                reason = 'unreliable_drive_feedback'
                break
            distance = math.hypot(t.x_m, t.y_m)
            if t.x_m >= .015 and first_progress is None:
                first_progress = clock()-sent
            if abs(t.yaw_rad) >= math.radians(12):
                reason = 'drive_heading_changed'
                break
            if distance >= .10 or t.x_m < -.015:
                reason = 'measured_local_step_complete' if t.x_m >= 0 else 'wrong_drive_direction'
                break
            check = assess(current, t)
            emit('startup_pulse_live_clearance', **check)
            if not check['safe']:
                reason = check['reason']
                break
            now = clock()
            # TTL also remains wholly within the overall breakaway deadline;
            # a blocked Python thread cannot keep the motor on past it.
            if (cancelled() or now >= expires
                    or now-current.received_s > .25):
                reason = 'lease_or_breakaway_deadline'
                break
            if now+ttl_ms/1000. > deadline:
                continue  # Observe the final lease; never extend past budget.
            if not renew(current, expires, deadline):  # STOP invalidates renewal.
                reason = 'drive_cancelled_before_renewal'
                break
            expires = now+ttl_ms/1000.
            renewals += 1
            emit('startup_drive_lease_renewed', sequence=sequence, ttl_ms=ttl_ms,
                 remaining_ms=round((deadline-now)*1000), background_refresh=False)
        return dict(reason=reason, elapsed_s=clock()-sent,
                    breakaway_s=first_progress, renewals=renewals)
    except Exception:
        reason = 'feedback_or_link_error'
        raise
    finally:
        stop()
        emit('startup_pulse_stopped', ttl_ms=ttl_ms, budget_ms=budget_ms,
             reason=reason, renewals=renewals, renewal_allowed=False)
