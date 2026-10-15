"""Recent, direction-specific turn response; one feedback-window lease at a time.

These are engineering control budgets, not a guarantee about tire friction.
The live adapter must stop on stale/unsafe scans and validate post-STOP motion.
"""
from dataclasses import dataclass
import math
import time


@dataclass
class DirectionResponse:
    rate: float = 0.
    ttl_ms: int = 70
    speed: int = 1200
    low_progress: int = 0
    last_progress_s: float | None = None
    recent_samples: int = 0
    output_stage: int = 0
    effective_speed: int = 0
    effective_ttl_ms: int = 0
    effective_stage: int = 0
    effective_rate_rad_s: float = 0.


class AdaptiveTurnResponse:
    def __init__(self, emit=lambda *a, **k: None, clock=time.monotonic):
        self.emit, self.clock = emit, clock
        self.directions = {1: DirectionResponse(), -1: DirectionResponse()}
        self.correction_scale = 1.
        self.active_direction = None

    def feedback(self, speed, ttl_ms, rotation, translation, reliable, error):
        if not speed or not reliable:
            return
        if not all(math.isfinite(v) for v in (rotation, translation, ttl_ms)):
            return
        row = self.directions[1 if speed > 0 else -1]
        row.ttl_ms, row.speed = int(ttl_ms), abs(int(speed))
        # Reliable SE(2) feedback includes pivot translation; angular response
        # remains useful even when the tires slip sideways.
        if not 50 <= ttl_ms <= 650:
            return
        if row.last_progress_s is None:
            row.last_progress_s = self.clock()
        required = max(.25, min(2., abs(math.degrees(error or 0.))*.08))
        meaningful = rotation*speed > 0 and abs(math.degrees(rotation)) >= required
        if meaningful:
            measured = abs(rotation)/(ttl_ms/1000.)
            row.rate = measured if not row.recent_samples else .5*row.rate+.5*measured
            row.recent_samples += 1
            # Retain this direction's JUST PROVEN breakaway command. A small
            # successful turn is not evidence that the old 1800/116ms floor
            # suddenly works on the same patch of floor.
            if abs(math.degrees(error or 0.)) >= (14 if speed > 0 else 12) and abs(rotation) <= abs(error):
                row.effective_speed = abs(int(speed))
                row.effective_ttl_ms = int(ttl_ms)
                row.effective_stage = max(2, row.output_stage)
                row.effective_rate_rad_s = measured
            row.low_progress = 0
            row.output_stage = 0
            row.last_progress_s = self.clock()
        else:
            row.low_progress += 1
            # Reliable failures replace a stale fastest-response assumption;
            # do not change the opposite direction or the archived evidence.
            row.rate *= .65
        self.emit('turn_recent_response', direction='left' if speed > 0 else 'right',
                  recent_rate_rad_s=row.rate, low_progress=row.low_progress,
                  meaningful_progress=meaningful, actual_ttl_ms=ttl_ms)
        # No attempt-count exit: a mechanical/communication failure still
        # cannot energize stalled motors indefinitely.
        if self.clock()-row.last_progress_s > 30. and row.low_progress:
            raise RuntimeError('转向持续30秒无有效进展；已逐步加力加时，需检查电源、轮子或底盘链路，禁止无限堵转')

    def reverse_correction(self):
        self.correction_scale = max(.25, self.correction_scale*.65)
        self.emit('turn_overshoot_correction', output_scale=self.correction_scale,
                  reverse_allowed=True)

    def plan(self, error, requested_speed, parent_ttl_ms, *, learned=None):
        direction = 1 if error > 0 else -1
        row = self.directions[direction]
        if not math.isfinite(error) or not 50 <= parent_ttl_ms <= 700:
            raise ValueError('Invalid adaptive turn budget')
        large = abs(math.degrees(error)) >= (14 if direction > 0 else 12)
        # At most one live feedback window (250ms), or the supervised 180ms
        # start duration for small corrections. NEVER renewable full-power.
        ceiling = min(int(parent_ttl_ms), 250 if large else 180)
        if direction != self.active_direction:
            # A new reversal is not a thirty-second stall inherited from
            # the opposite direction. Keep response samples, soften output.
            self.active_direction = direction
            row.last_progress_s = self.clock()
        reused = bool(learned and large and row.low_progress == 0)
        if reused:
            row.output_stage = 6  # Reproduce learned wheel gains, not speed alone.
        if row.low_progress:
            speed = min(3500, max(1800, row.speed)+250)
            ttl = min(ceiling, row.ttl_ms+40*min(3,row.low_progress))
            reason = 'current_low_progress_extend'
        elif large and row.effective_speed and self.correction_scale == 1.:
            speed = min(3500, row.effective_speed)
            budget = max(math.radians(1), min(math.radians(12), abs(error)-math.radians(2)))
            measured_rate = max(row.rate, row.effective_rate_rad_s)*1.5
            ttl = min(row.effective_ttl_ms, int(budget/max(measured_rate,.1)*1000.))
            row.output_stage = max(row.output_stage, row.effective_stage)
            reason = 'retain_proven_direction_breakaway'
        else:
            speed = 3500 if reused and large else min(1800, max(1200,abs(requested_speed)))
            rate = row.rate or (float(learned['fastest_rate_rad_s']) if reused else 0.)
            budget = max(math.radians(1), min(math.radians(12), abs(error)-math.radians(2)))
            # A slow floor sample must not authorize 180 ms at a reused
            # 3500 mrad/s motor command near a 19-degree goal. Estimate a
            # fast plausible response from the ACTUAL commanded speed; a
            # short pulse that does not break traction is extended by the
            # low-progress feedback branch on the next fresh scan.
            fast_rate = max(rate*1.5, abs(speed)/1000.*.75)
            ttl = (min(180, int(budget/max(fast_rate,.1)*1000.)) if reused
                   else min(70 if abs(math.degrees(error)) <= 35 else 140,
                            int(budget/max(rate*1.5,abs(speed)/1000.)*1000.)))
            speed = max(650, int(speed*self.correction_scale))
            ttl = int(ttl*self.correction_scale)
            reason = 'recent_response_precision'
        ttl = max(50, min(ceiling, ttl))
        row.output_stage = max(row.output_stage, 2+row.low_progress)
        self.emit('turn_dynamic_plan', direction='left' if direction > 0 else 'right',
                  angular_mrad_s=direction*speed, ttl_ms=ttl,
                  reason=reason, low_progress=row.low_progress,
                  recent_rate_rad_s=row.rate, inherited=reused,
                  live_feedback_required=True)
        return direction*speed, ttl, row.output_stage, reused
