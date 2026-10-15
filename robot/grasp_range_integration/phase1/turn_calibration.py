"""Narrow empirical start-pulse allowance, not a firmware acceleration guess.

Source: supervised turn_calibration_20260927_152449/raw_rotation.json.
Right: -3500mrad/s, 180ms, gains (1.5,1.30435,1.17391,1.30435),
10/10 reciprocal scans, yaw -2.691..-3.231 degrees, operator saw movement.
Left: turn_calibration_left_20260927_153634, same output/TTL, yaw
+3.880..+4.594 degrees, 10/10 reciprocal scans, operator saw left movement.
Never extrapolate duration/speed/direction or authorize continuous renewal.
Actual post-stop angle/pose and swept-clearance interlocks remain mandatory.
"""
import math

SOURCE_RUN = "turn_calibration_20260927_152449"
LEFT_SOURCE_RUN = "turn_calibration_left_20260927_153634"
CALIBRATED_TTL_MS = 180


def source_run_for(speed):
    return LEFT_SOURCE_RUN if speed > 0 else SOURCE_RUN


def calibrated_start_ttl_ms(error_rad, speed, stalled, parent_ttl_ms,
                            measured_rate_rad_s=0.):
    # Highest tier reproduces the tested speed AND wheel output gains.
    # Direction-specific envelopes: right 8deg, left 10deg (over twice
    # largest stopped angles 3.231/4.594deg). Left also requires >=14deg.
    # This envelope is an estimate, not a guarantee across floors/batteries.
    minimum_deg, envelope_deg = (14, 10) if speed > 0 else (12, 8)
    if (not math.isfinite(error_rad) or abs(error_rad) < math.radians(minimum_deg)
            or error_rad*speed <= 0 or abs(speed) != 3500 or stalled < 6 or parent_ttl_ms < 180
            or not math.isfinite(measured_rate_rad_s) or measured_rate_rad_s < 0):
        return None
    budget = min(math.radians(20), abs(error_rad)-math.radians(2))
    # Faster real motion must shrink the next lease immediately. Retain a
    # 1.5x margin; never extrapolate beyond the measured 180ms duration.
    conservative_rate = max(math.radians(envelope_deg)/.180, measured_rate_rad_s*1.5)
    allowed = min(CALIBRATED_TTL_MS, int(budget/conservative_rate*1000))
    return allowed if allowed >= 50 else None
