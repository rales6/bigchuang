"""Bounded scan odometry supported by reciprocal geometry, without a map fix.

The complete raw error distribution remains logged. A small group of long
range changes must not dominate the arithmetic mean, but accepting a bounded
estimate additionally requires multi-sector ray support and a reverse ICP
fit that closes the measured SE(2) motion. This grants no obstacle clearance.
"""
import math

from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence


def _usable(match):
    if match is None or not getattr(match, 'converged', False):
        return False
    t = match.transform
    return (all(math.isfinite(v) for v in
                (t.x_m, t.y_m, t.yaw_rad, match.rmse_m, match.inlier_ratio))
            and match.rmse_m <= .07 and match.inlier_ratio >= .55
            and abs(t.yaw_rad) <= math.radians(24))


def bounded_scan_odometry(localizer, before, after, motion, *, relative, raw=None):
    """Return evidence and extra uncertainty; retain the original fit metrics."""
    result = dict(accepted=False, reason='relative_fit_insufficient')
    if not _usable(motion):
        return result
    if raw is None:
        try:
            raw = turn_range_evidence(before, after, motion.transform,
                lidar_offset=localizer.lidar_offset, maximum_range=localizer.max_distance_m)
        except (AttributeError, TypeError, ValueError) as exc:
            return dict(result, reason='raw_scan_unavailable', error_type=type(exc).__name__)
    result['raw_range'] = raw
    for name in ('forward', 'reverse'):
        row = raw.get(name) or {}
        median, p80 = row.get('median_error_m'), row.get('p80_error_m')
        if (row.get('supported', 0) < 100 or row.get('overlap', 0.) < .75
                or row.get('agreement', 0.) < .80
                or row.get('supported_sectors', 0) < 4
                or row.get('bad_sectors', 8) > 1
                or median is None or p80 is None
                or not math.isfinite(median) or not math.isfinite(p80)
                or median > .10 or p80 > .20):
            return dict(result, reason='distributed_raw_evidence_insufficient')
    try:
        reverse = relative(after, before)
    except (AttributeError, TypeError, ValueError, RuntimeError, TimeoutError) as exc:
        return dict(result, reason='reverse_fit_unavailable', error_type=type(exc).__name__)
    if not _usable(reverse):
        return dict(result, reason='reverse_fit_insufficient')
    loop = motion.transform.compose(reverse.transform)
    shift, yaw = math.hypot(loop.x_m, loop.y_m), abs(loop.yaw_rad)
    result.update(reciprocal_translation_m=shift, reciprocal_yaw_deg=math.degrees(yaw),
                  reverse_rmse_m=reverse.rmse_m, reverse_inlier_ratio=reverse.inlier_ratio)
    if shift > .04 or yaw > math.radians(3):
        return dict(result, reason='reciprocal_motion_disagrees')
    return dict(result, accepted=True, reason='reciprocal_scan_odometry',
                extra_position_uncertainty_m=.03+shift,
                extra_yaw_uncertainty_deg=1.+math.degrees(yaw), map_fix=False)
