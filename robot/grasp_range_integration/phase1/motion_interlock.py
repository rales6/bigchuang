"""Validate scan evidence without assuming rotation has zero translation."""

import math


# Short scan-odometry navigation is constrained by growing position
# uncertainty and the existing live obstacle/glass envelopes. The former
# 12-second expiration stopped the 19:08 run while uncertainty was 0.233 m
# and only 0.28 m had been traversed in the bridge.
ESTIMATE_MAX_AGE_S = 30.
ESTIMATE_MAX_TRAVEL_M = 1.0
ESTIMATE_MAX_UNCERTAINTY_M = .35


def estimate_within_budget(estimate, now):
    """Continuous lidar estimates expire on evidence/uncertainty, not task age."""
    if not estimate:
        return False
    uncertainty = float(estimate.get('uncertainty_m', math.inf))
    if not math.isfinite(uncertainty) or not 0 <= uncertainty <= ESTIMATE_MAX_UNCERTAINTY_M:
        return False
    if estimate.get('policy') == 'continuous_lidar_odometry':
        return 0 <= now-estimate.get('last_evidence_s', -math.inf) <= .5
    return 0 <= now-estimate['started_s'] <= ESTIMATE_MAX_AGE_S


def estimate_command_horizon(estimate, now, speed_m_s):
    """Never reset uncertainty or extend commands past current sensor evidence."""
    if not estimate_within_budget(estimate, now) or speed_m_s <= 0:
        return 0.
    uncertainty_horizon = max(0., ESTIMATE_MAX_UNCERTAINTY_M-.02
                              -estimate.get('uncertainty_m', 0.))/(speed_m_s*.30)
    if estimate.get('policy') == 'continuous_lidar_odometry':
        return max(0., min(.5-(now-estimate['last_evidence_s']), uncertainty_horizon))
    return max(0., min(estimate['started_s']+ESTIMATE_MAX_AGE_S-now,
        (ESTIMATE_MAX_TRAVEL_M-estimate.get('distance_m', 0.))/speed_m_s, uncertainty_horizon))


def estimate_defer_reason(estimate, now, speed_m_s):
    enough_confidence = (ESTIMATE_MAX_UNCERTAINTY_M-.02-estimate.get('uncertainty_m', math.inf)
                         >= max(0., speed_m_s)*.30*.05)
    if (estimate.get('policy') == 'continuous_lidar_odometry' and enough_confidence
            and now-estimate.get('last_evidence_s', -math.inf) >= .45):
        return 'fresh_lidar_frame_required'
    return 'estimated_navigation_budget_exhausted'


def validate_turn_feedback(window, match):
    if not window.get("valid"):
        return "stale_or_unpaired_turn_scan"
    t = match.transform
    values = (t.x_m, t.y_m, t.yaw_rad, match.rmse_m, match.inlier_ratio)
    if not all(math.isfinite(v) for v in values):
        return "nonfinite_turn_feedback"
    if not match.converged or match.rmse_m > .09 or match.inlier_ratio < .65:
        return "unreliable_turn_feedback"
    # Retain angular feedback range, but consume measured pivot/slip
    # translation in full instead of rejecting it by a distance threshold.
    if abs(t.yaw_rad) > math.radians(24):
        return "turn_motion_outside_prediction_limit"
    return None


def predict_scan_motion(pose, match):
    """Phase-one SE(2) prediction; quality, not translation size, gates use.

    The frozen mission predictor rejects translation above 22cm. Keep its
    quality/angular contract here but apply the entire reliable transform.
    Neither clamping nor silently retaining the old pose is acceptable.
    """
    if match is None:
        return pose, False
    t = match.transform
    if (not all(math.isfinite(v) for v in
                (t.x_m, t.y_m, t.yaw_rad, match.rmse_m, match.inlier_ratio))
            or not match.converged or match.rmse_m > .16
            or match.inlier_ratio < .42
            or abs(t.yaw_rad) > math.radians(24)):
        return pose, False
    return pose.compose(t), True
