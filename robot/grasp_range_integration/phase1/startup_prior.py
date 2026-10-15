"""Single strict, inclusive start envelope for scoring, acceptance and vision."""

import math


def assess_start_prior(pose, prior):
    values = (pose.x_m, pose.y_m, pose.yaw_rad, prior.x_m, prior.y_m,
              prior.yaw_deg, prior.radius_m, prior.yaw_tolerance_deg)
    if not all(math.isfinite(float(v)) for v in values):
        return {"inside": False, "reason": "non_finite_start_pose"}
    distance = math.hypot(pose.x_m - prior.x_m, pose.y_m - prior.y_m)
    delta = pose.yaw_rad - math.radians(prior.yaw_deg)
    angle = abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))
    position_ok = distance <= prior.radius_m + 1e-9
    yaw_ok = angle <= prior.yaw_tolerance_deg + 1e-9
    return {"inside": position_ok and yaw_ok,
            "reason": ("inside_start_prior" if position_ok and yaw_ok else
                       "outside_start_position" if not position_ok else "outside_start_heading"),
            "position_error_m": distance, "yaw_error_deg": angle,
            "position_limit_m": prior.radius_m, "yaw_limit_deg": prior.yaw_tolerance_deg}


def inside_start_prior(pose, prior):
    return assess_start_prior(pose, prior)["inside"]
