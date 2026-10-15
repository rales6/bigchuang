"""Distinguish arrival inspection from task success; no hardware IO."""
import math


class ArrivalCoverageError(RuntimeError):
    """Request a bounded new station, not a fictitious frontal obstacle."""


def add_inspection_gate(assessment, pose, view):
    result = dict(assessment)
    xy = (view or {}).get("view_xy_m") or []
    near = (len(pose or []) == 3 and len(xy) == 2
            and all(math.isfinite(float(v)) for v in list(pose)+list(xy))
            and math.hypot(pose[0]-xy[0], pose[1]-xy[1]) <= .25)
    coverage = float(result.get("estimated_work_area_coverage") or 0)
    minimum = float(result.get("minimum_work_area_coverage") or .25)
    result["borderline_inspection_allowed"] = bool(
        result.get("eligible") is False and result.get("reason") == "approaching_task_region"
        and near and coverage >= max(.20, minimum*.8)
        and int(((view or {}).get("visibility_zone") or {}).get(
            "sweep_count", (view or {}).get("sweep_count") or 1)) == 1)
    return result


def arrival_action(route, pose, view, assess):
    """One diagnostic capture, then improve position if yaw alone won't help.

    Applies only to a semantic station, never single-camera object tracking.
    Returns a policy decision; the existing guarded executor sends commands.
    """
    if route.get("mode") != "semantic_visibility_regions":
        return {"action": "heading"}
    if int((view.get("visibility_zone") or {}).get("sweep_count",view.get("sweep_count") or 1)) != 1:
        return {"action":"heading"}  # Never skip required multi-angle sweeps.
    gate = assess(route, pose, view=view)
    yaw = float(view["target_yaw_rad"])
    error = math.atan2(math.sin(yaw-pose[2]), math.cos(yaw-pose[2]))
    if gate.get("eligible"):
        return {"action": "usable", "assessment": gate}
    if (gate.get("borderline_inspection_allowed") and abs(error) <= math.radians(15)
            and not view.get("arrival_diagnostic_done")):
        view["arrival_diagnostic_done"] = True
        return {"action": "diagnostic_capture", "assessment": gate}
    # Assess the SAME position at the desired heading. Do not spend time
    # rotating when recognition range/occlusion is the real limiting factor.
    ideal = assess(route, [pose[0], pose[1], yaw], view=view)
    if (gate.get("reason") == "approaching_task_region"
            and ideal.get("eligible") is False and ideal.get("reason") == "approaching_task_region"):
        xy = view["view_xy_m"]
        distance = math.hypot(pose[0]-xy[0], pose[1]-xy[1])
        if distance > .04:
            return {"action": "adjust_on_route", "assessment": gate, "distance_m": distance}
        raise ArrivalCoverageError("观察站实际视野仍不足，且已完成末段位置调整；保持停车，需要重新规划观察站")
    return {"action": "heading", "assessment": gate}


def linear_arrival_ttl(distance_m, speed_mm_s, original_ms):
    """Bound one approach pulse by remaining distance, including start kick.

    Not an odometry claim. Existing scan, frontal, glass, and swept-path
    checks must authorize it; no lease renewal or object-distance pursuit.
    """
    if not math.isfinite(distance_m) or distance_m <= .025 or speed_mm_s <= 0:
        raise ValueError("invalid arrival adjustment")
    ttl = int((distance_m-.025)/(speed_mm_s/1000)*1000)
    if ttl < 50:
        raise RuntimeError("末段调整剩余距离不足最短脉冲预算；禁止盲目前进")
    return min(int(original_ms), ttl, 350)
