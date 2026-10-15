"""Refresh clearance after slow map matching without rewriting odometry."""

import math
import time


def effective_scan(context, scan=None):
    scan = context.get("scan") if scan is None else scan
    pair = context.get("clearance_pair")
    return pair[1] if pair is not None and scan is pair[0] else scan


def ensure_navigation_freshness(context, refresh, emit, *, clock=time.monotonic):
    """A stale computation is a bounded zero-speed retry, not mission failure.

    refresh must ACK STOP, get a complete post-STOP revolution, and verify
    reference-to-new motion (ICP AND reciprocal raw beams) within 3cm/2deg.
    It reads independently: the mission's odometry baseline is NOT consumed.
    Neither scan timestamps nor the absolute map pose are fabricated.
    """
    now = clock()
    latest = effective_scan(context)
    latest_received = getattr(latest, "received_s", None)
    latest_age = now-latest_received if latest_received is not None else float("inf")
    pose_age = now-context.get("updated", 0.)
    if context.get("reliable") and 0 <= latest_age <= .5 and 0 <= pose_age <= .5:
        return True
    # Always compare to the scan behind the absolute map fit. Chaining tiny
    # motions between successive refreshed scans could accumulate large drift.
    reference = context.get("scan")
    received = getattr(reference, "received_s", None)
    age = now-received if received is not None else float("inf")

    def wait(reason, **details):
        context["freshness_wait_reason"] = reason
        emit("navigation_freshness_wait", reason=reason,
             reference_age_s=age if math.isfinite(age) else None,
             odometry_baseline_unchanged=True, **details)
        return False

    if not context.get("reliable"):
        return wait("map_pose_requires_normal_relocalization")
    if not math.isfinite(age) or not 0 <= age <= 1.5:
        return wait("missing_or_overage_reference")
    if refresh is None:
        return wait("fresh_scan_service_unavailable")
    try:
        pair = refresh(reference)
    except Exception as exc:
        return wait("post_stop_scan_unavailable", error=str(exc))
    if pair is None:
        return wait("fresh_scan_service_unavailable")
    current, small_motion = pair
    if not small_motion:
        # Main loop must apply actual motion and obtain a new absolute fit.
        # Do NOT label the old pose as fresh, nor paint an imaginary obstacle.
        context["reliable"] = False
        return wait("vehicle_moved_or_motion_unreliable")
    end = getattr(current, "received_s", None)
    start = getattr(current, "acquisition_start_s", None)
    now = clock()
    if (current is reference or end is None or start is None
            or not all(math.isfinite(v) for v in (end, start))
            or not received < start <= end or not 0 <= now-end <= .5):
        return wait("invalid_or_stale_post_stop_scan")
    # The absolute map fit remains unchanged; a validated tiny-motion bridge
    # extends its evidence only to this real scan receipt time. Extra geometry
    # margin includes the allowed 3cm position / 2deg heading discrepancy.
    source = context.get("scan")
    context["clearance_pair"] = (source, current)
    context["updated"] = end
    context["bridge_uncertainty_m"] = context.get("bridge_uncertainty_m", 0.) + .05
    context.setdefault("quality", {}).update(post_stop_small_motion=True,
        navigation_stationary_bridge=True)
    context.pop("freshness_wait_reason", None)
    emit("navigation_freshness_refreshed", reference_age_s=age,
         current_age_s=now-end, sequence=getattr(current, "sequence", None),
         stationary_pose_supported=True, extra_geometry_margin_m=.05,
         odometry_baseline_unchanged=True)
    return True
