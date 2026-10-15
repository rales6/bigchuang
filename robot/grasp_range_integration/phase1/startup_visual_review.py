"""Startup-only bounded visual review; protected DeepSeek source is untouched.

Whole-map matching remains available mid-mission. At a user-confirmed staging
area, visual review must not silently replace the prior with a remote basin.
"""

from copy import deepcopy
import math
from pathlib import Path
from grasp_range_integration.phase1.startup_prior import inside_start_prior


def pose_delta(a, b):
    return (math.hypot(a.x_m - b.x_m, a.y_m - b.y_m),
            abs(math.degrees(math.atan2(math.sin(a.yaw_rad - b.yaw_rad),
                                       math.cos(a.yaw_rad - b.yaw_rad)))))


def in_scope(result, anchor, prior):
    if result is None or not getattr(result, "reliable", False):
        return False
    p = result.pose
    if not all(math.isfinite(v) for v in (p.x_m, p.y_m, p.yaw_rad)):
        return False
    distance, angle = pose_delta(p, anchor.pose)
    return (inside_start_prior(p, prior)
            and distance <= .55 and angle <= 30.)


class ScopedStartupLocalizer:
    def __init__(self, original, anchor, prior, alternatives=()):
        self.original, self.anchor, self.prior = original, anchor, prior
        self.candidates = []
        for candidate in [anchor, *alternatives]:
            if not in_scope(candidate, anchor, prior):
                continue
            if any(pose_delta(candidate.pose, old.pose)[0] < .08
                   and pose_delta(candidate.pose, old.pose)[1] < 6
                   for old in self.candidates):
                continue
            self.candidates.append(candidate)

    def global_localization_candidates(self, scan, maximum=6):
        # The old checker calls this method. Deliberately do NOT delegate to
        # its global search or filter global results after that expensive call.
        return self.candidates[:max(1, int(maximum))]

    def localize_near_position(self, scan, hint, radius_m=.9):
        # No all-heading guided correction at startup either. Alternatives
        # already have lidar scores from the current stationary scan.
        ordered = sorted(self.candidates, key=lambda row: sum(pose_delta(row.pose, hint)))
        return ordered[0] if ordered else None

    def __getattr__(self, name):
        return getattr(self.original, name)


def normalize_descriptions(descriptions):
    """Do not turn uncertain map points/extended walls into exact left/right."""
    rows = deepcopy(descriptions)
    for description in rows:
        expected, context = [], list(description.get("context_only_structures", []))
        view = description.get("visibility_model", {})
        half_fov = float(view.get("dual_camera_horizontal_fov_deg", 82)) / 2
        margin = float(view.get("camera_bearing_margin_deg", 8))
        for item in description.get("expected_visible_structures", []):
            bearing = float(item.get("relative_bearing_deg", 0))
            uncertainty = float(item.get("bearing_uncertainty_deg", 0))
            if abs(bearing) > half_fov + margin:
                item["evidence_role"] = "context_only"
                item["startup_context_reason"] = "outside_nominal_camera_view"
                context.append(item)
                continue
            low, high = bearing - uncertainty, bearing + uncertainty
            allowed = []
            if low < -18:
                allowed.append("right")
            if low <= 18 and high >= -18:
                allowed.append("center")
            if high > 18:
                allowed.append("left")
            # A wall is an extended surface, not the reviewed centroid point.
            # Its presence is useful; its exact image side is not a point cue.
            if item.get("category") in {"glass_wall", "wall"}:
                allowed = ["left", "center", "right"]
            item["allowed_sides"] = allowed
            item["side"] = allowed[0] if len(allowed) == 1 else "uncertain"
            item["bearing_interval_deg"] = [round(low, 1), round(high, 1)]
            expected.append(item)
        description["expected_visible_structures"] = expected
        description["context_only_structures"] = context
        description["startup_review_policy"] = (
            "候选仅限已确认起点范围，不能按雷达分数猜选。先从真实图片独立辨认结构和方向，"
            "再比较候选；allowed_sides 是地图误差允许的方向。玻璃墙是有延展的面，"
            "不可把中心标注点当整面墙的左右位置。视野外的 context_only 不能证明匹配；"
            "至少两个不同稳定结构实际出现才可确认，否则返回 uncertain。")
    return rows


def audit_observations(diagnostics):
    result = diagnostics.get("deepseek_result") or {}
    selected = result.get("selected_candidate_id")
    description = next((row for row in diagnostics.get("candidate_descriptions", [])
                        if row.get("candidate_id") == selected), None)
    if description is None:
        return {"checked": False, "matched": [], "contradictions": []}
    expected = {row["name"]: row for row in description.get("expected_visible_structures", [])
                if row.get("evidence_role") == "stable_anchor"}
    matched, contradictions = set(), set()
    for observation in result.get("observed_structures", []):
        name = observation.get("name")
        item = expected.get(name)
        if item is None or float(observation.get("confidence") or 0) < .7:
            continue
        side = observation.get("side")
        if side in item.get("allowed_sides", []):
            matched.add(name)
        elif side in {"left", "center", "right"}:
            contradictions.add(name)
    return {"checked": True, "matched": sorted(matched),
            "contradictions": sorted(contradictions)}


def review_startup(checker, call_args, anchor, prior, alternatives=(), ds_module=None):
    """Wrap one real capture/rank call without changing global/midrun behavior."""
    args = list(call_args)
    scoped = ScopedStartupLocalizer(args[3], anchor, prior, alternatives)
    if not scoped.candidates:
        raise RuntimeError("启动视觉复核缺少范围内可靠雷达候选")
    args[3] = scoped
    original_descriptions = None
    if ds_module is not None:
        original_descriptions = ds_module.candidate_scene_descriptions
        ds_module.candidate_scene_descriptions = lambda *a, **kw: normalize_descriptions(
            original_descriptions(*a, **kw))
    try:
        status, visual, match, diagnostics = checker(*args)
    finally:
        if original_descriptions is not None:
            ds_module.candidate_scene_descriptions = original_descriptions
    diagnostics = dict(diagnostics)
    diagnostics["startup_candidate_scope"] = "confirmed_start_prior_and_anchor"
    diagnostics["startup_visual_is_global_localization"] = False
    diagnostics["startup_candidate_poses"] = [
        [row.pose.x_m, row.pose.y_m, math.degrees(row.pose.yaw_rad)] for row in scoped.candidates]
    if ds_module is not None:
        captures = diagnostics.get("capture") or []
        valid_pair = (diagnostics.get("camera_count") == 2 and len(captures) == 2
                      and Path(captures[0]).resolve() != Path(captures[1]).resolve()
                      and all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in captures))
        # API-test doubles may audit the pure normalization helpers separately;
        # deployed review must never treat one image copied twice as stereo.
        if not valid_pair:
            diagnostics["rejected_reason"] = "startup_missing_distinct_camera_pair"
            return "error", anchor, None, diagnostics
    if visual is None:
        diagnostics["rejected_reason"] = "startup_missing_visual_candidate"
        return "error", anchor, None, diagnostics
    if status in {"confirmed", "corrected"} and not in_scope(visual, anchor, prior):
        diagnostics["rejected_reason"] = "startup_visual_pose_out_of_scope"
        if visual is not None:
            diagnostics["startup_rejected_visual_pose"] = [
                visual.pose.x_m, visual.pose.y_m, math.degrees(visual.pose.yaw_rad)]
            distance, angle = pose_delta(visual.pose, anchor.pose)
            diagnostics["startup_rejected_delta"] = {"position_m": distance, "yaw_deg": angle}
        return "conflict", anchor, None, diagnostics
    if ds_module is not None and status in {"confirmed", "corrected"}:
        audit = audit_observations(diagnostics)
        diagnostics["startup_direction_audit"] = audit
        if audit["contradictions"]:
            diagnostics["rejected_reason"] = "startup_structure_direction_conflict"
            return "conflict", anchor, None, diagnostics
        if not audit["checked"] or len(audit["matched"]) < 2:
            # Do NOT turn a bad positive choice into a genuine DS abstention:
            # selected_candidate_id stays unchanged, so lidar-only gate fails.
            diagnostics["rejected_reason"] = "startup_insufficient_observed_anchors"
            return "uncertain", anchor, None, diagnostics
    return status, visual, match, diagnostics
