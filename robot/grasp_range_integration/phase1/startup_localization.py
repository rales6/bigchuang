"""Bounded first search, then independently score each stationary scan.

Uses existing map scoring/raycasting, never edits map or pose priors. Partial
work after a deadline is discarded. Visual candidate ranking stays intact.
"""

from dataclasses import replace
import math
import time

from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1.startup_prior import assess_start_prior, inside_start_prior


class StartupMatcher:
    def __init__(self, localizer, prior, *, search_budget_s=20., verify_budget_s=2.):
        if not 5 <= search_budget_s <= 45 or not .25 <= verify_budget_s <= 5:
            raise ValueError("启动搜索预算须为5–45秒，局部验证预算须为0.25–5秒")
        self.localizer, self.prior = localizer, prior
        self.search_budget_s, self.verify_budget_s = search_budget_s, verify_budget_s
        self.hint = None
        self.searches = 0
        self.diagnostic = {}
        self.visual_candidates = []
        # Supplied only during semantic-search startup. It filters the pose
        # lattice BEFORE radar ranking; never moves a pose to a free cell.
        self.start_mask = getattr(localizer, '_phase1_start_candidate_mask', None)

    def match(self, scan, overall_deadline):
        mode = "bounded_start_search" if self.hint is None else "stationary_local_verify"
        started = time.monotonic()
        budget = self.search_budget_s if self.hint is None else self.verify_budget_s
        deadline = min(overall_deadline, started + budget)
        self.diagnostic = {"mode": mode, "budget_s": budget, "result": "searching"}
        self.visual_candidates = []
        try:
            if self.hint is None:
                if self.searches >= 2:
                    self.diagnostic["result"] = "initial_search_limit"
                    return None
                self.searches += 1
            # Compatibility for non-scoring test/API doubles only. Production
            # ScanMapLocalizer exposes the bounded scoring API below.
            if not hasattr(self.localizer, "_rank"):
                result = (self.localizer.track(scan, self.hint)
                          if self.hint is not None and hasattr(self.localizer, "track")
                          else self.localizer.global_localize(scan, hint=None))
            else:
                result = self._score(scan, deadline)
            if result is not None:
                self.diagnostic.update(
                    pose=[result.pose.x_m, result.pose.y_m, math.degrees(result.pose.yaw_rad)],
                    prior_check=assess_start_prior(result.pose, self.prior),
                    rmse_m=getattr(result, "rmse_m", None),
                    inlier_ratio=getattr(result, "inlier_ratio", None))
            if time.monotonic() >= deadline:
                raise TimeoutError("startup scoring deadline exceeded")
            if result is None or not result.reliable:
                self.diagnostic["result"] = "no_reliable_match"
                return None
            p = result.pose
            if not inside_start_prior(p, self.prior):
                self.diagnostic["result"] = "outside_start_prior"
                return None
            if self.hint is not None:
                angle = math.atan2(math.sin(p.yaw_rad - self.hint.yaw_rad),
                                   math.cos(p.yaw_rad - self.hint.yaw_rad))
                if math.hypot(p.x_m - self.hint.x_m, p.y_m - self.hint.y_m) > .12 or abs(angle) > math.radians(6):
                    self.diagnostic["result"] = "stationary_pose_changed"
                    return None
            self.hint = result.pose
            self.diagnostic["result"] = "accepted_candidate"
            return result
        except TimeoutError:
            self.diagnostic["result"] = "budget_exhausted"
            return None
        finally:
            self.diagnostic["elapsed_s"] = round(time.monotonic() - started, 3)

    def _score(self, scan, deadline):
        localizer, prior = self.localizer, self.prior
        points = localizer.points(scan, maximum=220)
        profile = localizer.range_profile(scan, maximum=36)
        if len(points) < 80:
            return None
        def bounded_lattice(center, **kwargs):
            raw = localizer._pose_lattice(center, **kwargs)
            poses = [pose for pose in raw if inside_start_prior(pose, prior)]
            if self.start_mask is not None and poses:
                cells = localizer.map.world_to_grid([(p.x_m, p.y_m) for p in poses])
                allowed = [p for p, (c, r) in zip(poses, cells)
                    if localizer.map.in_bounds(c, r) and self.start_mask[r, c]]
                self.diagnostic['excluded_no_route_poses'] = (
                    self.diagnostic.get('excluded_no_route_poses', 0) + len(poses)-len(allowed))
                poses = allowed
            self.diagnostic["excluded_lattice_poses"] = (
                self.diagnostic.get("excluded_lattice_poses", 0) + len(raw) - len(poses))
            return poses
        if self.hint is None:
            center = Pose2D(prior.x_m, prior.y_m, math.radians(prior.yaw_deg))
            coarse = localizer._rank(points, bounded_lattice(
                center, translation_radius_m=prior.radius_m, translation_step_m=.22,
                yaw_radius_deg=prior.yaw_tolerance_deg, yaw_step_deg=10), keep=6,
                range_profile=profile, range_shortlist_min=72, deadline_s=deadline)
            refined = []
            for _, pose, _ in coarse:
                refined.extend(bounded_lattice(
                    pose, translation_radius_m=.18, translation_step_m=.06,
                    yaw_radius_deg=10, yaw_step_deg=2))
            ranked = localizer._rank(points, refined, keep=12, range_profile=profile,
                                     range_shortlist_min=72, deadline_s=deadline)
            result = localizer._result_from_ranked(ranked)
        else:
            candidates = bounded_lattice(
                self.hint, translation_radius_m=.08, translation_step_m=.04,
                yaw_radius_deg=4, yaw_step_deg=2)
            ranked = localizer._rank(points, candidates, keep=4, range_profile=profile,
                                     range_shortlist_min=12, deadline_s=deadline)
            result = localizer._result_from_ranked(ranked, ambiguity=False)
        # These are scored alternatives from this scan, not whole-map poses.
        # The visual adapter applies the start prior and anchor-jump gates too.
        self.visual_candidates = [localizer._result_from_ranked([row], ambiguity=False)
                                  for row in ranked]
        # Preserve the old constrained-prior ambiguity contract, only after
        # the existing reliable gate has passed.
        if result is not None and result.reliable:
            return replace(result, ambiguity_margin=max(result.ambiguity_margin, .20))
        return None
