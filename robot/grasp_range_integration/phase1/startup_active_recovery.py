"""Move for new localization evidence, never promote an uncertain pose to navigation.

Unknown rear space is supported by a free-map disk covering the ENTIRE user
start envelope, not by labelling a weak lidar candidate as a strong map fix.
"""
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1.rotation_guard import (
    assess_rotation, map_rotation_evidence, installed_body_mask, body_supported_blind_arcs)
from grasp_range_integration.phase1.glass_boundary_guard import (
    swept_glass_clearance, bounded_recovery_glass_sweep)
from grasp_range_integration.phase1.startup_localization import StartupMatcher
from grasp_range_integration.phase1.startup_visual_review import review_startup, pose_delta
from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence
from grasp_range_integration.phase1.motion_interlock import predict_scan_motion
from grasp_range_integration.phase1.motion_evidence import bounded_scan_odometry
from grasp_range_integration.phase1.startup_exploration import ExplorationPolicy, scan_geometry
from grasp_range_integration.phase1.search_handoff import absolute_map_range_evidence


def motor_exposure_s(pulse, ttl_ms):
    """Upper bound issued motor leases, excluding scan/clearance waiting."""
    data = pulse if isinstance(pulse, dict) else {}
    if data.get('command_sent') is False or data.get('status') == 'deferred':
        return 0.
    try:
        renewals = max(0, int(data.get('renewals', 0)))
        elapsed = float(data.get('elapsed_s', ttl_ms/1000.))
        if not math.isfinite(elapsed):
            return ttl_ms/1000.
        return max(0., min(elapsed, (renewals+1)*ttl_ms/1000.))
    except (ValueError, TypeError, OverflowError):
        return ttl_ms/1000.


@dataclass(frozen=True)
class RecoverySettings:
    phase: str = 'startup'
    time_budget_s: float = 180.
    heading_tolerance_deg: float = 180.
    operator_prompt: bool = True
    initial_fit: bool = True
    visual_attempt_base: int = 100
    fresh_start_handoff: bool = False


@dataclass(frozen=True)
class RecoveryEvidence:
    candidate: object
    scan: object
    predicted_pose: Pose2D


def assess_recovery_action(scan, known_map, prior, linear, angular, ttl_ms, *,
                           operator_clear=False, clock=time.monotonic, directional_translation=False,
                           dispatch_margin_s=0.):
    """Check bounded motion from fresh raw ranges and a conservative glass envelope."""
    pose = Pose2D(prior.x_m, prior.y_m, math.radians(prior.yaw_deg))
    age = clock()-getattr(scan, 'received_s', -math.inf)
    if not 0 <= age <= .3:
        return dict(safe=False, reason='startup_recovery_stale_scan')
    # Extra 15cm covers the commanded short advance and stopping uncertainty;
    # it is a clearance margin, not a permitted measured-slip threshold.
    # Midrun pure rotation has no commanded forward travel. Its physical
    # disk, measured pose uncertainty, boundary margin and 3cm local drift
    # allowance still apply; the 15cm FORWARD stopping reserve is not added.
    reserve = .03 if directional_translation and angular and not linear else .15
    if not math.isfinite(dispatch_margin_s) or not 0 <= dispatch_margin_s <= .2:
        raise ValueError('invalid dispatch latency margin')
    travel = abs(linear)/1000.*(max(0., age)+ttl_ms/1000.+.20+dispatch_margin_s)
    radius = .44+prior.radius_m+reserve+travel
    bounded_heading = directional_translation and 0 <= prior.yaw_tolerance_deg <= 45
    forward_only = bool(bounded_heading and linear > 0 and not angular)
    # The forward branch never uses a rear rotation disk to grant motion.
    # Computing that unused full-map disk on every 150ms probe made fresh
    # frames expire before the command could be checked/renewed.
    envelope = (dict(clear=None, reason='forward_uses_live_sweep_and_bounded_glass') if forward_only else
                map_rotation_evidence(known_map, pose, radius,
                    exact_glass_geometry=bool(bounded_heading and angular and not linear)))
    # Heading is not yet confirmed: enclose motion in ANY heading, rather
    # than testing only a trajectory along the center of the heading prior.
    if bounded_heading:
        glass = bounded_recovery_glass_sweep(known_map,pose,linear,angular,
            uncertainty_m=prior.radius_m,heading_tolerance_deg=prior.yaw_tolerance_deg,
            pose_age_s=age,ttl_ms=ttl_ms,reserve_m=reserve,
            dispatch_margin_s=dispatch_margin_s)
    else:
        glass = swept_glass_clearance(known_map, pose, 0, 0, reliable=True,
                                      pose_age_s=age, ttl_ms=ttl_ms, radius_m=radius)
    if not glass['safe']:
        return dict(safe=False, reason='startup_glass_clearance', glass=glass)
    if angular:
        result = assess_rotation(scan, known_map=known_map, pose=pose,
                                 map_pose_reliable=True,
                                  map_pose_uncertainty_m=prior.radius_m+reserve,
                                  exact_glass_geometry=bool(bounded_heading and not linear))
        angles, ranges = np.asarray(scan.angles_rad), np.asarray(scan.distances_m)
        body = installed_body_mask(angles, ranges)
        valid = np.isfinite(angles) & np.isfinite(ranges) & (ranges > 0) & (ranges <= 8) & ~body
        bearings = np.sort(angles[valid] % (2*math.pi))
        explained = False
        if len(bearings) >= 60:
            _, rear_supported = body_supported_blind_arcs(angles, body, bearings)
            gap = np.max(np.diff(np.r_[bearings, bearings[0]+2*math.pi]))
            explained = gap <= math.radians(45) or rear_supported
        # A supervised startup probe can use explicit clearance confirmation
        # for chassis-occluded space, as calibration probes do. NEVER override
        # live external near returns, glass or an unexplained/cropped scan gap.
        if (not result['safe'] and operator_clear
                and result['reason'] in ('installation_edge_occlusion_without_safe_map_support',
                                        'unobserved_sector_without_safe_map_support')
                and explained
                and result.get('min_distance_m', 0) >= .60
                and result.get('exterior_count', 0) >= 60):
            result.update(safe=True, reason='operator_confirmed_startup_probe_clearance')
    else:
        angles, ranges = np.asarray(scan.angles_rad), np.asarray(scan.distances_m)
        valid = np.isfinite(angles) & np.isfinite(ranges) & (ranges > 0)
        front = valid & (np.abs((angles+math.pi)%(2*math.pi)-math.pi) <= math.radians(50))
        count = int(np.count_nonzero(front))
        nearest = float(np.min(ranges[front])) if count else None
        observed = np.sort(((angles[front]+math.pi)%(2*math.pi)-math.pi))
        covered = (count >= 16 and observed[0] <= -math.radians(40)
                   and observed[-1] >= math.radians(40)
                   and np.max(np.diff(observed)) <= math.radians(10))
        required_front = max(.60, .44+reserve+travel)
        clear = covered and nearest >= required_front
        result = dict(safe=bool(clear),
                      reason='front_clear' if clear else 'front_not_clear',
                      front_count=count, nearest_front_m=nearest,
                      required_front_m=required_front)
        if directional_translation:
            # Midrun heading is bounded by measured odometry. A side return
            # inside a wide angular window is not a frontal wall: check its
            # ACTUAL distance to the straight center path, including the
            # installed +14cm lidar offset and the unchanged 44+15cm reserve.
            # Require observed +/-85deg coverage; missing side beams are NOT
            # clearance. This authorizes FORWARD translation, never rotation
            # or reversing into the chassis-occluded rear sector.
            wrapped = (angles+math.pi)%(2*math.pi)-math.pi
            seen = np.sort(wrapped[valid & (np.abs(wrapped) <= math.radians(90))])
            max_gap = float(np.max(np.diff(seen))) if len(seen) >= 2 else math.inf
            covered = (len(seen) >= 60 and seen[0] <= -math.radians(85)+1e-9
                       and seen[-1] >= math.radians(85)-1e-9
                       and max_gap <= math.radians(10)+1e-9)
            exterior = (valid & (ranges <= 8) & ~installed_body_mask(angles,ranges)
                        & (np.abs(wrapped) <= math.pi/2))
            xx, yy = ranges[exterior]*np.cos(angles[exterior])+.14, ranges[exterior]*np.sin(angles[exterior])
            distance = np.hypot(xx-np.clip(xx,0.,travel),yy)
            nearest_sweep = float(distance.min()) if len(distance) else 0.
            clear = covered and nearest_sweep >= .44+reserve and linear > 0
            reason = ('observed_forward_sweep_clear' if clear else
                      'forward_scan_coverage_incomplete' if not covered else
                      'observed_forward_sweep_blocked')
            result.update(safe=bool(clear), reason=reason, coverage_front_degrees=170,
                          coverage_complete=bool(covered), coverage_samples=len(seen),
                          coverage_min_deg=math.degrees(seen[0]) if len(seen) else None,
                          coverage_max_deg=math.degrees(seen[-1]) if len(seen) else None,
                          coverage_max_gap_deg=math.degrees(max_gap) if math.isfinite(max_gap) else None,
                          nearest_sweep_distance_m=nearest_sweep, sweep_radius_m=.44+reserve,
                          translation_heading_bounded=True, rear_turn_authorized=False)
    # For a short forward probe, the actual front corridor can be observed
    # directly. Rear occlusion is not a reason to request manual clearance of
    # the entire start disk. Continuous glass clearance remains mandatory.
    if angular and not envelope['clear'] and not operator_clear:
        result.update(safe=False, reason='startup_envelope_requires_operator_clearance')
    return dict(result, startup_envelope=envelope, glass=glass, operator_clear=operator_clear,
                authority=('bounded_pose_glass_envelope_and_raw_forward_sweep' if directional_translation and linear
                           else 'whole_user_start_envelope_not_weak_candidate'))


def propagate_prior(prior, transform):
    center = Pose2D(prior.x_m, prior.y_m, math.radians(prior.yaw_deg)).compose(transform)
    # A body-frame displacement rotates differently at each possible heading.
    # Include that uncertainty rather than shifting all hypotheses identically.
    expansion = 2*math.hypot(transform.x_m, transform.y_m)*math.sin(
        math.radians(min(180., prior.yaw_tolerance_deg))/2)
    return SimpleNamespace(x_m=center.x_m, y_m=center.y_m,
                           yaw_deg=math.degrees(center.yaw_rad),
                           radius_m=prior.radius_m+expansion,
                           yaw_tolerance_deg=prior.yaw_tolerance_deg)


def _strong_borderline_midrun_drive(motion, evidence):
    """Accept a 1-2 cm raw-range mean excess only with stronger independent fit.

    This applies solely to an already stopped, short forward recovery move;
    rotation, startup exploration and obstacle clearance keep their gates.
    """
    t = motion.transform
    if (not motion.converged or motion.rmse_m > .04 or motion.inlier_ratio < .85
            or not all(math.isfinite(v) for v in
                       (t.x_m,t.y_m,t.yaw_rad,motion.rmse_m,motion.inlier_ratio))
            or not 0 <= t.x_m <= .12 or abs(t.y_m) > .04
            or abs(t.yaw_rad) > math.radians(3)):
        return False
    for name in ('forward','reverse'):
        row = evidence.get(name) or {}
        mean = row.get('mean_error_m')
        if (row.get('supported',0) < 100 or row.get('overlap',0) < .80
                or row.get('agreement',0) < .85
                or row.get('supported_sectors',0) < 4
                or row.get('bad_sectors',1) != 0
                or mean is None or not math.isfinite(mean) or mean > .18):
            return False
    return True


def recover_startup(mission, args, client, scans, localizer, controller, stopping,
                    prior, *, camera, recognizer, gpu, directory, emit,
                    confirm=None, clock=time.monotonic, settings=None, reference_scan=None,
                    accept_candidate=None):
    """Bounded evidence-seeking phase; never drive a mission on a guessed start."""
    if not callable(getattr(client, 'startup_recovery_pulse', None)):
        raise RuntimeError('启动主动定位缺少受保护短动作接口；未授权运动')
    settings = settings or RecoverySettings()
    if (settings.phase not in ('startup', 'midrun') or not 30 <= settings.time_budget_s <= 180
            or not 10 <= settings.heading_tolerance_deg <= 180):
        raise ValueError('invalid localization recovery settings')
    phase_label = '途中' if settings.phase == 'midrun' else '启动'
    callback = emit or (lambda *a, **k: None)
    def emit(name, **fields):
        if settings.phase == 'midrun' and name.startswith('startup_'):
            name = 'midrun_'+name[len('startup_'):]
        callback(name, **fields)
    deadline = clock()+settings.time_budget_s
    # A candidate repeatedly clipped to the old 40deg edge is not evidence
    # that its true heading lies there. Exploration preserves the position
    # envelope with a full-heading startup search; midrun uses a measured,
    # bounded heading neighborhood instead of the original staging heading.
    current_prior = SimpleNamespace(x_m=prior.x_m, y_m=prior.y_m, yaw_deg=prior.yaw_deg,
                                   radius_m=prior.radius_m,
                                   yaw_tolerance_deg=settings.heading_tolerance_deg)
    ds_module = None
    if getattr(mission, '_phase1_deepseek_enabled', False):
        import deepseek_visual_localization as ds_module
    drive_experience = getattr(mission, '_phase1_startup_drive_experience', None)
    policy = ExplorationPolicy(drive_experience.ready() if drive_experience is not None else None)
    if not settings.initial_fit:
        policy.fitted()  # The failed stationary view has already been checked.
    operator_clear = False
    confirmation_requested = False
    last_progress = clock()
    index = 0
    anchor_scan = reference_scan
    pending_feedback = None
    pending_visual_attempted = False
    last_blocked_fit = -math.inf
    blocked_frames = 0
    next_drive_ttl_ms = None
    stalled_motor_s = 0.
    feedback_margin = {'position_m': 0., 'yaw_deg': 0.}
    emit('startup_active_recovery', status='started', time_budget_s=settings.time_budget_s,
         purpose='localization_only', target_search=False)

    def verified_motion(before, after, *, short_midrun_drive=False):
        feedback_margin.update(position_m=0., yaw_deg=0.)
        motion = localizer.relative_motion(before, after)
        evidence = turn_range_evidence(before, after, motion.transform,
            lidar_offset=localizer.lidar_offset, maximum_range=localizer.max_distance_m)
        _, usable = predict_scan_motion(Pose2D(), motion)
        borderline = (short_midrun_drive and settings.phase == 'midrun'
                      and not evidence['accepted']
                      and _strong_borderline_midrun_drive(motion,evidence))
        strict = (usable and motion.rmse_m <= .09 and motion.inlier_ratio >= .65
                  and (evidence['accepted'] or borderline))
        corroboration = ({'accepted': False, 'reason': 'strict_evidence_available'} if strict else
            bounded_scan_odometry(localizer, before, after, motion,
                                   relative=localizer.relative_motion, raw=evidence))
        emit('startup_recovery_feedback', rotation_deg=math.degrees(motion.transform.yaw_rad),
             dx_m=motion.transform.x_m, dy_m=motion.transform.y_m,
             rmse_m=motion.rmse_m, inlier_ratio=motion.inlier_ratio,
             raw_range=evidence, usable=usable,
             strong_borderline_drive=borderline, corroboration=corroboration)
        if not strict and not corroboration['accepted']:
            emit('startup_recovery_feedback_pending', terminal=False,
                 reason=corroboration['reason'], next_action='fresh_scan_or_stationary_map_fit')
            return None
        if corroboration['accepted']:
            feedback_margin.update(position_m=corroboration['extra_position_uncertainty_m'],
                                   yaw_deg=corroboration['extra_yaw_uncertainty_deg'])
        return motion

    def apply_motion(base, motion):
        updated = propagate_prior(base, motion.transform)
        return SimpleNamespace(x_m=updated.x_m, y_m=updated.y_m, yaw_deg=updated.yaw_deg,
            radius_m=updated.radius_m+feedback_margin['position_m'],
            yaw_tolerance_deg=min(180., updated.yaw_tolerance_deg+feedback_margin['yaw_deg']))

    def defer_motion(command=None, pulse=None):
        """Keep a search envelope and the original scan until motion is resolved.

        Unverified displacement is never replaced by zero or used to issue
        another move. New scans can prove relative motion without a map fix;
        otherwise a stationary map consensus must re-anchor this episode.
        """
        nonlocal current_prior, pending_feedback, pending_visual_attempted
        if pending_feedback is not None:
            return
        origin = current_prior
        if command is None:
            travel, yaw = .25, 45.
        else:
            linear, angular, ttl = command
            elapsed = max(ttl/1000., float((pulse or {}).get('elapsed_s', ttl/1000.)))+.2
            yaw = min(180., math.degrees(abs(angular)/1000.*elapsed*1.5)+15.)
            travel = abs(linear)/1000.*elapsed+.10
            if angular:
                travel += .44*math.radians(yaw)
        pending_feedback = dict(prior=origin, command=command, pulse=pulse)
        pending_visual_attempted = False
        current_prior = SimpleNamespace(x_m=origin.x_m, y_m=origin.y_m, yaw_deg=origin.yaw_deg,
            radius_m=origin.radius_m+travel,
            yaw_tolerance_deg=min(180., origin.yaw_tolerance_deg+yaw))
        emit('startup_recovery_feedback_search', terminal=False,
             radius_m=current_prior.radius_m, yaw_tolerance_deg=current_prior.yaw_tolerance_deg,
             motion_enabled=False, original_motion_baseline_retained=True)

    def finish(candidate, scan, matcher, *, visual_anchor=None):
        nonlocal current_prior, anchor_scan, pending_feedback
        if stopping() or clock() >= deadline:
            raise RuntimeError('换视角恢复已取消或时间预算耗尽；已停车')
        if settings.phase == 'midrun':
            center = Pose2D(current_prior.x_m, current_prior.y_m, math.radians(current_prior.yaw_deg))
            distance, angle = pose_delta(candidate.pose, center)
            # Midrun drift is already enclosed by the measured prior. Do not
            # reject a fresh consensus inside that envelope at an unrelated
            # 20cm cutoff, then enlarge the exploration disk again.
            position_limit = (current_prior.radius_m if pending_feedback is not None else
                              min(.35, max(.20, current_prior.radius_m)))
            angle_limit = current_prior.yaw_tolerance_deg if pending_feedback is not None else 10
            if (distance > position_limit or angle > angle_limit
                    or candidate.rmse_m > .09 or candidate.inlier_ratio < .65):
                emit('startup_recovery_correction_rejected', distance_m=distance, angle_deg=angle)
                return None
            def absolute_anchor_ok(row, frame):
                evidence = absolute_map_range_evidence(localizer,row,frame,deadline,clock=clock)
                emit('startup_recovery_absolute_range_anchor', **evidence)
                if not evidence['accepted']:
                    emit('startup_recovery_map_anchor_pending', terminal=False, **evidence)
                return evidence['accepted']
            if pending_feedback is not None and visual_anchor is None and not absolute_anchor_ok(candidate,scan):
                return None
            # Photos/map scoring can be slow. Re-verify on a REAL new scan;
            # never hand navigation a stale pre-DS frame or a made-up timestamp.
            if not 0 <= clock()-getattr(scan, 'received_s', -math.inf) <= .5:
                new = next(scans)
                while getattr(new, 'acquisition_start_s', -math.inf) <= scan.received_s:
                    if stopping() or clock() >= deadline:
                        raise RuntimeError('途中恢复缺少定位后的新扫描；已停车')
                    new = next(scans)
                previous_candidate = candidate
                if pending_feedback is None:
                    motion = verified_motion(scan, new)
                    if motion is None:
                        return None
                    current_prior = apply_motion(current_prior, motion)
                    anchor_scan = new
                    matcher.hint = candidate.pose.compose(motion.transform)
                else:
                    # A fresh independent absolute fit resolves the unknown
                    # old transform; no synthetic zero-motion update needed.
                    matcher.hint = candidate.pose
                matcher.prior = current_prior
                candidate = matcher.match(new, deadline)
                if (candidate is None or not candidate.reliable or candidate.rmse_m > .09
                        or candidate.inlier_ratio < .65):
                    return None
                scan = new
                if pending_feedback is not None:
                    refresh_distance, refresh_angle = pose_delta(candidate.pose,previous_candidate.pose)
                    if refresh_distance > .12 or refresh_angle > 6:
                        return None
                center = Pose2D(current_prior.x_m, current_prior.y_m, math.radians(current_prior.yaw_deg))
                distance, angle = pose_delta(candidate.pose, center)
                if distance > position_limit or angle > angle_limit or not 0 <= clock()-scan.received_s <= .5:
                    return None
                if visual_anchor is not None:
                    visual_distance, visual_angle = pose_delta(candidate.pose, visual_anchor)
                    if visual_distance > .12 or visual_angle > 6:
                        return None
                if pending_feedback is not None and visual_anchor is None and not absolute_anchor_ok(candidate,scan):
                    return None
            if accept_candidate is not None and not accept_candidate(candidate):
                if pending_feedback is not None:
                    # Localization and route availability are different gates.
                    # Retain a proven absolute fix for guarded exploration even
                    # if this map-mask cell has no navigation path yet. Drop
                    # only the unresolved-command expansion; retain the prior
                    # uncertainty and the four-scan consensus tolerance.
                    origin = pending_feedback['prior']
                    current_prior = SimpleNamespace(x_m=candidate.pose.x_m,
                        y_m=candidate.pose.y_m, yaw_deg=math.degrees(candidate.pose.yaw_rad),
                        radius_m=max(.22,origin.radius_m),
                        yaw_tolerance_deg=max(9.,origin.yaw_tolerance_deg))
                    anchor_scan = scan
                    pending_feedback = None
                    emit('startup_recovery_feedback_resolved',
                         source='stationary_absolute_anchor_route_pending',
                         transform_was_not_assumed_zero=True,
                         localization_retained=True, mission_navigation_enabled=False,
                         sequence=getattr(scan,'sequence',None),
                         pose=[candidate.pose.x_m,candidate.pose.y_m,candidate.pose.yaw_rad],
                         radius_m=current_prior.radius_m)
                emit('startup_recovery_route_pending', terminal=False,
                     pose=[candidate.pose.x_m, candidate.pose.y_m, candidate.pose.yaw_rad],
                     next_action='new_safe_view_and_route_check')
                return None
            if pending_feedback is not None:
                emit('startup_recovery_feedback_resolved', source='stationary_map_and_visual_anchor'
                     if visual_anchor is not None else 'stationary_map_consensus',
                     transform_was_not_assumed_zero=True)
                return RecoveryEvidence(candidate, scan, candidate.pose)
            return RecoveryEvidence(candidate, scan, center)
        if settings.fresh_start_handoff:
            # A carry-stage XY hint is from BEFORE visual pursuit, not a
            # measured motion anchor. Keep motors stopped and independently
            # score a real post-fit scan; never invent a zero-motion transform
            # from that old hint or enforce a compulsory drive/turn count.
            anchor = candidate.pose
            matcher.hint = anchor
            for _ in range(3):
                if stopping() or clock() >= deadline:
                    raise RuntimeError('搬运定位交接已取消或预算耗尽；已停车')
                new = next(scans)
                while getattr(new, 'acquisition_start_s', -math.inf) <= scan.received_s:
                    if stopping() or clock() >= deadline:
                        raise RuntimeError('搬运定位缺少复核后的新扫描；已停车')
                    new = next(scans)
                refreshed = matcher.match(new, deadline)
                if refreshed is None or not refreshed.reliable:
                    return None
                distance, angle = pose_delta(refreshed.pose, anchor)
                if (refreshed.rmse_m > .10 or refreshed.inlier_ratio < .70
                        or distance > .12 or angle > 6):
                    emit('startup_carry_handoff_pending', terminal=False,
                         reason='post_fit_scan_changed', distance_m=distance, angle_deg=angle)
                    return None
                candidate, scan = refreshed, new
                if 0 <= clock()-scan.received_s <= .5:
                    break
            else:
                emit('startup_carry_handoff_pending', terminal=False,
                     reason='post_fit_scan_stale')
                return None
            if stopping() or clock() >= deadline:
                raise RuntimeError('搬运定位交接已取消或预算耗尽；已停车')
            emit('startup_carry_handoff_verified', sequence=getattr(scan, 'sequence', None),
                 pose=[candidate.pose.x_m, candidate.pose.y_m, candidate.pose.yaw_rad],
                 required_validation_moves=0)
        if accept_candidate is not None and not accept_candidate(candidate):
            return None
        return candidate.pose, scan

    def try_fit(first_scan, *, allow_vision=True):
        """Only stationary, fresh four-scan consensus may release navigation."""
        nonlocal pending_visual_attempted
        policy.fitted()
        matcher = StartupMatcher(localizer, current_prior)
        results = []
        for sample in range(4):
            if stopping() or clock() >= deadline:
                raise RuntimeError(f'{phase_label}主动定位已取消或证据恢复预算耗尽；已停车')
            scan = first_scan if sample == 0 else next(scans)
            result = matcher.match(scan, deadline)
            emit('startup_recovery_scan', attempt=index, **matcher.diagnostic)
            results.append(result)
        ready, candidate, ratio, accepted, total = mission._localization_acceptance(results, args)
        if not ready or candidate is None or accepted < 4:
            return None
        strong_inlier = .65 if settings.phase == 'midrun' else .70
        strong_rmse = .09 if settings.phase == 'midrun' else .10
        if (candidate.reliable and candidate.rmse_m <= strong_rmse
                and candidate.inlier_ratio >= strong_inlier
                and (not (settings.phase == 'midrun' or settings.fresh_start_handoff) or
                     all(row is not None and row.reliable and row.rmse_m <= strong_rmse
                         and row.inlier_ratio >= strong_inlier for row in results))):
            outcome = finish(candidate, scan, matcher)
            if outcome is not None:
                emit('startup_active_recovery', status='localized', basis='fresh_strong_lidar',
                     pose=[candidate.pose.x_m, candidate.pose.y_m, candidate.pose.yaw_rad])
                return outcome
            if pending_feedback is None:
                return None  # Route/pose rejection is not a reason to re-rank the same photo.
            # A good four-frame map fit can still fail the raw-motion anchor.
            # Do not return before scoped vision gets to rank those MEASURED
            # candidates (153617 repeatedly fitted 59 times without DS).
        if not allow_vision:
            return None
        if pending_feedback is not None:
            if pending_visual_attempted:
                return None
            pending_visual_attempted = True
        try:
            status, winner, fit, diagnostics = review_startup(
                mission._visual_localization_check,
                (args, candidate, scan, localizer, camera, recognizer, gpu, directory,
                 settings.visual_attempt_base+policy.fit_attempts),
                candidate, current_prior, matcher.visual_candidates, ds_module)
        except Exception as exc:
            emit('startup_recovery_visual', status='error', attempt=index,
                 error_type=type(exc).__name__, error=str(exc), next_action='new_safe_view')
            return None
        emit('startup_recovery_visual', status=status, attempt=index, **diagnostics)
        if winner is not None:
            distance, angle = pose_delta(winner.pose, candidate.pose)
            from grasp_range_integration.phase1.startup_prior import inside_start_prior
            if (status in ('confirmed', 'corrected') and fit is not None
                    and inside_start_prior(winner.pose, current_prior)
                    and distance <= .55 and angle <= 30):
                outcome = finish(winner, scan, matcher, visual_anchor=winner.pose)
                if outcome is not None:
                    emit('startup_active_recovery', status='localized', basis='lidar_and_visual')
                return outcome
        return None

    while clock() < deadline:
        index += 1
        if stopping() or clock() >= deadline:
            raise RuntimeError(f'{phase_label}探索已取消或证据恢复时间预算耗尽；已停车')
        client.stop()
        stopped = clock()
        before = next(scans)
        while getattr(before, 'acquisition_start_s', -math.inf) < stopped+.05:
            if stopping() or clock() >= deadline:
                reason = '已取消' if stopping() else '证据恢复预算耗尽，运动反馈仍未确认'
                raise RuntimeError(f'{phase_label}主动定位{reason}；已停车')
            before = next(scans)
        if anchor_scan is not None:
            alignment = verified_motion(anchor_scan, before)
            if alignment is None:
                defer_motion()
                result = try_fit(before)
                if result is not None:
                    return result
                continue  # Retain original baseline; no new movement on an unknown transform.
            origin = pending_feedback['prior'] if pending_feedback is not None else current_prior
            current_prior = apply_motion(origin, alignment)
            if pending_feedback is not None:
                command = pending_feedback['command']
                if command is not None:
                    policy.observe(alignment.transform, command)
                pending_feedback = None
                last_progress = clock()
                emit('startup_recovery_feedback_resolved', source='new_relative_scan_evidence',
                     original_motion_applied_once=True)
        anchor_scan = before
        before_geometry = scan_geometry(before)
        worthwhile = (policy.wants_fit(before_geometry) or
                      (settings.fresh_start_handoff and settings.initial_fit
                       and policy.fit_attempts == 0))
        emit('startup_scan_geometry', attempt=index, **before_geometry, localization_requested=worthwhile,
             since_fit_distance_m=policy.since_fit_distance,
             since_fit_angle_deg=math.degrees(policy.since_fit_angle), map_pose_confirmed=False,
             before_motion=True, initial_full_heading_check=policy.fit_attempts == 0)
        if worthwhile:
            result = try_fit(before)
            if result is not None:
                return result
            # Scoring/vision time is not a motor stall. Reacquire a fresh scan
            # before authorizing movement; do not reuse this now old frame.
            last_progress = clock()
            continue
        options = []
        # A pre-send veto only shortens the NEXT selection, not every future
        # lease. Successful 150ms probes used to pin recovery at 150ms forever.
        frame_drive_ttl_limit = next_drive_ttl_ms
        next_drive_ttl_ms = None
        actions, directions = policy.candidates(before)
        emit('startup_exploration_plan', attempt=index, directions=directions,
             geometry=before_geometry, committed_direction=policy.direction,
             committed_arc_deg=math.degrees(policy.arc), low_progress=dict(policy.low_progress))
        for linear, angular, ttl in actions:
            if linear > 0 and angular == 0 and frame_drive_ttl_limit is not None:
                ttl = min(ttl, frame_drive_ttl_limit)
            assessment = assess_recovery_action(before, localizer.map, current_prior,
                                                 linear, angular, ttl,
                                                 operator_clear=operator_clear, clock=clock,
                                                 directional_translation=settings.phase == 'midrun',
                                                 dispatch_margin_s=.05 if settings.phase == 'midrun' else 0.)
            emit('startup_recovery_clearance', linear_mm_s=linear,
                 angular_mrad_s=angular, ttl_ms=ttl, **assessment)
            if assessment['safe']:
                options.append((linear, angular, ttl))
                # Only options[0] is ever executed. Scoring the unused
                # opposite turn afterwards needlessly ages the approved scan.
                break
        if (not options and settings.phase == 'midrun'
                and pending_feedback is None
                and any(linear > 0 and angular == 0 for linear, angular, _ttl in actions)):
            # The ordinary 400 ms breakaway lease can make the conservative
            # glass horizon fail even though a shorter forward sweep is clear.
            # Do not turn into the lidar's body-occluded rear sector or use a
            # weak map pose as free-space evidence. Try a fresh-scan-guarded,
            # forward-only probe whose lease is checked again before renewal.
            for ttl in (250, 200, 150):
                if frame_drive_ttl_limit is not None and ttl > frame_drive_ttl_limit:
                    continue
                assessment = assess_recovery_action(
                    before, localizer.map, current_prior, 400, 0, ttl,
                    operator_clear=False, clock=clock, directional_translation=True,
                    dispatch_margin_s=.05)
                emit('startup_recovery_clearance', linear_mm_s=400,
                     angular_mrad_s=0, ttl_ms=ttl, recovery='short_forward_probe',
                     **assessment)
                if assessment['safe']:
                    options.append((400, 0, ttl))
                    emit('startup_recovery_short_probe_selected', ttl_ms=ttl,
                         reason='full_forward_lease_glass_horizon_blocked',
                         raw_forward_sweep_required=True, glass_guard_required=True)
                    break
        if not options:
            if not settings.operator_prompt:
                # A short real new view may already be useful even when
                # there is no safe NEXT move. Spend one stationary check on
                # it, not a blind drive merely to reach the 12cm/15deg target.
                changed = policy.since_fit_distance >= .015 or abs(policy.since_fit_angle) >= math.radians(4)
                if changed and before_geometry['informative']:
                    emit('startup_recovery_last_safe_view', stationary_only=True)
                    result = try_fit(before)
                    if result is not None:
                        return result
                    last_progress = clock()
                    continue
                if settings.phase == 'midrun':
                    # A single incomplete/blocked frame is not a terminal
                    # mission result. No command is sent on it. Reacquire raw
                    # scans and retry bounded *lidar-only* consensus; DS
                    # abstention cannot force another identical photo loop.
                    blocked_frames += 1
                    emit('startup_recovery_waiting_for_scan', terminal=False,
                         blocked_frames=blocked_frames,
                         next_action='fresh_scan_and_bounded_lidar_consensus',
                         motion_enabled=False, original_search_bounds_retained=True)
                    if before_geometry['informative'] and clock()-last_blocked_fit >= 2.:
                        last_blocked_fit = clock()
                        result = try_fit(before, allow_vision=False)
                        if result is not None:
                            return result
                    continue
                raise RuntimeError('途中换视角无安全动作：实时净空、后方盲区或玻璃包络未通过；已停车，未盲目绕行')
            if not confirmation_requested:
                confirmation_requested = True
                emit('startup_recovery_operator_required', status='waiting',
                     reason='uncertain_pose_and_body_occlusion', motion_enabled=False)
                if confirm is None:
                    import sys
                    if not sys.stdin.isatty():
                        raise RuntimeError('启动恢复等待现场净空确认：请在交互终端运行脚本，确认小车含机械臂四周1米及前方已清空、远离玻璃且可急停；未移动')
                    answer = input('启动位置不确定，地图不能证明后方净空。请清空小车含机械臂四周至少1米及前方通道，现场看护可急停。确认后输入“已清空”继续短动作定位，其他输入取消：').strip()
                    operator_clear = answer == '已清空'
                else:
                    operator_clear = bool(confirm())
                # Waiting for the operator consumes no motor lease and is
                # not charged to the evidence-acquisition time budget.
                deadline = clock()+settings.time_budget_s
                last_progress = clock()
                emit('startup_recovery_operator_confirmation', confirmed=operator_clear)
                if operator_clear and not stopping():
                    continue  # Acquire a NEW stopped scan after the wait.
            raise RuntimeError('启动主动定位无安全换视角动作：起点范围、玻璃或实时净空未通过；已停车，请清空周围或重新摆放起点')
        command = options[0]
        blocked_frames = 0
        if command[1] and command[1]*policy.direction < 0:
            policy.direction, policy.arc = -policy.direction, 0.
        drive_budget_ms = (900 if settings.phase == 'midrun' and command[0] and command[2] < 400
                           else policy.continuous_drive_budget())
        emit('startup_recovery_motion', attempt=index, command=list(command),
             drive_budget_ms=drive_budget_ms if command[0] else None,
             mission_navigation_enabled=False)
        pulse = client.startup_recovery_pulse(before, current_prior, localizer.map, *command,
            operator_clear=operator_clear, localizer=localizer,
            drive_budget_ms=drive_budget_ms, stopping=stopping,
            recovery_phase=settings.phase)
        if isinstance(pulse, dict) and pulse.get('status') == 'deferred' and pulse.get('command_sent') is False:
            # Only an explicit pre-send veto can take this path. Do NOT catch
            # arbitrary transport failures or discard feedback after motion.
            # Keep the old scan/prior; next iteration measures any real drift
            # and reselects on a complete new scan, never the rejected frame.
            if settings.phase == 'midrun' and command[0] > 0 and command[1] == 0:
                shorter = next((ttl for ttl in (250, 200, 150) if ttl < command[2]), command[2])
                next_drive_ttl_ms = shorter
            emit('startup_recovery_dispatch_deferred', terminal=False,
                 reason=(pulse.get('assessment') or {}).get('reason','pre_send_veto'),
                 command=list(command), command_sent=False,
                 next_forward_ttl_ms=next_drive_ttl_ms,
                 next_action='fresh_scan_reselect_or_static_localization',
                 original_motion_baseline_retained=True)
            continue
        stopped = clock()
        after = next(scans)
        while getattr(after, 'acquisition_start_s', -math.inf) < stopped+.05:
            if stopping() or clock() >= deadline:
                raise RuntimeError(f'{phase_label}主动定位缺少动作后的新扫描；已停车')
            after = next(scans)
        motion = verified_motion(before, after,
                                 short_midrun_drive=bool(command[0]))
        if motion is None:
            defer_motion(command, pulse)
            anchor_scan = before
            continue
        if drive_experience is not None:
            drive_experience.record(command, pulse, motion.transform, True)
        moved = abs(motion.transform.yaw_rad) > math.radians(.5) or math.hypot(
            motion.transform.x_m, motion.transform.y_m) > .005
        policy.observe(motion.transform, command)
        if moved:
            last_progress = clock()
            stalled_motor_s = 0.
        else:
            # Waiting for scans, clearance, map fitting and DS is NOT motor
            # stall time. Charge only confirmed issued lease exposure, not
            # the 30s wall-clock interval surrounding one 150ms pulse.
            stalled_motor_s += motor_exposure_s(pulse, command[2])
            emit('startup_recovery_low_progress', stalled_motor_s=round(stalled_motor_s,3),
                 waiting_time_charged=False, next_action='fresh_scan_then_adaptive_reselection')
            if stalled_motor_s > 30:
                raise RuntimeError(f'{phase_label}探索累计实际下发运动30秒仍没有实测运动进展；已停车，请检查供电/轮子/底盘链路')
        current_prior = apply_motion(current_prior, motion)
        anchor_scan = after
        emit('startup_recovery_prior', x_m=current_prior.x_m, y_m=current_prior.y_m,
             yaw_deg=current_prior.yaw_deg, radius_m=current_prior.radius_m)
        # Complete map scoring is performed at the top of the next stopped
        # iteration, only if informative evidence and net novelty justify it.
    raise RuntimeError(f'{phase_label}探索时间预算耗尽，仍没有可验证的定位；已停车，请检查雷达辨识度/地图，不执行猜测导航')
