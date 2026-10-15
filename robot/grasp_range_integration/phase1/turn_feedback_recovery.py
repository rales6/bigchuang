"""Independent feedback recovery for ambiguous and unexpectedly large turns.

Never turn a weak ICP fit into motion permission just by lowering its inlier
threshold. Quality-ambiguous turns use stopped evidence; an unusually large
but strong turn needs its acknowledged motor pulse and independent raw/map
checks before its full motion can enter normal tracking.
"""
from dataclasses import replace
import inspect
import math
import time

from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.motion_interlock import validate_turn_feedback
from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence
from grasp_range_integration.phase1.motion_evidence import bounded_scan_odometry


def _difference(left, right):
    return (math.hypot(left.x_m-right.x_m, left.y_m-right.y_m),
            abs(math.atan2(math.sin(left.yaw_rad-right.yaw_rad),
                           math.cos(left.yaw_rad-right.yaw_rad))))


def _match_good(match, *, inlier=.65, max_yaw_deg=24.):
    if match is None or not getattr(match, 'converged', False):
        return False
    t = match.transform
    return (all(math.isfinite(v) for v in
                (t.x_m,t.y_m,t.yaw_rad,match.rmse_m,match.inlier_ratio))
            and match.rmse_m <= .09 and match.inlier_ratio >= inlier
            and (max_yaw_deg is None or abs(t.yaw_rad) <= math.radians(max_yaw_deg)))


def _raw_good(localizer, before, after, transform):
    return turn_range_evidence(before, after, transform,
        lidar_offset=localizer.lidar_offset,
        maximum_range=localizer.max_distance_m)['accepted']


def _map_good(candidate, hint):
    if candidate is None or getattr(candidate,'score',-1) == -1:
        return False
    position, yaw = _difference(candidate.pose, hint)
    agreement = getattr(candidate,'range_agreement',None)
    coverage = getattr(candidate,'range_coverage',None)
    return (candidate.reliable and candidate.rmse_m <= .09
            and candidate.inlier_ratio >= .65
            and agreement is not None and agreement >= .70
            and coverage is not None and coverage >= .50
            and position <= .12 and yaw <= math.radians(7))


def recover_ambiguous_turn(before, after, first, window, localizer, *,
                           relative, read_scan, stop, anchor=None,
                           emit=lambda *a,**kw:None, clock=time.monotonic):
    """Return a proven alternate transform or None; send no motor command.

    This helper verifies quality-ambiguous turns. Missing/stale pulse evidence,
    nonfinite motion or larger increments return to the caller's recovery
    state; this helper neither authorizes movement nor terminates a mission.
    """
    def report(status, **fields):
        emit('turn_feedback_recovery', status=status, **fields)

    if (not window.get('valid') or not _match_good(first,inlier=.42)):
        report('ineligible_original_evidence')
        return None
    evidence = bounded_scan_odometry(localizer, before, after, first, relative=relative)
    if evidence['accepted']:
        report('bounded_relative_odometry', **evidence)
        return first
    pivot = getattr(localizer,'relative_turn_motion',None)
    if callable(pivot):
        try:
            # The deployed protected API has only two positional arguments;
            # the newer local copy also accepts yaw_range_deg. Inspect the
            # signature instead of swallowing a TypeError as failed evidence.
            parameters = inspect.signature(pivot).parameters
            accepts_yaw = ('yaw_range_deg' in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))
            candidate = (pivot(before,after,yaw_range_deg=24.) if accepts_yaw
                         else pivot(before,after))
            delta,yaw = _difference(first.transform,candidate.transform)
            if (validate_turn_feedback(window,candidate) is None
                    and delta <= .07 and yaw <= math.radians(5)
                    and _raw_good(localizer,before,after,candidate.transform)):
                report('pivot_raw_verified', translation_disagreement_m=delta,
                       heading_disagreement_deg=math.degrees(yaw))
                return candidate
            report('pivot_not_correlated', translation_disagreement_m=delta,
                   heading_disagreement_deg=math.degrees(yaw))
        except (AttributeError,TypeError,ValueError,RuntimeError) as exc:
            report('pivot_unavailable', error_type=type(exc).__name__)

    # The prior turn lease is over. Explicit STOP ACK and a *new full scan*
    # are still required before trying another estimator.
    stop()
    stopped = clock()
    deadline = stopped + 1.5
    sequence = getattr(after,'sequence',None)
    if sequence is None:
        report('missing_scan_sequence')
        return None
    fresh = None
    while clock() < deadline:
        try:
            current = read_scan(sequence,min(.3,max(.01,deadline-clock())))
        except (RuntimeError,TimeoutError,StopIteration):
            break
        sequence = current.sequence
        if (current.acquisition_start_s >= stopped+.05
                and 0 <= clock()-current.received_s <= .3):
            fresh = current
            break
    if fresh is None:
        report('no_post_stop_full_scan')
        return None
    try:
        stationary = relative(after,fresh)
        stationary_distance, stationary_yaw = _difference(stationary.transform,
                                                           type(stationary.transform)())
        if (not _match_good(stationary,inlier=.70)
                or stationary_distance > .03 or stationary_yaw > math.radians(2)
                or not _raw_good(localizer,after,fresh,stationary.transform)):
            report('post_stop_motion_unverified',translation_m=stationary_distance,
                   heading_deg=math.degrees(stationary_yaw))
            return None
        corroborated = relative(before,fresh)
        delta,yaw = _difference(first.transform,corroborated.transform)
        # The mission still owns `after` as its current scan. Remove the
        # independently measured after->fresh residual before handing back
        # a before->after transform, otherwise that motion is counted twice.
        rebased = replace(corroborated,
                          transform=corroborated.transform.compose(stationary.transform.inverse()))
        if (validate_turn_feedback(window,corroborated) is None
                and delta <= .06 and yaw <= math.radians(5)
                and _raw_good(localizer,before,fresh,corroborated.transform)):
            report('second_scan_raw_verified',translation_disagreement_m=delta,
                   heading_disagreement_deg=math.degrees(yaw),
                   residual_stationary_translation_m=stationary_distance)
            return rebased
        evidence = bounded_scan_odometry(localizer, before, fresh, corroborated,
                                         relative=relative)
        if evidence['accepted'] and delta <= .06 and yaw <= math.radians(5):
            report('second_scan_bounded_relative_odometry', **evidence,
                   translation_disagreement_m=delta, heading_disagreement_deg=math.degrees(yaw),
                   rebased_to_after_sequence=getattr(after,'sequence',None))
            return rebased
        if anchor is None or anchor[0] is not before:
            report('relative_evidence_pending', map_anchor_available=False,
                   inlier_ratio=corroborated.inlier_ratio, evidence=evidence,
                   next_action='fresh_scan_and_bounded_localization')
            return None
        bounded = getattr(localizer,'localize_bounded',None)
        if not callable(bounded):
            report('bounded_map_recheck_unavailable')
            return None
        old_pose = anchor[1]
        hint = old_pose.compose(first.transform)
        on_after = bounded(after,hint,radius_m=.20,budget_s=.35,yaw_radius_deg=8)
        if not _map_good(on_after,hint):
            report('first_map_recheck_unverified')
            return None
        on_fresh = bounded(fresh,hint,radius_m=.20,budget_s=.35,yaw_radius_deg=8)
        if not _map_good(on_fresh,hint):
            report('second_map_recheck_unverified')
            return None
        map_shift,map_yaw = _difference(on_after.pose,on_fresh.pose)
        derived = old_pose.inverse().compose(on_after.pose)
        delta,yaw = _difference(first.transform,derived)
        if (map_shift > .05 or map_yaw > math.radians(3)
                or delta > .12 or yaw > math.radians(7)
                or abs(derived.yaw_rad) > math.radians(24)):
            report('map_pair_disagrees_with_motion',map_shift_m=map_shift,
                   map_yaw_deg=math.degrees(map_yaw),
                   motion_shift_m=delta,motion_yaw_deg=math.degrees(yaw))
            return None
        result = MatchResult(derived,max(on_after.rmse_m,on_fresh.rmse_m),
            getattr(first,'correspondences',0),True,
            min(on_after.inlier_ratio,on_fresh.inlier_ratio))
        report('two_map_frames_verified',map_shift_m=map_shift,
               map_yaw_deg=math.degrees(map_yaw),
               motion_shift_m=delta,motion_yaw_deg=math.degrees(yaw),
               residual_stationary_translation_m=stationary_distance)
        return result
    except (AttributeError,TypeError,ValueError,RuntimeError,TimeoutError) as exc:
        report('independent_recheck_error',error_type=type(exc).__name__,error=str(exc))
        return None


def recover_large_turn(before, after, first, window, localizer, *, intent,
                       relative, read_scan, stop, anchor=None,
                       emit=lambda *a,**kw:None, clock=time.monotonic):
    """Re-anchor a real turn that exceeded the old fixed 24-degree scan limit.

    The motor's acknowledged speed and duration bound plausible rotation;
    raw beams and a strong absolute map fit establish where the robot ended.
    If the immediate map fit fails, STOP and independently check a full new
    stationary scan. This never sends another motion command.
    """
    def report(status, **fields):
        emit('turn_feedback_recovery', status=status, **fields)

    if not window.get('valid') or not _match_good(first, inlier=.65, max_yaw_deg=None):
        report('large_turn_invalid_feedback')
        return None
    if anchor is None or anchor[0] is not before:
        report('large_turn_no_exact_map_anchor')
        return None
    if not isinstance(intent, dict):
        report('large_turn_no_acknowledged_pulse')
        return None
    try:
        speed = float(intent['angular_mrad_s'])
        ttl_s = float(intent['ttl_ms']) / 1000.
        multiplier = float(intent.get('output_multiplier', 1.))
        recorded = float(intent['recorded_s'])
        sent = float(window['sent_s'])
        if (not all(math.isfinite(v) for v in (speed, ttl_s, multiplier, recorded, sent))
                or not .05 <= ttl_s <= .4 or not 0 < abs(speed) <= 3500
                or not 1. <= multiplier <= 1.5
                or not sent-.15 <= recorded <= sent+1.0):
            report('large_turn_pulse_unpaired')
            return None
        observed = first.transform.yaw_rad
        # The limit is derived from this pulse, not a fixed per-scan angle.
        # The margin accommodates wheel-output boost and floor-dependent slip.
        plausible = abs(speed) / 1000. * ttl_s * multiplier * 1.35 + math.radians(6)
        if observed*speed <= 0 or abs(observed) > plausible:
            report('large_turn_outside_acknowledged_pulse',
                   observed_deg=math.degrees(observed),
                   pulse_bound_deg=math.degrees(plausible))
            return None
        bounded = getattr(localizer, 'localize_bounded', None)
        if not callable(bounded):
            report('large_turn_map_recheck_unavailable')
            return None
        old_pose = anchor[1]
        hint = old_pose.compose(first.transform)

        def map_motion(candidate, scan):
            if not _map_good(candidate, hint):
                return None
            derived = old_pose.inverse().compose(candidate.pose)
            shift, yaw = _difference(first.transform, derived)
            if (shift > .12 or yaw > math.radians(7)
                    or derived.yaw_rad*speed <= 0 or abs(derived.yaw_rad) > plausible
                    or not _raw_good(localizer, before, scan, derived)):
                return None
            return MatchResult(derived, candidate.rmse_m,
                               first.correspondences, True, candidate.inlier_ratio)

        on_after = bounded(after, hint, radius_m=.20, budget_s=.45,
                           yaw_radius_deg=10.)
        recovered = map_motion(on_after, after)
        if recovered is not None:
            report('large_turn_map_reanchored',
                   observed_deg=math.degrees(observed),
                   map_deg=math.degrees(recovered.transform.yaw_rad),
                   full_translation_m=math.hypot(recovered.transform.x_m,
                                                 recovered.transform.y_m))
            return recovered

        # An absolute map correction may time out in a feature-poor corridor.
        # Two independent raw-beam directions plus the strong relative fit and
        # a verified pre-turn map anchor still yield a bounded current pose.
        # The ordinary tracking/short odometry bridge must continue seeking a
        # fresh absolute map fix; this does not promote a weak ICP result.
        if _raw_good(localizer, before, after, first.transform):
            report('large_turn_bounded_scan_odometry',
                   observed_deg=math.degrees(observed),
                   map_recheck='not_confirmed',
                   full_translation_m=math.hypot(first.transform.x_m,
                                                 first.transform.y_m))
            return first

        stop()
        stopped = clock()
        deadline = stopped + 1.5
        sequence = getattr(after, 'sequence', None)
        if sequence is None:
            report('large_turn_missing_scan_sequence')
            return None
        fresh = None
        while clock() < deadline:
            try:
                current = read_scan(sequence, min(.3, max(.01, deadline-clock())))
            except (RuntimeError, TimeoutError, StopIteration):
                break
            sequence = current.sequence
            if (current.acquisition_start_s >= stopped+.05
                    and 0 <= clock()-current.received_s <= .3):
                fresh = current
                break
        if fresh is None:
            report('large_turn_no_post_stop_scan')
            return None
        stationary = relative(after, fresh)
        stationary_distance, stationary_yaw = _difference(
            stationary.transform, type(stationary.transform)())
        if (not _match_good(stationary, inlier=.70)
                or stationary_distance > .03 or stationary_yaw > math.radians(2)
                or not _raw_good(localizer, after, fresh, stationary.transform)):
            report('large_turn_post_stop_motion_unverified',
                   translation_m=stationary_distance,
                   heading_deg=math.degrees(stationary_yaw))
            return None
        on_fresh = bounded(fresh, hint, radius_m=.20, budget_s=.45,
                           yaw_radius_deg=10.)
        recovered = map_motion(on_fresh, fresh)
        if recovered is None:
            # Glass and the feature-poor side of this room can defeat two
            # absolute fits while the relative scan geometry is still clear.
            # Require a second pre-turn fit AND a reverse fit that closes the
            # SE(2) loop; then let normal map tracking/short odometry bridge
            # continue the absolute-position search. Never reuse old yaw.
            corroborated = relative(before, fresh)
            reverse = relative(fresh, before)
            shift, yaw = _difference(first.transform, corroborated.transform)
            loop = corroborated.transform.compose(reverse.transform)
            loop_shift, loop_yaw = _difference(loop, type(loop)())
            known_map = getattr(localizer, 'map', None)
            free = False
            if known_map is not None:
                predicted = old_pose.compose(corroborated.transform)
                column, row = known_map.world_to_grid(((predicted.x_m, predicted.y_m),))[0]
                column, row = int(column), int(row)
                free = bool(known_map.in_bounds(column, row)
                            and known_map.traversable[row, column])
            if (_match_good(corroborated, inlier=.65, max_yaw_deg=None)
                    and _match_good(reverse, inlier=.65, max_yaw_deg=None)
                    and shift <= .07 and yaw <= math.radians(5)
                    and loop_shift <= .06 and loop_yaw <= math.radians(4)
                    and corroborated.transform.yaw_rad*speed > 0
                    and abs(corroborated.transform.yaw_rad) <= plausible
                    and free):
                report('large_turn_reciprocal_odometry',
                       map_recheck='not_confirmed',
                       observed_deg=math.degrees(observed),
                       corroborated_deg=math.degrees(corroborated.transform.yaw_rad),
                       reciprocal_shift_m=loop_shift,
                       reciprocal_yaw_deg=math.degrees(loop_yaw))
                return corroborated
            report('large_turn_second_map_unverified',
                   first_map_diagnostic=getattr(localizer,'_last_bounded_diagnostic',None),
                   corroborated_inlier=getattr(corroborated,'inlier_ratio',None),
                   reverse_inlier=getattr(reverse,'inlier_ratio',None),
                   agreement_m=shift, agreement_deg=math.degrees(yaw),
                   reciprocal_shift_m=loop_shift,
                   reciprocal_yaw_deg=math.degrees(loop_yaw),
                   predicted_on_known_free_map=free)
            return None
        if _map_good(on_after, hint):
            shift, yaw = _difference(on_after.pose, on_fresh.pose)
            if shift > .05 or yaw > math.radians(3):
                report('large_turn_map_frames_disagree', shift_m=shift,
                       heading_deg=math.degrees(yaw))
                return None
        else:
            corroborated = relative(before, fresh)
            shift, yaw = _difference(first.transform, corroborated.transform)
            if (not _match_good(corroborated, inlier=.65, max_yaw_deg=None)
                    or shift > .07 or yaw > math.radians(5)
                    or not _raw_good(localizer, before, fresh, corroborated.transform)):
                report('large_turn_second_relative_unverified', shift_m=shift,
                       heading_deg=math.degrees(yaw))
                return None
        report('large_turn_stopped_map_reanchored',
               observed_deg=math.degrees(observed),
               map_deg=math.degrees(recovered.transform.yaw_rad),
               full_translation_m=math.hypot(recovered.transform.x_m,
                                             recovered.transform.y_m))
        return recovered
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError,
            TimeoutError) as exc:
        report('large_turn_recheck_error', error_type=type(exc).__name__,
               error=str(exc))
        return None
