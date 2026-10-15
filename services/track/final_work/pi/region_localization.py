"""Fresh-start localization for navigation without a pickup pose hint.

Do not enter the frozen compulsory-movement initializer. A global lidar
match narrows search only; the adaptive recovery must independently confirm
four stationary scans and a fresh post-fit frame before returning to driving.
"""
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from grasp_range_integration.phase1.startup_active_recovery import recover_startup, RecoverySettings


def map_search_prior(known_map):
    """An unknown start spans free map cells, never the destination coordinates."""
    rows, columns = np.nonzero(known_map.traversable)
    if not len(rows):
        raise ValueError('区域定位没有可搜索的地图自由区域')
    corners = np.asarray(known_map.grid_to_world([
        [int(columns.min()), int(rows.min())],
        [int(columns.max()), int(rows.max())]]), dtype=float)
    center = (corners[0] + corners[1]) / 2
    radius = float(np.linalg.norm(corners[1] - corners[0]) / 2 + known_map.resolution_m)
    if not np.isfinite(corners).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError('区域定位地图范围无效')
    return SimpleNamespace(x_m=float(center[0]), y_m=float(center[1]),
        yaw_deg=0., radius_m=radius, yaw_tolerance_deg=180.)


def seed_search_prior(localizer, candidate, args):
    """Only a strong whole-map result may narrow subsequent XY search.

The seed is NOT a navigation pose or a measured zero-motion transform.
Weak/ambiguous results leave the whole map in scope; do not turn the best
of several poor candidates into an authoritative starting point.
"""
    if candidate is None or not candidate.reliable:
        return None
    pose = candidate.pose
    values = (pose.x_m, pose.y_m, pose.yaw_rad, candidate.rmse_m, candidate.inlier_ratio)
    margin = float(candidate.ambiguity_margin)
    if (not all(math.isfinite(float(value)) for value in values)
            or not 0 <= candidate.rmse_m <= .10
            or not .70 <= candidate.inlier_ratio <= 1.
            or math.isnan(margin) or margin < max(.025, float(getattr(args, 'localization_min_margin', .025)))):
        return None
    mask = getattr(localizer, '_phase1_start_candidate_mask', None)
    if mask is not None:
        column, row = localizer.map.world_to_grid([(pose.x_m, pose.y_m)])[0]
        if not localizer.map.in_bounds(column, row) or not mask[row, column]:
            return None
    return SimpleNamespace(x_m=pose.x_m, y_m=pose.y_m,
        yaw_deg=math.degrees(pose.yaw_rad), radius_m=1.5, yaw_tolerance_deg=180.)


def point_bearings(candidates, semantic_map):
    """Include rear landmarks too: a front-door observation can reject a rear door.

    A missing object in a photo is never negative evidence. Extended glass
    walls have no point-bearing veto, and uncertain map points retain margins.
    """
    structures = [item for item in semantic_map.get('objects', []) if isinstance(item, dict)
        and item.get('semantic_reviewed') is True and item.get('category') in {'door', 'window', 'stairs'}
        and isinstance(item.get('map_xy'), (list, tuple)) and len(item['map_xy']) >= 2]
    rows = []
    for index, candidate in enumerate(candidates, 1):
        p = candidate.pose
        points = []
        for item in structures:
            dx, dy = float(item['map_xy'][0])-p.x_m, float(item['map_xy'][1])-p.y_m
            uncertainty = max(0., float(item.get('position_uncertainty_m') or 0.))
            distance = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(math.sin(math.atan2(dy, dx)-p.yaw_rad),
                                             math.cos(math.atan2(dy, dx)-p.yaw_rad)))
            if not all(math.isfinite(v) for v in (bearing, distance, uncertainty)) or distance <= max(.01, uncertainty):
                continue  # A centroid inside its uncertainty disk has no useful direction.
            points.append({'name': item.get('name'), 'category': item['category'],
                'relative_bearing_deg': bearing,
                'bearing_uncertainty_deg': min(45., math.degrees(math.asin(min(1., uncertainty/distance))))})
        rows.append({'candidate_id': f'C{index:02d}', 'points': points})
    return rows


def positive_structure_prior(candidates, descriptions, diagnostics):
    """Use a positive sighting to EXCLUDE contradictions, not invent a pose.

    DS may abstain between candidates while still reliably seeing a door in
    front. Do not let that abstention restore the high-score door-behind pose.
    This is only a rough radar search envelope, not visual localization success.
    """
    result = diagnostics.get('deepseek_result') or {}
    observed = set()
    for item in result.get('observed_structures', []):
        if not isinstance(item, dict):
            continue
        try:
            confidence = float(item.get('confidence') or 0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(confidence) and .8 <= confidence <= 1.:
            observed.add(str(item.get('name')))
    checked = []
    for row in descriptions:
        by_name = {}
        for point in row.get('points', []):
            by_name.setdefault(point['name'], []).append(point)
        contradictions = []
        for name in observed & by_name.keys():
            # 41deg half-FOV + camera toe-in/mounting allowance + map-point
            # uncertainty. Only gross front/back conflicts are excluded.
            if all(abs(p['relative_bearing_deg']) > 64.+p['bearing_uncertainty_deg']
                   for p in by_name[name]):
                contradictions.append(name)
        checked.append({'candidate_id': row['candidate_id'], 'contradictions': contradictions})
    rejected = {row['candidate_id'] for row in checked if row['contradictions']}
    survivors = [row for index, row in enumerate(candidates, 1) if f'C{index:02d}' not in rejected]
    audit = {'positive_observations': sorted(observed), 'candidates': checked,
             'excluded_candidate_ids': sorted(rejected), 'visually_verified': False}
    if not rejected or not survivors:
        return None, audit
    # An unresolved remote/symmetric alternative is not a reason to take
    # just the best of it. Narrow only a coherent surviving radar envelope.
    x = sum(row.pose.x_m for row in survivors)/len(survivors)
    y = sum(row.pose.y_m for row in survivors)/len(survivors)
    anchor = max(survivors, key=lambda row: float(getattr(row, 'score', 0.))).pose
    yaw_spread = max(abs(math.degrees(math.atan2(math.sin(row.pose.yaw_rad-anchor.yaw_rad),
                         math.cos(row.pose.yaw_rad-anchor.yaw_rad)))) for row in survivors)
    radius = max(.55, max(math.hypot(row.pose.x_m-x, row.pose.y_m-y) for row in survivors)+.18)
    if radius > 1.5 or yaw_spread > 33:
        audit['reason'] = 'surviving_basins_still_spatially_ambiguous'
        return None, audit
    audit['reason'] = 'positive_structure_constraints_on_radar_search_not_a_camera_fix'
    return SimpleNamespace(x_m=x, y_m=y, yaw_deg=math.degrees(anchor.yaw_rad), radius_m=radius,
        yaw_tolerance_deg=max(20., yaw_spread+12.), heading_evidence='positive_structure_constraints'), audit


def review_global_heading(mission, args, candidate, scan, localizer, camera,
                          recognizer, gpu, directory, emit):
    """Rank measured WHOLE-MAP basins once, before any heading is trusted.

    Repeated stationary fits can agree on the wrong symmetric basin. Unlike
    the fixed-staging-area reviewer, this has no user-confirmed XY/yaw scope.
    Vision still cannot introduce coordinates or refit all headings secretly.
    """
    checker = getattr(mission, '_visual_localization_check', None)
    if candidate is None or not callable(checker) or any(x is None for x in (camera, recognizer, gpu)):
        emit('region_initial_heading_review', status='unavailable', heading_verified=False)
        return None
    from grasp_range_integration.phase1.startup_visual_review import (
        pose_delta, normalize_descriptions, audit_observations)
    descriptions_module, original_descriptions = None, None
    full_bearings = []
    try:
        candidates = list(localizer.global_localization_candidates(scan, maximum=6))
        if not any(pose_delta(row.pose, candidate.pose)[0] < .25
                   and pose_delta(row.pose, candidate.pose)[1] < 15 for row in candidates):
            candidates.insert(0, candidate)
        mask = getattr(localizer, '_phase1_start_candidate_mask', None)
        measured = []
        for row in candidates:
            p = row.pose
            if (not row.reliable or not all(math.isfinite(float(v)) for v in
                    (p.x_m, p.y_m, p.yaw_rad, row.rmse_m, row.inlier_ratio))):
                continue
            if mask is not None:
                col, line = localizer.map.world_to_grid([(p.x_m, p.y_m)])[0]
                if not localizer.map.in_bounds(col, line) or not mask[line, col]:
                    continue
            measured.append(row)
        if not measured:
            emit('region_initial_heading_review', status='no_measured_candidate', heading_verified=False)
            return None

        class MeasuredBasins:
            def global_localization_candidates(self, scan, maximum=6):
                return measured[:maximum]

            def localize_near_position(self, scan, hint, radius_m=.9):
                # The legacy DS corrector searches all headings here. Only
                # return an ALREADY measured basin near the selected heading.
                rows = [row for row in measured if pose_delta(row.pose, hint)[0] <= radius_m
                        and pose_delta(row.pose, hint)[1] <= 15]
                return min(rows, key=lambda row: sum(pose_delta(row.pose, hint))) if rows else None

            def __getattr__(self, name):
                return getattr(localizer, name)

        if getattr(mission, '_phase1_deepseek_enabled', False):
            import deepseek_visual_localization as descriptions_module
            original_descriptions = descriptions_module.candidate_scene_descriptions

            def describe(*values, **options):
                nonlocal full_bearings
                rows = normalize_descriptions(original_descriptions(*values, **options))
                full_bearings = point_bearings(values[0], values[1])
                for row in rows:
                    row['positive_sighting_bearings_all_directions'] = next(
                        (v['points'] for v in full_bearings if v['candidate_id'] == row['candidate_id']), [])
                    row['startup_review_policy'] = (
                        '全局起点未知；比较全部实测雷达候选。至少两个稳定结构的实际出现与方向支持才可选，'
                        '不能按雷达最高分、地板纹理或单个柜子猜测。玻璃门和玻璃后的楼梯可用于区分朝向。')
                return rows
            descriptions_module.candidate_scene_descriptions = describe
        status, winner, match, diagnostics = checker(
            args, measured[0], scan, MeasuredBasins(), camera, recognizer,
            gpu, Path(directory), 4900)
        accepted = (status in {'confirmed', 'corrected'} and match is not None
                    and winner is not None and winner.reliable
                    and any(pose_delta(winner.pose, row.pose)[0] <= .12
                            and pose_delta(winner.pose, row.pose)[1] <= 6 for row in measured))
        constrained_prior = None
        if descriptions_module is not None:
            captures = diagnostics.get('capture') or []
            pair = (diagnostics.get('camera_count') == 2 and len(captures) == 2
                    and Path(captures[0]).resolve() != Path(captures[1]).resolve()
                    and all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in captures))
            audit = audit_observations(diagnostics)
            diagnostics['region_direction_audit'] = audit
            accepted = (accepted and pair and diagnostics.get('deepseek_reliable') is True
                        and len(audit['matched']) >= 2 and not audit['contradictions'])
            if (pair and not diagnostics.get('deepseek_error') and full_bearings
                    and int((diagnostics.get('deepseek_result') or {}).get('consistent_repeats') or 0) >= 2):
                constrained_prior, constraints = positive_structure_prior(measured[:len(full_bearings)], full_bearings, diagnostics)
                diagnostics['positive_structure_constraints'] = constraints
                # Even a positive model choice cannot override a gross
                # front/back contradiction revealed by its own observations.
                chosen = (diagnostics.get('deepseek_result') or {}).get('selected_candidate_id')
                if chosen in constraints['excluded_candidate_ids']:
                    accepted = False
        emit('region_initial_heading_review', status=status, heading_verified=bool(accepted),
             candidate_scope='whole_map_measured_basins', diagnostics=diagnostics,
             candidate_poses=[[row.pose.x_m, row.pose.y_m, row.pose.yaw_rad] for row in measured],
             positive_heading_constraint_applied=constrained_prior is not None and not accepted,
             pose=([winner.pose.x_m, winner.pose.y_m, winner.pose.yaw_rad] if accepted else None))
        if accepted:
            p = winner.pose
            return SimpleNamespace(x_m=p.x_m, y_m=p.y_m, yaw_deg=math.degrees(p.yaw_rad),
                                   radius_m=.55, yaw_tolerance_deg=35., heading_evidence='camera_confirmed')
        if constrained_prior is not None:
            return constrained_prior
    except Exception as exc:
        emit('region_initial_heading_review', status='error', heading_verified=False,
             error_type=type(exc).__name__, error=str(exc))
    finally:
        if original_descriptions is not None:
            descriptions_module.candidate_scene_descriptions = original_descriptions
    # Abstention is NOT confirmation. Keep lidar recovery available without
    # inventing a pose or making a routine API uncertainty a terminal fault.
    return None


def localize_region(mission, args, client, scans, localizer, controller, stopping,
                    *, clock=time.monotonic, **kwargs):
    """Use the same adaptive recovery as carrying, without assuming a start."""
    if stopping() or getattr(mission, '_phase1_motion_fault', None):
        raise RuntimeError('区域定位已取消或底盘故障；未授权新动作')
    callback = kwargs.get('event_callback')

    def emit(name, **fields):
        from grasp_range_integration.phase1.turn_policy import emit as telemetry
        telemetry(name, **fields)
        if callback is not None:
            callback(name, **fields)

    previous_mode = getattr(mission, '_phase1_recovery_mode', None)
    mission._phase1_recovery_mode = 'startup'
    try:
        client.stop()
        stopped_at = clock()
        seed_deadline = stopped_at + 5.
        # Do not score a buffered scan from the preceding task/movement.
        while True:
            if stopping() or getattr(mission, '_phase1_motion_fault', None):
                raise RuntimeError('区域定位已取消或底盘故障；未授权新动作')
            if clock() >= seed_deadline:
                raise RuntimeError('区域定位没有收到停车后的新雷达扫描')
            scan = next(scans)
            if (getattr(scan, 'acquisition_start_s', -math.inf) > stopped_at
                    and 0 <= clock() - getattr(scan, 'received_s', -math.inf) <= .5):
                break
        candidate = localizer.global_localize(scan, hint=None)
        if stopping() or getattr(mission, '_phase1_motion_fault', None):
            raise RuntimeError('区域定位已取消或底盘故障；未授权新动作')
        prior = seed_search_prior(localizer, candidate, args)
        source = 'fresh_global_lidar_search_only' if prior is not None else 'whole_free_map'
        visual_prior = review_global_heading(mission, args, candidate, scan, localizer,
            kwargs.get('camera'), kwargs.get('visual_recognizer'), kwargs.get('gpu'),
            kwargs.get('visual_directory') or Path('search_runs'), emit)
        if stopping() or getattr(mission, '_phase1_motion_fault', None):
            raise RuntimeError('区域视觉复核已取消或底盘故障；未授权新动作')
        if visual_prior is not None:
            prior = visual_prior
            source = ('positive_structure_constrained_lidar' if getattr(prior, 'heading_evidence', '')
                      == 'positive_structure_constraints' else 'whole_map_lidar_and_camera_heading')
        if prior is None:
            prior = map_search_prior(localizer.map)
        emit('region_localization_started', search_source=source,
             search_xy=[prior.x_m, prior.y_m], radius_m=prior.radius_m,
             heading_search_deg=2*prior.yaw_tolerance_deg, required_validation_moves=0,
             visual_heading_verified=visual_prior is not None and source != 'positive_structure_constrained_lidar',
             positive_heading_constraint_applied=source == 'positive_structure_constrained_lidar',
             seed_is_navigation_pose=False, source='adaptive_startup_not_frozen_localizer')
        print('区域启动定位：先做四帧雷达一致性及新帧复核，不强制换位验收；证据不足再按实时净空自适应换视角。', flush=True)
        pose, scan = recover_startup(mission, args, client, scans, localizer, controller,
            stopping, prior, camera=kwargs.get('camera'), recognizer=kwargs.get('visual_recognizer'),
            gpu=kwargs.get('gpu'), directory=kwargs.get('visual_directory') or Path('search_runs'),
            emit=emit, settings=RecoverySettings(phase='startup', initial_fit=True,
                operator_prompt=False, heading_tolerance_deg=prior.yaw_tolerance_deg, fresh_start_handoff=True,
                time_budget_s=max(30., min(180., float(getattr(args, 'localization_timeout_seconds', 120.)))),
                visual_attempt_base=5000))
    finally:
        try:
            client.stop()
        finally:
            controller.stopped()
            mission._phase1_recovery_mode = previous_mode
    if stopping() or getattr(mission, '_phase1_motion_fault', None):
        raise RuntimeError('区域定位交接已取消或底盘故障；未进入导航')
    if hasattr(localizer, 'track'):
        from grasp_range_integration.phase1.relationship_find_only_entry import _install_midrun_visual_recheck
        _install_midrun_visual_recheck(mission, localizer, args, client, controller,
            kwargs.get('camera'), kwargs.get('visual_recognizer'), kwargs.get('gpu'),
            kwargs.get('visual_directory') or Path('search_runs'), callback, scan,
            scan_iterator=scans, stopping=stopping)
    mission._region_start_localization = dict(pose=[pose.x_m, pose.y_m, pose.yaw_rad],
         yaw_deg=math.degrees(pose.yaw_rad), source=source,
         visual_heading_verified=visual_prior is not None and source != 'positive_structure_constrained_lidar',
         positive_heading_constraint_applied=source == 'positive_structure_constrained_lidar',
         final_sequence=getattr(scan, 'sequence', None))
    emit('region_localization_ready', pose=[pose.x_m, pose.y_m, pose.yaw_rad],
         yaw_deg=math.degrees(pose.yaw_rad),
         visual_heading_verified=visual_prior is not None and source != 'positive_structure_constrained_lidar',
         positive_heading_constraint_applied=source == 'positive_structure_constrained_lidar',
         final_sequence=getattr(scan, 'sequence', None), required_validation_moves=0,
         seed_is_navigation_pose=False)
    return pose, scan
