"""Return a measured pose AND scan only after first semantic route is usable.

No raw motor API, no mask relaxation, and no coordinate snapping. This runs
before the frozen mission receives its initial pose/scan, so real recovery
displacement cannot be followed by navigation from the old start.
"""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from grasp_range_integration.phase1.startup_active_recovery import recover_startup, RecoverySettings
from grasp_range_integration.phase1.visibility_route_planner import NoReachableVisibilityError, POLICY


def startup_candidate_mask(known_map, request):
    """Reject no-exit basins before lidar consensus, without pose snapping.

The lidar still chooses among scored poses inside the unchanged start prior.
Camera FOV is not a prerequisite for starting away from the search region.
    """
    if not all(hasattr(known_map, name) for name in ('occupied', 'unknown',
            'temporary_occupied', 'traversable', 'world_to_grid', 'grid_to_world')):
        return None  # Non-map test doubles / non-production callers.
    from grasp_range_integration.phase1.navigation_avoidance import turn_reserve_mask
    mask = turn_reserve_mask(known_map)
    center = request.get('center_xy_m') or []
    if len(center) != 2:
        return mask
    import cv2
    rows, columns = np.indices(mask.shape)
    world = known_map.grid_to_world(np.stack((columns, rows), axis=-1))
    distance = np.hypot(world[..., 0]-center[0], world[..., 1]-center[1])
    radius = float(request.get('radius_m', .95))
    approach = mask & (distance >= radius+.80) & (distance <= radius+1.65)
    if not approach.any():
        return mask  # The actual region planner will diagnose an unreachable region.
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    reachable = np.unique(labels[approach])
    return mask & np.isin(labels, reachable[reachable != 0])


def install_startup_route_recovery(mission):
    mission._localize = wrap_startup_route_recovery(mission, mission._localize)


def wrap_startup_route_recovery(mission, original):
    """Keep route/mask handoff when a skill supplies its own initial localizer."""

    def localize(args, client, scans, localizer, controller, stopping, **kwargs):
        original_track = getattr(localizer, 'track', None)
        request = json.loads(Path(args.decision).read_text(encoding='utf-8')).get('route_request') or {}
        semantic = request.get('planning_policy') == POLICY and request.get('mode') == 'semantic_search_region'
        old_mask = getattr(localizer, '_phase1_start_candidate_mask', None)
        try:
            if semantic:
                localizer._phase1_start_candidate_mask = startup_candidate_mask(localizer.map, request)
            pose, scan = original(args, client, scans, localizer, controller, stopping, **kwargs)
        finally:
            localizer._phase1_start_candidate_mask = old_mask
        if request.get('planning_policy') != POLICY or request.get('mode') != 'semantic_search_region':
            return pose, scan
        callback = kwargs.get('event_callback')

        def emit(name, **fields):
            from grasp_range_integration.phase1.turn_policy import emit as telemetry
            telemetry(name, **fields)
            if callback is not None:
                callback(name, **fields)

        def route_ready(candidate):
            if stopping() or getattr(mission, '_phase1_motion_fault', None):
                raise RuntimeError('起点路线恢复已取消或底盘反馈故障；未授权新动作')
            p = candidate.pose
            try:
                route = mission.build_search_route(localizer.map, (p.x_m, p.y_m), request)
            except NoReachableVisibilityError:
                emit('startup_route_blocked', pose=[p.x_m, p.y_m, p.yaw_rad],
                     terminal=False, next_action='guarded_view_change', mask_relaxed=False)
                return False
            return bool(route.get('views'))

        if route_ready(SimpleNamespace(pose=pose)):
            return pose, scan
        if not callable(getattr(client, 'startup_recovery_pulse', None)):
            raise RuntimeError('起点无可达路线且缺少受保护换视角接口；未发送动作')
        print('起点定位已通过，但地图规划掩膜暂时无出口：按新雷达安全走廊换视角/小幅前移，重新定位并检查路线；不直接结束任务。', flush=True)
        # The start has four-scan consensus, not the original 0.70m unknown
        # envelope. Preserve conservative 0.10m uncertainty around this fit.
        prior = SimpleNamespace(x_m=pose.x_m, y_m=pose.y_m,
            yaw_deg=math.degrees(pose.yaw_rad), radius_m=.10, yaw_tolerance_deg=30.)
        previous_mode = getattr(mission, '_phase1_recovery_mode', None)
        mission._phase1_recovery_mode = 'midrun'
        try:
            recovered = recover_startup(mission, args, client, scans, localizer, controller,
                stopping, prior, camera=kwargs.get('camera'), recognizer=kwargs.get('visual_recognizer'),
                gpu=kwargs.get('gpu'), directory=kwargs.get('visual_directory') or Path('search_runs'),
                emit=emit, reference_scan=scan, accept_candidate=route_ready,
                settings=RecoverySettings(phase='midrun', time_budget_s=90., heading_tolerance_deg=30.,
                    operator_prompt=False, initial_fit=False, visual_attempt_base=3000))
        finally:
            client.stop()
            controller.stopped()
            mission._phase1_recovery_mode = previous_mode
        # Rebind the already installed tracking adapter's source scan too.
        from grasp_range_integration.phase1.relationship_find_only_entry import _install_midrun_visual_recheck
        if hasattr(localizer, 'track'):
            # Replace the old initial-scan closure; never stack two tracking
            # adapters with different scan baselines after real movement.
            localizer.track = original_track
            _install_midrun_visual_recheck(mission, localizer, args, client, controller,
                kwargs.get('camera'), kwargs.get('visual_recognizer'), kwargs.get('gpu'),
                kwargs.get('visual_directory') or Path('search_runs'), callback, recovered.scan,
                scan_iterator=scans, stopping=stopping)
        emit('startup_route_recovered', pose=[recovered.candidate.pose.x_m,
            recovered.candidate.pose.y_m, recovered.candidate.pose.yaw_rad],
            final_sequence=getattr(recovered.scan, 'sequence', None), old_start_discarded=True)
        return recovered.candidate.pose, recovered.scan

    return localize
