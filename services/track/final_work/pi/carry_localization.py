"""Localize AFTER pickup without the frozen mandatory movement initializer.

No arm or raw motor writes. The earlier search pose bounds XY search only;
fresh map evidence owns the starting heading and the navigation scan.
"""
import math
from pathlib import Path
from types import SimpleNamespace

from grasp_range_integration.phase1.startup_active_recovery import recover_startup, RecoverySettings


def localize_carry(mission, hint, args, client, scans, localizer, controller, stopping, **kwargs):
    try:
        values = [float(v) for v in hint]
    except (TypeError, ValueError):
        values = []
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError('搬运任务缺少有效的上一阶段XY搜索提示；不能将未知位姿当作起点')
    if stopping() or getattr(mission, '_phase1_motion_fault', None):
        raise RuntimeError('搬运定位已取消或底盘故障；未授权新动作')
    prior = SimpleNamespace(x_m=values[0], y_m=values[1],
        yaw_deg=math.degrees(values[2]), radius_m=1.5, yaw_tolerance_deg=180.)
    callback = kwargs.get('event_callback')

    def emit(name, **fields):
        from grasp_range_integration.phase1.turn_policy import emit as telemetry
        telemetry(name, **fields)
        if callback is not None:
            callback(name, **fields)

    emit('carry_localization_started', hint_role='xy_search_only', radius_m=1.5,
         heading_search_deg=360, required_validation_moves=0,
         source='adaptive_startup_recovery_not_frozen_localizer')
    print('搬运定位：上一阶段位姿仅限定附近XY搜索，朝向重新匹配；四帧雷达一致即可衔接导航，不强制移动验收。', flush=True)
    previous_mode = getattr(mission, '_phase1_recovery_mode', None)
    mission._phase1_recovery_mode = 'startup'
    try:
        pose, scan = recover_startup(mission, args, client, scans, localizer, controller,
            stopping, prior, camera=kwargs.get('camera'), recognizer=kwargs.get('visual_recognizer'),
            gpu=kwargs.get('gpu'), directory=kwargs.get('visual_directory') or Path('search_runs'),
            emit=emit, settings=RecoverySettings(phase='startup', initial_fit=True,
                operator_prompt=False, heading_tolerance_deg=180., fresh_start_handoff=True,
                visual_attempt_base=4000))
    finally:
        # Preserve carrying/closed-claw state even if transport/scan fails.
        try:
            client.stop()
        finally:
            controller.stopped()
            mission._phase1_recovery_mode = previous_mode
    if stopping() or getattr(mission, '_phase1_motion_fault', None):
        raise RuntimeError('搬运定位交接已取消或底盘故障；未进入导航')
    if hasattr(localizer, 'track'):
        from grasp_range_integration.phase1.relationship_find_only_entry import _install_midrun_visual_recheck
        _install_midrun_visual_recheck(mission, localizer, args, client, controller,
            kwargs.get('camera'), kwargs.get('visual_recognizer'), kwargs.get('gpu'),
            kwargs.get('visual_directory') or Path('search_runs'), callback, scan,
            scan_iterator=scans, stopping=stopping)
    emit('carry_localization_ready', pose=[pose.x_m, pose.y_m, pose.yaw_rad],
         final_sequence=getattr(scan, 'sequence', None), old_pose_not_used_for_navigation=True)
    return pose, scan
