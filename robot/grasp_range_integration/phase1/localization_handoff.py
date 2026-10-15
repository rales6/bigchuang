"""Exactly-once transfer of internally measured recovery motion to navigation."""


def recovery_prediction(mission, result, predicted):
    handoff = getattr(mission, '_phase1_localization_handoff', None)
    if handoff is not None and result is handoff['result'] and not handoff.get('pose_consumed'):
        handoff['pose_consumed'] = True
        return handoff['predicted_pose'], handoff
    return predicted, None


def recovery_baseline(mission, localizer, before, after):
    handoff = getattr(mission, '_phase1_localization_handoff', None)
    if (handoff is not None and localizer is handoff['localizer']
            and before is handoff['source_scan'] and not handoff.get('baseline_consumed')):
        if not handoff.get('pose_consumed') and not handoff.get('deferred_prediction'):
            raise RuntimeError('恢复后的位姿尚未交接；禁止将运动计入旧位姿')
        if getattr(after, 'sequence', -1) <= getattr(handoff['final_scan'], 'sequence', -1):
            raise RuntimeError('恢复后的雷达基准尚未更新；禁止重复累计恢复位移')
        handoff['baseline_consumed'] = True
        return handoff['final_scan']
    return before


def deferred_recovery_pose(mission, old_pose):
    """Route-time recovery owns no frozen loop variables; replace them once.

    Called BEFORE applying final-scan -> next-scan motion. Normal in-track
    recovery remains strict and is fused by recovery_prediction instead.
    """
    handoff = getattr(mission, '_phase1_localization_handoff', None)
    if handoff and handoff.get('deferred_prediction') and not handoff.get('pose_consumed'):
        if not handoff.get('baseline_consumed'):
            raise RuntimeError('换视角路线恢复的新扫描基准尚未交接')
        handoff['pose_consumed'] = True
        return handoff['result'].pose
    return old_pose
