"""Use bounded navigation support for proven installed-body occlusion.

This is not a replacement for the lidar guard: real exterior near returns,
unexplained missing sectors, occupied footprint and continuous glass sweeps
remain blocking. A precision-fit uncertainty shell is not the physical body.
"""
import math

from .glass_boundary_guard import swept_glass_clearance
from .motion_interlock import ESTIMATE_MAX_UNCERTAINTY_M
from .rotation_guard import assess_rotation


def navigation_rotation_support(assessment, scan, context, *, now):
    if assessment.get('safe') or assessment.get('reason') not in {
            'unobserved_sector_without_safe_map_support',
            'installation_edge_occlusion_without_safe_map_support'}:
        return assessment
    quality = context.get('quality', {})
    age = now - context.get('updated', -math.inf)
    scan_age = now - getattr(scan, 'received_s', -math.inf)
    uncertainty = float(context.get('bridge_uncertainty_m', 0.))
    if (not context.get('reliable') or not 0 <= age <= .5
            or not 0 <= scan_age <= .5
            or not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= ESTIMATE_MAX_UNCERTAINTY_M
            or not (quality.get('reason') == 'strong_map_match'
                    or quality.get('estimated_navigation') is True)):
        return assessment
    # Recompute on the actual raw scan. No saved Boolean or broadened body mask
    # can authorize a missing sector. Physical disk still checks every map cell.
    physical = assess_rotation(scan, known_map=context.get('map'),
        pose=context.get('pose'), map_pose_reliable=True,
        map_pose_uncertainty_m=0., exact_glass_geometry=True)
    if (not physical['safe'] or physical.get('installed_body_blind_arc') is not True
            or physical.get('self_return_count', 0) < 24):
        return dict(assessment, navigation_support={'physical': physical})
    # Pose uncertainty is retained for transparent-wall half-plane boundaries;
    # it is not silently reset by a good relative scan match.
    glass = swept_glass_clearance(context.get('map'), context.get('pose'), 0, 3500,
        reliable=True, pose_age_s=age, radius_m=.44 + uncertainty)
    if not glass['safe']:
        return dict(assessment, navigation_support={'physical': physical, 'glass': glass})
    return dict(assessment, safe=True,
        reason='installed_body_occlusion_with_navigation_support',
        previous_reason=assessment['reason'], map_precision_required=False,
        navigation_support={'physical': physical, 'glass': glass,
                            'position_uncertainty_m': uncertainty,
                            'estimated_navigation': quality.get('estimated_navigation', False)})
