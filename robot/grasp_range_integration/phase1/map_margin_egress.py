"""Short forward transit escape from a map uncertainty shell, not a pivot veto override."""
import math
import time
from types import SimpleNamespace

from grasp_range_integration.phase1.rotation_guard import map_rotation_evidence
from grasp_range_integration.phase1.startup_active_recovery import assess_recovery_action


class MapMarginEgress:
    def __init__(self):
        self.origin=None
        self.started=0.
        self.exhausted=False

    def select(self, context, scan, route, assessment=None, *, clock=time.monotonic):
        pose=context.get('pose')
        result=dict(safe=False,reason='egress_not_applicable')
        if pose is None or not context.get('reliable') or not route or route.get('navigation_stage')!='approach':
            return result
        now=clock()
        if not 0 <= now-context.get('updated',0.) <= .5:
            return dict(result,reason='egress_pose_stale')
        uncertainty=float(context.get('bridge_uncertainty_m',0.))
        if not 0 <= uncertainty <= .35:
            return dict(result,reason='egress_uncertainty_excessive')
        region=route.get('target_region') or {}
        center=region.get('center_xy_m')
        if not center or len(center)!=2:
            return result
        distance=math.hypot(center[0]-pose.x_m,center[1]-pose.y_m)
        # Transit only. Never advance towards a single-camera object, grasp
        # target, or through the reserved object work area.
        reserve=float(region.get('radius_m',.95))+.80
        if distance <= reserve+uncertainty+.40:
            return dict(result,reason='egress_near_object_work_area')
        bearing=math.atan2(center[1]-pose.y_m,center[0]-pose.x_m)
        error=math.atan2(math.sin(bearing-pose.yaw_rad),math.cos(bearing-pose.yaw_rad))
        if abs(error)>math.radians(60):
            return dict(result,reason='egress_heading_away_from_region')
        known=context.get('map')
        physical=map_rotation_evidence(known,pose,.44)
        if not physical['clear']:
            return dict(result,reason='egress_physical_map_disk_blocked',physical_map=physical)
        if self.origin is None:
            if assessment is None or self.exhausted:
                return result
            layers=(assessment.get('map_support') or {}).get('layers') or {}
            shell_only=(assessment.get('installed_body_blind_arc') is True
                and assessment.get('map_pose_reliable') is True
                and assessment.get('reason')=='unobserved_sector_without_safe_map_support'
                and layers.get('occupied',{}).get('count',0)>0
                and all(layers.get(n,{}).get('count',0)==0
                        for n in ('unknown','temporary_occupied','verified_glass')))
            if not shell_only:
                return result
        else:
            progress=math.hypot(pose.x_m-self.origin[0],pose.y_m-self.origin[1])
            if progress>=.25 or now-self.started>=6.:
                self.origin=None
                self.exhausted=True
                return dict(result,reason='egress_view_changed_resume_normal_tracking',progress_m=progress)
            yaw=math.atan2(math.sin(pose.yaw_rad-self.origin[2]),math.cos(pose.yaw_rad-self.origin[2]))
            if abs(yaw)>math.radians(12):
                return dict(result,reason='egress_heading_changed')
        prior=SimpleNamespace(x_m=pose.x_m,y_m=pose.y_m,yaw_deg=math.degrees(pose.yaw_rad),
                              radius_m=uncertainty,yaw_tolerance_deg=30.)
        evidence=assess_recovery_action(scan,known,prior,400,0,650,clock=clock,
                                       directional_translation=True,dispatch_margin_s=.05)
        if not evidence['safe']:
            return dict(result,reason='egress_live_corridor_or_glass_blocked',evidence=evidence)
        if self.origin is None:
            self.origin=(pose.x_m,pose.y_m,pose.yaw_rad)
            self.started=now
        return dict(safe=True,reason='map_uncertainty_shell_forward_egress',linear_mm_s=400,
                    angular_mrad_s=0,evidence=evidence,physical_map=physical,
                    map_pose_confirmed=bool((context.get('quality') or {}).get('map_match')),
                    uncertainty_m=uncertainty,maximum_measured_egress_m=.25)

