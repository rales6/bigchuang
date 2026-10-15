"""Bounded command sweep regressions, including 174417 turn veto scalars."""
import math
from types import SimpleNamespace as NS
import unittest
import numpy as np
from raspberry_pi.mapping import Pose2D
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.glass_boundary_guard import (
    attach_boundaries,bounded_recovery_glass_sweep,USER_LEFT_POLICY)
from grasp_range_integration.phase1.startup_active_recovery import assess_recovery_action


class RecoverySweepTests(unittest.TestCase):
    def setUp(self):
        self.map=KnownOccupancyMap(np.full((240,240),254,dtype=np.uint8),.05,-6,-6)
        attach_boundaries(self.map,dict(segments_xy_m=[[[0,-5],[0,5]]],
            boundary_uncertainty_m=.20,boundary_policy=USER_LEFT_POLICY,
            allowed_reference_xy_m=[3,0],verified=False))
        self.scan=NS(angles_rad=np.linspace(-math.pi,math.pi,360,endpoint=False),
                     distances_m=np.full(360,3.),received_s=0.)

    def sweep(self,x,yaw,v,w=0,uncertainty=.15,tolerance=30):
        return bounded_recovery_glass_sweep(self.map,Pose2D(x,0,yaw),v,w,
            uncertainty_m=uncertainty,heading_tolerance_deg=tolerance,
            pose_age_s=.1,ttl_ms=400,reserve_m=.15 if v else .03)

    def test_forward_away_or_parallel_uses_actual_heading_not_all_direction_travel_disk(self):
        away=self.sweep(1.05,0,400)
        parallel=self.sweep(1.12,math.pi/2,400)
        self.assertTrue(away['safe']);self.assertTrue(parallel['safe'])
        self.assertFalse(away['travel_double_counted'])
        self.assertGreaterEqual(parallel['interpolation_reserve_m'],0.)

    def test_heading_envelope_can_block_even_when_center_parallel_trajectory_is_clear(self):
        self.assertTrue(self.sweep(.98,math.pi/2,400,uncertainty=.15,tolerance=0)['safe'])
        self.assertFalse(self.sweep(.98,math.pi/2,400,uncertainty=.15,tolerance=30)['safe'])

    def test_toward_glass_and_exterior_halfplane_still_block(self):
        self.assertFalse(self.sweep(1.05,math.pi,400)['safe'])
        self.assertFalse(self.sweep(-2,0,400)['safe'])

    def test_recorded_174417_937mm_glass_distance_can_short_turn_without_forward_stopping_reserve(self):
        prior=NS(x_m=.9370486240085008,y_m=0,yaw_deg=0,
            radius_m=.2236024511023677,yaw_tolerance_deg=30.)
        result=assess_recovery_action(self.scan,self.map,prior,0,3500,180,
            directional_translation=True,clock=lambda:.1)
        self.assertTrue(result['safe'],result)
        self.assertAlmostEqual(result['glass']['translation_reserve_m'],.03)
        self.assertLess(result['glass']['required_boundary_distance_m'],.9370486240085008)

    def test_continuous_glass_check_does_not_exempt_real_map_obstacles(self):
        from grasp_range_integration.phase1.rotation_guard import map_rotation_evidence
        pose=Pose2D(.9370486240085008,0,0)
        radius=.44+.2236024511023677+.03
        self.assertFalse(map_rotation_evidence(self.map,pose,radius)['clear'])
        self.assertTrue(map_rotation_evidence(self.map,pose,radius,exact_glass_geometry=True)['clear'])
        col,row=self.map.world_to_grid([(pose.x_m,pose.y_m)])[0]
        for layer in ('occupied','unknown','temporary_occupied'):
            getattr(self.map,layer)[row,col]=True
            self.assertFalse(map_rotation_evidence(self.map,pose,radius,exact_glass_geometry=True)['clear'])
            getattr(self.map,layer)[row,col]=False
        self.assertFalse(map_rotation_evidence(self.map,Pose2D(.8,0,0),radius,
                                              exact_glass_geometry=True)['clear'])

    def test_near_real_external_return_still_blocks_short_turn(self):
        self.scan.distances_m[180]=.2
        prior=NS(x_m=1.3,y_m=0,yaw_deg=0,radius_m=.1,yaw_tolerance_deg=30.)
        value=assess_recovery_action(self.scan,self.map,prior,0,3500,180,
                                     directional_translation=True,clock=lambda:.1)
        self.assertFalse(value['safe']);self.assertEqual(value['reason'],'exterior_near_obstacle')

    def test_short_turn_too_near_wall_or_large_pose_uncertainty_still_blocks(self):
        self.assertFalse(self.sweep(.75,0,0,3500,uncertainty=.22)['safe'])
        self.assertFalse(self.sweep(.94,0,0,3500,uncertainty=.5)['safe'])

    def test_unbounded_startup_or_stale_scan_not_promoted_to_known_heading_sweep(self):
        self.assertFalse(self.sweep(2,0,400,tolerance=180)['safe'])
        prior=NS(x_m=.94,y_m=0,yaw_deg=0,radius_m=.22,yaw_tolerance_deg=180.)
        value=assess_recovery_action(self.scan,self.map,prior,0,3500,180,
                                    directional_translation=False,clock=lambda:.1)
        self.assertFalse(value['safe'])
        self.assertEqual(assess_recovery_action(self.scan,self.map,prior,400,0,400,
            directional_translation=True,clock=lambda:.4)['reason'],'startup_recovery_stale_scan')


if __name__=='__main__':unittest.main()
