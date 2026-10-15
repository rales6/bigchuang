"""Software-only replay of the 20261002_163709 transit pause."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.glass_boundary_guard import DEFAULT_CONFIG, attach_boundaries
from grasp_range_integration.phase1.map_margin_egress import MapMarginEgress
from grasp_range_integration.phase1.relationship_find_only_entry import _install_waypoint_obstacle_guard
from grasp_range_integration.phase1 import visibility_route_planner as planner, turn_policy


class MapMarginEgressTests(unittest.TestCase):
    def setUp(self):
        f=json.loads((Path(__file__).parent/'fixtures/map_margin_stop_20261002.json').read_text(encoding='utf-8'))
        self.assessment=copy.deepcopy(f['assessment'])
        root=Path(__file__).resolve().parents[2]
        p=root/'maps/semantic_release_20260919/ble_scene_17.pgm'
        if not p.exists():p=root/'ble_scene_17.pgm'
        self.map=KnownOccupancyMap.load(p)
        attach_boundaries(self.map,json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8')))
        p=f['assessment']['map_support']['pose']
        self.pose=NS(x_m=p[0],y_m=p[1],yaw_rad=p[2])
        self.scan=NS(angles_rad=np.asarray(f['angles_rad']),distances_m=np.asarray(f['distances_m']),received_s=10.)
        self.context={'map':self.map,'pose':self.pose,'reliable':True,'updated':10.,
            'quality':f['assessment']['pose_quality'],
            'bridge_uncertainty_m':f['assessment']['pose_quality']['uncertainty_m']}
        self.route={'navigation_stage':'approach','target_region':{
            'center_xy_m':[.8751325645044502,1.0443147835391624],'radius_m':.95}}
        self.policy=MapMarginEgress()

    def select(self):
        return self.policy.select(self.context,self.scan,self.route,self.assessment,clock=lambda:10.1)

    def test_real_stopped_frame_selects_translation_not_rotation(self):
        decision=self.select()
        self.assertTrue(decision['safe'],decision)
        self.assertEqual(decision['angular_mrad_s'],0)
        self.assertFalse(decision['map_pose_confirmed'])
        self.assertFalse(decision['evidence']['rear_turn_authorized'])
        self.assertFalse(decision['evidence']['operator_clear'])

    def test_must_have_body_support_and_only_occupied_shell(self):
        for field,value in [('installed_body_blind_arc',False),('map_pose_reliable',False),
                            ('reason','exterior_near_obstacle')]:
            with self.subTest(field=field):
                original=self.assessment[field];self.assessment[field]=value
                self.assertFalse(self.select()['safe']);self.assessment[field]=original
        for name in ('unknown','temporary_occupied','verified_glass'):
            self.assessment['map_support']['layers'][name]['count']=1
            self.assertFalse(self.select()['safe'])
            self.assessment['map_support']['layers'][name]['count']=0

    def test_actual_map_body_overlap_still_blocks(self):
        c,r=self.map.world_to_grid(((self.pose.x_m,self.pose.y_m),))[0]
        self.map.occupied[r,c]=True
        self.assertFalse(self.select()['safe'])

    def test_front_object_and_missing_front_beams_block(self):
        angle=(self.scan.angles_rad+math.pi)%(2*math.pi)-math.pi
        self.scan.distances_m[np.argmin(abs(angle))]=.2
        self.assertFalse(self.select()['safe'])
        self.scan.distances_m[:]=np.nan
        self.assertFalse(self.select()['safe'])

    def test_continuous_glass_boundary_still_blocks(self):
        self.pose.x_m=-2.7
        self.assertFalse(self.select()['safe'])

    def test_stale_unreliable_or_unbounded_pose_blocks(self):
        self.context['reliable']=False
        self.assertFalse(self.select()['safe'])
        self.context['reliable']=True;self.context['updated']=9.
        self.assertFalse(self.select()['safe'])
        self.context['updated']=10.;self.scan.received_s=9.
        self.assertFalse(self.select()['safe'])
        self.scan.received_s=10.;self.context['bridge_uncertainty_m']=.36
        self.assertFalse(self.select()['safe'])

    def test_not_for_target_observation_or_approach_into_work_area(self):
        self.route['navigation_stage']='observation'
        self.assertFalse(self.select()['safe'])
        self.route['navigation_stage']='approach'
        self.route['target_region']['center_xy_m']=[self.pose.x_m+.8,self.pose.y_m]
        self.assertFalse(self.select()['safe'])

    def test_no_stall_rearming_or_unbounded_escape(self):
        self.assertTrue(self.select()['safe'])
        self.context['updated']=16.1;self.scan.received_s=16.1
        decision=self.policy.select(self.context,self.scan,self.route,self.assessment,clock=lambda:16.2)
        self.assertFalse(decision['safe'])
        self.assertTrue(self.policy.exhausted)
        self.assertFalse(self.policy.select(self.context,self.scan,self.route,self.assessment,clock=lambda:16.2)['safe'])

    def test_validated_translation_changes_view_before_normal_turn_retry(self):
        self.assertTrue(self.select()['safe'])
        self.pose.y_m+=.26
        self.assertFalse(self.select()['safe'])
        self.assertTrue(self.policy.exhausted)

    def test_does_not_modify_maps_or_raw_returns(self):
        layers=[self.map.occupied.copy(),self.map.unknown.copy(),self.map.temporary_occupied.copy()]
        ranges=self.scan.distances_m.copy()
        self.assertTrue(self.select()['safe'])
        for before,after in zip(layers,(self.map.occupied,self.map.unknown,self.map.temporary_occupied)):
            np.testing.assert_array_equal(before,after)
        np.testing.assert_array_equal(ranges,self.scan.distances_m)

    def test_entry_dispatches_egress_instead_of_zero_speed_wait(self):
        route_before=planner.LATEST_ROUTE
        try:
            planner.LATEST_ROUTE=self.route
            class Log:
                def event(self,*a,**k):return k
            class Localizer:
                def track(inner,scan,pose):return result
            result=NS(pose=self.pose,score=-1.,rmse_m=.013,inlier_ratio=.995)
            mission=NS(ScanMapLocalizer=Localizer,MissionLog=Log,
                build_search_route=lambda *a,**k:self.route,
                _strict_command_for_pose=lambda *a,**k:(0,1200,False),
                _fuse_tracking_pose=lambda *a:self.pose,
                _obstacle_safe=lambda *a,**k:True,_remember_temporary_obstacle=lambda *a:0,
                _phase1_navigation_estimate={'result':result,'started_s':10.,'uncertainty_m':.1096})
            _install_waypoint_obstacle_guard(mission)
            mission.build_search_route(self.map,(self.pose.x_m,self.pose.y_m),{})
            with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic',return_value=10.1):
                Localizer().track(self.scan,self.pose)
                mission._fuse_tracking_pose(self.pose,result)
                self.assertEqual(mission._strict_command_for_pose(),(400,0,False))
                self.assertEqual(turn_policy.LAST_REQUEST,(400,0))
                self.assertEqual(mission._strict_command_for_pose(),(400,0,False))
        finally:
            planner.LATEST_ROUTE=route_before
            turn_policy.LAST_REQUEST=(0,0)


if __name__=='__main__':unittest.main()
