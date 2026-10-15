"""Transit before perception; regress 20261002_143011 premature replan."""
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1 import visibility_route_planner as planner
from grasp_range_integration.phase1.navigation_avoidance import reserve_path
from grasp_range_integration.phase1.relationship_find_only_entry import (
    _install_phase1_supervisor, _install_waypoint_obstacle_guard)
from grasp_range_integration.phase1.localization_handoff import (
    recovery_baseline, deferred_recovery_pose, recovery_prediction)


class ApproachNavigationTests(unittest.TestCase):
    def setUp(self):
        self.old=(planner.LATEST_ROUTE,planner.LATEST_KNOWN_MAP,planner.ACTIVE_VIEW,
                  list(planner.VISITED_POINTS),planner.APPROACH_OBSERVATIONS,planner.OBSERVATIONS_COMPLETED)
        planner.VISITED_POINTS.clear()
        planner.APPROACH_OBSERVATIONS=planner.OBSERVATIONS_COMPLETED=0
        p=Path(__file__).resolve().parents[2]/'maps/semantic_release_20260919'
        if not (p/'ble_scene_17.pgm').is_file():
            p=Path(__file__).resolve().parents[2]
        self.map=KnownOccupancyMap.load(p/'ble_scene_17.pgm',p/'ble_scene_17.yaml',clearance_m=.30)
        original=self.map.path
        self.map.path=lambda a,b: reserve_path(self.map,a,b,original)
        self.request={'mode':'semantic_search_region','planning_policy':planner.POLICY,
                      'center_xy_m':[.8751325645044502,1.0443147835391624],
                      'radius_m':.95,'maximum_views':8}
        self.mission=NS(GpuMissionClient=NS())
        _install_phase1_supervisor(self.mission)

    def tearDown(self):
        (planner.LATEST_ROUTE,planner.LATEST_KNOWN_MAP,planner.ACTIVE_VIEW,
         visited,planner.APPROACH_OBSERVATIONS,planner.OBSERVATIONS_COMPLETED)=self.old
        planner.VISITED_POINTS[:]=visited

    def test_failed_run_position_has_transit_route_even_without_camera_view(self):
        before=self.map.traversable.copy()
        with patch.object(planner,'build_visibility_route',side_effect=AssertionError('premature perception')):
            route=planner.build_staged_route(self.map,(-1.368,-2.225),self.request)
        self.assertEqual(route['navigation_stage'],'approach')
        view=route['views'][0]
        self.assertNotIn('target_yaw_rad',view)
        self.assertGreater(view['path_length_m'],1.)
        self.assertEqual(view['view_xy_m'],view['path_xy_m'][-1])
        goal=view['view_xy_m'];center=self.request['center_xy_m']
        self.assertLessEqual(math.dist(goal,center),.95+1.45)
        for a,b in zip(view['path_xy_m'],view['path_xy_m'][1:]):
            # Every reserved path sample stays outside the object area.
            for t in np.linspace(0,1,30):
                self.assertGreaterEqual(math.dist((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])),center),1.75)
        np.testing.assert_array_equal(before,self.map.traversable)

    def test_start_does_not_require_target_capture_or_pivot(self):
        route=planner.build_staged_route(self.map,(-1.31,-2.8),self.request)
        self.assertEqual(len(route['views']),1)
        self.assertNotEqual(route['views'][0]['view_xy_m'],[-1.31,-2.8])
        with patch.object(planner,'_region_camera_coverage',side_effect=AssertionError('transit gate must not raycast')):
            gate=planner.task_region_view_assessment(route,[-1.31,-2.8,1.8])
        self.assertFalse(gate['eligible'])

    def test_near_region_uses_observation_planner(self):
        with patch.object(planner,'build_visibility_route',return_value={'mode':'actual_observation'}) as observe:
            route=planner.build_staged_route(self.map,(-.925,-.375),self.request)
        self.assertEqual(route['mode'],'actual_observation')
        observe.assert_called_once()

    def test_near_transit_checkpoint_can_inspect_current_camera_not_force_another_station(self):
        route=planner.build_staged_route(self.map,(-1.31,-2.8),self.request)
        with patch.object(planner,'_region_camera_coverage',return_value=.55):
            gate=planner.task_region_view_assessment(route,[-.9,0.,.5],known_map=self.map,view=route['views'][0])
        self.assertTrue(gate['eligible'])
        planner.LATEST_ROUTE=route
        recovery=NS(next=lambda *a:{'action':'inspect_current_view','terminal':False},last_status='locked')
        _install_phase1_supervisor(self.mission,recovery)
        state={'attempts':1,'current_view_index':0,'latest_event':{'type':'observation_complete'},
               'detections':[{'single_camera_match':True,'task_region_eligible':True}]}
        directive=self.mission.GpuMissionClient.next(None,state)
        self.assertEqual(directive['action'],'inspect_current_view')
        self.assertEqual(route['navigation_stage'],'observation')

    def test_untrusted_far_candidate_cannot_preempt_transit(self):
        planner.LATEST_ROUTE=planner.build_staged_route(self.map,(-1.31,-2.8),self.request)
        state={'attempts':1,'current_view_index':0,'latest_event':{'type':'observation_complete'},
               'detections':[{'single_camera_match':True,'task_region_eligible':False}]}
        directive=self.mission.GpuMissionClient.next(None,state)
        self.assertEqual(directive['action'],'replan')

    def test_no_view_near_region_tries_map_clear_alternate_perimeter(self):
        with patch.object(planner,'build_visibility_route',side_effect=planner.NoReachableVisibilityError('no view')):
            route=planner.build_staged_route(self.map,(-.925,-.375),self.request)
        self.assertEqual(route['navigation_stage'],'approach')
        self.assertGreater(math.dist(route['views'][0]['view_xy_m'],(-.925,-.375)),.35)

    def test_invalid_inputs_and_genuine_blocked_paths_are_not_hidden(self):
        with self.assertRaises(ValueError):
            planner.build_staged_route(self.map,(0,0),{'center_xy_m':[]})
        with patch.object(planner,'work_area_path',return_value=[]):
            with self.assertRaisesRegex(planner.NoReachableVisibilityError,'非相机视野失败'):
                planner.build_staged_route(self.map,(-1.31,-2.8),self.request)
        with patch.object(planner,'build_visibility_route',side_effect=RuntimeError('cancelled')):
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                planner.build_staged_route(self.map,(-.925,-.375),self.request)

    def test_far_segment_failure_does_not_exclude_future_station(self):
        planner.LATEST_ROUTE={'views':[{'view_xy_m':[0.,0.],
                    'visibility_zone':{'stop_rule':'turn_at_station_then_capture'}}]}
        state={'attempts':0,'remaining_view_indices':[0], 'latest_event':{
            'type':'navigation_no_progress','view_index':0,
            'recovery_cause':'navigation_turn_space_recovery','robot_pose':[-1.368,-2.225,1.47]}}
        directive=self.mission.GpuMissionClient.next(None,state)
        self.assertEqual(directive['action'],'replan')
        self.assertEqual(planner.VISITED_POINTS,[])

    def test_actual_station_block_can_still_select_another_station(self):
        planner.LATEST_ROUTE={'views':[{'view_xy_m':[0.,0.],
                    'visibility_zone':{'stop_rule':'turn_at_station_then_capture'}}]}
        state={'attempts':0,'remaining_view_indices':[0], 'latest_event':{
            'type':'navigation_no_progress','view_index':0,
            'recovery_cause':'navigation_turn_space_recovery','robot_pose':[.1,.1,1.47]}}
        self.mission.GpuMissionClient.next(None,state)
        self.assertEqual(planner.VISITED_POINTS,[(0.,0.)])

    def test_approach_photo_replans_once_without_target_failure_or_target_budget(self):
        planner.LATEST_ROUTE=planner.build_staged_route(self.map,(-1.31,-2.8),self.request)
        recovery=NS(next=lambda *a:self.fail('target recovery must not run in transit'),last_status='unused')
        _install_phase1_supervisor(self.mission,recovery)
        state={'attempts':1,'current_view_index':0,'remaining_view_indices':[],
               'latest_event':{'type':'observation_complete'}}
        for _ in range(2):
            directive=self.mission.GpuMissionClient.next(None,state)
            self.assertEqual(directive['action'],'replan')
            self.assertFalse(directive['terminal'])
        self.assertEqual(planner.APPROACH_OBSERVATIONS,1)
        self.assertIn(tuple(planner.LATEST_ROUTE['views'][0]['view_xy_m']),planner.VISITED_POINTS)


class DeferredRouteHandoffTests(unittest.TestCase):
    def setUp(self):
        self.source=NS(sequence=10)
        self.final=NS(sequence=20)
        self.future=NS(sequence=21)
        self.localizer=object()
        self.result=NS(pose=Pose2D(.15,.2,.1))
        self.h=dict(result=self.result,source_scan=self.source,final_scan=self.final,
                    localizer=self.localizer,predicted_pose=Pose2D(.14,.2,.1),
                    pose_consumed=False,baseline_consumed=False,deferred_prediction=True)
        self.mission=NS(_phase1_localization_handoff=self.h)

    def test_route_recovery_new_pose_then_only_one_adjacent_increment(self):
        baseline=recovery_baseline(self.mission,self.localizer,self.source,self.future)
        self.assertIs(baseline,self.final)
        old=Pose2D(-1,-2,0)
        base=deferred_recovery_pose(self.mission,old)
        self.assertIs(base,self.result.pose)
        next_pose=base.compose(Pose2D(.02,0,.01))
        self.assertLess(math.dist((next_pose.x_m,next_pose.y_m),(.15,.2)),.021)
        self.assertIs(deferred_recovery_pose(self.mission,next_pose),next_pose)
        self.assertEqual(recovery_prediction(self.mission,self.result,next_pose),(next_pose,None))
        self.assertIs(recovery_baseline(self.mission,self.localizer,self.source,self.future),self.source)

    def test_old_in_track_handoff_is_still_strict(self):
        self.h['deferred_prediction']=False
        with self.assertRaisesRegex(RuntimeError,'位姿尚未交接'):
            recovery_baseline(self.mission,self.localizer,self.source,self.future)

    def test_no_pose_update_before_fresh_scan_baseline(self):
        with self.assertRaisesRegex(RuntimeError,'扫描基准尚未交接'):
            deferred_recovery_pose(self.mission,Pose2D())
        with self.assertRaisesRegex(RuntimeError,'雷达基准尚未更新'):
            recovery_baseline(self.mission,self.localizer,self.source,self.final)

    def test_frozen_path_rebuild_uses_recovered_start_only_until_pose_consumed(self):
        self.h['old_route_start']=(-1.,-2.)
        m=NS(_phase1_route_handoff=self.h)
        with patch.object(planner,'work_area_path',wraps=planner.work_area_path):
            # A stationary request at the recovered start exits before map IO.
            path=planner.work_area_path(m,(-1.,-2.),(.15,.2),(3.,3.),.5,original=lambda *a:[])
        self.assertEqual(path,[(.15,.2)])
        self.h['pose_consumed']=True
        with self.assertRaises(AttributeError):
            planner.work_area_path(m,(-1.,-2.),(.15,.2),(3.,3.),.5,original=lambda *a:[])

    def test_outer_route_guard_recovers_without_rethrowing_camera_failure(self):
        calls=[]
        def build(m,start,request):
            calls.append(start)
            if len(calls)==1:raise planner.NoReachableVisibilityError('no route')
            return {'views':[{'view_xy_m':[2.,2.]}]}
        mission=NS(_obstacle_safe=lambda *a:True,build_search_route=build,
                   _fuse_tracking_pose=lambda p,r:r.pose,_phase1_navigation_scan=self.source)
        def recover(scan,pose,accept_candidate):
            self.assertIs(scan,self.source)
            self.assertTrue(accept_candidate(self.result))
            mission._phase1_localization_handoff=self.h
            return self.result
        mission._phase1_recover_route=recover
        _install_waypoint_obstacle_guard(mission)
        mission._fuse_tracking_pose(Pose2D(-1,-2,0),NS(pose=Pose2D(-1,-2,0),score=1.,rmse_m=.04,inlier_ratio=.8))
        m=NS()
        route=mission.build_search_route(m,(-1.,-2.),{})
        self.assertTrue(route['views'])
        self.assertEqual(calls,[(-1.,-2.),(.15,.2)])
        self.assertIs(m._phase1_route_handoff,self.h)
        self.assertEqual(self.h['old_route_start'],(-1.,-2.))

    def test_valid_route_with_persistent_blind_turn_enters_motion_recovery_not_same_replan(self):
        from grasp_range_integration.phase1 import rotation_guard, turn_policy
        events=[];builds=[];recoveries=[]
        candidate=NS(pose=Pose2D(-1,-2,0),score=1.,rmse_m=.04,inlier_ratio=.8)
        class Localizer:
            def track(self,*a):return candidate
        class Log:
            def __init__(self):pass
            def event(self,name,**fields):events.append((name,fields))
        def build(m,start,request):
            builds.append(start)
            return {'views':[{'view_xy_m':[2.,2.]}]}
        mission=NS(_obstacle_safe=lambda *a,**k:True,build_search_route=build,
            _fuse_tracking_pose=lambda p,r:r.pose,ScanMapLocalizer=Localizer,
            _strict_command_for_pose=lambda *a:(0,1200,False),
            _remember_temporary_obstacle=lambda *a:0,MissionLog=Log,
            _phase1_navigation_scan=self.source)
        def recover(scan,pose,accept_candidate):
            recoveries.append(scan)
            self.assertTrue(accept_candidate(self.result))
            mission._phase1_localization_handoff=self.h
            return self.result
        mission._phase1_recover_route=recover
        _install_waypoint_obstacle_guard(mission)
        m=NS();scan=NS(sequence=11,received_s=10.)
        with patch('grasp_range_integration.phase1.relationship_find_only_entry.time.monotonic',return_value=10.), \
             patch.object(rotation_guard,'assess_rotation',return_value={
                 'safe':False,'reason':'unobserved_sector_without_safe_map_support'}):
            mission.build_search_route(m,(-1,-2),{})
            Localizer().track(scan,candidate.pose)
            mission._fuse_tracking_pose(candidate.pose,candidate)
            for _ in range(8):mission._strict_command_for_pose(NS(),candidate.pose,(2.,2.))
            self.assertFalse(mission._obstacle_safe(scan,0,.45))
            mission._remember_temporary_obstacle()
            Log().event('temporary_obstacle',view_index=0)
            self.assertEqual(events[-1][0],'navigation_no_progress')
            mission.build_search_route(m,(-1,-2),{})
        self.assertEqual(recoveries,[self.source])
        self.assertEqual(builds,[(-1,-2),(.15,.2)])
        turn_policy.LAST_REQUEST=(0,0)


if __name__=='__main__':unittest.main()
