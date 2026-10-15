"""Task scope gates perception/recovery, never substitutes for collision safety."""

from copy import deepcopy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image

from grasp_range_integration.phase1 import visibility_route_planner as planner
from grasp_range_integration.phase1.observation_recovery import CandidateRecovery, install_observation_adapter, candidate
from grasp_range_integration.phase1.relationship_find_only_entry import _phase1_directive
from raspberry_pi.known_map_navigation import KnownOccupancyMap


class TaskRegionGateTests(unittest.TestCase):
    def setUp(self):
        self.known_map = KnownOccupancyMap(np.full((200, 200), 255, dtype=np.uint8), .05, -5, -5)
        self.route = {'mode': 'semantic_visibility_regions', 'target_region': {
            'center_xy_m': [3., 0.], 'radius_m': .5, 'recognition_distance_m': [.45, 1.8],
            'minimum_work_area_coverage': .25, 'camera_horizontal_fov_deg': 82, 'fov_margin_deg': 10},
            'views': [{'view_xy_m': [0, 0]}, {'view_xy_m': [1.7, 0]}]}
        self.original_route, self.original_map, self.original_view = planner.LATEST_ROUTE, planner.LATEST_KNOWN_MAP, planner.ACTIVE_VIEW

    def tearDown(self):
        planner.LATEST_ROUTE, planner.LATEST_KNOWN_MAP, planner.ACTIVE_VIEW = self.original_route, self.original_map, self.original_view

    def assess(self, pose, **kwargs):
        return planner.task_region_view_assessment(self.route, pose, known_map=self.known_map, **kwargs)

    def test_far_start_and_near_wrong_heading_do_not_authorize_target_search(self):
        self.assertFalse(self.assess([0, 0, 0])['eligible'])
        self.assertFalse(self.assess([1.7, 0, math.pi])['eligible'])
        self.assertTrue(self.assess([1.7, 0, 0])['eligible'])

    def test_planned_station_or_model_confidence_cannot_replace_actual_pose(self):
        self.route['views'][0]['view_xy_m'] = [1.7, 0]
        self.route['views'][0]['visibility_zone'] = {'estimated_work_area_coverage': 1.0}
        self.assertFalse(self.assess([0, 0, 0])['eligible'])
        self.assertFalse(self.assess([])['eligible'])
        self.assertFalse(self.assess([math.nan, 0, 0])['eligible'])

    def test_intervening_wall_and_insufficient_contract_reject(self):
        cell = self.known_map.world_to_grid(((2., 0),))[0]
        self.known_map.occupied[:, cell[0]] = True
        self.assertFalse(self.assess([1.7, 0, 0])['eligible'])
        self.route['target_region'] = {}
        self.assertFalse(self.assess([1.7, 0, 0])['eligible'])

    def test_active_band_camera_margin_is_preserved(self):
        view = {'visibility_zone': {'camera_horizontal_fov_deg': 82, 'fov_margin_deg': 4}}
        wide = self.assess([1.7, 0, math.radians(30)], view=view)
        narrow = self.assess([1.7, 0, math.radians(30)])
        self.assertGreaterEqual(wide['estimated_work_area_coverage'], narrow['estimated_work_area_coverage'])

    def test_outside_region_bottom_candidate_cannot_requeue_or_fail_mission(self):
        recovery = CandidateRecovery()
        recovery.context = {'robot_pose': [0, 0, 0], 'task_region_view': {'eligible': False}}
        state = {'latest_event': {'type': 'observation_complete'}, 'current_view_index': 0,
                 'remaining_view_indices': [1], 'attempts': 1,
                 'detections': [{'bbox_1000': [450, 950, 478, 984], 'confidence': .99, 'camera_id': 'left'}]}
        before = deepcopy(self.route)
        for _ in range(2):
            self.assertIsNone(recovery.next(state, self.route))
        self.assertEqual(state['remaining_view_indices'], [1])
        self.assertEqual(self.route, before)
        self.assertEqual(recovery.last_status, 'approaching_task_region')
        directive = _phase1_directive(state, self.route, allow_low=False)
        self.assertEqual(directive['action'], 'navigate_next_view')
        self.assertEqual(directive['next_view_index'], 1)

    def test_outside_region_stereo_flags_cannot_report_task_success(self):
        detections = [{'camera_id': camera, 'confirmed_match': True, 'dual_camera_confirmed': True,
                       'stereo_pair_id': 'same', 'task_region_eligible': False} for camera in ('left', 'right')]
        state = {'remaining_view_indices': [1], 'detections': detections}
        self.assertEqual(_phase1_directive(state, self.route, allow_low=False)['action'], 'navigate_next_view')
        self.assertIsNone(candidate(detections))

    def test_perception_runs_outside_but_success_remains_gated_with_photos_preserved(self):
        planner.LATEST_ROUTE, planner.LATEST_KNOWN_MAP = self.route, self.known_map
        planner.ACTIVE_VIEW = self.route['views'][0]
        calls = []
        class GPU:
            def perceive(self, left, right, context):
                calls.append(context)
                return {'detections': [{'camera_id': 'left', 'bbox_1000': [450, 950, 478, 984]}]}
        class Camera:
            def capture(self, path):
                Image.new('RGB', (100, 80), 'grey').save(path)
                right = Path(path).with_name('right.jpg')
                Image.new('RGB', (100, 80), 'grey').save(right)
                return {'camera_count': 2, 'camera_2_image': right.name}
        class Log:
            def event(self, *args, **kwargs): return {}
        mission = SimpleNamespace(GpuMissionClient=GPU, MissionLog=Log,
                                  confirm_detections=lambda ds, *args: ds,
                                  _fuse_tracking_pose=lambda pose, result: pose)
        recovery = install_observation_adapter(mission, Camera)
        with tempfile.TemporaryDirectory() as temporary:
            left = Path(temporary) / 'view.jpg'
            Camera().capture(left)
            right = left.with_name('right.jpg')
            value = GPU().perceive(left, right, {'robot_pose': [0, 0, 0]})
            self.assertEqual(len(value['detections']), 1)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0]['candidate_only'])
            self.assertEqual(calls[0]['search_phase'], 'transit_candidate_observation')
            self.assertFalse(value['perception_skipped'])
            self.assertFalse(value['detections'][0]['confirmed_match'])
            self.assertFalse(value['detections'][0]['dual_camera_confirmed'])
            self.assertTrue(value['detections'][0]['transit_candidate_allowed'])
            self.assertFalse(value['detections'][0]['candidate_recovery_allowed'])
            self.assertFalse(recovery.context['task_region_view']['eligible'])
            self.assertTrue((left.parent / 'observations/obs_0001_left_raw.jpg').exists())
            planner.ACTIVE_VIEW = self.route['views'][1]
            value = GPU().perceive(left, right, {'robot_pose': [1.7, 0, 0]})
            self.assertEqual(len(calls), 2)
            self.assertTrue(recovery.context['task_region_view']['eligible'])
            checked = mission.confirm_detections(value['detections'], {}, {})
            self.assertTrue(checked[0]['task_region_eligible'])

    def test_grounded_transit_candidate_cannot_steal_approach_route(self):
        from test_phase1_observation_recovery import detection
        item = detection('left', [600, 210, 630, 245])
        item.update(task_region_eligible=False, transit_candidate_allowed=True)
        recovery = CandidateRecovery()
        recovery.context = {'robot_pose': [0, 0, 0], 'task_region_view': {'eligible': False}}
        state = {'latest_event': {'type': 'observation_complete'}, 'current_view_index': 0,
                 'remaining_view_indices': [1], 'attempts': 1, 'detections': [item]}
        before=deepcopy(self.route)
        for attempt in range(1,4):
            state['attempts']=attempt
            self.assertIsNone(recovery.next(state,self.route))
            self.assertEqual(state['remaining_view_indices'],[1])
            self.assertEqual(self.route,before)
            directive=_phase1_directive(state,self.route,allow_low=False)
            self.assertEqual(directive['next_view_index'],1)
            self.assertNotIn('recovery',directive)
        self.assertEqual(recovery.last_status,'transit_candidate_archived_continue_approach')

    def test_latest_start_candidate_regression_keeps_navigation_waypoint(self):
        from test_phase1_observation_recovery import detection
        item=detection('left',[640,635,670,665])
        item.update(confidence=.95,task_region_eligible=False,transit_candidate_allowed=True)
        recovery=CandidateRecovery()
        recovery.context={'robot_pose':[-1.33,-2.82,math.radians(116.4)],
                          'task_region_view':{'eligible':False,'estimated_work_area_coverage':0.}}
        state={'latest_event':{'type':'observation_complete'},'current_view_index':0,
               'remaining_view_indices':[1],'attempts':5,'detections':[item]}
        before=deepcopy(self.route)
        self.assertIsNone(recovery.next(state,self.route))
        self.assertEqual(self.route,before)
        self.assertEqual(state['remaining_view_indices'],[1])
        self.assertIsNone(recovery.previous)

    def test_eligible_region_still_allows_single_camera_recheck(self):
        from test_phase1_observation_recovery import detection
        item=detection('left',[640,635,670,665]);item['task_region_eligible']=True
        recovery=CandidateRecovery()
        recovery.context={'robot_pose':[1.7,0,0], 'task_region_view':{'eligible':True}}
        state={'latest_event':{'type':'observation_complete'},'current_view_index':1,
               'remaining_view_indices':[],'attempts':1,'detections':[item]}
        directive=recovery.next(state,self.route)
        self.assertEqual(directive['reason'],'single_camera_stationary_recheck')
        self.assertFalse(directive['recovery']['linear_motion_allowed'])

    def test_transit_stereo_cannot_complete_task_even_if_candidate_marked(self):
        from test_phase1_observation_recovery import detection
        items = [detection(camera) for camera in ('left', 'right')]
        for item in items:
            item.update(task_region_eligible=False, transit_candidate_allowed=True,
                        dual_camera_confirmed=True, confirmed_match=True, stereo_pair_id='pair')
        self.assertEqual(_phase1_directive({'remaining_view_indices': [1], 'detections': items},
                                          self.route, allow_low=False)['action'], 'navigate_next_view')


if __name__ == '__main__':
    unittest.main()
