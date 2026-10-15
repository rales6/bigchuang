"""No hardware/network: green single-eye recovery and classifier abstention."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from grasp_range_integration.phase1.colour_target_review import verify_green_cues
from grasp_range_integration.phase1.observation_recovery import (
    colour_visual_cues, candidate, install_observation_adapter, reject_copied_stereo_boxes)


class GreenReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.images = {}
        for camera in ('left', 'right'):
            image = Image.new('RGB', (1280, 720), (105, 105, 105))
            if camera == 'right':
                ImageDraw.Draw(image).rectangle((148, 120, 188, 152), fill=(35, 115, 55))
            path = Path(self.tmp.name) / (camera + '.jpg');image.save(path);self.images[camera] = path
        self.cues = {i['camera_id']: i for i in colour_visual_cues(self.images, 'green')}
        self.config = SimpleNamespace(configured=True, api_key='fake', model='fake',
                                      base_url='https://example.invalid', timeout_s=1)

    def response(self, right=True):
        return {'choices': [{'message': {'content': json.dumps({
            'left': {'matches_target': False, 'confidence': .93},
            'right': {'matches_target': right, 'confidence': .94}, 'same_object': True})}}]}

    def test_no_green_pixels_in_left_does_not_create_left_candidate(self):
        self.assertEqual(set(self.cues), {'right'})
        self.assertFalse(self.cues['right']['confirmed_match'])
        self.assertEqual(candidate(list(self.cues.values()))['camera_id'], 'right')

    def test_single_eye_review_cannot_become_stereo_even_if_model_claims_same_object(self):
        calls = []
        result = verify_green_cues(self.images, self.cues, config=self.config,
                                  transport=lambda req: calls.append(req) or self.response())
        self.assertEqual(result['accepted_cameras'], ['right']);self.assertFalse(result['same_object'])
        self.assertEqual(len(calls), 2)

    def test_disagreeing_reviews_abstain(self):
        replies = iter([self.response(), self.response(False)])
        result = verify_green_cues(self.images, self.cues, config=self.config,
                                  transport=lambda req: next(replies))
        self.assertEqual(result['accepted_cameras'], [])

    def test_rejected_camera_does_not_reduce_valid_camera_confidence(self):
        image = Image.new('RGB', (1280,720), (105,105,105))
        ImageDraw.Draw(image).rectangle((350,142,395,187), fill=(35,115,55))
        image.save(self.images['left'])
        cues = {i['camera_id']:i for i in colour_visual_cues(self.images,'green')}
        reply = self.response();value = json.loads(reply['choices'][0]['message']['content'])
        value['left']['confidence'] = .1
        reply['choices'][0]['message']['content'] = json.dumps(value)
        result = verify_green_cues(self.images,cues,config=self.config,transport=lambda req:reply)
        self.assertEqual(result['accepted_cameras'],['right'])
        self.assertEqual(result['minimum_confidence'],.94)

    def test_network_failure_is_candidate_not_terminal_error(self):
        def unavailable(req): raise OSError('offline')
        result = verify_green_cues(self.images, self.cues, config=self.config, transport=unavailable)
        self.assertEqual(result['accepted_cameras'], [])
        self.assertIsNotNone(candidate(list(self.cues.values())))

    def test_invalid_box_and_duplicate_images_never_invoke_api(self):
        cues = dict(self.cues);cues['right'] = {**cues['right'], 'bbox_1000': [float('nan'), 0, 20, 30]}
        no_call = lambda req: self.fail('Unexpected model call')
        self.assertEqual(verify_green_cues(self.images, cues, config=self.config,
                                        transport=no_call)['reason'], 'invalid_pixel_candidate')
        both = {'left': self.cues['right'], 'right': self.cues['right']}
        self.assertEqual(verify_green_cues({'left':self.images['right'], 'right':self.images['right']},
            both, config=self.config, transport=no_call)['reason'], 'duplicate_camera_images')

    def test_copied_model_boxes_cannot_choose_wrong_steering_eye(self):
        detections = [{'source':'qwen', 'camera_id':c, 'bbox_1000':[100,150,140,180],
            'dual_camera_confirmed':True, 'single_camera_match':True, 'confirmed_match':True,
            'search_evidence':{'source_ok':True,'class_ok':True,'colour_ok':True,'bbox_valid':True}}
            for c in ('left', 'right')]
        checked = reject_copied_stereo_boxes(detections)
        self.assertIsNone(candidate(checked))

    def test_adapter_uses_actual_right_pixels_and_caches_review_across_both_passes(self):
        import raspberry_pi.object_search_mission as core
        mission = SimpleNamespace(GpuMissionClient=type('Gpu', (), {'perceive': lambda *a: {}}),
            MissionLog=type('Log', (), {'event': lambda *a, **k: None}),
            confirm_detections=core.confirm_detections, _same_stereo_target=core._same_stereo_target,
            _fuse_tracking_pose=lambda *a: None)
        camera = type('Camera', (), {'capture': lambda *a: {}})
        recovery = install_observation_adapter(mission, camera)
        recovery.context = {'task_region_view': {'eligible': True}}
        detections = [{'source':'qwen','camera_id':c,'class_name':'block','confidence':.95,
                       'attributes':{'color':'green'},'bbox_1000':[100,150,140,180]}
                      for c in ('left', 'right')]
        verdict = {'accepted_cameras':['right'],'same_object':False,'minimum_confidence':.92}
        with patch('grasp_range_integration.phase1.colour_target_review.verify_green_cues',
                   return_value=verdict) as review:
            checked = mission.confirm_detections(detections, {'class_name':'block','color':'green',
                'detection_mode':'qwen_fallback'}, self.images)
            checked = mission.confirm_detections(checked, {'class_name':'block','color':'green',
                'detection_mode':'qwen_fallback'}, self.images)
        self.assertEqual(review.call_count, 1)
        self.assertEqual(candidate(checked)['camera_id'], 'right')
        self.assertFalse(any(i.get('dual_camera_confirmed') for i in checked))

    def test_two_real_pixel_boxes_can_complete_confirmation_without_model_coordinates(self):
        import raspberry_pi.object_search_mission as core
        for c, x in [('left',350),('right',284)]:
            image = Image.new('RGB', (1280,720), (105,105,105))
            ImageDraw.Draw(image).rectangle((x,142,x+45,187), fill=(35,115,55))
            image.save(self.images[c])
        mission = SimpleNamespace(GpuMissionClient=type('Gpu', (), {'perceive':lambda *a:{}}),
            MissionLog=type('Log', (), {'event':lambda *a, **k:None}),
            confirm_detections=core.confirm_detections, _same_stereo_target=core._same_stereo_target,
            _fuse_tracking_pose=lambda *a:None)
        recovery = install_observation_adapter(mission, type('Camera', (), {'capture':lambda *a:{}}))
        recovery.context = {'task_region_view': {'eligible':True}}
        verdict = {'accepted_cameras':['left','right'],'same_object':True,'minimum_confidence':.92}
        with patch('grasp_range_integration.phase1.colour_target_review.verify_green_cues',
                   return_value=verdict) as review:
            first = mission.confirm_detections([], {'class_name':'block','color':'green'}, self.images)
            second = mission.confirm_detections(first, {'class_name':'block','color':'green'}, self.images)
        pairs = [i for i in second if i.get('dual_camera_confirmed')]
        self.assertEqual(len(pairs), 2);self.assertEqual(review.call_count, 1)
        self.assertNotEqual(pairs[0]['bbox_1000'], pairs[1]['bbox_1000'])
        self.assertTrue(all(i['search_evidence']['pursuit_entry']['ready'] for i in pairs))

    def test_green_lock_is_reviewed_after_small_yaw_lowers_semantic_coverage(self):
        import raspberry_pi.object_search_mission as core
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        mission = SimpleNamespace(GpuMissionClient=type('Gpu', (), {'perceive':lambda *a:{}}),
            MissionLog=type('Log', (), {'event':lambda *a, **k:None}),
            confirm_detections=core.confirm_detections, _same_stereo_target=core._same_stereo_target,
            _fuse_tracking_pose=lambda *a:None)
        recovery = install_observation_adapter(mission, type('Camera', (), {'capture':lambda *a:{}}))
        recovery.previous = dict(self.cues['right'])
        recovery.target_lock = {'scope': 'not-used-by-perceive'}
        previous = (planner.LATEST_ROUTE, planner.ACTIVE_VIEW)
        planner.LATEST_ROUTE = {'target_region': {'recognition_distance_m': [.45, 1.8]}}
        planner.ACTIVE_VIEW = {'view_xy_m': [0., 0.], 'visibility_zone': {
            'stop_rule': 'single_camera_locked_yaw_alignment'}}
        verdict = {'accepted_cameras':['right'], 'same_object':False, 'minimum_confidence':.94}
        try:
            with patch.object(planner, 'task_region_view_assessment', return_value={
                    'eligible':False, 'region_center_distance_m':1.81,
                    'estimated_work_area_coverage':.23}), patch(
                    'grasp_range_integration.phase1.colour_target_review.verify_green_cues',
                    return_value=verdict) as review:
                result = mission.GpuMissionClient().perceive(self.images['left'], self.images['right'], {
                    'robot_pose':[0., 0., .1], 'target':{'class_name':'block','color':'green'}})
            self.assertEqual(result['task_region_view']['reason'], 'live_green_reacquisition_at_station')
            self.assertTrue(result['task_region_view']['eligible'])
            review.assert_called_once()
            self.assertTrue(any(d.get('single_camera_match') for d in result['detections']))
            self.assertFalse(any(d.get('dual_camera_confirmed') for d in result['detections']))
        finally:
            planner.LATEST_ROUTE, planner.ACTIVE_VIEW = previous


if __name__ == '__main__': unittest.main()
