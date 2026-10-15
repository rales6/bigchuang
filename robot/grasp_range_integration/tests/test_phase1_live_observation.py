"""Offline transit perception, track age/identity, preemption and guard wiring."""
from copy import deepcopy
import math
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from PIL import Image,ImageDraw
from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1.live_observation import (
    BearingTracks,LiveObserver,LiveSettings,review_moving_frame,install_live_observation,
    template_cue,guided_segment_clear,load_settings,classify_transit)


def cue(eye='left',box=None):
    return dict(camera_id=eye,bbox_1000=box or [600.,250.,640.,295.],
        complete_in_frame=True,source='phase1_green_visual_cue')


class TrackTests(unittest.TestCase):
    def test_other_object_class_uses_unique_model_crop_match_on_actual_current_image(self):
        import numpy as np
        pixels=np.random.default_rng(42).integers(20,230,(30,32,3),dtype=np.uint8)
        scene=np.full((240,320,3),110,dtype=np.uint8);scene[80:110,170:202]=pixels
        with TemporaryDirectory() as directory:
            path=Path(directory)/'camera.png';Image.fromarray(scene).save(path)
            row=template_cue(path,dict(pixels=pixels),'left')
            self.assertIsNotNone(row)
            self.assertAlmostEqual(row['bbox_1000'][0],170/320*1000)
            scene[80:110,70:102]=pixels
            Image.fromarray(scene).save(path)
            self.assertIsNone(template_cue(path,dict(pixels=pixels),'left'))

    def test_live_settings_reject_unbounded_or_invalid_parameters(self):
        for settings in [dict(capture_interval_s=0),dict(maximum_frame_age_s=100),
                         dict(guidance_step_m=1),dict(maximum_guidance_deg=float('nan'))]:
            with self.assertRaises(ValueError):LiveSettings(**settings)

    def test_default_configuration_is_valid_and_unknown_keys_rejected(self):
        self.assertEqual(load_settings().capture_interval_s,1.5)
        with TemporaryDirectory() as directory:
            p=Path(directory)/'settings.json';p.write_text('{"skip_obstacles":true}')
            with self.assertRaises(ValueError):load_settings(p)

    def test_visible_colour_is_not_itself_object_classification(self):
        row=BearingTracks().update([cue()],[0,0,0],1)['left']
        self.assertNotIn('confirmed_match',row)

    def test_rotation_compensated_continuous_track_retains_identity(self):
        t=BearingTracks();a=t.update([cue()],[0,0,0],1)['left']
        b=t.update([cue()],[.1,0,.1],2)['left']
        self.assertEqual(a['track_id'],b['track_id'])

    def test_lost_reappearing_or_ambiguous_targets_do_not_inherit_identity(self):
        t=BearingTracks();a=t.update([cue()],[0,0,0],1)['left']
        self.assertEqual(t.update([cue(),cue(box=[200,250,240,295])],[0,0,0],2),{})
        b=t.update([cue()],[0,0,0],3)['left']
        self.assertNotEqual(a['track_id'],b['track_id'])
        t.update([],[0,0,0],4)
        self.assertNotEqual(t.update([cue()],[0,0,0],5)['left']['track_id'],b['track_id'])

    def test_large_displacement_jump_or_stale_capture_starts_new_identity(self):
        for p,stamp,box in [([1,0,0],2,None),([0,0,0],5,None),([0,0,0],2,[50,250,90,295])]:
            t=BearingTracks();a=t.update([cue()],[0,0,0],1)['left']
            b=t.update([cue(box=box)],p,stamp)['left']
            self.assertNotEqual(a['track_id'],b['track_id'])


class LivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.events=[];self.pose=Pose2D(0,0,0)
        self.route=dict(mode='semantic_visibility_regions',target_region={
            'center_xy_m':[2,0],'radius_m':.7,'recognition_distance_m':[.45,1.8]})
        self.o=LiveObserver(NS(),clock=lambda:10.)
        self.o.emit=lambda name,**kw:self.events.append((name,kw))
        self.tracks=BearingTracks().update([cue()],[0,0,0],9.5)
        self.o.latest=dict(stamp=9.5,pose=[0,0,0],tracks=self.tracks,sequence=1)
        self.o.proof=dict(frame_stamp=5.,classified={
            'left':{'track_id':self.tracks['left']['track_id']}},stereo_pair_id=None)

    def choose(self,**kw):
        return self.o.select(kw.get('pose',self.pose),self.route,NS(),schedule=False)

    def test_live_single_eye_guides_a_map_checked_segment_instead_of_old_waypoint(self):
        with patch('grasp_range_integration.phase1.live_observation.guided_segment_clear',return_value=True) as path:
            action=self.choose()
        self.assertEqual(action['action'],'guided_waypoint')
        self.assertEqual(action['camera'],'left')
        self.assertLess(action['step_deg'],0.)
        self.assertLessEqual(abs(action['step_deg']),15)
        self.assertFalse(action['object_depth_estimated']);path.assert_called_once()

    def test_slow_model_can_attach_only_to_the_same_unbroken_current_track(self):
        self.o.latest['tracks']['left']['track_id']+=1
        self.assertEqual(self.choose()['action'],'route')

    def test_stale_frame_or_expired_identity_returns_to_route(self):
        self.o.latest['stamp']=7.9
        self.assertEqual(self.choose()['action'],'route')
        self.o.latest['stamp']=9.5;self.o.proof['frame_stamp']=-11.
        self.assertEqual(self.choose()['action'],'route')

    def test_only_pixels_or_only_one_classified_eye_cannot_preempt_as_stereo(self):
        self.o.proof=None;self.assertEqual(self.choose()['action'],'route')
        self.setUp()
        with patch('grasp_range_integration.phase1.live_observation.guided_segment_clear',return_value=True):
            self.assertEqual(self.choose()['action'],'guided_waypoint')

    def test_two_current_model_confirmed_tracks_request_fresh_stopped_review_not_success(self):
        right=dict(self.tracks['left'],camera_id='right',track_id=2)
        self.o.latest['tracks']['right']=right
        self.o.proof['classified']['right']={'track_id':2};self.o.proof['stereo_pair_id']='pair'
        self.assertEqual(self.choose()['action'],'stopped_recheck')
        self.assertFalse(self.events[-1][1]['task_success_allowed'])
        self.assertEqual(self.events[-1][1]['next_action'],'fresh_stopped_stereo_and_tracker_entry_check')

    def test_obstacle_on_visual_segment_requests_review_not_unchecked_drive(self):
        with patch('grasp_range_integration.phase1.live_observation.guided_segment_clear',return_value=False):
            self.assertEqual(self.choose()['action'],'stopped_recheck')

    def test_far_start_does_not_turn_for_unrelated_object(self):
        self.assertEqual(self.choose(pose=Pose2D(-2,-3,0))['action'],'route')

    def test_stopped_capture_invalidates_old_classifier_and_handoff_latch(self):
        self.o.handoff=True;old=self.o.epoch;self.o.stopped_capture()
        self.assertIsNone(self.o.proof);self.assertFalse(self.o.handoff)
        self.assertEqual(self.o.epoch,old+1)

    def test_close_disables_future_results(self):
        self.o.close();self.assertTrue(self.o.closed);self.assertIsNone(self.o.proof)


class MovingModelReviewTests(unittest.TestCase):
    def setUp(self):
        import raspberry_pi.object_search_mission as core
        self.confirm=core.confirm_detections;self.same=core._same_stereo_target
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.images={};self.cues=[]
        from grasp_range_integration.phase1.observation_recovery import colour_visual_cues
        for c,x in [('left',720),('right',1010)]:
            image=Image.new('RGB',(1280,720),(110,110,110))
            ImageDraw.Draw(image).rectangle((x,130,x+32,162),fill=(30,120,50))
            p=Path(self.tmp.name)/(c+'.jpg');image.save(p);self.images[c]=p
        self.cues=colour_visual_cues(self.images,'green')
        self.target={'class_name':'block','color':'green','detection_mode':'qwen_fallback'}
        self.verdict=dict(accepted_cameras=['left','right'],same_object=True,minimum_confidence=.9)

    def review(self,raw,verdict=None):
        return review_moving_frame(raw,self.target,self.images,self.cues,confirm=self.confirm,
            same_target=self.same,verify_green=lambda *a:verdict or self.verdict)

    def test_174417_copied_qwen_boxes_use_ds_to_classify_actual_distinct_pixels(self):
        raw=[dict(source='qwen',class_name='block',camera_id=c,confidence=.85,
                  bbox_1000=[275,144,300,170],attributes={'color':'green'}) for c in ('left','right')]
        checked,review=self.review(raw)
        pairs=[i for i in checked if i.get('dual_camera_confirmed')]
        self.assertEqual(len(pairs),2)
        self.assertTrue(all(i['source']=='phase1_deepseek_green_block' for i in pairs))
        self.assertNotEqual(pairs[0]['bbox_1000'],pairs[1]['bbox_1000'])

    def test_174417_wrong_right_coordinates_and_missing_left_are_not_directly_trusted(self):
        raw=[dict(source='qwen',class_name='block',camera_id='right',confidence=.91,
                  bbox_1000=[550,248,585,285],attributes={'color':'green'})]
        checked,_=self.review(raw)
        self.assertFalse(checked[0]['single_camera_match'])
        self.assertEqual(len([i for i in checked if i.get('dual_camera_confirmed')]),2)

    def test_transit_green_pixels_use_deepseek_before_slow_or_wrong_box_qwen(self):
        mission=NS(_phase1_raw_confirm=self.confirm,_same_stereo_target=self.same,
            _phase1_raw_perceive=lambda *a:self.fail('DS verified, Qwen not needed'))
        with patch('grasp_range_integration.phase1.colour_target_review.verify_green_cues',return_value=self.verdict) as ds:
            checked,_=classify_transit(mission,NS(timeout_s=90),self.target,self.images,self.cues,{})
        self.assertEqual(ds.call_args.kwargs['request_timeout_s'],8.)
        self.assertEqual(len([i for i in checked if i.get('dual_camera_confirmed')]),2)

    def test_background_primary_model_failure_returns_no_target_without_mutating_stopped_timeout(self):
        mission=NS(_phase1_raw_confirm=self.confirm,_same_stereo_target=self.same,
            _phase1_raw_perceive=lambda *a:(_ for _ in ()).throw(TimeoutError('model timeout')))
        client=NS(timeout_s=90)
        checked,_=classify_transit(mission,client,{'class_name':'cup'},self.images,[],{})
        self.assertEqual(checked,[]);self.assertEqual(client.timeout_s,90)

    def test_ds_abstention_does_not_turn_pixels_into_found_object(self):
        checked,_=self.review([],dict(accepted_cameras=[],same_object=False))
        self.assertFalse(any(i.get('single_camera_match') for i in checked))

    def test_ds_single_eye_cannot_complete_stereo(self):
        checked,_=self.review([],dict(accepted_cameras=['right'],same_object=False,minimum_confidence=.9))
        self.assertEqual(len([i for i in checked if i.get('single_camera_match')]),1)
        self.assertFalse(any(i.get('dual_camera_confirmed') for i in checked))

    def test_duplicate_frame_content_cannot_steer_or_complete_stereo(self):
        self.images['right'].write_bytes(self.images['left'].read_bytes())
        checked,_=self.review([])
        self.assertFalse(any(i.get('single_camera_match') for i in checked))

    def test_stopped_adapter_also_reviews_pixels_when_coverage_gate_and_bad_qwen_boxes_block(self):
        import raspberry_pi.object_search_mission as core
        from grasp_range_integration.phase1.observation_recovery import install_observation_adapter
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        raw=[dict(source='qwen',class_name='block',camera_id=c,confidence=.85,
                  bbox_1000=[275,144,300,170],attributes={'color':'green'}) for c in ('left','right')]
        mission=NS(GpuMissionClient=type('Gpu',(),{'perceive':lambda *a: {'detections':raw}}),
            MissionLog=type('Log',(),{'event':lambda *a,**k:None}),confirm_detections=core.confirm_detections,
            _same_stereo_target=core._same_stereo_target,_fuse_tracking_pose=lambda *a:None)
        install_observation_adapter(mission,type('Camera',(),{'capture':lambda *a:{}}))
        route=dict(mode='semantic_visibility_regions',target_region={
            'center_xy_m':[2,0],'radius_m':.7,'recognition_distance_m':[.45,1.8]})
        with patch.object(planner,'LATEST_ROUTE',route),patch.object(planner,'ACTIVE_VIEW',None), \
             patch.object(planner,'task_region_view_assessment',return_value=dict(eligible=False,reason='approaching_task_region')), \
             patch('grasp_range_integration.phase1.colour_target_review.verify_green_cues',return_value=self.verdict) as ds:
            value=mission.GpuMissionClient().perceive(self.images['left'],self.images['right'],{
                'robot_pose':[0,0,0],'target':self.target})
        ds.assert_called_once();self.assertTrue(value['task_region_view']['eligible'])


class BackgroundTests(unittest.TestCase):
    def test_slow_camera_is_single_flight_and_does_not_wait_in_motion_loop(self):
        entered=threading.Event();release=threading.Event();done=threading.Event()
        o=LiveObserver(NS(),clock=lambda:10.);o.camera=object()
        route=dict(mode='semantic_visibility_regions',target_region={
            'center_xy_m':[2,0],'radius_m':.7,'recognition_distance_m':[.45,1.8]})
        def capture(*a):entered.set();release.wait(2);done.set()
        o._capture=capture
        try:
            self.assertEqual(o.select(Pose2D(0,0,0),route,None)['action'],'route')
            self.assertTrue(entered.wait(1));self.assertFalse(done.is_set())
            self.assertEqual(o.select(Pose2D(0,0,0),route,None)['action'],'route')
            self.assertTrue(o.capture_busy)
        finally:
            release.set();self.assertTrue(done.wait(1));o.close()

    def test_capture_failure_logs_without_cancelling_navigation(self):
        o=LiveObserver(NS(_phase1_capture_unarchived=lambda *a: (_ for _ in ()).throw(OSError('camera unavailable'))))
        with TemporaryDirectory() as directory:
            o.directory=Path(directory);o.pose=[0,0,0];o.capture_busy=True
            o._capture({})
        self.assertFalse(o.capture_busy)
        self.assertTrue(o.queue[-1][1]['navigation_continues'])

    def test_output_failure_before_classifier_launch_releases_single_flight_flag(self):
        def capture(camera,path):
            Image.new('RGB',(320,240),(70,70,70)).save(path)
            right=path.with_name('right.jpg');Image.new('RGB',(320,240),(80,80,80)).save(right)
            return {'camera_2_image':right.name}
        o=LiveObserver(NS(_phase1_capture_unarchived=capture),clock=lambda:10.)
        with TemporaryDirectory() as directory:
            o.directory=Path(directory);o.pose=[0,0,0];o.capture_busy=True
            with patch.object(Path,'write_text',side_effect=OSError('disk full')):o._capture({})
        self.assertFalse(o.capture_busy);self.assertFalse(o.model_busy)
        self.assertTrue(o.queue[-1][1]['navigation_continues'])

    def test_old_epoch_classifier_result_is_archived_but_cannot_steer(self):
        o=LiveObserver(NS(_phase1_raw_perceive=lambda *a:{'detections':[]},
            _phase1_raw_confirm=lambda *a:[],_same_stereo_target=lambda *a:True),clock=lambda:10.)
        with TemporaryDirectory() as directory:
            o.directory=Path(directory);images={}
            for eye,rgb in [('left',(50,50,50)),('right',(60,60,60))]:
                p=Path(directory)/(eye+'.jpg');Image.new('RGB',(320,240),rgb).save(p);images[eye]=str(p)
            frame=dict(sequence=1,epoch=0,stamp=9.,pose=[0,0,0],images=images,tracks={})
            o.stopped_capture();o.model_busy=True;o._review(frame,{},[])
            self.assertTrue((Path(directory)/'transit_0001.review.json').exists())
        self.assertIsNone(o.proof);self.assertFalse(o.model_busy)

    def test_raw_camera_capture_does_not_replace_archived_observation_record(self):
        from grasp_range_integration.phase1.observation_recovery import install_observation_adapter
        class Camera:
            def capture(self,path):Image.new('RGB',(320,240),(70,70,70)).save(path);return {}
        mission=NS(GpuMissionClient=type('Gpu',(),{'perceive':lambda *a:{'detections':[]}}),
            MissionLog=type('Log',(),{'event':lambda *a,**k:None}),confirm_detections=lambda *a:[],
            _same_stereo_target=lambda *a:True,_fuse_tracking_pose=lambda *a:None)
        with TemporaryDirectory() as directory:
            root=Path(directory);install_observation_adapter(mission,Camera)
            camera=Camera();camera.capture(root/'stopped.jpg')
            before=set(root.rglob('*'))
            mission._phase1_capture_unarchived(camera,root/'moving.jpg')
            self.assertEqual(set(root.rglob('*'))-before,{root/'moving.jpg'})

    def test_a_star_detour_does_not_authorize_direct_visual_drive_through_obstacle(self):
        import numpy as np
        from raspberry_pi.known_map_navigation import KnownOccupancyMap
        m=KnownOccupancyMap(np.full((200,200),254,dtype=np.uint8),.05,-5,-5)
        with patch('grasp_range_integration.phase1.visibility_route_planner.work_area_path',return_value=[(0,0),(0,1),(.3,0)]):
            self.assertTrue(guided_segment_clear(m,(0,0),(.3,0),(3,0),1.))
            c,r=m.world_to_grid([(.15,0)])[0];m.temporary_occupied[r,c]=True
            self.assertFalse(guided_segment_clear(m,(0,0),(.3,0),(3,0),1.))


class GuardWiringTests(unittest.TestCase):
    def setUp(self):
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        self.planner=planner;self.previous=(planner.LATEST_ROUTE,planner.ACTIVE_VIEW)
        planner.LATEST_ROUTE={};planner.ACTIVE_VIEW=None
        self.calls=[]
        class Localizer:
            def track(self,*a):return None
        def command(*a,**kw):self.calls.append(a);return 200,0,False
        self.mission=NS(ScanMapLocalizer=Localizer,_strict_command_for_pose=command,
            _obstacle_safe=lambda *a,**kw:True,build_search_route=lambda *a:{},_fuse_tracking_pose=lambda *a:None)
        from grasp_range_integration.phase1.relationship_find_only_entry import _install_waypoint_obstacle_guard
        _install_waypoint_obstacle_guard(self.mission)

    def tearDown(self):
        self.planner.LATEST_ROUTE,self.planner.ACTIVE_VIEW=self.previous

    def test_stereo_preemption_drains_remaining_waypoints_without_old_yaw_or_success(self):
        self.mission._phase1_live_observation=NS(select=lambda *a:{'action':'stopped_recheck'})
        for point in ((1,0),(1,1),(3,2)):
            self.assertEqual(self.mission._strict_command_for_pose(NS(),Pose2D(0,0,0),point),(0,0,True))
        self.assertEqual(self.calls,[])

    def test_single_eye_substitution_retains_guard_pipeline_and_does_not_mutate_original_args(self):
        self.mission._phase1_live_observation=NS(select=lambda *a:dict(action='guided_waypoint',waypoint=(.3,0),heading_tolerance_deg=2))
        args=NS(heading_tolerance_deg=10)
        self.assertEqual(self.mission._strict_command_for_pose(args,Pose2D(0,0,0),(2,2)),(200,0,False))
        self.assertEqual(self.calls[0][2],(.3,0));self.assertTrue(self.calls[0][0]._phase1_visual_guidance)
        self.assertEqual(args.heading_tolerance_deg,10)
        self.mission._phase1_verified_glass_required=True
        with patch('grasp_range_integration.phase1.navigation_freshness.ensure_navigation_freshness',return_value=False):
            self.assertEqual(self.mission._strict_command_for_pose(args,Pose2D(0,0,0),(2,2)),(0,0,False))

    def test_stationary_single_eye_recovery_does_not_inherit_transit_translation(self):
        self.planner.ACTIVE_VIEW={'visibility_zone':{'stop_rule':'single_camera_alignment'}}
        self.mission._phase1_live_observation=NS(select=lambda *a:self.fail('stationary recovery'))
        self.mission._strict_command_for_pose(NS(),Pose2D(0,0,0),(2,2))
        self.assertEqual(self.calls[0][2],(2,2))


if __name__=='__main__':unittest.main()
