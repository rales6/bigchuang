"""Regression of the 165830 near-region target / unset-range handoffs."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1.search_handoff import live_target_gate, absolute_map_range_evidence


class VisualHandoffTests(unittest.TestCase):
    def setUp(self):
        self.route=dict(mode='semantic_visibility_regions',navigation_stage='approach',target_region={
            'center_xy_m':[.8751325645044502,1.0443147835391624], 'radius_m':.95,
            'recognition_distance_m':[.45,1.8], 'camera_horizontal_fov_deg':82})
        self.pose=[-1.,-.519,1.057]
        self.gate=dict(eligible=False,reason='approaching_task_region',estimated_work_area_coverage=0.)
        self.checked=[dict(camera_id=c,source='qwen',confidence=.85,bbox_1000=b,
            single_camera_match=True,complete_in_frame=True,
            search_evidence={k:True for k in ['source_ok','class_ok','colour_ok','bbox_valid','bbox_complete']})
            for c,b in [('left',[348,217,375,247]),('right',[592,154,618,184])]]
        self.cues=[dict(camera_id=i['camera_id'],bbox_1000=i['bbox_1000']) for i in self.checked]

    def admit(self,checked=None,pose=None,cues=None):
        return live_target_gate(self.gate,self.route,pose or self.pose,
                                self.checked if checked is None else checked,
                                self.cues if cues is None else cues)

    def test_actual_4cm_semantic_boundary_excess_cannot_erase_two_real_candidates(self):
        result=self.admit()
        self.assertTrue(result['eligible'])
        self.assertEqual(result['visual_evidence_cameras'],['left','right'])
        self.assertFalse(result['object_depth_estimated'])
        self.assertTrue(result['visual_evidence_requires_confirmation'])

    def test_single_eye_gets_recheck_not_stereo_or_success(self):
        result=self.admit(checked=self.checked[:1])
        self.assertTrue(result['eligible'])
        self.assertEqual(result['visual_evidence_cameras'],['left'])
        self.assertNotIn('dual_camera_confirmed',result)

    def test_far_start_stays_in_transit(self):
        self.assertIs(self.admit(pose=[-1.55,-2.60,1.7]),self.gate)

    def test_wrong_heading_does_not_admit_distractor(self):
        self.assertIs(self.admit(pose=[-1.,-.519,-2.]),self.gate)

    def test_model_box_without_real_colour_pixels_cannot_override_map(self):
        self.assertIs(self.admit(cues=[]),self.gate)

    def test_wrong_colour_low_confidence_or_rejected_source_cannot_override_map(self):
        for key in ['source_ok','class_ok','colour_ok','bbox_valid','bbox_complete']:
            bad=deepcopy(self.checked)
            for i in bad:i['search_evidence'][key]=False
            self.assertIs(self.admit(checked=bad),self.gate)
        bad=deepcopy(self.checked)
        for i in bad:i['confidence']=.40
        self.assertIs(self.admit(checked=bad),self.gate)

    def test_adapter_keeps_stereo_and_real_tracker_entry_when_map_prior_is_borderline(self):
        import raspberry_pi.object_search_mission as core
        from grasp_range_integration.phase1.observation_recovery import install_observation_adapter
        from grasp_range_integration.phase1 import visibility_route_planner as planner
        with TemporaryDirectory() as tmp:
            images={}
            raw=[]
            for row in self.checked:
                c=row['camera_id'];image=Image.new('RGB',(1280,720),(105,105,105))
                b=row['bbox_1000'];box=(b[0]*1.28,b[1]*.72,b[2]*1.28,b[3]*.72)
                ImageDraw.Draw(image).rectangle(box,fill=(35,115,55))
                images[c]=Path(tmp)/(c+'.jpg');image.save(images[c])
                raw.append(dict(camera_id=c,source='qwen',confidence=.85,bbox_1000=b,
                    class_name='block',appearance_key='green_plastic_small',attributes={'color':'green'}))
            mission=NS(GpuMissionClient=type('Gpu',(),{'perceive':lambda *a: {'detections':raw}}),
                MissionLog=type('Log',(),{'event':lambda *a,**k:None}),
                confirm_detections=core.confirm_detections,_same_stereo_target=core._same_stereo_target,
                _fuse_tracking_pose=lambda *a:None)
            install_observation_adapter(mission,type('Camera',(),{'capture':lambda *a:{}}))
            with patch.object(planner,'LATEST_ROUTE',self.route),patch.object(planner,'ACTIVE_VIEW',None), \
                 patch.object(planner,'task_region_view_assessment',return_value=self.gate):
                result=mission.GpuMissionClient().perceive(images['left'],images['right'],{
                    'robot_pose':self.pose,'target':{'class_name':'block','color':'green','detection_mode':'qwen_fallback'}})
            self.assertTrue(result['task_region_view']['eligible'])
            self.assertEqual(len([i for i in result['detections'] if i.get('dual_camera_confirmed')]),2)
            self.assertTrue(all(i['search_evidence']['pursuit_entry']['ready'] for i in result['detections']))


class AbsoluteAnchorTests(unittest.TestCase):
    def setUp(self):
        self.scan=NS(sequence=1025)
        self.candidate=NS(pose=Pose2D(-.89,-.40,1.4),inlier_ratio=.77,
                          range_agreement=None,range_coverage=None)

    def test_probe_uses_actual_pose_and_actual_raw_scan_despite_unset_candidate_fields(self):
        calls=[]
        localizer=NS(range_profile=lambda scan,maximum:(scan,maximum),
            _range_consistency=lambda pose,profile,deadline_s: calls.append((pose,profile,deadline_s)) or (.85,.05,.80))
        result=absolute_map_range_evidence(localizer,self.candidate,self.scan,10,clock=lambda:1.)
        self.assertTrue(result['accepted']);self.assertEqual(result['sequence'],1025)
        self.assertEqual(calls,[(self.candidate.pose,(self.scan,36),1.35)])

    def test_missing_incomplete_weak_or_expired_probe_cannot_invent_map_anchor(self):
        self.assertFalse(absolute_map_range_evidence(NS(),self.candidate,self.scan,10,clock=lambda:1.)['accepted'])
        for values in [None,(.69,.1,.9),(.9,.1,.49),(float('nan'),.1,.9)]:
            loc=NS(range_profile=lambda *a,**k:[],_range_consistency=lambda *a,**k:values)
            self.assertFalse(absolute_map_range_evidence(loc,self.candidate,self.scan,10,clock=lambda:1.)['accepted'])
        no_call=NS(range_profile=lambda *a,**k:self.fail('Expired probe must not run'),_range_consistency=lambda *a:None)
        self.assertFalse(absolute_map_range_evidence(no_call,self.candidate,self.scan,1,clock=lambda:1.)['accepted'])


if __name__=='__main__':unittest.main()
