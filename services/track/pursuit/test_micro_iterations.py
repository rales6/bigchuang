"""Bounded iterative near adjustments; every hardware call is mocked."""
import json
import tempfile
import time
from dataclasses import replace
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO

from pursuit.hull_micro_adjust import (align, TargetNotVisible, SweptRetreatBudget,
    AdjustmentPlan, opposite_arc_repair)
from integrations.grab_visual.teaching_region import polygon_distance
from pursuit.pursuit_config import PursuitConfig


def obs(distance, left, right, goal_left=(435.2,315.6), goal_right=(364.8,242.4)):
    return dict(ok=True,inside_teaching_hull=distance==0,outside_distance_px=distance,
        centers=dict(left=left,right=right),
        nearest_hull_centers=dict(left=goal_left,right=goal_right))


class MicroIterationTests(unittest.TestCase):
    @staticmethod
    def observed_2d(left, right):
        # Reduced teaching polygons, same near-edge geometry as the 19:21 log.
        polygons={'left':[[201.6,366.7],[220.2,330.2],[435.2,315.6],
                          [568.3,339.4],[545.9,387.6],[262.4,399.8]],
                  'right':[[81.9,277.2],[83.8,257.5],[158.7,230.4],
                           [364.8,242.4],[471,295.9],[140.2,312.5]]}
        centers={'left':list(left),'right':list(right)}
        gaps={side:polygon_distance(centers[side],polygons[side])[0]
              for side in centers}
        nearest={side:list(polygon_distance(centers[side],polygons[side])[1])
                 for side in centers}
        return dict(ok=True,centers=centers,inside_teaching_hull=False,
            outside_distance_px=max(gaps.values()),grasp_region_mode='dual_2d_shell',
            view_hull_polygons_px=polygons,view_hull_distances_px=gaps,
            view_hull_centers=nearest,grasp_region_ok=all(x<=25 for x in gaps.values()),
            grasp_margin_px=25,projected_to_teaching_boundary=True)

    def simulate(self, observations, *, feedback=None, forward_reference=False, profile_changes=None, **kwargs):
        feedback_steps=iter(feedback) if isinstance(feedback,list) else None
        # Historical fixtures replay the pre-0.60s profile. New production
        # start/continuation behavior is covered in test_forward_patience.
        profile=dict(hull_forward_first_s=.4,hull_micro_forward_patience_steps=0,
                     hull_multicue_progress_enabled=False,hull_approach_arc_first_s=.4)
        profile.update(profile_changes or {})
        # Recorded pre-calibration fixtures intentionally retain their old
        # centre-third steering. Current line geometry is tested separately.
        with tempfile.TemporaryDirectory() as folder, \
             patch('pursuit.hull_micro_adjust._APPROACH_PROFILE',
                   replace(PursuitConfig(),forward_reference_enabled=forward_reference,**profile)), \
             patch('pursuit.hull_micro_adjust.verify_model_camera_order'), \
             patch('pursuit.hull_micro_adjust._ready'), \
             patch('pursuit.hull_micro_adjust.ScanGuard'), \
             patch('pursuit.hull_micro_adjust._observe',side_effect=observations), \
             patch('pursuit.hull_micro_adjust._pulse') as pulse:
            def step(*args,**kw):
                sample=next(feedback_steps) if feedback_steps is not None else feedback
                kw['feedback_sink'].update(sample or dict(termination='visual_motion_limit',
                    visual_feedback=dict(movement_px=dict(left=4.2,right=4.1))))
                return 1.0
            pulse.side_effect=step
            result=align('红色方块',real=True,log_dir=Path(folder),**kwargs)
            events=[json.loads(x) for x in Path(result.log_path).read_text(encoding='utf-8').splitlines()]
        return result,pulse.call_args_list,events

    def test_recorded_67_px_case_always_has_smaller_window_and_flow_stop(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        result,calls,_=self.simulate([before,before,obs(0,(444,274),(374,247))])
        self.assertTrue(result.ok)
        self.assertEqual(calls[0].args[2:4],('forward',.4))
        self.assertEqual(calls[0].kwargs['visual_motion_limit_px'],4)
        self.assertEqual(calls[0].kwargs['target_centers'],before['centers'])

    def test_worsening_arc_uses_measured_opposite_instead_of_stopping(self):
        before=self.observed_2d((233.0,297.4),(51.2,303.4))
        worse=self.observed_2d((244.2,306.5),(48.0,311.5))
        improving=self.observed_2d((229.0,306.5),(65.0,311.5))
        inside=self.observed_2d((213.0,306.5),(92.0,311.5))
        flow=dict(termination='time_limit',visual_feedback=dict(
            movement_px=dict(left=2.61,right=2.68),
            target_region_movement_px=dict(left=2.89,right=2.62)))
        repair=opposite_arc_repair(worse,(233.0,297.4,51.2,303.4),
            (244.2,306.5,48.0,311.5),'turn_left',flow,
            duration_s=.6,maximum_s=.68)
        self.assertIsNotNone(repair)
        self.assertEqual(repair.action,'turn_right')
        result,calls,events=self.simulate([before,before,worse,worse,improving,inside],
            feedback=flow,forward_reference=True)
        self.assertTrue(result.ok,result.reason)
        self.assertEqual([c.args[2] for c in calls],['turn_left','turn_right','turn_right'])
        self.assertEqual(calls[1].args[3],.5)
        self.assertFalse(calls[1].kwargs['arc_forward'])
        self.assertFalse(calls[2].kwargs['arc_forward'])
        self.assertIn('opposite_yaw_repair_selected',[e['event'] for e in events])
        self.assertIn('opposite_yaw_repair_continue',[e['event'] for e in events])

    def test_opposite_arc_requires_both_cameras_motion(self):
        worse=self.observed_2d((244.2,306.5),(48.0,311.5))
        repair=opposite_arc_repair(worse,(233.0,297.4,51.2,303.4),
            (244.2,306.5,48.0,311.5),'turn_left',
            dict(visual_feedback=dict(movement_px=dict(left=2.61))),
            duration_s=.6,maximum_s=.68)
        self.assertIsNone(repair)

    def test_only_explicit_detector_miss_is_classified_as_target_loss(self):
        from pursuit.hull_micro_adjust import _observe
        body=json.dumps({'ok':False,'error':'target not visible in both views'}).encode()
        error=HTTPError('http://fake',422,'miss',{},BytesIO(body))
        with patch('pursuit.hull_micro_adjust._ready'), \
             patch('pursuit.hull_micro_adjust._post_json',side_effect=error):
            with self.assertRaises(TargetNotVisible): _observe('红色方块')

    def test_model_error_is_not_classified_as_target_loss(self):
        from pursuit.hull_micro_adjust import _observe
        body=json.dumps({'ok':False,'error':'model unavailable'}).encode()
        error=HTTPError('http://fake',502,'model',{},BytesIO(body))
        with patch('pursuit.hull_micro_adjust._ready'), \
             patch('pursuit.hull_micro_adjust._post_json',side_effect=error):
            with self.assertRaises(HTTPError): _observe('红色方块')

    def test_confirmed_overrun_rechecks_then_rolls_back_and_completes(self):
        # Recorded before/after centres; final hull verdict is synthetic.
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        after=obs(101,(585.9,401.8),(313.6,392.9),(531.2,395.4),(317,329.6))
        inside=obs(0,(550,367),(330,352))
        result,calls,events=self.simulate([before,before,after,after,inside])
        self.assertTrue(result.ok,result.reason)
        self.assertEqual([c.args[2] for c in calls],['forward','backward'])
        self.assertEqual([c.args[3] for c in calls],[.4,.4])
        self.assertTrue(calls[1].kwargs['swept_retreat'])
        self.assertEqual(calls[1].kwargs['visual_motion_limit_px'],2)
        self.assertIn('overrun_stability_check',[x['event'] for x in events])

    def test_large_jump_without_flow_does_not_reverse(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        after=obs(101,(585.9,401.8),(313.6,392.9),(531.2,395.4),(317,329.6))
        result,calls,_=self.simulate([before,before,after],feedback={'termination':'time_limit'})
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),1)
        self.assertIn('缺少可信运动反馈',result.reason)

    def test_unstable_overrun_never_gets_another_motion(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        after=obs(101,(585.9,401.8),(313.6,392.9),(531.2,395.4),(317,329.6))
        shifted=obs(101,(555.9,380),(300,350))
        result,calls,_=self.simulate([before,before,after,shifted])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),1)

    def test_lost_after_forward_checks_first_then_short_rollback(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        found=obs(45,(444,280),(374,254))
        inside=obs(0,(448,286),(370,260))
        result,calls,events=self.simulate([before,before,TargetNotVisible('miss'),
            TargetNotVisible('miss'),found,found,inside])
        self.assertTrue(result.ok,result.reason)
        self.assertEqual([c.args[2] for c in calls],['forward','backward','forward'])
        self.assertEqual([c.args[3] for c in calls],[.4,.4,.4])
        self.assertIn('lost_target_stationary_recheck',[e['event'] for e in events])
        self.assertIn('rollback_reacquired',[e['event'] for e in events])

    def test_network_error_is_not_treated_as_lost_target(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        result,calls,_=self.simulate([before,before,TimeoutError('network')])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),1)

    def test_lost_without_flow_does_not_allow_blind_reverse(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        result,calls,_=self.simulate([before,before,TargetNotVisible('miss'),TargetNotVisible('miss')],
                                   feedback={'termination':'time_limit'})
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),1)

    def test_micro_retreat_cannot_refresh_an_exhausted_budget(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        budget=SweptRetreatBudget(time.monotonic()+60,'old',on_time_s=.8,pulses=2)
        result,calls,_=self.simulate([before,before,TargetNotVisible('miss'),TargetNotVisible('miss')],
                                   retreat_budget=budget)
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),1)

    def test_total_window_budget_prevents_next_step(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        after=obs(45,(444,280),(374,254))
        with patch('pursuit.hull_micro_adjust.MAX_TOTAL_PULSE_S',.6):
            result,calls,_=self.simulate([before,before,after])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),1)
        self.assertIn('累计脉冲',result.reason)

    def test_consecutive_correction_budget_stops_even_if_selector_keeps_proposing(self):
        before=obs(100,(300,300),(200,250))
        after=[obs(103+3*i,(308+8*i,300),(208+8*i,250)) for i in range(4)]
        # Deliberately permissive fake selector isolates the loop's stop guard.
        with patch('pursuit.hull_micro_adjust.select_adjustment',
                   return_value=AdjustmentPlan('turn_left',.4,'test_selector')):
            result,calls,events=self.simulate([before,before,
                after[0],after[0],after[1],after[1],after[2],after[2],after[3]])
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),4)
        self.assertEqual(sum(e['event']=='corrective_replan' for e in events),3)

    def test_lost_on_last_allowed_step_cannot_add_rollback(self):
        before=obs(67.4,(442.2,266.2),(376.3,239.0))
        result,calls,_=self.simulate([before,before,TargetNotVisible('miss'),TargetNotVisible('miss')],
                                   max_pulses=1)
        self.assertFalse(result.ok)
        self.assertEqual(len(calls),1)


if __name__=='__main__': unittest.main()
