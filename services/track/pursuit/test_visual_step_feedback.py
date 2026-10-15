import unittest
from pursuit.visual_step_feedback import VisualStepFeedback
from pursuit.pursuit_controller import PursuitController, Perception, Decision, Motion, State
from pursuit.pursuit_config import PursuitConfig, StereoConfig


class VisualStepTests(unittest.TestCase):
    def test_requires_two_stationary_pulses_then_only_one_level(self):
        f = VisualStepFeedback()
        for expected in (0, 1):
            f.baseline = (100, 40, 110, 42)
            self.assertEqual(f.complete((101, 41, 110, 43)), 'stationary')
            self.assertEqual(f.level, expected)

    def test_missing_frame_cannot_increase_startup_time(self):
        f = VisualStepFeedback(stationary=1, baseline=(100, 40, 110, 42))
        self.assertEqual(f.complete(None), 'unknown')
        self.assertEqual(f.level, 0)
        self.assertEqual(f.stationary, 0)

    def test_visible_movement_prevents_escalation(self):
        f = VisualStepFeedback(level=2)
        for _ in range(2):
            f.baseline = (100, 40, 110, 42)
            self.assertEqual(f.complete((108, 44, 119, 47)), 'moving')
        self.assertEqual(f.level, 1)

    def test_exhausted_feedback_is_bounded(self):
        f = VisualStepFeedback()
        for _ in range(8):
            f.baseline = (100, 40, 110, 42)
            f.complete(f.baseline)
        self.assertEqual(f.level, 3)
        self.assertTrue(f.exhausted)

    @staticmethod
    def pair(t, frame, cy=200):
        return Perception(True, (500, cy-20, 550, cy+20), None,
            tracking2=True, bbox2=(1880, cy-50, 1930, cy-10),
            received_at=t, frame_id=frame)

    def test_preserves_initial_profile_and_adapts_only_after_stationary_feedback(self):
        c = PursuitController(PursuitConfig(), StereoConfig(), 1280)
        base = Decision(Motion(405, 0, 400))
        for i, duration in enumerate((.8, .8, .85)):
            t=100+i*1.4
            d=c._visual_pulse_decision(base,self.pair(t,i+1),t,False)
            self.assertEqual(d.motion.linear_mm_s,405)
            self.assertAlmostEqual(c._visual_active_hold_s,duration)

    def test_close_band_allows_only_005_increment_not_two_second_escalation(self):
        c=PursuitController(PursuitConfig(),StereoConfig(),1280)
        p=self.pair(100,1,cy=290)
        c._visual_feedback.level=3
        c._visual_pulse_decision(Decision(Motion(405,0,400)),p,100,True)
        self.assertLessEqual(c._visual_active_hold_s,.85)

    def test_search_is_pulsed_and_has_global_on_time_budget(self):
        c=PursuitController(PursuitConfig(),StereoConfig(),1280,simulate_motion=True)
        c.on_ground_result(False,100)
        first=c.tick(None,None,100.01)
        self.assertNotEqual(first.motion.angular_mrad_s,0)
        rest=c.tick(None,None,100.9)
        self.assertEqual(rest.motion.angular_mrad_s,0)
        for i in range(1,50):
            d=c.tick(None,None,100.9+i*.1)
            if d.finished: break
            if d.reground_now: c.on_ground_result(False,100.9+i*.1)
        self.assertTrue(d.finished)
        self.assertEqual(c.state,State.GIVE_UP)
        self.assertLessEqual(c._search_total_on_s,2.4+1e-9)

    def test_lost_target_rechecked_stopped_before_rotation(self):
        c=PursuitController(PursuitConfig(),StereoConfig(),1280)
        c.on_ground_result(True,100)
        lost=Perception(False,None,None,received_at=100.01)
        for i in range(1,20):
            t=100+i*.1
            lost.received_at=t
            d=c.tick(lost,None,t)
            if d.reground_now: break
        self.assertTrue(d.reground_now)
        self.assertEqual((d.motion.linear_mm_s,d.motion.angular_mrad_s),(0,0))
        self.assertTrue(c._stationary_recheck_pending)
        lost.received_at=t+.1
        d=c.tick(lost,None,t+.1)
        self.assertFalse(d.reground_now)
        self.assertEqual((d.motion.linear_mm_s,d.motion.angular_mrad_s),(0,0))
        c.on_ground_result(True,t+.2)
        self.assertFalse(c._stationary_recheck_pending)

    def test_no_link_cannot_teach_static_friction(self):
        c=PursuitController(PursuitConfig(),StereoConfig(),1280)
        c.set_link_ready(False)
        for i in range(10):
            t=100+i*1.4
            c._visual_pulse_decision(Decision(Motion(405,0,400)),self.pair(t,i),t,False)
        self.assertEqual(c._visual_feedback.level,0)
        self.assertFalse(c._visual_feedback.exhausted)

if __name__=='__main__': unittest.main()
