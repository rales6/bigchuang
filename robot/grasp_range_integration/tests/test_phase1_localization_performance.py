import math
from types import SimpleNamespace
import unittest
import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap,ScanMapLocalizer
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.localization_performance import install_localization_performance
from grasp_range_integration.phase1.fresh_motion_feedback import install_fresh_motion_feedback,FreshScan


class VectorRayTests(unittest.TestCase):
    def setUp(self):
        class Localizer(ScanMapLocalizer): pass
        install_localization_performance(SimpleNamespace(ScanMapLocalizer=Localizer))
        image=np.full((85,91),255,dtype=np.uint8)
        image[0,:]=image[-1,:]=image[:,0]=image[:,-1]=0
        image[15:60,45]=0;image[63,10:70]=0;image[23,20:38]=128
        self.map=KnownOccupancyMap(image,.05,-2.25,-2.1)
        self.original=ScanMapLocalizer(self.map)
        self.fast=Localizer(self.map)

    def test_same_samples_same_first_hit_for_many_poses_and_angles(self):
        for pose in (Pose2D(),Pose2D(-.5,-.7,.31),Pose2D(.4,.5,-1.8),Pose2D(-2,-1.9,math.pi)):
            for angle in np.linspace(-math.pi,math.pi,65):
                self.assertEqual(self.fast._raycast_range(pose,angle),
                                 self.original._raycast_range(pose,angle))

    def test_maximum_distance_empty_rays_and_outside_map_unchanged(self):
        for maximum in (.05,.3,1,3,8):
            for pose in (Pose2D(),Pose2D(20,20,0)):
                for angle in (0,.5,1,2,math.pi):
                    self.assertEqual(self.fast._raycast_range(pose,angle,maximum),
                                     self.original._raycast_range(pose,angle,maximum))

    def test_unknown_cells_still_not_invented_as_occupied_or_free(self):
        self.assertEqual(self.fast._range_consistency(Pose2D(),np.column_stack((np.linspace(-3,3,28),np.full(28,2)))),
                         self.original._range_consistency(Pose2D(),np.column_stack((np.linspace(-3,3,28),np.full(28,2)))))

    def test_only_distances_cached_not_dynamic_map_obstacles(self):
        self.fast._raycast_range(Pose2D(),0)
        self.map.occupied[:,:]=True
        self.assertEqual(self.fast._raycast_range(Pose2D(),0),self.original._raycast_range(Pose2D(),0))


class PairFitTests(unittest.TestCase):
    def setUp(self):
        owner=self;self.calls=0
        class Localizer:
            def relative_motion(self,*args):
                owner.calls+=1
                return MatchResult(Pose2D(),.02,100,True,.9)
        self.mission=SimpleNamespace(N10LidarDriver=object,Esp32Client=object,ScanMapLocalizer=Localizer)
        self.fence=install_fresh_motion_feedback(self.mission)
        self.loc=self.mission.ScanMapLocalizer()
        angles=np.linspace(-math.pi,math.pi,100)
        self.a=FreshScan(angles.copy(),np.full(100,2.),1.,None,.8,1.,1)
        self.b=FreshScan(angles.copy(),np.full(100,2.),2.,None,1.8,2.,2)

    def test_exact_raw_pair_reuses_fit_not_whole_tracking_or_older_pair(self):
        first=self.loc.relative_motion(self.a,self.b)
        self.assertIs(self.loc.relative_motion(self.a,self.b),first)
        self.assertEqual(self.calls,1)
        self.loc.relative_motion(self.b,self.a)
        self.assertEqual(self.calls,2)

    def test_mutated_raw_data_never_reuses_old_fit(self):
        self.loc.relative_motion(self.a,self.b)
        self.b.distances_m[0]=.2
        self.loc.relative_motion(self.a,self.b)
        self.assertEqual(self.calls,2)

    def test_cache_does_not_bypass_new_pending_turn_feedback_gate(self):
        self.loc.relative_motion(self.a,self.b)
        self.fence.acknowledge(3.,3.1,180,-2600)
        with self.assertRaisesRegex(RuntimeError,"反馈异常"):
            self.loc.relative_motion(self.a,self.b)
        self.assertTrue(self.mission._phase1_motion_fault)


class ArrivalLeaseTests(unittest.TestCase):
    def create(self, ready=True):
        class Client:
            def __init__(self): self.calls=[]
            def set_twist(self,*a,**kw): self.calls.append(("regular",a,kw))
            def set_bounded_twist(self,*a,**kw): self.calls.append(("bounded",a,kw));return "ACK"
            def stop(self): self.calls.append(("stop",))
        class Localizer:
            def relative_motion(self,*a): return None
        mission=SimpleNamespace(N10LidarDriver=object,Esp32Client=Client,ScanMapLocalizer=Localizer,
            _phase1_arrival_distance_m=.132715285,
            _phase1_authorize_motion=lambda *a:ready)
        install_fresh_motion_feedback(mission)
        return mission.Esp32Client()

    def test_real_start_output_uses_bounded_269ms_even_without_glass_flag(self):
        client=self.create()
        self.assertEqual(client.set_twist(400,0,650),"ACK")
        self.assertEqual(client.calls[-1][0],"bounded")
        self.assertEqual(client.calls[-1][1],(400,0,269))
        self.assertLess(client.calls[-1][2]["refresh_deadline_s"],__import__("time").monotonic())

    def test_arrival_does_not_override_real_obstacle_or_freshness_rejection(self):
        client=self.create(False)
        client.set_twist(400,0,650)
        self.assertEqual(client.calls,[("stop",)])
