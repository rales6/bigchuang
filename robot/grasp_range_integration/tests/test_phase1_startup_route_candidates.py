import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from raspberry_pi.mapping import Pose2D
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.startup_localization import StartupMatcher
from grasp_range_integration.phase1.startup_route_recovery import startup_candidate_mask
from grasp_range_integration.phase1.startup_active_recovery import motor_exposure_s


class Candidates(unittest.TestCase):
    def test_yellow_190506_base_free_cell_has_no_reserve_exit(self):
        root = Path(__file__).resolve().parents[2]
        path = root/'maps/semantic_release_20260919/ble_scene_17.pgm'
        if not path.exists():path=root/'ble_scene_17.pgm'
        known=KnownOccupancyMap.load(path,clearance_m=.30)
        before=known.traversable.copy()
        mask=startup_candidate_mask(known,dict(center_xy_m=[.875,1.044],radius_m=.95))
        c,r=known.world_to_grid((-1.69,-3.10))
        self.assertTrue(known.traversable[r,c]);self.assertFalse(mask[r,c])
        c,r=known.world_to_grid((-1.55,-2.60));self.assertTrue(mask[r,c])
        np.testing.assert_array_equal(before,known.traversable)

    def test_lattice_excludes_no_exit_before_radar_ranking_and_handles_empty(self):
        for allowed in (False,True):
            with self.subTest(allowed=allowed):
                prior=NS(x_m=0.,y_m=0.,yaw_deg=0.,radius_m=.7,yaw_tolerance_deg=40.)
                known=NS(world_to_grid=lambda xy:np.array([(0,0) if x<.1 else (1,0) for x,y in xy]),
                         in_bounds=lambda c,r:0<=c<2 and r==0)
                bad=Pose2D(0,0,0);good=Pose2D(.2,0,0);calls=[]
                def rank(points,poses,**kw):
                    calls.extend(poses)
                    return [(0,p,0) for p in poses]
                localizer=NS(map=known,points=lambda *a,**kw:np.zeros((100,2)),
                    range_profile=lambda *a,**kw:[],_pose_lattice=lambda *a,**kw:[bad,good],
                    _rank=rank,_result_from_ranked=lambda *a,**kw:None,
                    _phase1_start_candidate_mask=np.array([[False,allowed]]))
                matcher=StartupMatcher(localizer,prior)
                self.assertIsNone(matcher._score(object(),math.inf))
                self.assertTrue(all(p is good for p in calls))
                self.assertEqual(bool(calls),allowed)
                self.assertGreater(matcher.diagnostic['excluded_no_route_poses'],0)

    def test_test_doubles_do_not_change_default_localization(self):
        self.assertIsNone(startup_candidate_mask(object(),{}))

    def test_195255_150ms_lease_is_not_30seconds_of_motor_attempt(self):
        self.assertAlmostEqual(motor_exposure_s(dict(elapsed_s=35.,renewals=0),150),.15)
        self.assertAlmostEqual(motor_exposure_s(dict(elapsed_s=40.,renewals=1),400),.8)
        self.assertEqual(motor_exposure_s(dict(command_sent=False,elapsed_s=40),400),0)
        self.assertEqual(motor_exposure_s(dict(status='deferred'),400),0)
        self.assertEqual(motor_exposure_s(dict(elapsed_s=-1),400),0)
        self.assertAlmostEqual(motor_exposure_s(dict(elapsed_s=float('nan')),400),.4)


if __name__=='__main__':unittest.main()
