import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from raspberry_pi.lidar.base import LaserScan
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.turn_range_feedback import turn_range_evidence
from grasp_range_integration.phase1.fresh_motion_feedback import install_fresh_motion_feedback
from grasp_range_integration.phase1 import turn_policy


def room_scan():
    angles=np.deg2rad(np.arange(-180,180))
    return LaserScan(angles,2+.5*np.sin(2*angles)+.3*np.cos(3*angles),0.)


class RangeFeedbackTests(unittest.TestCase):
    def evidence(self,before,after,transform):
        return turn_range_evidence(before,after,transform,lidar_offset=(.14,0),maximum_range=8.)

    def test_zero_motion_has_reciprocal_range_support(self):
        result=self.evidence(room_scan(),room_scan(),Pose2D())
        self.assertTrue(result['accepted'])

    def test_rotated_scan_uses_sensor_offset_in_both_directions(self):
        before=room_scan(); motion=Pose2D(0,0,math.radians(20))
        points=motion.inverse().transform_points(before.points()+np.array([.14,0]))-np.array([.14,0])
        after=LaserScan(np.arctan2(points[:,1],points[:,0]),np.linalg.norm(points,axis=1),0.)
        self.assertTrue(self.evidence(before,after,motion)['accepted'])
        self.assertFalse(self.evidence(before,after,Pose2D())['accepted'])

    def test_missing_or_changed_ranges_cannot_authorize_acceleration(self):
        before=room_scan()
        self.assertFalse(self.evidence(before,LaserScan(before.angles_rad,before.distances_m+1,0.),Pose2D())['accepted'])
        self.assertFalse(self.evidence(before,LaserScan(before.angles_rad[:20],before.distances_m[:20],0.),Pose2D())['accepted'])

    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def test_real_recorded_stationary_scans_have_evidence(self):
        data=json.loads((Path(__file__).parent/'fixtures/startup_stationary_20260926.json').read_text(encoding='utf-8'))
        scans=[LaserScan(np.array(f['raw_angles_rad']),np.array(f['raw_distances_m']),0.) for f in data['frames']]
        for current in scans:
            self.assertTrue(self.evidence(scans[0],current,Pose2D())['accepted'])

    def test_legacy_localizer_runs_actual_adapter_recovery_branch_without_new_api(self):
        class Driver:
            def scans(self):return iter([])
        class Client:
            def __init__(self):
                from raspberry_pi.config import SerialConfig
                self.calls=[]
                self.config=SerialConfig()
            def set_bounded_twist(self,*args,**kw):self.calls.append(('bounded',args,kw))
            def allow_twist_to_refresh_for(self,*args):return True
            def stop(self):self.calls.append(('stop',))
        class LegacyLocalizer:
            lidar_offset=np.array([.14,0]);max_distance_m=8.
            def relative_motion(self,*args):return MatchResult(Pose2D(),.02,100,True,.9)
        self.assertFalse(hasattr(LegacyLocalizer,'turn_range_evidence'))
        mission=SimpleNamespace(N10LidarDriver=Driver,Esp32Client=Client,ScanMapLocalizer=LegacyLocalizer,
                                _phase1_rotation_assess=lambda _: {'safe':True})
        stream=SimpleNamespace(latest_after=lambda *a:room_scan())
        with patch('grasp_range_integration.phase1.fresh_motion_feedback.FreshScanStream',return_value=stream):
            fence=install_fresh_motion_feedback(mission)
            mission.N10LidarDriver().scans()
        mission.ScanMapLocalizer().relative_motion(room_scan(),room_scan())
        turn_policy.TURN_START_REQUEST={'angular_mrad_s':-2600,'error_rad':math.radians(-50)}
        self.addCleanup(setattr,turn_policy,'TURN_START_REQUEST',None)
        def recovery(**callbacks):
            self.assertTrue(callbacks['measure'](room_scan(),room_scan()).converged)
            normal = client.config
            callbacks['send'](-2600,350,1.)
            self.assertIs(client.config,normal)
            callbacks['stop']()
        client=mission.Esp32Client()
        with patch('grasp_range_integration.phase1.turn_start_recovery.run_turn_start_recovery',side_effect=recovery) as run:
            client.set_twist(0,-2600,250)
        run.assert_called_once()
        self.assertEqual([c[0] for c in client.calls],['bounded','stop'])
        self.assertIsNotNone(fence.pending)


if __name__=='__main__':unittest.main()
