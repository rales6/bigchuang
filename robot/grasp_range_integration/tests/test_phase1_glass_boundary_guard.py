"""Software-only glass-gap, map binding and sweep regressions."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from raspberry_pi.known_map_navigation import KnownOccupancyMap
from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1.glass_boundary_guard import (
    DEFAULT_CONFIG, USER_LEFT_POLICY, load_verified_boundaries, attach_boundaries,
    swept_glass_clearance, install_glass_boundary_guard,
)
from grasp_range_integration.phase1.navigation_avoidance import reserve_path
from grasp_range_integration.phase1.rotation_guard import map_rotation_evidence, assess_rotation


def data():
    return dict(schema='phase1.verified_glass_boundaries.v1', verified=True,
                verification_note='Synthetic unit-test geometry, NOT a real field measurement',
                boundary_uncertainty_m=.10, segments_xy_m=[[[0.,-2.5],[0.,2.5]]])


class GlassTests(unittest.TestCase):
    def setUp(self):
        self.map=KnownOccupancyMap(np.full((120,120),254,dtype=np.uint8),.05,-3.,-3.)
        self.occupied=self.map.occupied.copy()
        self.free=self.map.free.copy()
        self.temporary=self.map.temporary_occupied.copy()
        attach_boundaries(self.map,data())

    def test_keepout_is_continuous_despite_all_lidar_map_cells_free(self):
        self.assertTrue(self.map._phase1_glass_keepout[:,59:61][10:110].all())
        np.testing.assert_array_equal(self.map.occupied,self.occupied)
        np.testing.assert_array_equal(self.map.free,self.free)
        np.testing.assert_array_equal(self.map.temporary_occupied,self.temporary)

    def test_route_cannot_cross_verified_glass_even_with_no_raw_obstacle(self):
        original=self.map.path
        path=reserve_path(self.map,(-1.,0.),(1.,0.),original)
        # The segment ends leave insufficient sweep clearance to go around.
        self.assertEqual(path,[])

    def test_safe_parallel_path_is_retained(self):
        self.assertTrue(reserve_path(self.map,(-1.,-1.),(-1.,1.),self.map.path))

    def test_map_rotation_intersection_includes_separate_glass_layer(self):
        result=map_rotation_evidence(self.map,Pose2D(-.35,0.,0.),.44)
        self.assertFalse(result['clear'])
        self.assertGreater(result['layers']['verified_glass']['count'],0)

    def test_full_clear_lidar_cannot_override_verified_glass_wall(self):
        angles=np.linspace(-math.pi,math.pi,360,endpoint=False)
        scan=SimpleNamespace(angles_rad=angles,distances_m=np.full(360,3.))
        result=assess_rotation(scan,known_map=self.map,pose=Pose2D(-.35,0.,0.),map_pose_reliable=True)
        self.assertFalse(result['safe'])
        self.assertEqual(result['reason'],'glass_sweep_blocked')

    def test_stale_or_unreliable_pose_never_authorizes_glass_sweep(self):
        for reliable,age in ((False,0.),(True,.51),(True,float('nan'))):
            result=swept_glass_clearance(self.map,Pose2D(-2.,0.,0.),400,0,
                reliable=reliable,pose_age_s=age)
            self.assertFalse(result['safe'])

    def test_acceleration_speed_is_used_in_actual_output_sweep(self):
        pose=Pose2D(-.9,0.,0.)
        low=swept_glass_clearance(self.map,pose,240,0,reliable=True,pose_age_s=.2)
        high=swept_glass_clearance(self.map,pose,460,0,reliable=True,pose_age_s=.2)
        self.assertTrue(low['safe'])
        self.assertFalse(high['safe'])

    def test_parallel_motion_clear_but_turn_near_wall_blocked(self):
        self.assertTrue(swept_glass_clearance(self.map,Pose2D(-1.,0.,math.pi/2),400,0,
            reliable=True,pose_age_s=.1)['safe'])
        self.assertFalse(swept_glass_clearance(self.map,Pose2D(-.5,0.,0.),0,3000,
            reliable=True,pose_age_s=.1)['safe'])

    def test_outside_map_endpoints_rejected(self):
        d=data();d['segments_xy_m']=[[[0.,0.],[0.,4.]]]
        with self.assertRaisesRegex(ValueError,'地图外'):attach_boundaries(self.map,d)

    def test_glass_map_installed_on_mission_only(self):
        mission=SimpleNamespace(KnownOccupancyMap=KnownOccupancyMap)
        install_glass_boundary_guard(mission,data())
        new=mission.KnownOccupancyMap(np.full((120,120),254,dtype=np.uint8),.05,-3.,-3.)
        self.assertTrue(new._phase1_glass_keepout.any())
        self.assertTrue(mission._phase1_verified_glass_required)

    def test_default_unverified_configuration_blocks_before_file_or_hardware_access(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'unverified.json'
            d=data();d['verified']=False
            config.write_text(json.dumps(d),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'尚未现场核验'):
                load_verified_boundaries(config,'nonexistent.pgm','nonexistent.yaml')

    def halfplane_data(self, reverse=False):
        d=data()
        d.update(verified=False,boundary_policy=USER_LEFT_POLICY,
                 user_confirmation={'source':'user','text':'Treat the whole left side as glass'},
                 geometry_source='bound_map_occupied_outline_estimate',
                 boundary_uncertainty_m=.20,allowed_reference_xy_m=[1.,0.],
                 segments_xy_m=[[[0.,-.5],[0.,.5]]])
        if reverse:d['segments_xy_m'][0].reverse()
        return d

    def test_whole_left_side_and_exterior_stay_blocked_beyond_endpoints(self):
        attach_boundaries(self.map,self.halfplane_data())
        for point in ((-1.,0.),(-1.,2.),(-1.,-2.),(.1,2.)):
            c,r=self.map.world_to_grid((point,))[0]
            self.assertTrue(self.map._phase1_glass_keepout[r,c])
        c,r=self.map.world_to_grid(((1.,2.),))[0]
        self.assertFalse(self.map._phase1_glass_keepout[r,c])

    def test_halfplane_orientation_uses_indoor_reference_not_endpoint_order(self):
        a=attach_boundaries(self.map,self.halfplane_data()).copy()
        b=attach_boundaries(self.map,self.halfplane_data(reverse=True))
        np.testing.assert_array_equal(a,b)

    def test_cannot_drive_or_rotate_outside_halfplane_even_far_from_segment(self):
        attach_boundaries(self.map,self.halfplane_data())
        for pose in (Pose2D(-1.,2.,0.),Pose2D(.5,2.,0.)):
            value=swept_glass_clearance(self.map,pose,0,2600,reliable=True,pose_age_s=.1)
            self.assertFalse(value['safe'])
        self.assertFalse(map_rotation_evidence(self.map,Pose2D(-1.,2.,0.),.44)['clear'])

    def test_route_cannot_go_around_short_halfplane_segment(self):
        attach_boundaries(self.map,self.halfplane_data())
        self.assertEqual(reserve_path(self.map,(1.,2.),(-1.,2.),self.map.path),[])

    def test_user_confirmation_can_authorize_policy_without_claiming_physical_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            pgm,yaml,config,base=self.verified_files(directory)
            d=self.halfplane_data()
            d.update(map_pgm_sha256=base['map_pgm_sha256'],map_yaml_sha256=base['map_yaml_sha256'])
            config.write_text(json.dumps(d),encoding='utf-8')
            accepted=load_verified_boundaries(config,pgm,yaml)
            self.assertFalse(accepted['verified'])
            self.assertEqual(accepted['boundary_policy'],USER_LEFT_POLICY)

    def test_missing_user_confirmation_or_small_margin_cannot_bypass_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            pgm,yaml,config,base=self.verified_files(directory)
            for key,value in (('user_confirmation',{}),('boundary_uncertainty_m',.1),
                              ('geometry_source','measured'),('allowed_reference_xy_m',None)):
                d=self.halfplane_data()
                d.update(map_pgm_sha256=base['map_pgm_sha256'],map_yaml_sha256=base['map_yaml_sha256'])
                d[key]=value
                config.write_text(json.dumps(d),encoding='utf-8')
                with self.assertRaises(ValueError):load_verified_boundaries(config,pgm,yaml)

    def test_published_map_policy_is_hash_bound_and_start_still_free(self):
        release=Path(__file__).resolve().parents[2]/'maps/semantic_release_20260919'
        if not release.exists():
            release=Path(__file__).resolve().parents[2]
        if not (release/'ble_scene_17.pgm').exists():
            self.skipTest('Published map fixture not present on this host')
        d=load_verified_boundaries(DEFAULT_CONFIG,release/'ble_scene_17.pgm',release/'ble_scene_17.yaml')
        m=KnownOccupancyMap.load(release/'ble_scene_17.pgm',release/'ble_scene_17.yaml')
        attach_boundaries(m,d)
        p=Pose2D(-1.55,-2.60,math.radians(100))
        self.assertTrue(swept_glass_clearance(m,p,240,0,reliable=True,pose_age_s=.1)['safe'])
        self.assertFalse(d['verified'])

    def verified_files(self, directory):
        pgm,yaml,config=(Path(directory)/name for name in ('map.pgm','map.yaml','glass.json'))
        pgm.write_bytes(b'unit-test-pgm');yaml.write_bytes(b'unit-test-yaml')
        d=data();d['map_pgm_sha256']=hashlib.sha256(pgm.read_bytes()).hexdigest()
        d['map_yaml_sha256']=hashlib.sha256(yaml.read_bytes()).hexdigest()
        config.write_text(json.dumps(d),encoding='utf-8')
        return pgm,yaml,config,d

    def test_verified_configuration_is_bound_to_both_map_files(self):
        with tempfile.TemporaryDirectory() as directory:
            pgm,yaml,config,d=self.verified_files(directory)
            self.assertEqual(load_verified_boundaries(config,pgm,yaml)['segments_xy_m'],d['segments_xy_m'])
            yaml.write_bytes(b'different-origin')
            with self.assertRaisesRegex(ValueError,'不匹配'):load_verified_boundaries(config,pgm,yaml)

    def test_invalid_geometry_and_margin_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pgm,yaml,config,original=self.verified_files(directory)
            for key,value in (('segments_xy_m',[]),('segments_xy_m',[[[0.,0.],[0.,0.]]]),
                              ('segments_xy_m',[[[0.,0.],[float('nan'),1.]]]),
                              ('boundary_uncertainty_m',.01),('verification_note','')):
                d=dict(original);d[key]=value
                config.write_text(json.dumps(d),encoding='utf-8')
                with self.assertRaises(ValueError):load_verified_boundaries(config,pgm,yaml)


if __name__=='__main__':unittest.main()
