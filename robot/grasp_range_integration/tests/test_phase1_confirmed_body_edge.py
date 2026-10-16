"""User-confirmed chassis echo, using immutable full stop scan, not a sketch."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.rotation_guard import assess_rotation, body_mask_components


class ConfirmedBodyEdgeTests(unittest.TestCase):
    @unittest.skipUnless((Path(__file__).parent / "fixtures").is_dir(), "Requires private recording bundle")
    def setUp(self):
        self.record=json.loads((Path(__file__).parent/'fixtures/body_confirmed_stop_20260927_162135.json').read_text(encoding='utf-8'))
        f=self.record['frame']
        self.scan=SimpleNamespace(angles_rad=np.asarray(f['angles_rad']),
                                  distances_m=np.asarray(f['distances_m']))
        self.pose=SimpleNamespace(x_m=0.,y_m=0.,yaw_rad=0.)
        # Synthetic free map is only a map-support safety test. Sensor rays
        # are the complete actual archived scan, unchanged.
        self.known=KnownOccupancyMap(np.full((80,80),255,dtype=np.uint8),.05,-2,-2)

    def test_real_terminal_red_point_changes_classification_only_one_ray(self):
        before_angles=self.scan.angles_rad.copy();before_ranges=self.scan.distances_m.copy()
        legacy,patch=body_mask_components(before_angles,before_ranges)
        self.assertEqual(np.flatnonzero(legacy).tolist(),self.record['frame']['legacy_mask_indices'])
        self.assertEqual(np.flatnonzero(patch).tolist(),[104])
        self.assertAlmostEqual(self.scan.distances_m[104],.156)
        self.assertAlmostEqual(math.degrees(self.scan.angles_rad[104]),-107)
        decision=assess_rotation(self.scan,known_map=self.known,pose=self.pose,map_pose_reliable=True)
        self.assertTrue(decision['safe'])
        self.assertAlmostEqual(decision['min_distance_m'],.741)
        np.testing.assert_array_equal(self.scan.distances_m,before_ranges)
        np.testing.assert_array_equal(self.scan.angles_rad,before_angles)

    def test_body_patch_does_not_claim_unseen_space_free_without_map(self):
        decision=assess_rotation(self.scan)
        self.assertFalse(decision['safe'])
        self.assertEqual(decision['reason'],'installation_edge_occlusion_without_safe_map_support')
        for layer in (self.known.occupied,self.known.unknown,self.known.temporary_occupied):
            c,r=self.known.world_to_grid(((.30,0),))[0]
            layer[r,c]=True
            self.assertFalse(assess_rotation(self.scan,known_map=self.known,pose=self.pose,map_pose_reliable=True)['safe'])
            layer[r,c]=False

    def test_front_other_side_and_longer_same_bearing_obstacles_remain_blocking(self):
        for angle,distance in [(0,.156),(107,.156),(-100,.156),(-113,.156),
                               (-107,.17),(-107,.22),(-107,.30),(-103,.141)]:
            current=SimpleNamespace(angles_rad=np.r_[self.scan.angles_rad,math.radians(angle)],
                                    distances_m=np.r_[self.scan.distances_m,distance])
            legacy,patch=body_mask_components(current.angles_rad,current.distances_m)
            self.assertFalse(legacy[-1] or patch[-1],(angle,distance))
            if distance<.30:
                result=assess_rotation(current,known_map=self.known,pose=self.pose,map_pose_reliable=True)
                self.assertFalse(result['safe'],(angle,distance))
                self.assertEqual(result['reason'],'exterior_near_obstacle')

    def test_isolated_identical_point_without_chassis_signature_not_filtered(self):
        current=SimpleNamespace(angles_rad=np.r_[np.linspace(-math.pi,math.pi,100),math.radians(-107)],
                                distances_m=np.r_[np.ones(100),.156])
        result=assess_rotation(current,known_map=self.known,pose=self.pose,map_pose_reliable=True)
        self.assertFalse(result['safe'])
        self.assertEqual(result['reason'],'exterior_near_obstacle')
