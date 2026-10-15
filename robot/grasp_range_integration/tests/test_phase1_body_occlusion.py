"""Software-only rear shoulder regression; no hardware IO.

Synthetic angular signatures below reconstruct the recorded mask support;
they are not represented as full raw recorded scans.
"""
import math
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

from grasp_range_integration.phase1.rotation_guard import assess_rotation, body_supported_blind_arcs
from grasp_range_integration.tests.test_phase1_rotation_guard import free_map, POSE


def recorded_support(start=100, first=147):
    body = np.r_[np.arange(first, 159), np.arange(173, 195),
                 np.arange(199, 221), np.arange(250, 261)]
    exterior = np.r_[np.arange(0, start+1), np.arange(262, 360)]
    return np.deg2rad(body), np.ones(len(body), dtype=bool), np.deg2rad(exterior)


def installed_scan(start=100, first=147):
    angles, _, _ = recorded_support(start, first)
    # Distances are synthetic, geometrically inside the legacy chassis mask.
    exterior = np.deg2rad(np.r_[np.arange(0, start+1), np.arange(262, 360)])
    body_r = np.where(np.degrees(angles) >= 245, .10, .14)
    return SimpleNamespace(angles_rad=np.r_[exterior, angles],
                           distances_m=np.r_[np.ones(len(exterior)), body_r])


class BodyOcclusionTests(unittest.TestCase):
    def test_four_unmodified_full_recorded_scans(self):
        data=json.loads((Path(__file__).parent/'fixtures/body_occlusion_20261002.json').read_text())
        self.assertEqual(data['source_run'],'20261002_155646')
        self.assertEqual(len(data['frames']),4)
        for frame in data['frames']:
            with self.subTest(source=frame['source_file']):
                current=SimpleNamespace(angles_rad=np.asarray(frame['angles_rad']),
                                        distances_m=np.asarray(frame['distances_m']))
                self.assertEqual(frame['recorded_reason'],'unobserved_sector_without_safe_map_support')
                self.assertTrue(frame['map_match']['reliable'])
                result=assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True,
                                       map_pose_uncertainty_m=frame['uncertainty_m'])
                self.assertTrue(result['safe'],result)
                self.assertEqual(result['reason'],'installed_body_occlusion_with_local_map_support')
                self.assertTrue(result['blind_arcs'][0]['recorded_rear_shoulder'])
                self.assertFalse(assess_rotation(current)['safe'])

    def test_recorded_47_48_50_degree_shoulder_signatures_explained(self):
        for start, first in ((100, 147), (99, 147), (98, 148)):
            arcs, explained = body_supported_blind_arcs(*recorded_support(start, first))
            self.assertTrue(explained)
            self.assertTrue(arcs[0]['recorded_rear_shoulder'])
            self.assertGreater(arcs[0]['unsupported_gap_deg'], 45)

    def test_extended_tolerance_only_applies_to_leading_shoulder(self):
        angles, body, exterior = recorded_support()
        # A similarly sized hole in the central rear is not the recorded shoulder.
        angles = angles[(np.degrees(angles) < 174) | (np.degrees(angles) > 223)]
        self.assertFalse(body_supported_blind_arcs(angles, np.ones(len(angles),bool), exterior)[1])

    def test_each_dense_band_is_required(self):
        angles, body, exterior = recorded_support()
        for lo, hi in ((145,165), (170,195), (198,225)):
            keep = (np.degrees(angles) < lo) | (np.degrees(angles) > hi)
            self.assertFalse(body_supported_blind_arcs(angles[keep], body[keep], exterior)[1])

    def test_recorded_jitter_with_dense_bands_is_explained(self):
        for start, first in ((90,147), (85,147), (82,147), (84,148)):
            self.assertTrue(body_supported_blind_arcs(*recorded_support(start,first))[1])

    def test_wider_or_unrecorded_shoulder_is_not_chassis(self):
        for start, first in ((81,147), (98,153), (82,150)):
            self.assertFalse(body_supported_blind_arcs(*recorded_support(start,first))[1])

    def test_frontward_edge_requires_dense_bands_even_with_short_gaps(self):
        angles, body, exterior = recorded_support(start=85, first=112)
        keep = (np.degrees(angles) < 170) | (np.degrees(angles) > 195)
        self.assertFalse(body_supported_blind_arcs(angles[keep], body[keep], exterior)[1])

    def test_three_runs_unmodified_scans_and_actual_map(self):
        from raspberry_pi.known_map_navigation import KnownOccupancyMap
        from grasp_range_integration.phase1.glass_boundary_guard import DEFAULT_CONFIG, attach_boundaries
        data=json.loads((Path(__file__).parent/'fixtures/body_occlusion_20261003.json').read_text(encoding='utf-8'))
        root=Path(__file__).resolve().parents[2]
        pgm=root/'maps/semantic_release_20260919/ble_scene_17.pgm'
        if not pgm.exists():pgm=root/'ble_scene_17.pgm'
        known=KnownOccupancyMap.load(pgm)
        attach_boundaries(known,json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8')))
        self.assertEqual(len(data['frames']),72)
        recovered=near=0
        for frame in data['frames']:
            with self.subTest(run=frame['source_run'],file=frame['source_file']):
                current=SimpleNamespace(angles_rad=np.asarray(frame['angles_rad'],dtype=float),
                                        distances_m=np.asarray(frame['distances_m'],dtype=float))
                before=current.distances_m.copy()
                recorded=frame['assessment']
                if recorded['reason']=='exterior_near_obstacle':
                    result=assess_rotation(current)
                    self.assertFalse(result['safe'])
                    self.assertEqual(result['reason'],'exterior_near_obstacle')
                    near+=1
                else:
                    p=recorded['map_support']['pose']
                    pose=SimpleNamespace(x_m=p[0],y_m=p[1],yaw_rad=p[2])
                    result=assess_rotation(current,known_map=known,pose=pose,
                        map_pose_reliable=recorded['map_pose_reliable'],
                        map_pose_uncertainty_m=recorded['map_support']['radius_m']-.44)
                    self.assertTrue(result['safe'],result)
                    self.assertEqual(result['reason'],'installed_body_occlusion_with_local_map_support')
                    self.assertFalse(assess_rotation(current)['safe'])
                    recovered+=1
                np.testing.assert_array_equal(before,current.distances_m)
        self.assertEqual((recovered,near),(60,12))

    def test_requires_reliable_free_map(self):
        current = installed_scan()
        self.assertTrue(assess_rotation(current, known_map=free_map(), pose=POSE,
                                       map_pose_reliable=True)['safe'])
        for known, reliable in ((None, True), (free_map(), False)):
            self.assertFalse(assess_rotation(current, known_map=known, pose=POSE,
                                            map_pose_reliable=reliable)['safe'])

    def test_every_blocking_map_layer_still_vetoes(self):
        for name in ('occupied','unknown','temporary_occupied','_phase1_glass_keepout'):
            known=free_map()
            if name.startswith('_'):setattr(known,name,np.zeros_like(known.occupied))
            cell=known.world_to_grid(((.3,0),))[0]
            getattr(known,name)[cell[1],cell[0]]=True
            self.assertFalse(assess_rotation(installed_scan(),known_map=known,pose=POSE,
                                            map_pose_reliable=True)['safe'])

    def test_live_front_obstacle_still_vetoes(self):
        current=installed_scan()
        current.distances_m[0]=.15
        result=assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True)
        self.assertFalse(result['safe'])
        self.assertEqual(result['reason'],'exterior_near_obstacle')

    def test_body_support_does_not_change_raw_scan(self):
        current=installed_scan()
        before=current.distances_m.copy()
        assess_rotation(current,known_map=free_map(),pose=POSE,map_pose_reliable=True)
        np.testing.assert_array_equal(before,current.distances_m)


if __name__=='__main__':unittest.main()
