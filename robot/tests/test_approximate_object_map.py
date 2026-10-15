import unittest

import numpy as np

from tools.render_approximate_object_map import bearing_offset, occupied_components, ray_surfaces


class ApproximateObjectMapTests(unittest.TestCase):
    def test_pinhole_bearings_use_image_left_as_positive_yaw(self):
        self.assertAlmostEqual(bearing_offset(0, 82), 41)
        self.assertAlmostEqual(bearing_offset(.5, 82), 0)
        self.assertAlmostEqual(bearing_offset(1, 82), -41)

    def test_ray_preserves_near_furniture_and_far_wall_surfaces(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        gray[:, 30] = 0
        gray[:, 60] = 0
        config = {'resolution': .05, 'origin': [0, 0, 0]}
        hits = ray_surfaces(gray, config, .5, 2, 0)
        self.assertEqual(hits[0]['column'], 30)
        self.assertEqual(hits[-1]['column'], 60)

    def test_ray_without_surface_refuses_to_invent_a_depth(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, 'No map surface'):
            ray_surfaces(gray, {'resolution': .05, 'origin': [0, 0, 0]}, 1, 1, 90)

    def test_components_keep_separate_lidar_outlines(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        gray[10:15, 10:15] = 0
        gray[30:35, 30:35] = 0
        components = occupied_components(gray, {})
        self.assertEqual(sorted(map(len, components)), [25, 25])


if __name__ == '__main__':
    unittest.main()
