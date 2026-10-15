import unittest
import numpy as np
from types import SimpleNamespace

from raspberry_pi.known_map_navigation import KnownOccupancyMap
from grasp_range_integration.phase1.temporary_obstacle_evidence import install_temporary_evidence
from grasp_range_integration.phase1.rotation_guard import map_rotation_evidence


class TemporaryEvidenceTests(unittest.TestCase):
    def maps(self):
        class AnnotatedMap(KnownOccupancyMap): pass
        install_temporary_evidence(AnnotatedMap)
        image = np.full((80, 80), 255, dtype=np.uint8)
        return AnnotatedMap(image, .05, -2, -2), KnownOccupancyMap(image, .05, -2, -2)

    def test_provenance_preserves_all_protected_masks_and_confirmation_threshold(self):
        annotated, original = self.maps()
        for index in range(2):
            for known in (annotated, original):
                added = known.mark_temporary_obstacles(((.55, 0.),), radius_m=.12, minimum_hits=2)
                if index == 0: self.assertEqual(added, 0)
            for field in ('temporary_occupied', 'temporary_obstacle_hits', 'traversable'):
                np.testing.assert_array_equal(getattr(annotated, field), getattr(original, field))
        result = map_rotation_evidence(annotated, SimpleNamespace(x_m=0., y_m=0., yaw_rad=0.), .44)
        layer = result['layers']['temporary_occupied']
        self.assertFalse(result['clear'])
        self.assertGreater(layer['padding_only_count'], 0)
        self.assertEqual(layer['confirmed_core_count'], 0)
        self.assertEqual(layer['unclassified_count'], 0)

    def test_core_confirmation_and_clear_reset_provenance(self):
        annotated, _ = self.maps()
        annotated.mark_temporary_obstacles(((.3, 0.),), radius_m=.12)
        pose = SimpleNamespace(x_m=0., y_m=0., yaw_rad=0.)
        layer = map_rotation_evidence(annotated, pose, .44)['layers']['temporary_occupied']
        self.assertGreater(layer['confirmed_core_count'], 0)
        annotated.clear_temporary_obstacles()
        self.assertTrue(map_rotation_evidence(annotated, pose, .44)['clear'])
        self.assertFalse(annotated._phase1_temporary_core.any())

    def test_preexisting_unknown_provenance_is_not_declared_padding(self):
        annotated, _ = self.maps()
        column, row = annotated.world_to_grid(((.3, 0.),))[0]
        annotated.temporary_occupied[row, column] = True
        annotated.mark_temporary_obstacles(((1., 0.),), radius_m=.12)
        layer = map_rotation_evidence(annotated, SimpleNamespace(x_m=0., y_m=0., yaw_rad=0.), .44)['layers']['temporary_occupied']
        self.assertEqual(layer['unclassified_count'], 1)
        self.assertEqual(layer['padding_only_count'], 0)


if __name__ == '__main__': unittest.main()
