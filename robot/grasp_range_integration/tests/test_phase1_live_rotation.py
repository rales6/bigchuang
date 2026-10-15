"""No hardware imports or calls; manual probe safety gating."""
import unittest
from types import SimpleNamespace
import numpy as np
from grasp_range_integration.phase1.probe_live_rotation import probe_clearance


class LiveProbeTests(unittest.TestCase):
    def frame(self):
        angles = np.deg2rad(np.arange(-180, 180))
        distances = np.ones(360)
        rear = np.abs(np.degrees(angles)) >= 110
        distances[rear] = .10
        return SimpleNamespace(angles_rad=angles, distances_m=distances)

    def test_rear_blind_probe_needs_explicit_operator_confirmation(self):
        self.assertFalse(probe_clearance(self.frame(), False)['safe'])
        self.assertTrue(probe_clearance(self.frame(), True)['safe'])

    def test_operator_cannot_override_new_102_degree_point(self):
        frame = self.frame(); frame.distances_m[282] = .158
        self.assertFalse(probe_clearance(frame, True)['safe'])

    def test_probe_keeps_050_external_clearance(self):
        frame = self.frame(); frame.distances_m[180] = .49
        self.assertFalse(probe_clearance(frame, True)['safe'])

    def test_documented_patch_needs_operator_clearance_only_for_manual_probe(self):
        frame = self.frame(); frame.distances_m[291] = .14
        self.assertFalse(probe_clearance(frame, False)['safe'])
        self.assertTrue(probe_clearance(frame, True)['safe'])

    def test_missing_half_scan_not_body_evidence(self):
        frame = self.frame(); frame.distances_m[:] = 1.
        frame.distances_m[frame.angles_rad < 0] = np.inf
        self.assertFalse(probe_clearance(frame, True)['safe'])
