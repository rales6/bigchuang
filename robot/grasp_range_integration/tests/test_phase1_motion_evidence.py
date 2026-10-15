"""Relative motion can be corroborated without inventing an absolute map fix."""
import math
from types import SimpleNamespace
import unittest

from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from grasp_range_integration.phase1.motion_evidence import bounded_scan_odometry


class MotionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.before, self.after = object(), object()
        self.localizer = SimpleNamespace()
        self.motion = MatchResult(Pose2D(.064311, .000869, math.radians(.605736)),
                                  .020309, 199, True, .975124)
        # Synthetic distribution with historical aggregate values; NOT a
        # replay of raw scans or proof that the real 221843 pair will pass.
        row = dict(supported=199, overlap=.984, agreement=.829, supported_sectors=5,
                   bad_sectors=0, mean_error_m=.28133, median_error_m=.025, p80_error_m=.15)
        self.raw = dict(accepted=False, forward=dict(row), reverse=dict(row))

    def evidence(self, reverse=None):
        backward = reverse or MatchResult(self.motion.transform.inverse(), .023, 188, True, .97)
        def relative(before, after):
            self.assertIs(before, self.after)
            self.assertIs(after, self.before)
            return backward
        return bounded_scan_odometry(self.localizer, self.before, self.after, self.motion,
                                     relative=relative, raw=self.raw)

    def test_tail_mean_alone_does_not_reject_reciprocal_distributed_evidence(self):
        result = self.evidence()
        self.assertTrue(result['accepted'])
        self.assertFalse(result['map_fix'])
        self.assertGreaterEqual(result['extra_position_uncertainty_m'], .03)
        self.assertEqual(self.motion.inlier_ratio, .975124)

    def test_reciprocal_disagreement_does_not_authorize_navigation(self):
        reverse = MatchResult(Pose2D(.10, 0., 0.), .02, 190, True, .95)
        self.assertEqual(self.evidence(reverse)['reason'], 'reciprocal_motion_disagrees')

    def test_bad_distribution_cannot_be_hidden_by_high_icp_score(self):
        self.raw['reverse']['p80_error_m'] = .4
        self.assertFalse(self.evidence()['accepted'])

    def test_missing_distribution_does_not_silently_pass(self):
        self.raw['forward'].pop('median_error_m')
        self.assertFalse(self.evidence()['accepted'])

    def test_insufficient_sector_support_does_not_silently_pass(self):
        self.raw['reverse']['supported_sectors'] = 2
        self.assertFalse(self.evidence()['accepted'])


if __name__ == '__main__':
    unittest.main()
