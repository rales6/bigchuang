"""Acceptance regressions for the 20260918_182646 field drift evidence."""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.known_map_navigation import ScanMapLocalizer, LocalizationResult
from raspberry_pi.mapping import Pose2D
from raspberry_pi.mapping.scan_matcher import MatchResult
from raspberry_pi.open_space_panorama import (
    _bounded_motion, _correction_rejection_reason, _transport_candidate,
)


class PoseQualityTests(unittest.TestCase):
    def motion(self, match, turning=False, model=None):
        before, current = object(), object()
        localizer = SimpleNamespace(relative_motion=lambda *_: match,
            _panorama_rotation_only=turning, _panorama_turn_model_diagnostic=model)
        origin = Pose2D(.2, -.1, .4)
        pose, _, accepted = _bounded_motion(localizer, before, current, origin)
        return pose, accepted, localizer, before, origin

    def test_field_33cm_33percent_turn_is_not_written_into_pose(self):
        match = MatchResult(Pose2D(.3352229345, 0, math.radians(1.0771258)),
                            .115125127, 69, True, .3365853658)
        pose, accepted, localizer, before, origin = self.motion(match, True)
        self.assertFalse(accepted)
        self.assertEqual(pose, origin)
        self.assertEqual(localizer._panorama_pending_motion, (before, origin))
        self.assertEqual(localizer._panorama_motion_diagnostic['rejection_reason'], 'scan_match_quality_low')

    def test_good_matching_does_not_remove_actual_translation(self):
        match = MatchResult(Pose2D(.19, .01, .03), .02, 150, True, .94)
        pose, accepted, _, _, origin = self.motion(match)
        self.assertTrue(accepted)
        self.assertEqual(pose, origin.compose(match.transform))

    def test_stopped_recovery_cannot_drop_turn_context_to_bypass_verification(self):
        weak = MatchResult(Pose2D(.335, 0, .02), .115, 69, True, .337)
        stronger = MatchResult(Pose2D(.2, 0, .02), .03, 120, True, .9)
        sequence = iter([weak, stronger])
        references = []
        def turn(before, current):
            references.append(before)
            return next(sequence)
        localizer = SimpleNamespace(relative_motion=lambda *_: self.fail('Turn recovery used ordinary ICP'),
            relative_turn_motion=turn, _panorama_rotation_only=True,
            _panorama_turn_model_diagnostic={'translation_verified': False})
        before = object()
        origin = Pose2D(.2, -.1, .4)
        self.assertFalse(_bounded_motion(localizer, before, object(), origin)[2])
        localizer._panorama_rotation_only = False
        pose, _, accepted = _bounded_motion(localizer, object(), object(), origin)
        self.assertFalse(accepted)
        self.assertEqual(pose, origin)
        self.assertEqual(references, [before, before])
        self.assertEqual(localizer._panorama_motion_diagnostic['rejection_reason'], 'turn_translation_unverified')

    def test_good_recovery_applies_original_anchor_motion_once(self):
        sequence = iter([MatchResult(Pose2D(.335, 0, .02), .115, 69, True, .337),
                         MatchResult(Pose2D(.05, 0, .02), .03, 120, True, .9)])
        localizer = SimpleNamespace(relative_motion=lambda *_: next(sequence))
        origin = Pose2D(.2, -.1, .4)
        self.assertFalse(_bounded_motion(localizer, object(), object(), origin)[2])
        pose, _, accepted = _bounded_motion(localizer, object(), object(), origin)
        self.assertTrue(accepted)
        self.assertEqual(pose, origin.compose(Pose2D(.05, 0, .02)))
        self.assertIsNone(localizer._panorama_pending_motion)

    def test_invalid_or_absent_quality_never_counts_as_motion(self):
        good = MatchResult(Pose2D(.1, 0, 0), .02, 100, True, .9)
        for match in (replace(good, inlier_ratio=0.), replace(good, rmse_m=math.nan),
                      replace(good, inlier_ratio=math.nan), replace(good, correspondences=10),
                      replace(good, rmse_m=.2)):
            with self.subTest(match=match):
                self.assertFalse(self.motion(match)[1])

    def test_large_turn_translation_requires_reciprocal_evidence(self):
        match = MatchResult(Pose2D(.2, .03, .2), .03, 150, True, .9)
        self.assertFalse(self.motion(match, True, {'translation_verified': False})[1])
        pose, accepted, _, _, origin = self.motion(match, True, {'translation_verified': True})
        self.assertTrue(accepted)
        self.assertEqual(pose, origin.compose(match.transform))

    def test_reciprocal_match_verifies_real_skid_without_forcing_zero(self):
        localizer = ScanMapLocalizer(SimpleNamespace())
        free = MatchResult(Pose2D(.2, .05, .2), .02, 150, True, .95)
        reverse = replace(free, transform=free.transform.inverse())
        bad_pivot = MatchResult(Pose2D(0, 0, .2), .2, 80, True, .5)
        with patch.object(localizer, 'points', return_value=np.ones((150, 2))), patch.object(
            localizer.motion_matcher, 'match', side_effect=[free, reverse]), patch.object(
            localizer.motion_matcher, 'match_rotation', return_value=bad_pivot):
            self.assertIs(localizer.relative_turn_motion(None, None), free)
        self.assertTrue(localizer._panorama_turn_model_diagnostic['translation_verified'])

    def test_one_way_alias_is_not_verified(self):
        localizer = ScanMapLocalizer(SimpleNamespace())
        free = MatchResult(Pose2D(.2, 0, .2), .02, 150, True, .95)
        with patch.object(localizer, 'points', return_value=np.ones((150, 2))), patch.object(
            localizer.motion_matcher, 'match', return_value=free), patch.object(
            localizer.motion_matcher, 'match_rotation', return_value=replace(free, rmse_m=.2)):
            localizer.relative_turn_motion(None, None)
        self.assertFalse(localizer._panorama_turn_model_diagnostic['translation_verified'])

    def candidate(self, **changes):
        return replace(LocalizationResult(Pose2D(.4, 0, .1), .04, .85, .8, .1,
                                          .9, .06, .9), **changes)

    def reason(self, candidate):
        return _correction_rejection_reason(candidate, Pose2D(), .65, 35.)

    def test_field_38percent_candidate_cannot_enter_confirmation_window(self):
        candidate = self.candidate(rmse_m=.0847986632, inlier_ratio=.38125)
        for _ in range(2):
            self.assertEqual(self.reason(candidate), 'candidate_quality_low')

    def test_large_map_correction_requires_metric_evidence(self):
        self.assertEqual(self.reason(self.candidate(range_agreement=None)), 'candidate_metric_evidence_missing')

    def test_endpoint_fit_cannot_override_bad_absolute_ranges(self):
        for changes in ({'range_agreement': .4}, {'range_mean_error_m': .5}, {'range_coverage': .3}):
            self.assertEqual(self.reason(self.candidate(**changes)), 'candidate_metric_distance_mismatch')

    def test_large_map_correction_requires_proven_distinct_basin_margin(self):
        for margin in (math.inf, math.nan, 0., .01):
            self.assertEqual(self.reason(self.candidate(ambiguity_margin=margin)), 'candidate_uniqueness_unproven')

    def test_strong_unique_metric_candidate_is_still_eligible(self):
        self.assertEqual(self.reason(self.candidate()), '')

    def test_transport_preserves_metric_evidence(self):
        candidate = self.candidate()
        transported = _transport_candidate(candidate, Pose2D(), Pose2D(.02, 0, .01))
        for name in ('range_agreement', 'range_mean_error_m', 'range_coverage', 'ambiguity_margin'):
            self.assertEqual(getattr(transported, name), getattr(candidate, name))

    def test_bounded_match_cannot_return_candidate_without_metric_ray_check(self):
        localizer = ScanMapLocalizer(SimpleNamespace())
        ranked = [(.9, Pose2D(), (.03, .9))]
        with patch.object(localizer, '_pose_lattice', return_value=[Pose2D()]), patch.object(
            localizer, 'points', return_value=np.ones((100, 2))), patch.object(
            localizer, 'range_profile', return_value=np.ones((18, 2))), patch.object(
            localizer, '_rank', side_effect=[ranked, ranked]), patch.object(
            localizer, '_range_consistency', return_value=None):
            self.assertIsNone(localizer.localize_bounded(None, Pose2D()))
        self.assertEqual(localizer._last_bounded_diagnostic['result'], 'metric_evidence_unavailable')

    def test_timeout_during_winner_ray_validation_does_not_leak_a_candidate(self):
        localizer = ScanMapLocalizer(SimpleNamespace())
        ranked = [(.9, Pose2D(), (.03, .9))]
        with patch.object(localizer, '_pose_lattice', return_value=[Pose2D()]), patch.object(
            localizer, 'points', return_value=np.ones((100, 2))), patch.object(
            localizer, 'range_profile', return_value=np.ones((18, 2))), patch.object(
            localizer, '_rank', side_effect=[ranked, ranked]), patch.object(
            localizer, '_range_consistency', side_effect=TimeoutError):
            self.assertIsNone(localizer.localize_bounded(None, Pose2D()))
        self.assertEqual(localizer._last_bounded_diagnostic['result'], 'budget_exhausted')


if __name__ == '__main__':
    unittest.main()
