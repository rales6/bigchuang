import math
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from grasp_range_integration.phase1.startup_visual_review import (
    ScopedStartupLocalizer, normalize_descriptions, audit_observations, review_startup,
)


def candidate(x=-1.55, y=-2.60, yaw=100., reliable=True):
    return SimpleNamespace(pose=SimpleNamespace(x_m=x, y_m=y, yaw_rad=math.radians(yaw)),
                           reliable=reliable, rmse_m=.04, inlier_ratio=.85)


class StartupVisualReviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        images = [Path(self.directory.name) / name for name in ('left.jpg', 'right.jpg')]
        for image in images:
            image.write_bytes(b'camera IO fixture; not a real decoded image')
        self.capture_diagnostics = {'camera_count': 2, 'capture': [str(image) for image in images]}
        self.prior = SimpleNamespace(x_m=-1.55, y_m=-2.6, yaw_deg=100.,
                                     radius_m=.7, yaw_tolerance_deg=40.)
        self.anchor = candidate()
        self.original = SimpleNamespace(
            global_localization_candidates=lambda *a, **kw: self.fail('whole-map search forbidden'),
            localize_near_position=lambda *a, **kw: self.fail('all-heading correction forbidden'))
        self.descriptions = [{'candidate_id': 'C01', 'expected_visible_structures': [
            {'name': '玻璃墙', 'category': 'glass_wall', 'relative_bearing_deg': -35.1,
             'bearing_uncertainty_deg': 18.7, 'evidence_role': 'stable_anchor', 'side': 'right'},
            {'name': '玻璃门', 'category': 'door', 'relative_bearing_deg': -7.4,
             'bearing_uncertainty_deg': 17.9, 'evidence_role': 'stable_anchor', 'side': 'center'},
            {'name': '远处栏杆', 'category': 'stairs', 'relative_bearing_deg': 71.1,
             'bearing_uncertainty_deg': 28., 'evidence_role': 'stable_anchor', 'side': 'left'},
        ], 'context_only_structures': []}]

    def test_candidates_never_call_global_or_all_heading_search(self):
        scoped = ScopedStartupLocalizer(self.original, self.anchor, self.prior,
                                       [candidate(x=.77, yaw=9.5), candidate(x=-1.35, yaw=110),
                                        candidate(yaw=150), candidate(reliable=False)])
        rows = scoped.global_localization_candidates(object())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], self.anchor)
        self.assertIs(scoped.localize_near_position(object(), self.anchor.pose), self.anchor)

    def test_real_2_32m_90_5deg_failure_cannot_become_start_pose(self):
        distant = candidate(x=-1.55+2.32, yaw=100.+90.5)
        checker = lambda *a: ('corrected', distant, object(), {})
        status, visual, match, diagnostics = review_startup(
            checker, [None, self.anchor, object(), self.original, None, None, None, None, 1],
            self.anchor, self.prior)
        self.assertEqual(status, 'conflict')
        self.assertIs(visual, self.anchor)
        self.assertIsNone(match)
        self.assertEqual(diagnostics['rejected_reason'], 'startup_visual_pose_out_of_scope')

    def test_wide_glass_is_not_a_single_direction_and_offscreen_anchor_is_context(self):
        rows = normalize_descriptions(self.descriptions)
        expected = rows[0]['expected_visible_structures']
        self.assertEqual(expected[0]['allowed_sides'], ['left', 'center', 'right'])
        self.assertEqual(expected[0]['side'], 'uncertain')
        self.assertEqual(expected[1]['allowed_sides'], ['right', 'center'])
        self.assertEqual(rows[0]['context_only_structures'][0]['name'], '远处栏杆')
        self.assertEqual(self.descriptions[0]['expected_visible_structures'][0]['side'], 'right')

    def test_audit_requires_two_distinct_actual_stable_anchors(self):
        diagnostics = {'candidate_descriptions': normalize_descriptions(self.descriptions),
                       'deepseek_result': {'selected_candidate_id': 'C01', 'observed_structures': [
                           {'name': '玻璃墙', 'side': 'center', 'confidence': .9},
                           {'name': '玻璃门', 'side': 'center', 'confidence': .8}]}}
        audit = audit_observations(diagnostics)
        self.assertEqual(len(audit['matched']), 2)
        self.assertEqual(audit['contradictions'], [])
        diagnostics['deepseek_result']['observed_structures'][1]['side'] = 'left'
        self.assertEqual(audit_observations(diagnostics)['contradictions'], ['玻璃门'])

    def test_description_hook_restored_after_checker_exception(self):
        original = lambda *a: self.descriptions
        module = SimpleNamespace(candidate_scene_descriptions=original)
        def checker(*a):
            self.assertEqual(module.candidate_scene_descriptions()[0]['expected_visible_structures'][0]['side'], 'uncertain')
            raise RuntimeError('camera failure')
        with self.assertRaisesRegex(RuntimeError, 'camera failure'):
            review_startup(checker, [None, self.anchor, object(), self.original], self.anchor,
                           self.prior, ds_module=module)
        self.assertIs(module.candidate_scene_descriptions, original)

    def test_bad_positive_choice_cannot_disguise_itself_as_abstention(self):
        module = SimpleNamespace(candidate_scene_descriptions=lambda *a: self.descriptions)
        def checker(*a):
            return 'confirmed', self.anchor, object(), {
                **self.capture_diagnostics,
                'candidate_descriptions': module.candidate_scene_descriptions(),
                'deepseek_result': {'selected_candidate_id': 'C01', 'observed_structures': [
                    {'name': '玻璃墙', 'side': 'center', 'confidence': .9}]}}
        status, _, match, diagnostics = review_startup(
            checker, [None, self.anchor, object(), self.original], self.anchor,
            self.prior, ds_module=module)
        self.assertEqual(status, 'uncertain')
        self.assertIsNone(match)
        self.assertEqual(diagnostics['deepseek_result']['selected_candidate_id'], 'C01')
        self.assertEqual(diagnostics['rejected_reason'], 'startup_insufficient_observed_anchors')

    def test_true_direction_conflict_stays_conflict(self):
        module = SimpleNamespace(candidate_scene_descriptions=lambda *a: self.descriptions)
        def checker(*a):
            return 'confirmed', self.anchor, object(), {
                **self.capture_diagnostics,
                'candidate_descriptions': module.candidate_scene_descriptions(),
                'deepseek_result': {'selected_candidate_id': 'C01', 'observed_structures': [
                    {'name': '玻璃门', 'side': 'left', 'confidence': .9}]}}
        status, _, _, diagnostics = review_startup(
            checker, [None, self.anchor, object(), self.original], self.anchor,
            self.prior, ds_module=module)
        self.assertEqual(status, 'conflict')
        self.assertEqual(diagnostics['rejected_reason'], 'startup_structure_direction_conflict')

    def test_released_deepseek_checker_runs_through_bounded_adapter(self):
        # Exercise the actual unchanged install/check implementation, with
        # camera and API as IO doubles. No camera, network or motor access.
        path = Path(__file__).resolve().parents[2] / 'deepseek_visual_localization.py'
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / 'maps/semantic_release_20260919/deepseek_visual_localization.py'
        spec = importlib.util.spec_from_file_location('startup_review_release_test', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        config = SimpleNamespace(configured=True, semantic_map=Path('unused'), source_path=Path('unused'),
                                 model='mock', maximum_candidates=6, minimum_confidence=.7,
                                 minimum_margin=.15, repeat_count=2)
        rank_result = SimpleNamespace(selected_candidate_id='C01', second_confidence=0., confidence=.85,
                                      margin=.85, consistent_repeats=2)
        rank_result.to_dict = lambda: {
            'selected_candidate_id': 'C01', 'observed_structures': [
                {'name': '玻璃墙', 'side': 'center', 'confidence': .9},
                {'name': '玻璃门', 'side': 'center', 'confidence': .8}]}
        rank_calls = []
        def rank(left, right, descriptions):
            rank_calls.append(descriptions)
            return rank_result
        mission = SimpleNamespace(VisualPlaceMatch=SimpleNamespace)
        with patch.object(module, 'load_config', return_value=config), \
             patch.object(module, '_load_semantic_map', return_value={'objects': [{}]}), \
             patch.object(module, 'DeepSeekVisionClient', return_value=SimpleNamespace(rank=rank)), \
             patch.object(module, 'candidate_scene_descriptions', return_value=self.descriptions):
            self.assertTrue(module.install(mission))
            with tempfile.TemporaryDirectory() as directory:
                def capture(path):
                    path.write_bytes(b'left camera IO fixture')
                    second = path.with_name(path.stem+'_camera_2.jpg')
                    second.write_bytes(b'right camera IO fixture')
                    return {'camera_count': 2, 'camera_2_image': second.name}
                args = SimpleNamespace(visual_localization_max_position_error=.55,
                                       visual_localization_max_yaw_error_deg=30.)
                status, visual, _, diagnostics = review_startup(
                    mission._visual_localization_check,
                    [args, self.anchor, object(), self.original, SimpleNamespace(capture=capture),
                     None, None, Path(directory), 1], self.anchor, self.prior, ds_module=module)
        self.assertEqual(status, 'confirmed')
        self.assertIs(visual, self.anchor)
        self.assertEqual(len(rank_calls), 1)
        self.assertEqual(rank_calls[0][0]['expected_visible_structures'][0]['side'], 'uncertain')
        self.assertFalse(diagnostics['startup_visual_is_global_localization'])
        self.assertEqual(len(diagnostics['startup_direction_audit']['matched']), 2)

    def test_same_image_twice_cannot_approve_even_genuine_abstention(self):
        module = SimpleNamespace(candidate_scene_descriptions=lambda *a: self.descriptions)
        diagnostics = {**self.capture_diagnostics, 'deepseek_result': {
            'selected_candidate_id': 'uncertain', 'consistent_repeats': 2}}
        diagnostics['capture'] = [diagnostics['capture'][0]] * 2
        status, _, _, details = review_startup(
            lambda *a: ('uncertain', self.anchor, None, diagnostics),
            [None, self.anchor, object(), self.original], self.anchor, self.prior, ds_module=module)
        self.assertEqual(status, 'error')
        self.assertEqual(details['rejected_reason'], 'startup_missing_distinct_camera_pair')


if __name__ == '__main__':
    unittest.main()
