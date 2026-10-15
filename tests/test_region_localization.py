"""Ordinary region navigation must not use compulsory-move initialization."""
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1 import startup_active_recovery as recovery
from services.track.final_work.pi import region_localization as region


INSTALL = 'grasp_range_integration.phase1.relationship_find_only_entry._install_midrun_visual_recheck'


class RegionLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.cancel = False
        self.pose = Pose2D(-1.1, -2.3, 1.7)
        self.candidate = NS(pose=self.pose, reliable=True, rmse_m=.03,
                            inlier_ratio=.9, ambiguity_margin=.15)
        self.mission = NS(_phase1_recovery_mode='navigation', _phase1_motion_fault=None,
            _localization_acceptance=lambda rows, args: (True, self.candidate, 1., 4, 4))
        self.client = NS(stop=Mock(), startup_recovery_pulse=Mock())
        self.controller = NS(stopped=Mock())
        self.known_map = NS(traversable=np.ones((5, 9), dtype=bool), resolution_m=.1,
            grid_to_world=lambda cells: np.asarray(cells, dtype=float)*.1+.05)
        self.localizer = NS(global_localize=Mock(return_value=self.candidate),
                            map=self.known_map, track=Mock())
        self.args = NS(localization_timeout_seconds=120., localization_min_margin=.025,
                       localization_required_moves=3)
        self.events = []
        self.scans = self.new_scans()

    def new_scans(self):
        for sequence in range(1, 100):
            self.now += .2
            yield NS(sequence=sequence, acquisition_start_s=self.now, received_s=self.now,
                angles_rad=np.linspace(-math.pi, math.pi, 360, endpoint=False),
                distances_m=np.full(360, 2.))

    def run_localizer(self):
        return region.localize_region(self.mission, self.args, self.client, self.scans,
            self.localizer, self.controller, lambda: self.cancel, clock=lambda: self.now,
            event_callback=lambda name, **fields: self.events.append((name, fields)))

    def actual_recovery(self, match):
        actual = recovery.recover_startup
        def timed(*args, **kwargs):
            self.assertFalse(kwargs['settings'].operator_prompt)
            self.assertTrue(kwargs['settings'].fresh_start_handoff)
            return actual(*args, clock=lambda: self.now, **kwargs)
        matcher = NS(diagnostic={}, visual_candidates=[], match=Mock(side_effect=match))
        with patch.object(region, 'recover_startup', side_effect=timed), \
                patch.object(recovery, 'StartupMatcher', return_value=matcher), patch(INSTALL) as install:
            result = self.run_localizer()
        return result, matcher, install

    def test_reproduced_strong_near_zero_motion_does_not_require_any_move(self):
        (pose, scan), matcher, install = self.actual_recovery(lambda *args: self.candidate)
        self.assertIs(pose, self.pose)
        self.assertEqual(scan.sequence, 6)  # seed + four fits + new independent frame
        self.assertEqual(matcher.match.call_count, 5)
        self.assertEqual(self.localizer.global_localize.call_count, 1)
        self.client.startup_recovery_pulse.assert_not_called()
        self.assertEqual(self.args.localization_required_moves, 3)  # no global parameter mutation
        self.assertIs(install.call_args.args[-1], scan)
        self.assertIs(install.call_args.kwargs['scan_iterator'], self.scans)
        self.assertEqual(self.mission._phase1_recovery_mode, 'navigation')
        self.assertTrue(any(name == 'region_localization_ready' for name, _ in self.events))

    def test_weak_or_ambiguous_global_seed_keeps_whole_map_not_goal_or_origin(self):
        for candidate in (None, NS(**{**vars(self.candidate), 'reliable': False}),
                NS(**{**vars(self.candidate), 'ambiguity_margin': .001}),
                NS(**{**vars(self.candidate), 'rmse_m': .15}),
                NS(**{**vars(self.candidate), 'inlier_ratio': .6})):
            self.assertIsNone(region.seed_search_prior(self.localizer, candidate, self.args))
        self.localizer.global_localize.return_value = None
        with patch.object(region, 'recover_startup', return_value=(self.pose, NS(sequence=40))) as recover, patch(INSTALL):
            self.run_localizer()
        prior = recover.call_args.args[7]
        self.assertAlmostEqual(prior.x_m, .45)
        self.assertAlmostEqual(prior.y_m, .25)
        self.assertGreater(prior.radius_m, math.hypot(.4, .2))
        self.assertEqual(prior.yaw_tolerance_deg, 180.)
        self.assertEqual(recover.call_args.kwargs['settings'].time_budget_s, 120.)

    def test_strong_seed_only_narrows_search_final_pose_can_differ(self):
        new_pose = Pose2D(-.7, -1.9, -.8)
        fresh = NS(sequence=70)
        with patch.object(region, 'recover_startup', return_value=(new_pose, fresh)) as recover, patch(INSTALL):
            pose, scan = self.run_localizer()
        self.assertIs(pose, new_pose)
        self.assertIs(scan, fresh)
        self.assertEqual(recover.call_args.args[7].radius_m, 1.5)
        self.assertEqual(recover.call_args.args[7].yaw_tolerance_deg, 180.)
        self.assertFalse(next(fields for name, fields in self.events
                              if name == 'region_localization_started')['seed_is_navigation_pose'])

    def test_scoring_delay_reacquires_new_frame_before_handoff(self):
        def match(scan, deadline):
            if scan.sequence == 6:
                self.now += .6
            return self.candidate
        (pose, scan), _, _ = self.actual_recovery(match)
        self.assertEqual(scan.sequence, 7)
        self.client.startup_recovery_pulse.assert_not_called()

    def test_changed_or_cancelled_post_fit_cannot_release_old_pose(self):
        def match(scan, deadline):
            if scan.sequence == 6:
                self.cancel = True
                return NS(**{**vars(self.candidate), 'pose': Pose2D(2., 2., .1)})
            return self.candidate
        with self.assertRaises(RuntimeError):
            self.actual_recovery(match)
        self.assertFalse(any(name == 'region_localization_ready' for name, _ in self.events))
        self.client.startup_recovery_pulse.assert_not_called()

    def test_stale_buffered_seed_is_not_scored(self):
        import itertools
        stale = NS(sequence=-1, acquisition_start_s=-2., received_s=-1.)
        self.scans = itertools.chain([stale], self.scans)
        with patch.object(region, 'recover_startup', return_value=(self.pose, NS(sequence=9))), patch(INSTALL):
            self.run_localizer()
        self.assertEqual(self.localizer.global_localize.call_args.args[0].sequence, 1)

    def test_fault_during_seed_does_not_enter_motion_recovery(self):
        def failed(scan, hint):
            self.mission._phase1_motion_fault = 'link_failed'
            return self.candidate
        self.localizer.global_localize.side_effect = failed
        with patch.object(region, 'recover_startup') as recover, patch(INSTALL) as install:
            with self.assertRaises(RuntimeError):
                self.run_localizer()
        recover.assert_not_called()
        install.assert_not_called()
        self.assertEqual(self.mission._phase1_recovery_mode, 'navigation')

    def test_cancel_before_start_never_scores_or_sends_motion(self):
        self.cancel = True
        with patch.object(region, 'recover_startup') as recover:
            with self.assertRaises(RuntimeError):
                self.run_localizer()
        recover.assert_not_called()
        self.localizer.global_localize.assert_not_called()
        self.client.startup_recovery_pulse.assert_not_called()

    def test_transport_failure_restores_controller_and_mode_without_handoff(self):
        self.localizer.global_localize.side_effect = OSError('scan disconnected')
        with patch(INSTALL) as install:
            with self.assertRaises(OSError):
                self.run_localizer()
        self.controller.stopped.assert_called_once()
        self.assertEqual(self.mission._phase1_recovery_mode, 'navigation')
        install.assert_not_called()

    def test_empty_map_is_not_guessed(self):
        self.known_map.traversable[:] = False
        with self.assertRaises(ValueError):
            region.map_search_prior(self.known_map)

    def test_whole_map_camera_heading_can_select_opposite_basin_before_fresh_consensus(self):
        opposite = NS(**{**vars(self.candidate), 'pose': Pose2D(-2., -.5, -2.6)})
        self.localizer.global_localization_candidates = Mock(return_value=[self.candidate, opposite])
        self.mission._visual_localization_check = Mock(return_value=(
            'corrected', opposite, NS(reliable=True), {}))
        new_scan = NS(sequence=100)
        with patch.object(region, 'recover_startup', return_value=(opposite.pose, new_scan)) as recover, patch(INSTALL):
            pose, scan = region.localize_region(self.mission, self.args, self.client, self.scans,
                self.localizer, self.controller, lambda: False, clock=lambda: self.now,
                camera=object(), visual_recognizer=object(), gpu=object(),
                event_callback=lambda name, **fields: self.events.append((name, fields)))
        prior = recover.call_args.args[7]
        self.assertAlmostEqual(prior.yaw_deg, math.degrees(-2.6))
        self.assertEqual(prior.radius_m, .55)
        self.assertEqual(recover.call_args.kwargs['settings'].heading_tolerance_deg, 35.)
        # Still run the independent four scans and post-fit handoff, never
        # directly promote the camera-selected candidate into a drive pose.
        self.assertTrue(recover.call_args.kwargs['settings'].fresh_start_handoff)
        self.assertIs(pose, opposite.pose)
        self.assertIs(scan, new_scan)
        proxy = self.mission._visual_localization_check.call_args.args[3]
        self.assertEqual(len(proxy.global_localization_candidates(None)), 2)
        self.assertIs(proxy.localize_near_position(None, opposite.pose), opposite)
        self.assertFalse(self.localizer.track.called)

    def test_global_visual_abstention_or_invented_pose_does_not_override_lidar(self):
        self.localizer.global_localization_candidates = Mock(return_value=[self.candidate])
        invented = NS(**{**vars(self.candidate), 'pose': Pose2D(9, 9, 0)})
        for status, winner in [('uncertain', self.candidate), ('corrected', invented)]:
            self.mission._visual_localization_check = Mock(return_value=(status, winner, NS(), {}))
            prior = region.review_global_heading(self.mission, self.args, self.candidate, NS(),
                self.localizer, object(), object(), object(), Path('.'), lambda *a, **kw: None)
            self.assertIsNone(prior)
        self.client.startup_recovery_pulse.assert_not_called()

    def test_legacy_guided_correction_cannot_change_selected_heading(self):
        opposite = NS(**{**vars(self.candidate), 'pose': Pose2D(-1.1, -2.3, -1.4)})
        self.localizer.global_localization_candidates = Mock(return_value=[self.candidate, opposite])
        def check(*args):
            proxy = args[3]
            self.assertIs(proxy.localize_near_position(None, opposite.pose), opposite)
            self.assertIsNone(proxy.localize_near_position(None, Pose2D(8, 8, 0)))
            return 'confirmed', opposite, NS(), {}
        self.mission._visual_localization_check = check
        prior = region.review_global_heading(self.mission, self.args, self.candidate, NS(),
            self.localizer, object(), object(), object(), Path('.'), lambda *a, **kw: None)
        self.assertAlmostEqual(prior.yaw_deg, math.degrees(-1.4))

    def test_deepseek_global_selection_requires_distinct_pair_and_two_observed_anchors(self):
        import sys
        import tempfile
        module = NS(candidate_scene_descriptions=lambda *a: [{
            'candidate_id': 'C01', 'expected_visible_structures': [
                {'name': '门', 'category': 'door', 'relative_bearing_deg': 0,
                 'evidence_role': 'stable_anchor'},
                {'name': '楼梯', 'category': 'stairs', 'relative_bearing_deg': 25,
                 'evidence_role': 'stable_anchor'}]}])
        original = module.candidate_scene_descriptions
        self.mission._phase1_deepseek_enabled = True
        self.localizer.global_localization_candidates = Mock(return_value=[self.candidate])
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'deepseek_visual_localization': module}):
            left, right = Path(directory)/'left.jpg', Path(directory)/'right.jpg'
            left.write_bytes(b'left'); right.write_bytes(b'right')
            diagnostics = {'camera_count': 2, 'capture': [str(left), str(right)],
                'deepseek_reliable': True, 'deepseek_result': {
                'selected_candidate_id': 'C01', 'observed_structures': [
                    {'name': '门', 'side': 'center', 'confidence': .9},
                    {'name': '楼梯', 'side': 'left', 'confidence': .9}]}}
            def check(*args):
                diagnostics['candidate_descriptions'] = module.candidate_scene_descriptions([self.candidate], {'objects': []})
                return 'confirmed', self.candidate, NS(), diagnostics
            self.mission._visual_localization_check = check
            def run():
                result = region.review_global_heading(self.mission, self.args, self.candidate, NS(),
                    self.localizer, object(), object(), object(), Path(directory), lambda *a, **kw: None)
                self.assertIs(module.candidate_scene_descriptions, original)
                return result
            self.assertIsNotNone(run())
            diagnostics['capture'] = [str(left), str(left)]
            self.assertIsNone(run())
            diagnostics['capture'] = [str(left), str(right)]
            diagnostics['deepseek_result']['observed_structures'][1]['side'] = 'right'
            self.assertIsNone(run())
            diagnostics['deepseek_result']['observed_structures'].pop()
            self.assertIsNone(run())

    def test_front_door_observation_excludes_rear_door_basin_even_when_ds_abstains(self):
        semantic = {'objects': [{'name': '玻璃门', 'category': 'door', 'map_xy': [3., 0.],
            'semantic_reviewed': True, 'position_uncertainty_m': .85}]}
        bad = NS(pose=Pose2D(0., 0., math.pi), reliable=True, score=1.2)
        good = NS(pose=Pose2D(0., 0., 0.), reliable=True, score=.8)
        diagnostics = {'deepseek_result': {'selected_candidate_id': 'uncertain',
            'observed_structures': [{'name': '玻璃门', 'side': 'right', 'confidence': .8}]}}
        prior, audit = region.positive_structure_prior([bad, good],
            region.point_bearings([bad, good], semantic), diagnostics)
        self.assertIsNotNone(prior)
        self.assertAlmostEqual(prior.yaw_deg, 0.)
        self.assertEqual(prior.heading_evidence, 'positive_structure_constraints')
        self.assertEqual(audit['excluded_candidate_ids'], ['C01'])
        self.assertFalse(audit['visually_verified'])

    def test_live_style_ds_abstention_keeps_front_door_constraint_through_recovery(self):
        import sys
        import tempfile
        bad = NS(**{**vars(self.candidate), 'pose': Pose2D(.205, .365, math.radians(-36)), 'score': 1.2})
        good = NS(**{**vars(self.candidate), 'pose': Pose2D(-1.55, -2.6, math.radians(140)), 'score': .8})
        semantic = {'objects': [{'name': '玻璃门', 'category': 'door',
            'map_xy': [-2.9196918033, -.8362125834], 'semantic_reviewed': True, 'position_uncertainty_m': .85}]}
        module = NS(candidate_scene_descriptions=lambda *a: [
            {'candidate_id': 'C01', 'expected_visible_structures': []},
            {'candidate_id': 'C02', 'expected_visible_structures': []}])
        original = module.candidate_scene_descriptions
        self.mission._phase1_deepseek_enabled = True
        self.localizer.global_localize.return_value = bad
        self.localizer.global_localization_candidates = Mock(return_value=[bad, good])
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'deepseek_visual_localization': module}):
            left, right = Path(directory)/'left.jpg', Path(directory)/'right.jpg'
            left.write_bytes(b'left'); right.write_bytes(b'right')
            def check(*args):
                module.candidate_scene_descriptions([bad, good], semantic, NS())
                return 'uncertain', bad, None, {'camera_count': 2, 'capture': [str(left), str(right)],
                    'deepseek_reliable': False, 'deepseek_result': {'selected_candidate_id': 'uncertain',
                    'consistent_repeats': 2, 'observed_structures': [
                    {'name': '玻璃门', 'side': 'center', 'confidence': .8}]}}
            self.mission._visual_localization_check = check
            with patch.object(region, 'recover_startup', return_value=(good.pose, NS(sequence=40))) as recover, patch(INSTALL):
                region.localize_region(self.mission, self.args, self.client, self.scans,
                    self.localizer, self.controller, lambda: False, clock=lambda: self.now,
                    camera=object(), visual_recognizer=object(), gpu=object())
            prior = recover.call_args.args[7]
            self.assertAlmostEqual(prior.x_m, good.pose.x_m)
            self.assertAlmostEqual(prior.y_m, good.pose.y_m)
            self.assertAlmostEqual(prior.yaw_deg, 140.)
            self.assertTrue(self.mission._region_start_localization['positive_heading_constraint_applied'])
            self.assertFalse(self.mission._region_start_localization['visual_heading_verified'])
            self.assertIs(module.candidate_scene_descriptions, original)

    def test_not_seen_is_not_a_contradiction_and_remote_survivors_are_not_collapsed(self):
        rows = [NS(pose=Pose2D(0., 0., math.pi), score=1.2),
                NS(pose=Pose2D(-3., 0., 0.), score=.8), NS(pose=Pose2D(3., 0., 0.), score=.7)]
        descriptions = [{'candidate_id': f'C{i+1:02d}', 'points': [{
            'name': '玻璃门', 'relative_bearing_deg': 180. if i == 0 else 0.,
            'bearing_uncertainty_deg': 20.}]} for i in range(3)]
        self.assertIsNone(region.positive_structure_prior(rows, descriptions,
            {'deepseek_result': {'observed_structures': []}})[0])
        prior, audit = region.positive_structure_prior(rows, descriptions,
            {'deepseek_result': {'observed_structures': [{'name': '玻璃门', 'confidence': .9}]}})
        self.assertIsNone(prior)
        self.assertEqual(audit['reason'], 'surviving_basins_still_spatially_ambiguous')

    def test_extended_walls_or_uncertainty_disk_cannot_be_point_bearing_veto(self):
        rows = [NS(pose=Pose2D(0., 0., 0.))]
        semantic = {'objects': [
            {'name': '玻璃墙', 'category': 'glass_wall', 'map_xy': [-3., 0.],
             'semantic_reviewed': True, 'position_uncertainty_m': .5},
            {'name': '门', 'category': 'door', 'map_xy': [.3, 0.],
             'semantic_reviewed': True, 'position_uncertainty_m': .85}]}
        self.assertEqual(region.point_bearings(rows, semantic)[0]['points'], [])

    def test_maintenance_and_deployed_source_copies_match(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('region_navigation.py', 'region_localization.py'):
            self.assertEqual((root / 'services/track/final_work/pi' / name).read_bytes(),
                (root / 'robot/grasp_range_integration/track_web_integration/pi' / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
