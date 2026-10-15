"""Boundary regression using the real acceptance helper, without hardware."""

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from raspberry_pi.mapping import Pose2D
from raspberry_pi.known_map_navigation import LocalizationResult, ScanMapLocalizer
from raspberry_pi.map_camera_inspection import _localization_acceptance
from grasp_range_integration.phase1.startup_prior import assess_start_prior, inside_start_prior
from grasp_range_integration.phase1.startup_localization import StartupMatcher
from grasp_range_integration.phase1.startup_visual_review import in_scope
from grasp_range_integration.phase1.relationship_find_only_entry import (
    _inside_start_prior, _install_stationary_start_localization,
)


class StartupPriorTests(unittest.TestCase):
    def setUp(self):
        self.prior = SimpleNamespace(x_m=-1.55, y_m=-2.6, yaw_deg=100.,
                                     radius_m=.7, yaw_tolerance_deg=40.)

    def result(self, yaw=140., x=-1.31, y=-2.66):
        return LocalizationResult(Pose2D(x, y, math.radians(yaw)), .04, .85, 1., .2)

    def test_all_gates_agree_at_140_degree_inclusive_boundary(self):
        row = self.result(yaw=140.+1e-12)
        matcher = StartupMatcher(SimpleNamespace(global_localize=lambda *a, **kw: row), self.prior)
        self.assertIsNotNone(matcher.match(object(), time.monotonic()+10))
        self.assertTrue(inside_start_prior(row.pose, self.prior))
        self.assertTrue(_inside_start_prior(row.pose, self.prior))
        self.assertTrue(in_scope(row, row, self.prior))

    def test_142_degree_and_old_position_slack_rejected_by_every_gate(self):
        for row in [self.result(yaw=142.), self.result(yaw=100., x=-.80, y=-2.6)]:
            with self.subTest(pose=row.pose):
                matcher = StartupMatcher(SimpleNamespace(global_localize=lambda *a, **kw: row), self.prior)
                self.assertIsNone(matcher.match(object(), time.monotonic()+10))
                self.assertEqual(matcher.diagnostic['result'], 'outside_start_prior')
                self.assertFalse(_inside_start_prior(row.pose, self.prior))
                self.assertFalse(in_scope(row, row, self.prior))
                self.assertIn('pose', matcher.diagnostic)
                self.assertFalse(matcher.diagnostic['prior_check']['inside'])

    def test_position_boundary_wraparound_and_nonfinite(self):
        self.assertTrue(inside_start_prior(self.result(yaw=100., x=-.85, y=-2.6).pose, self.prior))
        self.assertFalse(inside_start_prior(self.result(yaw=100., x=-.849999, y=-2.6).pose, self.prior))
        prior = SimpleNamespace(x_m=0.,y_m=0.,yaw_deg=175.,radius_m=.7,yaw_tolerance_deg=10.)
        self.assertTrue(inside_start_prior(Pose2D(0.,0.,math.radians(-175.)), prior))
        self.assertFalse(inside_start_prior(Pose2D(0.,0.,math.nan), prior))

    def test_scoring_never_ranks_outside_pose_even_if_it_would_score_best(self):
        valid, outside = self.result(), self.result(yaw=142.)
        calls = []
        def rank(points, poses, **kwargs):
            calls.append(poses)
            self.assertTrue(poses)
            self.assertTrue(all(inside_start_prior(p, self.prior) for p in poses))
            return [(1., p, (.04,.85)) for p in poses]
        localizer = SimpleNamespace(
            points=lambda *a, **kw: np.zeros((100,2)), range_profile=lambda *a, **kw: None,
            _pose_lattice=lambda *a, **kw: [outside.pose, valid.pose], _rank=rank,
            _result_from_ranked=ScanMapLocalizer._result_from_ranked)
        matcher = StartupMatcher(localizer, self.prior)
        for _ in range(2):
            row = matcher.match(object(), time.monotonic()+10)
            self.assertIsNotNone(row, matcher.diagnostic)
            self.assertAlmostEqual(math.degrees(row.pose.yaw_rad),140.)
        self.assertEqual(len(calls),3)  # coarse, refined, next-frame local
        self.assertGreater(matcher.diagnostic['excluded_lattice_poses'],0)

    def run_entry(self, row, *, frames=7, acceptance=_localization_acceptance):
        args = SimpleNamespace(localization_window=5,localization_min_accepted=3,
                               localization_acceptance_ratio=.6,localization_timeout_seconds=10.)
        visual_calls, stops, events = [], [], []
        def check(*a):
            visual_calls.append(a[-1])
            return 'confirmed', row, object(), {}
        mission = SimpleNamespace(_localization_acceptance=acceptance,_visual_localization_check=check)
        _install_stationary_start_localization(mission,self.prior)
        client = SimpleNamespace(stop=lambda:stops.append(True))
        localizer = SimpleNamespace(global_localize=lambda *a,**kw:row,track=lambda *a:row)
        with redirect_stdout(io.StringIO()):
            try:
                result = mission._localize(args,client,iter([object()]*frames),localizer,object(),
                    lambda:False,camera=object(),visual_recognizer=SimpleNamespace(available=True),
                    visual_directory=Path('search_runs'),
                    event_callback=lambda name,**data:events.append((name,data)))
                return result,visual_calls,stops,events,None
            except RuntimeError as exc:
                return None,visual_calls,stops,events,str(exc)

    def test_real_consensus_at_heading_boundary_enters_visual_review(self):
        result,calls,stops,events,error = self.run_entry(self.result())
        self.assertIsNone(error)
        self.assertEqual(calls,[1])
        self.assertTrue(_inside_start_prior(result[0],self.prior))
        self.assertEqual(sum(n=='startup_scan_scored' for n,d in events),4)
        self.assertGreaterEqual(len(stops),2)

    def test_outside_candidate_reports_boundary_not_scan_inconsistency(self):
        result,calls,stops,events,error = self.run_entry(self.result(yaw=142.))
        self.assertIsNone(result)
        self.assertEqual(calls,[])
        self.assertIn('超出固定起点范围',error)
        rejected = next(d for n,d in events if n=='startup_lidar_rejected')
        self.assertEqual(rejected['reason'],'outside_start_heading')
        self.assertAlmostEqual(rejected['prior_rejection']['prior_check']['yaw_error_deg'],42.)
        self.assertGreaterEqual(len(stops),2)

    def test_three_frames_even_if_consistent_cannot_authorize_start(self):
        result,calls,stops,events,error = self.run_entry(self.result(),frames=3)
        self.assertIsNone(result)
        self.assertEqual(calls,[])
        self.assertIn('扫描流提前结束',error)
        self.assertTrue(stops)

    def test_persistent_consensus_rejection_not_overwritten_by_loop_else(self):
        outside = self.result(yaw=142.)
        fake_acceptance = lambda rows,args: (len(rows)>=4,outside,1.,len(rows),len(rows))
        result,calls,stops,events,error = self.run_entry(self.result(),acceptance=fake_acceptance)
        self.assertIsNone(result)
        self.assertEqual(calls,[])
        self.assertIn('超出固定起点范围',error)
        self.assertTrue(stops)
        final = next(d for n,d in events if n=='startup_lidar_rejected')
        self.assertEqual(final['reason'],'consensus_outside_start_prior')

    def test_missing_reliable_match_not_mislabeled_as_consensus_failure(self):
        result,calls,stops,events,error = self.run_entry(None)
        self.assertIsNone(result)
        self.assertEqual(calls,[])
        self.assertIn('没有达到质量要求',error)
        final = next(d for n,d in events if n=='startup_lidar_rejected')
        self.assertEqual(final['reason'],'no_reliable_lidar_match')

    def test_scoring_timeout_has_distinct_rejection(self):
        fake = SimpleNamespace(match=lambda *a:None, diagnostic={
            'mode':'bounded_start_search','elapsed_s':20.,'result':'budget_exhausted'})
        with patch('grasp_range_integration.phase1.startup_localization.StartupMatcher',return_value=fake):
            result,calls,stops,events,error = self.run_entry(self.result())
        self.assertIsNone(result)
        self.assertEqual(calls,[])
        self.assertIn('定位超时',error)
        final = next(d for n,d in events if n=='startup_lidar_rejected')
        self.assertEqual(final['reason'],'startup_scoring_budget_exhausted')


if __name__ == '__main__':
    unittest.main()
