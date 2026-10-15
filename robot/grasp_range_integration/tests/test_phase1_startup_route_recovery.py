"""Offline startup-route handoff tests; no hardware or API connections."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from raspberry_pi.mapping import Pose2D
from grasp_range_integration.phase1 import startup_route_recovery as adapter
from grasp_range_integration.phase1.visibility_route_planner import NoReachableVisibilityError, POLICY


class StartupRouteRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.decision = Path(self.tmp.name)/'decision.json'
        self.decision.write_text(json.dumps({'route_request': {'mode':'semantic_search_region',
            'planning_policy':POLICY, 'center_xy_m':[.875, 1.044], 'radius_m':.95}}), encoding='utf-8')
        self.args = NS(decision=self.decision)
        self.old_pose = Pose2D(-1.61, -3.12, 2.443)
        self.old_scan = NS(sequence=1)
        self.new_pose = Pose2D(-1.534, -2.981, 2.4)
        self.new_scan = NS(sequence=20)
        self.mission = NS(_localize=Mock(return_value=(self.old_pose,self.old_scan)),
                          build_search_route=Mock(return_value={'views':[{}]}))
        self.client = NS(stop=Mock(), startup_recovery_pulse=Mock())
        self.controller = NS(stopped=Mock())
        self.localizer = NS(map=object())
        self.scans = iter([])
        adapter.install_startup_route_recovery(self.mission)

    def run_adapter(self):
        return self.mission._localize(self.args,self.client,self.scans,self.localizer,
                                     self.controller,lambda:False)

    def test_reachable_start_never_starts_recovery(self):
        with patch.object(adapter,'recover_startup') as recover:
            pose,scan=self.run_adapter()
        self.assertIs(pose,self.old_pose); self.assertIs(scan,self.old_scan)
        recover.assert_not_called();self.client.startup_recovery_pulse.assert_not_called()

    def test_blocked_start_recovers_and_returns_new_pose_and_scan_together(self):
        self.mission.build_search_route.side_effect=[NoReachableVisibilityError('none'),
            NoReachableVisibilityError('same old basin'), {'views':[{}]}]
        def recover(*args,**kwargs):
            self.assertFalse(kwargs['accept_candidate'](NS(pose=self.old_pose)))
            self.assertTrue(kwargs['accept_candidate'](NS(pose=self.new_pose)))
            self.assertIs(kwargs['reference_scan'],self.old_scan)
            self.assertFalse(kwargs['settings'].operator_prompt)
            self.assertFalse(kwargs['settings'].initial_fit)
            self.assertEqual(self.mission._phase1_recovery_mode,'midrun')
            return NS(candidate=NS(pose=self.new_pose),scan=self.new_scan)
        with patch.object(adapter,'recover_startup',side_effect=recover):
            pose,scan=self.run_adapter()
        self.assertIs(pose,self.new_pose);self.assertIs(scan,self.new_scan)
        self.assertIsNone(self.mission._phase1_recovery_mode)
        self.client.stop.assert_called_once();self.controller.stopped.assert_called_once()

    def test_program_or_transport_error_cannot_authorize_recovery(self):
        self.mission.build_search_route.side_effect=OSError('transport failure')
        with patch.object(adapter,'recover_startup') as recover:
            with self.assertRaises(OSError):self.run_adapter()
        recover.assert_not_called()

    def test_tracking_adapter_replaces_old_baseline_instead_of_stacking(self):
        raw=Mock(); old_adapter=Mock();self.localizer.track=raw
        def localize(*args,**kwargs):
            self.localizer.track=old_adapter
            return self.old_pose,self.old_scan
        self.mission._localize=localize;adapter.install_startup_route_recovery(self.mission)
        self.mission.build_search_route.side_effect=NoReachableVisibilityError('none')
        result=NS(candidate=NS(pose=self.new_pose),scan=self.new_scan)
        with patch.object(adapter,'recover_startup',return_value=result), patch(
                'grasp_range_integration.phase1.relationship_find_only_entry._install_midrun_visual_recheck') as install:
            self.run_adapter()
        self.assertIs(self.localizer.track,raw)
        self.assertIs(install.call_args.args[-1],self.new_scan)


if __name__=='__main__':unittest.main()
