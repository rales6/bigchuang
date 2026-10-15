"""The pursuit mask is a staging stop, not a graspability verdict."""
from __future__ import annotations

import threading
import unittest
import tempfile
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from final_work.pi import pursuit_session


class PursuitHandoffTests(unittest.TestCase):
    def handoff_fixture(self, metadata_changes=None, row_changes=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        log_root = root / 'logs' / 'pursuit'
        run = log_root / 'current'
        run.mkdir(parents=True)
        (log_root / 'LATEST.txt').write_text(str(run), encoding='utf-8')
        metadata = {'pid': 1234, 'mode': 'esp32', 'active_instruction': '红色方块',
                    'final_result': 'alignment_pending', 'error': '',
                    'ended_at': datetime.now().astimezone().isoformat()}
        metadata.update(metadata_changes or {})
        (run / 'run.json').write_text(json.dumps(metadata), encoding='utf-8')
        rows = [dict(t_s=t, tracking='1', tracking2='1', state='TRACKING',
                     v_mm_s='405', w_mrad_s='0', bbox_cy_px=str(300 + 10 * t),
                     bbox_cy2_px=str(250 + 10 * t)) for t in (12, 13, 14)]
        for row in rows:
            row.update(row_changes or {})
        with (run / 'telemetry.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return root

    def test_fresh_matching_straight_dual_approach_grants_short_retreat(self):
        root = self.handoff_fixture()
        with patch.object(pursuit_session, 'TRACK_ROOT', root):
            budget = pursuit_session._recent_pursuit_retreat('红色方块', 1234)
        self.assertIsNotNone(budget)
        self.assertGreater(budget.expires_at, time.monotonic())
        self.assertEqual(budget.pulses, 0)

    def test_wrong_child_mode_target_or_failed_run_cannot_grant_retreat(self):
        for changes in ({'pid': 9999}, {'mode': 'dry_run'},
                        {'active_instruction': '手'}, {'final_result': 'give_up'},
                        {'error': 'BLE disconnected'}, {'ended_at': '2020-01-01T00:00:00+08:00'}):
            with self.subTest(changes=changes):
                root = self.handoff_fixture(metadata_changes=changes)
                with patch.object(pursuit_session, 'TRACK_ROOT', root):
                    self.assertIsNone(pursuit_session._recent_pursuit_retreat('红色方块', 1234))

    def test_turn_reverse_one_eye_stationary_or_no_visual_progress_blocks_retreat(self):
        for changes in ({'w_mrad_s': '320'}, {'v_mm_s': '-405'},
                        {'tracking2': '0'}, {'v_mm_s': '0'},
                        {'bbox_cy2_px': '300'}, {'w_mrad_s': 'nan'}):
            with self.subTest(changes=changes):
                root = self.handoff_fixture(row_changes=changes)
                with patch.object(pursuit_session, 'TRACK_ROOT', root):
                    self.assertIsNone(pursuit_session._recent_pursuit_retreat('红色方块', 1234))

    def test_handoff_forwards_budget_to_alignment(self):
        budget = object()
        with patch.object(pursuit_session, '_restore_preview'), \
             patch.object(pursuit_session, '_wait_preview_ready', return_value=True), \
             patch.object(pursuit_session, '_event'), \
             patch('pursuit.hull_micro_adjust.align', return_value=SimpleNamespace(
                 ok=True, reason='inside', pulses=1, log_path='test')) as align:
            pursuit_session._finish_at_exact_hull('红色方块', threading.Event(), budget)
        self.assertIs(align.call_args.kwargs['retreat_budget'], budget)

    def test_extracts_exact_target_without_red_only_branch(self):
        self.assertEqual(pursuit_session._target_from_args(
            ["--instruction", "矿泉水瓶", "--esp32"]), "矿泉水瓶")
        self.assertEqual(pursuit_session._target_from_args(
            ["--instruction=奶茶杯"]), "奶茶杯")

    def test_mask_stop_requires_exact_hull_before_success(self):
        with patch.object(pursuit_session, "_restore_preview"), \
             patch.object(pursuit_session, "_wait_preview_ready", return_value=True), \
             patch.object(pursuit_session, "_event"), \
             patch("pursuit.hull_micro_adjust.align", return_value=SimpleNamespace(
                 ok=False, reason="rear obstacle", pulses=0,
                 log_path="/tmp/test-hull.jsonl")) as align:
            self.assertEqual(pursuit_session._finish_at_exact_hull(
                "红色方块", threading.Event()), 2)
        align.assert_called_once()
        self.assertTrue(align.call_args.kwargs["real"])

    def test_inside_hull_can_complete_handoff(self):
        with patch.object(pursuit_session, "_restore_preview"), \
             patch.object(pursuit_session, "_wait_preview_ready", return_value=True), \
             patch.object(pursuit_session, "_event"), \
             patch("pursuit.hull_micro_adjust.align", return_value=SimpleNamespace(
                 ok=True, reason="inside hull", pulses=1,
                 log_path="/tmp/test-hull.jsonl")):
            self.assertEqual(pursuit_session._finish_at_exact_hull(
                "红色方块", threading.Event()), 0)

    def test_no_ble_never_calls_alignment(self):
        with patch.object(pursuit_session, "_restore_preview"), \
             patch.object(pursuit_session, "_wait_preview_ready", return_value=False), \
             patch.object(pursuit_session, "_event"), \
             patch("pursuit.hull_micro_adjust.align") as align:
            self.assertEqual(pursuit_session._finish_at_exact_hull(
                "红色方块", threading.Event()), 2)
        align.assert_not_called()


if __name__ == "__main__":
    unittest.main()
