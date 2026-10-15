"""Command encoding tests without importing Flask/cv2 or any BLE worker."""
import ast
import math
import unittest
import threading
from types import SimpleNamespace
from pathlib import Path


class MicroDriveProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'integrations/grab_visual/arm_teach.py'
        if not source.exists():
            source = Path('/home/dachuang/grab_visual/arm_teach.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in {'_micro_command', '_run_chassis_pulse'}
            or isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '_MICRO_COMMANDS'
                for target in node.targets))]
        clock = [0.0]
        namespace = {'math': math, '_micro_cancel': threading.Event(),
                     'time': SimpleNamespace(monotonic=lambda: clock[0],
                                             sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        cls.command = staticmethod(namespace['_micro_command'])
        cls.run_pulse = staticmethod(namespace['_run_chassis_pulse'])
        cls.clock = clock
        cls.cancel = namespace['_micro_cancel']

    def setUp(self):
        self.clock[0] = 0.0
        self.cancel.clear()

    def test_default_verified_approach_speed_is_retained_in_both_directions(self):
        self.assertEqual(self.command('forward', 405), '#MOVF400L000S074!')
        self.assertEqual(self.command('backward', 405), '#MOVB400L000S074!')

    def test_speed_setting_is_not_ignored_by_the_ble_owner(self):
        self.assertEqual(self.command('forward', 350), '#MOVF400L000S064!')

    def test_invalid_or_excessive_speed_never_produces_a_motor_command(self):
        for speed in (float('nan'), float('inf'), -1, 99, 406, 550):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                self.command('backward', speed)

    def test_turn_command_and_unknown_action_are_not_treated_as_forward(self):
        self.assertEqual(self.command('turn_left'), '#MOVF000L120S013!')
        with self.assertRaises(ValueError):
            self.command('unknown')

    def test_turn_has_separate_bounded_setting_without_changing_forward(self):
        self.assertEqual(self.command('turn_left',405,900), '#MOVF000L120S026!')
        self.assertEqual(self.command('turn_right',405,2100), '#MOVF000R120S060!')
        self.assertEqual(self.command('forward',405,2100), '#MOVF400L000S074!')
        for value in (449,2401,float('nan'),float('inf')):
            with self.assertRaises(ValueError): self.command('turn_left',405,value)

    def test_configured_speed_reaches_motor_and_every_step_ends_with_stop(self):
        commands = []
        self.run_pulse(lambda command: commands.append(command) or 'OK',
                       'forward', 0.8, 350)
        self.assertEqual(commands[0], '#MOVF400L000S064!')
        self.assertEqual(commands[-1], '#STOP!')
        self.assertAlmostEqual(self.clock[0], 0.8)

    def test_fast_visual_feedback_stops_yaw_before_time_limit(self):
        commands=[]
        result=self.run_pulse(lambda command: commands.append(command) or 'OK',
            'turn_left',.8,405,2400,
            movement_feedback=lambda:{'stop':True,'movement_px':{'left':9,'right':10}})
        self.assertEqual(commands[-1],'#STOP!')
        self.assertEqual(result['termination'],'visual_motion_limit')
        self.assertLess(self.clock[0],.8)

    def test_visual_feedback_failure_still_stops(self):
        commands=[]
        def failed(): raise RuntimeError('camera stale')
        with self.assertRaisesRegex(RuntimeError,'camera stale'):
            self.run_pulse(lambda command: commands.append(command) or 'OK',
                'turn_right',.8,405,2400,movement_feedback=failed)
        self.assertEqual(commands[-1],'#STOP!')

    def test_motor_write_failure_still_sends_stop(self):
        commands = []
        def send(command):
            commands.append(command)
            return 'ERROR' if command.startswith('#MOV') else 'OK'
        with self.assertRaisesRegex(RuntimeError, 'MOV 未写入'):
            self.run_pulse(send, 'backward', 0.8)
        self.assertEqual(commands[-1], '#STOP!')


if __name__ == '__main__':
    unittest.main()
