"""No-hardware tests for the shared-BLE, local-only chassis pulse endpoint."""
from __future__ import annotations

import os
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

try:
    from flask import Flask
    try:
        from integrations.grab_visual import arm_teach
    except ImportError:
        import importlib.util
        deployed = Path("/home/dachuang/grab_visual/arm_teach.py")
        spec = importlib.util.spec_from_file_location("deployed_arm_teach", deployed)
        if spec is None or spec.loader is None:
            raise ImportError("Pi arm_teach.py unavailable")
        arm_teach = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arm_teach)
except ImportError:  # The Windows logic-only venv intentionally lacks Flask/cv2.
    Flask = None
    arm_teach = None


@unittest.skipUnless(arm_teach is not None, "requires Pi grab_visual dependencies")
class ArmTeachMicroTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.token_path = Path(self.directory.name) / "token"
        self.token_path.write_text("unit-test-secret", encoding="utf-8")
        environment = patch.dict(os.environ, {
            "FINAL_WORK_HARDWARE_TOKEN_FILE": str(self.token_path)})
        environment.start()
        self.addCleanup(environment.stop)
        self.commands: list[str] = []
        arm_teach._micro_cancel.clear()
        arm_teach._micro_cancelled_ids.clear()
        self.app = Flask(__name__)
        arm_teach.register(self.app, jpegs={"left": b"", "right": b""},
                           ble_send=self._send)
        self.client = self.app.test_client()

    def _send(self, command: str) -> str:
        self.commands.append(command)
        return "OK"

    def _post(self, action: str, *, token: str = "unit-test-secret",
              remote: str = "127.0.0.1"):
        return self.client.post(
            "/api/arm-teach/chassis-pulse",
            json={"action": action, "duration_s": 0.4,
                  "pulse_id": uuid.uuid4().hex},
            headers={"X-Robot-Safety-Token": token},
            environ_base={"REMOTE_ADDR": remote})

    def test_nonlocal_and_missing_token_never_write_ble(self):
        self.assertEqual(self._post("forward", remote="192.0.2.1").status_code, 403)
        self.assertEqual(self._post("forward", token="incorrect").status_code, 403)
        self.assertEqual(self.commands, [])

    def test_shared_ble_forward_pulse_always_stops(self):
        response = self._post("forward")
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(self.commands[0], "#MOVF400L000S074!")
        self.assertEqual(self.commands[-1], "#STOP!")

    def test_invalid_action_never_writes_ble(self):
        self.assertEqual(self._post("spin_forever").status_code, 400)
        self.assertEqual(self.commands, [])

    def test_missing_turn_frames_prevent_any_motor_command(self):
        response=self._post('turn_left')
        self.assertEqual(response.status_code,503)
        self.assertEqual(self.commands,[])

    def test_fast_flow_detects_background_shift_not_small_local_object_change(self):
        import cv2
        import numpy as np
        image=np.random.default_rng(42).integers(0,256,(480,640),dtype=np.uint8)
        def jpeg(value): return cv2.imencode('.jpg',value)[1].tobytes()
        frames={'left':jpeg(image),'right':jpeg(image)}
        feedback=arm_teach._make_turn_feedback(frames)
        local=image.copy()
        local[220:260,300:340]=0
        frames.update(left=jpeg(local),right=jpeg(local))
        self.assertFalse(feedback()['stop'])
        shifted=cv2.warpAffine(image,np.float32([[1,0,10],[0,1,0]]),(640,480))
        frames.update(left=jpeg(shifted),right=jpeg(shifted))
        result=feedback()
        self.assertTrue(result['stop'])
        self.assertGreater(result['movement_px']['left'],8)

    def test_frozen_turn_frame_stops_instead_of_blind_motion(self):
        import cv2
        import numpy as np
        image=np.random.default_rng(43).integers(0,256,(480,640),dtype=np.uint8)
        encoded=cv2.imencode('.jpg',image)[1].tobytes()
        with patch.object(arm_teach.time,'monotonic',return_value=100):
            feedback=arm_teach._make_turn_feedback({'left':encoded,'right':encoded})
        with patch.object(arm_teach.time,'monotonic',return_value=100.6):
            with self.assertRaisesRegex(RuntimeError,'冻结'): feedback()

    def test_arm_job_blocks_chassis_pulse(self):
        with patch.object(arm_teach, "_arm_jobs_pending", 1):
            self.assertEqual(self._post("backward").status_code, 409)
        self.assertEqual(self.commands, [])

    def test_cancel_before_pulse_arrival_prevents_motion(self):
        pulse_id = uuid.uuid4().hex
        headers = {"X-Robot-Safety-Token": "unit-test-secret"}
        cancelled = self.client.post(
            "/api/arm-teach/chassis-cancel",
            json={"pulse_id": pulse_id}, headers=headers)
        self.assertEqual(cancelled.status_code, 200)
        response = self.client.post(
            "/api/arm-teach/chassis-pulse",
            json={"action": "forward", "duration_s": 0.4,
                  "pulse_id": pulse_id}, headers=headers)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.commands, [])

    def test_stop_on_ble_heartbeat_failure(self):
        commands = []

        def send(command):
            commands.append(command)
            return "ERROR" if command == "#PING!" else "OK"

        with self.assertRaisesRegex(RuntimeError, "心跳失败"):
            arm_teach._run_chassis_pulse(send, "forward", 0.4)
        self.assertEqual(commands[-1], "#STOP!")

    def test_stop_request_cancels_pulse(self):
        started = threading.Event()
        commands = []

        def send(command):
            commands.append(command)
            if command.startswith("#MOV"):
                started.set()
            return "OK"

        result = []

        def run():
            try:
                arm_teach._run_chassis_pulse(send, "backward", 0.8)
            except Exception as exc:
                result.append(str(exc))

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(started.wait(1.0))
        arm_teach._micro_cancel.set()
        worker.join(2.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(result)
        self.assertEqual(commands[-1], "#STOP!")


if __name__ == "__main__":
    unittest.main()
