from __future__ import annotations

from grab_target.automation_logic import (
    INITIAL_DEPLOY_SEQUENCE,
    autonomous_motion_ready,
    choose_camera_from_qwen,
    grasp_motion_verified,
    select_requested_object,
    stored_tracking_boxes,
)
from grab_target.arm_calibration import DEFAULT_CALIBRATION
from grab_target.visual_servo import VisualServoController
from grab_target.grasp_sequence import GraspSequence, Phase
from grab_target.grasp_runner import Runner


def test_select_requested_object_does_not_substitute_high_confidence_item():
    objects = [
        {"name": "蓝色瓶子", "confidence": 0.99, "pickable": True},
        {"name": "红色方块", "confidence": 0.72, "pickable": True},
    ]
    selected = select_requested_object(objects, "红色方块")
    assert selected is not None
    assert selected["name"] == "红色方块"


def test_select_requested_object_rejects_non_matching_items():
    objects = [{"name": "蓝色瓶子", "confidence": 0.99, "pickable": True}]
    assert select_requested_object(objects, "红色方块") is None


def test_select_requested_object_rejects_wrong_color_same_shape():
    objects = [{"name": "蓝色方块", "confidence": 0.99, "pickable": True}]
    assert select_requested_object(objects, "红色方块") is None


def test_grasp_verification_requires_target_to_follow_claw():
    before = {
        "claw_x": 100.0, "claw_y": 200.0,
        "target_x": 112.0, "target_y": 176.0,
        "rel_x": 12.0, "rel_y": -24.0,
    }
    followed = {
        "claw_x": 100.0, "claw_y": 170.0,
        "target_x": 113.0, "target_y": 146.0,
        "rel_x": 13.0, "rel_y": -24.0,
    }
    stayed = {
        "claw_x": 100.0, "claw_y": 170.0,
        "target_x": 112.0, "target_y": 176.0,
        "rel_x": 12.0, "rel_y": 6.0,
    }
    assert grasp_motion_verified(before, followed)[0] is True
    assert grasp_motion_verified(before, stayed)[0] is False


def test_visual_recovery_replays_motion_history_in_reverse_order():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    servo.record_move(20, 0, 0)
    servo.record_move(0, -8, 6)

    first = servo.rollback_cmd()
    second = servo.rollback_cmd()

    assert first is not None
    # One axis at a time, clamped below half a manual-button step.
    assert (first.forward_mm, first.up_mm, first.left_deg) == (0, 0, -5)
    assert second is not None
    assert (second.forward_mm, second.up_mm, second.left_deg) == (-8, 0, 0)


def test_sequence_stays_in_carry_until_explicit_release():
    sequence = GraspSequence("red block")
    sequence.phase = Phase.RETRACT
    sequence.advance_to_carry()
    assert sequence.phase == Phase.CARRY
    sequence.advance_to_release()
    assert sequence.phase == Phase.RELEASE


def test_runner_uses_injected_mjpeg_capture_without_reimporting_web_app():
    jpeg = b"\xff\xd8" + b"x" * 1200 + b"\xff\xd9"
    runner = Runner("ws://unused", "red block", capture=lambda: jpeg)
    assert runner._capture() == jpeg


def test_runner_can_switch_between_injected_camera_streams():
    left = b"\xff\xd8" + b"L" * 1200 + b"\xff\xd9"
    right = b"\xff\xd8" + b"R" * 1200 + b"\xff\xd9"
    runner = Runner(
        "ws://unused", "red block",
        captures={"left": lambda: left, "right": lambda: right},
    )
    assert runner._capture() == left
    runner._active_camera = "right"
    assert runner._capture() == right


def test_qwen_camera_choice_prefers_clearer_confident_view():
    parsed = {
        "left": {"found": True, "confidence": 0.62, "bbox_2d": [100, 200, 220, 400]},
        "right": {"found": True, "confidence": 0.91, "bbox_2d": [600, 180, 760, 420]},
    }
    assert choose_camera_from_qwen(parsed) == "right"


def test_empty_tracker_overlay_does_not_require_target_ready_property():
    from types import SimpleNamespace

    tracker = SimpleNamespace(
        claw=SimpleNamespace(initialized=False, bbox=None),
        target=SimpleNamespace(initialized=False, bbox=None),
    )
    assert stored_tracking_boxes(tracker) is None

    tracker.claw.initialized = True
    tracker.claw.bbox = (10, 20, 30, 40)
    payload = stored_tracking_boxes(tracker)
    assert payload is not None
    assert payload["claw_ok"] is True
    assert payload["target_ok"] is False


def test_csrt_tuple_is_converted_to_domain_bbox():
    import numpy as np
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from grab_target.dual_tracker import DualTracker

    src = Path(__file__).resolve().parents[1] / "3090" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    class FakeTracker:
        def update(self, _frame):
            return True, (10, 20, 30, 40)

    ok, bbox = DualTracker._update_csrt(
        np.zeros((100, 120, 3), dtype=np.uint8),
        SimpleNamespace(tracker=FakeTracker()),
    )
    assert ok is True
    assert (bbox.x1, bbox.y1, bbox.width, bbox.height) == (10, 20, 30, 40)


def test_initial_deploy_is_exactly_six_web_d20_commands():
    assert len(INITIAL_DEPLOY_SEQUENCE) == 6
    assert all((forward, up) == (20, 0)
               for _label, forward, up, _duration in INITIAL_DEPLOY_SEQUENCE)

    sent = []
    runner = Runner("ws://unused", "red block", ble_send=lambda cmd: sent.append(cmd) or "OK")
    for _label, forward, up, duration in INITIAL_DEPLOY_SEQUENCE:
        runner._arm({
            "action": "move", "forward_mm": forward, "up_mm": up,
            "left_deg": 0, "claw_open": -1, "duration_ms": duration,
        })
    assert sent == ["#ARMD20!"] * 6
def test_deploy_first_frame_returns_d20_command_not_bool():
    import numpy as np
    from types import SimpleNamespace
    from grab_target.arm_grasp_server import GraspSession
    from grab_target.grasp_sequence import GraspSequence

    session = GraspSession.__new__(GraspSession)
    session.sequence = GraspSequence("red block")
    session.sequence.start()
    session._deploy_seq = None
    session._make_response = lambda arm_cmd, tracking=None: {"arm_cmd": arm_cmd}
    response = session._handle_deploy(np.zeros((480, 640, 3), dtype=np.uint8))
    assert isinstance(response, dict)
    assert response["arm_cmd"].forward_mm == 20




def test_visual_motion_limits_are_below_half_manual_button_steps():
    cal = DEFAULT_CALIBRATION
    assert cal.max_left_per_step_deg < 15 / 2
    assert cal.max_up_per_step_mm < 15 / 2
    assert cal.max_forward_per_step_mm < 20 / 2
    probe = VisualServoController(cal).probe_cmd()
    assert probe.forward_mm == 0
    assert probe.up_mm == 0
    assert 0 < abs(probe.left_deg) <= 5
    assert probe.claw_open == -1


def test_autonomous_motion_requires_same_frame_stable_claw_and_target():
    assert autonomous_motion_ready(
        claw_detected_now=True, claw_stable_frames=3, target_ok=True,
    ) is True
    assert autonomous_motion_ready(
        claw_detected_now=False, claw_stable_frames=20, target_ok=True,
    ) is False


def _box(x1, y1, x2, y2):
    from types import SimpleNamespace
    return SimpleNamespace(
        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
        width=float(x2 - x1), height=float(y2 - y1),
    )


def test_overhead_approach_uses_four_mm_depth_steps():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    claw = _box(100, 100, 140, 140)
    target = _box(80, 220, 160, 300)  # tip gap=80px: coarse forward stage
    cmd = servo.compute(claw, target, (480, 640, 3))
    assert cmd.action == "move"
    assert (cmd.forward_mm, cmd.up_mm, cmd.left_deg) == (4, 0, 0)


def test_coarse_depth_can_back_out_after_overshoot():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    claw = _box(100, 100, 140, 160)
    target = _box(95, 172, 145, 222)
    cmd = servo.compute(claw, target, (480, 640, 3))
    assert cmd.action == "move"
    assert (cmd.forward_mm, cmd.up_mm, cmd.left_deg) == (-2, 0, 0)


def test_hover_correction_can_rise_after_downward_overshoot():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    servo._approach_ready = True
    claw = _box(100, 100, 140, 200)
    target = _box(95, 172, 145, 222)
    cmd = servo.compute(claw, target, (480, 640, 3))
    assert cmd.action == "move" and cmd.up_mm > 0


def test_close_range_descent_is_two_mm_single_axis():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    claw = _box(100, 100, 140, 140)
    servo._approach_ready = True
    target = _box(95, 158, 145, 208)  # close enough, still above target
    cmd = servo.compute(claw, target, (480, 640, 3))
    assert cmd.action == "move"
    assert (cmd.forward_mm, cmd.up_mm, cmd.left_deg) == (0, -2, 0)


def test_grasp_height_requires_two_consecutive_frames():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    claw = _box(100, 100, 140, 140)
    servo._was_aligned_v = True
    servo._approach_ready = True
    target = _box(95, 132.5, 145, 182.5)
    first = servo.compute(claw, target, (480, 640, 3))
    second = servo.compute(claw, target, (480, 640, 3))
    assert first.action == "idle"
    assert second.action == "grasp"


def test_edge_recovery_is_gentler_than_normal_alignment():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    cmd = servo.pull_back_cmd(_box(590, 200, 630, 260), (480, 640, 3))
    assert cmd.action == "move"
    assert abs(cmd.left_deg) <= 2
    assert cmd.left_deg > 0
    assert cmd.forward_mm == 0 and cmd.up_mm == 0


def test_horizontal_servo_uses_installed_camera_yaw_sign():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    claw = _box(300, 100, 340, 140)
    target_left = _box(100, 153, 180, 233)
    cmd_left = servo.compute(claw, target_left, (480, 640, 3))
    assert cmd_left.action == "move" and cmd_left.left_deg > 0

    servo.reset()
    target_right = _box(460, 153, 540, 233)
    cmd_right = servo.compute(claw, target_right, (480, 640, 3))
    assert cmd_right.action == "move" and cmd_right.left_deg < 0
    assert autonomous_motion_ready(
        claw_detected_now=True, claw_stable_frames=2, target_ok=True,
    ) is False
    assert autonomous_motion_ready(
        claw_detected_now=True, claw_stable_frames=3, target_ok=False,
    ) is False
def test_near_center_large_target_does_not_oscillate_by_two_degrees():
    servo = VisualServoController(DEFAULT_CALIBRATION)
    claw = _box(200, 100, 300, 180)
    target = _box(219, 198, 323, 278)
    cmd = servo.compute(claw, target, (480, 640, 3))
    assert cmd.state != "align_h"
    assert cmd.left_deg == 0


def test_small_step_budgets_allow_full_gradual_descent():
    sequence = GraspSequence("red block")
    assert DEFAULT_CALIBRATION.max_attempts_per_step >= 32
    assert DEFAULT_CALIBRATION.max_total_attempts >= 100
    assert sequence._max_phase_attempts["fine_align"] >= 40




def test_pi_serializer_uses_web_ui_physical_axis_mapping():
    sent = []
    runner = Runner("ws://unused", "red block", ble_send=lambda cmd: sent.append(cmd) or "OK")
    runner._arm({"action": "move", "forward_mm": 8, "up_mm": 0, "left_deg": 0})
    runner._arm({"action": "move", "forward_mm": -8, "up_mm": 0, "left_deg": 0})
    runner._arm({"action": "move", "forward_mm": 0, "up_mm": 6, "left_deg": 0})
    runner._arm({"action": "move", "forward_mm": 0, "up_mm": -6, "left_deg": 0})
    assert sent == ["#ARMD8!", "#ARMU8!", "#ARMB6!", "#ARMF6!"]


def test_runner_builds_labelled_side_by_side_camera_jpeg():
    import cv2
    import numpy as np

    def jpeg(color):
        frame = np.full((120, 160, 3), color, dtype=np.uint8)
        frame[:, ::2] = 255 - color
        ok, encoded = cv2.imencode(".jpg", frame)
        assert ok
        return encoded.tobytes()

    runner = Runner(
        "ws://unused", "red block", w=80, h=60,
        captures={"left": lambda: jpeg(30), "right": lambda: jpeg(220)},
    )
    combined = runner._capture_comparison()
    assert combined is not None
    decoded = cv2.imdecode(np.frombuffer(combined, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (60, 160)
