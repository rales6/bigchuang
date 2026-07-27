from qwen_grounded_tracker.domain import (
    LidarObservation,
    MotionGuidance,
    ObstacleObservation,
)
from qwen_grounded_tracker.safety.arbiter import SafetyArbiter


def test_null_lidar_does_not_block_demo_when_not_required() -> None:
    arbiter = SafetyArbiter(require_lidar_ready=False)
    decision = arbiter.decide(
        MotionGuidance("FORWARD", 0.1, 0.0, "test"),
        tracking_visible=True,
        yolo_obstacles=ObstacleObservation(danger=False),
        lidar=LidarObservation(False, False, None, "placeholder"),
        emergency_stop=False,
    )
    assert not decision.blocked
    assert decision.guidance.direction == "FORWARD"


def test_null_lidar_blocks_when_required() -> None:
    arbiter = SafetyArbiter(require_lidar_ready=True)
    decision = arbiter.decide(
        MotionGuidance("FORWARD", 0.1, 0.0, "test"),
        tracking_visible=True,
        yolo_obstacles=ObstacleObservation(danger=False),
        lidar=LidarObservation(False, False, None, "placeholder"),
        emergency_stop=False,
    )
    assert decision.blocked
    assert decision.guidance.direction == "STOP"
