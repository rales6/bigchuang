from qwen_grounded_tracker.domain import BBox
from qwen_grounded_tracker.navigation.direction_planner import DirectionPlanner


def test_direction_planner_turns_and_moves() -> None:
    planner = DirectionPlanner(center_tolerance=0.05, stop_area_ratio=0.15)
    assert planner.plan(BBox(0, 100, 100, 200), 640, 480).direction == "TURN_LEFT"
    assert planner.plan(BBox(540, 100, 640, 200), 640, 480).direction == "TURN_RIGHT"
    assert planner.plan(BBox(280, 180, 360, 260), 640, 480).direction == "FORWARD"


def test_direction_planner_stops_without_target() -> None:
    planner = DirectionPlanner()
    assert planner.plan(None, 640, 480).direction == "STOP"
