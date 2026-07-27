import numpy as np

from qwen_grounded_tracker.domain import BBox
from qwen_grounded_tracker.tracking.identity_guard import IdentityGuard


def test_identity_guard_accepts_same_color_target() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[50:150, 50:150] = (255, 0, 0)
    bbox = BBox(50, 50, 150, 150)
    guard = IdentityGuard(minimum_score=0.2)
    guard.initialize(frame, bbox)
    result = guard.evaluate(frame, bbox)
    assert result.accepted
    assert result.score > 0.8
