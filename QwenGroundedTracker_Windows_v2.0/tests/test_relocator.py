import numpy as np

from qwen_grounded_tracker.domain import BBox
from qwen_grounded_tracker.tracking.relocator import ReferenceRelocator


def test_relocator_fallback_preserves_normalized_box() -> None:
    snapshot = np.zeros((100, 200, 3), dtype=np.uint8)
    current = np.zeros((200, 400, 3), dtype=np.uint8)
    result = ReferenceRelocator(template_match_threshold=1.1).relocate(
        snapshot,
        BBox(50, 20, 150, 80),
        current,
    )
    assert result.method == "normalized_snapshot_fallback"
    assert result.bbox.to_xywh() == (100, 40, 200, 120)
