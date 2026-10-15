from __future__ import annotations

from typing import Protocol

import numpy as np

from qwen_grounded_tracker.domain import GroundingResult


class TargetGrounder(Protocol):
    def ground(
        self,
        frame: np.ndarray,
        instruction: str,
        reference_image: np.ndarray | None = None,
    ) -> GroundingResult:
        ...
