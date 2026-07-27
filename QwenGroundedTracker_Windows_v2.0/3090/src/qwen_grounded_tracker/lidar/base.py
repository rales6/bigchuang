from __future__ import annotations

from typing import Protocol

from qwen_grounded_tracker.domain import LidarObservation


class LidarProvider(Protocol):
    def open(self) -> None:
        ...

    def read(self) -> LidarObservation:
        ...

    def close(self) -> None:
        ...
