from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic

import numpy as np

from qwen_grounded_tracker.domain import GroundingResult
from qwen_grounded_tracker.grounding.base import TargetGrounder


@dataclass
class GroundingResponse:
    request_id: int
    result: GroundingResult
    snapshot: np.ndarray
    instruction: str


class GroundingWorker:
    def __init__(self, grounder: TargetGrounder) -> None:
        self.grounder = grounder
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-grounding")
        self.future: Future[GroundingResponse] | None = None
        self.started_at: float | None = None
        self.request_id = 0

    @property
    def busy(self) -> bool:
        return self.future is not None and not self.future.done()

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, monotonic() - self.started_at)

    def submit(
        self,
        frame: np.ndarray,
        instruction: str,
        reference_image: np.ndarray | None = None,
    ) -> bool:
        if self.busy:
            return False

        self.request_id += 1
        request_id = self.request_id
        snapshot = frame.copy()
        reference = None if reference_image is None else reference_image.copy()
        self.started_at = monotonic()

        def task() -> GroundingResponse:
            result = self.grounder.ground(snapshot, instruction, reference)
            return GroundingResponse(request_id, result, snapshot, instruction)

        self.future = self.executor.submit(task)
        print(f"[Qwen grounder] submitted request {request_id}")
        return True

    def poll(self) -> GroundingResponse | None:
        if self.future is None or not self.future.done():
            return None

        future = self.future
        self.future = None
        self.started_at = None
        return future.result()

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
