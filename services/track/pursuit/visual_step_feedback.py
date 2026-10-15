"""Session-local stop/look pulse adaptation, with no hardware dependencies."""
from dataclasses import dataclass
import math


@dataclass
class VisualStepFeedback:
    level: int = 0
    stationary: int = 0
    moving: int = 0
    baseline: tuple | None = None
    last_change_px: float | None = None
    exhausted: bool = False

    def complete(self, signature, *, max_level=3):
        if signature is None or self.baseline is None:
            self.baseline = None
            self.stationary = self.moving = 0
            return 'unknown'
        change = max(abs(a-b) for a, b in zip(signature, self.baseline))
        self.baseline = None
        self.last_change_px = change
        if not math.isfinite(change):
            return 'unknown'
        if change <= 3.0:
            self.stationary += 1
            self.moving = 0
            if self.stationary >= 2:
                self.stationary = 0
                if self.level >= max_level:
                    self.exhausted = True
                else:
                    self.level += 1
            return 'stationary'
        if change >= 6.0:
            self.stationary = 0
            self.moving += 1
            if self.moving >= 2:
                self.level = max(0, self.level-1)
                self.moving = 0
            return 'moving'
        self.stationary = self.moving = 0
        return 'uncertain'

    def cancel_pending(self):
        # Lost/stale frames must never teach that a pulse failed to move.
        self.baseline = None
        self.stationary = self.moving = 0
