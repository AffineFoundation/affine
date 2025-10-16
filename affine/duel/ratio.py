from __future__ import annotations

import math
import time
from typing import Callable


class RatioSchedule:
    """Track dynamic ratio-to-beat with exponential decay."""

    def __init__(
        self,
        initial: float = 0.51,
        *,
        baseline: float = 0.5,
        half_life_seconds: float = 7 * 24 * 3600,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._baseline = baseline
        self._value = max(initial, baseline)
        self._half_life = max(half_life_seconds, 1.0)
        self._clock = clock or time.monotonic
        self._updated = self._clock()

    def current(self) -> float:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        decay = math.exp(-elapsed * math.log(2) / self._half_life)
        return self._baseline + (self._value - self._baseline) * decay

    def update(self, ratio: float) -> None:
        self._value = max(self._baseline, min(0.95, ratio))
        self._updated = self._clock()


__all__ = ["RatioSchedule"]
