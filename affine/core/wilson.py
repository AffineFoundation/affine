from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Tuple


_NORMAL = NormalDist()


def wilson_interval(wins: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    if wins < 0 or wins > trials:
        raise ValueError("wins must be between 0 and trials (inclusive).")
    z = _NORMAL.inv_cdf(0.5 + confidence / 2.0)
    phat = wins / trials
    denom = 1.0 + z * z / trials
    center = phat + z * z / (2 * trials)
    margin = z * sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def beats_target(wins: int, trials: int, target: float, confidence: float) -> bool:
    lower, _ = wilson_interval(wins, trials, confidence)
    return lower > target


def loses_to_target(wins: int, trials: int, target: float, confidence: float) -> bool:
    _, upper = wilson_interval(wins, trials, confidence)
    return upper < target


@dataclass(frozen=True)
class SequentialState:
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def trials(self) -> int:
        return self.wins + self.losses

    def record(self, outcome: str) -> "SequentialState":
        if outcome == "contender":
            return SequentialState(self.wins + 1, self.losses, self.ties)
        if outcome == "champion":
            return SequentialState(self.wins, self.losses + 1, self.ties)
        return SequentialState(self.wins, self.losses, self.ties + 1)


__all__ = ["SequentialState", "beats_target", "loses_to_target", "wilson_interval"]
