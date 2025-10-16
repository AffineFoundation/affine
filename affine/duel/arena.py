from __future__ import annotations

from statistics import NormalDist
from typing import Iterable, Iterator, Optional

from ..core.types import DuelResult
from ..core.wilson import wilson_interval

_NORMAL = NormalDist()


def _confidence_from_z(z: float) -> float:
    return max(0.0, min(1.0, 2 * (_NORMAL.cdf(z) - 0.5)))


def _iter_stream(stream: Iterable[Optional[bool]]) -> Iterator[Optional[bool]]:
    if callable(stream):  # type: ignore
        return iter(stream())  # type: ignore[arg-type]
    return iter(stream)


def duel_env(
    stream_results: Iterable[Optional[bool]],
    *,
    env_id: str = "",
    ratio_to_beat_env: float = 0.51,
    z: float = 1.96,
    max_budget: int = 200,
) -> DuelResult:
    """Sequential duel on a single environment using Wilson stopping."""
    wins = losses = ties = 0
    trials = 0
    confidence = _confidence_from_z(z)
    iterator = _iter_stream(stream_results)
    for outcome in iterator:
        if outcome is None:
            ties += 1
            continue
        if outcome:
            wins += 1
        else:
            losses += 1
        trials = wins + losses
        if trials == 0:
            continue
        lower, upper = wilson_interval(wins, trials, confidence)
        if lower > ratio_to_beat_env:
            return DuelResult(
                env_id=env_id,
                outcome="contender",
                wins=wins,
                losses=losses,
                ties=ties,
                trials=trials,
                ci=(lower, upper),
            )
        if upper < ratio_to_beat_env:
            return DuelResult(
                env_id=env_id,
                outcome="champion",
                wins=wins,
                losses=losses,
                ties=ties,
                trials=trials,
                ci=(lower, upper),
            )
        if trials >= max_budget:
            break
    lower, upper = wilson_interval(wins, max(trials, 1), confidence)
    return DuelResult(
        env_id=env_id,
        outcome="inconclusive",
        wins=wins,
        losses=losses,
        ties=ties,
        trials=trials,
        ci=(lower, upper),
    )


__all__ = ["duel_env"]
