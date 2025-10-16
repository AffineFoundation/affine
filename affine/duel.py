from __future__ import annotations

import math
import time
from statistics import NormalDist
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence

from .core import DuelResult


_NORMAL = NormalDist()


def wilson_interval(
    wins: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    if wins < 0 or wins > trials:
        raise ValueError("wins must be between 0 and trials (inclusive).")
    z = _NORMAL.inv_cdf(0.5 + confidence / 2.0)
    phat = wins / trials
    denom = 1.0 + z * z / trials
    center = phat + z * z / (2 * trials)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def beats_target(wins: int, trials: int, target: float, confidence: float) -> bool:
    lower, _ = wilson_interval(wins, trials, confidence)
    return lower > target


def loses_to_target(wins: int, trials: int, target: float, confidence: float) -> bool:
    _, upper = wilson_interval(wins, trials, confidence)
    return upper < target


def _z_to_confidence(z: float) -> float:
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
    wins = losses = ties = 0
    trials = 0
    confidence = _z_to_confidence(z)
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
                env_id, "contender", wins, losses, ties, trials, (lower, upper)
            )
        if upper < ratio_to_beat_env:
            return DuelResult(
                env_id, "champion", wins, losses, ties, trials, (lower, upper)
            )
        if trials >= max_budget:
            break
    lower, upper = wilson_interval(wins, max(trials, 1), confidence)
    return DuelResult(
        env_id, "inconclusive", wins, losses, ties, trials, (lower, upper)
    )


class RatioSchedule:
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


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def duel_many_envs(
    contender_uid: int,
    champion_uid: int,
    env_ids: Sequence[str],
    *,
    stream_factory: Callable[[str], Iterable[Optional[bool]]],
    ratio_to_beat_global: float = 0.51,
    per_env_ratio: float = 0.51,
    z: float = 1.96,
    max_budget: int = 200,
    ratio_schedule: RatioSchedule | None = None,
) -> Dict[str, object]:
    if not env_ids:
        raise ValueError("At least one environment required.")

    ratio_before = ratio_to_beat_global
    if ratio_schedule is not None:
        ratio_current = ratio_schedule.current()
        ratio_to_beat_global = max(ratio_to_beat_global, ratio_current)
        ratio_before = ratio_to_beat_global

    per_env: Dict[str, DuelResult] = {}
    env_wins_cont = 0
    env_wins_champ = 0
    total_envs = len(env_ids)
    wins_required = max(1, math.ceil(ratio_to_beat_global * total_envs))
    stopped_early = False
    samples_used = 0

    for env_id in env_ids:
        stream = stream_factory(env_id)
        result = duel_env(
            stream,
            env_id=env_id,
            ratio_to_beat_env=per_env_ratio,
            z=z,
            max_budget=max_budget,
        )
        per_env[env_id] = result
        samples_used += result.wins + result.losses + result.ties
        if result.outcome == "contender":
            env_wins_cont += 1
        elif result.outcome == "champion":
            env_wins_champ += 1

        remaining = total_envs - len(per_env)
        if env_wins_cont >= wins_required:
            stopped_early = remaining > 0
            break
        if env_wins_cont + remaining < wins_required:
            stopped_early = remaining > 0
            break

    if env_wins_cont >= wins_required:
        winner_uid = contender_uid
        winning_ratios = [
            result.wins / max(1, result.wins + result.losses)
            for result in per_env.values()
            if result.outcome == "contender"
        ]
        ratio_snapshot = (
            min(0.95, _geometric_mean(winning_ratios))
            if winning_ratios
            else ratio_to_beat_global
        )
        dethroned = True
        if ratio_schedule:
            ratio_schedule.update(ratio_snapshot)
    else:
        winner_uid = champion_uid
        ratio_snapshot = ratio_to_beat_global
        dethroned = False

    if ratio_schedule:
        ratio_after = ratio_schedule.current()
    else:
        ratio_after = ratio_snapshot if dethroned else ratio_to_beat_global

    return {
        "winner_uid": winner_uid,
        "contender_uid": contender_uid,
        "champion_uid": champion_uid,
        "dethroned": dethroned,
        "ratio_to_beat": ratio_before,
        "ratio_snapshot": ratio_snapshot,
        "ratio_next": ratio_after,
        "per_env": per_env,
        "samples_used": samples_used,
        "stopped_early": stopped_early,
        "wins_required": wins_required,
        "env_wins": {"contender": env_wins_cont, "champion": env_wins_champ},
    }


__all__ = [
    "RatioSchedule",
    "beats_target",
    "duel_env",
    "duel_many_envs",
    "loses_to_target",
    "wilson_interval",
]
