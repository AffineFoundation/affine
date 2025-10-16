from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

from ..core.types import DuelResult
from .arena import duel_env
from .ratio import RatioSchedule


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
    """
    Evaluate contender vs champion across multiple environments.

    Args:
        stream_factory: returns an iterable of per-challenge outcomes for the given env.
                        True=contender win, False=champion win, None=tie.
    """
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
        result = duel_env(stream, env_id=env_id, ratio_to_beat_env=per_env_ratio, z=z, max_budget=max_budget)
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
        ratio_snapshot = min(0.95, _geometric_mean(winning_ratios)) if winning_ratios else ratio_to_beat_global
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


__all__ = ["duel_many_envs"]
