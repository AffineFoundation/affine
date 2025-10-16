from __future__ import annotations

from typing import Iterable, Optional

import pytest

from affine import RatioSchedule, duel_env, duel_many_envs


def test_duel_env_contender_wins_fast() -> None:
    outcomes = [True] * 8
    result = duel_env(outcomes, env_id="env", ratio_to_beat_env=0.5, z=1.96, max_budget=20)
    assert result.outcome == "contender"
    assert result.trials <= len(outcomes)


def test_duel_env_champion_holds() -> None:
    outcomes = [False, False, False, False]
    result = duel_env(outcomes, env_id="env", ratio_to_beat_env=0.6, z=1.96, max_budget=20)
    assert result.outcome == "champion"
    assert result.trials <= len(outcomes)


class _FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def __call__(self) -> float:
        return self._now


def _stream_factory(mapping: dict[str, Iterable[Optional[bool]]]):
    def factory(env_id: str):
        def generator():
            for value in mapping[env_id]:
                yield value

        return generator()

    return factory


def test_duel_many_envs_updates_ratio_schedule() -> None:
    streams = {
        "env/a": [True] * 4,
        "env/b": [True] * 4,
        "env/c": [False] * 4,
    }
    clock = _FakeClock()
    schedule = RatioSchedule(initial=0.51, half_life_seconds=1000, clock=clock)
    factory = _stream_factory(streams)
    result = duel_many_envs(
        contender_uid=42,
        champion_uid=7,
        env_ids=list(streams.keys()),
        stream_factory=factory,
        ratio_to_beat_global=0.51,
        per_env_ratio=0.51,
        ratio_schedule=schedule,
        max_budget=10,
    )
    assert result["winner_uid"] == 42
    assert result["dethroned"] is True
    assert result["stopped_early"] is True
    assert pytest.approx(result["ratio_snapshot"], rel=1e-6) == result["ratio_next"]
    assert result["ratio_to_beat"] == pytest.approx(0.51)
    assert schedule.current() == result["ratio_next"]
