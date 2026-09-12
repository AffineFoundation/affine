"""Outcome of one verifiers trace, standalone.

Same rules as affine.corpus.view.rollout_outcome / affine.corpus.trace
(turn-cap artifact, primary reward keys), duplicated here because the
datagen pods carry an older affine tree that lacks these helpers and the
driver must not depend on the pod's copy.
"""

from __future__ import annotations

TURN_CAP_STOP = "max_turns"
TURN_CAP_ARTIFACT = "rollout stopped: max_turns"
CLEAN_STOP_CONDITIONS = frozenset({"agent_completed", "max_turns"})
PRIMARY_REWARD_KEYS = ("solved", "correct", "passed_fraction")


def is_turn_cap_artifact(err: dict, trace: dict) -> bool:
    if trace.get("stop_condition") != TURN_CAP_STOP:
        return False
    return TURN_CAP_ARTIFACT in str(err.get("message") or "")


def real_errors(trace: dict) -> list[dict]:
    return [e for e in (trace.get("errors") or [])
            if not is_turn_cap_artifact(e, trace)]


def primary_score(trace: dict):
    rewards = trace.get("rewards") or {}
    return next(((rewards.get(k) or {}).get("score")
                 for k in PRIMARY_REWARD_KEYS if rewards.get(k)), None)


def rollout_outcome(trace: dict) -> str:
    """"solved" / "failed" / "errored" / "unscored"."""
    if real_errors(trace):
        return "errored"
    if trace.get("stop_condition") not in CLEAN_STOP_CONDITIONS:
        return "errored"
    score = primary_score(trace)
    if isinstance(score, bool) or not isinstance(score, (int, float, str)):
        if trace.get("stop_condition") == TURN_CAP_STOP:
            return "failed"
        return "unscored"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "unscored"
    return "solved" if value >= 1.0 else "failed"
