from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Tuple

from ..net import bittensor as bt_net
from .merge import VerifiedSample, decide_group


def scoreboard(
    buckets: Mapping[tuple[str, str], Mapping[str, Mapping[str, VerifiedSample]]],
    vtrust: Mapping[str, float],
) -> Dict[int, float]:
    scores: Dict[int, float] = defaultdict(float)
    for (_env_id, _challenge_id), validators in buckets.items():
        for validator, roles in validators.items():
            if "contender" not in roles or "champion" not in roles:
                continue
            if not roles["contender"].valid or not roles["champion"].valid:
                continue
            winner = decide_group(roles)
            if winner is None:
                continue
            weight = vtrust.get(validator, 0.0)
            if weight <= 0.0:
                continue
            sample = roles[winner].sample
            try:
                uid = int(sample.miner_id)
            except ValueError:
                continue
            scores[uid] += weight
    return dict(scores)


def winner_takes_all(scores: Mapping[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    winner = max(scores.items(), key=lambda item: item[1])[0]
    return {winner: 1.0}


def set_weights(
    netuid: int,
    scores: Mapping[int, float],
    *,
    dry_run: bool = False,
    wallet=None,
):
    weights = winner_takes_all(scores)
    if not weights:
        return {}
    uids = list(weights.keys())
    wvals = [weights[uid] for uid in uids]
    return bt_net.set_weights(netuid, uids, wvals, wallet=wallet, dry_run=dry_run)


__all__ = ["scoreboard", "set_weights", "winner_takes_all"]
