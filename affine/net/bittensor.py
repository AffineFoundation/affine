from __future__ import annotations

from typing import Sequence


def metagraph(netuid: int):
    import bittensor as bt
    return bt.subtensor().metagraph(netuid)


def set_weights(
    netuid: int,
    uids: Sequence[int],
    weights: Sequence[float],
    *,
    wallet=None,
    wait_for_inclusion: bool = True,
    prompt: bool = False,
    dry_run: bool = False,
):
    if dry_run:
        return {"uids": list(uids), "weights": list(weights)}
    import bittensor as bt
    return bt.subtensor().set_weights(
        netuid=netuid,
        uids=list(uids),
        weights=list(weights),
        wait_for_finalization=wait_for_inclusion,
        prompt=prompt,
        wallet=wallet,
    )


__all__ = ["metagraph", "set_weights"]
