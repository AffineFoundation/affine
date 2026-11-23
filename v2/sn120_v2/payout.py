"""Payout. Pure functions: journal events and chain registrations in, weights out.

By default the payees are the last ``kings_window`` distinct winner UIDs,
newest first. With ``null_prune_from_round``, every judged-null round from
that round onward removes the oldest member of that set, down to one. New
crowns move their UID to the front, and a full set evicts its oldest member.
Pruned UIDs return only by winning again. ``payees`` remains the original
last-window rule used for copy protection.

A payee is paid only while the hotkey that earned the win is still registered
on that UID. Registered, non-burn payees split all weight equally. The pure
function falls back to ``burn_uid`` when nobody qualifies; the weights writer
refuses that table and keeps the last valid weights instead.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

from .errors import ContractError
from .params import Params
from .types import King

_ROUND_ID = re.compile(r"r\d{6}")


def payees(kings: Sequence[King], window: int) -> list[King]:
    """The newest win of each of the last ``window`` distinct UIDs, newest first. ``kings`` is oldest first."""
    seen: set[int] = set()
    chosen: list[King] = []
    for king in reversed(kings):
        if king.uid in seen:
            continue
        seen.add(king.uid)
        chosen.append(king)
        if len(chosen) == window:
            break
    return chosen


def _round_number(round_id: str) -> int:
    """Imported v1 crowns precede every v2 round. Other ids must have their canonical six-digit form."""
    if round_id.startswith("v1-e"):
        return -1
    if _ROUND_ID.fullmatch(round_id):
        return int(round_id[1:])
    raise ContractError(f"payout round id is neither v1-e<epoch> nor rNNNNNN: {round_id!r}")


def _simulate_paid_set(kings: Sequence[King], null_rounds: Sequence[str], params: Params) -> tuple[list[King], int]:
    """Return the paid candidates and the cumulative number of removals by judged-null rounds."""
    if not params.null_prune_from_round:
        return payees(kings, params.kings_window), 0
    cutoff = int(params.null_prune_from_round[1:])
    events: list[tuple[int, int, King | str]] = []
    for king in kings:
        if _round_number(king.round_id) >= cutoff:
            events.append((_round_number(king.round_id), 0, king))
    for round_id in null_rounds:
        if _round_number(round_id) >= cutoff:
            # A crown cannot share a round with a null close. The tie order makes
            # this total order deterministic even for malformed synthetic inputs.
            events.append((_round_number(round_id), 1, round_id))
    events.sort(key=lambda event: event[:2])
    paid = payees([king for king in kings if _round_number(king.round_id) < cutoff], params.kings_window)
    pruned = 0
    for _, kind, event in events:
        if kind == 0:
            king = event
            assert isinstance(king, King)
            paid = [king, *(old for old in paid if old.uid != king.uid)][:params.kings_window]
        elif len(paid) > 1:
            paid.pop()
            pruned += 1
    return paid, pruned


def paid_set(kings: Sequence[King], null_rounds: Sequence[str], params: Params) -> list[King]:
    """Paid-set candidates after crowns and judged nulls, newest first. ``kings`` and ``null_rounds`` are oldest first."""
    return _simulate_paid_set(kings, null_rounds, params)[0]


def pruned_by_null(kings: Sequence[King], null_rounds: Sequence[str], params: Params) -> int:
    """How many kings judged-null rounds actually removed (floor events at one do not count)."""
    return _simulate_paid_set(kings, null_rounds, params)[1]


def paid_payees(kings: Sequence[King], registered: Mapping[int, str], window: int) -> list[King]:
    """Payees whose earning hotkey is still registered on the same UID, under the original window rule."""
    return [king for king in payees(kings, window) if registered.get(king.uid) == king.hotkey]


def weights(kings: Sequence[King], registered: Mapping[int, str], params: Params, null_rounds: Sequence[str] = ()) -> dict[int, float]:
    """Chain weights by UID. They always sum to 1 and are never all zero.

    ``registered`` maps each UID on chain to its current hotkey.
    """
    paid = [king for king in paid_set(kings, null_rounds, params) if registered.get(king.uid) == king.hotkey and king.uid != params.burn_uid]
    if not paid:
        return {params.burn_uid: 1.0}
    share = 1.0 / len(paid)
    return check_weights({king.uid: share for king in paid})


def check_weights(table: Mapping[int, float]) -> dict[int, float]:
    """Refuse a weight table that is empty, all zero, negative, non-finite or not normalised."""
    if not table:
        raise ContractError("refusing an empty weight table")
    for uid, weight in table.items():
        if type(uid) is not int or uid < 0 or not math.isfinite(weight) or weight < 0:
            raise ContractError(f"refusing weight {weight!r} for uid {uid!r}")
    total = math.fsum(table.values())
    if total <= 0 or not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ContractError(f"refusing weights that sum to {total!r}")
    return dict(table)
