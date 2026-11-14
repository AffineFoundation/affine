"""Payout. Pure functions: the list of kings and the chain's registrations in, weights out.

Rule at launch (last-10-kings):

* A king is a round winner that was applied. A null round adds no king.
* The payees are the last ``kings_window`` distinct winner UIDs. A new
  distinct UID pushes out the oldest. A UID that wins again keeps one place,
  with its newest win.
* A payee is paid only while the hotkey that earned the win is still
  registered on that UID.
* Each paid payee gets an equal share, 1/N, with N = the number of paid payees.
* Weight that no one earned goes to ``burn_uid``. So the weights are never
  all zero: with no paid payee, all weight goes to ``burn_uid``.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .errors import ContractError
from .params import Params
from .types import King


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


def paid_payees(kings: Sequence[King], registered: Mapping[int, str], window: int) -> list[King]:
    """Payees whose earning hotkey is still registered on the same UID."""
    return [king for king in payees(kings, window) if registered.get(king.uid) == king.hotkey]


def weights(kings: Sequence[King], registered: Mapping[int, str], params: Params) -> dict[int, float]:
    """Chain weights by UID. They always sum to 1 and are never all zero.

    ``registered`` maps each UID on chain to its current hotkey.
    """
    paid = [king for king in paid_payees(kings, registered, params.kings_window) if king.uid != params.burn_uid]
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
