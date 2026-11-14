"""The miner queue. Pure functions: no I/O and no clock.

A seat is one paid registration. Turn order has three zones:

  front   seats sent back, not charged: no_decision (our failure), not_trained or not_scored (the round's time ended)
  normal  everyone else
  back    seats moved back because they had no valid file for too long

Inside each zone the order is registration block, then UID. So seats that
return to the front keep their original order. A demoted seat with a valid
file at selection leaves the back (``restore_from_back``) before the round
picks entries.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping

from .errors import InfraError, V2Error
from .types import ChainSnapshot, Entry, FrozenBatch, QueueEntry, Registration, Verdict

RegKey = tuple[int, int]
Queue = Mapping[RegKey, QueueEntry]

FRONT, NORMAL, BACK = 0, 1, 2
SECONDS_PER_DAY = 86400


def zone(seat: QueueEntry) -> int:
    if seat.front:
        return FRONT
    return BACK if seat.demoted else NORMAL


def turn_key(seat: QueueEntry) -> tuple[int, int, int]:
    return (zone(seat), seat.reg.registration_block, seat.reg.uid)


def waiting(queue: Queue) -> list[QueueEntry]:
    """Seats that are not in an open round, in turn order."""
    return sorted((seat for seat in queue.values() if not seat.leased), key=turn_key)


def coldkey_share_report(
    queue: Queue,
    registrations: Iterable[Registration],
) -> dict[str, object]:
    """How many waiting seats share an owner coldkey (read-only impact of the one-seat-per-coldkey rule).

    ``registrations`` is usually a chain snapshot: each seat's coldkey is taken
    from the matching registration when present, else from the seat itself.
    Seats with an empty coldkey are counted under ``missing_coldkey``.
    """
    by_key = {reg.key: reg for reg in registrations}
    groups: dict[str, list[int]] = {}
    missing = 0
    for seat in waiting(queue):
        reg = by_key.get(seat.reg.key, seat.reg)
        coldkey = reg.coldkey
        if not coldkey:
            missing += 1
            continue
        groups.setdefault(coldkey, []).append(seat.reg.uid)
    shared = {coldkey: uids for coldkey, uids in groups.items() if len(uids) > 1}
    return {
        "waiting": len(waiting(queue)),
        "with_coldkey": sum(len(uids) for uids in groups.values()),
        "missing_coldkey": missing,
        "coldkeys_with_2plus_waiting_seats": len(shared),
        "waiting_seats_that_share_a_coldkey": sum(len(uids) for uids in shared.values()),
        "largest_share": max((len(uids) for uids in shared.values()), default=0),
        "shares": [
            {"coldkey": coldkey, "uids": uids, "count": len(uids)}
            for coldkey, uids in sorted(shared.items(), key=lambda item: (-len(item[1]), item[0]))
        ],
    }


@dataclass(frozen=True)
class Selection:
    """The result of one walk of the queue."""

    entries: tuple[Entry, ...]  # admitted seats, in turn order
    skipped: tuple[RegKey, ...]  # seats passed over with no valid file: not charged, place kept


def select_round(
    queue: Queue,
    verdicts: Mapping[str, Verdict],
    freeze: Callable[[Registration], FrozenBatch | None],
    cap: int,
) -> Selection:
    """Walk the queue and admit up to ``cap`` seats that have a valid frozen file.

    A seat with no valid file is skipped. A held seat is passed over. A miner
    gets one entry per round: a second seat of the same hotkey or coldkey
    stays queued for a later round.
    """
    entries: list[Entry] = []
    skipped: list[RegKey] = []
    miners: set[str] = set()
    for seat in waiting(queue):
        if len(entries) >= cap:
            break
        reg = seat.reg
        if seat.held:
            continue
        owners = {f"hotkey:{reg.hotkey}"} | ({f"coldkey:{reg.coldkey}"} if reg.coldkey else set())
        if owners & miners:
            continue
        verdict = verdicts.get(reg.hotkey)
        batch = freeze(reg) if verdict is not None and verdict.valid else None
        if batch is None:
            skipped.append(reg.key)
            continue
        entries.append(Entry(queue_pos=len(entries), reg=reg, batch=batch))
        miners |= owners
    return Selection(tuple(entries), tuple(skipped))


def split_among_boxes(entries: tuple[Entry, ...], boxes: int) -> list[tuple[Entry, ...]]:
    """Give each box a block of consecutive seats. Earlier boxes get the earlier seats."""
    boxes = max(1, min(boxes, len(entries)))
    size, extra = divmod(len(entries), boxes)
    blocks, start = [], 0
    for index in range(boxes):
        stop = start + size + (1 if index < extra else 0)
        blocks.append(entries[start:stop])
        start = stop
    return blocks


def squat_trim(
    queue: Queue,
    verdicts: Mapping[str, Verdict],
    now: int,
    turns_last_day: int,
    min_wait_s: int,
    age_factor: float,
) -> list[RegKey]:
    """Seats to move to the back: no valid file, and older than ``age_factor`` times the expected wait.

    Expected wait = max(``min_wait_s``, seats ahead / turns per second), with
    the turn rate taken from the last 24 hours. A seat with a valid file is
    never moved. With no turns in the last day the wait is unknown, so no
    seat is moved.
    """
    if turns_last_day <= 0:
        return []
    seconds_per_turn = SECONDS_PER_DAY / turns_last_day
    moved: list[RegKey] = []
    ahead = 0
    for seat in waiting(queue):
        if zone(seat) != NORMAL:
            continue
        expected_wait = max(float(min_wait_s), ahead * seconds_per_turn)
        ahead += 1
        verdict = verdicts.get(seat.reg.hotkey)
        has_file = verdict is not None and verdict.valid
        age = now - seat.reg.registered_at
        if not has_file and seat.reg.registered_at > 0 and age > age_factor * expected_wait:
            moved.append(seat.reg.key)
    return moved


def restore_from_back(queue: Queue, verdicts: Mapping[str, Verdict]) -> list[RegKey]:
    """Seats that leave the back zone: demoted, waiting, and a valid file at selection time.

    Matches v1's recheck each turn: the back flag is not sticky once a valid
    file is present. Order is turn order so the journal list is deterministic.
    A front seat that is also demoted is included: clearing demoted leaves it
    at the front.
    """
    restored: list[RegKey] = []
    for seat in waiting(queue):
        if not seat.demoted:
            continue
        verdict = verdicts.get(seat.reg.hotkey)
        if verdict is not None and verdict.valid:
            restored.append(seat.reg.key)
    return restored


@dataclass(frozen=True)
class FrontMove:
    """One seat that the operator sends to the front: the seat as it is, and its place in the turn order before and after."""

    seat: QueueEntry
    position_before: int
    position_after: int

    @property
    def changed(self) -> bool:
        return not self.seat.front or self.seat.demoted


def plan_front(queue: Queue, served_hotkeys: Iterable[str], hotkeys: Iterable[str]) -> tuple[list[FrontMove], list[QueueEntry]]:
    """The seats of ``hotkeys`` at the front and not demoted: one move for each hotkey, and the turn order after.

    All or nothing: one hotkey with no waiting seat refuses the whole call.
    Inside the front zone the order stays registration block, then UID. No
    other flag of a seat changes, and no other seat changes its flags.
    """
    names = list(dict.fromkeys(hotkeys))
    served, problems, picked = set(served_hotkeys), [], []
    for hotkey in names:
        seats = [seat for seat in queue.values() if seat.reg.hotkey == hotkey]
        if len(seats) == 1 and not seats[0].leased:
            picked.append(seats[0])
        elif len(seats) == 1:
            problems.append(f"{hotkey}: its seat (UID {seats[0].reg.uid}) is in an open round")
        elif seats:
            problems.append(f"{hotkey}: {len(seats)} seats have this hotkey (UIDs {', '.join(str(seat.reg.uid) for seat in seats)})")
        else:
            problems.append(f"{hotkey}: " + ("it had its turn and was charged, so it has no seat" if hotkey in served else "no seat in the queue has this hotkey"))
    if problems:
        raise V2Error("nothing was changed: " + "; ".join(problems))
    before = {seat.reg.key: place for place, seat in enumerate(waiting(queue))}
    moved = {seat.reg.key: replace(seat, front=True, demoted=False) for seat in picked}
    order = waiting({**queue, **moved})
    after = {seat.reg.key: place for place, seat in enumerate(order)}
    return [FrontMove(seat, before[seat.reg.key], after[seat.reg.key]) for seat in picked], order


@dataclass(frozen=True)
class SyncPlan:
    """Changes that bring the queue in line with one chain snapshot."""

    added: tuple[Registration, ...]
    removed: tuple[RegKey, ...]
    rehotkeyed: tuple[Registration, ...]
    refreshed: tuple[Registration, ...] = ()  # same hotkey; owner coldkey (or other snapshot fields) caught up

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.rehotkeyed or self.refreshed)


def plan_sync(
    queue: Queue,
    served_regs: Iterable[RegKey],
    served_hotkeys: Iterable[str],
    snapshot: ChainSnapshot,
    netuid: int,
    burn_uid: int,
    max_remove_frac: float = 0.25,
    force: bool = False,
) -> SyncPlan:
    """Compare the queue with the chain.

    A registration enters the queue once. A seat that already had its turn,
    or a hotkey that already had its turn, never enters again. A seat whose
    registration left the chain is removed, unless it is in an open round.

    The owner coldkey of each seat is the ``SubtensorModule.Owner`` value of
    its hotkey at the snapshot block. A seat whose hotkey stayed but whose
    coldkey in the queue is empty or stale is refreshed so the one-entry-per-
    coldkey rule of ``select_round`` can apply.

    A bad chain read must not empty the paid queue. So an empty snapshot, a
    snapshot of another subnet, or a plan that removes more than
    ``max_remove_frac`` of the queue raises ``InfraError`` unless ``force``.
    """
    if snapshot.netuid != netuid:
        raise InfraError(f"chain snapshot is for netuid {snapshot.netuid}, expected {netuid}")
    if not snapshot.registrations:
        raise InfraError("chain snapshot has no registrations; refusing to sync")
    used_regs, used_hotkeys = set(served_regs), set(served_hotkeys)
    on_chain: dict[RegKey, Registration] = {}
    for reg in snapshot.registrations:
        if reg.uid != burn_uid and reg.key not in used_regs and reg.hotkey not in used_hotkeys:
            on_chain[reg.key] = reg
    added = [reg for key, reg in on_chain.items() if key not in queue]
    rehotkeyed = [
        reg for key, reg in on_chain.items() if key in queue and not queue[key].leased and queue[key].reg.hotkey != reg.hotkey
    ]
    refreshed = [
        reg
        for key, reg in on_chain.items()
        if key in queue
        and not queue[key].leased
        and queue[key].reg.hotkey == reg.hotkey
        and queue[key].reg.coldkey != reg.coldkey
    ]
    removed = [key for key, seat in queue.items() if key not in on_chain and not seat.leased]
    if not force and len(queue) >= 8 and len(removed) > max_remove_frac * len(queue):
        raise InfraError(
            f"sync would remove {len(removed)} of {len(queue)} seats; refusing without force (possible bad chain read). "
            "If the chain is right, an operator applies it with the force-sync command"
        )
    by_turn = lambda reg: (reg.registration_block, reg.uid)
    return SyncPlan(
        tuple(sorted(added, key=by_turn)),
        tuple(sorted(removed)),
        tuple(sorted(rehotkeyed, key=by_turn)),
        tuple(sorted(refreshed, key=by_turn)),
    )
