"""The miner queue. Pure functions: no I/O and no clock.

A seat is one paid registration. Turn order has three zones:

  front   seats sent back, not charged: no_decision (our failure), not_trained or not_scored (the round's time ended)
  normal  everyone else
  back    seats moved back because they had no valid file for too long

Inside each zone the order is registration block, then UID. So seats that
return to the front keep their original order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .errors import InfraError
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


@dataclass(frozen=True)
class SyncPlan:
    """Changes that bring the queue in line with one chain snapshot."""

    added: tuple[Registration, ...]
    removed: tuple[RegKey, ...]
    rehotkeyed: tuple[Registration, ...]

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.rehotkeyed)


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
    removed = [key for key, seat in queue.items() if key not in on_chain and not seat.leased]
    if not force and len(queue) >= 8 and len(removed) > max_remove_frac * len(queue):
        raise InfraError(
            f"sync would remove {len(removed)} of {len(queue)} seats; refusing without force (possible bad chain read). "
            "If the chain is right, an operator applies it with the force-sync command"
        )
    by_turn = lambda reg: (reg.registration_block, reg.uid)
    return SyncPlan(tuple(sorted(added, key=by_turn)), tuple(sorted(removed)), tuple(sorted(rehotkeyed, key=by_turn)))
