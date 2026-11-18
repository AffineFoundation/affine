"""The miner queue. Pure functions: no I/O and no clock.

A seat is one paid registration. Turn order has three zones:

  front   seats sent back, not charged: no_decision (our failure), not_trained or not_scored (the round's time ended)
  normal  everyone else
  back    seats moved back because they had no valid file for too long

Inside each zone the order is registration block, then UID. So seats that
return to the front keep their original order.
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
    coldkeyed: tuple[Registration, ...] = ()  # seats that get the coldkey of the chain: a seat saved before the coldkey was read has ""

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.rehotkeyed or self.coldkeyed)


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

    A waiting seat gets the coldkey that the chain has for it. That changes
    the coldkey only: the seat keeps its place and its flags. A seat in an
    open round is not touched.

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
    coldkeyed = [
        replace(queue[key].reg, coldkey=reg.coldkey)
        for key, reg in on_chain.items()
        if key in queue and not queue[key].leased and queue[key].reg.hotkey == reg.hotkey and reg.coldkey not in ("", queue[key].reg.coldkey)
    ]
    removed = [key for key, seat in queue.items() if key not in on_chain and not seat.leased]
    if not force and len(queue) >= 8 and len(removed) > max_remove_frac * len(queue):
        raise InfraError(
            f"sync would remove {len(removed)} of {len(queue)} seats; refusing without force (possible bad chain read). "
            "If the chain is right, an operator applies it with the force-sync command"
        )
    by_turn = lambda regs: tuple(sorted(regs, key=lambda reg: (reg.registration_block, reg.uid)))
    return SyncPlan(by_turn(added), tuple(sorted(removed)), by_turn(rehotkeyed), by_turn(coldkeyed))
