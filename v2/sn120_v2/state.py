"""The controller's state, rebuilt from the journal.

``replay`` folds journal records into one ``State``. Nothing else changes a
``State``. So a restarted controller, the weights writer, the website and
the benchmark service all see the same head, queue and kings.

Every record kind has one handler below. A record that does not fit the
state (for example a second open round) raises ``JournalCorrupt``: the
controller refuses to run on a journal it does not understand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from .canonical import Record
from .errors import JournalCorrupt, V2Error
from .journal import LEADER, JournalRecord, read_journal
from .params import GenesisSpec
from .types import Entry, HeadManifest, HeadVersion, King, QueueEntry, Registration, RoundDecision, V1Import

RegKey = tuple[int, int]
HISTORY = 200  # closed rounds kept in memory for the public snapshot
DAY = 86400

# Journal record kinds.
GENESIS = "genesis"
V1_IMPORTED = "v1_imported"
REGISTRATIONS_SYNCED = "registrations_synced"
SEATS_DEMOTED = "seats_demoted"
SEAT_RELEASED = "seat_released"
ROUND_OPENED = "round_opened"
LEASE_FAILED = "lease_failed"
ROUND_SETTLED = "round_settled"
APPLY_DEFERRED = "apply_deferred"
HEAD_APPLIED = "head_applied"
HEAD_FOLDED = "head_folded"
ROUND_CLOSED = "round_closed"

# How a round ended.
APPLIED = "applied"  # a winner was applied to the head
NULL = "null"  # scored, and M stays
NO_DECISION = "no_decision"  # our infrastructure failed: nobody is charged
ABANDONED = "abandoned"  # a winner was named but its files could not be verified: nobody is charged


@dataclass(frozen=True)
class Lease(Record):
    """One box's part of a round."""

    lease_id: str
    box_id: str
    issued_at: int
    expires_at: int
    ticket_digest: str
    entries: tuple[Entry, ...]
    status: str = "open"  # open, done (result accepted) or failed (expired or box failure)
    result_digest: str = ""


@dataclass(frozen=True)
class RoundRecord(Record):
    """A round as the journal knows it."""

    round_id: str
    head_version_id: str
    params_hash: str
    recipe_id: str
    round_seed: str
    snapshot_id: str
    n_tasks: int
    opened_at: int
    leases: tuple[Lease, ...]
    skipped: tuple[tuple[int, int], ...] = ()
    decision: RoundDecision | None = None
    applied: bool = False  # the winner is on the head
    apply_attempts: int = 0
    outcome: str = ""  # empty while the round is open
    closed_at: int = 0

    def lease(self, lease_id: str) -> Lease:
        return next(lease for lease in self.leases if lease.lease_id == lease_id)


@dataclass
class State:
    genesis: GenesisSpec | None = None
    heads: list[HeadVersion] = field(default_factory=list)  # every head, oldest first
    manifest: HeadManifest | None = None  # file maps of the current head; the store's pointer follows it
    queue: dict[RegKey, QueueEntry] = field(default_factory=dict)  # seats that have not had their turn
    served_regs: set[RegKey] = field(default_factory=set)
    served_hotkeys: set[str] = field(default_factory=set)
    kings: list[King] = field(default_factory=list)  # applied winners, oldest first
    round: RoundRecord | None = None  # the open round
    rounds: list[RoundRecord] = field(default_factory=list)  # closed rounds, oldest first
    round_seq: int = 0
    turn_times: list[int] = field(default_factory=list)  # when each charged entry was served, last day only
    snapshot_id: str = ""
    failed_leases: int = 0  # lease failures in a row
    last_failure_at: int = 0
    demote_on_arrival: set[RegKey] = field(default_factory=set)  # flags carried over from v1
    front_on_arrival: set[RegKey] = field(default_factory=set)
    leader: str = ""
    last_seq: int = -1

    @property
    def head(self) -> HeadVersion:
        if not self.heads:
            raise V2Error("the journal has no genesis yet")
        return self.heads[-1]

    def turns_last_day(self, now: int) -> int:
        return sum(1 for at in self.turn_times if at > now - DAY)


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise JournalCorrupt(message)


def _keys(pairs: Iterable[Iterable[int]]) -> list[RegKey]:
    return [(int(uid), int(block)) for uid, block in pairs]


def _edit(state: State, key: RegKey, **changes: Any) -> None:
    if key in state.queue:
        state.queue[key] = replace(state.queue[key], **changes)


def _new_head(state: State, data: dict) -> HeadVersion:
    state.manifest = HeadManifest.from_dict(data["manifest"])
    state.heads.append(state.manifest.head)
    return state.manifest.head


def _genesis(state: State, data: dict, at: int) -> None:
    _need(not state.heads, "a second genesis record")
    state.genesis = GenesisSpec.from_dict(data["genesis"])
    head = _new_head(state, data)
    _need(head.parent_version_id is None and head.base_hash == state.genesis.base_hash, "genesis head does not match its spec")


def _v1_imported(state: State, data: dict, at: int) -> None:
    v1 = V1Import.from_dict(data["v1"])
    _need(not state.kings and not state.rounds and state.round is None, "v1 state can only be imported before the first round")
    state.kings.extend(v1.kings)
    state.served_regs.update(_keys(v1.served_regs))
    state.served_hotkeys.update(v1.served_hotkeys)
    state.demote_on_arrival.update(_keys(v1.demoted))
    state.front_on_arrival.update(_keys(v1.front))


def _registrations_synced(state: State, data: dict, at: int) -> None:
    for key in _keys(data["removed"]):
        state.queue.pop(key, None)
    for raw in data["rehotkeyed"]:
        reg = Registration.from_dict(raw)
        _edit(state, reg.key, reg=reg)
    for raw in data["added"]:
        reg = Registration.from_dict(raw)
        _need(reg.key not in state.served_regs and reg.key not in state.queue, f"registration {reg.key} is already known")
        state.queue[reg.key] = QueueEntry(reg, front=reg.key in state.front_on_arrival, demoted=reg.key in state.demote_on_arrival)


def _seats_demoted(state: State, data: dict, at: int) -> None:
    for key in _keys(data["keys"]):
        _edit(state, key, demoted=True)


def _seat_released(state: State, data: dict, at: int) -> None:
    (key,) = _keys([data["key"]])
    _edit(state, key, held=False, infra_attempts=0)


def _round_opened(state: State, data: dict, at: int) -> None:
    record = RoundRecord.from_dict(data["round"])
    _need(state.round is None, "a round opened while another round is open")
    _need(record.head_version_id == state.head.version_id, "a round opened against a head that is not current")
    for lease in record.leases:
        for entry in lease.entries:
            _need(entry.reg.key in state.queue and not state.queue[entry.reg.key].leased, "a round took a seat that is not waiting")
            _edit(state, entry.reg.key, leased=True)
    state.round, state.round_seq = record, state.round_seq + 1


def _open_round(state: State, data: dict) -> RoundRecord:
    _need(state.round is not None and state.round.round_id == data["round_id"], "a record names a round that is not open")
    return state.round


def _lease_failed(state: State, data: dict, at: int) -> None:
    record = _open_round(state, data)
    lease = record.lease(data["lease_id"])
    _need(lease.status == "open", "a lease failed twice or after its result")
    for entry in lease.entries:
        _edit(state, entry.reg.key, leased=False, front=True)
    leases = tuple(replace(l, status="failed") if l.lease_id == lease.lease_id else l for l in record.leases)
    state.round = replace(record, leases=leases)
    state.failed_leases, state.last_failure_at = state.failed_leases + 1, at


def _round_settled(state: State, data: dict, at: int) -> None:
    record = _open_round(state, data)
    _need(record.decision is None, "a round settled twice")
    digests = data["results"]
    leases = tuple(
        replace(l, status="done", result_digest=digests[l.lease_id]) if l.lease_id in digests else l for l in record.leases
    )
    _need(all(l.status != "open" for l in leases), "a round settled while a lease is still open")
    state.round = replace(record, leases=leases, decision=RoundDecision.from_dict(data["decision"]))
    state.failed_leases = 0


def _apply_deferred(state: State, data: dict, at: int) -> None:
    record = _open_round(state, data)
    state.round = replace(record, apply_attempts=record.apply_attempts + 1)
    state.last_failure_at = at


def _head_applied(state: State, data: dict, at: int) -> None:
    record, parent = _open_round(state, data), state.head
    winner = record.decision.winner if record.decision else None
    _need(winner is not None and not record.applied, "a head was applied without a settled winner, or twice")
    head, king = _new_head(state, data), King.from_dict(data["king"])
    _need(head.parent_version_id == parent.version_id and not head.folded, "an applied head does not follow the current head")
    _need((king.uid, king.hotkey, king.head_version_id) == (winner.uid, winner.hotkey, head.version_id), "king does not match the winner")
    state.kings.append(king)
    state.round = replace(record, applied=True)


def _head_folded(state: State, data: dict, at: int) -> None:
    parent = state.head
    head = _new_head(state, data)
    _need(head.parent_version_id == parent.version_id and head.folded, "a folded head does not follow the current head")


def _round_closed(state: State, data: dict, at: int) -> None:
    record = _open_round(state, data)
    outcome = data["outcome"]
    _need(outcome in (APPLIED, NULL, NO_DECISION, ABANDONED), f"unknown round outcome {outcome!r}")
    _need((outcome == APPLIED) == record.applied, "a round can close as applied exactly when its winner is on the head")
    for uid, block, hotkey in data["served"]:
        state.queue.pop((uid, block), None)
        state.served_regs.add((uid, block))
        state.served_hotkeys.add(hotkey)
        state.turn_times.append(at)
    for key in _keys(data["front"]):
        _edit(state, key, leased=False, front=True)
    for key in _keys(data["attempted"]):
        _edit(state, key, infra_attempts=state.queue[key].infra_attempts + 1)
    for key in _keys(data["held"]):
        _edit(state, key, held=True)
    for key in _keys(data["released"]):
        _edit(state, key, leased=False)
    _need(not any(state.queue[e.reg.key].leased for l in record.leases for e in l.entries if e.reg.key in state.queue), "a closed round left a seat leased")
    state.rounds = (state.rounds + [replace(record, outcome=outcome, closed_at=at)])[-HISTORY:]
    state.round = None
    state.snapshot_id = data.get("snapshot_id", state.snapshot_id)
    state.turn_times = [t for t in state.turn_times if t > at - 2 * DAY]


HANDLERS: dict[str, Callable[[State, dict, int], None]] = {
    GENESIS: _genesis,
    V1_IMPORTED: _v1_imported,
    REGISTRATIONS_SYNCED: _registrations_synced,
    SEATS_DEMOTED: _seats_demoted,
    SEAT_RELEASED: _seat_released,
    ROUND_OPENED: _round_opened,
    LEASE_FAILED: _lease_failed,
    ROUND_SETTLED: _round_settled,
    APPLY_DEFERRED: _apply_deferred,
    HEAD_APPLIED: _head_applied,
    HEAD_FOLDED: _head_folded,
    ROUND_CLOSED: _round_closed,
}


def apply(state: State, record: JournalRecord) -> None:
    """Fold one journal record into the state."""
    _need(record.seq == state.last_seq + 1, f"journal record {record.seq} is out of order")
    if record.kind == LEADER:
        state.leader = record.data.get("owner", "")
    else:
        handler = HANDLERS.get(record.kind)
        _need(handler is not None, f"unknown journal record kind {record.kind!r}")
        _need(record.kind == GENESIS or bool(state.heads), "the journal must start with a genesis record")
        try:
            handler(state, record.data, record.at)
        except (KeyError, StopIteration, TypeError) as exc:
            raise JournalCorrupt(f"journal record {record.seq} ({record.kind}) is malformed: {exc!r}") from exc
    state.last_seq = record.seq


def replay(records: Iterable[JournalRecord]) -> State:
    state = State()
    for record in records:
        apply(state, record)
    return state


def load(path: str | Path) -> State:
    """Read-only: the state of a journal file. Safe while a controller is writing to it."""
    return replay(read_journal(path))
