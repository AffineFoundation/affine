"""Pure state machine of the batch controller.

This module has no I/O, no clock and no randomness. Three functions hold all
the logic:

``decide(cfg, state, event)``  -> one journal record, or None.
``resolve(cfg, state)``        -> one follow-up record, or None.
``evolve(state, kind, body)``  -> applies one record to the state.

The runtime calls ``decide``, appends the record to the journal (fsync), then
calls ``evolve``. After each record it calls ``resolve`` until it returns
None. On restart the state is rebuilt by ``evolve`` over the journal, then
``resolve`` runs to a fixed point before anything else. ``plan`` lists the
side effects that must be in progress for a state; every one is idempotent.

Each record is one complete transition. No record needs a second record to
make the state consistent, so a crash between two records is always safe.

Batch state machine
-------------------
::

    QUEUED -> RENTING -> RUNNING -> SETTLING -> RED                 (final)
       ^         |          |          |------> GREEN -> APPLYING -> APPLIED
       |         |          |          |          '----> CREDITED   (final)
       '---------+----------+----------'  attempt failed: back to QUEUED,
                                          or LOST_INFRA after N failures

HEAD epoch state machine
------------------------
::

    OPEN(n) --apply_intent--> APPLYING(n+1) --applied--> OPEN(n+1)

``APPLYING`` is never aborted. It ends only when the pointer in R2 holds the
new link. Any other pointer value halts the controller (fail-stop).

A halt blocks new leases, new credits and new apply intents. Leases that run
already go on to a decision, so no GREEN is lost while the controller waits
for an operator.

Events may arrive late, twice, or out of order. ``decide`` checks every event
against the fence, the phase and the rent try of the current state, and drops
what does not fit. The model checker explores exactly this function.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Set, Tuple

from .layout import GENESIS_FENCE, link_id as compute_link_id


class Phase(str, enum.Enum):
    QUEUED = 'queued'
    RENTING = 'renting'
    RUNNING = 'running'
    SETTLING = 'settling'
    GREEN = 'green'
    APPLYING = 'applying'
    APPLIED = 'applied'
    CREDITED = 'credited'
    RED = 'red'
    LOST_INFRA = 'lost_infra'


TERMINAL = frozenset({Phase.APPLIED, Phase.CREDITED, Phase.RED, Phase.LOST_INFRA})
HOLDS_SLOT = frozenset({Phase.RENTING, Phase.RUNNING})
# Outcomes that put the miner on the payout list, with the kind of credit.
# The payout layer sets the rate of each kind; this module only names it.
CREDIT_GREEN = 'green'
CREDIT_LOST_INFRA = 'lost_infra'
CREDIT: Mapping[Phase, str] = {
    Phase.APPLIED: CREDIT_GREEN,
    Phase.CREDITED: CREDIT_GREEN,
    Phase.LOST_INFRA: CREDIT_LOST_INFRA,
}

VERDICT_GREEN = 'green'
VERDICT_RED = 'red'

# Records that carry no state change for the core.
PASSIVE_KINDS = frozenset({'init', 'controller_started', 'lease_renewed', 'note'})

Q, RT, RN, ST = Phase.QUEUED, Phase.RENTING, Phase.RUNNING, Phase.SETTLING
TRANSITIONS: Mapping[str, frozenset] = {
    'lease_issued': frozenset({(Q, RT)}),
    'rent_try': frozenset({(RT, RT)}),
    'box_bound': frozenset({(RT, RN)}),
    'job_launched': frozenset({(RN, RN)}),
    'upload_observed': frozenset({(RN, ST)}),
    'attempt_failed': frozenset({
        (RN, Q), (ST, Q), (RN, Phase.LOST_INFRA), (ST, Phase.LOST_INFRA),
    }),
    'decided': frozenset({(ST, Phase.GREEN), (ST, Phase.RED)}),
    'apply_intent': frozenset({(Phase.GREEN, Phase.APPLYING)}),
    'credited': frozenset({(Phase.GREEN, Phase.CREDITED)}),
    'applied': frozenset({(Phase.APPLYING, Phase.APPLIED)}),
}


class CoreError(RuntimeError):
    """A record does not fit the state. The journal or the code is wrong."""


@dataclass(frozen=True)
class Rung:
    """One step of the rental ladder: where to rent and what."""

    provider: str
    gpu: str
    tier: str


@dataclass(frozen=True)
class CoreConfig:
    max_boxes: int
    max_infra_failures: int
    ladder: Tuple[Rung, ...]

    def __post_init__(self) -> None:
        if self.max_boxes < 1 or self.max_infra_failures < 1 or not self.ladder:
            raise ValueError('max_boxes, max_infra_failures and ladder must be set')


class Head(NamedTuple):
    seq: int
    state_id: str
    link_id: str


class Pending(NamedTuple):
    """An apply intent: the next HEAD, chosen but not yet confirmed in R2."""

    batch_id: Optional[str]
    seq: int
    state_id: str
    link_id: str
    parent_link: Optional[str]
    result_sha: Optional[str]
    fence: int
    evidence: Tuple


class Decision(NamedTuple):
    verdict: str
    fence: int
    state_id: Optional[str]
    result_sha: str
    base_state: str
    base_seq: int
    order: Optional[int]
    evidence: Tuple


class Lease:
    """One attempt of one batch. ``fence`` is unique for the life of the run."""

    __slots__ = ('fence', 'try_no', 'rung', 'box_id', 'provider', 'base_state', 'base_seq', 'launched')

    def __init__(self, fence: int, try_no: int = 0, rung: int = 0):
        self.fence = fence
        self.try_no = try_no
        self.rung = rung
        self.box_id: Optional[str] = None
        self.provider: Optional[str] = None
        self.base_state: Optional[str] = None
        self.base_seq: Optional[int] = None
        self.launched = False

    def copy(self) -> 'Lease':
        other = Lease(self.fence, self.try_no, self.rung)
        other.box_id = self.box_id
        other.provider = self.provider
        other.base_state = self.base_state
        other.base_seq = self.base_seq
        other.launched = self.launched
        return other

    def key(self) -> Tuple:
        return (
            self.fence, self.try_no, self.rung, self.box_id, self.provider,
            self.base_state, self.base_seq, self.launched,
        )


class Batch:
    __slots__ = (
        'batch_id', 'seq', 'miner', 'ref', 'phase', 'infra_failures', 'attempts',
        'lease', 'decision', 'note', 'head_seq',
    )

    def __init__(self, batch_id: str, seq: int, miner: str, ref: str):
        self.batch_id = batch_id
        self.seq = seq
        self.miner = miner
        self.ref = ref
        self.phase = Phase.QUEUED
        self.infra_failures = 0
        self.attempts = 0
        self.lease: Optional[Lease] = None
        self.decision: Optional[Decision] = None
        self.note: Optional[str] = None
        self.head_seq: Optional[int] = None

    def copy(self) -> 'Batch':
        other = Batch(self.batch_id, self.seq, self.miner, self.ref)
        other.phase = self.phase
        other.infra_failures = self.infra_failures
        other.attempts = self.attempts
        other.lease = self.lease.copy() if self.lease is not None else None
        other.decision = self.decision
        other.note = self.note
        other.head_seq = self.head_seq
        return other

    def key(self) -> Tuple:
        return (
            self.batch_id, self.seq, self.miner, self.ref, self.phase.value,
            self.infra_failures, self.attempts,
            self.lease.key() if self.lease is not None else None,
            self.decision, self.note, self.head_seq,
        )


class State:
    """Everything the controller knows. Rebuilt from the journal on restart."""

    def __init__(self) -> None:
        self.head: Optional[Head] = None
        self.pending: Optional[Pending] = None
        self.halted: Optional[str] = None
        self.active: Dict[str, Batch] = {}
        self.done: Dict[str, Batch] = {}
        self.by_fence: Dict[int, str] = {}
        self.next_seq = 0
        self.next_fence = 1
        self.next_order = 0
        self.doomed: Dict[str, str] = {}
        self.dirty: Set[Tuple[str, int]] = set()
        self.unpublished: Set[str] = set()
        self.rejected: Set[str] = set()

    def clone(self) -> 'State':
        other = State()
        other.head = self.head
        other.pending = self.pending
        other.halted = self.halted
        other.active = {k: v.copy() for k, v in self.active.items()}
        # Terminal batches never change, so the objects are shared.
        other.done = dict(self.done)
        other.by_fence = dict(self.by_fence)
        other.next_seq = self.next_seq
        other.next_fence = self.next_fence
        other.next_order = self.next_order
        other.doomed = dict(self.doomed)
        other.dirty = set(self.dirty)
        other.unpublished = set(self.unpublished)
        other.rejected = set(self.rejected)
        return other

    def key(self) -> Tuple:
        """A hashable, canonical form. Equal keys mean equal states."""
        return (
            self.head, self.pending, self.halted,
            tuple(b.key() for b in sorted(self.active.values(), key=lambda b: b.seq)),
            tuple(b.key() for b in sorted(self.done.values(), key=lambda b: b.seq)),
            self.next_seq, self.next_fence, self.next_order,
            tuple(sorted(self.doomed.items())),
            tuple(sorted(self.dirty)),
            tuple(sorted(self.unpublished)),
            tuple(sorted(self.rejected)),
        )

    def batch(self, batch_id: str) -> Optional[Batch]:
        return self.active.get(batch_id) or self.done.get(batch_id)

    def leased(self, fence: int) -> Optional[Batch]:
        batch_id = self.by_fence.get(fence)
        return self.active.get(batch_id) if batch_id is not None else None

    def slots_in_use(self) -> int:
        return sum(1 for bid in self.by_fence.values() if self.active[bid].phase in HOLDS_SLOT)

    def live_box_ids(self) -> Set[str]:
        """Boxes that a running lease still needs. Never destroy these."""
        out = set()
        for bid in self.by_fence.values():
            batch = self.active[bid]
            if batch.phase is Phase.RUNNING and batch.lease.box_id is not None:
                out.add(batch.lease.box_id)
        return out

    def base(self) -> Optional[Tuple[str, int]]:
        """The state a new attempt trains on: the pending HEAD if one exists.

        A pending apply is never aborted and its state is already verified in
        R2, so it is the next HEAD for certain.
        """
        if self.pending is not None:
            return self.pending.state_id, self.pending.seq
        if self.head is not None:
            return self.head.state_id, self.head.seq
        return None


# ----------------------------------------------------------------------------
# Events: what the outside world reports. All are hashable.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Arrived:
    batch_id: str
    miner: str
    ref: str = ''


@dataclass(frozen=True)
class Rejected:
    digest: str
    reason: str


@dataclass(frozen=True)
class Tick:
    """Ask the scheduler for one new lease. ``hold`` blocks new leases."""

    hold: bool = False


@dataclass(frozen=True)
class RentOk:
    fence: int
    try_no: int
    box_id: str
    provider: str
    gpu: str = ''
    tier: str = ''
    price: float = 0.0
    name: str = ''


@dataclass(frozen=True)
class RentFailed:
    """The rent try ended without a box: no offer, an error, or a timeout."""

    fence: int
    try_no: int
    reason: str


@dataclass(frozen=True)
class Launched:
    fence: int


@dataclass(frozen=True)
class Polled:
    """One look at a running lease.

    ``complete``: ``COMPLETE.json`` exists in the staging prefix of the fence.
    ``gone``: the provider says the box does not exist any more.
    ``expired``: the lease deadline passed and the upload was not complete at
    a time after that deadline.
    """

    fence: int
    complete: bool = False
    gone: bool = False
    expired: bool = False


@dataclass(frozen=True)
class Renewal:
    fence: int
    hb_seq: int
    until: float


@dataclass(frozen=True)
class Settled:
    """The result of this fence was verified in R2 and judged.

    ``state_id`` is set only when the full state was also verified in R2. A
    GREEN result without a verified state is still a GREEN: it is credited,
    but it can never be applied.
    """

    fence: int
    verdict: str
    result_sha: str
    state_id: Optional[str] = None
    evidence: Tuple = ()
    note: str = ''


@dataclass(frozen=True)
class SettleInvalid:
    """The package of this fence is definitely bad (missing or wrong bytes)."""

    fence: int
    reason: str


@dataclass(frozen=True)
class ApplyDone:
    seq: int
    link_id: str


@dataclass(frozen=True)
class ApplyDiverged:
    seq: int
    link_id: str
    reason: str


@dataclass(frozen=True)
class BoxGone:
    """The provider does not list the box any more, or it is not ours."""

    box_id: str
    how: str = 'absent'


@dataclass(frozen=True)
class OrphanSeen:
    """A box with our run prefix that no lease needs.

    ``fence`` and ``try_no`` come from the box name, or are None when the
    name does not parse.
    """

    box_id: str
    provider: str
    fence: Optional[int] = None
    try_no: Optional[int] = None


@dataclass(frozen=True)
class StagingCleared:
    batch_id: str
    fence: int


@dataclass(frozen=True)
class Published:
    batch_id: str


@dataclass(frozen=True)
class GenesisVerified:
    state_id: str
    evidence: Tuple = ()


@dataclass(frozen=True)
class Fault:
    """An effect found durable data that must never be wrong, and it is wrong
    (for example the manifest of the HEAD state does not match its id).

    The controller stops new work (fail-stop) until an operator resumes it.
    """

    reason: str


@dataclass(frozen=True)
class Resume:
    note: str


# ----------------------------------------------------------------------------
# Effects: what must be in progress. All are hashable and idempotent.
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class RentOp:
    fence: int
    try_no: int
    rung: Rung
    batch_id: str


@dataclass(frozen=True)
class LaunchOp:
    fence: int
    batch_id: str
    attempt: int
    box_id: str
    provider: str
    base_state: str
    base_seq: int
    ref: str


@dataclass(frozen=True)
class PollOp:
    fence: int
    batch_id: str
    box_id: str
    provider: str


@dataclass(frozen=True)
class SettleOp:
    fence: int
    batch_id: str
    base_state: str


@dataclass(frozen=True)
class ApplyOp:
    batch_id: Optional[str]
    seq: int
    state_id: str
    link_id: str
    parent_link: Optional[str]
    result_sha: Optional[str]
    fence: int
    evidence: Tuple


@dataclass(frozen=True)
class DestroyOp:
    box_id: str
    provider: str


@dataclass(frozen=True)
class GcOp:
    batch_id: str
    fence: int


@dataclass(frozen=True)
class PublishOp:
    batch_id: str


Record = Tuple[str, Dict[str, Any]]


# ----------------------------------------------------------------------------
# decide
# ----------------------------------------------------------------------------

def _fail(cfg: CoreConfig, batch: Batch, reason: str) -> Record:
    failures = batch.infra_failures + 1
    return 'attempt_failed', {
        'batch_id': batch.batch_id,
        'fence': batch.lease.fence,
        'reason': reason,
        'infra_failures': failures,
        'terminal': failures >= cfg.max_infra_failures,
    }


def _box_known(state: State, box_id: str) -> bool:
    return box_id in state.doomed or box_id in state.live_box_ids()


def _first_queued(state: State) -> Optional[Batch]:
    best = None
    for batch in state.active.values():
        if batch.phase is Phase.QUEUED and (best is None or batch.seq < best.seq):
            best = batch
    return best


def decide(cfg: CoreConfig, state: State, ev: Any) -> Optional[Record]:
    """Turn one event into at most one record. Never changes ``state``."""
    kind = type(ev)

    if kind is Arrived:
        if state.batch(ev.batch_id) is not None:
            return None
        return 'batch_admitted', {
            'batch_id': ev.batch_id, 'seq': state.next_seq, 'miner': ev.miner, 'ref': ev.ref,
        }

    if kind is Rejected:
        if ev.digest in state.rejected:
            return None
        return 'batch_rejected', {'digest': ev.digest, 'reason': ev.reason}

    if kind is Tick:
        if ev.hold or state.halted is not None or state.head is None:
            return None
        if state.slots_in_use() >= cfg.max_boxes:
            return None
        batch = _first_queued(state)
        if batch is None:
            return None
        return 'lease_issued', {
            'batch_id': batch.batch_id, 'fence': state.next_fence,
            'attempt': batch.attempts + 1, 'rung': 0,
        }

    if kind is RentOk:
        batch = state.leased(ev.fence)
        if (
            batch is not None and batch.phase is Phase.RENTING
            and batch.lease.try_no == ev.try_no and not _box_known(state, ev.box_id)
        ):
            base_state, base_seq = state.base()
            return 'box_bound', {
                'batch_id': batch.batch_id, 'fence': ev.fence, 'try': ev.try_no,
                'box_id': ev.box_id, 'provider': ev.provider, 'gpu': ev.gpu,
                'tier': ev.tier, 'price': ev.price, 'name': ev.name,
                'base_state': base_state, 'base_seq': base_seq,
            }
        if _box_known(state, ev.box_id):
            return None
        return 'box_doomed', {
            'box_id': ev.box_id, 'provider': ev.provider, 'reason': 'late_rent',
        }

    if kind is RentFailed:
        batch = state.leased(ev.fence)
        if batch is None or batch.phase is not Phase.RENTING or batch.lease.try_no != ev.try_no:
            return None
        nxt = batch.lease.rung + 1
        wrap = nxt >= len(cfg.ladder)
        return 'rent_try', {
            'batch_id': batch.batch_id, 'fence': ev.fence, 'try': ev.try_no + 1,
            'rung': 0 if wrap else nxt, 'backoff': wrap, 'why': ev.reason,
        }

    if kind is Launched:
        batch = state.leased(ev.fence)
        if batch is None or batch.phase is not Phase.RUNNING or batch.lease.launched:
            return None
        return 'job_launched', {'batch_id': batch.batch_id, 'fence': ev.fence}

    if kind is Polled:
        batch = state.leased(ev.fence)
        if batch is None or batch.phase is not Phase.RUNNING:
            return None
        # A complete upload always wins over a dead box or a passed deadline:
        # the work is in R2, so it is adopted, not thrown away.
        if ev.complete:
            return 'upload_observed', {'batch_id': batch.batch_id, 'fence': ev.fence}
        if ev.gone:
            return _fail(cfg, batch, 'box_gone')
        if ev.expired:
            return _fail(cfg, batch, 'lease_expired')
        return None

    if kind is Renewal:
        batch = state.leased(ev.fence)
        if batch is None or batch.phase is not Phase.RUNNING:
            return None
        return 'lease_renewed', {
            'batch_id': batch.batch_id, 'fence': ev.fence, 'hb_seq': ev.hb_seq, 'until': ev.until,
        }

    if kind is Settled:
        batch = state.leased(ev.fence)
        if batch is None or batch.phase is not Phase.SETTLING:
            return None
        green = ev.verdict == VERDICT_GREEN
        if not green and ev.verdict != VERDICT_RED:
            return None
        has_state = green and ev.state_id is not None
        return 'decided', {
            'batch_id': batch.batch_id, 'fence': ev.fence, 'verdict': ev.verdict,
            'state_id': ev.state_id if has_state else None, 'result_sha': ev.result_sha,
            'base_state': batch.lease.base_state, 'base_seq': batch.lease.base_seq,
            'order': state.next_order if green else None,
            'evidence': [list(item) for item in ev.evidence] if has_state else [],
            'note': ev.note,
        }

    if kind is SettleInvalid:
        batch = state.leased(ev.fence)
        if batch is None or batch.phase is not Phase.SETTLING:
            return None
        return _fail(cfg, batch, 'invalid_upload:%s' % ev.reason)

    if kind is ApplyDone:
        pending = state.pending
        if pending is None or pending.seq != ev.seq or pending.link_id != ev.link_id:
            return None
        return 'applied', {
            'seq': pending.seq, 'link_id': pending.link_id, 'state_id': pending.state_id,
            'batch_id': pending.batch_id,
        }

    if kind is ApplyDiverged:
        pending = state.pending
        if state.halted is not None or pending is None:
            return None
        if pending.seq != ev.seq or pending.link_id != ev.link_id:
            return None
        return 'halted', {'reason': ev.reason, 'seq': ev.seq}

    if kind is BoxGone:
        if ev.box_id not in state.doomed:
            return None
        return 'box_gone', {'box_id': ev.box_id, 'how': ev.how}

    if kind is OrphanSeen:
        if _box_known(state, ev.box_id):
            return None
        if ev.fence is not None:
            batch = state.leased(ev.fence)
            # The box of a rent try that is still open is not an orphan: the
            # rent effect adopts it by its exact name.
            if (
                batch is not None and batch.phase is Phase.RENTING
                and batch.lease.try_no == ev.try_no
            ):
                return None
        return 'box_doomed', {'box_id': ev.box_id, 'provider': ev.provider, 'reason': 'orphan'}

    if kind is StagingCleared:
        if (ev.batch_id, ev.fence) not in state.dirty:
            return None
        return 'staging_cleared', {'batch_id': ev.batch_id, 'fence': ev.fence}

    if kind is Published:
        if ev.batch_id not in state.unpublished:
            return None
        return 'outcome_published', {'batch_id': ev.batch_id}

    if kind is GenesisVerified:
        if state.head is not None or state.pending is not None:
            return None
        link = compute_link_id(
            seq=0, state_id=ev.state_id, parent=None, batch_id=None, result_sha=None,
            fence=GENESIS_FENCE,
        )
        return 'apply_intent', {
            'batch_id': None, 'seq': 0, 'state_id': ev.state_id, 'parent_link': None,
            'link_id': link, 'result_sha': None, 'fence': GENESIS_FENCE,
            'evidence': [list(item) for item in ev.evidence],
        }

    if kind is Fault:
        if state.halted is not None:
            return None
        return 'halted', {'reason': ev.reason, 'seq': None}

    if kind is Resume:
        if state.halted is None:
            return None
        return 'resumed', {'note': ev.note}

    raise CoreError('unknown event %r' % (ev,))


# ----------------------------------------------------------------------------
# resolve: follow-up records that depend only on the state
# ----------------------------------------------------------------------------

def resolve(cfg: CoreConfig, state: State) -> Optional[Record]:
    """Settle the GREEN batches against the HEAD.

    The rule, in order of the GREEN decisions in the journal:

    * no verified state                      -> credited (paid, not applied).
    * base is older than the HEAD            -> credited (paid, not applied).
    * base is the HEAD, no apply is pending  -> this batch wins: apply intent.
    * base is the HEAD, an apply is pending  -> credited (paid, not applied).
    * base is the pending HEAD               -> wait for that apply to finish.
    """
    if state.halted is not None or state.head is None:
        return None
    greens = [b for b in state.active.values() if b.phase is Phase.GREEN]
    greens.sort(key=lambda b: b.decision.order)
    for batch in greens:
        decision = batch.decision
        if decision.state_id is None:
            return 'credited', {'batch_id': batch.batch_id, 'reason': 'no_verified_state'}
        if decision.base_seq < state.head.seq:
            return 'credited', {'batch_id': batch.batch_id, 'reason': 'stale_base'}
        if decision.base_seq == state.head.seq:
            if state.pending is not None:
                return 'credited', {'batch_id': batch.batch_id, 'reason': 'not_first'}
            seq = state.head.seq + 1
            link = compute_link_id(
                seq=seq, state_id=decision.state_id, parent=state.head.link_id,
                batch_id=batch.batch_id, result_sha=decision.result_sha, fence=decision.fence,
            )
            return 'apply_intent', {
                'batch_id': batch.batch_id, 'seq': seq, 'state_id': decision.state_id,
                'parent_link': state.head.link_id, 'link_id': link,
                'result_sha': decision.result_sha, 'fence': decision.fence,
                'evidence': [list(item) for item in decision.evidence],
            }
    return None


# ----------------------------------------------------------------------------
# evolve
# ----------------------------------------------------------------------------

def _active(state: State, batch_id: str) -> Batch:
    batch = state.active.get(batch_id)
    if batch is None:
        raise CoreError('record for a batch that is not active: %r' % (batch_id,))
    return batch


def _leased(state: State, body: Mapping[str, Any]) -> Batch:
    batch = state.leased(body['fence'])
    if batch is None or batch.batch_id != body['batch_id']:
        raise CoreError('record for a fence that is not live: %r' % (body['fence'],))
    return batch


def _move(batch: Batch, kind: str, to: Phase) -> None:
    if (batch.phase, to) not in TRANSITIONS[kind]:
        raise CoreError('illegal transition %s: %s -> %s' % (kind, batch.phase.value, to.value))
    batch.phase = to


def _finish(state: State, batch: Batch) -> None:
    del state.active[batch.batch_id]
    state.done[batch.batch_id] = batch
    state.unpublished.add(batch.batch_id)


def _end_lease(state: State, batch: Batch) -> None:
    lease = batch.lease
    state.dirty.add((batch.batch_id, lease.fence))
    del state.by_fence[lease.fence]
    batch.lease = None


def _tuple2(items: Any) -> Tuple:
    return tuple(tuple(item) for item in items)


def _ev_batch_admitted(state: State, b: Mapping[str, Any]) -> None:
    if state.batch(b['batch_id']) is not None or b['seq'] != state.next_seq:
        raise CoreError('bad admission of %r' % (b['batch_id'],))
    state.active[b['batch_id']] = Batch(b['batch_id'], b['seq'], b['miner'], b.get('ref', ''))
    state.next_seq += 1


def _ev_batch_rejected(state: State, b: Mapping[str, Any]) -> None:
    state.rejected.add(b['digest'])


def _ev_lease_issued(state: State, b: Mapping[str, Any]) -> None:
    batch = _active(state, b['batch_id'])
    if b['fence'] != state.next_fence or b['attempt'] != batch.attempts + 1:
        raise CoreError('bad lease for %r' % (batch.batch_id,))
    _move(batch, 'lease_issued', Phase.RENTING)
    batch.attempts = b['attempt']
    batch.lease = Lease(b['fence'], 0, b['rung'])
    state.by_fence[b['fence']] = batch.batch_id
    state.next_fence += 1


def _ev_rent_try(state: State, b: Mapping[str, Any]) -> None:
    batch = _leased(state, b)
    _move(batch, 'rent_try', Phase.RENTING)
    if b['try'] != batch.lease.try_no + 1:
        raise CoreError('rent try out of order for fence %r' % (b['fence'],))
    batch.lease.try_no = b['try']
    batch.lease.rung = b['rung']


def _ev_box_bound(state: State, b: Mapping[str, Any]) -> None:
    batch = _leased(state, b)
    if b['try'] != batch.lease.try_no or _box_known(state, b['box_id']):
        raise CoreError('bad box binding for fence %r' % (b['fence'],))
    _move(batch, 'box_bound', Phase.RUNNING)
    lease = batch.lease
    lease.box_id = b['box_id']
    lease.provider = b['provider']
    lease.base_state = b['base_state']
    lease.base_seq = b['base_seq']


def _ev_job_launched(state: State, b: Mapping[str, Any]) -> None:
    batch = _leased(state, b)
    _move(batch, 'job_launched', Phase.RUNNING)
    batch.lease.launched = True


def _ev_upload_observed(state: State, b: Mapping[str, Any]) -> None:
    batch = _leased(state, b)
    _move(batch, 'upload_observed', Phase.SETTLING)
    # Everything the controller needs is in R2 now. The box is released.
    state.doomed[batch.lease.box_id] = batch.lease.provider


def _ev_attempt_failed(state: State, b: Mapping[str, Any]) -> None:
    batch = _leased(state, b)
    was_running = batch.phase is Phase.RUNNING
    _move(batch, 'attempt_failed', Phase.LOST_INFRA if b['terminal'] else Phase.QUEUED)
    if b['infra_failures'] != batch.infra_failures + 1:
        raise CoreError('infra failure count out of order for %r' % (batch.batch_id,))
    batch.infra_failures = b['infra_failures']
    if was_running:
        state.doomed[batch.lease.box_id] = batch.lease.provider
    _end_lease(state, batch)
    if b['terminal']:
        batch.note = b['reason']
        _finish(state, batch)


def _ev_decided(state: State, b: Mapping[str, Any]) -> None:
    batch = _leased(state, b)
    lease = batch.lease
    if b['base_state'] != lease.base_state or b['base_seq'] != lease.base_seq:
        raise CoreError('decision base differs from the lease base for %r' % (batch.batch_id,))
    green = b['verdict'] == VERDICT_GREEN
    if green:
        if b['order'] != state.next_order:
            raise CoreError('bad GREEN decision for %r' % (batch.batch_id,))
        state.next_order += 1
    elif b['verdict'] != VERDICT_RED or b['state_id'] is not None:
        raise CoreError('bad RED decision for %r' % (batch.batch_id,))
    _move(batch, 'decided', Phase.GREEN if green else Phase.RED)
    batch.decision = Decision(
        b['verdict'], b['fence'], b['state_id'], b['result_sha'], b['base_state'],
        b['base_seq'], b['order'], _tuple2(b.get('evidence', ())),
    )
    batch.note = b.get('note') or None
    _end_lease(state, batch)
    if not green:
        _finish(state, batch)


def _ev_apply_intent(state: State, b: Mapping[str, Any]) -> None:
    if state.pending is not None:
        raise CoreError('apply intent while another apply is pending')
    want_seq = state.head.seq + 1 if state.head is not None else 0
    want_parent = state.head.link_id if state.head is not None else None
    if b['seq'] != want_seq or b['parent_link'] != want_parent:
        raise CoreError('apply intent does not extend the HEAD')
    if b['batch_id'] is None:
        if state.head is not None:
            raise CoreError('genesis intent after genesis')
    else:
        batch = _active(state, b['batch_id'])
        decision = batch.decision
        if (
            decision is None or decision.state_id is None or decision.state_id != b['state_id']
            or decision.base_seq != state.head.seq or decision.base_state != state.head.state_id
        ):
            raise CoreError('apply intent for a state that was not trained on the HEAD')
        _move(batch, 'apply_intent', Phase.APPLYING)
    state.pending = Pending(
        b['batch_id'], b['seq'], b['state_id'], b['link_id'], b['parent_link'],
        b['result_sha'], b['fence'], _tuple2(b.get('evidence', ())),
    )


def _ev_applied(state: State, b: Mapping[str, Any]) -> None:
    pending = state.pending
    if pending is None or pending.seq != b['seq'] or pending.link_id != b['link_id']:
        raise CoreError('applied record without a matching intent')
    state.head = Head(pending.seq, pending.state_id, pending.link_id)
    state.pending = None
    if pending.batch_id is not None:
        batch = _active(state, pending.batch_id)
        _move(batch, 'applied', Phase.APPLIED)
        batch.head_seq = pending.seq
        _finish(state, batch)


def _ev_credited(state: State, b: Mapping[str, Any]) -> None:
    batch = _active(state, b['batch_id'])
    _move(batch, 'credited', Phase.CREDITED)
    batch.note = b['reason']
    _finish(state, batch)


def _ev_halted(state: State, b: Mapping[str, Any]) -> None:
    state.halted = b['reason']


def _ev_resumed(state: State, b: Mapping[str, Any]) -> None:
    state.halted = None


def _ev_box_doomed(state: State, b: Mapping[str, Any]) -> None:
    if b['box_id'] in state.live_box_ids():
        raise CoreError('refuse to doom a box that a running lease needs')
    state.doomed[b['box_id']] = b['provider']


def _ev_box_gone(state: State, b: Mapping[str, Any]) -> None:
    if b['box_id'] not in state.doomed:
        raise CoreError('box_gone for a box that was not doomed')
    del state.doomed[b['box_id']]


def _ev_staging_cleared(state: State, b: Mapping[str, Any]) -> None:
    key = (b['batch_id'], b['fence'])
    if key not in state.dirty:
        raise CoreError('staging_cleared for a prefix that was not dirty')
    state.dirty.remove(key)


def _ev_outcome_published(state: State, b: Mapping[str, Any]) -> None:
    if b['batch_id'] not in state.unpublished:
        raise CoreError('outcome_published for a batch that was not pending')
    state.unpublished.remove(b['batch_id'])


_EVOLVE = {
    'batch_admitted': _ev_batch_admitted,
    'batch_rejected': _ev_batch_rejected,
    'lease_issued': _ev_lease_issued,
    'rent_try': _ev_rent_try,
    'box_bound': _ev_box_bound,
    'job_launched': _ev_job_launched,
    'upload_observed': _ev_upload_observed,
    'attempt_failed': _ev_attempt_failed,
    'decided': _ev_decided,
    'apply_intent': _ev_apply_intent,
    'applied': _ev_applied,
    'credited': _ev_credited,
    'halted': _ev_halted,
    'resumed': _ev_resumed,
    'box_doomed': _ev_box_doomed,
    'box_gone': _ev_box_gone,
    'staging_cleared': _ev_staging_cleared,
    'outcome_published': _ev_outcome_published,
}


def evolve(state: State, kind: str, body: Mapping[str, Any]) -> None:
    """Apply one journal record. Raises ``CoreError`` if it does not fit."""
    handler = _EVOLVE.get(kind)
    if handler is not None:
        handler(state, body)
    elif kind not in PASSIVE_KINDS:
        raise CoreError('unknown record kind %r' % (kind,))


def step(cfg: CoreConfig, state: State, ev: Any) -> List[Record]:
    """decide + evolve + resolve to a fixed point, in place. For tests and
    the model checker. The runtime does the same with a journal in between."""
    out: List[Record] = []
    rec = decide(cfg, state, ev)
    while rec is not None:
        evolve(state, rec[0], rec[1])
        out.append(rec)
        rec = resolve(cfg, state)
    return out


# ----------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------

def plan(cfg: CoreConfig, state: State) -> List[Any]:
    """The side effects that must be in progress for this state."""
    ops: List[Any] = []
    for fence in sorted(state.by_fence):
        batch = state.active[state.by_fence[fence]]
        lease = batch.lease
        if batch.phase is Phase.RENTING:
            rung = cfg.ladder[lease.rung % len(cfg.ladder)]
            ops.append(RentOp(fence, lease.try_no, rung, batch.batch_id))
        elif batch.phase is Phase.RUNNING:
            if not lease.launched:
                ops.append(LaunchOp(
                    fence, batch.batch_id, batch.attempts, lease.box_id, lease.provider,
                    lease.base_state, lease.base_seq, batch.ref,
                ))
            ops.append(PollOp(fence, batch.batch_id, lease.box_id, lease.provider))
        elif batch.phase is Phase.SETTLING:
            ops.append(SettleOp(fence, batch.batch_id, lease.base_state))
    pending = state.pending
    if pending is not None and state.halted is None:
        ops.append(ApplyOp(
            pending.batch_id, pending.seq, pending.state_id, pending.link_id,
            pending.parent_link, pending.result_sha, pending.fence, pending.evidence,
        ))
    for box_id in sorted(state.doomed):
        ops.append(DestroyOp(box_id, state.doomed[box_id]))
    for batch_id, fence in sorted(state.dirty):
        ops.append(GcOp(batch_id, fence))
    for batch_id in sorted(state.unpublished):
        ops.append(PublishOp(batch_id))
    return ops


def outcome_fields(batch: Batch) -> Dict[str, Any]:
    """The published outcome of a terminal batch. A pure function of the
    journal, so a repeated publish writes the same bytes."""
    if batch.phase not in TERMINAL:
        raise CoreError('outcome of a batch that is not terminal')
    decision = batch.decision
    credit = CREDIT.get(batch.phase)
    return {
        'batch_id': batch.batch_id,
        'seq': batch.seq,
        'miner': batch.miner,
        'outcome': batch.phase.value,
        'payout_list': credit is not None,
        'credit': credit,
        'applied': batch.phase is Phase.APPLIED,
        'head_seq': batch.head_seq,
        'base_seq': decision.base_seq if decision is not None else None,
        'base_state': decision.base_state if decision is not None else None,
        'state_id': decision.state_id if decision is not None else None,
        'result_sha': decision.result_sha if decision is not None else None,
        'fence': decision.fence if decision is not None else None,
        'attempts': batch.attempts,
        'infra_failures': batch.infra_failures,
        'note': batch.note,
    }
