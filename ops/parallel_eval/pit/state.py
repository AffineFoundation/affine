"""PIT item state machine.

One registration = one item = one box lifecycle. Transitions are explicit and
exhaustive; unknown edges raise TransitionError.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple


class ItemState(str, Enum):
    QUEUED = 'queued'
    BOUND = 'bound'
    RENTING = 'renting'
    DOWNLOADING = 'downloading'
    TRAINING = 'training'
    PROBING = 'probing'
    SCORING = 'scoring'
    GREEN = 'green'
    RED = 'red'
    MERGE_BLOCKED = 'merge_blocked'
    APPLIED = 'applied'
    FAILED = 'failed'
    RETRY_WAIT = 'retry_wait'
    RELEASED = 'released'
    ABORTED = 'aborted'


# Terminal outcomes that still release the box.
TERMINAL = frozenset({
    ItemState.APPLIED,
    ItemState.MERGE_BLOCKED,
    ItemState.RED,
    ItemState.ABORTED,
    ItemState.RELEASED,
})

# Edges: from -> frozenset(to)
ALLOWED: Dict[ItemState, FrozenSet[ItemState]] = {
    ItemState.QUEUED: frozenset({ItemState.BOUND, ItemState.ABORTED}),
    ItemState.BOUND: frozenset({ItemState.RENTING, ItemState.ABORTED}),
    ItemState.RENTING: frozenset({
        ItemState.DOWNLOADING, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.DOWNLOADING: frozenset({
        ItemState.TRAINING, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.TRAINING: frozenset({
        ItemState.PROBING, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.PROBING: frozenset({
        ItemState.SCORING, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.SCORING: frozenset({
        ItemState.GREEN, ItemState.RED, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.GREEN: frozenset({
        ItemState.APPLIED, ItemState.MERGE_BLOCKED, ItemState.RELEASED,
    }),
    ItemState.RED: frozenset({ItemState.RELEASED}),
    ItemState.MERGE_BLOCKED: frozenset({ItemState.RELEASED}),
    ItemState.APPLIED: frozenset({ItemState.RELEASED}),
    ItemState.FAILED: frozenset({ItemState.RETRY_WAIT, ItemState.ABORTED, ItemState.RELEASED}),
    ItemState.RETRY_WAIT: frozenset({ItemState.RENTING, ItemState.ABORTED}),
    ItemState.RELEASED: frozenset(),
    ItemState.ABORTED: frozenset({ItemState.RELEASED}),
}


class TransitionError(ValueError):
    """Raised when a state transition is not in ALLOWED."""


@dataclass
class ItemRecord:
    """Durable-ish in-memory record for one queue item.

    Secrets, seeds, presigned URLs and private keys must never be stored here.
    Use opaque handle placeholders only.
    """

    item_id: str
    registration_index: int
    miner_hotkey_prefix: str  # truncated public id only (no secrets)
    state: ItemState = ItemState.QUEUED
    bound_head: Optional[str] = None
    bound_head_epoch: Optional[int] = None
    bound_master_digest: Optional[str] = None
    candidate_id: Optional[str] = None
    box_id: Optional[str] = None
    attempt: int = 0
    last_error: Optional[str] = None
    decision: Optional[str] = None  # 'green' | 'red' | None
    z: Optional[float] = None
    merge_blocked: bool = False
    paid: bool = False
    shadow: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            'item_id': self.item_id,
            'registration_index': self.registration_index,
            'miner_hotkey_prefix': self.miner_hotkey_prefix,
            'state': self.state.value,
            'bound_head': self.bound_head,
            'bound_head_epoch': self.bound_head_epoch,
            'bound_master_digest': self.bound_master_digest,
            'candidate_id': self.candidate_id,
            'box_id': self.box_id,
            'attempt': self.attempt,
            'last_error': self.last_error,
            'decision': self.decision,
            'z': self.z,
            'merge_blocked': self.merge_blocked,
            'paid': self.paid,
            'shadow': self.shadow,
            'meta': dict(self.meta),
        }


def apply_transition(
    record: ItemRecord,
    to: ItemState,
    *,
    reason: Optional[str] = None,
    updates: Optional[Mapping[str, Any]] = None,
) -> ItemRecord:
    """Return a new ItemRecord after a validated transition."""
    allowed = ALLOWED.get(record.state, frozenset())
    if to not in allowed:
        raise TransitionError(
            'illegal transition %s -> %s (item=%s)' % (record.state.value, to.value, record.item_id)
        )
    kwargs: Dict[str, Any] = {'state': to}
    if reason is not None:
        kwargs['last_error'] = reason if to in {
            ItemState.FAILED, ItemState.RETRY_WAIT, ItemState.ABORTED,
        } else record.last_error
        meta = dict(record.meta)
        meta['last_transition_reason'] = reason
        kwargs['meta'] = meta
    if updates:
        for key, value in updates.items():
            if key in ('item_id', 'state'):
                raise TransitionError('cannot rewrite item_id/state via updates')
            if not hasattr(record, key):
                raise TransitionError('unknown ItemRecord field %r' % (key,))
            kwargs[key] = value
    return replace(record, **kwargs)


def can_transition(record: ItemRecord, to: ItemState) -> bool:
    return to in ALLOWED.get(record.state, frozenset())


def next_states(record: ItemRecord) -> Tuple[ItemState, ...]:
    return tuple(sorted(ALLOWED.get(record.state, frozenset()), key=lambda s: s.value))
