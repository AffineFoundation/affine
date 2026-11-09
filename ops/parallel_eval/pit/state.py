"""PIT item state machine (DESIGN §2).

QUEUED → BOUND → RENTING → PROVISIONING → TRAINING → SEALED → SERVING →
PROBING → RESULT → SCORED → DECIDED → RELEASED

Boxes are rented only while an item is in RENTING…DECIDED. RELEASED (or
ABORTED→RELEASED) always frees the box. No warm pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple


class ItemState(str, Enum):
    QUEUED = 'queued'
    BOUND = 'bound'
    RENTING = 'renting'
    PROVISIONING = 'provisioning'
    TRAINING = 'training'
    SEALED = 'sealed'
    SERVING = 'serving'
    PROBING = 'probing'
    RESULT = 'result'
    SCORED = 'scored'
    DECIDED = 'decided'
    RELEASED = 'released'
    RETRY_WAIT = 'retry_wait'
    FAILED = 'failed'
    ABORTED = 'aborted'
    NO_DECISION = 'no_decision'


# States that may hold a live rental. Empty queue ⇒ zero of these.
RENTED_STATES: FrozenSet[ItemState] = frozenset({
    ItemState.RENTING, ItemState.PROVISIONING, ItemState.TRAINING,
    ItemState.SEALED, ItemState.SERVING, ItemState.PROBING,
    ItemState.RESULT, ItemState.SCORED, ItemState.DECIDED,
})

TERMINAL = frozenset({
    ItemState.RELEASED, ItemState.ABORTED, ItemState.NO_DECISION,
})

ALLOWED: Dict[ItemState, FrozenSet[ItemState]] = {
    ItemState.QUEUED: frozenset({ItemState.BOUND, ItemState.ABORTED}),
    ItemState.BOUND: frozenset({ItemState.RENTING, ItemState.ABORTED}),
    ItemState.RENTING: frozenset({
        ItemState.PROVISIONING, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.PROVISIONING: frozenset({
        ItemState.TRAINING, ItemState.RETRY_WAIT, ItemState.FAILED,
        ItemState.ABORTED, ItemState.NO_DECISION,
    }),
    ItemState.TRAINING: frozenset({
        ItemState.SEALED, ItemState.RETRY_WAIT, ItemState.FAILED,
        ItemState.ABORTED, ItemState.NO_DECISION,
    }),
    ItemState.SEALED: frozenset({
        ItemState.SERVING, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.SERVING: frozenset({
        ItemState.PROBING, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.PROBING: frozenset({
        ItemState.RESULT, ItemState.RETRY_WAIT, ItemState.FAILED,
        ItemState.ABORTED, ItemState.NO_DECISION,
    }),
    ItemState.RESULT: frozenset({
        ItemState.SCORED, ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.ABORTED,
    }),
    ItemState.SCORED: frozenset({ItemState.DECIDED, ItemState.ABORTED}),
    ItemState.DECIDED: frozenset({ItemState.RELEASED}),
    ItemState.NO_DECISION: frozenset({ItemState.RELEASED}),
    ItemState.FAILED: frozenset({ItemState.RETRY_WAIT, ItemState.ABORTED, ItemState.RELEASED}),
    ItemState.RETRY_WAIT: frozenset({ItemState.RENTING, ItemState.ABORTED}),
    ItemState.ABORTED: frozenset({ItemState.RELEASED}),
    ItemState.RELEASED: frozenset(),
}


class TransitionError(ValueError):
    """Illegal state transition."""


@dataclass
class ItemRecord:
    """One queue item. Never stores secrets, seeds, or presigned URLs."""

    item_id: str
    registration_index: int
    epoch: int
    uid: int = 0
    hotkey_prefix: str = ''
    state: ItemState = ItemState.QUEUED
    bound_head: Optional[str] = None
    bound_master_digest: Optional[str] = None
    bound_round_id: Optional[str] = None
    bound_plan_seed: Optional[str] = None  # cand_seed digest handle only; never raw seed
    candidate_id: Optional[str] = None
    box_id: Optional[str] = None
    box_name: Optional[str] = None
    attempt: int = 0
    sealed: bool = False
    decision: Optional[str] = None  # green_fresh | green_stale | red | no_decision
    z: Optional[float] = None
    fresh: Optional[bool] = None
    merge_blocked: bool = False
    paid: bool = False
    result_stored: bool = False
    shadow: bool = True
    last_error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def miner_hotkey_prefix(self) -> str:
        return self.hotkey_prefix

    def snapshot(self) -> dict:
        return {
            'item_id': self.item_id,
            'registration_index': self.registration_index,
            'epoch': self.epoch,
            'uid': self.uid,
            'hotkey_prefix': self.hotkey_prefix,
            'state': self.state.value,
            'bound_head': self.bound_head,
            'bound_master_digest': self.bound_master_digest,
            'bound_round_id': self.bound_round_id,
            'bound_plan_seed': self.bound_plan_seed,
            'candidate_id': self.candidate_id,
            'box_id': self.box_id,
            'box_name': self.box_name,
            'attempt': self.attempt,
            'sealed': self.sealed,
            'decision': self.decision,
            'z': self.z,
            'fresh': self.fresh,
            'merge_blocked': self.merge_blocked,
            'paid': self.paid,
            'result_stored': self.result_stored,
            'shadow': self.shadow,
            'last_error': self.last_error,
            'meta': dict(self.meta),
        }


def apply_transition(
    record: ItemRecord,
    to: ItemState,
    *,
    reason: Optional[str] = None,
    updates: Optional[Mapping[str, Any]] = None,
) -> ItemRecord:
    allowed = ALLOWED.get(record.state, frozenset())
    if to not in allowed:
        raise TransitionError(
            'illegal transition %s -> %s (item=%s)' % (record.state.value, to.value, record.item_id)
        )
    kwargs: Dict[str, Any] = {'state': to}
    if reason is not None:
        if to in {ItemState.FAILED, ItemState.RETRY_WAIT, ItemState.ABORTED, ItemState.NO_DECISION}:
            kwargs['last_error'] = reason
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


def holds_rental(record: ItemRecord) -> bool:
    return record.state in RENTED_STATES and bool(record.box_id)
