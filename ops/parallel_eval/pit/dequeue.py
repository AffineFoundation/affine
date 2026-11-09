"""Dequeue + head-binding for PIT.

When an item leaves the queue it binds to the current head. Items dequeued after
a GREEN that becomes the new head bind to that new head. A later GREEN whose
``bound_head`` is no longer the live head is stale: it is paid but
``merge_blocked`` and never applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .state import ItemRecord, ItemState, TransitionError, apply_transition


@dataclass(frozen=True)
class Binding:
    """Immutable head binding stamped at dequeue time."""

    head: str
    head_epoch: int
    master_digest: str
    dequeued_at_mono: float

    def as_dict(self) -> dict:
        return {
            'head': self.head,
            'head_epoch': self.head_epoch,
            'master_digest': self.master_digest,
            'dequeued_at_mono': self.dequeued_at_mono,
        }


def _hex64(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError('%s must be a 64-char hex digest' % field)
    if any(c not in '0123456789abcdef' for c in value):
        raise ValueError('%s must be lowercase hex' % field)
    return value


def bind_item(
    record: ItemRecord,
    *,
    head: str,
    head_epoch: int,
    master_digest: str,
    dequeued_at_mono: float,
) -> ItemRecord:
    """QUEUED -> BOUND with the current head / fp32 master digest.

    ``master_digest`` is the expected published fp32 master hash (fail-closed
    check on the box). Never a secret.
    """
    if record.state is not ItemState.QUEUED:
        raise TransitionError('bind requires queued item, got %s' % record.state.value)
    if type(head_epoch) is not int or head_epoch < 0:
        raise ValueError('head_epoch must be a non-negative int')
    if type(dequeued_at_mono) is not float and type(dequeued_at_mono) is not int:
        raise ValueError('dequeued_at_mono must be numeric')
    head = _hex64(head, 'head')
    master_digest = _hex64(master_digest, 'master_digest')
    binding = Binding(
        head=head,
        head_epoch=head_epoch,
        master_digest=master_digest,
        dequeued_at_mono=float(dequeued_at_mono),
    )
    return apply_transition(
        record,
        ItemState.BOUND,
        reason='bound-to-head',
        updates={
            'bound_head': binding.head,
            'bound_head_epoch': binding.head_epoch,
            'bound_master_digest': binding.master_digest,
            'meta': {**record.meta, 'binding': binding.as_dict()},
        },
    )


def is_stale_green(record: ItemRecord, current_head: str) -> bool:
    """True when a GREEN item's bound head is no longer the live head.

    Stale GREENs are paid and merge_blocked; they must never be applied.
    """
    if record.state not in (ItemState.GREEN, ItemState.MERGE_BLOCKED, ItemState.SCORING):
        # Only meaningful once we know the item is (or will be) green.
        if record.decision != 'green':
            return False
    if not record.bound_head:
        raise ValueError('green item missing bound_head')
    current_head = _hex64(current_head, 'current_head')
    return record.bound_head != current_head


def mark_green_outcome(
    record: ItemRecord,
    *,
    current_head: str,
    z: float,
    candidate_id: str,
) -> ItemRecord:
    """SCORING -> GREEN, then immediately APPLIED or MERGE_BLOCKED.

    First GREEN on the current head applies. Later GREENs on an older bound
    head are merge_blocked (paid, never applied).
    """
    if record.state is not ItemState.SCORING:
        raise TransitionError('green outcome requires scoring, got %s' % record.state.value)
    candidate_id = _hex64(candidate_id, 'candidate_id')
    current_head = _hex64(current_head, 'current_head')
    green = apply_transition(
        record,
        ItemState.GREEN,
        reason='gate-green',
        updates={
            'decision': 'green',
            'z': float(z),
            'candidate_id': candidate_id,
            'paid': True,
        },
    )
    if is_stale_green(green, current_head):
        return apply_transition(
            green,
            ItemState.MERGE_BLOCKED,
            reason='stale-bound-head',
            updates={'merge_blocked': True, 'paid': True},
        )
    if green.bound_head != current_head:
        # Defensive: bind_item always sets bound_head; mismatch without stale means corruption.
        raise TransitionError('bound_head/current_head invariant broken')
    return apply_transition(
        green,
        ItemState.APPLIED,
        reason='first-green-for-head',
        updates={'merge_blocked': False, 'paid': True},
    )


def mark_red_outcome(record: ItemRecord, *, z: Optional[float] = None) -> ItemRecord:
    if record.state is not ItemState.SCORING:
        raise TransitionError('red outcome requires scoring, got %s' % record.state.value)
    updates = {'decision': 'red', 'paid': False, 'merge_blocked': False}
    if z is not None:
        updates['z'] = float(z)
    return apply_transition(record, ItemState.RED, reason='gate-red', updates=updates)


def order_greens(items: Sequence[ItemRecord]) -> List[ItemRecord]:
    """GREEN / MERGE_BLOCKED / APPLIED items in decision-time order.

    Ordering key: (green_at_mono from meta, registration_index, item_id).
    Missing green_at_mono sorts last among undecided timestamps.
    """
    def key(item: ItemRecord):
        meta = item.meta or {}
        green_at = meta.get('green_at_mono')
        # Only finished greens participate.
        if item.state not in (ItemState.GREEN, ItemState.MERGE_BLOCKED, ItemState.APPLIED):
            return (float('inf'), item.registration_index, item.item_id)
        if green_at is None:
            return (float('inf'), item.registration_index, item.item_id)
        return (float(green_at), item.registration_index, item.item_id)

    finished = [
        i for i in items
        if i.state in (ItemState.GREEN, ItemState.MERGE_BLOCKED, ItemState.APPLIED)
        and i.decision == 'green'
    ]
    return sorted(finished, key=key)


def select_next_queued(
    items: Iterable[ItemRecord],
    *,
    in_flight: int,
    box_cap: int,
) -> Optional[ItemRecord]:
    """FIFO next QUEUED item if under the box cap."""
    if box_cap < 1:
        raise ValueError('box_cap must be >= 1')
    if in_flight >= box_cap:
        return None
    queued = [i for i in items if i.state is ItemState.QUEUED]
    if not queued:
        return None
    queued.sort(key=lambda i: (i.registration_index, i.item_id))
    return queued[0]
