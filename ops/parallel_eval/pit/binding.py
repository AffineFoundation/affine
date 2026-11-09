"""Dequeue + head binding (BRIEF §1.2, DESIGN §3).

Dequeue is the only binding point. If M(H) is unpublished, dequeue pauses
(returns None). Indices are never copied into ITEM.json.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .expected_master import lookup_expected_master
from .state import ItemRecord, ItemState, TransitionError, apply_transition


def _hex64(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError('%s must be a 64-char hex digest' % field)
    if any(c not in '0123456789abcdef' for c in value):
        raise ValueError('%s must be lowercase hex' % field)
    return value


def live_cand_seed(secret_seed: str, epoch: int) -> str:
    """Live candidate plan seed — matches ``baseline_v1.cand_seed`` exactly."""
    if not isinstance(secret_seed, str) or not secret_seed:
        raise ValueError('secret_seed required')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be non-negative int')
    return hashlib.sha256(
        ('%s:prior-baseline:cand:%d' % (secret_seed, int(epoch))).encode()
    ).hexdigest()


def cand_seed_handle(round_seed_commitment: str, epoch: int) -> str:
    """Shadow/test plan-seed handle only — **not** live-equivalent to ``live_cand_seed``.

    Uses only a commitment/handle string already safe to store, hashed with epoch.
    """
    if not isinstance(round_seed_commitment, str) or not round_seed_commitment:
        raise ValueError('round_seed_commitment required')
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be non-negative int')
    material = ('pit-cand-seed-v1|%s|%d' % (round_seed_commitment, epoch)).encode()
    return hashlib.sha256(material).hexdigest()


def _resolve_plan_seed(
    epoch: int,
    *,
    secret_seed: Optional[str] = None,
    plan_seed_fn: Optional[Callable[[int], str]] = None,
    round_seed_commitment: Optional[str] = None,
) -> str:
    if secret_seed is not None:
        return live_cand_seed(secret_seed, epoch)
    if plan_seed_fn is not None:
        return plan_seed_fn(epoch)
    if round_seed_commitment is not None:
        return cand_seed_handle(round_seed_commitment, epoch)
    raise ValueError('secret_seed, plan_seed_fn, or round_seed_commitment required')


@dataclass(frozen=True)
class Binding:
    head: str
    master_digest: str
    round_id: str
    plan_seed: str
    head_epoch: int
    dequeued_at_mono: float

    def as_dict(self) -> dict:
        return {
            'H': self.head,
            'master_digest': self.master_digest,
            'round_id': self.round_id,
            'plan_seed': self.plan_seed,
            'head_epoch': self.head_epoch,
            'dequeued_at_mono': self.dequeued_at_mono,
        }


def dequeue_and_bind(
    record: ItemRecord,
    *,
    head: str,
    head_epoch: int,
    round_id: str,
    round_seed_commitment: Optional[str] = None,
    secret_seed: Optional[str] = None,
    plan_seed_fn: Optional[Callable[[int], str]] = None,
    master_digest: Optional[str] = None,
    chain_override: Optional[Dict[str, str]] = None,
    public_manifest: Optional[Dict[str, Any]] = None,
    index: Optional[Dict[str, str]] = None,
    training_result: Optional[Dict[str, Any]] = None,
    dequeued_at_mono: Optional[float] = None,
) -> Optional[ItemRecord]:
    """QUEUED → BOUND, or None if master unpublished (pause dequeue)."""
    if record.state is not ItemState.QUEUED:
        raise TransitionError('dequeue requires queued item, got %s' % record.state.value)
    head = _hex64(head, 'head')
    dig = master_digest or lookup_expected_master(
        head,
        chain_override=chain_override,
        public_manifest=public_manifest,
        index=index,
        training_result=training_result,
    )
    if dig is None:
        return None  # pause — head not bindable yet
    dig = _hex64(dig, 'master_digest')
    if type(head_epoch) is not int or head_epoch < 0:
        raise ValueError('head_epoch must be non-negative int')
    if not isinstance(round_id, str) or not round_id:
        raise ValueError('round_id required')
    plan = _resolve_plan_seed(
        record.epoch,
        secret_seed=secret_seed,
        plan_seed_fn=plan_seed_fn,
        round_seed_commitment=round_seed_commitment,
    )
    now = float(dequeued_at_mono if dequeued_at_mono is not None else time.monotonic())
    binding = Binding(
        head=head,
        master_digest=dig,
        round_id=round_id,
        plan_seed=plan,
        head_epoch=head_epoch,
        dequeued_at_mono=now,
    )
    return apply_transition(
        record,
        ItemState.BOUND,
        reason='bound-to-head',
        updates={
            'bound_head': binding.head,
            'bound_master_digest': binding.master_digest,
            'bound_round_id': binding.round_id,
            'bound_plan_seed': binding.plan_seed,
            'meta': {**record.meta, 'binding': binding.as_dict()},
        },
    )


def rebind_if_unsealed(
    record: ItemRecord,
    *,
    head: str,
    head_epoch: int,
    round_id: str,
    round_seed_commitment: Optional[str] = None,
    secret_seed: Optional[str] = None,
    plan_seed_fn: Optional[Callable[[int], str]] = None,
    master_digest: str,
    dequeued_at_mono: Optional[float] = None,
) -> ItemRecord:
    """Retry before SEALED: rebind to the *current* head (DESIGN §3 / §8)."""
    if record.sealed or record.state in {
        ItemState.SEALED, ItemState.SERVING, ItemState.PROBING,
        ItemState.RESULT, ItemState.SCORED, ItemState.DECIDED, ItemState.RELEASED,
    }:
        raise TransitionError('rebind refused after SEALED')
    if record.state not in {ItemState.RETRY_WAIT, ItemState.FAILED, ItemState.BOUND, ItemState.RENTING}:
        # Allow rebind from retry path; caller resets toward RENTING afterward.
        if record.state is not ItemState.QUEUED:
            pass
    # Rebind fields in place via a synthetic BOUND update from RETRY_WAIT/FAILED
    # is not a state transition — only rewrite binding metadata, then caller rents.
    head = _hex64(head, 'head')
    dig = _hex64(master_digest, 'master_digest')
    plan = _resolve_plan_seed(
        record.epoch,
        secret_seed=secret_seed,
        plan_seed_fn=plan_seed_fn,
        round_seed_commitment=round_seed_commitment,
    )
    now = float(dequeued_at_mono if dequeued_at_mono is not None else time.monotonic())
    binding = Binding(
        head=head, master_digest=dig, round_id=round_id, plan_seed=plan,
        head_epoch=head_epoch, dequeued_at_mono=now,
    )
    record.bound_head = binding.head
    record.bound_master_digest = binding.master_digest
    record.bound_round_id = binding.round_id
    record.bound_plan_seed = binding.plan_seed
    record.candidate_id = None
    record.sealed = False
    record.meta = {**record.meta, 'binding': binding.as_dict(), 'rebound': True}
    return record


def is_stale_green(record: ItemRecord, current_head: str) -> bool:
    if not record.bound_head:
        raise ValueError('missing bound_head')
    return record.bound_head != _hex64(current_head, 'current_head')


def mark_scored_decision(
    record: ItemRecord,
    *,
    current_head: str,
    decision: str,
    z: Optional[float],
    candidate_id: str,
) -> ItemRecord:
    """SCORED → DECIDED with fresh/stale classification (Option B)."""
    if record.state is not ItemState.SCORED:
        raise TransitionError('decision requires scored, got %s' % record.state.value)
    candidate_id = _hex64(candidate_id, 'candidate_id')
    current_head = _hex64(current_head, 'current_head')
    if decision == 'red':
        return apply_transition(
            record, ItemState.DECIDED, reason='gate-red',
            updates={
                'decision': 'red', 'z': None if z is None else float(z),
                'candidate_id': candidate_id, 'paid': False, 'fresh': None,
                'merge_blocked': False, 'result_stored': True,
            },
        )
    if decision != 'green':
        raise ValueError('decision must be green|red')
    stale = is_stale_green(record, current_head)
    if stale:
        # Jacob: later GREEN from an old head uploads, is paid, stranded/merge_blocked;
        # it never becomes HEAD.
        return apply_transition(
            record, ItemState.DECIDED, reason='green-stale-stranded',
            updates={
                'decision': 'green_stale', 'z': float(z if z is not None else 0.0),
                'candidate_id': candidate_id, 'paid': True, 'fresh': False,
                'merge_blocked': True, 'stranded': True, 'result_stored': True,
            },
        )
    return apply_transition(
        record, ItemState.DECIDED, reason='green-fresh',
        updates={
            'decision': 'green_fresh', 'z': float(z if z is not None else 0.0),
            'candidate_id': candidate_id, 'paid': True, 'fresh': True,
            'merge_blocked': False, 'stranded': False, 'result_stored': True,
        },
    )


def select_next_queued(
    items: Iterable[ItemRecord],
    *,
    rented: int,
    max_boxes: int,
) -> Optional[ItemRecord]:
    """FIFO next QUEUED item if under the box cap.

    Hard rule: rentals ≤ min(queued_or_in_flight_demand, max_boxes). The
    controller only calls rent after bind; empty queue ⇒ rented stays 0.
    """
    if max_boxes < 1:
        raise ValueError('max_boxes must be >= 1')
    if rented >= max_boxes:
        return None
    queued = [i for i in items if i.state is ItemState.QUEUED]
    if not queued:
        return None
    queued.sort(key=lambda i: (i.registration_index, i.epoch, i.item_id))
    return queued[0]


def order_greens(items: Sequence[ItemRecord]) -> List[ItemRecord]:
    def key(item: ItemRecord):
        gat = (item.meta or {}).get('decided_at_mono')
        if item.decision not in ('green_fresh', 'green_stale'):
            return (float('inf'), item.registration_index, item.item_id)
        if gat is None:
            return (float('inf'), item.registration_index, item.item_id)
        return (float(gat), item.registration_index, item.item_id)

    finished = [i for i in items if i.decision in ('green_fresh', 'green_stale')]
    return sorted(finished, key=key)
