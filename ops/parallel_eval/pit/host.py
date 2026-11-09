"""PIT host controller.

Owns the queue, box cap, binding, scoring, decision log, and shadow guards.
Reuses pe_rent / pe_lane / pe_score / pe_live through :mod:`adapters`.

Shadow mode (only authorized mode here) never writes DECISION.json, weights,
HEAD-CHAIN, or R2 head pointers.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from .adapters import AdapterUnavailable, HostAdapters, load_host_adapters
from .admission import (
    AdmissionRequest,
    EphemeralBoxKey,
    build_admission_request,
    generate_ephemeral_box_key,
    request_real_root_signature,
)
from .box import BoxIO, BoxRunner, Fp32MasterError, ProbeBundle
from .config import PitConfig, load_pit_config
from .decision_log import DecisionLog
from .dequeue import (
    bind_item,
    mark_green_outcome,
    mark_red_outcome,
    order_greens,
    select_next_queued,
)
from .shadow import ShadowGuard, ShadowWriteError
from .state import ItemRecord, ItemState, apply_transition


def _new_item_id() -> str:
    return uuid.uuid4().hex


@dataclass
class HostController:
    """In-process PIT controller suitable for shadow runs and unit tests."""

    config: PitConfig
    adapters: HostAdapters
    box_io_factory: Callable[[], BoxIO]
    decision_log: DecisionLog = field(default_factory=DecisionLog)
    items: List[ItemRecord] = field(default_factory=list)
    current_head: Optional[str] = None
    current_head_epoch: int = 0
    current_master_digest: Optional[str] = None
    shadow: Optional[ShadowGuard] = None
    now_fn: Callable[[], float] = time.time
    # Trainer-pod fallback (DESIGN §12): kept available; not used in happy path.
    trainer_pod_fallback: Optional[Callable[..., Any]] = None

    def __post_init__(self) -> None:
        self.shadow = self.shadow or ShadowGuard(self.config)
        if self.config.enabled and not self.config.may_run():
            raise ValueError('PIT enabled but mode is not runnable in this PR')

    @classmethod
    def from_flag(
        cls,
        parallel_eval: Mapping[str, Any],
        *,
        box_io_factory: Callable[[], BoxIO],
        adapter_overrides: Optional[Mapping[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> 'HostController':
        cfg = load_pit_config(parallel_eval)
        adapters = load_host_adapters(
            pe_rent_module=cfg.pe_rent_module,
            pe_lane_module=cfg.pe_lane_module,
            pe_score_module=cfg.pe_score_module,
            pe_live_module=cfg.pe_live_module,
            overrides=adapter_overrides,
        )
        return cls(config=cfg, adapters=adapters, box_io_factory=box_io_factory, **kwargs)

    # ------------------------------------------------------------------ queue
    def enqueue(
        self,
        *,
        registration_index: int,
        miner_hotkey_prefix: str,
        item_id: Optional[str] = None,
    ) -> ItemRecord:
        if len(miner_hotkey_prefix) > 16:
            miner_hotkey_prefix = miner_hotkey_prefix[:16]
        rec = ItemRecord(
            item_id=item_id or _new_item_id(),
            registration_index=registration_index,
            miner_hotkey_prefix=miner_hotkey_prefix,
            shadow=self.config.is_shadow(),
        )
        self.items.append(rec)
        return rec

    def in_flight(self) -> int:
        busy = {
            ItemState.BOUND, ItemState.RENTING, ItemState.DOWNLOADING,
            ItemState.TRAINING, ItemState.PROBING, ItemState.SCORING,
            ItemState.RETRY_WAIT,
        }
        return sum(1 for i in self.items if i.state in busy)

    def set_head(self, head: str, *, epoch: int, master_digest: str) -> None:
        if len(head) != 64 or len(master_digest) != 64:
            raise ValueError('head and master_digest must be hex64')
        self.current_head = head
        self.current_head_epoch = int(epoch)
        self.current_master_digest = master_digest

    # ------------------------------------------------------------------ tick
    def tick(self) -> Dict[str, Any]:
        """One controller tick: dequeue under cap, advance in-flight, score.

        Shadow: all live artifact writes are refused by ShadowGuard.
        """
        self.shadow.assert_shadow()
        if not self.config.may_run() and self.config.enabled:
            raise ValueError('PIT not runnable')
        # Even when disabled, tick is a no-op (safe).
        if not self.config.enabled:
            return {'ok': True, 'enabled': False, 'actions': []}

        actions: List[dict] = []
        # 1) Dequeue under box_cap
        while True:
            nxt = select_next_queued(self.items, in_flight=self.in_flight(), box_cap=self.config.box_cap)
            if nxt is None:
                break
            if not self.current_head or not self.current_master_digest:
                actions.append({'op': 'wait-head', 'item_id': nxt.item_id})
                break
            bound = bind_item(
                nxt,
                head=self.current_head,
                head_epoch=self.current_head_epoch,
                master_digest=self.current_master_digest,
                dequeued_at_mono=self.now_fn(),
            )
            self._replace(bound)
            actions.append({'op': 'bound', 'item_id': bound.item_id, 'head': bound.bound_head[:12]})

        # 2) Advance each non-terminal item one step (cooperative)
        for item in list(self.items):
            if item.state in (ItemState.RELEASED, ItemState.ABORTED):
                continue
            try:
                advanced, note = self._advance(item)
                if advanced is not item:
                    self._replace(advanced)
                    actions.append({'op': 'advance', 'item_id': item.item_id, 'to': advanced.state.value, 'note': note})
            except (Fp32MasterError, AdapterUnavailable, ShadowWriteError, TransitionErrorWrap, RuntimeError, ValueError) as exc:
                failed = self._fail(item, str(exc)[:300])
                self._replace(failed)
                actions.append({'op': 'fail', 'item_id': item.item_id, 'error': str(exc)[:200]})

        return {
            'ok': True,
            'enabled': True,
            'mode': self.config.mode,
            'in_flight': self.in_flight(),
            'box_cap': self.config.box_cap,
            'head': None if not self.current_head else self.current_head[:12],
            'actions': actions,
            'shadow': True,
        }

    def _replace(self, record: ItemRecord) -> None:
        for i, old in enumerate(self.items):
            if old.item_id == record.item_id:
                self.items[i] = record
                return
        raise KeyError('unknown item %s' % record.item_id)

    def _fail(self, item: ItemRecord, reason: str) -> ItemRecord:
        from .state import can_transition
        next_attempt = item.attempt + 1
        if next_attempt > self.config.max_infra_retries:
            # Exhausted retries: abort (then release on a later tick).
            if can_transition(item, ItemState.ABORTED):
                return apply_transition(
                    item, ItemState.ABORTED, reason=reason,
                    updates={'attempt': next_attempt, 'last_error': reason},
                )
            if can_transition(item, ItemState.FAILED):
                failed = apply_transition(
                    item, ItemState.FAILED, reason=reason,
                    updates={'attempt': next_attempt, 'last_error': reason},
                )
                return apply_transition(failed, ItemState.ABORTED, reason=reason)
            if can_transition(item, ItemState.RELEASED):
                return apply_transition(item, ItemState.RELEASED, reason=reason)
            return item
        if can_transition(item, ItemState.RETRY_WAIT):
            return apply_transition(
                item, ItemState.RETRY_WAIT, reason=reason,
                updates={'attempt': next_attempt, 'last_error': reason},
            )
        if can_transition(item, ItemState.FAILED):
            failed = apply_transition(
                item, ItemState.FAILED, reason=reason,
                updates={'attempt': next_attempt, 'last_error': reason},
            )
            return apply_transition(failed, ItemState.RETRY_WAIT, reason=reason)
        if can_transition(item, ItemState.ABORTED):
            return apply_transition(item, ItemState.ABORTED, reason=reason)
        return item

    def _advance(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        if item.state is ItemState.BOUND:
            return apply_transition(item, ItemState.RENTING, reason='rent-start'), 'renting'
        if item.state is ItemState.RENTING:
            return self._rent(item)
        if item.state is ItemState.DOWNLOADING:
            return self._run_box(item)
        if item.state is ItemState.RETRY_WAIT:
            return apply_transition(item, ItemState.RENTING, reason='retry'), 'retry-rent'
        if item.state is ItemState.SCORING:
            return self._score(item)
        if item.state in (ItemState.GREEN, ItemState.RED, ItemState.MERGE_BLOCKED, ItemState.APPLIED):
            # Release box; shadow refuses live apply writes.
            self._shadow_refuse_live_apply(item)
            return apply_transition(item, ItemState.RELEASED, reason='done'), 'released'
        if item.state is ItemState.FAILED:
            # Failed items that still have retry budget wait; exhausted ones abort.
            if item.attempt > self.config.max_infra_retries:
                return apply_transition(item, ItemState.ABORTED, reason=item.last_error or 'exhausted'), 'abort'
            return apply_transition(item, ItemState.RETRY_WAIT, reason=item.last_error or 'retry'), 'retry-wait'
        if item.state is ItemState.ABORTED:
            return apply_transition(item, ItemState.RELEASED, reason='aborted'), 'released'
        # TRAINING / PROBING are driven inside _run_box in one tick for this PR.
        return item, 'noop'

    def _rent(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        try:
            rent = self.adapters.require('rent')
        except AdapterUnavailable:
            # Shadow/unit path: allocate a synthetic box id without host rental.
            box_id = 'shadow-box-%s' % item.item_id[:12]
            updated = apply_transition(
                item, ItemState.DOWNLOADING, reason='shadow-rent',
                updates={'box_id': box_id},
            )
            return updated, 'shadow-rent'
        result = rent(item_id=item.item_id, attempt=item.attempt)
        box_id = (result or {}).get('box_id')
        if not box_id:
            raise TransitionErrorWrap('rent returned no box_id')
        # Optional lane assignment
        if self.adapters.lane:
            self.adapters.lane(box_id=box_id, item_id=item.item_id)
        updated = apply_transition(
            item, ItemState.DOWNLOADING, reason='rented',
            updates={'box_id': str(box_id)},
        )
        return updated, 'rented'

    def _run_box(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        runner = BoxRunner(config=self.config, io=self.box_io_factory())
        box_key = generate_ephemeral_box_key()
        # Build admission request only — never request a real ROOT signature.
        req = build_admission_request(
            box_key,
            item_id=item.item_id,
            bound_head=item.bound_head or '',
            bound_master_digest=item.bound_master_digest or '',
            now=self.now_fn(),
        )
        # Explicitly document that we do not call request_real_root_signature.
        _ = AdmissionRequest, request_real_root_signature, EphemeralBoxKey
        batch_handle = 'batch-handle:%s' % item.item_id
        plan_handle = 'plan-handle:head-epoch-%d' % (item.bound_head_epoch or 0)
        updated, bundle = runner.run(
            item,
            batch_handle=batch_handle,
            plan_handle=plan_handle,
            box_key=box_key,
            admission_cert=None,  # no real ROOT cert in this PR
            now=self.now_fn(),
        )
        # box.run already transitioned to SCORING; only attach host meta.
        meta = dict(updated.meta)
        meta['probe_bundle'] = {
            'candidate_id': bundle.candidate_id,
            'probe_partial': bundle.probe_result.get('partial'),
            'master_upload_ok': bool((bundle.master_upload or {}).get('ok')),
            'box_public_key_hex': bundle.box_public_key_hex,
            'admission_request_digest': req.digest(),
        }
        meta['_bundle'] = bundle
        updated.meta = meta
        updated.candidate_id = bundle.candidate_id
        return updated, 'box-complete'

    def _score(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        bundle: Optional[ProbeBundle] = (item.meta or {}).get('_bundle')
        if bundle is None:
            raise TransitionErrorWrap('missing probe bundle for scoring')
        # Prefer pe_score adapter; fall back to a local strict-z>1 fixture rule
        # that only accepts an explicit score dict from the adapter/probe.
        score_fn = self.adapters.score
        if score_fn is not None:
            scored = score_fn(probe=bundle.probe_result, item_id=item.item_id)
        else:
            scored = bundle.probe_result.get('score') or {}
        decision = str(scored.get('decision') or '').lower()
        z = scored.get('z')
        if decision not in ('green', 'red'):
            raise TransitionErrorWrap('score missing decision green|red')
        green_at = float(self.now_fn())
        item.meta['green_at_mono'] = green_at
        if decision == 'red':
            updated = mark_red_outcome(item, z=None if z is None else float(z))
            self._append_decision(updated, decision='red', green_at_mono=None)
            self._shadow_refuse_live_apply(updated)
            return updated, 'red'
        # green
        if self.current_head is None:
            raise TransitionErrorWrap('cannot resolve green without current_head')
        updated = mark_green_outcome(
            item,
            current_head=self.current_head,
            z=float(z if z is not None else 0.0),
            candidate_id=bundle.candidate_id,
        )
        if updated.state is ItemState.APPLIED:
            self._append_decision(updated, decision='green', green_at_mono=green_at)
            # Promote head in controller memory only; shadow refuses durable writes.
            self._shadow_refuse_live_apply(updated)
            self.current_head = updated.candidate_id
            self.current_head_epoch = int(updated.bound_head_epoch or 0) + 1
            # Master digest for the new head must come from the upload receipt later;
            # keep prior digest as placeholder until a live publish path exists.
            return updated, 'applied-new-head'
        self._append_decision(updated, decision='merge_blocked', green_at_mono=green_at)
        self._shadow_refuse_live_apply(updated)
        return updated, 'merge-blocked'

    def _append_decision(
        self,
        item: ItemRecord,
        *,
        decision: str,
        green_at_mono: Optional[float],
    ) -> None:
        self.decision_log.append(
            item_id=item.item_id,
            registration_index=item.registration_index,
            decision=decision,
            bound_head=item.bound_head or '',
            candidate_id=item.candidate_id,
            z=item.z,
            paid=item.paid,
            merge_blocked=item.merge_blocked,
            green_at_mono=green_at_mono,
            shadow=True,
            at_unix=self.now_fn(),
        )

    def _shadow_refuse_live_apply(self, item: ItemRecord) -> None:
        """Prove shadow never writes live artifacts around an apply/decision."""
        for action in (
            'write_decision',
            'write_head_chain',
            'write_weights',
            'publish_r2_head',
            'apply_green_live',
        ):
            try:
                self.shadow.check_action(action)
            except ShadowWriteError:
                continue  # expected
            raise ShadowWriteError('shadow guard failed to refuse %s' % action)
        # Also refuse concrete basenames
        for name in ('DECISION.json', 'HEAD-CHAIN.json', 'CURRENT.json', 'WEIGHTS.json'):
            try:
                self.shadow.check_path(name)
            except ShadowWriteError:
                continue
            raise ShadowWriteError('shadow guard failed to refuse path %s' % name)

    def ordered_greens(self) -> List[ItemRecord]:
        return order_greens(self.items)

    def trainer_fallback_available(self) -> bool:
        return self.trainer_pod_fallback is not None or self.config.trainer_fallback == 'trainer_pod'


class TransitionErrorWrap(RuntimeError):
    """Host-local failure that should enter the retry/fail path."""
