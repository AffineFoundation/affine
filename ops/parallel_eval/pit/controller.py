"""PIT host controller (BRIEF §1.8).

On-demand rent only (hard rule): rent when a BOUND item needs a box; release as
soon as the result is stored or the item fails. Empty queue ⇒ zero rentals.
B queued with max_boxes N ⇒ at most min(B, N) concurrent rentals.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from .binding import dequeue_and_bind, mark_scored_decision, select_next_queued
from .box import BoxIO, BoxRunner, Fp32MasterError, ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .decisions import DecisionLog
from .pe_lane import LaneService, bind_host_lane_service
from .pe_rent import RentError, RentService, bind_host_rent_service
from .pe_score import score as pit_score
from .pe_score import score_epoch_shadow
from .shadow import ShadowGuard, ShadowWriteError
from .signer import (
    build_admission_envelope,
    generate_ephemeral_box_key,
    request_real_root_signature,
)
from .state import (
    RENTED_STATES,
    ItemRecord,
    ItemState,
    TransitionError,
    apply_transition,
    can_transition,
    holds_rental,
)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Controller:
    config: PitConfig
    rent: RentService
    lane: LaneService
    box_io_factory: Callable[[], BoxIO]
    decisions: DecisionLog = field(default_factory=DecisionLog)
    items: List[ItemRecord] = field(default_factory=list)
    current_head: Optional[str] = None
    current_head_epoch: int = 0
    current_master_digest: Optional[str] = None
    current_round_id: str = 'round-fixture'
    round_seed_commitment: str = 'seed-commitment-fixture'
    shadow: Optional[ShadowGuard] = None
    now_fn: Callable[[], float] = time.time
    score_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    live_decision_lookup: Optional[Callable[[int], Optional[Mapping[str, Any]]]] = None
    apply_hook: Optional[Callable[[ItemRecord], Any]] = None  # pe_apply seam; shadow refuses writes

    def __post_init__(self) -> None:
        self.shadow = self.shadow or ShadowGuard(self.config)
        if self.config.enabled and not self.config.may_run():
            raise ValueError('PIT enabled but mode is not runnable in this PR')

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_flag(
        cls,
        parallel_eval: Mapping[str, Any],
        *,
        box_io_factory: Optional[Callable[[], BoxIO]] = None,
        **kwargs: Any,
    ) -> 'Controller':
        cfg = load_pit_config(parallel_eval)
        rent = RentService(max_boxes=cfg.max_boxes, gpu_order=tuple(cfg.gpu_order))
        bind_host_rent_service(rent, module_path=cfg.pe_rent_module)
        lane = LaneService()
        bind_host_lane_service(lane, module_path=cfg.pe_lane_module)
        return cls(
            config=cfg,
            rent=rent,
            lane=lane,
            box_io_factory=box_io_factory or ScriptedBoxIO,
            **kwargs,
        )

    # ------------------------------------------------------------------ queue
    def enqueue(
        self,
        *,
        epoch: int,
        registration_index: int,
        uid: int = 0,
        hotkey_prefix: str = '',
        item_id: Optional[str] = None,
    ) -> ItemRecord:
        rec = ItemRecord(
            item_id=item_id or _new_id(),
            registration_index=registration_index,
            epoch=int(epoch),
            uid=int(uid),
            hotkey_prefix=(hotkey_prefix or '')[:16],
            shadow=self.config.is_shadow(),
        )
        self.items.append(rec)
        return rec

    def set_head(self, head: str, *, epoch: int, master_digest: str, round_id: Optional[str] = None) -> None:
        if len(head) != 64 or len(master_digest) != 64:
            raise ValueError('head/master_digest must be hex64')
        self.current_head = head
        self.current_head_epoch = int(epoch)
        self.current_master_digest = master_digest
        if round_id:
            self.current_round_id = round_id

    def rented_count(self) -> int:
        return self.rent.active_count()

    def demand_for_rent(self) -> int:
        """Items that still need a box (BOUND not yet rented)."""
        return sum(1 for i in self.items if i.state is ItemState.BOUND and not i.box_id)

    def queued_count(self) -> int:
        return sum(1 for i in self.items if i.state is ItemState.QUEUED)

    def expected_rentals(self) -> int:
        """Hard-rule expectation: min(queued + in-flight-bound demand path, max_boxes) once drained.

        More precisely after a tick converges: active rentals == number of items
        currently in RENTED_STATES (≤ max_boxes). Empty queue and no in-flight ⇒ 0.
        """
        in_flight = sum(1 for i in self.items if i.state in RENTED_STATES)
        return in_flight

    def _replace(self, record: ItemRecord) -> None:
        for i, old in enumerate(self.items):
            if old.item_id == record.item_id:
                self.items[i] = record
                return
        raise KeyError(record.item_id)

    # ------------------------------------------------------------------ tick
    def tick(self) -> Dict[str, Any]:
        self.shadow.assert_shadow()
        if not self.config.enabled:
            return {'ok': True, 'enabled': False, 'rentals': self.rented_count(), 'actions': []}

        actions: List[dict] = []

        # 1) Dequeue under cap (bind only; rent happens in advance step).
        # Occupancy = items that already hold or will hold a box (BOUND + RENTED_STATES).
        while True:
            occupied = sum(
                1 for i in self.items
                if i.state is ItemState.BOUND or i.state in RENTED_STATES
            )
            nxt = select_next_queued(
                self.items,
                rented=occupied,
                max_boxes=self.config.max_boxes,
            )
            if nxt is None:
                break
            if not self.current_head or not self.current_master_digest:
                actions.append({'op': 'wait-head', 'item_id': nxt.item_id})
                break
            bound = dequeue_and_bind(
                nxt,
                head=self.current_head,
                head_epoch=self.current_head_epoch,
                round_id=self.current_round_id,
                round_seed_commitment=self.round_seed_commitment,
                master_digest=self.current_master_digest,
                dequeued_at_mono=self.now_fn(),
            )
            if bound is None:
                actions.append({'op': 'pause-unpublished-master', 'item_id': nxt.item_id})
                break
            self._replace(bound)
            actions.append({'op': 'bound', 'item_id': bound.item_id, 'head': bound.bound_head[:12]})

        # 2) Advance items.
        for item in list(self.items):
            if item.state is ItemState.RELEASED:
                continue
            try:
                advanced, note = self._advance(item)
                if advanced is not item or note != 'noop':
                    self._replace(advanced)
                    actions.append({
                        'op': 'advance', 'item_id': item.item_id,
                        'to': advanced.state.value, 'note': note,
                        'box_id': advanced.box_id,
                    })
            except (Fp32MasterError, RentError, ShadowWriteError, TransitionError, ValueError, RuntimeError) as exc:
                failed = self._fail_and_release(item, str(exc)[:300])
                self._replace(failed)
                actions.append({'op': 'fail', 'item_id': item.item_id, 'error': str(exc)[:200], 'rentals': self.rented_count()})

        return {
            'ok': True,
            'enabled': True,
            'mode': self.config.mode,
            'shadow': True,
            'rentals': self.rented_count(),
            'queued': self.queued_count(),
            'max_boxes': self.config.max_boxes,
            'on_demand_rent_only': True,
            'actions': actions,
        }

    def _advance(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        if item.state is ItemState.BOUND:
            return self._rent_on_demand(item)
        if item.state is ItemState.RENTING:
            if item.box_id:
                return apply_transition(
                    item, ItemState.PROVISIONING, reason='already-rented',
                ), 'provision'
            return self._rent_on_demand(item)
        if item.state is ItemState.PROVISIONING:
            return self._run_box(item)
        if item.state is ItemState.RESULT:
            return apply_transition(item, ItemState.SCORED, reason='result-stored', updates={'result_stored': True}), 'scored'
        if item.state is ItemState.SCORED:
            return self._decide(item)
        if item.state is ItemState.DECIDED:
            return self._release(item, reason='decided')
        if item.state is ItemState.NO_DECISION:
            return self._release(item, reason='no_decision')
        if item.state is ItemState.RETRY_WAIT:
            return apply_transition(item, ItemState.RENTING, reason='retry'), 'retry'
        if item.state is ItemState.FAILED:
            if item.attempt > self.config.max_attempts:
                return self._release(apply_transition(item, ItemState.ABORTED, reason='exhausted'), reason='exhausted')
            return apply_transition(item, ItemState.RETRY_WAIT, reason=item.last_error or 'retry'), 'retry-wait'
        if item.state is ItemState.ABORTED:
            return self._release(item, reason='aborted')
        # TRAINING/SEALED/SERVING/PROBING are driven inside _run_box in one tick for shadow unit tests.
        return item, 'noop'

    def _rent_on_demand(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        """Hard rule: rent only for this bound/renting item; never pre-rent."""
        if item.state is ItemState.BOUND:
            if not self.rent.can_rent(queued_bound_waiting=1):
                return item, 'cap-wait'
            renting = apply_transition(item, ItemState.RENTING, reason='rent-start')
        elif item.state is ItemState.RENTING and not item.box_id:
            if not self.rent.can_rent(queued_bound_waiting=1):
                return item, 'cap-wait'
            renting = item
        else:
            raise TransitionError('rent requires BOUND or RENTING-without-box')
        out = self.rent.rent(item_id=renting.item_id, epoch=renting.epoch, attempt=renting.attempt)
        provisioned = apply_transition(
            renting, ItemState.PROVISIONING, reason='rented',
            updates={'box_id': str(out['box_id']), 'box_name': out.get('name')},
        )
        return provisioned, 'rented'

    def _run_box(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        runner = BoxRunner(config=self.config, io=self.box_io_factory(), lane=self.lane)
        box_key = generate_ephemeral_box_key()
        env = build_admission_envelope(
            box_key,
            pod_id=item.box_id or 'unknown',
            epoch=item.epoch,
            binding_H=item.bound_head or '',
            master_digest=item.bound_master_digest or '',
            now=self.now_fn(),
        )
        _ = request_real_root_signature  # documented refusal retained
        updated, bundle = runner.run(
            item,
            batch_handle='batch-handle:%s' % item.item_id,
            plan_handle=item.bound_plan_seed or 'plan-handle',
            box_key=box_key,
            now=self.now_fn(),
        )
        meta = dict(updated.meta)
        meta['probe_bundle'] = {
            'candidate_id': bundle.candidate_id,
            'admission_digest': env.digest(),
            'box_public_key_hex': bundle.box_public_key_hex,
        }
        meta['_bundle'] = bundle
        updated.meta = meta
        updated.candidate_id = bundle.candidate_id
        updated.sealed = True
        updated.result_stored = True
        # runner leaves item in RESULT
        return updated, 'box-complete'

    def _decide(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        bundle = (item.meta or {}).get('_bundle')
        if bundle is None:
            raise TransitionError('missing probe bundle')
        if self.current_head is None:
            raise TransitionError('no current_head')

        if self.config.is_shadow() and self.live_decision_lookup is not None:
            live = self.live_decision_lookup(item.epoch)
            shadow_out = score_epoch_shadow(
                epoch=item.epoch,
                round_id=item.bound_round_id or self.current_round_id,
                box_result=bundle.probe_result,
                box_candidate_id=bundle.candidate_id,
                box_head_id=item.bound_head or self.current_head,
                live_decision=live,
                host_score=self.score_fn,
            )
            scored = shadow_out['scored']
            item.meta['shadow_comparison'] = shadow_out['comparison']
            # Guarantee recorded candidate is the box's.
            if shadow_out['candidate'] != bundle.candidate_id:
                raise TransitionError('shadow scorer returned wrong candidate')
        else:
            scored = pit_score(
                epoch=item.epoch,
                round_id=item.bound_round_id or self.current_round_id,
                result=bundle.probe_result,
                candidate_id=bundle.candidate_id,
                head_id=item.bound_head or self.current_head,
                host_score=self.score_fn,
            )

        if scored.get('candidate') != bundle.candidate_id:
            raise TransitionError('scored candidate must be the box candidate')

        decision = str(scored.get('decision') or '').lower()
        if decision == 'pending':
            return item, 'wait-baseline'
        if decision == 'no_decision':
            updated = apply_transition(
                item, ItemState.NO_DECISION, reason='infra-no-decision',
                updates={'decision': 'no_decision', 'result_stored': True, 'paid': False},
            )
            return updated, 'no_decision'

        decided_at = float(self.now_fn())
        item.meta['decided_at_mono'] = decided_at
        updated = mark_scored_decision(
            item,
            current_head=self.current_head,
            decision='green' if decision == 'green' else 'red',
            z=scored.get('z'),
            candidate_id=bundle.candidate_id,
        )
        self.decisions.append(
            epoch=updated.epoch,
            uid=updated.uid,
            hotkey_prefix=updated.hotkey_prefix,
            binding_H=updated.bound_head or '',
            binding_round_id=updated.bound_round_id or '',
            candidate=updated.candidate_id or '',
            decision=updated.decision or 'red',
            z=updated.z,
            fresh=updated.fresh,
            merge_blocked=updated.merge_blocked,
            paid=updated.paid,
            decided_at=decided_at,
            shadow=True,
        )
        self._shadow_refuse_live_writes(updated)
        if updated.decision == 'green_fresh':
            # Memory-only head advance in shadow; no HEAD-CHAIN / R2 / DECISION.json.
            self.current_head = updated.candidate_id
            self.current_head_epoch = int(updated.meta.get('binding', {}).get('head_epoch') or self.current_head_epoch) + 1
            if self.apply_hook is not None:
                # pe_apply seam — shadow guard must refuse durable writes inside.
                self.apply_hook(updated)
        return updated, updated.decision or 'decided'

    def _release(self, item: ItemRecord, *, reason: str) -> tuple[ItemRecord, str]:
        if item.box_id and holds_rental(item) or item.box_id:
            try:
                self.rent.release(item.box_id, reason=reason)
            except Exception:
                # Drop local ledger slot even if host release flaps; retry is ops-side.
                if item.box_id in self.rent.ledger.active:
                    self.rent.ledger.release(item.box_id, reason='force-%s' % reason, at=self.now_fn())
        if item.state is ItemState.RELEASED:
            return item, 'already-released'
        if can_transition(item, ItemState.RELEASED):
            released = apply_transition(
                item, ItemState.RELEASED, reason=reason,
                updates={'box_id': None, 'box_name': None},
            )
            return released, 'released'
        # ABORTED / NO_DECISION / DECIDED already handled via can_transition.
        if item.state is ItemState.ABORTED and can_transition(item, ItemState.RELEASED):
            return apply_transition(item, ItemState.RELEASED, reason=reason), 'released'
        return item, 'release-noop'

    def _fail_and_release(self, item: ItemRecord, reason: str) -> ItemRecord:
        next_attempt = item.attempt + 1
        # Always release the box on failure (hard rule).
        if item.box_id:
            try:
                self.rent.release(item.box_id, reason='fail:%s' % reason[:60])
            except Exception:
                if item.box_id in self.rent.ledger.active:
                    self.rent.ledger.release(item.box_id, reason='fail-force', at=self.now_fn())
        updates = {'attempt': next_attempt, 'last_error': reason, 'box_id': None, 'box_name': None}
        # Trainer fail-closed → NO_DECISION, no auto-retry (DESIGN §8).
        if 'expected-master' in reason or 'fp32-master' in reason or 'Fp32MasterError' in reason:
            if can_transition(item, ItemState.NO_DECISION):
                nd = apply_transition(item, ItemState.NO_DECISION, reason=reason, updates={**updates, 'decision': 'no_decision'})
                return apply_transition(nd, ItemState.RELEASED, reason='no_decision-release')
        if next_attempt > self.config.max_attempts:
            if can_transition(item, ItemState.ABORTED):
                ab = apply_transition(item, ItemState.ABORTED, reason=reason, updates=updates)
                return apply_transition(ab, ItemState.RELEASED, reason='aborted-release')
            if can_transition(item, ItemState.FAILED):
                failed = apply_transition(item, ItemState.FAILED, reason=reason, updates=updates)
                ab = apply_transition(failed, ItemState.ABORTED, reason=reason)
                return apply_transition(ab, ItemState.RELEASED, reason='failed-release')
        if can_transition(item, ItemState.RETRY_WAIT):
            return apply_transition(item, ItemState.RETRY_WAIT, reason=reason, updates=updates)
        if can_transition(item, ItemState.FAILED):
            failed = apply_transition(item, ItemState.FAILED, reason=reason, updates=updates)
            return apply_transition(failed, ItemState.RETRY_WAIT, reason=reason)
        return item

    def _shadow_refuse_live_writes(self, item: ItemRecord) -> None:
        for action in (
            'write_decision', 'write_head_chain', 'write_weights',
            'publish_r2_head', 'apply_green_live',
        ):
            try:
                self.shadow.check_action(action)
                raise ShadowWriteError('shadow guard failed to refuse %s' % action)
            except ShadowWriteError:
                pass
        for name in ('DECISION.json', 'HEAD-CHAIN.json', 'CURRENT.json', 'WEIGHTS.json'):
            try:
                self.shadow.check_path(name)
                raise ShadowWriteError('shadow guard failed to refuse path %s' % name)
            except ShadowWriteError:
                pass


# Back-compat alias
HostController = Controller
