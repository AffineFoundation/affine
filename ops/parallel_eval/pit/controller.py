"""PIT host controller (BRIEF §1.8, SPEC §3–§4).

On-demand rent only: rent when a BOUND item needs a box; release as soon as the
result is stored or the item fails. Empty queue ⇒ zero item rentals.
B queued with max_boxes N ⇒ at most min(B, N) concurrent item rentals.

Live mode (Jacob 2026-11-09): batches dispatch from the host PIT queue (cap N);
each box binds current head at spawn. First GREEN_FRESH uploads C+M(C) and becomes
HEAD; later old-head GREEN is paid + stranded/merge_blocked; RED earns nothing;
non-GREEN destroys box state. Full redo with the same binding if a box is lost
after SEALED (SPEC D3). Lifecycle never waits on these decisions.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from .apply_live import ApplyLiveError, LiveApplicator, PromoteRefused
from .binding import dequeue_and_bind, mark_scored_decision, select_next_queued
from .box import BoxIO, BoxRunner, Fp32MasterError, ProbeBundle, ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .cutover import plan_cutover
from .decisions import DecisionLog
from .pe_lane import LaneService, bind_host_lane_service
from .pe_rent import RentError, RentService, bind_host_rent_service
from .pe_score import score as pit_score
from .pe_score import score_epoch_shadow
from .shadow import ShadowGuard, ShadowWriteError
from .signer import (
    RootSigner,
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
    apply_hook: Optional[Callable[[ItemRecord], Any]] = None
    live_applicator: Optional[LiveApplicator] = None
    root_signer: Optional[RootSigner] = None
    on_green_fresh: Optional[Callable[[ItemRecord], Any]] = None  # bench/prior/baseline hooks
    destroy_box_state_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    plan_seed_fn: Optional[Callable[[int], str]] = None
    indices_fn: Optional[Callable[[int], Any]] = None  # epoch → secret-draw indices (host-private)

    def __post_init__(self) -> None:
        self.shadow = self.shadow or ShadowGuard(self.config)
        if self.config.enabled and not self.config.may_run():
            raise ValueError('PIT enabled but mode is not runnable')
        if self.config.is_live() and self.live_applicator is None:
            self.live_applicator = LiveApplicator(
                config=self.config,
                accept_pit_decision=None,
                destroy_state=self.destroy_box_state_fn,
            )

    @classmethod
    def from_flag(
        cls,
        parallel_eval: Mapping[str, Any],
        *,
        box_io_factory: Optional[Callable[[], BoxIO]] = None,
        **kwargs: Any,
    ) -> 'Controller':
        cfg = load_pit_config(parallel_eval)
        rent = RentService(
            max_boxes=cfg.max_boxes,
            gpu_order=tuple(cfg.gpu_order),
            price_ceiling_usd_h=float(cfg.price_ceiling_usd_h),
            disable_auto_backup=bool(cfg.disable_lium_auto_backup),
        )
        bind_host_rent_service(rent, module_path=cfg.pe_rent_module)
        lane = LaneService()
        bind_host_lane_service(lane, module_path=cfg.pe_lane_module)
        if cfg.is_live():
            rent.require_host = True
            lane.require_host = True
            if box_io_factory is None:
                raise ValueError('live mode requires explicit box_io_factory (refuse ScriptedBoxIO default)')
            if box_io_factory is ScriptedBoxIO:
                raise ValueError('live mode refuses ScriptedBoxIO')
            if rent.host_rent is None:
                raise ValueError('live mode requires host pe_rent (bind failed)')
        else:
            box_io_factory = box_io_factory or ScriptedBoxIO
        return cls(
            config=cfg,
            rent=rent,
            lane=lane,
            box_io_factory=box_io_factory,
            **kwargs,
        )

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
        return sum(1 for i in self.items if i.state is ItemState.BOUND and not i.box_id)

    def queued_count(self) -> int:
        return sum(1 for i in self.items if i.state is ItemState.QUEUED)

    def expected_rentals(self) -> int:
        return sum(1 for i in self.items if i.state in RENTED_STATES)

    def cutover_plan(self) -> dict:
        return plan_cutover(self.config).as_dict()

    def _replace(self, record: ItemRecord) -> None:
        for i, old in enumerate(self.items):
            if old.item_id == record.item_id:
                self.items[i] = record
                return
        raise KeyError(record.item_id)

    def tick(self) -> Dict[str, Any]:
        if self.config.is_shadow():
            self.shadow.assert_shadow()
        if not self.config.enabled or self.config.revert:
            return {
                'ok': True, 'enabled': False, 'rentals': self.rented_count(),
                'actions': [], 'cutover': self.cutover_plan(),
            }

        actions: List[dict] = []

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
                round_seed_commitment=self.round_seed_commitment if self.plan_seed_fn is None else None,
                plan_seed_fn=self.plan_seed_fn,
                master_digest=self.current_master_digest,
                dequeued_at_mono=self.now_fn(),
            )
            if bound is None:
                actions.append({'op': 'pause-unpublished-master', 'item_id': nxt.item_id})
                break
            self._replace(bound)
            actions.append({'op': 'bound', 'item_id': bound.item_id, 'head': bound.bound_head[:12]})

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
            except PromoteRefused as exc:
                failed = self._fail_and_release(item, 'promote-refused:%s' % str(exc)[:200])
                self._replace(failed)
                actions.append({'op': 'no_decision', 'item_id': item.item_id, 'error': str(exc)[:200]})
            except (
                ApplyLiveError, Fp32MasterError, RentError, ShadowWriteError,
                TransitionError, TypeError, ValueError, RuntimeError,
            ) as exc:
                failed = self._fail_and_release(item, str(exc)[:300])
                self._replace(failed)
                actions.append({'op': 'fail', 'item_id': item.item_id, 'error': str(exc)[:200], 'rentals': self.rented_count()})

        return {
            'ok': True,
            'enabled': True,
            'mode': self.config.mode,
            'shadow': self.config.is_shadow(),
            'live': self.config.is_live(),
            'rentals': self.rented_count(),
            'queued': self.queued_count(),
            'max_boxes': self.config.max_boxes,
            'on_demand_rent_only': True,
            'price_ceiling_usd_h': self.config.price_ceiling_usd_h,
            'gpu_pinned': list(self.config.gpu_order),
            'cutover': self.cutover_plan(),
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
            return self._finish_decision(item)
        if item.state is ItemState.NO_DECISION:
            return self._destroy_and_release(item, reason='no_decision')
        if item.state is ItemState.RETRY_WAIT:
            # Full redo keeps binding after SEALED (SPEC D3); rebind only if unsealed.
            if item.sealed and self.config.full_redo_after_sealed:
                item.meta = dict(item.meta or {}, full_redo=True)
            return apply_transition(item, ItemState.RENTING, reason='retry'), 'retry'
        if item.state is ItemState.FAILED:
            if item.attempt > self.config.max_attempts:
                return self._destroy_and_release(
                    apply_transition(item, ItemState.ABORTED, reason='exhausted'),
                    reason='exhausted',
                )
            return apply_transition(item, ItemState.RETRY_WAIT, reason=item.last_error or 'retry'), 'retry-wait'
        if item.state is ItemState.ABORTED:
            return self._destroy_and_release(item, reason='aborted')
        return item, 'noop'

    def _rent_on_demand(self, item: ItemRecord) -> tuple[ItemRecord, str]:
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
        if not out.get('auto_backup_disabled', True) and self.config.disable_lium_auto_backup:
            raise RentError('lium auto-backup must be disabled')
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
        if self.root_signer is not None:
            cert = request_real_root_signature(env, signer=self.root_signer, issued_at_unix=self.now_fn())
            runner.root_pubkey_hex = self.root_signer.public_key_hex()
            runner.admit(box_key, cert, now=self.now_fn())
        meta0 = item.meta or {}
        # Prefer real K1 batch path from ingest. Synthetic batch-handle is only for
        # ScriptedBoxIO tests; live HostBoxIO.train_v834e refuses it (fail closed).
        batch_handle = (
            meta0.get('batch_path')
            or meta0.get('batch_handle')
            or meta0.get('batch')
            or ('batch-handle:%s' % item.item_id)
        )
        indices = None
        if self.indices_fn is not None:
            indices = list(self.indices_fn(int(item.epoch)))
        elif self.lane.require_host or self.lane.host_run_probe is not None:
            raise TransitionError('host run_probe requires indices_fn (pe_round_seed)')
        updated, bundle = runner.run(
            item,
            batch_handle=str(batch_handle),
            plan_handle=item.bound_plan_seed or 'plan-handle',
            box_key=box_key,
            now=self.now_fn(),
            indices=indices,
        )
        meta = dict(updated.meta)
        meta['probe_bundle'] = {
            'item_id': bundle.item_id,
            'candidate_id': bundle.candidate_id,
            'bound_head': bundle.bound_head,
            'bound_master_digest': bundle.bound_master_digest,
            'probe_result': dict(bundle.probe_result),
            'master_upload': dict(bundle.master_upload),
            'train_receipt': dict(bundle.train_receipt),
            'box_public_key_hex': bundle.box_public_key_hex,
            'admission_digest': env.digest(),
        }
        meta['candidate_path'] = (bundle.train_receipt or {}).get('candidate_path')
        meta['candidate_master_path'] = (bundle.train_receipt or {}).get('candidate_master_path')
        meta['candidate_master_digest'] = (bundle.train_receipt or {}).get('candidate_master_digest')
        meta['candidate_sha256'] = (bundle.train_receipt or {}).get('candidate_sha256') or bundle.candidate_id
        updated.meta = meta
        updated.candidate_id = bundle.candidate_id
        updated.sealed = True
        updated.result_stored = True
        return updated, 'box-complete'

    def _probe_bundle_from_item(self, item: ItemRecord) -> ProbeBundle:
        meta = item.meta or {}
        raw = meta.get('_bundle')
        if isinstance(raw, ProbeBundle):
            return raw
        snap = meta.get('probe_bundle')
        if not isinstance(snap, Mapping):
            raise TransitionError('missing probe bundle')
        return ProbeBundle(
            item_id=str(snap.get('item_id') or item.item_id),
            bound_head=str(snap.get('bound_head') or item.bound_head or ''),
            bound_master_digest=str(snap.get('bound_master_digest') or item.bound_master_digest or ''),
            candidate_id=str(snap.get('candidate_id') or item.candidate_id or ''),
            probe_result=dict(snap.get('probe_result') or {}),
            master_upload=dict(snap.get('master_upload') or {}),
            train_receipt=dict(snap.get('train_receipt') or {}),
            box_public_key_hex=str(snap.get('box_public_key_hex') or ''),
            meta=dict(snap),
        )

    def _staleness_reference_head(self) -> str:
        if self.current_head is None:
            raise TransitionError('no current_head')
        if not self.config.is_live():
            return self.current_head
        leaders: List[tuple[float, str]] = []
        for rec in self.items:
            if rec.decision != 'green_fresh' or not rec.candidate_id:
                continue
            if rec.state is not ItemState.DECIDED:
                continue
            gat = (rec.meta or {}).get('decided_at_mono')
            if gat is None:
                continue
            leaders.append((float(gat), rec.candidate_id))
        if not leaders:
            return self.current_head
        leaders.sort(key=lambda row: row[0])
        return leaders[0][1]

    def _decide(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        bundle = self._probe_bundle_from_item(item)
        if self.current_head is None:
            raise TransitionError('no current_head')

        probe = bundle.probe_result or {}
        if probe.get('partial') or probe.get('timeout') or probe.get('incomplete'):
            updated = apply_transition(
                item, ItemState.NO_DECISION, reason='incomplete-probe',
                updates={'decision': 'no_decision', 'result_stored': True, 'paid': False},
            )
            return updated, 'no_decision-incomplete'

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
            if shadow_out['candidate'] != bundle.candidate_id:
                raise TransitionError('shadow scorer returned wrong candidate')
        else:
            if self.config.is_live() and self.score_fn is None:
                raise TransitionError('live decide requires host score_fn')
            scored = pit_score(
                epoch=item.epoch,
                round_id=item.bound_round_id or self.current_round_id,
                result=bundle.probe_result,
                candidate_id=bundle.candidate_id,
                head_id=item.bound_head or self.current_head,
                host_score=self.score_fn,
                require_host=self.config.is_live(),
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
        staleness_head = self._staleness_reference_head()
        updated = mark_scored_decision(
            item,
            current_head=staleness_head,
            decision='green' if decision == 'green' else 'red',
            z=scored.get('z'),
            candidate_id=bundle.candidate_id,
        )
        # Hard rule: REDs earn nothing.
        if updated.decision == 'red' and updated.paid:
            raise TransitionError('RED must not be paid (last-10-GREEN rule)')

        # Keep host score (incl. heldout) for gate emit after verify.
        updated.meta = dict(updated.meta or {}, scored=dict(scored))
        # Live: defer DecisionLog until finish/verify so green-time order stays
        # consistent and unpaid greens never land in the sealed log.
        # Shadow seals immediately (no upload verify gate).
        if self.config.is_shadow():
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
                self.current_head = updated.candidate_id
                self.current_head_epoch = int(updated.meta.get('binding', {}).get('head_epoch') or self.current_head_epoch) + 1
                if self.apply_hook is not None:
                    self.apply_hook(updated)
        else:
            updated.meta['decision_log_pending'] = True
        return updated, updated.decision or 'decided'

    def _seal_decision_log(self, item: ItemRecord) -> None:
        if not (item.meta or {}).get('decision_log_pending'):
            return
        self.decisions.append(
            epoch=item.epoch,
            uid=item.uid,
            hotkey_prefix=item.hotkey_prefix,
            binding_H=item.bound_head or '',
            binding_round_id=item.bound_round_id or '',
            candidate=item.candidate_id or '',
            decision=item.decision or 'red',
            z=item.z,
            fresh=item.fresh,
            merge_blocked=item.merge_blocked,
            paid=item.paid,
            decided_at=float((item.meta or {}).get('decided_at_mono') or self.now_fn()),
            shadow=False,
        )
        item.meta = dict(item.meta or {}, decision_log_pending=False)

    def _finish_decision(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        if self.config.is_live() and self.live_applicator is not None:
            scored = dict((item.meta or {}).get('scored') or {'decision': item.decision})
            try:
                result = self.live_applicator.apply(item, scored=scored)
            except ApplyLiveError as exc:
                item.meta = dict(item.meta or {}, live_apply_error=str(exc)[:300])
                released = apply_transition(
                    item,
                    ItemState.RELEASED,
                    reason='apply-failed:%s' % str(exc)[:120],
                    updates={
                        'decision': 'no_decision', 'paid': False, 'fresh': None,
                        'merge_blocked': False, 'box_id': None, 'box_name': None,
                    },
                )
                # Seal as no_decision only after verify failed — never a paid GREEN.
                released.meta = dict(released.meta or {}, decision_log_pending=True, gate_emit_ready=True)
                self._seal_decision_log(released)
                return released, 'apply-failed'
            item.meta['live_apply'] = result.as_dict()
            item.meta['gate_emit_ready'] = True
            self._seal_decision_log(item)
            if result.head_advanced and item.candidate_id:
                self.current_head = item.candidate_id
                self.current_head_epoch = int(self.current_head_epoch) + 1
                if self.on_green_fresh is not None:
                    self.on_green_fresh(item)
            if result.destroyed_box_state or item.decision != 'green_fresh':
                return self._destroy_and_release(item, reason=item.decision or 'decided')
            return self._release(item, reason='green_fresh-applied')
        # Shadow: just release; state already destroyed conceptually.
        if item.decision and item.decision != 'green_fresh':
            return self._destroy_and_release(item, reason=item.decision)
        return self._release(item, reason='decided')

    def _destroy_and_release(self, item: ItemRecord, *, reason: str) -> tuple[ItemRecord, str]:
        if self.destroy_box_state_fn is not None and item.box_id:
            try:
                self.destroy_box_state_fn(
                    box_id=item.box_id, candidate_id=item.candidate_id, reason=reason,
                )
            except Exception as exc:
                item.meta = dict(item.meta or {}, destroy_error=str(exc)[:200])
        return self._release(item, reason=reason)

    def _release(self, item: ItemRecord, *, reason: str) -> tuple[ItemRecord, str]:
        if item.box_id and holds_rental(item) or item.box_id:
            try:
                self.rent.release(item.box_id, reason=reason)
            except Exception:
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
        if item.state is ItemState.ABORTED and can_transition(item, ItemState.RELEASED):
            return apply_transition(item, ItemState.RELEASED, reason=reason), 'released'
        return item, 'release-noop'

    def _fail_and_release(self, item: ItemRecord, reason: str) -> ItemRecord:
        if item.state is ItemState.DECIDED:
            item.last_error = reason
            return self._destroy_and_release(item, reason=reason)
        next_attempt = item.attempt + 1
        if item.box_id:
            try:
                self.rent.release(item.box_id, reason='fail:%s' % reason[:60])
            except Exception:
                if item.box_id in self.rent.ledger.active:
                    self.rent.ledger.release(item.box_id, reason='fail-force', at=self.now_fn())
        updates = {'attempt': next_attempt, 'last_error': reason, 'box_id': None, 'box_name': None}
        if 'expected-master' in reason or 'fp32-master' in reason or 'Fp32MasterError' in reason or 'promote-refused' in reason:
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


HostController = Controller
