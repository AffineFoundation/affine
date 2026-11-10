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

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import PREFIX_LIVE, PREFIX_SHADOW
from .apply_live import ApplyLiveError, LiveApplicator, PromoteRefused
from .binding import dequeue_and_bind, mark_scored_decision, select_next_queued
from .box import BoxIO, BoxRunner, Fp32MasterError, ProbeBundle, ScriptedBoxIO
from .config import PitConfig, load_pit_config
from .cutover import plan_cutover
from .cutover_apply import (
    confirm_head_advanced,
    drive_cutover_apply,
    master_index_is_complete,
)
from .decisions import DecisionLog
from .durable import item_from_dict, item_to_dict, pit_state_root
from .pe_lane import LaneService, bind_host_lane_service
from .pe_rent import RentError, RentService, bind_host_rent_service
from .pe_score import score as pit_score
from .pe_score import score_epoch_shadow
from .queue_ingest import training_job_checkpoint
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


# SSH/transport re-adopt after restart: retry before releasing a still-listed box.
READOPT_MAX_ATTEMPTS = 3
# GREEN apply (upload/verify) failure: keep the box and retry with backoff.
GREEN_APPLY_BACKOFF_S = 30.0
GREEN_APPLY_BACKOFF_CAP_S = 300.0


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
    # Prod: never default to a fixture round_id. Bind from SAMPLE.private.json
    # via round_id_fn (host-side only; never sent to the box).
    current_round_id: str = ''
    round_seed_commitment: str = ''
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
    round_id_fn: Optional[Callable[[int], str]] = None  # epoch → SAMPLE.private.json round_id
    durable_save_fn: Optional[Callable[[ItemRecord], None]] = None
    _workers: Dict[str, threading.Thread] = field(default_factory=dict, init=False, repr=False)
    _worker_out: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _worker_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

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
        shadow = bool(cfg.is_shadow())
        rent = RentService(
            max_boxes=cfg.max_boxes,
            gpu_order=tuple(cfg.gpu_order),
            price_ceiling_usd_h=float(cfg.price_ceiling_usd_h),
            disable_auto_backup=bool(cfg.disable_lium_auto_backup),
            shadow=shadow,
            name_prefix=PREFIX_SHADOW if shadow else PREFIX_LIVE,
            share_pe_cap=not shadow,
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

    def _round_id_for_item(self, item: ItemRecord) -> str:
        """Host-side SAMPLE.private.json round_id for the item's epoch.

        Never invent a fixture. Never send the raw SAMPLE/ROUND to the box —
        only the round_id string is bound onto the item for pe_score.
        """
        rid = ''
        if self.round_id_fn is not None:
            try:
                rid = str(self.round_id_fn(int(item.epoch)) or '')
            except Exception as exc:
                raise ValueError(
                    'round_id_fn failed for epoch %s: %s' % (item.epoch, str(exc)[:200])
                ) from exc
        if not rid:
            rid = str(self.current_round_id or '')
        if not rid or rid in ('round-fixture', 'fixture', 'seed-commitment-fixture'):
            raise ValueError(
                'refuse bind: SAMPLE.private.json round_id missing for epoch %s '
                '(no fixture default on prod paths)' % item.epoch
            )
        return rid

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
            # Free box slots only: count every item that still needs or holds a
            # slot (BOUND / in-rental / retryable FAILED / RETRY_WAIT). Price
            # failures stay waiting so they do not free a slot for a newer bind.
            occupied = sum(1 for i in self.items if self._occupies_box_slot(i))
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
            try:
                round_id = self._round_id_for_item(nxt)
            except ValueError as exc:
                actions.append({
                    'op': 'refuse-bind-round-id',
                    'item_id': nxt.item_id,
                    'epoch': nxt.epoch,
                    'error': str(exc)[:200],
                })
                break
            bound = dequeue_and_bind(
                nxt,
                head=self.current_head,
                head_epoch=self.current_head_epoch,
                round_id=round_id,
                round_seed_commitment=(
                    self.round_seed_commitment if self.plan_seed_fn is None else None
                ),
                plan_seed_fn=self.plan_seed_fn,
                master_digest=self.current_master_digest,
                dequeued_at_mono=self.now_fn(),
            )
            if bound is None:
                actions.append({'op': 'pause-unpublished-master', 'item_id': nxt.item_id})
                break
            self._replace(bound)
            actions.append({
                'op': 'bound',
                'item_id': bound.item_id,
                'head': bound.bound_head[:12],
                'round_id': bound.bound_round_id,
            })

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
                failed = self._fail_and_release(
                    item, 'promote-refused:%s' % str(exc)[:200],
                )
                self._replace(failed)
                actions.append({
                    'op': 'no_decision', 'item_id': item.item_id,
                    'error': str(exc)[:200],
                })
            except Exception as exc:
                # Isolate per item: one failure must never abort the tick for peers
                # (e.g. e1056 still evaluating while e1055 post-decision retries).
                failed = self._fail_and_release(item, str(exc)[:300])
                self._replace(failed)
                actions.append({
                    'op': 'fail', 'item_id': item.item_id,
                    'error': str(exc)[:200], 'rentals': self.rented_count(),
                    'exc_type': type(exc).__name__,
                })

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
            return self._run_box_async(item)
        if item.state in (ItemState.TRAINING, ItemState.SEALED, ItemState.SERVING, ItemState.PROBING):
            # Worker in flight or re-adopted after restart — wait / resume.
            return self._poll_box_worker(item)
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

    def _occupies_box_slot(self, item: ItemRecord) -> bool:
        """True when this item still consumes a max_boxes slot (blocks new binds)."""
        if item.state is ItemState.BOUND:
            return True
        if item.state in RENTED_STATES:
            return True
        if item.state is ItemState.RETRY_WAIT:
            return True
        if item.state is ItemState.FAILED and item.attempt <= self.config.max_attempts:
            return True
        return False

    def _is_rent_waiter(self, item: ItemRecord) -> bool:
        """Oldest-first rent queue: bound / renting-without-box / retryable waiters."""
        if item.state is ItemState.BOUND:
            return True
        if item.state is ItemState.RENTING and not item.box_id:
            return True
        if item.state is ItemState.RETRY_WAIT:
            return True
        if item.state is ItemState.FAILED and item.attempt <= self.config.max_attempts:
            return True
        return False

    def _has_older_rent_waiter(self, item: ItemRecord) -> bool:
        waiters = [i for i in self.items if self._is_rent_waiter(i)]
        waiters.sort(key=lambda i: (i.registration_index, i.epoch, i.item_id))
        for w in waiters:
            if w.item_id == item.item_id:
                return False
            return True
        return False

    @staticmethod
    def _is_price_rent_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            'ceiling' in msg
            or 'above ceiling' in msg
            or 'no eligible b300' in msg
            or 'no b300 offer' in msg
        )

    def _rent_on_demand(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        if item.state is ItemState.BOUND:
            # Stale-head: TRAINING-JOB checkpoint must equal current head — else
            # skip with no rental (item trained against an obsolete head).
            if self.current_head:
                job_ck = training_job_checkpoint(int(item.epoch))
                if job_ck and job_ck != self.current_head:
                    meta = dict(item.meta or {}, stale_head_skip=True,
                                job_checkpoint=job_ck, current_head=self.current_head)
                    skipped = apply_transition(
                        item, ItemState.NO_DECISION, reason='stale-head-skip',
                        updates={
                            'decision': 'no_decision', 'paid': False,
                            'result_stored': False, 'meta': meta,
                        },
                    )
                    return skipped, 'stale-head-skip'
            if self._has_older_rent_waiter(item):
                return item, 'rent-wait-older'
            if not self.rent.can_rent(queued_bound_waiting=1):
                return item, 'cap-wait'
            renting = apply_transition(item, ItemState.RENTING, reason='rent-start')
        elif item.state is ItemState.RENTING and not item.box_id:
            if self._has_older_rent_waiter(item):
                return item, 'rent-wait-older'
            if not self.rent.can_rent(queued_bound_waiting=1):
                return item, 'cap-wait'
            renting = item
        else:
            raise TransitionError('rent requires BOUND or RENTING-without-box')
        try:
            out = self.rent.rent(item_id=renting.item_id, epoch=renting.epoch, attempt=renting.attempt)
        except (RentError, AssertionError, RuntimeError, ValueError, TypeError) as exc:
            # Per-item catch: a bad rent name/cap/offer must not crash the tick loop.
            # Price / offer-ceiling failures: stay RENTING without a box, do not
            # burn attempts, and keep the slot so newer items cannot jump ahead.
            if self._is_price_rent_error(exc):
                renting.last_error = 'rent:%s' % str(exc)[:240]
                meta = dict(renting.meta or {})
                meta['rent_price_wait'] = True
                meta['rent_price_error'] = str(exc)[:200]
                renting.meta = meta
                return renting, 'rent-price-wait'
            failed = apply_transition(
                renting, ItemState.FAILED, reason='rent-error',
                updates={'last_error': 'rent:%s' % str(exc)[:240]},
            )
            return failed, 'rent-error'
        if not out.get('auto_backup_disabled', True) and self.config.disable_lium_auto_backup:
            raise RentError('lium auto-backup must be disabled')
        meta = dict(renting.meta or {})
        gpu = out.get('gpu') or out.get('gpu_model') or out.get('gpu_name')
        if gpu:
            meta['train_gpu'] = str(gpu)
            meta['eval_gpu'] = str(gpu)  # same box trains then probes
        # Persist the full pe_rent dict so pe_lane.serve/run_probe get a real pod handle.
        meta['pod'] = {
            k: out.get(k) for k in (
                'id', 'box_id', 'name', 'ip', 'port', 'huid', 'gpu', 'price_h',
                'price_usd_h', 'price_per_hour', 'ssh_command', 'executor',
            ) if out.get(k) is not None
        }
        meta['pod']['id'] = meta['pod'].get('id') or out.get('box_id')
        provisioned = apply_transition(
            renting, ItemState.PROVISIONING, reason='rented',
            updates={
                'box_id': str(out['box_id']),
                'box_name': out.get('name'),
                'meta': meta,
            },
        )
        return provisioned, 'rented'


    def _persist(self, item: ItemRecord) -> None:
        if self.durable_save_fn is not None:
            try:
                self.durable_save_fn(item)
            except Exception:
                pass

    def _worker_result_path(self, item_id: str) -> Path:
        return pit_state_root() / 'items' / ('%s.worker-result.json' % item_id)

    def _save_worker_result(self, item_id: str, payload: Mapping[str, Any]) -> Path:
        """Atomically persist finished train+eval before advancing to decided."""
        path = self._worker_result_path(item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = dict(payload)
        item = doc.get('item')
        if isinstance(item, ItemRecord):
            doc['item'] = item_to_dict(item)
        tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + '\n')
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        return path

    def _load_worker_result(self, item_id: str) -> Optional[dict]:
        path = self._worker_result_path(item_id)
        try:
            raw = json.loads(path.read_text())
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        item_raw = raw.get('item')
        if isinstance(item_raw, dict):
            raw = dict(raw)
            raw['item'] = item_from_dict(item_raw)
        return raw

    def _run_box_async(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        """Start one worker thread per item (up to cap); durable so restarts re-adopt.

        Scripted / non-host lanes stay synchronous so unit tests remain deterministic.
        """
        if not self.lane.require_host and self.lane.host_run_probe is None:
            return self._run_box(item)
        with self._worker_lock:
            existing = self._workers.get(item.item_id)
            if existing is not None and existing.is_alive():
                return item, 'box-worker-running'
            done = self._worker_out.pop(item.item_id, None)
        if done is not None:
            return self._apply_worker_result(item, done)
        disk = self._load_worker_result(item.item_id)
        if disk is not None:
            return self._apply_worker_result(item, disk)
        # Snapshot stays PROVISIONING for BoxRunner; controller view moves to TRAINING.
        snapshot = item
        working = apply_transition(item, ItemState.TRAINING, reason='box-worker-start')
        working.meta = dict(working.meta or {}, worker_started_at=self.now_fn(), worker_box_id=working.box_id)
        self._persist(working)

        def _target(snap: ItemRecord) -> None:
            try:
                # BoxRunner requires PROVISIONING at entry.
                if snap.state is not ItemState.PROVISIONING:
                    snap = apply_transition(snap, ItemState.PROVISIONING, reason='worker-entry')
                updated, note = self._run_box(snap)
                payload = dict(ok=True, item=updated, note=note)
                # Persist BEFORE in-memory handoff so a restart never re-runs train+probe.
                try:
                    self._save_worker_result(snap.item_id, payload)
                except Exception as persist_exc:
                    payload = dict(
                        ok=False,
                        error='worker-result-persist:%s' % str(persist_exc)[:200],
                        item_id=snap.item_id,
                    )
            except Exception as exc:
                payload = dict(ok=False, error=str(exc)[:400], item_id=snap.item_id)
            with self._worker_lock:
                self._worker_out[snap.item_id] = payload
                self._workers.pop(snap.item_id, None)

        thr = threading.Thread(
            target=_target, args=(snapshot,),
            name='pit-box-%s' % working.item_id[:12], daemon=True,
        )
        with self._worker_lock:
            self._workers[working.item_id] = thr
        thr.start()
        return working, 'box-worker-started'

    def _poll_box_worker(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        with self._worker_lock:
            thr = self._workers.get(item.item_id)
            done = self._worker_out.pop(item.item_id, None)
        if done is not None:
            return self._apply_worker_result(item, done)
        if thr is not None and thr.is_alive():
            return item, 'box-worker-running'
        disk = self._load_worker_result(item.item_id)
        if disk is not None:
            return self._apply_worker_result(item, disk)
        # No live worker and no durable result after restart: re-adopt the box
        # and re-run train+probe.
        if item.box_id and (item.meta or {}).get('pod'):
            meta = dict(item.meta or {})
            meta.pop('probe_bundle', None)
            item.meta = meta
            item.sealed = False
            item.result_stored = False
            item.candidate_id = None
            if item.state is not ItemState.PROVISIONING:
                try:
                    item = apply_transition(item, ItemState.PROVISIONING, reason='readopt-worker')
                except TransitionError:
                    # Force via snapshot fields when mid-pipeline state can't step back.
                    item.state = ItemState.PROVISIONING
            return self._run_box_async(item)
        return item, 'box-worker-wait'

    def _box_still_in_lium(self, item: ItemRecord) -> bool:
        """True when the item's box is still visible in lium (or listing unavailable)."""
        if not item.box_id:
            return False
        if self.rent.list_pods is None:
            # No listing hook: treat pod meta as still live so we retry rather than
            # release a potentially billing box after a transient SSH blip.
            return bool((item.meta or {}).get('pod'))
        try:
            rows = list(self.rent.list_pods() or [])
        except Exception:
            return True
        box_id = str(item.box_id)
        box_name = str(item.box_name or '')
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            bid = str(row.get('id') or row.get('box_id') or '')
            name = str(row.get('name') or row.get('pod_name') or '')
            if bid and bid == box_id:
                return True
            if name and (name == box_id or (box_name and name == box_name)):
                return True
        return False

    @staticmethod
    def _is_readopt_transport_error(err: str) -> bool:
        msg = (err or '').lower()
        return any(
            tok in msg
            for tok in (
                'rc=255', 'ssh', 'connection refused', 'connection reset',
                'broken pipe', 'timed out', 'timeout', 'no route',
                'network is unreachable', 'readopt',
            )
        )

    def _maybe_retry_readopt(
        self, item: ItemRecord, err: str,
    ) -> Optional[tuple[ItemRecord, str]]:
        """Retry a failed re-adopt a few times when the box is still in lium."""
        if not self._is_readopt_transport_error(err):
            return None
        if not item.box_id:
            return None
        attempts = int((item.meta or {}).get('readopt_attempts') or 0)
        if attempts >= READOPT_MAX_ATTEMPTS:
            return None
        if not self._box_still_in_lium(item):
            return None
        meta = dict(item.meta or {})
        meta['readopt_attempts'] = attempts + 1
        meta['last_readopt_error'] = (err or '')[:200]
        meta.pop('probe_bundle', None)
        item.meta = meta
        item.sealed = False
        item.result_stored = False
        item.candidate_id = None
        if item.state is not ItemState.PROVISIONING:
            try:
                item = apply_transition(item, ItemState.PROVISIONING, reason='readopt-retry')
            except TransitionError:
                item.state = ItemState.PROVISIONING
        self._persist(item)
        return item, 'readopt-retry'

    def _apply_worker_result(self, item: ItemRecord, done: Mapping[str, Any]) -> tuple[ItemRecord, str]:
        if not done.get('ok'):
            err = 'box-worker:%s' % (done.get('error') or 'failed')
            retry = self._maybe_retry_readopt(item, err)
            if retry is not None:
                return retry
            return self._fail_and_release(item, err), 'box-worker-fail'
        updated = done.get('item')
        if not isinstance(updated, ItemRecord):
            return self._fail_and_release(item, 'box-worker:bad-result'), 'box-worker-fail'
        # Ensure durable result exists before any decided transition consumes it.
        try:
            self._save_worker_result(item.item_id, done)
        except Exception as exc:
            return (
                self._fail_and_release(item, 'worker-result-persist:%s' % str(exc)[:200]),
                'box-worker-fail',
            )
        # Successful worker clears prior re-adopt retry counters.
        if (updated.meta or {}).get('readopt_attempts'):
            meta = dict(updated.meta or {})
            meta.pop('readopt_attempts', None)
            meta.pop('last_readopt_error', None)
            updated.meta = meta
        self._persist(updated)
        return updated, str(done.get('note') or 'box-complete')


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
            train_gpu=(item.meta or {}).get('train_gpu'),
            eval_gpu=(item.meta or {}).get('eval_gpu'),
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
            'training_report': dict(bundle.training_report or {}),
            'train_gpu': bundle.train_gpu,
            'eval_gpu': bundle.eval_gpu,
            'admission_digest': env.digest(),
        }
        receipt = bundle.train_receipt or {}
        meta['candidate_path'] = receipt.get('candidate_path')
        meta['candidate_on_box'] = bool(receipt.get('candidate_on_box') or receipt.get('defer_candidate_upload'))
        meta['box_candidate_dir'] = receipt.get('box_candidate_dir')
        meta['box_master_dir'] = receipt.get('box_master_dir')
        meta['candidate_files'] = dict(receipt.get('candidate_files') or {})
        meta['defer_candidate_upload'] = bool(receipt.get('defer_candidate_upload'))
        meta['candidate_master_path'] = receipt.get('candidate_master_path')
        meta['candidate_master_digest'] = receipt.get('candidate_master_digest')
        meta['candidate_sha256'] = receipt.get('candidate_sha256') or bundle.candidate_id
        # Persist training report + GPUs on the item before any release path.
        meta['training_report'] = dict(bundle.training_report or {})
        meta['train_gpu'] = bundle.train_gpu
        meta['eval_gpu'] = bundle.eval_gpu
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
            training_report=dict(snap.get('training_report') or meta.get('training_report') or {}),
            train_gpu=snap.get('train_gpu') or meta.get('train_gpu'),
            eval_gpu=snap.get('eval_gpu') or meta.get('eval_gpu'),
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
                box_report=bundle.training_report or (item.meta or {}).get('training_report'),
                live_report=(live or {}).get('training_report') if isinstance(live, Mapping) else None,
                box_meta={
                    'round_id': item.bound_round_id or self.current_round_id,
                    'plan_seed': item.bound_plan_seed,
                    'train_gpu': bundle.train_gpu or (item.meta or {}).get('train_gpu'),
                    'eval_gpu': bundle.eval_gpu or (item.meta or {}).get('eval_gpu'),
                    'training_report': bundle.training_report,
                },
                live_meta=dict(live or {}) if isinstance(live, Mapping) else None,
                train_gpu=bundle.train_gpu or (item.meta or {}).get('train_gpu'),
                eval_gpu=bundle.eval_gpu or (item.meta or {}).get('eval_gpu'),
            )
            scored = shadow_out['scored']
            item.meta['shadow_comparison'] = shadow_out['comparison']
            item.meta['shadow_pass'] = shadow_out.get('shadow_pass')
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


    def _green_apply_backoff_s(self, item: ItemRecord) -> float:
        n = int((item.meta or {}).get('apply_retry_count') or 0)
        return float(min(GREEN_APPLY_BACKOFF_CAP_S, GREEN_APPLY_BACKOFF_S * (2 ** max(0, n))))

    def _alert_green_apply_failure(self, item: ItemRecord, reason: str) -> Optional[Path]:
        """Loud durable alert: never silently drop a GREEN that still owns a box."""
        doc = {
            'ok': False,
            'event': 'green_apply_failure',
            'at_unix': self.now_fn(),
            'item_id': item.item_id,
            'epoch': item.epoch,
            'candidate_id': item.candidate_id,
            'box_id': item.box_id,
            'decision': item.decision,
            'error': reason[:800],
            'apply_retry_count': int((item.meta or {}).get('apply_retry_count') or 0),
            'apply_retry_at': (item.meta or {}).get('apply_retry_at'),
            'candidate_on_box': bool((item.meta or {}).get('candidate_on_box')),
            'defer_candidate_upload': bool((item.meta or {}).get('defer_candidate_upload')),
        }
        alert: Optional[Path] = None
        try:
            root = pit_state_root()
            root.mkdir(parents=True, exist_ok=True)
            alert = root / 'GREEN-APPLY-ALERT.json'
            tmp = alert.with_name('.%s.%d.tmp' % (alert.name, os.getpid()))
            tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + '\n')
            os.chmod(tmp, 0o600)
            os.replace(tmp, alert)
        except Exception as exc:
            doc['alert_write_error'] = str(exc)[:200]
        print(
            json.dumps({
                'component': 'pit-controller',
                'level': 'error',
                'event': 'green_apply_failure',
                'item_id': item.item_id,
                'epoch': item.epoch,
                'box_id': item.box_id,
                'error': reason[:400],
                'alert': str(alert) if alert is not None else None,
            }, default=str),
            flush=True,
        )
        return alert

    def _keep_green_for_retry(self, item: ItemRecord, reason: str) -> ItemRecord:
        """Stay DECIDED with box held; schedule backoff retry. Never silent-drop."""
        meta = dict(item.meta or {})
        retries = int(meta.get('apply_retry_count') or 0) + 1
        delay = float(min(
            GREEN_APPLY_BACKOFF_CAP_S,
            GREEN_APPLY_BACKOFF_S * (2 ** max(0, retries - 1)),
        ))
        meta.update({
            'live_apply_error': reason[:300],
            'apply_retry_count': retries,
            'apply_retry_at': float(self.now_fn()) + delay,
            'gate_emit_ready': False,
            'green_apply_pending': True,
        })
        item.meta = meta
        item.last_error = reason[:300]
        self._alert_green_apply_failure(item, reason)
        self._persist(item)
        return item

    def _drive_pending_head(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        """Drive pending → APPLY ready → record_applied for a queued GREEN.

        ``head_advanced`` / box release only after HEAD-CHAIN append is verified.
        Incomplete fp32 master index blocks apply and keeps the box.
        """
        cand = item.candidate_id or ''
        if not cand or not master_index_is_complete(cand):
            kept = self._keep_green_for_retry(
                item, 'incomplete-fp32-master:refuse-head-apply',
            )
            # Keep gate_emit_ready if DECISION already landed; still block HEAD.
            if ((item.meta or {}).get('live_apply') or {}).get('candidate_verified'):
                kept.meta = dict(kept.meta or {}, gate_emit_ready=True, green_apply_pending=True)
                self._persist(kept)
            return kept, 'green-master-incomplete'

        def _finalize_advanced() -> tuple[ItemRecord, str]:
            item.meta = dict(item.meta or {})
            item.meta.pop('green_apply_pending', None)
            item.meta['head_advanced'] = True
            live = dict(item.meta.get('live_apply') or {})
            live['head_advanced'] = True
            item.meta['live_apply'] = live
            self.current_head = cand
            self.current_head_epoch = int(self.current_head_epoch) + 1
            if self.on_green_fresh is not None:
                self.on_green_fresh(item)
            self._persist(item)
            return self._release(item, reason='green_fresh-applied')

        if confirm_head_advanced(cand):
            return _finalize_advanced()
        try:
            drive_cutover_apply(
                alert_fn=lambda doc: self._alert_green_apply_failure(
                    item, str(doc.get('event') or 'cutover-apply'),
                ),
            )
        except Exception as exc:
            if confirm_head_advanced(cand):
                return _finalize_advanced()
            kept = self._keep_green_for_retry(
                item, 'cutover-apply:%s' % str(exc)[:200],
            )
            return kept, 'green-head-drive-error'
        if not confirm_head_advanced(cand):
            # Lifecycle may still need an epoch tick to offer/consume APPLY ready.
            item.meta = dict(item.meta or {}, green_apply_pending=True, head_advanced=False)
            self._persist(item)
            return item, 'green-head-pending'
        return _finalize_advanced()

    def _finish_decision(self, item: ItemRecord) -> tuple[ItemRecord, str]:
        # Recovery (e1055): decided + box already gone + non-GREEN decision on disk.
        # Still mark gate_emit_ready and move to RELEASED so turns/PIT-RESULT finish.
        # Never take this path for unverified GREEN — that would drop C on the box.
        decision = (item.decision or '').lower()
        if (
            self.config.is_live()
            and item.decision
            and not item.box_id
            and decision not in ('green_fresh', 'green')
            and (item.meta or {}).get('decision_emitted') is not True
        ):
            item.meta = dict(item.meta or {}, gate_emit_ready=True)
            self._seal_decision_log(item)
            if item.state is ItemState.DECIDED and can_transition(item, ItemState.RELEASED):
                return apply_transition(
                    item, ItemState.RELEASED, reason='post-decision-recovery',
                    updates={'box_id': None, 'box_name': None},
                ), 'post-decision-recovery'
            return item, 'post-decision-recovery'
        if decision in ('green_fresh', 'green') and not item.box_id:
            kept = self._keep_green_for_retry(
                item, 'green-missing-box:cannot-apply-without-pod',
            )
            return kept, 'green-apply-retry'
        if self.config.is_live() and self.live_applicator is not None:
            retry_at = (item.meta or {}).get('apply_retry_at')
            if (
                decision in ('green_fresh', 'green')
                and retry_at is not None
                and float(retry_at) > float(self.now_fn())
            ):
                return item, 'green-apply-backoff'
            # Already queued GREEN waiting on HEAD-CHAIN: drive apply_tick /
            # reconcile without re-uploading; refuse incomplete master.
            if (
                decision in ('green_fresh', 'green')
                and (item.meta or {}).get('green_apply_pending')
                and ((item.meta or {}).get('live_apply') or {}).get('pe_apply_accepted')
            ):
                return self._drive_pending_head(item)
            scored = dict((item.meta or {}).get('scored') or {'decision': item.decision})
            try:
                result = self.live_applicator.apply(item, scored=scored)
            except PromoteRefused as exc:
                # Never promote on timeout / incomplete — demote loudly.
                item.meta = dict(item.meta or {}, live_apply_error=str(exc)[:300], gate_emit_ready=True)
                self._alert_green_apply_failure(item, 'promote-refused:%s' % str(exc)[:200])
                demoted = item
                if can_transition(item, ItemState.RELEASED):
                    demoted = apply_transition(
                        item,
                        ItemState.RELEASED,
                        reason='promote-refused:%s' % str(exc)[:120],
                        updates={
                            'decision': 'no_decision', 'paid': False, 'fresh': None,
                            'merge_blocked': False,
                        },
                    )
                demoted.meta = dict(demoted.meta or {}, gate_emit_ready=True)
                return self._destroy_and_release(demoted, reason='promote-refused')
            except ApplyLiveError as exc:
                item.meta = dict(item.meta or {}, live_apply_error=str(exc)[:300])
                if item.decision in ('red', 'green_stale'):
                    item.meta['gate_emit_ready'] = True
                    self._seal_decision_log(item)
                    return self._destroy_and_release(
                        item, reason='apply-failed-keep-decision:%s' % str(exc)[:80],
                    )
                if decision in ('green_fresh', 'green'):
                    kept = self._keep_green_for_retry(
                        item, 'green-apply:%s' % str(exc)[:200],
                    )
                    return kept, 'green-apply-retry'
                released = apply_transition(
                    item,
                    ItemState.RELEASED,
                    reason='apply-failed:%s' % str(exc)[:120],
                    updates={
                        'decision': 'no_decision', 'paid': False, 'fresh': None,
                        'merge_blocked': False, 'box_id': None, 'box_name': None,
                    },
                )
                released.meta = dict(released.meta or {}, decision_log_pending=True, gate_emit_ready=True)
                self._seal_decision_log(released)
                return released, 'apply-failed'
            item.meta['live_apply'] = result.as_dict()
            item.meta['gate_emit_ready'] = True
            item.meta.pop('apply_retry_at', None)
            self._seal_decision_log(item)
            if item.decision == 'green_fresh' and not result.head_advanced:
                # Queued in GREENS pending; HEAD-CHAIN not yet appended. Keep the
                # box until cutover_apply confirms record_applied (or master fix).
                item.meta['green_apply_pending'] = True
                item.meta['head_advanced'] = False
                self._persist(item)
                return item, 'green-head-pending'
            item.meta.pop('green_apply_pending', None)
            item.meta['head_advanced'] = bool(result.head_advanced)
            if result.head_advanced and item.candidate_id:
                self.current_head = item.candidate_id
                self.current_head_epoch = int(self.current_head_epoch) + 1
                if self.on_green_fresh is not None:
                    self.on_green_fresh(item)
            if result.destroyed_box_state or item.decision != 'green_fresh':
                return self._destroy_and_release(item, reason=item.decision or 'decided')
            return self._release(item, reason='green_fresh-applied')
        # Shadow / live without applicator: release; still mark emit ready for RED.
        item.meta = dict(item.meta or {}, gate_emit_ready=True)
        if item.decision and item.decision != 'green_fresh':
            return self._destroy_and_release(item, reason=item.decision)
        return self._release(item, reason='decided')

    def _unverified_green(self, item: ItemRecord) -> bool:
        """Live GREEN whose candidate/master uploads are not yet verified.

        Shadow / unit paths without a live applicator must still release.
        """
        if not self.config.is_live():
            return False
        decision = (item.decision or '').lower()
        if decision not in ('green_fresh', 'green'):
            return False
        apply = (item.meta or {}).get('live_apply') or {}
        return not (
            apply.get('candidate_verified') and apply.get('master_verified')
        )

    def _destroy_and_release(self, item: ItemRecord, *, reason: str) -> tuple[ItemRecord, str]:
        # Never destroy a pod that still holds the only copy of a GREEN candidate.
        if self._unverified_green(item) and reason != 'green_fresh-applied':
            kept = self._keep_green_for_retry(
                item, 'refuse-destroy-unverified-green:%s' % reason[:160],
            )
            return kept, 'green-hold'
        if self.destroy_box_state_fn is not None and item.box_id:
            try:
                self.destroy_box_state_fn(
                    box_id=item.box_id, candidate_id=item.candidate_id, reason=reason,
                )
            except Exception as exc:
                item.meta = dict(item.meta or {}, destroy_error=str(exc)[:200])
        return self._release(item, reason=reason)

    def _release(self, item: ItemRecord, *, reason: str) -> tuple[ItemRecord, str]:
        if self._unverified_green(item) and reason != 'green_fresh-applied':
            kept = self._keep_green_for_retry(
                item, 'refuse-release-unverified-green:%s' % reason[:160],
            )
            return kept, 'green-hold'
        if item.box_id and (holds_rental(item) or item.box_id):
            # Skip rent.release when already out of the active ledger (idempotent
            # decided-item retries must not grow history).
            if item.box_id in self.rent.ledger.active:
                try:
                    self.rent.release(item.box_id, reason=reason)
                except Exception:
                    if item.box_id in self.rent.ledger.active:
                        self.rent.ledger.release(
                            item.box_id, reason='force-%s' % reason, at=self.now_fn(),
                        )
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
        """Always return an ItemRecord (never a (item, note) tuple)."""
        if item.state is ItemState.DECIDED:
            item.last_error = reason
            decision = (item.decision or '').lower()
            # Unverified GREEN: keep the box and retry apply — never terminal release
            # without DECISION/PIT-RESULT (candidate may exist only on the pod).
            if decision in ('green_fresh', 'green') and not (
                ((item.meta or {}).get('live_apply') or {}).get('candidate_verified')
                and ((item.meta or {}).get('live_apply') or {}).get('master_verified')
            ):
                return self._keep_green_for_retry(item, 'fail-keep-green:%s' % reason[:200])
            # Post-decision failures for RED/stale/verified GREEN: clear box and
            # surface gate_emit_ready so later ticks finish turns / PIT-RESULT.
            released, _note = self._destroy_and_release(item, reason=reason)
            released.meta = dict(released.meta or {}, gate_emit_ready=True)
            return released
        next_attempt = item.attempt + 1
        if item.box_id and item.box_id in self.rent.ledger.active:
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
