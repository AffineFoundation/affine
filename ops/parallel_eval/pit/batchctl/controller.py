"""The runtime of the batch controller.

One ``Controller`` object owns the journal and the state. It is the only
writer of both. Its loop is:

1. ``commit(event)``: ask the core for a record, write it to the journal
   (fsync), then let the core apply it. Then write the follow-up records
   that the state calls for.
2. ``due_ops()``: ask the core which side effects must be in progress.
3. ``execute(op)``: run one side effect. It returns events for step 1.

Recovery is the same loop. ``open`` rebuilds the state from the journal and
starts at step 1. No effect needs to know that a crash happened: each one
reads the world before it changes it.

``step`` runs the loop once on the calling thread. ``run`` runs it with
worker threads, one per effect, so a slow rental or a slow R2 call never
blocks another batch. Only the thread that calls ``run`` writes the journal.
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from . import MODE_SHADOW, VERSION
from . import core as C
from .effects import APPLY_VERIFY_SHA256, EffectConfig, Effects, FenceView, LeaseView
from .guard import (
    JOURNAL, LOCK, STATUS, GuardedStore, StateRoot, check_mode, check_run_id,
)
from .interfaces import (
    BatchSource, BoxAgent, Clock, Judge, ObjectStore, PreconditionFailed, Provider, StoreError,
)
from .journal import FileOps, Journal, JournalError, RootLock
from .layout import MAX_HEAD_BYTES, Layout, is_hex64, run_body
from .timers import TimerConfig, Timers
from .. import (
    ALLOW_SHARDED_GPUS_DEFAULT, ALLOWED_GPUS, PRICE_CEILING_USD_H, REQUIRED_GPU_COUNT,
)
from ..gpu_policy import effective_gpu_order
from ..shadow import ShadowWriteError
from ...durable import canonical

# Smallest first, then larger. Within one GPU class: secure first, then spot.
LADDER_GPUS = ('RTX5090', 'H100', 'B300')
LADDER_TIERS = ('secure', 'spot')


class ControllerError(RuntimeError):
    """The controller cannot go on safely. The process must stop."""


def default_ladder(
    provider: str,
    *,
    allow_sharded_gpus: bool = ALLOW_SHARDED_GPUS_DEFAULT,
    secure_first: bool = False,
) -> Tuple[C.Rung, ...]:
    """8x RTX 5090 first, then larger classes; secure before spot.

    The existing promotion gate decides which classes may be used at all.
    ``secure_first``: try a secure box of every class before any spot box.
    Without it, both tiers of one class come before the next class.
    """
    order = effective_gpu_order(LADDER_GPUS, allow_sharded_gpus=allow_sharded_gpus)
    if secure_first:
        return tuple(C.Rung(provider, gpu, tier) for tier in LADDER_TIERS for gpu in order)
    return tuple(C.Rung(provider, gpu, tier) for gpu in order for tier in LADDER_TIERS)


@dataclass(frozen=True)
class Config:
    run_id: str
    genesis_state: str
    core: C.CoreConfig
    mode: str = MODE_SHADOW
    allow_sharded_gpus: bool = ALLOW_SHARDED_GPUS_DEFAULT
    gpu_count: int = REQUIRED_GPU_COUNT
    price_ceiling_usd_h: float = PRICE_CEILING_USD_H
    # Rental.
    rent_timeout_s: float = 900.0
    rent_backoff_s: float = 120.0
    box_ttl_s: float = 8 * 3600.0
    # Lease of a running box.
    lease_ttl_s: float = 4 * 3600.0
    heartbeat_grace_s: float = 1200.0
    heartbeat_every_s: float = 60.0
    renew_every_s: float = 300.0
    absent_polls: int = 2
    # Spacing between two runs of the same effect.
    retry_s: float = 10.0
    poll_every_s: float = 20.0
    ingest_every_s: float = 10.0
    sweep_every_s: float = 300.0
    status_every_s: float = 15.0
    # Money and failure limits.
    max_cost_usd: Optional[float] = None
    breaker_failures: Optional[int] = 8
    # The same effect fails this many times in a row with an error that the
    # code did not expect: stop new work until an operator has looked.
    effect_error_limit: int = 5
    apply_verify: str = APPLY_VERIFY_SHA256
    # Worker threads of ``run``. An effect that runs longer than
    # ``effect_timeout_s`` is given up: the same work may start again.
    max_workers: int = 64
    effect_timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        check_mode(self.mode)
        check_run_id(self.run_id)
        if not is_hex64(self.genesis_state):
            raise ValueError('genesis_state must be a state id (sha256)')
        gpus = []
        for rung in self.core.ladder:
            if rung.gpu not in ALLOWED_GPUS:
                raise ValueError('ladder has a GPU class that is not allowed: %r' % (rung.gpu,))
            if rung.gpu not in gpus:
                gpus.append(rung.gpu)
        if effective_gpu_order(gpus, allow_sharded_gpus=self.allow_sharded_gpus) != tuple(gpus):
            raise ValueError('ladder has a GPU class that the promotion gate blocks')
        if not 0 < self.price_ceiling_usd_h <= PRICE_CEILING_USD_H:
            raise ValueError('price ceiling must be above zero and not above the policy ceiling')
        if self.gpu_count != REQUIRED_GPU_COUNT:
            raise ValueError('a box must have %d GPUs' % REQUIRED_GPU_COUNT)
        if self.box_ttl_s < self.lease_ttl_s:
            raise ValueError('box_ttl_s must not be shorter than lease_ttl_s')
        if self.breaker_failures is not None and self.breaker_failures < 1:
            raise ValueError('breaker_failures must be at least 1')
        if self.effect_error_limit < 1:
            raise ValueError('effect_error_limit must be at least 1')
        if self.effect_timeout_s <= 0 or self.rent_timeout_s <= 0:
            raise ValueError('timeouts must be above zero')
        if self.max_workers < 2 * self.core.max_boxes + 8:
            # Every box can have one slow rental in flight. The rest of the
            # work (polls, verification, the apply) must still find a worker.
            raise ValueError('max_workers must be at least 2 * max_boxes + 8')


@dataclass(frozen=True)
class IngestOp:
    pass


@dataclass(frozen=True)
class SweepOp:
    pass


@dataclass(frozen=True)
class GenesisOp:
    state_id: str


def op_key(op: Any) -> Tuple:
    """Two ops with the same key are the same piece of work."""
    kind = type(op)
    if kind is C.RentOp:
        return ('rent', op.fence, op.try_no)
    if kind is C.LaunchOp:
        return ('launch', op.fence)
    if kind is C.PollOp:
        return ('poll', op.fence)
    if kind is C.SettleOp:
        return ('settle', op.fence)
    if kind is C.ApplyOp:
        return ('apply', op.seq)
    if kind is C.DestroyOp:
        return ('destroy', op.box_id)
    if kind is C.GcOp:
        return ('gc', op.batch_id, op.fence)
    if kind is C.PublishOp:
        return ('publish', op.batch_id)
    if kind is IngestOp:
        return ('ingest',)
    if kind is SweepOp:
        return ('sweep',)
    if kind is GenesisOp:
        return ('genesis',)
    raise ControllerError('unknown op %r' % (op,))


# Errors that must stop the process. Every other error of an effect is
# reported and the effect runs again (see ``Controller.attempt``).
FATAL = (ShadowWriteError, JournalError, C.CoreError, ControllerError)


class Controller:
    def __init__(
        self,
        *,
        root: StateRoot,
        config: Config,
        store: ObjectStore,
        providers: Mapping[str, Provider],
        agent: BoxAgent,
        source: BatchSource,
        judge: Judge,
        clock: Clock,
        file_ops: Optional[FileOps] = None,
    ):
        check_mode(config.mode)
        if root.run_id != config.run_id:
            raise ShadowWriteError('state root is for another run')
        for rung in config.core.ladder:
            if rung.provider not in providers:
                raise ValueError('ladder names an unknown provider: %r' % (rung.provider,))
        self.config = config
        self.root = root
        self.clock = clock
        self.store = GuardedStore(store, config.run_id)
        self.layout = Layout(self.store.prefix)
        self.providers = dict(providers)
        self.timers = Timers(TimerConfig(
            lease_ttl_s=config.lease_ttl_s,
            heartbeat_grace_s=config.heartbeat_grace_s,
            rent_backoff_s=config.rent_backoff_s,
            renew_every_s=config.renew_every_s,
            absent_polls=config.absent_polls,
        ))
        self.effects = Effects(
            EffectConfig(
                run_id=config.run_id,
                gpu_count=config.gpu_count,
                price_ceiling_usd_h=config.price_ceiling_usd_h,
                rent_timeout_s=config.rent_timeout_s,
                box_ttl_s=config.box_ttl_s,
                job_ttl_s=config.lease_ttl_s,
                heartbeat_every_s=config.heartbeat_every_s,
                apply_verify=config.apply_verify,
            ),
            store=self.store, providers=self.providers, agent=agent, source=source, judge=judge,
            clock=clock, timers=self.timers, layout=self.layout,
            fences=self._fence_view, lease=self._lease_view, done_batch=self._done_batch,
            known_refs=self._known_refs, alert=self.alert,
        )
        self.state = C.State()
        self.journal: Optional[Journal] = None
        self._ops = file_ops or FileOps()
        self._lock = threading.RLock()
        self._root_lock: Optional[RootLock] = None
        self._dead: Optional[str] = None
        self._streak = 0
        self._alerts: Deque[Dict[str, Any]] = collections.deque(maxlen=50)
        self._alert_counts: Dict[str, int] = collections.Counter()
        self._alert_lock = threading.Lock()
        # op key -> unexpected errors in a row. Under ``_alert_lock``.
        self._errors: Dict[Tuple, int] = {}
        self._status_at = 0.0

    # -- open / close --------------------------------------------------------

    def _init_body(self) -> Dict[str, Any]:
        return {
            'version': VERSION,
            'mode': self.config.mode,
            'run_id': self.config.run_id,
            'r2_prefix': self.store.prefix,
            'genesis_state': self.config.genesis_state,
        }

    def open(self) -> 'Controller':
        """Take the lock, rebuild the state from the journal, and bind R2."""
        with self._lock:
            if self.journal is not None:
                raise ControllerError('controller is already open')
            self._root_lock = RootLock(self.root.file(LOCK)).acquire()
            try:
                path = self.root.file(JOURNAL)
                if self._ops.exists(path):
                    journal = Journal.open(path, ops=self._ops)
                else:
                    journal = Journal.create(
                        path, kind='init', body=self._init_body(), at=self.clock.now(), ops=self._ops,
                    )
                self.journal = journal
                records = journal.replay()
                first = records[0]
                if first['t'] != 'init' or first['b'] != self._init_body():
                    raise ControllerError('the journal belongs to another run, prefix or version')
                for rec in records:
                    C.evolve(self.state, rec['t'], rec['b'])
                    self.timers.fold(rec['t'], rec['b'], rec['at'])
                    self._fold(rec['t'], rec['b'])
                self._bind_prefix(first['h'])
                self._append('controller_started', {'records': len(records)})
                self._follow_up()
            except BaseException:
                self.close()
                raise
        return self

    def _bind_prefix(self, journal_id: str) -> None:
        """Tie the R2 prefix to this journal, once and for ever.

        If the journal is lost, a new journal cannot take over the prefix
        and silently start from nothing. The operator must decide.
        """
        body = run_body(run_id=self.config.run_id, journal_id=journal_id)
        try:
            self.store.put(self.layout.run, body, if_none_match=True)
        except PreconditionFailed:
            pass
        except StoreError:
            # Outcome unknown. The read below decides.
            pass
        bound = self.store.get(self.layout.run, max_bytes=MAX_HEAD_BYTES).data
        if bound != body:
            raise ControllerError('the R2 prefix is bound to another journal')

    def close(self) -> None:
        """Release the files. Writes nothing."""
        with self._lock:
            if self.journal is not None:
                self.journal.close()
                self.journal = None
            if self._root_lock is not None:
                self._root_lock.release()
                self._root_lock = None
            self._dead = self._dead or 'closed'

    # -- the single writer ---------------------------------------------------

    def _now(self) -> float:
        return max(self.clock.now(), self.timers.last_at)

    def _check_alive(self) -> None:
        if self._dead is not None or self.journal is None:
            raise ControllerError('controller is not usable: %s' % (self._dead or 'not open'))

    def _append(self, kind: str, body: Mapping[str, Any]) -> C.Record:
        self._check_alive()
        try:
            # The exact form that a replay will read back.
            body = json.loads(canonical(body))
            # The core refuses a record that does not fit before it is written.
            C.evolve(self.state, kind, body)
            at = self._now()
            # Write-ahead: nothing acts on this record before the fsync.
            self.journal.append(kind, body, at)
        except BaseException as exc:
            # The state in memory and the journal may differ now. Stop.
            self._dead = 'journal append failed: %r' % (exc,)
            raise
        self.timers.fold(kind, body, at)
        self._fold(kind, body)
        return kind, body

    def _fold(self, kind: str, body: Mapping[str, Any]) -> None:
        if kind == 'attempt_failed':
            self._streak += 1
        elif kind in ('decided', 'resumed'):
            self._streak = 0

    def _follow_up(self) -> List[C.Record]:
        out: List[C.Record] = []
        while True:
            rec = C.resolve(self.config.core, self.state)
            if rec is None:
                break
            out.append(self._append(*rec))
        limit = self.config.breaker_failures
        if limit is not None and self._streak >= limit:
            # Many attempts in a row were lost and none reached a verdict.
            # Something is wrong with the pipeline, not with one box. Stop
            # new rentals; the batches keep their place in the queue.
            rec = C.decide(self.config.core, self.state, C.Fault('infra_breaker:%d_lost_in_a_row' % self._streak))
            if rec is not None:
                out.append(self._append(*rec))
        return out

    def commit(self, event: Any) -> List[C.Record]:
        """Turn one event into journal records. Returns the records."""
        with self._lock:
            self._check_alive()
            rec = C.decide(self.config.core, self.state, event)
            if rec is None:
                return []
            out = [self._append(*rec)]
            out.extend(self._follow_up())
            return out

    def resume(self, note: str) -> List[C.Record]:
        """Operator action: end a halt."""
        return self.commit(C.Resume(note))

    def tick(self) -> int:
        """Give a lease to every queued batch that a free slot can take."""
        count = 0
        while True:
            if not self.commit(C.Tick(hold=self._hold())):
                return count
            count += 1

    def _hold(self) -> bool:
        cap = self.config.max_cost_usd
        return cap is not None and self.timers.cost(self._now())['usd_spent_estimate'] >= cap

    # -- views for the effects ----------------------------------------------

    # A view is never ahead of the journal: after a failed append the
    # controller is dead, and a dead controller gives no view.

    def _fence_view(self) -> FenceView:
        with self._lock:
            self._check_alive()
            return FenceView(
                frozenset(self.state.by_fence), frozenset(self.state.dirty), self.state.next_fence,
            )

    def _lease_view(self, fence: int) -> Optional[LeaseView]:
        with self._lock:
            self._check_alive()
            batch = self.state.leased(fence)
            if batch is None:
                return None
            return LeaseView(batch.phase, batch.lease.try_no, batch.lease.box_id)

    def _done_batch(self, batch_id: str) -> Optional[C.Batch]:
        with self._lock:
            self._check_alive()
            return self.state.done.get(batch_id)

    def _known_refs(self) -> Dict[str, Tuple[str, str]]:
        with self._lock:
            self._check_alive()
            every = list(self.state.active.values()) + list(self.state.done.values())
            return {batch.batch_id: (batch.miner, batch.ref) for batch in every}

    def alert(self, code: str, detail: Any) -> None:
        with self._alert_lock:
            self._alert_counts[code] += 1
            self._alerts.append({'at': self._now(), 'code': code, 'detail': str(detail)[:200]})

    # -- effects -------------------------------------------------------------

    def due_ops(self) -> List[Any]:
        """The side effects that must run now."""
        with self._lock:
            self._check_alive()
            planned = C.plan(self.config.core, self.state)
            need_genesis = self.state.head is None and self.state.pending is None
        ops: List[Any] = [IngestOp()]
        if need_genesis:
            ops.append(GenesisOp(self.config.genesis_state))
        ops.extend(planned)
        ops.append(SweepOp())
        keep = frozenset(op_key(op) for op in ops)
        self.timers.forget(keep)
        with self._alert_lock:
            for key in [k for k in self._errors if k not in keep]:
                del self._errors[key]
        now = self._now()
        return [op for op in ops if self.timers.due(op_key(op), now)]

    def _spacing(self, op: Any) -> float:
        kind = type(op)
        if kind is C.PollOp:
            return self.config.poll_every_s
        if kind is IngestOp:
            return self.config.ingest_every_s
        if kind is SweepOp:
            return self.config.sweep_every_s
        return self.config.retry_s

    def execute(self, op: Any) -> List[Any]:
        """Run one side effect. Returns events for ``commit``."""
        self._check_alive()
        key, spacing = op_key(op), self._spacing(op)
        self.timers.defer(key, self._now() + spacing)
        try:
            kind = type(op)
            if kind is IngestOp:
                return self.effects.ingest()
            if kind is SweepOp:
                return self.effects.sweep()
            if kind is GenesisOp:
                return self.effects.genesis(op.state_id)
            return self.effects.run(op)
        finally:
            # The pause before the next run counts from the end of this one.
            self.timers.defer(key, self._now() + spacing)

    def attempt(self, op: Any) -> List[Any]:
        """Run one side effect, and take in an error that it did not expect.

        Such an error is a bug, or an adapter that let an error through. It
        never decides anything: the effect runs again later. When the same
        effect fails ``effect_error_limit`` times in a row, new work stops
        and the status shows why. A person must look.
        """
        key = op_key(op)
        try:
            events = self.execute(op)
        except FATAL:
            raise
        except Exception as exc:
            with self._alert_lock:
                count = self._errors[key] = self._errors.get(key, 0) + 1
            self.alert('effect_error', '%s: %r' % (key[0], exc))
            if count >= self.config.effect_error_limit:
                return [C.Fault('effect_error:%s' % key[0])]
            return []
        with self._alert_lock:
            self._errors.pop(key, None)
        return events

    def step(self) -> int:
        """One pass on the calling thread. Returns the number of records."""
        count = self.tick()
        for op in self.due_ops():
            for event in self.attempt(op):
                count += len(self.commit(event))
            count += self.tick()
        return count

    # -- threaded runtime ----------------------------------------------------

    def _patience(self, op: Any) -> float:
        if type(op) is C.RentOp:
            return 2 * self.config.rent_timeout_s
        return self.config.effect_timeout_s

    def run(self, stop: threading.Event, *, idle_s: float = 0.2) -> None:
        """Run until ``stop`` is set.

        A fatal error ends the loop with an exception. The caller must then
        end the process (``os._exit``); a supervisor starts a new one, which
        recovers from the journal and R2. The workers are daemon threads, so
        a call that hangs for ever cannot keep a dead process alive.
        """
        results: 'queue.Queue[Tuple[int, Any, Any]]' = queue.Queue()
        tokens = itertools.count(1)
        # key -> (token, op, start time). At most one worker per key.
        inflight: Dict[Tuple, Tuple[int, Any, float]] = {}
        # Workers that were given up and have not come back.
        hung: Dict[int, Tuple] = {}

        def work(token: int, op: Any) -> None:
            try:
                out: Any = self.attempt(op)
            except BaseException as exc:  # fatal: handed to the loop below
                out = exc
            results.put((token, op, out))

        while not stop.is_set():
            self.tick()
            for op in self.due_ops():
                key = op_key(op)
                if key in inflight:
                    continue
                if len(inflight) + len(hung) >= self.config.max_workers:
                    break
                token = next(tokens)
                inflight[key] = (token, op, self._now())
                threading.Thread(
                    target=work, args=(token, op), daemon=True, name='batchctl-%s' % key[0],
                ).start()
            try:
                item = results.get(timeout=idle_s)
            except queue.Empty:
                item = None
            while item is not None:
                token, op, out = item
                key = op_key(op)
                if key in inflight and inflight[key][0] == token:
                    del inflight[key]
                # A worker that was given up can still come back. Its events
                # are facts like any other: the core checks each one against
                # the fence and the phase.
                hung.pop(token, None)
                if isinstance(out, BaseException):
                    raise out
                for event in out:
                    self.commit(event)
                try:
                    item = results.get_nowait()
                except queue.Empty:
                    item = None
            now = self._now()
            for key in list(inflight):
                token, op, started = inflight[key]
                if now - started <= self._patience(op):
                    continue
                # The call did not keep its own timeout. Free the key, so the
                # same work can start again on a new worker.
                del inflight[key]
                hung[token] = key
                self.alert('effect_watchdog', key)
                if type(op) is C.RentOp:
                    # Move on to the next rung. If a box shows up later, it
                    # has the name of a closed try and the sweep destroys it.
                    self.commit(C.RentFailed(op.fence, op.try_no, 'watchdog_timeout'))
            if len(hung) > self.config.max_workers // 2:
                raise ControllerError('too many hung effects: %d' % len(hung))
            if now - self._status_at >= self.config.status_every_s:
                try:
                    self.write_status()
                except OSError as exc:
                    # The status file is a view. Its loss never stops the work.
                    self._status_at = now
                    self.alert('status_write_failed', repr(exc))

    # -- status --------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """What an operator needs to see: HEAD, batches, boxes and cost."""
        with self._lock:
            state = self.state
            now = self._now()
            phases: Dict[str, int] = collections.Counter()
            for batch in state.active.values():
                phases[batch.phase.value] += 1
            for batch in state.done.values():
                phases[batch.phase.value] += 1
            queued = [b for b in state.active.values() if b.phase is C.Phase.QUEUED]
            first = min(queued, key=lambda b: b.seq).batch_id if queued else None
            renting = sum(1 for b in state.active.values() if b.phase is C.Phase.RENTING)
            running = len(state.live_box_ids())
            cost = self.timers.cost(now)
            cost['max_cost_usd'] = self.config.max_cost_usd
            cost['new_rentals_on_hold'] = self._hold()
            head = state.head
            pending = state.pending
            out = {
                'version': VERSION,
                'mode': self.config.mode,
                'run_id': self.config.run_id,
                'at': now,
                'halted': state.halted,
                'head': None if head is None else {
                    'seq': head.seq, 'state_id': head.state_id, 'link_id': head.link_id,
                },
                'pending_apply': None if pending is None else {
                    'seq': pending.seq, 'state_id': pending.state_id, 'batch_id': pending.batch_id,
                },
                'batches': {
                    'total': len(state.active) + len(state.done),
                    'rejected': len(state.rejected),
                    'by_phase': dict(sorted(phases.items())),
                    'first_queued': first,
                },
                'boxes': {
                    'max_boxes': self.config.core.max_boxes,
                    'slots_in_use': state.slots_in_use(),
                    'renting': renting,
                    'running': running,
                    'to_destroy': len(state.doomed),
                },
                'cost': cost,
                'infra': {
                    'lost_attempts_in_a_row': self._streak,
                    'breaker_failures': self.config.breaker_failures,
                    'max_infra_failures': self.config.core.max_infra_failures,
                },
                'housekeeping': {
                    'staging_to_clear': len(state.dirty),
                    'outcomes_to_publish': len(state.unpublished),
                },
                'journal': {
                    'records': self.journal.count if self.journal is not None else None,
                    'last_hash': self.journal.last_hash if self.journal is not None else None,
                },
            }
        with self._alert_lock:
            out['alerts'] = {'counts': dict(self._alert_counts), 'recent': list(self._alerts)}
        return out

    def write_status(self) -> None:
        """Write ``status.json`` by atomic rename. The file is a view only:
        nothing reads it back, and losing it loses nothing."""
        status = self.status()
        self._status_at = status['at']
        path = self.root.file(STATUS)
        tmp = self.root.file(STATUS + '.tmp')
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(status, sort_keys=True, indent=1).encode() + b'\n')
        finally:
            os.close(fd)
        os.replace(tmp, path)
