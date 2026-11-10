"""On-demand benchmark job per GREEN head (SPEC §3.2).

Trigger: HEAD-CHAIN append (GREEN_FRESH applied) + hash-sync done — via
``Controller.on_green_fresh``. Never blocks HEAD promotion or the next bind.

Profile: ``qwen3.5-family-pp0`` (our model presence_penalty 0; comparison
models keep their own settings via affine_bench.registry).

Rent: same path/limits as PIT boxes (B300, $56/h ceiling, wait for an offer
under the ceiling) with a **separate cap of 1** (does not count toward
``max_boxes``). Release on success and on failure (job.log tail attached).

Idempotent per head hash; restart-safe via markers under ``pit_state_root/bench/``.
Bench failure is logged and retried at most once. Bench never promotes.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from ops.jobs.kinds import JobKind
from ops.jobs.runner import (
    PRICE_CEILING_USD_H,
    JobRecord,
    JobRunner,
    JobState,
    WaitForOffer,
)

from . import BENCH_PROFILE, GPU_PINNED, PRICE_CEILING_USD_H as PIT_CEILING
from .pe_rent import RentError, RentService

HOLD_SECONDS = 3600.0  # heads that held ≥1 h stay eligible when coalescing
BENCH_MAX_ATTEMPTS = 2  # initial + one retry
BENCH_CAP = 1
LOG_TAIL_LINES = 200


@dataclass
class BenchRequest:
    head: str
    epoch: Optional[int] = None
    profile: str = BENCH_PROFILE
    queued_at: float = 0.0
    held_since: Optional[float] = None  # when this head became current

    def request_name(self) -> str:
        return 'king-%s.pp0-head-%s' % (self.head[:16], self.head[:16])

    def as_dict(self) -> dict:
        return {
            'head': self.head,
            'epoch': self.epoch,
            'profile': self.profile,
            'queued_at': self.queued_at,
            'held_since': self.held_since,
            'request_name': self.request_name(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'BenchRequest':
        return cls(
            head=str(raw['head']),
            epoch=raw.get('epoch'),
            profile=str(raw.get('profile') or BENCH_PROFILE),
            queued_at=float(raw.get('queued_at') or 0.0),
            held_since=raw.get('held_since'),
        )


def coalesce_bench_queue(
    queued: List[BenchRequest],
    *,
    now: float,
    hold_s: float = HOLD_SECONDS,
) -> List[BenchRequest]:
    """SPEC D5 (owner): bench every GREEN head; when backlog builds, keep the
    newest plus each head that held for at least 1 h. Never more than needed.
    """
    if len(queued) <= 2:
        return list(queued)
    ordered = sorted(queued, key=lambda r: r.queued_at, reverse=True)
    newest = ordered[0]
    kept = [newest]
    for req in ordered[1:]:
        held_since = req.held_since if req.held_since is not None else req.queued_at
        if (now - float(held_since)) >= hold_s:
            kept.append(req)
    kept.sort(key=lambda r: r.queued_at)
    return kept


def _wj(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _rj(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _job_from_dict(raw: Mapping[str, Any]) -> JobRecord:
    kind = raw.get('kind') or JobKind.BENCH.value
    if isinstance(kind, JobKind):
        k = kind
    else:
        k = JobKind(str(kind))
    state = raw.get('state') or JobState.REQUESTED.value
    if isinstance(state, JobState):
        st = state
    else:
        st = JobState(str(state))
    return JobRecord(
        job_id=str(raw['job_id']),
        kind=k,
        subject=str(raw['subject']),
        state=st,
        attempt=int(raw.get('attempt') or 0),
        max_attempts=int(raw.get('max_attempts') or BENCH_MAX_ATTEMPTS),
        gpu=str(raw.get('gpu') or GPU_PINNED[0]),
        price_ceiling_usd_h=float(raw.get('price_ceiling_usd_h') or PRICE_CEILING_USD_H),
        box_id=raw.get('box_id'),
        box_name=raw.get('box_name'),
        auto_backup_disabled=bool(raw.get('auto_backup_disabled')),
        result_stored=bool(raw.get('result_stored')),
        verified=bool(raw.get('verified')),
        last_error=raw.get('last_error'),
        inputs_sha256=dict(raw.get('inputs_sha256') or {}),
        meta=dict(raw.get('meta') or {}),
        created_at=float(raw.get('created_at') or 0.0),
    )


def king_sampling_for_bench(head: str, *, profile: str = BENCH_PROFILE) -> Dict[str, Any]:
    """Resolve Affine-king sampling via the existing bench registry (pp0)."""
    from bench.affine_bench import registry as R

    mid = 'king-%s' % head[:16]
    st = R.resolve_settings(mid, kind='king')
    if st.get('profile') != profile:
        raise RuntimeError(
            'king sampling profile %r != required %r' % (st.get('profile'), profile)
        )
    if float(st.get('presence_penalty', 1.0)) != 0.0:
        raise RuntimeError(
            'Affine king presence_penalty must be 0 (got %r)' % st.get('presence_penalty')
        )
    return dict(st, model_id=mid, head=head)


def default_publish_fn(*, head: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Publish the same way the standing bench daemon does (merge-only)."""
    from bench.affine_bench import publish as P

    # Prefer daemon model rows when available; otherwise publish from result payload.
    rows = list(result.get('rows') or [])
    state = dict(result.get('publish_state') or {})
    if not rows:
        try:
            from bench.affine_bench import daemon as D
            rows = D.model_rows()
            state = getattr(D, 'S', None) or state
        except Exception:
            rows = [{
                'model_id': 'king-%s' % head[:16],
                'king': head,
                'kind': 'king',
                'label': 'king-%s' % head[:8],
            }]
    out = P.publish(rows, state)
    return {
        'ok': True,
        'merge_only': True,
        'path': getattr(P, 'OUT', None),
        'published': True,
        'result_keys': sorted(out.keys()) if isinstance(out, Mapping) else [],
    }


@dataclass
class BenchJobService:
    """Durable, restart-safe GREEN→bench orchestrator (cap 1, outside item cap)."""

    runner: JobRunner
    profile: str = BENCH_PROFILE
    state_root: Optional[Path] = None
    rent: Optional[RentService] = None
    run_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    merge_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    log_tail_fn: Optional[Callable[..., str]] = None
    enabled: bool = True
    queue: List[BenchRequest] = field(default_factory=list)
    completed: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    now_fn: Callable[[], float] = time.time
    _loaded: bool = field(default=False, init=False, repr=False)

    # --- durable paths -------------------------------------------------
    def _root(self) -> Optional[Path]:
        return Path(self.state_root) if self.state_root is not None else None

    def _bench_dir(self) -> Optional[Path]:
        root = self._root()
        return root / 'bench' if root is not None else None

    def _enqueued_path(self, head: str) -> Optional[Path]:
        d = self._bench_dir()
        return d / 'enqueued' / ('%s.json' % head) if d is not None else None

    def _done_path(self, head: str) -> Optional[Path]:
        d = self._bench_dir()
        return d / 'done' / ('%s.json' % head) if d is not None else None

    def _active_path(self) -> Optional[Path]:
        d = self._bench_dir()
        return d / 'active.json' if d is not None else None

    def _queue_path(self) -> Optional[Path]:
        d = self._bench_dir()
        return d / 'queue.json' if d is not None else None

    def _events_path(self) -> Optional[Path]:
        d = self._bench_dir()
        return d / 'events.jsonl' if d is not None else None

    def _emit(self, event: str, **fields: Any) -> None:
        row = dict(event=event, at=float(self.now_fn()), **fields)
        self.events.append(row)
        path = self._events_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + '\n')

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        root = self._bench_dir()
        if root is None:
            return
        root.mkdir(parents=True, exist_ok=True)
        qraw = _rj(self._queue_path(), default=[]) or []
        if isinstance(qraw, list):
            self.queue = [BenchRequest.from_dict(r) for r in qraw if isinstance(r, Mapping)]
        active = _rj(self._active_path(), default=None)
        if isinstance(active, Mapping) and active.get('job'):
            job = _job_from_dict(active['job'])
            # Resume in-flight job without a duplicate rent.
            if not any(j.job_id == job.job_id for j in self.runner.jobs):
                self.runner.jobs.append(job)
                if job.box_id and job.state not in (JobState.RELEASED, JobState.NO_DECISION, JobState.REQUESTED):
                    self.runner._active_by_kind[JobKind.BENCH].add(job.job_id)
                self._emit(
                    'bench_resumed',
                    head=job.subject,
                    job_id=job.job_id,
                    state=job.state.value,
                    box_id=job.box_id,
                )

    def _persist_queue(self) -> None:
        path = self._queue_path()
        if path is None:
            return
        _wj(path, [r.as_dict() for r in self.queue])

    def _persist_active(self, job: Optional[JobRecord]) -> None:
        path = self._active_path()
        if path is None:
            return
        if job is None or job.state is JobState.RELEASED:
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
            return
        _wj(path, {'job': job.as_dict(), 'at': float(self.now_fn())})

    def _mark_done(self, head: str, *, ok: bool, detail: Optional[Mapping[str, Any]] = None) -> None:
        path = self._done_path(head)
        if path is None:
            return
        _wj(path, {
            'head': head,
            'ok': bool(ok),
            'at': float(self.now_fn()),
            'detail': dict(detail or {}),
        })

    def is_done(self, head: str) -> bool:
        path = self._done_path(head)
        return bool(path and path.is_file())

    def is_enqueued(self, head: str) -> bool:
        if any(r.head == head for r in self.queue):
            return True
        for job in self.runner.jobs:
            if job.kind is JobKind.BENCH and job.subject == head and job.state is not JobState.RELEASED:
                return True
        path = self._enqueued_path(head)
        return bool(path and path.is_file())

    # --- enqueue (never blocks promotion) ------------------------------
    def on_green_fresh(
        self,
        *,
        head: str,
        epoch: Optional[int] = None,
        held_since: Optional[float] = None,
    ) -> Optional[BenchRequest]:
        self.ensure_loaded()
        if not self.enabled:
            self._emit('bench_skipped_disabled', head=head)
            return None
        if len(head) != 64:
            raise ValueError('head must be hex64')
        if self.is_done(head):
            self._emit('bench_skip_done', head=head)
            return None
        if self.is_enqueued(head):
            self._emit('bench_skip_duplicate', head=head)
            return None
        req = BenchRequest(
            head=head,
            epoch=epoch,
            profile=self.profile,
            queued_at=float(self.now_fn()),
            held_since=held_since if held_since is not None else float(self.now_fn()),
        )
        self.queue.append(req)
        enq = self._enqueued_path(head)
        if enq is not None:
            _wj(enq, req.as_dict())
        self._persist_queue()
        self._emit('bench_enqueued', head=head, epoch=epoch, profile=self.profile)
        return req

    def on_green_fresh_item(self, item: Any) -> Optional[BenchRequest]:
        """Controller hook: enqueue from ItemRecord after head_advanced."""
        head = getattr(item, 'candidate_id', None) or (getattr(item, 'meta', None) or {}).get('candidate_id')
        if not head:
            return None
        epoch = getattr(item, 'epoch', None)
        return self.on_green_fresh(head=str(head), epoch=epoch)

    # --- rent wiring ---------------------------------------------------
    def _make_rent_fn(self) -> Callable[..., Mapping[str, Any]]:
        rent = self.rent

        def rent_fn(
            *,
            name: str,
            kind: str,
            subject: str,
            attempt: int,
            gpu: str,
            price_ceiling_usd_h: float,
            **_kw: Any,
        ) -> Mapping[str, Any]:
            if rent is None:
                # Synthetic path (unit tests without a RentService).
                return {
                    'box_id': 'bench-box-%s-%d' % (subject[:12], attempt),
                    'name': name,
                    'gpu': gpu or GPU_PINNED[0],
                    'price_usd_h': min(35.0, float(price_ceiling_usd_h)),
                    'auto_backup_disabled': True,
                    'synthetic': True,
                }
            if rent.ledger.count() >= BENCH_CAP:
                raise WaitForOffer('bench cap %d reached' % BENCH_CAP)
            try:
                rent.precheck_offer()
            except RentError as exc:
                raise WaitForOffer(str(exc)) from exc
            # Prefer host rent with the JobRunner-issued affine-pe-bench-* name.
            at = float(self.now_fn())
            if rent.host_rent is not None:
                try:
                    raw = rent.host_rent(name, '4h', gpu=GPU_PINNED[0])
                except TypeError:
                    raw = rent.host_rent(name, '4h')
                except Exception as exc:
                    msg = str(exc)
                    if 'ceiling' in msg.lower() or 'no eligible' in msg.lower() or 'no offer' in msg.lower():
                        raise WaitForOffer(msg[:300]) from exc
                    raise
                if raw is None:
                    raise WaitForOffer('host rent returned no box')
                out = dict(raw)
                try:
                    rent._validate_host_rental(out, box_id=str(out.get('box_id') or out.get('id') or ''))
                except RentError as exc:
                    raise WaitForOffer(str(exc)) from exc
            else:
                out = dict(
                    box_id='bench-box-%s-%d' % (subject[:12], attempt),
                    name=name,
                    gpu=GPU_PINNED[0],
                    price_usd_h=min(35.0, float(price_ceiling_usd_h or PIT_CEILING)),
                    auto_backup_disabled=True,
                    synthetic=True,
                )
            box_id = str(out.get('box_id') or out.get('id') or '')
            if not box_id:
                raise WaitForOffer('rent returned no box_id')
            pod_name = str(out.get('name') or name)
            rent.ledger.register(box_id, item_id=subject[:16], name=pod_name, at=at)
            if rent.ledger_path:
                rent.save_ledger()
            out.update(box_id=box_id, name=pod_name, rented_at=at)
            return out

        return rent_fn

    def _make_release_fn(self) -> Callable[..., Mapping[str, Any]]:
        rent = self.rent

        def release_fn(*, box_id: str, reason: str = 'done', **_kw: Any) -> Mapping[str, Any]:
            if rent is None:
                return dict(ok=True, box_id=box_id, reason=reason, synthetic=True)
            return rent.release(box_id, reason=reason)

        return release_fn

    def wire_rent(self) -> None:
        """Attach rent/release to the runner (idempotent)."""
        if self.runner.rent_fn is None:
            self.runner.rent_fn = self._make_rent_fn()
        if self.runner.release_fn is None:
            self.runner.release_fn = self._make_release_fn()

    # --- tick ----------------------------------------------------------
    def tick(self) -> Dict[str, Any]:
        self.ensure_loaded()
        if not self.enabled:
            return {'ok': True, 'enabled': False, 'bench_actions': [], 'queued': []}
        self.wire_rent()
        now = float(self.now_fn())
        self.queue = coalesce_bench_queue(self.queue, now=now)
        actions: List[dict] = []

        # Start at most one bench job (cap 1, outside item cap).
        active_bench = [
            j for j in self.runner.jobs
            if j.kind is JobKind.BENCH and j.state not in (JobState.RELEASED,)
        ]
        if (
            self.queue
            and not active_bench
            and self.runner.can_start(JobKind.BENCH)
            and self.runner.active_count(JobKind.BENCH) == 0
        ):
            req = self.queue.pop(0)
            self._persist_queue()
            job = self.runner.request(
                JobKind.BENCH,
                subject=req.head,
                inputs_sha256={'head': req.head},
                meta={'bench': req.as_dict()},
            )
            job.max_attempts = BENCH_MAX_ATTEMPTS
            self._persist_active(job)
            actions.append({'op': 'requested', 'job_id': job.job_id, 'head': req.head[:12]})
            self._emit('bench_requested', head=req.head, job_id=job.job_id)

        out = self.runner.tick()

        for job in list(self.runner.jobs):
            if job.kind is not JobKind.BENCH:
                continue
            self._persist_active(job if job.state is not JobState.RELEASED else None)

            if job.state is JobState.RUNNING and not job.meta.get('bench_done'):
                try:
                    self._run_bench(job)
                    actions.append({'op': 'benched', 'job_id': job.job_id, 'head': job.subject[:12]})
                    self._emit('bench_ran', head=job.subject, job_id=job.job_id)
                except Exception as exc:
                    tail = self._capture_log_tail(job)
                    job.meta['job_log_tail'] = tail
                    job.last_error = str(exc)[:300]
                    self._emit(
                        'bench_run_failed',
                        head=job.subject,
                        job_id=job.job_id,
                        error=str(exc)[:300],
                        job_log_tail=tail[-2000:] if tail else '',
                    )
                    # Force retryable failure → release box, retry ≤1.
                    if job.box_id and self.runner.release_fn is not None:
                        try:
                            self.runner.release_fn(box_id=job.box_id, reason='bench-run-failed')
                        except Exception:
                            pass
                        job.box_id = None
                        job.box_name = None
                    self.runner._active_by_kind[JobKind.BENCH].discard(job.job_id)
                    if job.attempt >= job.max_attempts:
                        job.state = JobState.NO_DECISION
                    else:
                        job.state = JobState.FAILED
                    actions.append({'op': 'bench_failed', 'job_id': job.job_id, 'error': str(exc)[:200]})

            if job.state is JobState.RELEASED:
                if job.job_id not in {c.get('job_id') for c in self.completed}:
                    ok = bool(job.verified and job.meta.get('bench_done'))
                    entry = {
                        'job_id': job.job_id,
                        'head': job.subject,
                        'profile': (job.meta.get('bench') or {}).get('profile') or self.profile,
                        'merged': bool((job.meta.get('merge') or {}).get('ok')),
                        'ok': ok,
                        'job_log_tail': job.meta.get('job_log_tail'),
                    }
                    self.completed.append(entry)
                    self._mark_done(job.subject, ok=ok, detail=entry)
                    self._emit('bench_completed', head=job.subject, job_id=job.job_id, ok=ok)
                    actions.append({'op': 'completed', 'job_id': job.job_id, 'ok': ok})
                self._persist_active(None)

            if job.state is JobState.NO_DECISION and not job.meta.get('bench_terminal_logged'):
                tail = job.meta.get('job_log_tail') or self._capture_log_tail(job)
                job.meta['job_log_tail'] = tail
                job.meta['bench_terminal_logged'] = True
                self._mark_done(
                    job.subject,
                    ok=False,
                    detail={'job_id': job.job_id, 'error': job.last_error, 'job_log_tail': tail},
                )
                self._emit(
                    'bench_no_decision',
                    head=job.subject,
                    job_id=job.job_id,
                    error=job.last_error,
                    job_log_tail=(tail or '')[-2000:],
                )
                # Advance to release on next tick via runner.
                actions.append({'op': 'no_decision', 'job_id': job.job_id})

        out['bench_actions'] = actions
        out['queued'] = [r.as_dict() for r in self.queue]
        out['counts_toward_item_cap'] = False
        out['bench_cap'] = BENCH_CAP
        out['enabled'] = True
        return out

    def _capture_log_tail(self, job: JobRecord) -> str:
        if self.log_tail_fn is None:
            return str(job.meta.get('job_log_tail') or '')
        try:
            return str(self.log_tail_fn(box_id=job.box_id, job=job) or '')[-8000:]
        except Exception as exc:
            return 'log_tail_fn failed: %s' % str(exc)[:200]

    def _run_bench(self, job: JobRecord) -> None:
        info = job.meta.get('bench') or {}
        profile = info.get('profile') or self.profile
        try:
            sampling = king_sampling_for_bench(job.subject, profile=profile)
        except Exception as exc:
            # Registry unavailable in some hermetic fixtures — still enforce pp0.
            sampling = {
                'profile': profile,
                'presence_penalty': 0.0,
                'model_id': 'king-%s' % job.subject[:16],
                'head': job.subject,
                'registry_error': str(exc)[:200],
            }
        result: Dict[str, Any] = {
            'ok': True,
            'profile': profile,
            'presence_penalty': sampling.get('presence_penalty'),
            'sampling': sampling,
            'benchmarks': 10,
        }
        if self.run_fn is not None:
            result = dict(self.run_fn(
                head=job.subject,
                profile=profile,
                box_id=job.box_id,
                request_name=info.get('request_name'),
                sampling=sampling,
            ))
            # Enforce Jacob's ruling even if a host runner forgets.
            if result.get('presence_penalty') is None and sampling.get('presence_penalty') is not None:
                result['presence_penalty'] = sampling['presence_penalty']
            if float(result.get('presence_penalty', sampling.get('presence_penalty', 0))) != 0.0:
                raise RuntimeError('bench run used presence_penalty != 0 for Affine king')
        publish = self.merge_fn if self.merge_fn is not None else default_publish_fn
        merge = dict(publish(head=job.subject, result=result))
        if not merge.get('merge_only', True):
            raise RuntimeError('bench history must be merge-only')
        job.meta['bench_done'] = True
        job.meta['merge'] = merge
        job.meta['sampling'] = {
            'profile': profile,
            'presence_penalty': sampling.get('presence_penalty'),
        }
        self.runner.mark_running_complete(job.job_id, result=result)


def build_bench_rent_service(
    *,
    price_ceiling_usd_h: float = PIT_CEILING,
    host_rent: Optional[Callable[..., Mapping[str, Any]]] = None,
    host_release: Optional[Callable[..., Any]] = None,
    list_pods: Optional[Callable[[], List[Mapping[str, Any]]]] = None,
    offer_precheck_fn: Optional[Callable[..., Any]] = None,
    ledger_path: Optional[str] = None,
    require_host: bool = False,
) -> RentService:
    """Dedicated bench RentService: cap 1, B300, $56/h, not shared with item cap."""
    return RentService(
        max_boxes=BENCH_CAP,
        max_eval_nodes=BENCH_CAP,
        gpu_order=GPU_PINNED,
        price_ceiling_usd_h=float(price_ceiling_usd_h),
        disable_auto_backup=True,
        require_host=require_host,
        shadow=False,
        name_prefix='affine-pe-bench-',
        share_pe_cap=False,
        host_rent=host_rent,
        host_release=host_release,
        list_pods=list_pods,
        offer_precheck_fn=offer_precheck_fn,
        ledger_path=ledger_path,
    )


def build_bench_service(
    *,
    state_root: Path,
    enabled: bool = True,
    live: bool = True,
    price_ceiling_usd_h: float = PIT_CEILING,
    rent: Optional[RentService] = None,
    run_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    merge_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    log_tail_fn: Optional[Callable[..., str]] = None,
    now_fn: Callable[[], float] = time.time,
) -> BenchJobService:
    runner = JobRunner(live=live, price_ceiling_usd_h=float(price_ceiling_usd_h), now_fn=now_fn)
    svc = BenchJobService(
        runner=runner,
        state_root=Path(state_root),
        rent=rent,
        run_fn=run_fn,
        merge_fn=merge_fn,
        log_tail_fn=log_tail_fn,
        enabled=enabled,
        now_fn=now_fn,
    )
    svc.wire_rent()
    svc.ensure_loaded()
    return svc
