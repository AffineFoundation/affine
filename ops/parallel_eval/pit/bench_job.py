"""On-demand GREEN bench — detached box job, non-blocking host ticks.

Trigger: HEAD-CHAIN append (``Controller.on_green_fresh``). Never blocks HEAD
promotion or the next bind.

Lifecycle mirrors PIT ``box_job``:
  rent → push descriptor/urls → launch setup/fetch/run detached →
  each tick only ``inspect`` → pull RESULT → land results for
  publish_fleet → release.

State lives only under ``pit_state/bench/<head>/``. Never reads/writes
``bench-results/daemon.json`` or standing-daemon globals. Cap 1, B300 ≤ $56/h,
separate from item ``max_boxes``. Bench failure: log + ≤1 retry; never promotes.
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
from .bench_box import (
    BENCH_BOX_GIT_REF,
    BenchBoxController,
    MemBenchBoxStore,
)
from .bench_fetch_inputs import flatten_files, push_fetch_inputs
from .bench_publish import write_results
from .box_job import STATUS_NEED_LAUNCH
from .pe_rent import RentError, RentService

HOLD_SECONDS = 3600.0
BENCH_MAX_ATTEMPTS = 2
BENCH_CAP = 1
LOG_TAIL_LINES = 200


@dataclass
class BenchRequest:
    head: str
    epoch: Optional[int] = None
    profile: str = BENCH_PROFILE
    queued_at: float = 0.0
    held_since: Optional[float] = None
    # Flat {name: sha256} from the GREEN item — never URLs.
    files: Optional[Dict[str, str]] = None

    def as_dict(self) -> dict:
        d = {
            'head': self.head,
            'epoch': self.epoch,
            'profile': self.profile,
            'queued_at': self.queued_at,
            'held_since': self.held_since,
        }
        if self.files:
            d['files'] = dict(self.files)
        return d

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> 'BenchRequest':
        files = raw.get('files')
        return cls(
            head=str(raw['head']),
            epoch=raw.get('epoch'),
            profile=str(raw.get('profile') or BENCH_PROFILE),
            queued_at=float(raw.get('queued_at') or 0.0),
            held_since=raw.get('held_since'),
            files=dict(files) if isinstance(files, Mapping) else None,
        )


def coalesce_bench_queue(
    queued: List[BenchRequest],
    *,
    now: float,
    hold_s: float = HOLD_SECONDS,
) -> List[BenchRequest]:
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


def head_state_dir(state_root: Path, head: str) -> Path:
    return Path(state_root) / 'bench' / head


@dataclass
class BenchJobService:
    """Durable GREEN→bench orchestrator: rent + detached inspect, never blocks."""

    runner: JobRunner
    state_root: Path
    profile: str = BENCH_PROFILE
    git_ref: str = BENCH_BOX_GIT_REF
    rent: Optional[RentService] = None
    box_store_factory: Optional[Callable[[JobRecord], Any]] = None
    publish_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    results_dir: Optional[str] = None  # BENCH_RESULTS_DIR — publish_fleet input root
    resolve_files_fn: Optional[Callable[..., Mapping[str, str]]] = None
    presign_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    enabled: bool = True
    queue: List[BenchRequest] = field(default_factory=list)
    completed: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    now_fn: Callable[[], float] = time.time
    # head → box store (re-adopted across ticks / restart)
    _stores: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _head_files: Dict[str, Dict[str, str]] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def _dir(self, head: str) -> Path:
        return head_state_dir(self.state_root, head)

    def _emit(self, event: str, *, head: str, **fields: Any) -> None:
        row = dict(event=event, head=head, at=float(self.now_fn()), **fields)
        self.events.append(row)
        path = self._dir(head) / 'events.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + '\n')

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        root = Path(self.state_root) / 'bench'
        if not root.is_dir():
            return
        # Restore queue from undoned heads with request.json and no active job.
        for child in sorted(root.iterdir()):
            if not child.is_dir() or len(child.name) != 64:
                continue
            head = child.name
            if (child / 'done.json').is_file():
                continue
            req = _rj(child / 'request.json')
            active = _rj(child / 'active.json')
            if isinstance(active, Mapping) and active.get('job'):
                job = self._job_from_dict(active['job'])
                if not any(j.job_id == job.job_id for j in self.runner.jobs):
                    self.runner.jobs.append(job)
                    if job.box_id and job.state not in (
                        JobState.RELEASED, JobState.NO_DECISION, JobState.REQUESTED,
                    ):
                        self.runner._active_by_kind[JobKind.BENCH].add(job.job_id)
                    self._emit(
                        'bench_resumed', head=head, job_id=job.job_id,
                        state=job.state.value, box_id=job.box_id,
                    )
                bench_meta = (job.meta.get('bench') or {})
                if isinstance(bench_meta.get('files'), Mapping) and bench_meta['files']:
                    self._head_files[head] = {
                        str(k): str(v) for k, v in bench_meta['files'].items()
                    }
            elif isinstance(req, Mapping) and not any(r.head == head for r in self.queue):
                restored = BenchRequest.from_dict(req)
                self.queue.append(restored)
                if restored.files:
                    self._head_files[head] = dict(restored.files)
            if head not in self._head_files and isinstance(req, Mapping):
                req_files = req.get('files')
                if isinstance(req_files, Mapping) and req_files:
                    self._head_files[head] = {
                        str(k): str(v) for k, v in req_files.items()
                    }

    def _job_from_dict(self, raw: Mapping[str, Any]) -> JobRecord:
        state = raw.get('state') or JobState.REQUESTED.value
        st = state if isinstance(state, JobState) else JobState(str(state))
        kind = raw.get('kind') or JobKind.BENCH.value
        k = kind if isinstance(kind, JobKind) else JobKind(str(kind))
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

    def _persist_active(self, job: Optional[JobRecord]) -> None:
        if job is None:
            return
        path = self._dir(job.subject) / 'active.json'
        if job.state is JobState.RELEASED:
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
            return
        _wj(path, {'job': job.as_dict(), 'at': float(self.now_fn()), 'git_ref': self.git_ref})

    def _mark_done(self, head: str, *, ok: bool, detail: Optional[Mapping[str, Any]] = None) -> None:
        _wj(self._dir(head) / 'done.json', {
            'head': head, 'ok': bool(ok), 'at': float(self.now_fn()),
            'detail': dict(detail or {}),
        })

    def is_done(self, head: str) -> bool:
        return (self._dir(head) / 'done.json').is_file()

    def is_enqueued(self, head: str) -> bool:
        if any(r.head == head for r in self.queue):
            return True
        for job in self.runner.jobs:
            if job.kind is JobKind.BENCH and job.subject == head and job.state is not JobState.RELEASED:
                return True
        return (self._dir(head) / 'request.json').is_file() and not self.is_done(head)

    def on_green_fresh(
        self,
        *,
        head: str,
        epoch: Optional[int] = None,
        held_since: Optional[float] = None,
        files: Optional[Mapping[str, Any]] = None,
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
        flat: Optional[Dict[str, str]] = None
        if files:
            flat = flatten_files(files)
            self._head_files[head] = flat
        req = BenchRequest(
            head=head,
            epoch=epoch,
            profile=self.profile,
            queued_at=float(self.now_fn()),
            held_since=held_since if held_since is not None else float(self.now_fn()),
            files=flat,
        )
        self.queue.append(req)
        _wj(self._dir(head) / 'request.json', req.as_dict())
        self._emit(
            'bench_enqueued', head=head, epoch=epoch, profile=self.profile,
            git_ref=self.git_ref, n_files=(len(flat) if flat else 0),
        )
        return req

    def on_green_fresh_item(self, item: Any) -> Optional[BenchRequest]:
        head = getattr(item, 'candidate_id', None)
        if not head:
            return None
        meta = getattr(item, 'meta', None) or {}
        files = meta.get('candidate_files') or meta.get('files')
        if not files:
            live = meta.get('live_apply') or {}
            files = live.get('candidate_files') or live.get('files')
        return self.on_green_fresh(
            head=str(head),
            epoch=getattr(item, 'epoch', None),
            files=files if isinstance(files, Mapping) else None,
        )

    # --- rent ------------------------------------------------------------
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
            at = float(self.now_fn())
            if rent.host_rent is not None:
                try:
                    raw = rent.host_rent(name, '4h', gpu=GPU_PINNED[0])
                except TypeError:
                    raw = rent.host_rent(name, '4h')
                except Exception as exc:
                    msg = str(exc)
                    if any(x in msg.lower() for x in ('ceiling', 'no eligible', 'no offer')):
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
        if self.runner.rent_fn is None:
            self.runner.rent_fn = self._make_rent_fn()
        if self.runner.release_fn is None:
            self.runner.release_fn = self._make_release_fn()

    def _box_store(self, job: JobRecord) -> Any:
        head = job.subject
        if head in self._stores:
            return self._stores[head]
        if self.box_store_factory is not None:
            store = self.box_store_factory(job)
        else:
            store = MemBenchBoxStore()
        self._stores[head] = store
        return store

    def _prepare_fetch(self, job: JobRecord) -> Callable[[Any, str], Mapping[str, Any]]:
        head = job.subject
        bench = job.meta.get('bench') or {}
        files = self._head_files.get(head) or bench.get('files')

        def _prep(store: Any, job_dir: str) -> Mapping[str, Any]:
            receipt = push_fetch_inputs(
                store,
                job_dir,
                head=head,
                files=files,
                resolve_fn=self.resolve_files_fn,
                presign_fn=self.presign_fn,
            )
            # Persist descriptor sha map under pit_state — never URLs.
            desc_text = store.read_text('%s/descriptor.json' % job_dir.rstrip('/'))
            try:
                desc_doc = json.loads(desc_text) if desc_text else {'id': head}
            except Exception:
                desc_doc = {'id': head, 'files': {}}
            host_desc = {
                'id': head,
                'files': dict(desc_doc.get('files') or {}),
                'n_files': receipt.get('n_files'),
                'at': float(self.now_fn()),
            }
            _wj(self._dir(head) / 'descriptor.json', host_desc)
            self._emit(
                'bench_fetch_inputs', head=head, job_id=job.job_id,
                n_files=receipt.get('n_files'), names=receipt.get('names'),
            )
            return receipt

        return _prep

    def _controller(self, job: JobRecord) -> BenchBoxController:
        return BenchBoxController(
            store=self._box_store(job),
            head=job.subject,
            profile=(job.meta.get('bench') or {}).get('profile') or self.profile,
            git_ref=self.git_ref,
            now_fn=self.now_fn,
            prepare_fetch_fn=self._prepare_fetch(job),
        )

    def _release_box(self, job: JobRecord, *, reason: str) -> None:
        if job.box_id and self.runner.release_fn is not None:
            try:
                self.runner.release_fn(box_id=job.box_id, reason=reason)
            except Exception as exc:
                job.meta['release_error'] = str(exc)[:200]
        job.box_id = None
        job.box_name = None
        self.runner._active_by_kind[JobKind.BENCH].discard(job.job_id)

    def _capture_log_tail(self, job: JobRecord) -> str:
        try:
            ctrl = self._controller(job)
            return str(ctrl.read_log_tail(n_lines=LOG_TAIL_LINES) or '')
        except Exception as exc:
            return 'log_tail failed: %s' % str(exc)[:200]

    def _publish(self, head: str, result: Mapping[str, Any]) -> Dict[str, Any]:
        # Keep a copy under pit_state/bench/<head>/; land into publish_fleet results dir.
        local = self._dir(head) / 'results'
        local.mkdir(parents=True, exist_ok=True)
        _wj(local / 'RESULT.json', dict(result))
        if self.publish_fn is not None:
            out = dict(self.publish_fn(
                head=head, result=result, results_dir_path=self.results_dir,
            ))
        else:
            out = dict(write_results(
                head=head, result=result,
                results_dir_path=self.results_dir, now=self.now_fn(),
            ))
        # Never rewrite website/benchmarks.json from PIT.
        if out.get('path') and str(out.get('path')).endswith('benchmarks.json'):
            raise RuntimeError('refuse website/benchmarks.json write from PIT bench')
        _wj(self._dir(head) / 'publish.json', {
            k: v for k, v in out.items() if k != 'urls'
        })
        return out

    # --- tick (non-blocking) --------------------------------------------
    def tick(self) -> Dict[str, Any]:
        """One host tick: rent/launch/inspect at most one step. Never sleeps."""
        t0 = float(self.now_fn())
        self.ensure_loaded()
        if not self.enabled:
            return {'ok': True, 'enabled': False, 'bench_actions': [], 'queued': [], 'elapsed_s': 0.0}
        self.wire_rent()
        now = float(self.now_fn())
        self.queue = coalesce_bench_queue(self.queue, now=now)
        actions: List[dict] = []

        active = [
            j for j in self.runner.jobs
            if j.kind is JobKind.BENCH and j.state is not JobState.RELEASED
        ]
        if (
            self.queue
            and not active
            and self.runner.can_start(JobKind.BENCH)
            and self.runner.active_count(JobKind.BENCH) == 0
        ):
            req = self.queue.pop(0)
            job = self.runner.request(
                JobKind.BENCH,
                subject=req.head,
                inputs_sha256={'head': req.head},
                meta={
                    'bench': req.as_dict(),
                    'git_ref': self.git_ref,
                    # Hold RUNNING across host ticks while detached stages run.
                    'hold_running': True,
                },
            )
            job.max_attempts = BENCH_MAX_ATTEMPTS
            self._persist_active(job)
            actions.append({'op': 'requested', 'job_id': job.job_id, 'head': req.head[:12]})
            self._emit('bench_requested', head=req.head, job_id=job.job_id)

        # Advance rent/boot machine one step (non-blocking; WaitForOffer stays REQUESTED).
        out = self.runner.tick()

        for job in list(self.runner.jobs):
            if job.kind is not JobKind.BENCH:
                continue
            self._persist_active(job if job.state is not JobState.RELEASED else None)

            # Once RUNNING on a rented box, drive detached stages via inspect only.
            if job.state is JobState.RUNNING and not job.meta.get('bench_done'):
                note = self._drive_detached(job)
                actions.append(note)
                if note.get('op') == 'published':
                    # Drop hold so subsequent ticks advance STORING→…→RELEASED.
                    job.meta.pop('hold_running', None)
                    self.runner.mark_running_complete(job.job_id, result=note.get('result'))
                elif note.get('op') == 'failed':
                    self._fail_job(job, error=str(note.get('error') or 'bench failed'))

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
                    job.subject, ok=False,
                    detail={'job_id': job.job_id, 'error': job.last_error, 'job_log_tail': tail},
                )
                self._emit(
                    'bench_no_decision', head=job.subject, job_id=job.job_id,
                    error=job.last_error, job_log_tail=(tail or '')[-2000:],
                )
                actions.append({'op': 'no_decision', 'job_id': job.job_id})

        elapsed = float(self.now_fn()) - t0
        out['bench_actions'] = actions
        out['queued'] = [r.as_dict() for r in self.queue]
        out['counts_toward_item_cap'] = False
        out['bench_cap'] = BENCH_CAP
        out['enabled'] = True
        out['elapsed_s'] = elapsed
        out['git_ref'] = self.git_ref
        return out

    def _drive_detached(self, job: JobRecord) -> Dict[str, Any]:
        """Non-blocking: inspect / launch-one-stage / finish. Never sleeps."""
        ctrl = self._controller(job)
        st = ctrl.status()
        # Stage previously launched this attempt but exited with no marker → fail.
        launched = job.meta.setdefault('launched_stages', {})
        if (
            st.state == STATUS_NEED_LAUNCH
            and st.next_stage
            and st.next_stage in launched
            and st.next_stage not in st.markers
        ):
            tail = self._capture_log_tail(job)
            job.meta['job_log_tail'] = tail
            return {
                'op': 'failed',
                'error': 'detached stage %s exited without marker' % st.next_stage,
                'job_log_tail': tail,
            }

        step = ctrl.tick_once()
        op = step.get('op')
        if op == 'launched':
            job.meta.setdefault('launched_stages', {})[str(step.get('stage'))] = step.get('pid')
        if op == 'running' or op == 'launched':
            self._emit(
                'bench_detached', head=job.subject, job_id=job.job_id,
                op=op, stage=step.get('stage'), pid=step.get('pid'),
                markers=step.get('markers'),
            )
            return dict(step, head=job.subject[:12])

        if op == 'done':
            result = dict(step.get('result') or {})
            if float(result.get('presence_penalty', 0) or 0) != 0.0:
                return {
                    'op': 'failed',
                    'error': 'presence_penalty != 0',
                    'job_log_tail': self._capture_log_tail(job),
                }
            try:
                merge = self._publish(job.subject, result)
            except Exception as exc:
                tail = self._capture_log_tail(job)
                job.meta['job_log_tail'] = tail
                return {'op': 'failed', 'error': 'publish: %s' % str(exc)[:200], 'job_log_tail': tail}
            job.meta['bench_done'] = True
            job.meta['merge'] = merge
            job.meta['sampling'] = {
                'profile': result.get('profile') or self.profile,
                'presence_penalty': 0.0,
            }
            self._emit('bench_published', head=job.subject, job_id=job.job_id, merge=merge)
            return {'op': 'published', 'result': result, 'merge': merge}

        # missing_result or unknown → fail this attempt
        tail = self._capture_log_tail(job)
        job.meta['job_log_tail'] = tail
        return {
            'op': 'failed',
            'error': step.get('error') or ('detached %s' % op),
            'job_log_tail': tail,
        }

    def _fail_job(self, job: JobRecord, *, error: str) -> None:
        job.last_error = error[:300]
        tail = job.meta.get('job_log_tail') or self._capture_log_tail(job)
        job.meta['job_log_tail'] = tail
        job.meta['launched_stages'] = {}
        self._emit(
            'bench_run_failed', head=job.subject, job_id=job.job_id,
            error=error[:300], job_log_tail=(tail or '')[-2000:],
        )
        self._release_box(job, reason='bench-run-failed')
        if job.attempt >= job.max_attempts:
            job.state = JobState.NO_DECISION
        else:
            job.state = JobState.FAILED
            # Next rent gets a fresh hold for detached stages.
            job.meta['hold_running'] = True


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
    box_store_factory: Optional[Callable[[JobRecord], Any]] = None,
    publish_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    results_dir: Optional[str] = None,
    resolve_files_fn: Optional[Callable[..., Mapping[str, str]]] = None,
    presign_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
    git_ref: str = BENCH_BOX_GIT_REF,
    now_fn: Callable[[], float] = time.time,
) -> BenchJobService:
    runner = JobRunner(live=live, price_ceiling_usd_h=float(price_ceiling_usd_h), now_fn=now_fn)
    svc = BenchJobService(
        runner=runner,
        state_root=Path(state_root),
        rent=rent,
        box_store_factory=box_store_factory,
        publish_fn=publish_fn,
        results_dir=results_dir or os.environ.get('BENCH_RESULTS_DIR') or os.environ.get('AFFINE_BENCH_RESULTS'),
        resolve_files_fn=resolve_files_fn,
        presign_fn=presign_fn,
        enabled=enabled,
        git_ref=git_ref,
        now_fn=now_fn,
    )
    svc.wire_rent()
    svc.ensure_loaded()
    return svc
