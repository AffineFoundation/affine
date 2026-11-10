"""Shared serverless job lifecycle (SPEC §4).

REQUESTED → RENTING → BOOTING → LOADING → RUNNING → STORING → VERIFIED →
WIPING → RELEASED.

Hard rules:
* Rent only when a JOB.json exists (no warm pool).
* Price ceiling $56/h for B300 item/baseline (SPEC D1).
* Disable Lium auto-backup on every rental.
* Never promote / never score a partial or timed-out probe.
* Fail-closed errors are not auto-retried.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Set

from .kinds import JOB_KINDS, JobKind, NAME_PREFIX, counts_toward_item_cap, kind_cap

PRICE_CEILING_USD_H = 56.0
PREFERRED_PRICE_USD_H = 35.0
PINNED_GPU = 'B300'
MAX_ATTEMPTS_DEFAULT = 3


class JobState(str, Enum):
    REQUESTED = 'requested'
    RENTING = 'renting'
    BOOTING = 'booting'
    LOADING = 'loading'
    RUNNING = 'running'
    STORING = 'storing'
    VERIFIED = 'verified'
    WIPING = 'wiping'
    RELEASED = 'released'
    FAILED = 'failed'
    NO_DECISION = 'no_decision'


_FORWARD = {
    JobState.REQUESTED: {JobState.RENTING, JobState.FAILED},
    JobState.RENTING: {JobState.BOOTING, JobState.FAILED, JobState.REQUESTED},
    JobState.BOOTING: {JobState.LOADING, JobState.FAILED},
    JobState.LOADING: {JobState.RUNNING, JobState.FAILED},
    JobState.RUNNING: {JobState.STORING, JobState.FAILED, JobState.NO_DECISION},
    JobState.STORING: {JobState.VERIFIED, JobState.FAILED},
    JobState.VERIFIED: {JobState.WIPING},
    JobState.WIPING: {JobState.RELEASED},
    JobState.FAILED: {JobState.REQUESTED, JobState.WIPING, JobState.RELEASED, JobState.NO_DECISION},
    JobState.NO_DECISION: {JobState.WIPING, JobState.RELEASED},
    JobState.RELEASED: set(),
}


class JobError(RuntimeError):
    pass


class FailClosedError(JobError):
    """fp32 / expected-hash / serve-flag / bundle-sha — never auto-retry."""


class WaitForOffer(JobError):
    """No eligible offer under the price ceiling yet.

    Keeps the job in REQUESTED without burning an attempt. Used by bench/item
    rent paths that must wait for B300 ≤ ceiling rather than fail-closed.
    """


@dataclass
class JobRecord:
    job_id: str
    kind: JobKind
    subject: str  # epoch id, head id, round id, …
    state: JobState = JobState.REQUESTED
    attempt: int = 0
    max_attempts: int = MAX_ATTEMPTS_DEFAULT
    gpu: str = PINNED_GPU
    price_ceiling_usd_h: float = PRICE_CEILING_USD_H
    box_id: Optional[str] = None
    box_name: Optional[str] = None
    auto_backup_disabled: bool = False
    result_stored: bool = False
    verified: bool = False
    last_error: Optional[str] = None
    inputs_sha256: Dict[str, str] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def as_dict(self) -> dict:
        return {
            'job_id': self.job_id,
            'kind': self.kind.value,
            'subject': self.subject,
            'state': self.state.value,
            'attempt': self.attempt,
            'max_attempts': self.max_attempts,
            'gpu': self.gpu,
            'price_ceiling_usd_h': self.price_ceiling_usd_h,
            'box_id': self.box_id,
            'box_name': self.box_name,
            'auto_backup_disabled': self.auto_backup_disabled,
            'result_stored': self.result_stored,
            'verified': self.verified,
            'last_error': self.last_error,
            'inputs_sha256': dict(self.inputs_sha256),
            'meta': dict(self.meta),
            'created_at': self.created_at,
            'counts_toward_item_cap': counts_toward_item_cap(self.kind),
        }


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class JobRunner:
    """In-memory job runner used by unit tests and the PIT controller.

    Host Lium rental is injected via ``rent_fn`` / ``release_fn``. When absent,
    synthetic boxes are used (same pattern as PIT pe_rent).
    """

    live: bool = True
    price_ceiling_usd_h: float = PRICE_CEILING_USD_H
    jobs: List[JobRecord] = field(default_factory=list)
    rent_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    release_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    disable_auto_backup_fn: Optional[Callable[[str], bool]] = None
    now_fn: Callable[[], float] = time.time
    _active_by_kind: Dict[JobKind, Set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._active_by_kind:
            self._active_by_kind = {k: set() for k in JobKind}

    def active_count(self, kind: Optional[JobKind] = None) -> int:
        if kind is None:
            return sum(len(v) for v in self._active_by_kind.values())
        return len(self._active_by_kind[kind])

    def item_cap_occupancy(self) -> int:
        """Only item jobs count toward the PIT box_cap."""
        return self.active_count(JobKind.ITEM)

    def can_start(self, kind: JobKind) -> bool:
        cap = kind_cap(kind, live=self.live)
        if cap <= 0:
            return False
        return self.active_count(kind) < cap

    def request(
        self,
        kind: JobKind | str,
        subject: str,
        *,
        inputs_sha256: Optional[Mapping[str, str]] = None,
        gpu: Optional[str] = None,
        meta: Optional[Mapping[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> JobRecord:
        if isinstance(kind, str):
            if kind not in JOB_KINDS:
                raise ValueError('unknown job kind %r' % kind)
            kind = JobKind(kind)
        if kind in (JobKind.ITEM, JobKind.BASELINE):
            gpu = gpu or PINNED_GPU
            if gpu != PINNED_GPU:
                raise ValueError('item/baseline jobs are pinned to B300 (no fallback)')
        else:
            gpu = gpu or PINNED_GPU
        if not self.can_start(kind):
            raise JobError('%s cap reached' % kind.value)
        rec = JobRecord(
            job_id=job_id or _new_id(),
            kind=kind,
            subject=subject,
            gpu=gpu,
            price_ceiling_usd_h=float(self.price_ceiling_usd_h),
            inputs_sha256=dict(inputs_sha256 or {}),
            meta=dict(meta or {}),
            created_at=float(self.now_fn()),
        )
        self.jobs.append(rec)
        return rec

    def tick(self) -> Dict[str, Any]:
        actions: List[dict] = []
        for job in list(self.jobs):
            if job.state is JobState.RELEASED:
                continue
            try:
                note = self._advance(job)
                if note != 'noop':
                    actions.append({'job_id': job.job_id, 'kind': job.kind.value, 'state': job.state.value, 'note': note})
            except WaitForOffer as exc:
                job.last_error = str(exc)[:300]
                # Stay REQUESTED; attempt not burned (see _rent).
                if job.state is not JobState.REQUESTED:
                    job.state = JobState.REQUESTED
                actions.append({
                    'job_id': job.job_id,
                    'kind': job.kind.value,
                    'state': job.state.value,
                    'note': 'wait-offer',
                })
            except FailClosedError as exc:
                job.last_error = str(exc)[:300]
                self._fail_closed(job)
                actions.append({'job_id': job.job_id, 'kind': job.kind.value, 'state': job.state.value, 'note': 'fail-closed'})
            except JobError as exc:
                job.last_error = str(exc)[:300]
                self._fail_retryable(job)
                actions.append({'job_id': job.job_id, 'kind': job.kind.value, 'state': job.state.value, 'note': 'fail'})
        return {
            'ok': True,
            'actions': actions,
            'active': {k.value: self.active_count(k) for k in JobKind},
            'item_cap_occupancy': self.item_cap_occupancy(),
        }

    def _advance(self, job: JobRecord) -> str:
        if job.state is JobState.REQUESTED:
            return self._rent(job)
        if job.state is JobState.RENTING:
            return self._boot(job)
        if job.state is JobState.BOOTING:
            job.state = JobState.LOADING
            return 'loading'
        if job.state is JobState.LOADING:
            job.state = JobState.RUNNING
            return 'running'
        if job.state is JobState.RUNNING:
            # Kind-specific work is driven by callers; default marks stored.
            job.state = JobState.STORING
            job.result_stored = True
            return 'storing'
        if job.state is JobState.STORING:
            job.state = JobState.VERIFIED
            job.verified = True
            return 'verified'
        if job.state is JobState.VERIFIED:
            job.state = JobState.WIPING
            return 'wiping'
        if job.state is JobState.WIPING:
            return self._release(job, reason='verified')
        if job.state is JobState.FAILED:
            if job.attempt >= job.max_attempts:
                job.state = JobState.NO_DECISION
                return 'no_decision'
            # Retry: new box, same subject/inputs.
            return self._rent(job, retry=True)
        if job.state is JobState.NO_DECISION:
            return self._release(job, reason='no_decision')
        return 'noop'

    def _rent(self, job: JobRecord, *, retry: bool = False) -> str:
        if not retry and not self.can_start(job.kind) and job.box_id is None:
            # Already counted if re-entering; for first rent check cap.
            if job.job_id not in self._active_by_kind[job.kind]:
                raise JobError('%s cap reached' % job.kind.value)
        next_attempt = int(job.attempt) + 1
        job.state = JobState.RENTING
        name = '%s%s-a%d' % (NAME_PREFIX[job.kind], job.subject[:16], next_attempt)
        at = float(self.now_fn())
        if self.rent_fn is not None:
            try:
                out = dict(self.rent_fn(
                    name=name, kind=job.kind.value, subject=job.subject,
                    attempt=next_attempt, gpu=job.gpu,
                    price_ceiling_usd_h=job.price_ceiling_usd_h,
                    disable_auto_backup=True,
                ))
            except WaitForOffer:
                job.state = JobState.REQUESTED
                raise
            price = out.get('price_usd_h')
            if price is not None and float(price) > float(job.price_ceiling_usd_h):
                # Offer slipped above ceiling after precheck — wait, do not burn attempt.
                if self.release_fn is not None:
                    bid = str(out.get('box_id') or out.get('id') or '')
                    if bid:
                        try:
                            self.release_fn(box_id=bid, reason='rent-price-ceiling')
                        except Exception:
                            pass
                job.state = JobState.REQUESTED
                raise WaitForOffer(
                    'price %.2f above ceiling %.2f' % (float(price), job.price_ceiling_usd_h)
                )
            box_id = str(out.get('box_id') or out.get('id') or '')
            if not box_id:
                raise JobError('rent returned no box_id')
            pod_name = out.get('name') or name
        else:
            box_id = 'job-box-%s-%d' % (job.job_id[:12], next_attempt)
            pod_name = name
            out = dict(box_id=box_id, name=pod_name, synthetic=True, gpu=job.gpu,
                       price_usd_h=min(PREFERRED_PRICE_USD_H, job.price_ceiling_usd_h))
        # Disable Lium auto-backup (SPEC §8) — required before any private data lands.
        disabled = True
        if self.disable_auto_backup_fn is not None:
            disabled = bool(self.disable_auto_backup_fn(box_id))
        if not disabled:
            # Refuse to proceed with backup left on: wipe would not be real.
            if self.release_fn is not None:
                try:
                    self.release_fn(box_id=box_id, reason='auto-backup-still-on')
                except Exception:
                    pass
            raise FailClosedError('lium auto-backup must be disabled on every rental')
        job.attempt = next_attempt
        job.box_id = box_id
        job.box_name = pod_name
        job.auto_backup_disabled = True
        job.meta['rented_at'] = at
        job.meta['rent'] = {k: out[k] for k in out if k not in ('presigned_url', 'secret', 'private_key')}
        self._active_by_kind[job.kind].add(job.job_id)
        return 'rented'

    def _boot(self, job: JobRecord) -> str:
        if not job.auto_backup_disabled:
            raise FailClosedError('auto-backup not confirmed disabled')
        job.state = JobState.BOOTING
        return 'booting'

    def _release(self, job: JobRecord, *, reason: str) -> str:
        if job.box_id:
            if self.release_fn is not None:
                try:
                    self.release_fn(box_id=job.box_id, reason=reason)
                except Exception as exc:
                    job.meta['release_error'] = str(exc)[:200]
            job.box_id = None
            job.box_name = None
        self._active_by_kind[job.kind].discard(job.job_id)
        job.state = JobState.RELEASED
        return 'released'

    def _fail_closed(self, job: JobRecord) -> None:
        # Alert path: no auto-retry. Wipe + release.
        job.state = JobState.NO_DECISION
        if job.box_id:
            self._release(job, reason='fail-closed')
            job.state = JobState.RELEASED

    def _fail_retryable(self, job: JobRecord) -> None:
        if job.box_id:
            if self.release_fn is not None:
                try:
                    self.release_fn(box_id=job.box_id, reason='retry')
                except Exception:
                    pass
            job.box_id = None
            job.box_name = None
        self._active_by_kind[job.kind].discard(job.job_id)
        if job.attempt >= job.max_attempts:
            job.state = JobState.NO_DECISION
        else:
            job.state = JobState.FAILED

    def mark_running_complete(self, job_id: str, *, result: Optional[Mapping[str, Any]] = None) -> JobRecord:
        job = self._get(job_id)
        if job.state is not JobState.RUNNING:
            raise JobError('mark_running_complete requires RUNNING')
        if result is not None:
            job.meta['result'] = dict(result)
        job.result_stored = True
        job.state = JobState.STORING
        return job

    def _get(self, job_id: str) -> JobRecord:
        for j in self.jobs:
            if j.job_id == job_id:
                return j
        raise KeyError(job_id)
