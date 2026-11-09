"""Benchmark job per GREEN head (SPEC §3.2).

Trigger: HEAD-CHAIN append (GREEN_FRESH applied). Profile: qwen3.5-family-pp0.
Merge-only history. Coalesce: never >1 bench at a time; when backlog builds,
bench the newest head (plus any that held ≥1 h when >2 queued — D5 newest-first).

Does **not** count toward the item box_cap.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from ops.jobs.kinds import JobKind
from ops.jobs.runner import JobRecord, JobRunner, JobState

from . import BENCH_PROFILE

HOLD_SECONDS = 3600.0  # heads that held ≥1 h stay eligible when coalescing


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
    # Newest first by queued_at (HEAD-CHAIN append order ≈ queue order).
    ordered = sorted(queued, key=lambda r: r.queued_at, reverse=True)
    newest = ordered[0]
    kept = [newest]
    for req in ordered[1:]:
        held_since = req.held_since if req.held_since is not None else req.queued_at
        if (now - float(held_since)) >= hold_s:
            kept.append(req)
    # Preserve chronological order for the runner (oldest remaining first).
    kept.sort(key=lambda r: r.queued_at)
    return kept


@dataclass
class BenchJobService:
    runner: JobRunner
    profile: str = BENCH_PROFILE
    run_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    merge_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    queue: List[BenchRequest] = field(default_factory=list)
    completed: List[Dict[str, Any]] = field(default_factory=list)
    now_fn: Callable[[], float] = time.time

    def on_green_fresh(self, *, head: str, epoch: Optional[int] = None, held_since: Optional[float] = None) -> BenchRequest:
        if len(head) != 64:
            raise ValueError('head must be hex64')
        req = BenchRequest(
            head=head,
            epoch=epoch,
            profile=self.profile,
            queued_at=float(self.now_fn()),
            held_since=held_since if held_since is not None else float(self.now_fn()),
        )
        self.queue.append(req)
        return req

    def tick(self) -> Dict[str, Any]:
        now = float(self.now_fn())
        self.queue = coalesce_bench_queue(self.queue, now=now)
        actions: List[dict] = []
        # Cap 1 bench job.
        if self.queue and self.runner.can_start(JobKind.BENCH) and self.runner.active_count(JobKind.BENCH) == 0:
            req = self.queue.pop(0)
            job = self.runner.request(
                JobKind.BENCH,
                subject=req.head,
                inputs_sha256={'head': req.head},
                meta={'bench': req.as_dict()},
            )
            actions.append({'op': 'requested', 'job_id': job.job_id, 'head': req.head[:12]})
        out = self.runner.tick()
        for job in list(self.runner.jobs):
            if job.kind is not JobKind.BENCH:
                continue
            if job.state is JobState.RUNNING and not job.meta.get('bench_done'):
                self._run_bench(job)
                actions.append({'op': 'benched', 'job_id': job.job_id})
            if job.state is JobState.RELEASED and job.verified:
                if job.job_id not in {c.get('job_id') for c in self.completed}:
                    self.completed.append({
                        'job_id': job.job_id,
                        'head': job.subject,
                        'profile': (job.meta.get('bench') or {}).get('profile'),
                        'merged': bool((job.meta.get('merge') or {}).get('ok')),
                    })
        out['bench_actions'] = actions
        out['queued'] = [r.as_dict() for r in self.queue]
        out['counts_toward_item_cap'] = False
        return out

    def _run_bench(self, job: JobRecord) -> None:
        info = job.meta.get('bench') or {}
        result = {'ok': True, 'profile': info.get('profile') or self.profile, 'benchmarks': 10}
        if self.run_fn is not None:
            result = dict(self.run_fn(
                head=job.subject,
                profile=info.get('profile') or self.profile,
                box_id=job.box_id,
                request_name=info.get('request_name'),
            ))
        merge = {'ok': True, 'merge_only': True}
        if self.merge_fn is not None:
            merge = dict(self.merge_fn(head=job.subject, result=result))
            if not merge.get('merge_only', True):
                raise RuntimeError('bench history must be merge-only')
        job.meta['bench_done'] = True
        job.meta['merge'] = merge
        self.runner.mark_running_complete(job.job_id, result=result)
