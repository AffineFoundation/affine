"""Prior sampler as a cheap per-head job (SPEC §3.4).

One pass per new GREEN head. Does not block scoring (draw uses the current
table). Max 1 prior job at a time. Does **not** count toward the item box_cap.
Private pool stays host-side; grades return 0600 and the box is wiped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from ops.jobs.kinds import JobKind
from ops.jobs.runner import JobRecord, JobRunner, JobState


@dataclass
class PriorRequest:
    head: str
    epoch: Optional[int] = None
    gpu: str = 'H100'  # cheap pass; SPEC proposes 8×H100

    def as_dict(self) -> dict:
        return {'head': self.head, 'epoch': self.epoch, 'gpu': self.gpu}


@dataclass
class PriorJobService:
    runner: JobRunner
    run_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    store_grades_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    queue: List[PriorRequest] = field(default_factory=list)
    completed: List[Dict[str, Any]] = field(default_factory=list)

    def on_green_fresh(self, *, head: str, epoch: Optional[int] = None) -> PriorRequest:
        if len(head) != 64:
            raise ValueError('head must be hex64')
        req = PriorRequest(head=head, epoch=epoch)
        self.queue.append(req)
        return req

    def tick(self) -> Dict[str, Any]:
        actions: List[dict] = []
        if self.queue and self.runner.can_start(JobKind.PRIOR) and self.runner.active_count(JobKind.PRIOR) == 0:
            req = self.queue.pop(0)
            job = self.runner.request(
                JobKind.PRIOR,
                subject=req.head,
                inputs_sha256={'head': req.head},
                gpu=req.gpu,
                meta={'prior': req.as_dict()},
            )
            actions.append({'op': 'requested', 'job_id': job.job_id, 'head': req.head[:12]})
        out = self.runner.tick()
        for job in list(self.runner.jobs):
            if job.kind is not JobKind.PRIOR:
                continue
            if job.state is JobState.RUNNING and not job.meta.get('prior_done'):
                self._run_pass(job)
                actions.append({'op': 'prior-pass', 'job_id': job.job_id})
            if job.state is JobState.RELEASED and job.verified:
                if job.job_id not in {c.get('job_id') for c in self.completed}:
                    self.completed.append({'job_id': job.job_id, 'head': job.subject, 'ok': True})
        out['prior_actions'] = actions
        out['counts_toward_item_cap'] = False
        return out

    def _run_pass(self, job: JobRecord) -> None:
        result = {'ok': True, 'pass': 'full'}
        if self.run_fn is not None:
            result = dict(self.run_fn(head=job.subject, box_id=job.box_id))
        if self.store_grades_fn is not None:
            # Grades only — never write the pool file into job.meta.
            grades = result.get('grades')
            stored = dict(self.store_grades_fn(head=job.subject, grades=grades))
            job.meta['grades_stored'] = bool(stored.get('ok'))
        job.meta['prior_done'] = True
        safe = {k: result[k] for k in result if k not in ('grades', 'pool', 'private')}
        self.runner.mark_running_complete(job.job_id, result=safe)
