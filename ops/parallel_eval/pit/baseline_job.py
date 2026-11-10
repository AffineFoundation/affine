"""Baseline / secret-draw short job per new head (SPEC §3.3).

Host draws N=1000 secret indices (private). The job prefers B300 then any
allowed 8-GPU under the ceiling, serves the head, receives the draw over stdin
at PROBE time only, runs k=12 at 64K, streams cells home, wipes, and releases.
Does **not** count toward the item box_cap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from ops.jobs.kinds import JobKind
from ops.jobs.runner import ALLOWED_GPUS, PINNED_GPU, JobRecord, JobRunner, JobState

from . import GATE_HEAD_ROLLOUTS, GATE_MAX_TOKENS, GATE_N_DRAWN


@dataclass
class BaselineRequest:
    head: str
    round_id: str
    seed_commitment: str  # public commitment only — never the raw seed/indices
    n_drawn: int = GATE_N_DRAWN
    k_head: int = GATE_HEAD_ROLLOUTS
    max_tokens: int = GATE_MAX_TOKENS
    gpu: str = PINNED_GPU

    def as_dict(self) -> dict:
        return {
            'head': self.head,
            'round_id': self.round_id,
            'seed_commitment': self.seed_commitment,
            'n_drawn': self.n_drawn,
            'k_head': self.k_head,
            'max_tokens': self.max_tokens,
            'gpu': self.gpu,
        }


@dataclass
class BaselineJobService:
    runner: JobRunner
    # Host-only: returns private indices at probe time. Never persisted in JOB.json.
    draw_provider: Optional[Callable[[str], Mapping[str, Any]]] = None
    probe_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    store_fn: Optional[Callable[..., Mapping[str, Any]]] = None
    queue: List[BaselineRequest] = field(default_factory=list)
    completed: List[Dict[str, Any]] = field(default_factory=list)

    def enqueue(self, req: BaselineRequest) -> BaselineRequest:
        if req.gpu not in ALLOWED_GPUS:
            raise ValueError(
                'baseline GPU must be one of %s; got %r'
                % (list(ALLOWED_GPUS), req.gpu)
            )
        if len(req.head) != 64:
            raise ValueError('head must be hex64')
        payload = req.as_dict()
        banned = frozenset(('indices', 'raw_seed', 'secret_draw', 'presign'))
        for key in payload:
            if key in banned:
                raise ValueError('baseline request must not include banned field: %s' % key)
        self.queue.append(req)
        return req

    def tick(self) -> Dict[str, Any]:
        actions: List[dict] = []
        while self.queue and self.runner.can_start(JobKind.BASELINE):
            req = self.queue.pop(0)
            job = self.runner.request(
                JobKind.BASELINE,
                subject=req.round_id,
                inputs_sha256={'head': req.head, 'seed_commitment': req.seed_commitment},
                gpu=PINNED_GPU,
                meta={'baseline': req.as_dict()},
            )
            actions.append({'op': 'requested', 'job_id': job.job_id, 'round_id': req.round_id})
        out = self.runner.tick()
        for job in list(self.runner.jobs):
            if job.kind is not JobKind.BASELINE:
                continue
            if job.state is JobState.RUNNING and not job.meta.get('baseline_done'):
                self._run_probe(job)
                actions.append({'op': 'probed', 'job_id': job.job_id})
            if job.state is JobState.RELEASED and job.job_id not in {c.get('job_id') for c in self.completed}:
                if job.result_stored and job.verified:
                    self.completed.append({
                        'job_id': job.job_id,
                        'round_id': job.subject,
                        'head': (job.meta.get('baseline') or {}).get('head'),
                        'ok': True,
                    })
        out['baseline_actions'] = actions
        out['counts_toward_item_cap'] = False
        return out

    def _run_probe(self, job: JobRecord) -> None:
        req = job.meta.get('baseline') or {}
        round_id = job.subject
        # Draw is fetched at probe time only and must not be written into job.meta.
        draw = {}
        if self.draw_provider is not None:
            draw = dict(self.draw_provider(round_id))
        indices = draw.get('indices')
        if indices is not None and not isinstance(indices, (list, tuple)):
            raise ValueError('draw indices malformed')
        if self.probe_fn is None:
            raise RuntimeError('baseline_job refuses to run without probe_fn')
        # Pass indices only as a transient arg — probe_fn must not log them.
        result = dict(self.probe_fn(
            head=req.get('head'),
            round_id=round_id,
            indices=indices,
            k=GATE_HEAD_ROLLOUTS,
            max_tokens=GATE_MAX_TOKENS,
            box_id=job.box_id,
        ))
        if result.get('partial') or result.get('timeout'):
            raise RuntimeError('baseline partial/timeout → never score')
        # Store cells on host via store_fn; do not keep indices on the job record.
        stored = {'ok': True}
        if self.store_fn is not None:
            cells = result.get('cells')
            stored = dict(self.store_fn(round_id=round_id, cells=cells, head=req.get('head')))
        job.meta['baseline_done'] = True
        job.meta['baseline_store'] = {k: stored[k] for k in stored if k not in ('indices', 'seed', 'cells')}
        self.runner.mark_running_complete(job.job_id, result={'ok': True, 'partial': False})
