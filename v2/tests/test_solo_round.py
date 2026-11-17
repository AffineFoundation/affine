"""A complete solo round with fake training and independent GPU workers."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from sn120_v2.gpu.solo import SoloRunner, SoloSettings
from sn120_v2.judging import check_result
from sn120_v2.sampling import rollout_seed
from sn120_v2.types import NULL_MODEL


class Worker:
    def __init__(self, gpu, events, artifacts=None):
        self.gpu, self.events, self.artifacts = gpu, events, artifacts or {}

    def run(self, *, steal, result, judge, alive, deadline):
        self.events.append(("serve", self.gpu, time.monotonic()))
        self.judge = judge
        while True:
            claim = steal(self.gpu, NULL_MODEL)
            if claim is None:
                return
            model, start, stop = claim
            for index in range(start, stop):
                result(model, 0, index, "P")


def test_solo_round_steals_and_preserves_seed_identity(build):
    ticket = build.ticket(n_entries=3, n_tasks=16, params=build.params(n_tasks=16, n_gpus=3, looks=(1.0,), eval_chunk_tasks=4))
    artifacts = {entry.queue_pos: build.artifact(ticket.head, entry, ticket.params) for entry in ticket.entries}
    events = []

    class Evaluator:
        http = None

        def solo_worker(self, ticket, trained, deadline, gpu):
            return Worker(gpu, events, artifacts)

        def prewarm(self, *args, **kwargs):
            events.append(("warm", kwargs.get("gpu"), time.monotonic()))

        def close(self):
            pass

        @property
        def _solo_artifacts(self):
            return artifacts

    class Trainer:
        def train(self, request):
            events.append(("train", request.entry.queue_pos, time.monotonic()))
            if request.entry.queue_pos == 0:
                time.sleep(0.05)
                events.append(("train_end", request.entry.queue_pos, time.monotonic()))
            return artifacts[request.entry.queue_pos]

        def close(self):
            pass

    runner = SoloRunner(Trainer(), Evaluator(), lambda: 1000, build.image, settings=SoloSettings(steal_block_tasks=2), poll_s=0.01)
    result = runner.run(ticket)
    assert result.null_outcomes == ("P" * 16,)
    assert all(entry.outcomes == ("P" * 16,) for entry in result.entries)
    assert not result.stopped_at_limit
    check_result(ticket, result)
    assert {kind for kind, *_ in events} == {"train", "train_end", "warm", "serve"}
    train_zero_end = next(at for kind, pos, at in events if kind == "train_end" and pos == 0)
    serve_fast = next(at for kind, gpu, at in events if kind == "serve" and gpu != 0)
    assert serve_fast < train_zero_end
    for task in ticket.draw.task_ids:
        assert rollout_seed(ticket.draw.round_seed, task, 0) == rollout_seed(ticket.draw.round_seed, task, 0)
