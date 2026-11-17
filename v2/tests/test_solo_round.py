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

        def _serve_one(self, head, params, table, gpu):
            events.append(("warm", gpu, time.monotonic()))
            return f"http://gpu{gpu}"

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


def test_no_server_starts_on_a_training_gpu_and_failed_idle_start_is_retried(build):
    ticket = build.ticket(n_entries=1, n_tasks=8, params=build.params(n_tasks=8, n_gpus=2, looks=(1.0,), eval_chunk_tasks=4))
    artifact = build.artifact(ticket.head, ticket.entries[0], ticket.params)
    lock = threading.Lock()
    active = set()
    events = []
    attempts = []

    class Evaluator:
        http = None
        failures = 1

        def _serve_one(self, head, params, table, gpu):
            with lock:
                assert gpu not in active, "vLLM started on a GPU whose trainer is active"
                attempts.append(gpu)
                if self.failures:
                    self.failures -= 1
                    raise RuntimeError("early server start failed")
                events.append(("serve", gpu, time.monotonic()))
            return f"http://gpu{gpu}"

        def solo_worker(self, ticket, trained, deadline, gpu):
            return Worker(gpu, events, {0: artifact})

        def close(self):
            pass

    class Trainer:
        def train(self, request):
            with lock:
                active.add(request.gpu_ids[0])
                events.append(("train", request.gpu_ids[0], time.monotonic()))
            time.sleep(0.03)
            with lock:
                active.discard(request.gpu_ids[0])
            return artifact

        def close(self):
            pass

    result = SoloRunner(Trainer(), Evaluator(), lambda: 1000, build.image, settings=SoloSettings(prewarm=True, steal_block_tasks=2), poll_s=0.01).run(ticket)
    assert result.null_outcomes == ("P" * 8,)
    assert any(kind == "serve" and gpu == 1 for kind, gpu, _ in events)
    assert any(kind == "serve" and gpu == 0 for kind, gpu, _ in events)
    assert len(attempts) >= 2 and attempts.count(0) >= 1 and attempts.count(1) >= 1


def test_more_batches_than_gpus_never_prewarms_between_training_jobs(build):
    ticket = build.ticket(n_entries=3, n_tasks=8, params=build.params(n_tasks=8, n_gpus=2, looks=(1.0,), eval_chunk_tasks=4))
    artifacts = {entry.queue_pos: build.artifact(ticket.head, entry, ticket.params) for entry in ticket.entries}
    active = set()
    events = []
    lock = threading.Lock()

    class Evaluator:
        http = None

        def _serve_one(self, head, params, table, gpu):
            with lock:
                assert gpu not in active, "server started between training jobs"
                events.append(("serve", gpu))
            return f"http://gpu{gpu}"

        def solo_worker(self, ticket, trained, deadline, gpu):
            return Worker(gpu, events, artifacts)

        def close(self):
            pass

    class Trainer:
        def train(self, request):
            with lock:
                active.add(request.gpu_ids[0])
            time.sleep(0.02)
            with lock:
                active.discard(request.gpu_ids[0])
            return artifacts[request.entry.queue_pos]

        def close(self):
            pass

    result = SoloRunner(Trainer(), Evaluator(), lambda: 1000, build.image, poll_s=0.01).run(ticket)
    assert result.null_outcomes == ("P" * 8,)
    assert all(entry.outcomes == ("P" * 8,) for entry in result.entries)
