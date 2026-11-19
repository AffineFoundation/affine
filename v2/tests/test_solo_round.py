"""A complete solo round with fake training and independent GPU workers."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from sn120_v2.gpu.evaluator import _Deadline
from sn120_v2.gpu.solo import SoloRunner, SoloScheduler, SoloSettings, SoloWorker
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


def test_idle_gpus_serve_null_before_slow_training_adds_loras(build):
    ticket = build.ticket(n_entries=6, n_tasks=64, params=build.params(n_tasks=64, n_gpus=8, looks=(1.0,), eval_chunk_tasks=4))
    artifacts = {entry.queue_pos: build.artifact(ticket.head, entry, ticket.params) for entry in ticket.entries}
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=1), evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 10000)
    gate = threading.Condition()
    null_gpus = set()
    lora_gpus = set()
    served = []
    rollouts = []
    stopped = []

    def worker(gpu):
        def steal(gpu, owner):
            if scheduler.training_done:
                with gate:
                    # A non-idle GPU may not drain LoRAs before both idle GPUs
                    # have had their first LoRA claim. An idle GPU yields after
                    # its first claim too, so the other idle GPU can join.
                    while lora_gpus != {6, 7} and (gpu not in (6, 7) or gpu in lora_gpus):
                        assert gate.wait(timeout=30), f"idle LoRA workers missing: {lora_gpus}"
            claim = scheduler.steal(gpu, owner)
            if claim is not None and claim[0] != NULL_MODEL and gpu in (6, 7):
                with gate:
                    lora_gpus.add(gpu)
                    gate.notify_all()
            if claim is not None and claim[0] == NULL_MODEL and gpu not in null_gpus:
                with gate:
                    null_gpus.add(gpu)
                    if len(null_gpus) < 8:
                        assert gate.wait(timeout=30), f"only GPUs {null_gpus} reached null work"
                    else:
                        gate.notify_all()
            return claim

        def record(model, sample, index, code):
            rollouts.append((gpu, model))
            if model != NULL_MODEL:
                with gate:
                    gate.notify_all()
            scheduler.result(model, sample, index, code)

        Worker(gpu, served).run(steal=steal, result=record, judge=scheduler.judge, alive=lambda: True, deadline=10000)
        stopped.append(gpu)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(8)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30
    with gate:
        while len(null_gpus) < 8:
            assert not stopped, f"idle GPUs exited before training: {stopped}"
            assert time.monotonic() < deadline, null_gpus
            gate.wait(timeout=0.05)
    assert not stopped
    for entry in ticket.entries:
        scheduler.add_lora(entry.queue_pos, artifacts[entry.queue_pos], gpu=entry.queue_pos)
    scheduler.training_finished()
    for thread in threads:
        thread.join(timeout=30)
    assert all(not thread.is_alive() for thread in threads)
    assert {event[1] for event in served if event[0] == "serve"} == set(range(8))
    assert {gpu for gpu, model in rollouts} == set(range(8))
    assert lora_gpus == {6, 7}
    null, scored, dropped, stopped_at_limit = scheduler.finish()
    assert null == ("P" * 64,) and all(scored[pos] == ("P" * 64,) for pos in range(6))
    assert not dropped and not stopped_at_limit


def test_worker_deadline_before_runner_poll_keeps_shared_cutoff_prefix(build):
    params = build.params(n_tasks=8, n_gpus=2, looks=(0.5, 1.0), eval_chunk_tasks=4, min_scored_tasks=2, promote_reserve_s=10)
    ticket = build.ticket(n_entries=2, n_tasks=8, params=params)
    artifacts = {entry.queue_pos: build.artifact(ticket.head, entry, params) for entry in ticket.entries}
    evaluator = SimpleNamespace(http=None, _solo_clients=1, settings=SimpleNamespace(clients_per_server=1), clients_for=lambda params: 1, _solo_servers={gpu: SimpleNamespace(alive=lambda: True) for gpu in range(2)})
    evaluator._serve_one = lambda *args: "http://gpu"
    evaluator._load_one = lambda *args: None
    evaluator._rollout = lambda *args: (_ for _ in ()).throw(_Deadline())
    evaluator._solo_scheduler = None

    def solo_worker(ticket, trained, deadline, gpu):
        scheduler = evaluator._solo_scheduler
        for index in range(4):
            for model in (NULL_MODEL, *(lora.lora_hash for lora in artifacts.values())):
                scheduler.result(model, 0, index, "P")
        # Extra null work does not extend the shared prefix at the cutoff.
        for index in range(4, 6):
            scheduler.result(NULL_MODEL, 0, index, "P")
        return SoloWorker(evaluator, ticket, trained, deadline, gpu)

    evaluator.solo_worker = solo_worker
    evaluator.close = lambda: None
    runner = SoloRunner(SimpleNamespace(train=lambda request: artifacts[request.entry.queue_pos], close=lambda: None), evaluator, lambda: 1000, build.image, poll_s=1000)
    result = runner.run(ticket)
    assert result.stopped_at_limit
    assert result.null_outcomes == ("P" * 4 + "-" * 4,)
    assert all(entry.outcomes == ("P" * 4 + "-" * 4,) for entry in result.entries)
    assert evaluator._solo_scheduler.cut
    check_result(ticket, result)


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
            worker = Worker(gpu, events, {0: artifact})
            original_run = worker.run

            def run(**callbacks):
                # Match SoloWorker.run: prewarm is only an optimization; the
                # worker always performs the normal startup/retry itself.
                self._serve_one(ticket.head, ticket.params, len(ticket.entries), gpu)
                return original_run(**callbacks)

            worker.run = run
            return worker

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
                # Another GPU may still be training; this GPU may never train
                # again after its first server start (asserted below).
                events.append(("serve", gpu))
            return f"http://gpu{gpu}"

        def solo_worker(self, ticket, trained, deadline, gpu):
            worker = Worker(gpu, events, artifacts)
            original_run = worker.run

            def run(**callbacks):
                events.append(("worker", gpu))
                self._serve_one(ticket.head, ticket.params, len(ticket.entries), gpu)
                return original_run(**callbacks)

            worker.run = run
            return worker

        def close(self):
            pass

    class Trainer:
        def train(self, request):
            with lock:
                active.add(request.gpu_ids[0])
                events.append(("train", request.gpu_ids[0]))
            time.sleep(0.02)
            with lock:
                active.discard(request.gpu_ids[0])
                events.append(("train_end", request.gpu_ids[0]))
            return artifacts[request.entry.queue_pos]

        def close(self):
            pass

    evaluator = Evaluator()
    result = SoloRunner(Trainer(), evaluator, lambda: 1000, build.image, poll_s=0.01).run(ticket)
    assert result.null_outcomes == ("P" * 8,)
    assert all(entry.outcomes == ("P" * 8,) for entry in result.entries)
    for gpu in (0, 1):
        first_train = next(i for i, event in enumerate(events) if event == ("train", gpu))
        last_train_end = max(i for i, event in enumerate(events) if event == ("train_end", gpu))
        first_server = next((i for i, event in enumerate(events) if event == ("serve", gpu)), None)
        assert first_server is not None, events
        assert first_train < last_train_end < first_server
        assert not any(event == ("train", gpu) for event in events[first_server + 1:])
