"""Streaming is a scheduling change, not a change to the look rules."""

from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import pytest

from sn120_v2.errors import InfraError
from sn120_v2.gpu.evaluator import CancellableHttp
from sn120_v2.judging import check_result, decide
from sn120_v2.types import FAIL, NULL_MODEL, PASS, EvalChunk
from sn120_v2.worker import LocalRunner


class OutcomeEvaluator:
    """One fixed simulated outcome set, independent of the scheduling path."""

    image = "sim-image"

    def __init__(self, ticket, effects):
        self.ticket, self.effects = ticket, effects
        self.calls, self.sent, self.cancelled = [], [], set()
        self.names = {}

    def outcome(self, model, sample, index):
        if model == NULL_MODEL:
            return self.effects[0][index]
        pos = next(pos for pos, name in self.names.items() if name == model)
        return self.effects[pos + 1][index]

    def evaluate(self, request):
        self.calls.append(request)
        names = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
        outcomes = {}
        for model in names:
            outcomes[model] = tuple("".join(self.outcome(model, sample, self.ticket.draw.task_ids.index(task)) for task in request.task_ids) for sample in range(request.params.null_samples if model == NULL_MODEL else request.params.samples_per_task))
        return EvalChunk(request.look, request.task_ids, outcomes, self.image)

    def close(self):
        pass


class StreamingOutcomeEvaluator(OutcomeEvaluator):
    def stream(self, request, result, eliminated, done=lambda: False):
        self.calls.append(request)
        models = [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]
        for index in range(len(request.task_ids)):
            if done():
                break
            for model in models:
                if eliminated(model):
                    self.cancelled.add(model)
                    continue
                self.sent.append((model, index))
                for sample in range(request.params.null_samples if model == NULL_MODEL else request.params.samples_per_task):
                    result(model, sample, index, self.outcome(model, sample, index))
        return len(request.task_ids)


def make_world(build, n_tasks=64, n_loras=3, effects=None, params=None):
    params = params or build.params(n_tasks=n_tasks, looks=(0.25, 0.5, 1.0), eval_chunk_tasks=8, min_scored_tasks=8, batch_cap=n_loras)
    ticket = build.ticket(n_entries=n_loras, n_tasks=n_tasks, params=params)
    ticket = dataclasses.replace(ticket, issued_at=1000, expires_at=100000)
    trained = {entry.queue_pos: build.artifact(ticket.head, entry, params) for entry in ticket.entries}
    trainer = SimpleNamespace(train=lambda request: trained[request.entry.queue_pos], close=lambda: None)
    effects = effects or {0: [PASS] * n_tasks, **{pos + 1: [PASS if index % (pos + 2) else FAIL for index in range(n_tasks)] for pos in range(n_loras)}}
    return ticket, trained, trainer, effects


def compare(build, effects=None, params=None, n_tasks=64, n_loras=3):
    ticket, trained, trainer, effects = make_world(build, n_tasks, n_loras, effects, params)
    clock = lambda: ticket.issued_at + 100
    old = OutcomeEvaluator(ticket, effects)
    new = StreamingOutcomeEvaluator(ticket, effects)
    for seam in (old, new):
        seam.names = {pos: trained[pos].lora_hash for pos in trained}
    baseline = LocalRunner(trainer, old, clock, build.image).run(ticket)
    streamed = LocalRunner(trainer, new, clock, build.image).run(ticket)
    assert streamed.null_outcomes == baseline.null_outcomes
    assert [(e.queue_pos, e.status, e.outcomes, e.eliminated_at_look) for e in streamed.entries] == [(e.queue_pos, e.status, e.outcomes, e.eliminated_at_look) for e in baseline.entries]
    assert streamed.stopped_at_limit == baseline.stopped_at_limit
    assert [(e.uid, e.status, e.score) for e in decide([(ticket, streamed)]).entries] == [(e.uid, e.status, e.score) for e in decide([(ticket, baseline)]).entries]
    check_result(ticket, streamed)
    return ticket, streamed, baseline, new


def test_king_two_samples_match_block_and_stream_paths(build):
    n = 64
    effects = {0: [PASS] * n}
    for pos in range(3):
        effects[pos + 1] = [FAIL if (index * (pos + 1)) % 5 < 3 else PASS for index in range(n)]
    params = build.params(n_tasks=n, null_samples_per_task=2, min_scored_tasks=8)
    ticket, streamed, baseline, seam = compare(build, effects, params=params)
    assert len(streamed.null_outcomes) == 2
    assert all(len(entry.outcomes) == 1 for entry in streamed.entries)
    assert len([model for model, index in seam.sent if model == NULL_MODEL]) == n
    assert len(seam.calls) == 1 and seam.calls[0].params.null_samples == 2


def test_look_decisions_match_the_chunked_path_on_the_same_simulated_outcomes(build):
    n = 64
    effects = {0: [PASS] * n}
    for pos in range(3):
        effects[pos + 1] = [FAIL if (index * (pos + 1)) % 5 < 3 else PASS for index in range(n)]
    compare(build, effects)


@pytest.mark.parametrize("drop_look", [0, 1])
def test_every_drop_is_the_same_and_speculation_does_not_leak_into_the_result(build, drop_look):
    n = 64
    effects = {0: [PASS] * n, 1: [FAIL if index % 3 else PASS for index in range(n)], 2: [PASS] * n}
    if drop_look:
        effects[1][:16] = [PASS] * 16
        effects[1][16:32] = [FAIL] * 16
    ticket, streamed, baseline, seam = compare(build, effects, n_loras=2)
    expected = [(entry.queue_pos, entry.eliminated_at_look) for entry in baseline.entries]
    assert [(entry.queue_pos, entry.eliminated_at_look) for entry in streamed.entries] == expected
    assert (0, drop_look) in expected
    assert not any(model == seam.names[0] and index > (16 if drop_look == 0 else 32) for model, index in seam.sent)


def test_cancellation_disconnects_only_the_eliminated_model():
    class Reply:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    http = CancellableHttp()
    replies = {"bad": Reply(), "good": Reply()}
    with http._lock:
        http._active = {name: {reply} for name, reply in replies.items()}
    http.cancel("bad")
    assert replies["bad"].closed and not replies["good"].closed
    assert "bad" in http._cancelled and "good" not in http._cancelled


class RetryEvaluator(StreamingOutcomeEvaluator):
    def __init__(self, ticket, effects):
        super().__init__(ticket, effects)
        self.attempts = 0

    def stream(self, request, result, eliminated, done=lambda: False):
        self.attempts += 1
        if self.attempts == 1:
            for model in [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]:
                for index in range(8):
                    result(model, 0, index, self.outcome(model, 0, index))
            raise InfraError("simulated dead server")
        return super().stream(request, result, eliminated, done)


def test_retry_reuses_outcomes_and_rechecks_looks(build):
    ticket, trained, trainer, effects = make_world(build)
    seam = RetryEvaluator(ticket, effects)
    seam.names = {pos: trained[pos].lora_hash for pos in trained}
    result = LocalRunner(trainer, seam, lambda: ticket.issued_at + 100, build.image).run(ticket)
    assert seam.attempts == 2
    assert all(entry.outcomes == ("P" * 64,) or entry.outcomes for entry in result.entries)
    check_result(ticket, result)


class DeadlineEvaluator(OutcomeEvaluator):
    def __init__(self, ticket, effects, stop_after_blocks):
        super().__init__(ticket, effects)
        self.stop_after_blocks = stop_after_blocks
        self.completed = 0
        self.clock = ticket.issued_at + 100

    def evaluate(self, request):
        chunk = super().evaluate(request)
        self.completed += 1
        if self.completed == self.stop_after_blocks:
            self.clock = request.deadline + 1
        return chunk


class StreamingDeadlineEvaluator(StreamingOutcomeEvaluator):
    def __init__(self, ticket, effects, stop_after_blocks):
        super().__init__(ticket, effects)
        self.stop_after_blocks = stop_after_blocks
        self.clock = ticket.issued_at + 100

    def stream(self, request, result, eliminated, done=lambda: False):
        limit = (self.stop_after_blocks - 1) * request.params.eval_chunk_tasks
        for model in [NULL_MODEL, *(lora.lora_hash for lora in request.loras)]:
            for index in range(limit + 4):
                result(model, 0, index, self.outcome(model, 0, index))
        self.clock = request.deadline + 1
        self.calls.append(request)
        return limit


def test_deadline_keeps_only_complete_shared_blocks(build):
    ticket, trained, trainer, effects = make_world(build)
    old = DeadlineEvaluator(ticket, effects, 3)
    new = StreamingDeadlineEvaluator(ticket, effects, 3)
    for seam in (old, new):
        seam.names = {pos: trained[pos].lora_hash for pos in trained}
    baseline = LocalRunner(trainer, old, lambda: old.clock, build.image).run(ticket)
    streamed = LocalRunner(trainer, new, lambda: new.clock, build.image).run(ticket)
    assert baseline.null_outcomes == streamed.null_outcomes
    assert [entry.outcomes for entry in baseline.entries] == [entry.outcomes for entry in streamed.entries]
    assert streamed.stopped_at_limit and not baseline.stopped_at_limit
    assert all(len(text.rstrip("-")) == 16 for text in streamed.null_outcomes)
    check_result(ticket, streamed)
