"""Streaming looks and deadlines: exact prefixes, old-chunk parity and refilling fake servers."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from sn120_v2 import judging
from sn120_v2.judging import check_result
from sn120_v2.errors import InfraError
from sn120_v2.types import EvalBlock, EvalRequest, NULL_MODEL, NOT_RUN, eval_blocks
from sn120_v2.worker import LocalRunner


def codes(seed: int, model: str, n: int, samples: int) -> tuple[str, ...]:
    """Deterministic heterogeneous codes, independent of the artifact's hash."""
    return tuple("".join("P" if (seed * 17 + index * 13 + sample * 7) % 5 < 2 else "F" for index in range(n)) for sample in range(samples))


class FakeStream:
    """Full deterministic outcomes, delivered in arbitrary model/block order."""

    def __init__(self, ticket, seed: int = 0, shuffle: bool = True, build=None):
        self.ticket, self.seed, self.shuffle, self.build = ticket, seed, shuffle, build
        self.outcomes = {NULL_MODEL: codes(seed, NULL_MODEL, len(ticket.draw.task_ids), ticket.params.samples_per_task)}
        self.outcomes.update({model: codes(seed + position + 1, model, len(ticket.draw.task_ids), ticket.params.samples_per_task) for position, model in enumerate(self.models(ticket))})
        self.sent = []
        self.closed = 0

    def models(self, ticket):
        return tuple(self.build.artifact(ticket.head, entry, ticket.params).lora_hash for entry in ticket.entries)

    def evaluate(self, request: EvalRequest) -> SimpleNamespace:
        """The old interface, used only by the reference chunk loop below."""
        start = self.ticket.draw.task_ids.index(request.task_ids[0])
        stop = start + len(request.task_ids)
        return SimpleNamespace(task_ids=request.task_ids, outcomes={model: tuple(row[start:stop] for row in self.outcomes[model]) for model in (NULL_MODEL, *(lora.lora_hash for lora in request.loras))})

    def stream(self, request: EvalRequest, limit):
        assert request.task_ids == self.ticket.draw.task_ids
        blocks = []
        size = request.params.eval_chunk_tasks
        for model in (NULL_MODEL, *(lora.lora_hash for lora in request.loras)):
            self.models_sent = getattr(self, "models_sent", set()) | {model}
            for start, end in eval_blocks(size, request.params.look_sizes(len(request.task_ids))):
                blocks.append(EvalBlock(model, start, end, tuple(row[start:end] for row in self.outcomes[model])))
        if self.shuffle:
            random.Random(self.seed).shuffle(blocks)
        for block in blocks:
            self.sent.append(block)
            yield block
            if all(limit(model) == 0 for model in (NULL_MODEL, *(lora.lora_hash for lora in request.loras))):
                break

    def close(self):
        self.closed += 1


def old_evaluate(runner: LocalRunner, ticket, trained, deadline: int):
    """The pre-stream LocalRunner._evaluate, verbatim in its decisions, on EvalChunk fakes."""
    params, draw = ticket.params, ticket.draw
    n, samples = len(draw.task_ids), params.samples_per_task
    null = [""] * samples
    scored = {pos: [""] * samples for pos in trained}
    dropped = {}
    alive, start, stopped = sorted(trained), 0, False
    try:
        for look, stop in enumerate(params.look_sizes(n)):
            while start < stop and not stopped:
                if runner.clock() >= deadline:
                    stopped = True
                    break
                end = min(stop, start + params.eval_chunk_tasks)
                loras = tuple(trained[pos] for pos in alive)
                chunk = runner.evaluator.evaluate(EvalRequest(ticket.round_id, ticket.head, loras, draw.task_ids[start:end], draw.round_seed, look, params, deadline, len(ticket.entries)))
                if chunk is None:
                    stopped = True
                    break
                for sample in range(samples):
                    null[sample] += chunk.outcomes[NULL_MODEL][sample]
                    for pos in alive:
                        scored[pos][sample] += chunk.outcomes[trained[pos].lora_hash][sample]
                start = end
            if stopped:
                break
            for pos in list(alive):
                if judging.drop_at_look(judging.score_entry(draw.weights, null, scored[pos], params), look, n, params):
                    alive.remove(pos)
                    dropped[pos] = look
            if not alive:
                break
    finally:
        runner.evaluator.close()
    pad = lambda strings: tuple(text + NOT_RUN * (n - len(text)) for text in strings)
    return pad(null), {pos: pad(strings) for pos, strings in scored.items()}, dropped, stopped


def trained(build, ticket):
    return {entry.queue_pos: build.artifact(ticket.head, entry, ticket.params) for entry in ticket.entries}


@pytest.mark.parametrize("seed", range(40))
def test_stream_matches_the_old_chunk_loop_with_out_of_order_blocks(build, seed):
    n = 12 if seed % 2 else 20
    params = build.params(n_tasks=n, looks=(0.25, 0.5, 1.0), eval_chunk_tasks=4, samples_per_task=2, drop_alpha=0.5, z_bar=0.0)
    ticket = build.ticket(n_entries=3, n_tasks=n, params=params)
    artifacts = trained(build, ticket)
    first = LocalRunner(SimpleNamespace(), FakeStream(ticket, seed, build=build), lambda: 1000, build.image)
    second = LocalRunner(SimpleNamespace(), FakeStream(ticket, seed, build=build), lambda: 1000, build.image)
    old = old_evaluate(first, ticket, artifacts, 10**9)
    new = second._evaluate(ticket, artifacts, 10**9)
    assert new == old
    assert len({block.model for block in second.evaluator.sent}) == 4


@pytest.mark.parametrize("drop_at", [0, 1, None])
def test_drop_lengths_speculation_and_null_stop(build, drop_at):
    n, size = 12, 4
    params = build.params(n_tasks=n, looks=(1 / 3, 2 / 3, 1.0), eval_chunk_tasks=size, drop_alpha=0.5, z_bar=0.0)
    ticket = build.ticket(n_entries=1, n_tasks=n, params=params)
    artifacts = trained(build, ticket)
    seam = FakeStream(ticket, shuffle=False, build=build)
    bad = "PFPFFFFF" if drop_at == 1 else "FFFF" if drop_at == 0 else "PPPP"
    name = artifacts[0].lora_hash
    seam.outcomes[name] = (bad + "P" * (n - len(bad)),)
    runner = LocalRunner(SimpleNamespace(), seam, lambda: 1000, build.image)
    null, scored, dropped, stopped = runner._evaluate(ticket, artifacts, 10**9)
    expected = 12 if drop_at is None else params.look_sizes(n)[drop_at]
    assert scored[0] == ((bad + "P" * (n - len(bad))) if drop_at is None else bad + NOT_RUN * (n - len(bad)),)
    assert dropped == ({} if drop_at is None else {0: drop_at}) and not stopped
    if drop_at == 0:
        assert scored[0][0].count(NOT_RUN) == n - size, "speculative replies past the drop are discarded from the outcome"
    if drop_at is None:
        assert null == seam.outcomes[NULL_MODEL]
    elif drop_at == 0:
        assert null == (seam.outcomes[NULL_MODEL][0][:expected] + NOT_RUN * (n - expected),)


@pytest.mark.parametrize("cut", [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
def test_deadline_at_many_points_has_common_prefix_and_no_cut_errors(build, cut):
    n = 20
    params = build.params(n_tasks=n, looks=(0.25, 0.5, 1.0), eval_chunk_tasks=4, samples_per_task=1, drop_alpha=0.5, z_bar=0.0)
    ticket = build.ticket(n_entries=2, n_tasks=n, params=params)
    artifacts = trained(build, ticket)
    seam = FakeStream(ticket, seed=3, shuffle=True, build=build)
    clock = [1000]
    original = seam.stream
    def stream(request, limit):
        for block in original(request, limit):
            yield block
            if block.end >= cut:
                clock[0] = ticket.expires_at + 1
                if all(limit(model) == 0 for model in (NULL_MODEL, *(lora.lora_hash for lora in request.loras))):
                    break
    seam.stream = stream
    trainer = SimpleNamespace(train=lambda request: artifacts[request.entry.queue_pos], close=lambda: None)
    runner = LocalRunner(trainer, seam, lambda: clock[0], build.image)
    result = runner.run(ticket)
    check_result(ticket, result)
    if cut < params.look_sizes(n)[0]:
        assert result.stopped_at_limit and all(entry.status == "not_scored" for entry in result.entries)
    else:
        assert all("E" not in text for text in result.null_outcomes)
        assert all(len(entry.outcomes[0].rstrip(NOT_RUN)) == len(result.null_outcomes[0].rstrip(NOT_RUN)) for entry in result.entries if entry.status == "scored")


def test_deadline_restores_a_drop_beyond_the_common_prefix(build):
    n = 12
    params = build.params(n_tasks=n, looks=(1 / 3, 2 / 3, 1.0), eval_chunk_tasks=4, drop_alpha=0.5, z_bar=0.0)
    ticket = build.ticket(n_entries=1, n_tasks=n, params=params)
    artifacts = trained(build, ticket)
    seam = FakeStream(ticket, shuffle=False, build=build)
    name = artifacts[0].lora_hash
    seam.outcomes[name] = ("PFPFFFFF" + "P" * 4,)
    original = seam.stream
    clock = [1000]
    def stream(request, limit):
        for block in original(request, limit):
            if block.model == name and block.start >= 4:
                clock[0] = ticket.expires_at + 1
                break
            yield block
    seam.stream = stream
    trainer = SimpleNamespace(train=lambda request: artifacts[request.entry.queue_pos], close=lambda: None)
    result = LocalRunner(trainer, seam, lambda: clock[0], build.image).run(ticket)
    check_result(ticket, result)
    assert result.stopped_at_limit
    assert result.entries[0].eliminated_at_look is None
    assert result.entries[0].outcomes[0].rstrip(NOT_RUN) == "PFPF"
    assert len(result.null_outcomes[0].rstrip(NOT_RUN)) == 4


def test_eval_block_rejects_not_run_and_partial_samples(build):
    with pytest.raises(Exception):
        EvalBlock(NULL_MODEL, 0, 2, ("P-",))
    with pytest.raises(Exception):
        EvalBlock(NULL_MODEL, 0, 2, ("PP", "P"))
