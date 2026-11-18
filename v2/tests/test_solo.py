"""Solo scheduling: LPT, null priority, work stealing and seed pairing."""

from __future__ import annotations

import dataclasses
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from sn120_v2.errors import InfraError
from sn120_v2.gpu.solo import SoloRunner, SoloScheduler, SoloSettings, choose_claim, lpt_train_all
from sn120_v2.judging import check_result
from sn120_v2.public import round_text
from sn120_v2.sampling import rollout_seed
from sn120_v2.types import FAIL, NOT_RUN, NULL_MODEL, PASS


def test_lpt_is_longest_first_and_stable(build):
    entries = [dataclasses.replace(build.entry(i), batch=dataclasses.replace(build.entry(i).batch, n_rollouts=n)) for i, n in enumerate((5, 100, 20))]
    assert [entry.queue_pos for entry in lpt_train_all(entries)] == [1, 2, 0]
    assert [entry.queue_pos for entry in lpt_train_all(entries, {0: 900, 1: 5, 2: 5})] == [0, 1, 2]


def test_stealing_claims_each_block_once_and_prefers_null():
    remaining = {NULL_MODEL: 0, "a": 0, "b": 0}
    first = choose_claim(remaining, {"a", "b"}, 100, 16, "a")
    assert first == (NULL_MODEL, 0, 16)
    remaining[NULL_MODEL] = 16
    own = choose_claim(remaining, {"a", "b"}, 100, 16, "a")
    assert own == ("a", 0, 16)
    remaining["a"] = 16
    stolen = choose_claim(remaining, {"a", "b"}, 100, 16, "other")
    assert stolen == ("b", 0, 16)


def test_scheduler_buffers_out_of_order_results_and_pairs_seeds(build):
    ticket = build.ticket(n_entries=2, n_tasks=8, params=build.params(n_tasks=8, looks=(0.5, 1.0), eval_chunk_tasks=4))
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=2), clock=lambda: 1100, evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 10000)
    loras = {entry.queue_pos: build.artifact(ticket.head, entry, ticket.params) for entry in ticket.entries}
    for pos, lora in loras.items():
        scheduler.add_lora(pos, lora)
    seen = []
    for pos, lora in loras.items():
        for index in reversed(range(8)):
            scheduler.result(lora.lora_hash, 0, index, "P")
    for index in reversed(range(8)):
        scheduler.result(NULL_MODEL, 0, index, "P")
    assert all(len(text) == 8 for text in scheduler.null)
    assert all(len(text) == 8 for texts in scheduler.scored.values() for text in texts)
    for model in (NULL_MODEL, *(lora.lora_hash for lora in loras.values())):
        assert [rollout_seed(ticket.draw.round_seed, task, 0) for task in ticket.draw.task_ids] == [rollout_seed(ticket.draw.round_seed, task, 0) for task in ticket.draw.task_ids]


def test_deadline_scores_the_shared_completed_block_prefix(build):
    params = build.params(n_tasks=20, looks=(0.5, 1.0), eval_chunk_tasks=4, min_scored_tasks=2)
    ticket = build.ticket(n_entries=2, n_tasks=20, params=params)
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=1), evaluator=SimpleNamespace(http=None), clock=lambda: 1200, image_digest=build.image)
    scheduler = SoloScheduler(runner, ticket, 10000)
    loras = {entry.queue_pos: build.artifact(ticket.head, entry, ticket.params) for entry in ticket.entries}
    for pos, lora in loras.items():
        scheduler.add_lora(pos, lora, gpu=pos)
    for index in range(12):
        for model in (NULL_MODEL, *(lora.lora_hash for lora in loras.values())):
            scheduler.result(model, 0, index, PASS)
    for model in (NULL_MODEL, loras[0].lora_hash):
        for index in range(12, 17):
            scheduler.result(model, 0, index, PASS)
    scheduler.deadline_cut()
    null, scored, dropped, stopped = scheduler.finish()
    assert stopped and not dropped
    assert null == (PASS * 12 + NOT_RUN * 8,)
    assert all(scored[pos] == (PASS * 12 + NOT_RUN * 8,) for pos in loras)
    result = SoloRunner._solo_result(runner, ticket, loras, null, scored, dropped, stopped, 1000, 1100)
    assert result.stopped_at_limit
    assert result.null_outcomes == null and all(entry.outcomes == scored[pos] for pos, entry in enumerate(result.entries))
    summary = {"round_id": ticket.round_id, "outcome": "no_winner", "stopped_at_limit": result.stopped_at_limit, "head_version_id": ticket.head.version_id, "seats": len(ticket.entries), "n_tasks": len(ticket.draw.task_ids)}
    assert "(stopped at the time limit)" in round_text(summary)
    check_result(ticket, result)


def test_deadline_with_no_shared_completed_block_fails_closed(build):
    ticket = build.ticket(n_entries=1, n_tasks=8, params=build.params(n_tasks=8, eval_chunk_tasks=4))
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=1), evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 10000)
    scheduler.add_lora(0, build.artifact(ticket.head, ticket.entries[0], ticket.params))
    for index in range(4):
        scheduler.result(NULL_MODEL, 0, index, PASS)
    scheduler.deadline_cut()
    with pytest.raises(InfraError, match="before null and every live LoRA completed a shared block"):
        scheduler.finish()


def test_steal_waits_after_null_for_a_lora_then_exits_when_all_work_is_done(build):
    ticket = build.ticket(n_entries=1, n_tasks=4, params=build.params(n_tasks=4, n_gpus=1, looks=(1.0,), eval_chunk_tasks=1))
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=1), evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 10000)
    lora = build.artifact(ticket.head, ticket.entries[0], ticket.params)
    claims = []
    stopped = threading.Event()

    def worker():
        while (claim := scheduler.steal(0, NULL_MODEL)) is not None:
            claims.append(claim)
        stopped.set()

    thread = threading.Thread(target=worker)
    thread.start()
    for index in range(4):
        scheduler.result(NULL_MODEL, 0, index, PASS)
    assert not stopped.wait(0.05), "null exhaustion must not end a worker while training can add a LoRA"
    scheduler.add_lora(0, lora, gpu=0)
    deadline = time.monotonic() + 5
    while len(claims) < 5:
        assert time.monotonic() < deadline, claims
        time.sleep(0.005)
    scheduler.training_finished()
    thread.join(timeout=5)
    assert not thread.is_alive() and stopped.is_set()
    assert [model for model, _, _ in claims] == [NULL_MODEL] * 4 + [lora.lora_hash] * 4


def test_lora_that_finishes_a_look_before_null_is_dropped_when_null_catches_up(build):
    ticket = build.ticket(n_entries=1, n_tasks=16, params=build.params(n_tasks=16, looks=(0.25, 1.0), eval_chunk_tasks=4, min_scored_tasks=2))
    runner = SimpleNamespace(settings=SoloSettings(steal_block_tasks=2), clock=lambda: 1100, evaluator=SimpleNamespace(http=None))
    scheduler = SoloScheduler(runner, ticket, 10000)
    lora = build.artifact(ticket.head, ticket.entries[0], ticket.params)
    scheduler.add_lora(0, lora, gpu=0)
    # The candidate has completed the whole first look and speculated beyond
    # it, but null has not reached the look yet: no decision is possible yet.
    for index in range(8):
        scheduler.result(lora.lora_hash, 0, index, FAIL if index % 3 else PASS)
    assert scheduler.dropped == {} and scheduler.judged == set()
    for index in range(4):
        scheduler.result(NULL_MODEL, 0, index, PASS)
    assert scheduler.dropped == {0: 0}
    assert scheduler.scored[0] == ["".join(FAIL if index % 3 else PASS for index in range(4))]
    assert len(scheduler.scored[0][0]) == 4
    scheduler.result(lora.lora_hash, 0, 8, PASS)
    assert len(scheduler.scored[0][0]) == 4
