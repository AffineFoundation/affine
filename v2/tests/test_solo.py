"""Solo scheduling: LPT, null priority, work stealing and seed pairing."""

from __future__ import annotations

import dataclasses
import queue
import threading
from types import SimpleNamespace

from sn120_v2.gpu.solo import SoloScheduler, SoloSettings, choose_claim, lpt_train_all
from sn120_v2.sampling import rollout_seed
from sn120_v2.types import FAIL, NULL_MODEL, PASS


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
