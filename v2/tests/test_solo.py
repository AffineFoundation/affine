"""Solo scheduling: LPT, null priority, work stealing and seed pairing."""

from __future__ import annotations

import dataclasses
import queue
import threading
from types import SimpleNamespace

from sn120_v2.gpu.solo import SoloScheduler, SoloSettings, choose_claim, lpt_train_all
from sn120_v2.sampling import rollout_seed
from sn120_v2.types import NULL_MODEL


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
