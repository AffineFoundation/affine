"""Generator scheduling, job validation, budget cutoff and result aggregation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter

import pytest

from sn120_v2.canonical import canonical_json
from sn120_v2.envs.math_grader import Grade
from sn120_v2.errors import ContractError, VerifyError
from sn120_v2.generator import decisions
from sn120_v2.generator.constants import ACCEPT, CONTINUE, POOL_ID, REJECT
from sn120_v2.generator.epoch import epoch_deadline
from sn120_v2.generator.fake_vllm import PASS
from sn120_v2.generator.scoring import (
    ROLLOUT_VERSION,
    JOB_VERSION,
    UNSCORED,
    ScoringJob,
    TaskSpec,
    TaskState,
    append_rollout,
    hotkey_results,
    initial_states,
    make_results,
    next_batch,
    finish_grade,
    rollout_seed,
    settle,
    stage_one_rounds,
    undecided_order,
)


def make_job(owners=("a", "a", "b", "c"), **changes):
    fields = dict(
        version=JOB_VERSION,
        epoch="20261120T0000Z",
        tasks=tuple(TaskSpec(f"{index + 1:020x}", owner) for index, owner in enumerate(owners)),
        head_version="a" * 64,
        seed_secret="unit-test-secret",
        budget_deadline=epoch_deadline("20261120T0000Z").timestamp() + 13500,
        pool_pin=POOL_ID,
    )
    return ScoringJob(**(fields | changes))


def record(job, task_id, j, passed):
    return {
        "version": ROLLOUT_VERSION,
        "epoch": job.epoch,
        "task_id": task_id,
        "hotkey": job.owners[task_id],
        "j": j,
        "seed": rollout_seed(job.seed_secret, task_id, j),
        "head_version": job.head_version,
        "messages": [],
        "response": PASS,
        "capped": False,
        "grade": {"code": "P" if passed else "F", "passed": passed},
    }


def observe(job, states, task_id, count, passes):
    for j in range(count):
        row = record(job, task_id, j, j < passes)
        row["grade"] = {}
        append_rollout(states[task_id], row, job=job)
    state = states[task_id]
    while not state.terminal and state.n in state.responses:
        row = state.responses.pop(state.n)
        row["grade"] = {"code": "P" if row["j"] < passes else "F", "passed": row["j"] < passes}
        state.rollouts.append(row)
        finish_grade(state, row)
    settle(state)


def test_seed_matches_specification_and_differs_from_v2_round_seed():
    secret, task_id = "entropy", "a" * 20
    expected = int(hashlib.sha256(f"{secret}:{task_id}:0".encode()).hexdigest()[:12], 16)
    assert rollout_seed(secret, task_id, 0) == expected
    assert 0 <= expected < 2**48
    assert rollout_seed(secret, task_id, 0) != rollout_seed(secret, task_id, 1)


def test_job_round_trip_owner_order_and_exact_pool_pin():
    job = make_job()
    assert ScoringJob.from_json(canonical_json(job.to_dict())) == job
    assert ScoringJob.from_dict(job.to_dict() | {"head_version": {"version_id": "a" * 64}}).head_version == "a" * 64
    assert job.task_ids == tuple(f"{i + 1:020x}" for i in range(4))
    for field, value in (
        ("epoch", "20261120T0100Z"),
        ("pool_pin", "another-pool"),
        ("head_version", ""),
        ("seed_secret", ""),
        ("budget_deadline", True),
    ):
        with pytest.raises(ContractError):
            ScoringJob.from_dict(job.to_dict() | {field: value})
    with pytest.raises(ContractError):
        ScoringJob.from_dict(job.to_dict() | {"extra": "not-allowed"})
    with pytest.raises(ContractError):
        make_job(tasks=(TaskSpec("a" * 20, "a"), TaskSpec("a" * 20, "b")))
    with pytest.raises(ContractError):
        make_job(tasks=(TaskSpec("bad-id", "a"),))


def test_stage_one_fair_round_robin_keeps_all_owners_at_equal_progress():
    job = make_job(("a", "a", "a", "a", "b", "b", "c"))
    states = initial_states(job)
    observed = []
    for _ in range(7):
        batch, stage = next_batch(job, states, 8)
        assert stage == "stage1" and len(batch) == 8
        task_id = batch[0][0]
        assert [j for _, j in batch] == list(range(8))
        observe(job, states, task_id, 8, 4)
        observed.append(job.owners[task_id])
    assert observed == ["a", "b", "c", "a", "b", "a", "a"]
    assert {task.task_id: states[task.task_id].n for task in job.tasks} == {
        task.task_id: 8 for task in job.tasks
    }


def test_capacity_batches_do_not_let_one_owner_take_the_whole_farm():
    job = make_job(tuple("a" for _ in range(16)) + tuple("b" for _ in range(16)))
    states = initial_states(job)
    batch, stage = next_batch(job, states, 256)
    assert stage == "stage1"
    owners = Counter(job.owners[task_id] for task_id, _ in batch)
    assert owners == {"a": 128, "b": 128}
    first_tasks = [task_id for task_id, sample in batch if sample == 0]
    assert first_tasks[:4] == [f"{i:020x}" for i in (1, 17, 2, 18)]


def test_early_decisions_do_not_leave_scheduler_stuck():
    job = make_job(("a", "b", "c"))
    states = initial_states(job)
    observe(job, states, job.task_ids[0], 8, 4)
    assert states[job.task_ids[0]].decision == ACCEPT
    batch, _ = next_batch(job, states, 32)
    assert Counter(task_id for task_id, _ in batch) == {job.task_ids[1]: 8, job.task_ids[2]: 8}
    assert all(len(group) == 8 for group in stage_one_rounds(job, states))


def test_closest_q_first_with_job_order_ties():
    job = make_job(("a", "b", "c", "d"))
    states = initial_states(job)
    # q(2,8) is closer to .5 than q(1,8); q(6,8) mirrors q(2,8).
    for task_id, passed in zip(job.task_ids, (1, 2, 6, 4)):
        observe(job, states, task_id, 8, passed)
    ordered = undecided_order(job, states)
    assert ordered
    assert [group[0][0] for group in ordered[:2]] == [job.task_ids[1], job.task_ids[2]]
    assert all(len(group) == 8 for group in ordered)
    assert all(group[0][1] == 8 for group in ordered)


def test_decision_thresholds_are_delegated_and_minimum_is_enforced():
    state = TaskState(TaskSpec("a" * 20, "a"))
    for n, s, expected in ((8, 4, ACCEPT), (8, 0, REJECT), (8, 2, UNSCORED)):
        state.s, state.n, state.final, state.decision = s, n, False, UNSCORED
        settle(state)
        assert state.decision == expected
        assert state.q == decisions.q(s, n)
    for n in range(8):
        state = TaskState(TaskSpec("a" * 20, "a"), n=n, s=n)
        settle(state, exhausted=True)
        assert state.final and state.decision == UNSCORED


def test_final_decision_at_32_uses_q_half():
    state = TaskState(TaskSpec("a" * 20, "a"), n=32, s=8)
    settle(state)
    assert state.decision == ACCEPT and state.final
    state = TaskState(TaskSpec("a" * 20, "a"), n=32, s=25)
    settle(state)
    assert state.decision == REJECT and state.final


def test_budget_exhausted_undecided_uses_final_half_threshold():
    state = TaskState(TaskSpec("a" * 20, "a"), n=16, s=7)
    settle(state, exhausted=True)
    assert state.decision == ACCEPT and state.final
    assert "budget exhausted" in state.reason
    state = TaskState(TaskSpec("a" * 20, "a"), n=16, s=1)
    settle(state, exhausted=True)
    assert state.decision == REJECT and state.final


def test_unscored_never_counts_as_accepted_and_hotkey_counts_add():
    job = make_job(("a", "a", "b"))
    states = initial_states(job)
    for state in states.values():
        state.final = True
    states[job.task_ids[0]].n = states[job.task_ids[0]].s = 8
    states[job.task_ids[0]].decision = ACCEPT
    states[job.task_ids[0]].rollouts = [record(job, job.task_ids[0], j, True) for j in range(8)]
    states[job.task_ids[1]].n = 3
    states[job.task_ids[1]].rollouts = [record(job, job.task_ids[1], j, False) for j in range(3)]
    counts = hotkey_results(job, states)
    assert counts["a"] | {} == {
        "submitted": 2, "eligible": 2, "scored": 1, "accepted": 1,
        "rejected": 0, "unscored": 1, "rollouts": 11, "passes": 8,
    }
    result = make_results(job, states, timings={}, throughput={}, servers=[])
    assert result["seed_secret"] == job.seed_secret
    assert result["seed_commitment"] == hashlib.sha256(job.seed_secret.encode()).hexdigest()
    assert result["tasks"][0]["q"] is not None


def test_replay_rejects_different_job_seed_head_owner_or_sequence():
    job = make_job(("a",))
    task_id = job.task_ids[0]
    state = initial_states(job)[task_id]
    row = record(job, task_id, 0, True)
    row["grade"] = {}
    for change in ({"seed": 0}, {"head_version": "b" * 64}, {"hotkey": "b"}, {"j": 1}, {"epoch": "20261120T0400Z"}):
        fresh = TaskState(TaskSpec(task_id, "a"))
        with pytest.raises(VerifyError):
            append_rollout(fresh, row | change, job=job)
    append_rollout(state, row, job=job)
    with pytest.raises(VerifyError):
        append_rollout(state, row, job=job)


def test_out_of_order_raw_responses_wait_for_the_missing_prefix():
    job = make_job(("a",))
    task_id = job.task_ids[0]
    state = initial_states(job)[task_id]
    for j in (2, 0, 1):
        row = record(job, task_id, j, True)
        row["grade"] = {}
        append_rollout(state, row, job=job)
    assert state.n == 0 and list(state.responses) == [2, 0, 1]
    for j in range(3):
        row = state.responses.pop(state.n)
        row["grade"] = {"code": "P", "passed": True}
        state.rollouts.append(row)
        finish_grade(state, row)
    assert (state.s, state.n) == (3, 3)
    assert [row["j"] for row in state.rollouts] == [0, 1, 2]
