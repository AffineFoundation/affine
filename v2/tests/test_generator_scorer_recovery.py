"""Fault-injection tests for the Generator pod scorer."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from sn120_v2.canonical import canonical_json
from sn120_v2.envs.math_grader import Grade
from sn120_v2.errors import ConfigError, InfraError, VerifyError
from sn120_v2.generator.epoch import epoch_deadline
from sn120_v2.generator.fake_vllm import FAIL, PASS, FakeFarm, FakePool
from sn120_v2.generator.http import BudgetDeadline, LocalHttp, loopback_url
from sn120_v2.generator.runner import recover, run_job
from sn120_v2.generator.scoring import JOB_VERSION, ScoringJob, TaskSpec, initial_states, load_rollouts, rollout_seed
from sn120_v2.generator.worker import BatchRunner, atomic_write, repair_checkpoints, request_body, trim_torn_line


def job_for(count=2, *, deadline=None, secret="recovery-secret"):
    epoch = "20261120T0000Z"
    return ScoringJob(
        version=JOB_VERSION,
        epoch=epoch,
        tasks=tuple(TaskSpec(f"{i + 40:020x}", f"owner-{i % 2}") for i in range(count)),
        head_version="b" * 64,
        seed_secret=secret,
        budget_deadline=deadline if deadline is not None else epoch_deadline(epoch).timestamp() + 13500,
        pool_pin="sn120-math-prior-v1:45b793b4d00d1f7225d34fd35e58018612a0ded00d96a9c23b750cfb18994115",
    )


def register(farm, job):
    for task in job.tasks:
        for sample in range(32):
            farm.seeds[rollout_seed(job.seed_secret, task.task_id, sample)] = (task.task_id, sample)


class Grades:
    def __init__(self):
        self.calls = 0

    def grade_many(self, jobs):
        self.calls += 1
        return [
            Grade("P", 1.0, False, "\\boxed{1}", "live") if "\\boxed{1}" in output and not capped
            else Grade("C" if capped else "F", 0.0, capped, None, "capped" if capped else "wrong")
            for _, output, capped in jobs
        ]


class BrokenGrader(Grades):
    def grade_many(self, jobs):
        raise InfraError("injected crash before any grade commit")


def test_request_body_is_exact_prod_math_sampling_contract():
    task = FakePool().tasks(("a" * 20,))[0]
    body = request_body(task, 42)
    assert body | {} == {
        "model": "null", "messages": task.messages(), "seed": 42, "n": 1,
        "max_tokens": 65536, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "skip_special_tokens": False, "chat_template_kwargs": {"enable_thinking": True},
    }


def test_only_literal_loopback_endpoints_are_accepted():
    assert loopback_url("http://127.0.0.1:8100") == "http://127.0.0.1:8100"
    assert loopback_url("http://[::1]:8100/") == "http://[::1]:8100"
    for bad in (
        "http://localhost:8100", "http://example.com:8100", "https://127.0.0.1:8100",
        "http://127.0.0.1.evil.test:8100", "http://127.0.0.1:8100/path",
        "http://user@127.0.0.1:8100", "http://169.254.0.1:8100", "http://127.0.0.1",
    ):
        with pytest.raises(ConfigError):
            loopback_url(bad)


def test_real_http_eight_fake_servers_balance_and_exact_parameters(tmp_path):
    job = job_for(16)
    states = initial_states(job)
    grader = Grades()
    with FakeFarm(first_port=0, mode="pass") as farm:
        register(farm, job)
        runner = BatchRunner(job, states, tmp_path, FakePool(), grader, urls=farm.urls, http=LocalHttp(), clients_per_server=4)
        batch = [(task.task_id, j) for task in job.tasks for j in range(2)]
        runner.run([], batch)
        assert Counter(request["seed"] for request in farm.requests).most_common(1)[0][1] == 1
        assert len({request["messages"][1]["content"] for request in farm.requests}) == 1
        assert all(request["n"] == 1 for request in farm.requests)
        servers = Counter(row["server"] for state in states.values() for row in state.rollouts)
        assert len(servers) == 8
        assert set(servers.values()) == {4}
        # With no early rejects, all 32 responses are committed in order.
        assert sum(state.n for state in states.values()) == 32


def test_crash_before_grade_recovers_without_duplicate_inference(tmp_path):
    job = job_for(1)
    out = tmp_path / "out"
    with FakeFarm(first_port=0, mode="pass") as farm:
        register(farm, job)
        with pytest.raises(InfraError, match="injected crash"):
            run_job(job, out, pool=FakePool(), grader=BrokenGrader(), urls=farm.urls, clients_per_server=2)
        first_requests = len(farm.requests)
        assert first_requests == 8
        assert not (out / "DONE").exists()
        # Eight durable raw responses, but no graded counts have been committed.
        assert len((out / "rollouts" / "responses" / f"{job.task_ids[0]}.jsonl").read_text().splitlines()) == 8
        states, completed = recover(job, out)
        assert not completed and states[job.task_ids[0]].n == 0
        result = run_job(job, out, pool=FakePool(), grader=Grades(), urls=farm.urls, clients_per_server=2)
        assert len(farm.requests) == first_requests
        assert result["tasks"][0]["n"] == 8 and result["tasks"][0]["s"] == 8
        assert (out / "DONE").is_file()
        assert len((out / "rollouts" / f"{job.task_ids[0]}.jsonl").read_text().splitlines()) == 8


def test_torn_tail_is_discarded_and_completed_runs_are_idempotent(tmp_path):
    job = job_for(1)
    out = tmp_path / "out"
    with FakeFarm(first_port=0, mode="mixed") as farm:
        register(farm, job)
        first = run_job(job, out, pool=FakePool(), grader=Grades(), urls=farm.urls, clients_per_server=2)
        requests = len(farm.requests)
        raw_path = out / "rollouts" / "responses" / f"{job.task_ids[0]}.jsonl"
        with raw_path.open("ab") as stream:
            stream.write(b'{"partial":')
        trim_torn_line(raw_path)
        assert all(json.loads(line) for line in raw_path.read_text().splitlines())
        assert run_job(job, out, pool=FakePool(), grader=Grades(), urls=farm.urls, clients_per_server=2) == first
        assert len(farm.requests) == requests


def test_incomplete_cutoff_output_can_be_safely_finalized_on_resume(tmp_path):
    job = job_for(3)
    out = tmp_path / "out"
    with FakeFarm(first_port=0, mode="mixed") as farm:
        register(farm, job)
        deadline = [job.budget_deadline - 1]
        original = farm.record

        def advance(body):
            original(body)
            deadline[0] += 0.1

        farm.record = advance
        result = run_job(job, out, pool=FakePool(), grader=Grades(), urls=farm.urls, clients_per_server=2, clock=lambda: deadline[0])
        assert result["timings"]["budget_exhausted"]
        assert all(row["decision"] != "unscored" or row["n"] < 8 and row["q"] is None for row in result["tasks"])
        # Same absolute deadline means no new inference; finalize the same evidence.
        before = len(farm.requests)
        resumed = run_job(job, out, pool=FakePool(), grader=Grades(), urls=farm.urls, clients_per_server=2, clock=lambda: deadline[0])
        assert resumed == result and len(farm.requests) == before


def test_budget_cancellation_closes_active_http_and_does_not_start_more(tmp_path):
    job = job_for(1)
    states = initial_states(job)

    class Transport:
        def __init__(self):
            self.calls = 0
            self.cancelled = threading.Event()

        def reset(self):
            self.cancelled.clear()

        def cancel(self):
            self.cancelled.set()

        def __call__(self, url, body, timeout):
            self.calls += 1
            job_deadline[0] = job.budget_deadline
            self.cancelled.wait(1)
            raise TimeoutError("budget elapsed")

    job_deadline = [time.time()]
    transport = Transport()
    runner = BatchRunner(job, states, tmp_path, FakePool(), Grades(), urls=[f"http://127.0.0.1:{19100 + i}" for i in range(8)], http=transport, clock=lambda: job_deadline[0], clients_per_server=1)
    assert runner.run([], [(job.task_ids[0], 0)]) == 0
    assert transport.calls == 1
    assert transport.cancelled.is_set()
    assert runner.cutoff


def test_done_rejects_a_different_private_job_or_missing_rollout(tmp_path):
    job = job_for(1)
    out = tmp_path / "out"
    with FakeFarm(first_port=0, mode="pass") as farm:
        register(farm, job)
        run_job(job, out, pool=FakePool(), grader=Grades(), urls=farm.urls)
    with pytest.raises(VerifyError, match="different private job"):
        recover(job_for(1, secret="changed-secret"), out)
    path = out / "rollouts" / f"{job.task_ids[0]}.jsonl"
    path.write_text("")
    with pytest.raises(VerifyError):
        recover(job, out)


def test_empty_unreached_rollout_file_exists(tmp_path):
    job = job_for(2)
    # Validation requires a post-epoch deadline; this timestamp is past only
    # relative to the actual wall clock, not relative to the epoch.
    out = tmp_path / "out"
    result = run_job(job, out, pool=FakePool(), grader=Grades(), urls=[f"http://127.0.0.1:{19200 + i}" for i in range(8)], http=LocalHttp(), clock=lambda: job.budget_deadline + 1)
    assert all(row["decision"] == "unscored" and row["q"] is None for row in result["tasks"])
    for task in job.tasks:
        assert (out / "rollouts" / f"{task.task_id}.jsonl").read_bytes() == b""
