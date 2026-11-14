"""The time limit of a row never holds the start of its worker: known-good cases make the workers of the grade pool ready first."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from types import SimpleNamespace

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import TIME_LIMIT, WARM_TRIES, Invalid, StoreAdmission, check_batch, env_grader, pool_grader
from sn120_v2.canonical import sha256_hex
from sn120_v2.envs.math_pool import load_pool
from sn120_v2.errors import InfraError

HOTKEY = "5MinerA"
THINK = "step " * 40
WORKERS = 8
START_ENV = "SN120_TEST_WORKER_START_S"  # seconds that a worker process of the real pool sleeps at its start
LIMIT_S, START_S, ROW_S = 4.0, 2.0, 2.5  # the start and the row are each inside the limit; the two together are not

if multiprocessing.current_process().name != "MainProcess":  # a worker process of the real pool imports this module at its start
    time.sleep(float(os.environ.get(START_ENV, "0")))


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


class Store(dict):
    put = dict.__setitem__
    get = dict.__getitem__

    def delete(self, key: str) -> None:
        self.pop(key, None)

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


class Pool:
    """The task "double N" has the answer 2N."""

    task_ids = tuple(f"t{n}" for n in range(50))

    def task(self, task_id: str) -> SimpleNamespace:
        n = int(task_id[1:])
        return SimpleNamespace(n=n, answer=str(2 * n), batch_prompt=lambda: f"double {n}")

    def task_for_prompt(self, prompt: str) -> SimpleNamespace | None:
        return self.task(f"t{prompt.split()[1]}") if prompt.startswith("double ") else None


class ColdPool:
    """``GradePool.grade_many`` as a model with no process and no clock.

    Job k of a call goes to worker k. A new worker needs most of the time limit to start. So a "slow" row (it
    needs most of the limit too) passes on a ready worker and gets the time limit on a new one. A "hang" row
    gets the time limit on every worker. The worker of a late job is killed, and a new one takes its place.
    With ``starts`` false a new worker never gets ready inside the limit.
    """

    def __init__(self, starts: bool = True) -> None:
        self.new, self.starts, self.calls = [True] * WORKERS, starts, []

    def grade_many(self, jobs) -> list[SimpleNamespace]:
        self.calls.append(len(jobs))
        return [self._one(k % WORKERS, *job) for k, job in enumerate(jobs)]

    def _one(self, worker: int, task, output: str, capped: bool) -> SimpleNamespace:
        late = "hang" in output or (self.new[worker] and ("slow" in output or not self.starts))
        self.new[worker] = late
        right = output.rsplit("\\boxed{", 1)[-1].rstrip("}") == task.answer
        return SimpleNamespace(reward=float(right and not late), machine_cause="time-limit" if late else None)


def file(word: str = "", at: int = 2, hotkey: str = HOTKEY) -> bytes:
    """10 prompts "double N", each with a right and a wrong rollout: 20 rows. ``word`` is in the right rollout of prompt ``at`` (row 2 * at)."""
    groups = [{"prompt": f"double {n}", "rollouts": [{"output": f"{THINK}{word + ' ' if n == at else ''}\\boxed{{{2 * n}}}", "reward": 1},
                                                      {"output": f"{THINK}\\boxed{{-1}}", "reward": 0}]} for n in range(10)]
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": groups}).encode()


def rule(raw: bytes, params, grader) -> str:
    try:
        check_batch(raw, HOTKEY, params, tokens=WordTokens(), grader=grader)
    except Invalid as exc:
        return exc.rule
    return ""


def test_first_file_of_a_new_process_does_not_pay_for_the_start_of_the_workers(build):
    """A right row that needs most of the time limit: valid, also as the first row that a new worker gets."""
    pool = ColdPool()
    assert rule(file("slow"), build.params(), pool_grader(Pool(), pool, chunk=WORKERS)) == ""
    assert pool.calls == [WORKERS, 8, 8, 4], "one known-good case for each worker, then the 20 rows of the file"
    assert rule(file("slow"), build.params(), pool_grader(Pool(), ColdPool(), chunk=WORKERS)) == rule(file(), build.params(), pool_grader(Pool(), ColdPool(), chunk=WORKERS)) == ""


def test_worker_that_replaces_a_killed_one_is_ready_before_the_next_row(build):
    """Row 12 of a file hangs: its worker (number 4) is killed. The known-good cases after it reach workers 0 to 3 only.

    The next file has a slow right row at the same place. It must not meet the new worker 4 before that worker is ready.
    """
    pool = ColdPool()
    official = pool_grader(Pool(), pool, chunk=WORKERS)
    assert admission.PROBE_TASKS <= 4 < WORKERS
    assert rule(file("hang", at=6), build.params(), official) == TIME_LIMIT
    assert pool.calls == [WORKERS, 8, 8, WORKERS, admission.PROBE_TASKS], "the file stops at its second chunk; the new worker is made ready; then the known-good cases"
    assert rule(file("slow", at=6), build.params(), official) == ""
    assert pool.calls[5:] == [8, 8, 4], "every worker was ready: no known-good case is sent again"


def test_worker_lost_at_a_known_good_case_is_ready_before_the_next_row(build):
    """The pool loses worker 2 at the known-good cases that follow a failed row. The next file has a slow right row for that worker."""

    class LossyPool(ColdPool):
        lost = False

        def grade_many(self, jobs):
            grades = super().grade_many(jobs)
            if len(jobs) == admission.PROBE_TASKS and not self.lost:  # once: the first call of this size is the known-good cases
                self.lost, self.new[2], grades[2] = True, True, SimpleNamespace(reward=0.0, machine_cause="worker-died")
            return grades

    pool = LossyPool()
    official = pool_grader(Pool(), pool, chunk=WORKERS)
    with pytest.raises(Invalid) as failed:
        check_batch(file("hang", at=6), HOTKEY, build.params(), tokens=WordTokens(), grader=official)
    assert failed.value.rule == TIME_LIMIT and not failed.value.proven, "a known-good case was lost: the row is not counted"
    assert rule(file("slow", at=1), build.params(), official) == ""
    assert pool.calls[5:] == [WORKERS, 8, 8, 4], "the known-good cases go to the new worker before the file"


def test_rows_after_a_killed_worker_in_the_same_call_wait_for_the_new_worker(build):
    """One call of the grader for many rows: the chunk after a row with no grade starts with the known-good cases."""
    pool = ColdPool()
    official = pool_grader(Pool(), pool, chunk=WORKERS)
    rows = [(f"double {n}", f"{THINK}{'hang ' if n == 4 else 'slow ' if n == 12 else ''}\\boxed{{{2 * n}}}", False) for n in range(16)]
    found = list(official.many(rows))
    assert [type(outcome).__name__ for outcome in found] == ["float"] * 4 + ["GraderTimeLimit"] + ["float"] * 11 and found[12] == 1.0
    assert pool.calls == [WORKERS, 8, WORKERS, 8]


def test_workers_that_do_not_get_ready_are_our_failure_with_no_verdict_and_no_count(build):
    pool = ColdPool(starts=False)
    adm = StoreAdmission(Store({layout.miner_object(HOTKEY): file()}), build.params(), tokens=WordTokens(), grader=pool_grader(Pool(), pool, chunk=WORKERS))
    for tries in (1, 2):  # loud each time; the next check tries again, and nothing is held against the file
        with pytest.raises(InfraError, match=f"the workers of the official grader are not ready: a known-good case has no grade after {WARM_TRIES} tries"):
            adm.verdicts([HOTKEY])
        assert pool.calls == [WORKERS] * WARM_TRIES * tries, "only known-good cases were sent: no row of the file"
        assert adm._tries == {} and adm._seen == {} and adm._held == {}
    pool.starts = True
    assert adm.verdicts([HOTKEY])[HOTKEY].valid


def test_grader_in_the_calling_process_sends_no_known_good_case_first(build):
    seen = []

    def grade(task, output: str, capped: bool) -> SimpleNamespace:
        seen.append(output)
        return SimpleNamespace(reward=float(output.endswith(f"{{{task.answer}}}")), machine_cause=None)

    assert rule(file(), build.params(), env_grader(Pool(), grade)) == "" and len(seen) == 20


def slow_row(task, output: str, capped: bool):
    """A grade job for the real pool: a row with "slow" in its text needs ``ROW_S`` seconds."""
    if "slow" in output:
        time.sleep(ROW_S)
    return pytest.importorskip("sn120_v2.envs.math_grader").grade(task, output, capped)


def test_real_grade_pool_slow_right_row_passes_in_a_new_process_whose_worker_starts_slowly(build, monkeypatch):
    """The real ``GradePool`` and the real math env. Without the known-good cases first, the row gets the time limit: start + row > limit."""
    math_grader = pytest.importorskip("sn120_v2.envs.math_grader", reason="install the envs extra: uv pip install -e '.[dev,envs]'")
    assert START_S < LIMIT_S and ROW_S < LIMIT_S < START_S + ROW_S
    monkeypatch.setenv(START_ENV, str(START_S))
    pool = load_pool()
    task = pool.task(pool.task_ids[0])
    rollouts = [{"output": f"{THINK}slow \\boxed{{{task.answer}}}", "reward": 1}, {"output": f"{THINK}\\boxed{{{task.answer}+1}}", "reward": 0}]
    raw = json.dumps({"version": "affine-batch-v3", "hotkey": HOTKEY, "groups": [{"prompt": task.batch_prompt(), "rollouts": rollouts}]}).encode()
    grade_pool = math_grader.GradePool(workers=1, timeout_s=LIMIT_S, job=slow_row)
    try:
        started = time.monotonic()
        batch = check_batch(raw, HOTKEY, build.params(), tokens=WordTokens(), grader=pool_grader(pool, grade_pool, chunk=1))
        assert batch.checked == ("format", "tokens", "grader")
        assert time.monotonic() - started >= START_S + ROW_S, "the worker did start slowly, and the row was slow"
    finally:
        grade_pool.close()
