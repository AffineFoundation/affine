"""The real grader pool stays bounded across rounds and recovers from a failure.

These tests use a tiny, importable grading function rather than the math env,
so they run without the heavy grader and GPU extras. Worker processes are real.
"""

from __future__ import annotations

import os
import resource
import threading
from types import SimpleNamespace

import pytest

from sn120_v2.envs.math_grader import GradePool
from sn120_v2.envs.math_pool import Task
from sn120_v2.types import PASS


def count_fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def scripted_job(task: Task, output: str, capped: bool):
    return SimpleNamespace(code=output, machine_cause=None)


def dying_job(task: Task, output: str, capped: bool):
    if output == "die":
        os._exit(1)
    return SimpleNamespace(code=PASS, machine_cause=None)


@pytest.fixture
def pool():
    graders = GradePool(workers=2, timeout_s=20.0, job=scripted_job)
    try:
        yield graders
    finally:
        graders.close()


def task() -> Task:
    return Task("test", "test", "question", "answer")


def test_many_grading_calls_leave_the_same_number_of_open_files(pool):
    jobs = [(task(), PASS, False)] * 200
    assert len(pool.grade_many(jobs)) == len(jobs)
    before = count_fds()
    for _ in range(10):
        assert len(pool.grade_many(jobs)) == len(jobs)
    assert count_fds() == before


def test_ten_rounds_of_two_thousand_grades_close_every_worker_pipe():
    before = count_fds()
    jobs = [(task(), PASS, False)] * 200
    for _ in range(10):
        graders = GradePool(workers=2, timeout_s=20.0, job=scripted_job)
        try:
            assert len(graders.grade_many(jobs)) == len(jobs)
        finally:
            graders.close()
        assert count_fds() == before


def test_worker_death_and_replacement_do_not_grow_open_files():
    import sn120_v2.envs.math_grader as math_grader

    graders = GradePool(workers=2, timeout_s=20.0, job=dying_job)
    try:
        graders.grade_many([(task(), PASS, False)] * 8)
        before = count_fds()
        for _ in range(10):
            grades = graders.grade_many([(task(), "die", False), (task(), PASS, False)])
            assert grades[0].machine_cause == math_grader.WORKER_DIED and grades[1].code == PASS
            assert count_fds() == before
    finally:
        graders.close()


def test_concurrent_gpu_batches_do_not_mix_worker_state(pool):
    jobs = [(task(), PASS, False)] * 8
    barrier = threading.Barrier(2)
    errors = []

    def grade():
        try:
            barrier.wait()
            assert len(pool.grade_many(jobs)) == len(jobs)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=grade) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads)
    assert not errors


def test_a_failed_call_does_not_poison_the_next_round(pool, monkeypatch):
    jobs = [(task(), PASS, False)] * 16
    assert len(pool.grade_many(jobs)) == len(jobs)
    import sn120_v2.envs.math_grader as math_grader

    def broken_wait(links, timeout):
        raise OSError("handle is closed")

    monkeypatch.setattr(math_grader, "wait", broken_wait)
    with pytest.raises(OSError, match="handle is closed"):
        pool.grade_many(jobs)
    monkeypatch.undo()
    assert [grade.code for grade in pool.grade_many(jobs)] == [PASS] * len(jobs)


def test_partial_worker_startup_closes_every_started_pipe(monkeypatch):
    import sn120_v2.envs.math_grader as math_grader

    warm = GradePool(workers=1, timeout_s=20.0, job=scripted_job)
    warm.close()
    before = count_fds()
    real_start = math_grader.multiprocessing.context.SpawnProcess.start
    starts = 0

    def fail_start(self):
        nonlocal starts
        starts += 1
        real_start(self)
        if starts % 3 == 0:
            raise OSError(24, "Too many open files")

    monkeypatch.setattr(math_grader.multiprocessing.context.SpawnProcess, "start", fail_start)
    for _ in range(10):
        with pytest.raises(OSError):
            math_grader.GradePool(workers=4, job=scripted_job)
        assert count_fds() == before


def test_worker_start_failure_closes_its_pipe_and_sentinel(monkeypatch):
    import sn120_v2.envs.math_grader as math_grader

    real_start = math_grader.multiprocessing.context.SpawnProcess.start

    def fail_start(self):
        real_start(self)
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(math_grader.multiprocessing.context.SpawnProcess, "start", fail_start)
    before = count_fds()
    with pytest.raises(OSError):
        math_grader._Worker(math_grader.multiprocessing.get_context("spawn"), scripted_job)
    assert count_fds() <= before + 1  # the process-wide resource tracker may start once


def test_evaluator_replaces_a_failed_lazy_pool_and_round_close_releases_it(monkeypatch):
    from sn120_v2.admission import LazyPool
    from sn120_v2.gpu.evaluator import VllmEvaluator

    pools = []

    def start():
        graders = GradePool(workers=2, timeout_s=20.0, job=scripted_job)
        pools.append(graders)
        return graders

    graders = LazyPool(start)
    seam = VllmEvaluator.__new__(VllmEvaluator)
    seam.grader = graders
    seam._grader_lock = threading.Lock()
    seam._grader_failure = False
    jobs = [(task(), PASS, False)] * 200
    assert len(seam._grade_batch(jobs)) == len(jobs)
    before = count_fds()
    assert len(seam._grade_batch(jobs)) == len(jobs)
    assert count_fds() == before and len(pools) == 1
    pools[0]._workers[0].link.close()
    import sn120_v2.envs.math_grader as math_grader

    def broken_wait(links, timeout):
        raise OSError("handle is closed")

    monkeypatch.setattr(math_grader, "wait", broken_wait)
    with pytest.raises(OSError, match="handle is closed"):
        seam._grade_batch(jobs)
    monkeypatch.setattr(math_grader, "wait", math_grader.multiprocessing.connection.wait)
    assert len(seam._grade_batch(jobs)) == len(jobs)
    assert len(pools) == 2 and pools[0]._workers == []
    seam._reset_grader()
    assert len(seam._grade_batch(jobs)) == len(jobs)
    assert len(pools) == 3
    seam._servers, seam._urls, seam._loaded = [], [], set()
    seam._head = None
    seam._solo_servers, seam._solo_urls, seam._solo_loaded = {}, {}, {}
    seam._solo_scheduler = None
    seam.close()
    assert count_fds() < before


def test_solo_round_failure_closes_resources_for_the_next_round(build, monkeypatch):
    from sn120_v2.errors import InfraError
    from sn120_v2.gpu.solo import SoloRunner

    evaluator = SimpleNamespace(close=lambda: calls.append("evaluator"), solo_worker=lambda *args: None)
    trainer = SimpleNamespace(close=lambda: calls.append("trainer"))
    calls = []
    runner = SoloRunner(trainer, evaluator, lambda: 1000, build.image)
    attempts = [InfraError("the grader failed: handle is closed"), "clean result"]

    def run_round(ticket):
        outcome = attempts.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(runner, "_solo_run_round", run_round)
    with pytest.raises(InfraError, match="handle is closed"):
        runner.run(build.ticket())
    assert runner.run(build.ticket()) == "clean result"
    assert calls == ["trainer", "evaluator"] * 2


def test_run_start_raises_the_soft_nofile_limit(monkeypatch, capsys):
    from sn120_v2 import cli

    calls = []
    monkeypatch.setattr(cli.resource, "getrlimit", lambda which: (1024, 524288))
    monkeypatch.setattr(cli.resource, "setrlimit", lambda which, limits: calls.append((which, limits)))
    cli.raise_nofile_limit()
    assert calls == [(resource.RLIMIT_NOFILE, (65536, 524288))]
    logged = capsys.readouterr().err
    assert "RLIMIT_NOFILE" in logged and "1024" in logged and "65536" in logged
