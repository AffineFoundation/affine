"""The official math grader: equal to v1 on every recorded case, and safe when a grade hangs or a worker dies.

``data/math_grader_v1_verdicts.jsonl`` holds inputs and the verdicts of the v1
grader (recorded by ``tools/record_v1_math_verdicts.py``, which imports v1 and
so lives outside ``v2/``). These tests need the ``envs`` extra.
"""

from __future__ import annotations

import inspect
import json
import os
import threading
import time
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from sn120_v2.canonical import is_sha256
from sn120_v2.envs.math_pool import Task, load_pool
from sn120_v2.errors import InfraError
from sn120_v2.types import CAP, ERROR, FAIL, PASS

math_grader = pytest.importorskip("sn120_v2.envs.math_grader", reason="install the envs extra: uv pip install -e '.[dev,envs]'")
TIME_LIMIT, WORKER_DIED = math_grader.TIME_LIMIT, math_grader.WORKER_DIED

V2 = Path(__file__).resolve().parents[1]
RECORDED = V2 / "tests/data/math_grader_v1_verdicts.jsonl"
THINK = "<think>\nLet me work it out step by step.\n</think>\n\n"


def scripted_job(task: Task, output: str, capped: bool):
    """A grade job for the supervisor test. The output text says what the worker does."""
    if output == "die":
        os._exit(1)
    if output == "hang":
        time.sleep(600)
    if output.startswith("log:"):  # hangs at the first call; a second call would be a pass
        log = Path(output[4:])
        earlier = log.read_text() if log.exists() else ""
        log.write_text(earlier + f"{os.getpid()}\n")
        if not earlier:
            time.sleep(600)
        return math_grader.Grade(PASS, 1.0, False, None, "live")
    return math_grader.grade(task, output, capped)


@pytest.fixture(scope="module")
def task() -> Task:
    return load_pool().task("math:0")


@pytest.fixture(scope="module")
def recorded():
    """The header, the recorded v1 cases, and the v2 grade of each case."""
    header, *cases = [json.loads(line) for line in RECORDED.read_text().splitlines()]
    pool = load_pool()
    jobs = [
        (pool.task(case["task_id"]) if "task_id" in case else Task("case", "test", case["problem"], case["gold"]), case["output"], case["stop"] != "eos")
        for case in cases
    ]
    graders = math_grader.GradePool(workers=min(8, os.cpu_count() or 1), timeout_s=120.0)
    try:
        return header["meta"], cases, graders.grade_many(jobs)
    finally:
        graders.close()


def as_v1(grade) -> tuple[str, str | None]:
    """A v2 grade as the v1 verdict and, for a pass, the v1 name of the rule that passed it."""
    if grade.code == PASS:
        return "positive", grade.rule
    if grade.code == ERROR:
        return "indeterminate", None
    return ("negative" if grade.code == FAIL and grade.rule == "wrong" else "neutral"), None


# ---- lesson: math-verify 0.9.0, then the lenient rules, the same for everyone ----


def test_every_recorded_v1_verdict_is_matched(recorded):
    meta, cases, grades = recorded
    assert len(cases) == len(grades) == meta["cases"] == 2458
    assert meta["packages"] == math_grader.PINNED, "v1 was recorded with the packages that v2 pins"
    differ = [case["id"] for case, grade in zip(cases, grades) if as_v1(grade) != (case["kind"], case["how"] if case["kind"] == "positive" else None)]
    assert not differ, f"{len(differ)} cases differ from v1, first ids {differ[:20]}"


def test_the_recorded_cases_pass_by_every_lenient_rule(recorded):
    _, cases, _ = recorded
    rules = {rule for case in cases if case["kind"] == "positive" for rule in case["how"].split("+")}
    # E2 (letter sets) is not here: in v1, math-verify or E1 accepts every letter list before E2 can.
    assert rules == {"live", "P1", "P2", "E1", "E3", "E4", "E5", "E6", "E7"}
    assert Counter(case["kind"] for case in cases) == recorded[0]["kinds"]


def test_the_grade_depends_on_the_task_and_the_text_only(task):
    assert list(inspect.signature(math_grader.grade).parameters) == ["task", "output", "capped"], "no model, no miner"
    right = THINK + "\\boxed{" + task.answer + "}"
    assert math_grader.grade(task, right, False) == math_grader.grade_answer(task.answer, task.problem, right, False)
    assert math_grader.grade(task, right, False) == math_grader.Grade(PASS, 1.0, False, "\\boxed{" + task.answer + "}", "live")
    wrong = math_grader.grade(task, THINK + "\\boxed{" + task.answer + "1}", False)
    assert (wrong.code, wrong.reward, wrong.rule) == (FAIL, 0.0, "wrong")


# ---- lesson: a capped reply is unresolved ----


def test_a_capped_output_is_never_graded_and_counts_as_a_cap_hit(task, recorded):
    right = THINK + "\\boxed{" + task.answer + "}"
    capped = math_grader.grade(task, right, True)
    assert (capped.code, capped.reward, capped.cap_hit, capped.rule) == (CAP, 0.0, True, "capped")
    _, cases, grades = recorded
    assert [grade.code for case, grade in zip(cases, grades) if case["stop"] != "eos"] == [CAP] * 4
    assert all(grade.cap_hit == (grade.code == CAP) and grade.reward == (grade.code == PASS) for grade in grades)


# ---- lesson: the grade supervisor; a time limit is not a pass ----


def test_a_hung_or_dead_grade_is_not_passed_and_is_never_retried(task, tmp_path):
    right, log = THINK + "\\boxed{" + task.answer + "}", tmp_path / "calls"
    jobs = [(task, right, False), (task, "die", False), (task, "hang", False), (task, f"log:{log}", False), (task, right, False)]
    graders = math_grader.GradePool(workers=2, timeout_s=6.0, job=scripted_job)
    try:
        started = time.monotonic()
        grades = graders.grade_many(jobs)
        assert time.monotonic() - started < 60, "a hung grade must not stall the round"
        assert [(grade.code, grade.machine_cause) for grade in grades] == [
            (PASS, None), (ERROR, WORKER_DIED), (ERROR, TIME_LIMIT), (ERROR, TIME_LIMIT), (PASS, None)
        ]  # fmt: skip
        assert all(grade.reward == 0.0 and not grade.cap_hit for grade in grades[1:4])
        (hung_worker,) = log.read_text().split()  # one line: the job ran one time, so its pass was never reached
        with pytest.raises(ProcessLookupError):
            os.kill(int(hung_worker), 0)  # the supervisor killed exactly that process
        assert [grade.code for grade in graders.grade_many([(task, right, False)] * 3)] == [PASS] * 3, "fresh workers took over"
    finally:
        graders.close()


def test_a_time_limit_of_math_verify_is_a_machine_error(monkeypatch, recorded):
    def expire(*args, **kwargs):
        raise math_grader.TimeoutException("Operation timed out!")

    monkeypatch.setattr(math_grader, "verify", expire)
    grade = math_grader.grade_answer("987650", "p", THINK + "\\boxed{987651}", False)
    assert (grade.code, grade.reward, grade.machine_cause) == (ERROR, 0.0, TIME_LIMIT)
    # The recorded set holds four cases where the real 5 second limit of math-verify expires, and eleven stable errors.
    causes = Counter((grade.code, grade.machine_cause) for grade in recorded[2])
    assert causes[(ERROR, TIME_LIMIT)] == 4 and causes[(ERROR, None)] == 11
    assert set(causes) == {(PASS, None), (FAIL, None), (CAP, None), (ERROR, None), (ERROR, TIME_LIMIT)}


def test_a_stable_error_has_no_machine_cause():
    no_reference = math_grader.grade_answer("", "p", THINK + "\\boxed{5}", False)
    raised = math_grader.grade_answer("y \\neq \\sqrt{2}", "p", THINK + "\\boxed{y \\leq \\sqrt{2}}", False)
    assert (no_reference.code, no_reference.rule, no_reference.machine_cause) == (ERROR, "reference-unparseable", None)
    assert (raised.code, raised.rule, raised.machine_cause) == (ERROR, "grader-error", None)


def test_the_grader_refuses_to_run_outside_the_main_thread(task):
    seen: list[object] = []

    def call(capped: bool) -> None:
        try:
            seen.append(math_grader.grade(task, THINK + "\\boxed{1}", capped).code)
        except InfraError as error:
            seen.append(error)

    for capped in (False, True):
        thread = threading.Thread(target=call, args=(capped,))
        thread.start()
        thread.join()
    assert isinstance(seen[0], InfraError) and seen[1] == CAP


# ---- lesson: the grader packages are pinned ----


def test_another_package_version_is_a_box_failure(monkeypatch):
    extra = tomllib.loads((V2 / "pyproject.toml").read_text())["project"]["optional-dependencies"]["envs"]
    assert sorted(extra) == sorted(f"{name}=={version}" for name, version in math_grader.PINNED.items())
    math_grader.check_runtime()
    monkeypatch.setattr(math_grader.metadata, "version", lambda name: "0.9.1")
    with pytest.raises(InfraError, match="math-verify==0.9.0"):
        math_grader.check_runtime()
    with pytest.raises(InfraError):
        math_grader.GradePool(workers=1)


def test_the_env_pin_changes_with_the_pool():
    pin = math_grader.env_pin()
    assert is_sha256(pin) and pin == math_grader.env_pin() and pin != math_grader.env_pin("0" * 64)
