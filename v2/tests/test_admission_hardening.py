"""Admission at launch: one file never stops the others (grader, tokenizer or checker fault), a hard time limit, no file that trains nothing, and a cap on the training cost."""

from __future__ import annotations

import ast
import inspect
import json
import random
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import CHECK_FAULT, FAULT, FAULT_HOLD_S, FAULT_TRIES, NO_SIGNAL, RULES, TIME_LIMIT, TOKENIZER_FAULT, TRAIN_COST, TRANSIENT, GraderFault, GraderTimeLimit, Invalid, StoreAdmission, check_batch, env_grader, final_answer, pool_grader, trained_tokens
from sn120_v2.canonical import sha256_hex
from sn120_v2.cli import main
from sn120_v2.errors import InfraError, VerifyError
from sn120_v2.gpu.batch import fit_tokens, training_rows
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.sim import STORE_DIR, SimAdmission
from sn120_v2.state import load
from sn120_v2.store import LocalStore

HOTKEY = "5MinerA"
THINK = "step " * 40


class WordTokens:
    def prompt_tokens(self, prompt: str) -> int:
        return 5 + len(prompt.split())

    def output_tokens(self, text: str) -> int:
        return len(text.split())


class Clock:
    """A clock that the test moves. ``wait()`` ends the hold of a fault verdict."""

    def __init__(self, now: float = 7.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def wait(self, seconds: float = FAULT_HOLD_S) -> None:
        self.now += seconds


class Store(dict):
    put = dict.__setitem__
    get = dict.__getitem__

    def delete(self, key: str) -> None:
        self.pop(key, None)

    def size(self, key: str) -> int:
        return len(self[key])

    def sha256(self, key: str) -> str:
        return sha256_hex(self[key])


def row(answer, reward, think: str = THINK) -> dict:
    return {"output": f"{think}\\boxed{{{answer}}}", "reward": reward}


def file(groups=None, hotkey: str = HOTKEY, train=None) -> bytes:
    """By default 10 prompts "double N", each with a right and a wrong rollout."""
    groups = [{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0)]} for n in range(10)] if groups is None else groups
    return json.dumps({"version": "affine-batch-v3", "hotkey": hotkey, "groups": groups} | ({} if train is None else {"train": train})).encode()


def rule(raw: bytes, params, tokens=WordTokens(), **inputs) -> str:
    try:
        check_batch(raw, HOTKEY, params, tokens=tokens, **inputs)
    except Invalid as exc:
        return exc.rule
    return ""


# --- a toy environment with the shape of the math env: a pool of tasks and grade(task, output, capped) ---


class Pool:
    """The task "double N" has the answer 2N."""

    task_ids = tuple(f"t{n}" for n in range(50))

    def task(self, task_id: str) -> SimpleNamespace:
        n = int(task_id[1:])  # "double x" is a prompt that breaks this pool: a fault of the checker, not of the grade function
        return SimpleNamespace(n=n, answer=str(2 * n), batch_prompt=lambda: f"double {n}")

    def task_for_prompt(self, prompt: str) -> SimpleNamespace | None:
        return self.task(f"t{prompt.split()[1]}") if prompt.startswith("double ") else None


def toy_grade(task, output: str, capped: bool) -> SimpleNamespace:
    """The output text says what happens: "raise" makes the grade function raise, "die" and "slow" give a machine cause."""
    if "raise" in output:
        raise ZeroDivisionError("the text of this row breaks the grader")
    cause = "worker-died" if "die" in output else "time-limit" if "slow" in output else None
    return SimpleNamespace(reward=float(cause is None and not capped and final_answer(output) == task.answer), machine_cause=cause)


class ToyGradePool:
    """``GradePool.grade_many`` without processes: a job that raises is a dead worker, as in the real pool."""

    def __init__(self, job=toy_grade) -> None:
        self.job, self.calls = job, []

    def grade_many(self, jobs) -> list[SimpleNamespace]:
        self.calls.append(len(jobs))
        return [self._one(job) for job in jobs]

    def _one(self, job) -> SimpleNamespace:
        try:
            return self.job(*job)
        except Exception:
            return SimpleNamespace(reward=0.0, machine_cause="worker-died")


def graders() -> dict:
    return {"env_grader": env_grader(Pool(), toy_grade), "pool_grader": pool_grader(Pool(), ToyGradePool())}


def test_env_grader_joins_the_pool_and_the_grade_function(build):
    for official in graders().values():
        assert (official("double 2", "\\boxed{4}", False), official("double 2", "\\boxed{4}", True), official("other", "x", False)) == (1.0, 0.0, None)
        assert rule(file(), build.params(), grader=official) == ""
        assert rule(file([{"prompt": "double 4", "rollouts": [row(8, 0), row(8, 1)]}]), build.params(), grader=official) == "grader"
        with pytest.raises(GraderFault):
            official("double 2", "die", False)


# --- one file never stops the others ---


@pytest.mark.parametrize("name", ["env_grader", "pool_grader"])
@pytest.mark.parametrize("poison, code", [("raise", FAULT), ("die", FAULT), ("slow", TIME_LIMIT)])
def test_a_file_that_breaks_the_grader_is_skipped_and_the_other_files_are_checked(name, poison, code, build):
    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=graders()[name], clock=Clock())
    bad = file([{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0, THINK + poison + " " if n == 3 else THINK)]} for n in range(10)], "bad")
    for hotkey, raw in (("first", file(hotkey="first")), ("bad", bad), ("last", file(hotkey="last"))):
        adm.store.put(layout.miner_object(hotkey), raw)
    got = adm.verdicts(["first", "bad", "last"])
    assert [(hotkey, verdict.valid, verdict.rule) for hotkey, verdict in got.items()] == [("first", True, ""), ("bad", False, code), ("last", True, "")]
    assert "groups[3].rollouts[1]" in got["bad"].reason and poison not in got["bad"].reason
    assert adm.freeze(build.reg(2, hotkey="bad"), "r000001") is None  # a skip: not charged, the place is kept
    assert adm.freeze(build.reg(3, hotkey="last"), "r000001") is not None
    assert got["bad"].reason.endswith("(check 1 of 3)")
    assert adm.verdicts(["bad"])["bad"] is got["bad"], "the verdict holds for 10 minutes"
    adm.clock.wait()
    assert adm.verdicts(["bad"])["bad"].reason.endswith("(check 2 of 3)"), "then the file is checked again"
    assert adm.verdicts(["first"])["first"] is got["first"]
    adm.store.put(layout.miner_object("bad"), file(hotkey="bad"))  # the miner uploads a file the grader can read: checked with no wait
    assert adm.verdicts(["bad"])["bad"].valid and adm.freeze(build.reg(2, hotkey="bad"), "r000001") is not None


@pytest.mark.parametrize("name", ["env_grader", "pool_grader"])
def test_a_grader_that_is_broken_for_every_file_stays_loud(name, build):
    def broken(task, output, capped):
        raise OSError("the math package is gone")

    official = env_grader(Pool(), broken) if name == "env_grader" else pool_grader(Pool(), ToyGradePool(broken))
    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=official)
    adm.store.put(layout.miner_object(HOTKEY), file())
    for call in (lambda: rule(file(), build.params(), grader=official), lambda: adm.verdicts([HOTKEY]), lambda: adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001")):
        with pytest.raises(InfraError, match="known-good"):
            call()
    assert adm._seen == {} and list(adm.store) == [layout.miner_object(HOTKEY)]


def test_the_known_good_case_is_a_pool_task_with_its_own_answer(build):
    seen = []

    def watched(task, output, capped):
        seen.append((task.n, output))
        return toy_grade(task, output, capped)

    official = env_grader(Pool(), watched)
    official.probe()
    assert seen == [(n, f"\\boxed{{{2 * n}}}") for n in range(admission.PROBE_TASKS)]
    official = env_grader(Pool(), lambda task, output, capped: SimpleNamespace(reward=0.0, machine_cause=None))  # it answers, but wrongly
    with pytest.raises(InfraError, match="known-good"):
        official.probe()
    with pytest.raises(InfraError, match="2 grades for 3 rows"):  # a pool that loses a row is our failure
        list(pool_grader(Pool(), SimpleNamespace(grade_many=lambda jobs: [toy_grade(*job) for job in jobs][:2])).many([("double 1", "x", False)] * 3))


def test_the_grade_pool_gets_the_rows_in_chunks_and_the_check_stops_at_the_first_bad_row(build):
    rollouts = [row(8, 1, f"{THINK}{k} ") for k in range(20)] + [row(-1, 0, f"{THINK}{k} ") for k in range(20)]
    big = [{"prompt": "my own task", "rollouts": [row(1, 1), row(2, 0)]}, {"prompt": "double 4", "rollouts": rollouts}]
    pool = ToyGradePool()
    assert rule(file(big), build.params(), grader=pool_grader(Pool(), pool, chunk=16)) == ""
    assert pool.calls == [14, 16, 10], "42 rows in chunks of 16; the 2 rows outside the pool are not sent"
    rollouts[10] = row(8, 1, THINK + "slow ")
    pool = ToyGradePool()
    assert rule(file(big), build.params(), grader=pool_grader(Pool(), pool, chunk=16)) == TIME_LIMIT
    assert pool.calls == [14, admission.PROBE_TASKS], "the bad row is in the first chunk: the other rows are never graded; then the known-good case"


# --- a text that breaks the tokenizer skips its own file only ---


class Panic(BaseException):
    """A Rust tokenizer reports a panic with an exception that is not an ``Exception``."""


class FragileTokens(WordTokens):
    """Fails on a text that ``breaks`` it. With ``dead`` it fails on every text."""

    def __init__(self, error: type[BaseException], breaks=lambda text: "poison" in text, dead: bool = False) -> None:
        self.error, self.breaks, self.dead = error, breaks, dead

    def prompt_tokens(self, prompt: str) -> int:
        self.output_tokens(prompt)
        return super().prompt_tokens(prompt)

    def output_tokens(self, text: str) -> int:
        if self.dead or self.breaks(text):
            raise self.error("secret text of the batch")
        return super().output_tokens(text)


BREAKS_THE_TOKENIZER = {
    "prompt": ([{"prompt": "double 3 poison", "rollouts": [row(6, 1), row(-1, 0)]}], lambda text: "poison" in text),
    "output": ([{"prompt": "double 3", "rollouts": [row(6, 1), row(-1, 0, THINK + "poison ")]}], lambda text: "poison" in text),
    # Only the count of the text before the first boxed answer fails: the whole output is counted first, and passes.
    "reasoning": ([{"prompt": "double 3", "rollouts": [row(6, 1), row(-1, 0, THINK + "poison ")]}], lambda text: "poison" in text and "boxed" not in text),
}


@pytest.mark.parametrize("error", [ValueError, MemoryError, Panic])
@pytest.mark.parametrize("where", sorted(BREAKS_THE_TOKENIZER))
def test_file_that_breaks_the_tokenizer_is_skipped_and_the_other_files_are_checked(build, error, where):
    groups, breaks = BREAKS_THE_TOKENIZER[where]
    store = Store({layout.miner_object("5A"): file(hotkey="5A"), layout.miner_object("5B"): file(groups, hotkey="5B"), layout.miner_object("5C"): file(hotkey="5C")})
    adm = StoreAdmission(store, build.params(), tokens=FragileTokens(error, breaks), grader=None, clock=lambda: 7)
    found = adm.verdicts(["5A", "5B", "5C"])
    assert [(found[key].valid, found[key].rule) for key in ("5A", "5B", "5C")] == [(True, ""), (False, TOKENIZER_FAULT), (True, "")]
    assert "secret" not in found["5B"].reason and "poison" not in found["5B"].reason, "a reason never holds batch text"
    assert "5B" not in adm._seen and {"5A", "5C"} <= set(adm._seen), "the fault verdict is not kept at the first check"
    assert adm.freeze(build.reg(2, hotkey="5B"), "r000001") is None and layout.frozen_batch("r000001", "5B") not in store
    assert adm.freeze(build.reg(1, hotkey="5A"), "r000001") is not None


def test_tokenizer_that_is_broken_for_every_file_stays_loud(build):
    store = Store({layout.miner_object("5A"): file(hotkey="5A")})
    adm = StoreAdmission(store, build.params(), tokens=FragileTokens(ValueError, dead=True), grader=None)
    with pytest.raises(InfraError, match="known-good"):
        adm.verdicts(["5A"])
    with pytest.raises(InfraError, match="known-good"):
        adm.freeze(build.reg(1, hotkey="5A"), "r000001")
    with pytest.raises(KeyboardInterrupt):
        check_batch(file(), HOTKEY, build.params(), tokens=FragileTokens(KeyboardInterrupt, dead=True))


# --- a file that breaks the checker itself, with any exception, skips its own file only ---

BUG = "marker"  # the prompt of the one file that hits the bug


def bug_in(monkeypatch, step: str, error: type[BaseException], every_file: bool = False) -> None:
    """Put a bug in one step of the checker: it raises on a file that holds the ``BUG`` prompt, or on every file."""
    real = getattr(admission, step)

    def buggy(first, *rest):
        prompts = [group["prompt"] for group in first["groups"]] if isinstance(first, dict) else [prompt for prompt, _ in first]
        if every_file or BUG in prompts:
            raise error("secret text of the batch")
        return real(first, *rest)

    monkeypatch.setattr(admission, step, buggy)


def three_files(bad: list, **inputs) -> StoreAdmission:
    store = Store({layout.miner_object("5A"): file(hotkey="5A"), layout.miner_object("5B"): file(bad, hotkey="5B"), layout.miner_object("5C"): file(hotkey="5C")})
    return StoreAdmission(store, inputs.pop("params"), tokens=WordTokens(), clock=Clock(), **inputs)


@pytest.mark.parametrize("error", [KeyError, RecursionError, MemoryError, Panic])
@pytest.mark.parametrize("step", ["_groups", "_cap", "_kept", "_trains", "_graded"])
def test_file_that_breaks_the_checker_is_skipped_and_the_other_files_are_checked(build, monkeypatch, step, error):
    bug_in(monkeypatch, step, error)
    adm = three_files([{"prompt": BUG, "rollouts": [row(6, 1), row(-1, 0)]}], params=build.params(), grader=graders()["pool_grader"])
    found = adm.verdicts(["5A", "5B", "5C"])
    assert [(found[key].valid, found[key].rule) for key in ("5A", "5B", "5C")] == [(True, ""), (False, CHECK_FAULT), (True, "")]
    counted = "" if error is MemoryError else r" \(check 1 of 3\)"
    assert re.fullmatch(rf"the checker fails on this file: {error.__name__} in buggy, test_admission_hardening\.py line \d+{counted}", found["5B"].reason), "the class and the place in the code, never the message"
    assert "5B" not in adm._seen and {"5A", "5C"} <= set(adm._seen), "the fault verdict is not kept at the first check"
    assert adm.verdicts(["5B"])["5B"] is found["5B"] and adm.verdicts(["5A"])["5A"] is found["5A"]
    adm.clock.wait()
    assert adm.verdicts(["5B"])["5B"] is not found["5B"], "after the hold the file is checked again"
    assert adm.freeze(build.reg(2, hotkey="5B"), "r000001") is None and layout.frozen_batch("r000001", "5B") not in adm.store  # a skip: not charged, the place is kept
    assert adm.freeze(build.reg(3, hotkey="5C"), "r000001") is not None
    adm.store.put(layout.miner_object("5B"), file(hotkey="5B"))  # the miner uploads a file that does not hit the bug
    assert adm.verdicts(["5B"])["5B"].valid and adm.freeze(build.reg(2, hotkey="5B"), "r000001") is not None


@pytest.mark.parametrize("name", ["env_grader", "pool_grader"])
def test_prompt_that_breaks_the_pool_lookup_skips_its_own_file_only(build, name):
    adm = three_files([{"prompt": "double x", "rollouts": [row(6, 1), row(-1, 0)]}], params=build.params(), grader=graders()[name])
    found = adm.verdicts(["5A", "5B", "5C"])
    assert [(found[key].valid, found[key].rule) for key in ("5A", "5B", "5C")] == [(True, ""), (False, CHECK_FAULT), (True, "")]
    assert found["5B"].reason.startswith("the checker fails on this file: ValueError in task, test_admission_hardening.py line ")
    assert rule(file([{"prompt": "double x", "rollouts": [row(6, 1), row(-1, 0)]}]), build.params()) == "", "with no grader the same file hits no fault"


@pytest.mark.parametrize("step", ["_groups", "_cap", "_kept", "_trains", "_graded"])
def test_checker_that_is_broken_for_every_file_stays_loud(build, monkeypatch, step):
    bug_in(monkeypatch, step, KeyError, every_file=True)
    adm = StoreAdmission(Store({layout.miner_object(HOTKEY): file()}), build.params(), tokens=WordTokens(), grader=graders()["pool_grader"])
    for call in (lambda: rule(file(), build.params(), grader=adm.grader), lambda: adm.verdicts([HOTKEY]), lambda: adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001")):
        with pytest.raises(InfraError, match=r"KeyError in buggy.* does not pass its known-good file \(KeyError in buggy") as caught:
            call()
        assert "secret" not in str(caught.value)
    assert adm._seen == {} and list(adm.store) == [layout.miner_object(HOTKEY)]


def test_stop_request_and_our_own_failure_pass_through_the_net(build, monkeypatch):
    for error in (KeyboardInterrupt, SystemExit, InfraError):
        bug_in(monkeypatch, "_cap", error)
        with pytest.raises(error):
            check_batch(file([{"prompt": BUG, "rollouts": [row(6, 1), row(-1, 0)]}]), HOTKEY, build.params(), tokens=WordTokens())
        monkeypatch.undo()


def test_known_good_file_passes_every_rule_with_every_set_of_inputs(build):
    plain = lambda prompt, output, capped: None  # noqa: E731 - a grader with no known-good cases: it knows no prompt
    for tokens in (None, WordTokens()):
        for grader, n_files in ((None, 1), (plain, 1), (graders()["env_grader"], admission.PROBE_TASKS), (graders()["pool_grader"], admission.PROBE_TASKS)):
            files = list(admission._known_good(build.params(), grader))
            assert len(files) == n_files
            for raw in files:
                batch = check_batch(raw, admission.PROBE_HOTKEY, build.params(), tokens=tokens, grader=grader)
                assert (batch.n_prompts, batch.n_rollouts, batch.format) == (1, 2, "affine-batch-v1")
                assert batch.checked == ("format",) + (("tokens",) if tokens else ()) + (("grader",) if grader else ())
    pool_task = json.loads(next(admission._known_good(build.params(), graders()["env_grader"])))["groups"][0]
    assert pool_task["prompt"] == "double 0" and final_answer(pool_task["rollouts"][0]["output"]) == "0" and final_answer(pool_task["rollouts"][1]["output"]) is None


def test_known_good_file_that_the_rules_reject_stays_loud(build, monkeypatch):
    bug_in(monkeypatch, "_cap", KeyError)
    bad = file([{"prompt": BUG, "rollouts": [row(6, 1), row(-1, 0)]}])
    assert rule(bad, build.params()) == CHECK_FAULT
    with pytest.raises(InfraError, match=r"does not pass its known-good file \(rule train_cost\)"):  # no file passes these params
        rule(bad, build.params(max_train_token_steps=1))
    wrong = env_grader(Pool(), lambda task, output, capped: SimpleNamespace(reward=0.0, machine_cause=None))  # it answers, but wrongly
    with pytest.raises(InfraError, match=r"does not pass its known-good file \(rule grader\)"):
        rule(bad, build.params(), grader=wrong)
    one_good = env_grader(Pool(), lambda task, output, capped: SimpleNamespace(reward=float(task.n == 3 and "boxed" in output), machine_cause=None))
    assert rule(bad, build.params(), grader=one_good) == CHECK_FAULT, "one known-good case that passes is enough, as in the probe of the grader"


# --- the limit: the same fault of the same file, proven three times in a row, is a verdict that stays ---

STAYS = f"(checked {FAULT_TRIES} times: this verdict stays until a new file is uploaded)"


class CountingStore(Store):
    """Counts the downloads: a kept verdict costs no download and no check."""

    reads = 0

    def get(self, key: str) -> bytes:
        self.reads += 1
        return self[key]


def poisoned(word: str, at: int = 3) -> bytes:
    """The default file with ``word`` in the wrong rollout of group ``at``."""
    return file([{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0, THINK + word + " " if n == at else THINK)]} for n in range(10)])


def one_file(raw: bytes, params, **inputs) -> StoreAdmission:
    return StoreAdmission(CountingStore({layout.miner_object(HOTKEY): raw}), params, **{"tokens": WordTokens(), "grader": None, "clock": Clock()} | inputs)


def checks(adm: StoreAdmission, calls: int) -> list:
    """The verdict of each of ``calls`` checks. The clock moves past the hold after each one."""
    found = []
    for _ in range(calls):
        found.append(adm.verdicts([HOTKEY])[HOTKEY])
        adm.clock.wait()
    return found


def reasons(adm: StoreAdmission, calls: int) -> list[str]:
    return [verdict.reason for verdict in checks(adm, calls)]


def with_fault(code: str, build, monkeypatch) -> StoreAdmission:
    """One stored file that gets ``code`` at every check, while the known-good case passes."""
    if code == TOKENIZER_FAULT:
        return one_file(poisoned("poison"), build.params(), tokens=FragileTokens(Panic), grader=None)
    if code == CHECK_FAULT:
        bug_in(monkeypatch, "_cap", KeyError)
        return one_file(file([{"prompt": BUG, "rollouts": [row(6, 1), row(-1, 0)]}]), build.params(), grader=graders()["pool_grader"])
    return one_file(poisoned("slow" if code == TIME_LIMIT else "die"), build.params(), grader=graders()["pool_grader"])


@pytest.mark.parametrize("code", TRANSIENT)
def test_same_fault_three_times_in_a_row_is_a_verdict_that_stays(code, build, monkeypatch):
    adm = with_fault(code, build, monkeypatch)
    first, second, third = checks(adm, FAULT_TRIES)
    assert {first.rule, second.rule, third.rule} == {code} and not third.valid
    assert first.reason.endswith("(check 1 of 3)") and second.reason.endswith("(check 2 of 3)") and third.reason.endswith(STAYS)
    assert first.reason.split(" (")[0] == third.reason.split(" (")[0], "the reason names the same place each time"
    assert adm.store.reads == FAULT_TRIES
    assert all(adm.verdicts([HOTKEY])[HOTKEY] is third for _ in range(5)) and adm.store.reads == FAULT_TRIES, "kept: no download and no check"
    assert adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001") is None and list(adm.store) == [layout.miner_object(HOTKEY)]  # an invalid file like any other
    again = StoreAdmission(adm.store, adm.params, tokens=adm.tokens, grader=adm.grader, clock=lambda: 7)  # the controller starts again
    assert again.verdicts([HOTKEY])[HOTKEY].reason.endswith("(check 1 of 3)")
    adm.params = build.params(max_prompts=adm.params.max_prompts + 1)  # new params: the file is checked again, from zero
    assert [reason.rsplit(" (", 1)[1] for reason in reasons(adm, 2)] == ["check 1 of 3)", "check 2 of 3)"]
    adm.store.put(layout.miner_object(HOTKEY), file() + b" ")  # a new file
    assert adm.verdicts([HOTKEY])[HOTKEY].valid and HOTKEY not in adm._tries and adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001") is not None
    adm.store.pop(layout.miner_object(HOTKEY))  # the file is gone: nothing is remembered
    assert adm.verdicts([HOTKEY]) == {} and (adm._seen, adm._tries, adm._held) == ({}, {}, {})


@pytest.mark.parametrize("code", TRANSIENT)
def test_new_bytes_with_the_same_fault_start_the_count_again(code, build, monkeypatch):
    adm = with_fault(code, build, monkeypatch)
    assert reasons(adm, 2)[1].endswith("(check 2 of 3)")
    adm.store.put(layout.miner_object(HOTKEY), adm.store[layout.miner_object(HOTKEY)] + b" ")
    assert [reason.rsplit(" (", 1)[1] for reason in reasons(adm, 4)] == ["check 1 of 3)", "check 2 of 3)", STAYS[1:], STAYS[1:]]


@pytest.mark.parametrize("error", [MemoryError, OSError, BrokenPipeError])
def test_failure_with_a_cause_on_our_machine_is_never_counted(error, build, monkeypatch):
    broken_tokens = one_file(poisoned("poison"), build.params(), tokens=FragileTokens(error), grader=None)
    bug_in(monkeypatch, "_cap", error)
    broken_check = one_file(file([{"prompt": BUG, "rollouts": [row(6, 1), row(-1, 0)]}]), build.params())

    class LostPool(ToyGradePool):
        def grade_many(self, jobs):
            if any("pipe" in output for _, output, _ in jobs):
                raise error("the pipe to the workers broke")
            return super().grade_many(jobs)

    lost_pool = one_file(poisoned("pipe"), build.params(), grader=pool_grader(Pool(), LostPool()))
    for adm, code in ((broken_tokens, TOKENIZER_FAULT), (broken_check, CHECK_FAULT), (lost_pool, FAULT)):
        found = checks(adm, 3 * FAULT_TRIES)
        assert {verdict.rule for verdict in found} == {code} and not any("(check" in verdict.reason for verdict in found)
        assert adm._seen == {} and adm.store.reads == 3 * FAULT_TRIES, "checked again after each hold, never kept"


def test_time_limit_of_a_grader_with_no_known_good_case_is_never_counted(build):
    def timed(prompt: str, output: str, capped: bool) -> float | None:
        raise GraderTimeLimit

    adm = one_file(file(), build.params(), grader=timed)
    assert set(reasons(adm, 3 * FAULT_TRIES)) == {"groups[0].rollouts[0]: the official grader did not finish this row inside its time limit"}
    assert adm._seen == {} and adm.store.reads == 3 * FAULT_TRIES


@pytest.mark.parametrize("cause", ["time-limit", "worker-died"])
def test_grade_pool_that_is_slow_or_dead_for_every_row_is_loud_and_counts_nothing(cause, build):
    state = {"down": True}

    def job(task, output, capped):
        return SimpleNamespace(reward=0.0, machine_cause=cause) if state["down"] else toy_grade(task, output, capped)

    adm = one_file(file(), build.params(), grader=pool_grader(Pool(), ToyGradePool(job)))
    for _ in range(3 * FAULT_TRIES):
        with pytest.raises(InfraError, match="known-good"):
            adm.verdicts([HOTKEY])
    assert adm._tries == {} and adm._seen == {} and adm._held == {}, "our failure leaves no count and no hold"
    state["down"] = False  # the pool is back: the honest file passes, with no wait
    assert adm.verdicts([HOTKEY])[HOTKEY].valid


def test_row_that_gets_its_grade_at_the_second_try_is_never_counted(build):
    """A pool that loses one row of each file at random: the known-good cases pass, and so does the same row alone."""
    calls = iter(range(1000))

    class FlakyPool(ToyGradePool):
        def grade_many(self, jobs):
            grades = super().grade_many(jobs)
            if len(jobs) == 20:  # the 20 rows of the file; not the known-good cases, not the second try of one row
                grades[next(calls) % 20] = SimpleNamespace(reward=0.0, machine_cause="worker-died")
            return grades

    pool = FlakyPool()
    adm = one_file(file(), build.params(), grader=pool_grader(Pool(), pool))
    found = reasons(adm, 3 * FAULT_TRIES)
    assert found[:3] == [f"groups[{i}].rollouts[{j}]: the official grader fails on this row" for i, j in ((0, 0), (0, 1), (1, 0))]
    assert not any("(check" in reason for reason in found) and adm._seen == {}
    assert pool.calls[:3] == [20, admission.PROBE_TASKS, 1], "the file, the known-good cases, then the same row alone"


def test_known_good_case_that_gets_no_grade_means_no_failed_row_is_counted(build):
    """A pool that loses rows often: the row of the file dies twice, but one known-good case dies too."""

    def job(task, output, capped):
        lost = "die" in output or (task.n == 2 and "step" not in output)  # the known-good case of task 2 is the bare answer
        return SimpleNamespace(reward=0.0, machine_cause="worker-died") if lost else toy_grade(task, output, capped)

    official = pool_grader(Pool(), ToyGradePool(job))
    assert official.probe() is False and graders()["pool_grader"].probe() is True
    adm = one_file(poisoned("die"), build.params(), grader=official)
    assert set(reasons(adm, 3 * FAULT_TRIES)) == {"groups[3].rollouts[1]: the official grader fails on this row"} and adm._seen == {}


def test_row_over_its_time_limit_is_not_graded_a_second_time(build):
    pool = ToyGradePool()
    adm = one_file(poisoned("slow"), build.params(), grader=pool_grader(Pool(), pool))
    assert reasons(adm, FAULT_TRIES)[-1].endswith(STAYS)
    assert pool.calls == [20, admission.PROBE_TASKS] * FAULT_TRIES, "a file that hangs costs its time limit once for each check"


def test_fault_that_moves_from_row_to_row_never_reaches_the_limit(build):
    target = {"word": "seven"}

    def job(task, output, capped):
        return SimpleNamespace(reward=0.0, machine_cause="worker-died") if target["word"] in output else toy_grade(task, output, capped)

    groups = [{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0, THINK + {3: "three ", 7: "seven "}.get(n, ""))]} for n in range(10)]
    adm = one_file(file(groups), build.params(), grader=pool_grader(Pool(), ToyGradePool(job)))
    found = []
    for word in ("seven", "three", "seven", "three", "three"):
        target["word"] = word
        found += reasons(adm, 1)
    assert found == [f"groups[{n}].rollouts[1]: the official grader fails on this row (check {count} of 3)" for n, count in ((7, 1), (3, 1), (7, 1), (3, 1), (3, 2))]
    assert adm._seen == {}


def test_check_that_is_not_counted_or_that_passes_starts_the_count_again(build):
    mode = {"now": "die"}

    class Machine(ToyGradePool):
        """The row with "odd" in it: the worker dies, the pipe breaks, or the grade is given."""

        def grade_many(self, jobs):
            if mode["now"] == "pipe" and any("odd" in output for _, output, _ in jobs):
                raise OSError("the pipe to the workers broke")
            return [SimpleNamespace(reward=0.0, machine_cause="worker-died") if mode["now"] == "die" and "odd" in job[1] else toy_grade(*job) for job in jobs]

    adm = one_file(poisoned("odd"), build.params(), grader=pool_grader(Pool(), Machine()))
    assert [reason.rsplit(" (", 1)[1] for reason in reasons(adm, 2)] == ["check 1 of 3)", "check 2 of 3)"]
    mode["now"] = "pipe"
    assert reasons(adm, 1) == ["groups[0].rollouts[0]: the official grader fails on this row"], "the whole call failed, with a sign of trouble on our machine: not counted"
    mode["now"] = "die"
    assert [reason.rsplit(" (", 1)[1] for reason in reasons(adm, 2)] == ["check 1 of 3)", "check 2 of 3)"], "the count started again"
    mode["now"] = "fine"
    passed = adm.verdicts([HOTKEY])[HOTKEY]
    assert passed.valid and adm.verdicts([HOTKEY])[HOTKEY] is passed and HOTKEY not in adm._tries, "the same bytes pass: a valid verdict, kept"


def test_run_names_the_cause_at_each_try_when_our_failure_stops_the_opening_of_a_round(build, tmp_path, monkeypatch, capsys):
    """What the operator sees: one "waiting:" line with the cause at each try. ``run`` goes on and opens no round."""
    directory = tmp_path / "state"
    shutil.copytree(Path(__file__).parent / "data" / "state_515d7e9f" / "idle", directory)
    SimAdmission(LocalStore(directory / STORE_DIR)).upload("hk4", 0.3)
    dead = pool_grader(Pool(), ToyGradePool(lambda task, output, capped: SimpleNamespace(reward=0.0, machine_cause="time-limit")))  # too slow for every row
    monkeypatch.setattr(SimAdmission, "verdicts", lambda self, hotkeys: check_batch(file(), HOTKEY, build.params(), tokens=WordTokens(), grader=dead))
    rounds = len(load(directory / JOURNAL_FILE).rounds)
    assert main(["run", "--state-dir", str(directory), "--stack", "sn120_v2.sim:stack", "--rounds", "1", "--max-idle", "3", "--idle-sleep", "0"]) == 0
    lines = capsys.readouterr().err.splitlines()
    assert [line for line in lines if line.startswith("waiting")] == ["waiting: the official grader gives no known-good case its reward"] * 3
    assert sum(line.startswith("idle: ") for line in lines) == 3 and len(load(directory / JOURNAL_FILE).rounds) == rounds, "no round opens"


# --- the hold: 10 minutes between two checks of a file with such a code, counted or not ---


def polled(adm: StoreAdmission, seconds: int, every: int = 30) -> list:
    """The verdicts that ``run`` gets when it tries to open a round each ``every`` seconds."""
    found = []
    for _ in range(0, seconds, every):
        found.append(adm.verdicts([HOTKEY])[HOTKEY])
        adm.clock.wait(every)
    return found


@pytest.mark.parametrize("code", TRANSIENT)
def test_file_with_a_fault_code_is_checked_again_only_after_the_hold(code, build, monkeypatch):
    adm = with_fault(code, build, monkeypatch)
    first = adm.verdicts([HOTKEY])[HOTKEY]
    adm.clock.wait(FAULT_HOLD_S - 1)
    assert adm.verdicts([HOTKEY])[HOTKEY] is first and adm.store.reads == 1, "inside the hold: the same verdict, no download, no check"
    assert not first.valid and first.rule == code, "the file keeps its code, so no round takes it"
    adm.clock.wait(1)
    second = adm.verdicts([HOTKEY])[HOTKEY]
    assert second.reason.endswith("(check 2 of 3)") and adm.store.reads == 2 and second.checked_at == first.checked_at + FAULT_HOLD_S


@pytest.mark.parametrize("code", TRANSIENT)
def test_verdict_that_stays_needs_a_fault_that_lasts_two_holds(code, build, monkeypatch):
    adm = with_fault(code, build, monkeypatch)
    found = polled(adm, 3 * FAULT_HOLD_S)  # 30 minutes of tries, one each 30 s
    notes = [verdict.reason.rsplit(" (", 1)[1] for verdict in found]
    per_hold = FAULT_HOLD_S // 30
    assert notes == ["check 1 of 3)"] * per_hold + ["check 2 of 3)"] * per_hold + [STAYS[1:]] * per_hold
    assert adm.store.reads == FAULT_TRIES and {verdict.rule for verdict in found} == {code}


def test_stall_of_our_machine_that_is_shorter_than_the_hold_never_makes_a_verdict_stay(build):
    """The reason for the hold: the machine is too slow for one heavy row of an honest file, for 3 tries in a row."""
    clock = Clock()

    def job(task, output, capped):
        stalled = clock.now < 7.0 + 90 and "heavy" in output
        return SimpleNamespace(reward=0.0, machine_cause="time-limit") if stalled else toy_grade(task, output, capped)

    adm = one_file(poisoned("heavy"), build.params(), grader=pool_grader(Pool(), ToyGradePool(job)), clock=clock)
    found = polled(adm, 2 * FAULT_HOLD_S)
    per_hold = FAULT_HOLD_S // 30
    assert [verdict.rule for verdict in found] == [TIME_LIMIT] * per_hold + [""] * per_hold, "one counted check; after the hold the file passes"
    assert found[0].reason.endswith("(check 1 of 3)") and found[-1].valid and adm.store.reads == 2 and adm._held == {}


@pytest.mark.parametrize("error", [MemoryError, OSError])
def test_failed_check_that_is_not_counted_is_held_too(error, build):
    adm = one_file(poisoned("poison"), build.params(), tokens=FragileTokens(error))
    found = polled(adm, 2 * FAULT_HOLD_S)
    assert {verdict.reason for verdict in found} == {"groups[3].rollouts[1].output: the tokenizer fails on this text"} and adm.store.reads == 2


def test_new_file_new_params_or_a_clock_that_goes_back_end_the_hold(build, monkeypatch):
    adm = with_fault(TIME_LIMIT, build, monkeypatch)
    first = adm.verdicts([HOTKEY])[HOTKEY]
    adm.clock.wait(-3600)  # the clock of the machine was set back: the hold must not last an hour more
    assert adm.verdicts([HOTKEY])[HOTKEY].reason.endswith("(check 2 of 3)") and adm.store.reads == 2
    adm.params = build.params(max_prompts=adm.params.max_prompts + 1)
    assert adm.verdicts([HOTKEY])[HOTKEY].reason.endswith("(check 1 of 3)") and adm.store.reads == 3, "new params: checked with no wait"
    adm.store.put(layout.miner_object(HOTKEY), poisoned("slow") + b" ")
    assert adm.verdicts([HOTKEY])[HOTKEY].reason.endswith("(check 1 of 3)") and adm.store.reads == 4, "new bytes with the same fault: checked with no wait"
    adm.store.put(layout.miner_object(HOTKEY), file())
    assert adm.verdicts([HOTKEY])[HOTKEY].valid and adm.store.reads == 5 and adm._held == {}, "a new good file: valid with no wait"
    adm.store.put(layout.miner_object(HOTKEY), poisoned("slow"))  # the first bytes again, 0 s after their last check
    assert adm.verdicts([HOTKEY])[HOTKEY].reason.endswith("(check 1 of 3)") and first.reason.endswith("(check 1 of 3)")


def test_hold_in_the_texts_is_the_number_in_the_code():
    assert FAULT_HOLD_S == 600 and FAULT_TRIES == 3
    for code in TRANSIENT:
        assert RULES[code].endswith("; checked again at a try to open a round, at most once each 10 minutes; after 3 failed checks in a row the verdict stays until a new file is uploaded")


# --- a file must be able to train: the trainer's own arithmetic on the rewards ---


def test_file_with_no_training_signal_is_invalid(build):
    params = build.params()
    one_each = file([{"prompt": f"double {n}", "rollouts": [row(2 * n, 1)]} for n in range(10)])
    assert rule(one_each, params) == NO_SIGNAL, "one rollout per prompt: every reward is its own mean"
    same = [{"prompt": f"double {n}", "rollouts": [row(2 * n, n % 2), row(2 * n + 1, n % 2)]} for n in range(10)]
    assert rule(file(same), params) == NO_SIGNAL, "the rewards differ between prompts but not inside one prompt"
    assert rule(file(same + [{"prompt": "double 77", "rollouts": [row(154, 1), row(-1, 0)]}]), params) == ""
    assert rule(file(same + [{"prompt": "double 77", "rollouts": [row(154, 0.25), row(-1, 0.75)]}]), params) == ""
    assert rule(one_each, params, tokens=None) == NO_SIGNAL, "the rule needs no tokenizer and no grader"
    reason = admission.verdict(one_each, HOTKEY, params)
    assert (reason.valid, reason.rule) == (False, NO_SIGNAL) and "mean reward of its training group" in reason.reason


def test_rewards_too_large_for_a_mean_make_the_file_invalid(build):
    """Each reward is finite but their sum is not. The trainer's mean raises an error; one file must never stop the check of the others."""
    huge = file([{"prompt": "double 4", "rollouts": [row(k, 1e308, f"{THINK}{k} ") for k in range(3)]}])
    assert rule(huge, build.params()) == "reward" and rule(huge, build.params(), tokens=None) == "reward"
    apart = file([{"prompt": "double 4", "rollouts": [row(0, 1e308), row(1, -1e308), row(2, 1.0)]}])
    assert rule(apart, build.params()) == "", "rewards that are far apart are fine: the mean exists"


def test_training_signal_follows_the_group_size_of_the_file(build):
    def four(rewards) -> bytes:
        return lambda train=None: file([{"prompt": "double 4", "rollouts": [row(k, reward, f"{THINK}{k} ") for k, reward in enumerate(rewards)]}], train=train)

    params = build.params()
    assert rule(four([1, 1, 0, 0])(), params) == ""  # one group of 4: the mean is 0.5
    assert rule(four([1, 1, 0, 0])({"group_size": 2}), params) == NO_SIGNAL  # groups (1, 1) and (0, 0)
    assert rule(four([1, 0, 1, 0])({"group_size": 2}), params) == ""
    assert rule(four([1, 1, 1, 1, 0])({"group_size": 4}), params) == NO_SIGNAL  # a last group of one rollout stays a group
    assert rule(four([1, 1, 1, 1, 0])({"group_size": 5}), params) == ""


def test_training_signal_is_the_trainers_own_answer(build):
    """For any rewards and any group size: the file is valid exactly when the trainer finds a row with an advantage."""
    rng, params, seen = random.Random(7), build.params(), set()
    for _ in range(300):
        size = rng.choice([0, 2, 3, 4])
        rewards = [[rng.choice([0, 1, 1, 0.1, 0.3]) for _ in range(rng.randint(1, 6))] for _ in range(rng.randint(1, 3))]
        groups = [{"prompt": f"double {n}", "rollouts": [row(k, reward, f"{THINK}{k} ") for k, reward in enumerate(group)]} for n, group in enumerate(rewards)]
        raw = file(groups, train={"group_size": size} if size else None)
        batch_groups = [(group["prompt"], [(r["output"], r["reward"]) for r in group["rollouts"]]) for group in groups]
        trains = any(training_row["advantage"] != 0 for training_row in training_rows(batch_groups, size))
        assert (rule(raw, params) == "") == trains, (size, rewards)
        seen.add(trains)
    assert seen == {True, False}


def test_rollout_that_the_trainer_drops_gives_no_signal(build):
    """The trainer drops a rollout with no output token, or with a prompt that fills the training context."""
    quiet = [{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(2 * n, 1, THINK + "again ")]} for n in range(20)]
    empty = file(quiet + [{"prompt": "double 77", "rollouts": [{"output": "", "reward": 1}, {"output": "", "reward": 0}]}])
    assert rule(empty, build.params()) == NO_SIGNAL
    assert rule(empty, build.params(), tokens=None) == ""  # without a tokenizer the checker cannot know
    params = build.params()
    short = replace(params, recipe=replace(params.recipe, train_max_output_tokens=8, train_max_context_tokens=12))
    long_prompt = file(quiet + [{"prompt": "double 77 " + "and " * 5, "rollouts": [row(154, 1), row(-1, 0)]}])  # 5 + 7 words = 12 tokens
    assert (rule(long_prompt, params), rule(long_prompt, short)) == ("", NO_SIGNAL)
    assert rule(file(quiet + [{"prompt": "double 77 " + "and " * 4, "rollouts": [row(154, 1), row(-1, 0)]}]), short) == ""


# --- one valid file must fit the training time of one seat: steps x the tokens that the trainer runs ---


def test_trained_tokens_is_the_trainers_own_cut(build):
    """For any sizes: the count is the prompt plus exactly the output tokens that ``fit_tokens`` keeps, or 0 for a dropped row."""
    rng, params = random.Random(11), build.params()
    recipe = replace(params.recipe, train_max_output_tokens=8, train_max_context_tokens=12)
    seen = set()
    for _ in range(2000):
        n_prompt, n_output = rng.randint(1, 14), rng.randint(0, 12)
        kept, capped, cut = fit_tokens(n_prompt, list(range(n_output)), -1, recipe.train_max_output_tokens, recipe.train_max_context_tokens)
        assert trained_tokens(n_prompt, n_output, recipe) == (n_prompt + len(kept) if kept else 0), (n_prompt, n_output)
        seen.add((bool(kept), capped, cut))
    assert seen == {(False, False, False), (True, False, False), (True, True, False), (True, True, True)}
    assert trained_tokens(100, 32767, params.recipe) == 100 + 32768, "below the cap: the output and one end-of-turn token"
    assert trained_tokens(100, 32768, params.recipe) == 100 + 32768, "at the cap: no end-of-turn token"
    assert trained_tokens(100, 65536, params.recipe) == 100 + 32768, "above the training cap: cut, not rejected"
    assert trained_tokens(36864 - 10, 500, params.recipe) == 36864, "a long prompt leaves less room"


def test_file_above_the_training_cost_cap_is_invalid(build):
    """Default file: 10 prompts of 7 tokens, each with two rollouts of 41 words (+1 end token) that both train: 20 x 49 = 980."""
    raw = file()
    assert check_batch(raw, HOTKEY, build.params(), tokens=WordTokens()).train_token_steps == 980
    assert rule(raw, build.params(max_train_token_steps=980)) == ""
    assert rule(raw, build.params(max_train_token_steps=979)) == TRAIN_COST
    assert rule(raw, build.params(max_train_token_steps=0)) == "", "0 means no limit"
    found = admission.verdict(raw, HOTKEY, build.params(max_train_token_steps=979), tokens=WordTokens())
    assert (found.valid, found.rule) == (False, TRAIN_COST)
    assert "costs 980 token-steps (1 steps times 980 tokens in 20 rollouts that train); the cap is 979" in found.reason
    steps = file(train={"steps": 3})
    assert check_batch(steps, HOTKEY, build.params(), tokens=WordTokens()).train_token_steps == 3 * 980, "the miner's own steps count"
    assert (rule(steps, build.params(max_train_token_steps=2940)), rule(steps, build.params(max_train_token_steps=2939))) == ("", TRAIN_COST)
    blind = check_batch(raw, HOTKEY, build.params(max_train_token_steps=1), tokens=None)
    assert (blind.train_token_steps, blind.checked) == (0, ("format",)), "without a tokenizer the cost rule does not run"


def test_the_cost_of_a_frozen_file_is_kept_at_the_freeze_and_another_process_counts_it_again(build):
    """The source of the GPU plan of a round (``assembly.train_load``): the same file gives the same numbers in every pass."""
    store, params = Store(), build.params()
    store[layout.miner_object(HOTKEY)] = file(train={"steps": 3})
    adm = StoreAdmission(store, params, tokens=WordTokens(), grader=None)
    frozen = adm.freeze(build.reg(1, hotkey=HOTKEY), "r000001")
    checked = check_batch(store[frozen.uri], HOTKEY, params, tokens=WordTokens())
    assert (checked.train_token_steps, checked.train_rows) == (3 * 980, 20), "20 rollouts train, each in each of the 3 steps"
    reads = []
    real = store.get
    store.get = lambda key: reads.append(key) or real(key)
    assert adm.train_cost(frozen, params) == (2940, 20) and reads == [], "the process that froze the file reads nothing"
    later = StoreAdmission(store, params, tokens=WordTokens(), grader=None)  # run started again in the round
    assert later.train_cost(frozen, params) == (2940, 20) and reads == [frozen.uri], "the frozen copy, counted again"
    assert later.train_cost(frozen, params) == (2940, 20) and len(reads) == 1, "one read for each file"
    store[frozen.uri] = file()  # not the frozen bytes
    with pytest.raises(VerifyError):
        StoreAdmission(store, params, tokens=WordTokens(), grader=None).train_cost(frozen, params)
    wrong = file([{"prompt": "double 1", "rollouts": [row(2, 1), row(2, 1)]}])  # nothing trains: the file is not valid
    store[frozen.uri] = wrong
    with pytest.raises(Invalid):
        StoreAdmission(store, params, tokens=WordTokens(), grader=None).train_cost(replace(frozen, batch_sha256=sha256_hex(wrong)), params)


def test_only_rollouts_that_train_count_toward_the_cost(build):
    """A rollout with advantage zero is never run through the model, and a long output counts only up to the training cap."""
    params = build.params()
    flat = [{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(2 * n, 1, THINK + "again ")]} for n in range(20)]
    mixed = file(flat + [{"prompt": "double 77", "rollouts": [row(154, 1), row(-1, 0)]}])
    assert check_batch(mixed, HOTKEY, params, tokens=WordTokens()).train_token_steps == 2 * (7 + 41 + 1)
    short = replace(params, recipe=replace(params.recipe, train_max_output_tokens=8, train_max_context_tokens=12))
    assert check_batch(mixed, HOTKEY, short, tokens=WordTokens()).train_token_steps == 2 * (7 + 5), "room: min(8, 12 - 7) = 5 output tokens"
    grouped = file(train={"group_size": 2, "steps": 2})  # every prompt is one group of a right and a wrong rollout
    assert check_batch(grouped, HOTKEY, params, tokens=WordTokens()).train_token_steps == 2 * 980


# --- the rule codes ---


def test_rules_table_names_every_rule_code_of_the_checker():
    used = set()
    for node in ast.walk(ast.parse(inspect.getsource(admission))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("Invalid", "_need"):
            code = node.args[0 if node.func.id == "Invalid" else 1]
            if isinstance(code, ast.Constant) or getattr(code, "id", "").isupper():  # a literal or a constant of the module, not the argument inside _need
                used.add(code.value if isinstance(code, ast.Constant) else getattr(admission, code.id))
    assert used == set(RULES) and all(isinstance(code, str) and code for code in used)
    order = list(RULES)
    assert order[-1] == CHECK_FAULT, "it can come from any step, so it is last"
    assert order.index("answer_repeat") < order.index(NO_SIGNAL) < order.index(TRAIN_COST) < order.index(TIME_LIMIT) < order.index(FAULT) < order.index("grader") < order.index("pool")


def test_verdict_carries_the_rule_code(build):
    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=None, clock=lambda: 5)
    adm.store.put(layout.miner_object("a"), file(hotkey="a"))
    adm.store.put(layout.miner_object("b"), file(hotkey="a"))
    adm.store.put(layout.miner_object("c"), file(hotkey="c", train={"lr": 1}))
    adm.store.put(layout.miner_object("d"), b"x" * (build.params().max_batch_bytes + 1))
    assert {hotkey: verdict.rule for hotkey, verdict in adm.verdicts(["a", "b", "c", "d"]).items()} == {"a": "", "b": "hotkey", "c": "train", "d": "size"}
