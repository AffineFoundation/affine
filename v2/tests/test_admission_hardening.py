"""Admission at launch: one file never stops the others, and a hard time limit on every grade."""

from __future__ import annotations

import ast
import inspect
import json
from types import SimpleNamespace

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import FAULT, RULES, TIME_LIMIT, GraderFault, Invalid, StoreAdmission, check_batch, env_grader, final_answer, pool_grader
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import InfraError

HOTKEY = "5MinerA"
THINK = "step " * 40


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
        return SimpleNamespace(n=int(task_id[1:]), answer=str(2 * int(task_id[1:])))

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
    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=graders()[name], clock=lambda: 5)
    bad = file([{"prompt": f"double {n}", "rollouts": [row(2 * n, 1), row(-1, 0, THINK + poison + " " if n == 3 else THINK)]} for n in range(10)], "bad")
    for hotkey, raw in (("first", file(hotkey="first")), ("bad", bad), ("last", file(hotkey="last"))):
        adm.store.put(layout.miner_object(hotkey), raw)
    got = adm.verdicts(["first", "bad", "last"])
    assert [(hotkey, verdict.valid, verdict.rule) for hotkey, verdict in got.items()] == [("first", True, ""), ("bad", False, code), ("last", True, "")]
    assert "groups[3].rollouts[1]" in got["bad"].reason and poison not in got["bad"].reason
    assert adm.freeze(build.reg(2, hotkey="bad"), "r000001") is None  # a skip: not charged, the place is kept
    assert adm.freeze(build.reg(3, hotkey="last"), "r000001") is not None
    assert adm.verdicts(["bad"])["bad"] is not got["bad"], "the verdict holds for one call: the file is checked again"
    assert adm.verdicts(["first"])["first"] is got["first"]
    adm.store.put(layout.miner_object("bad"), file(hotkey="bad"))  # the miner uploads a file the grader can read
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
    assert pool.calls == [14], "the bad row is in the first chunk: the other rows are never graded"


# --- the rule codes ---


def test_rules_table_names_every_rule_code_of_the_checker():
    used = set()
    for node in ast.walk(ast.parse(inspect.getsource(admission))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("Invalid", "_need"):
            code = node.args[0 if node.func.id == "Invalid" else 1]
            if isinstance(code, ast.Constant) or code.id.isupper():  # a literal or a constant of the module, not the argument inside _need
                used.add(code.value if isinstance(code, ast.Constant) else getattr(admission, code.id))
    assert used == set(RULES) and all(isinstance(code, str) and code for code in used)
    order = list(RULES)
    assert order.index("answer_repeat") < order.index(TIME_LIMIT) < order.index(FAULT) < order.index("grader") < order.index("pool")


def test_verdict_carries_the_rule_code(build):
    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=None, clock=lambda: 5)
    adm.store.put(layout.miner_object("a"), file(hotkey="a"))
    adm.store.put(layout.miner_object("b"), file(hotkey="a"))
    adm.store.put(layout.miner_object("c"), file(hotkey="c", train={"lr": 1}))
    adm.store.put(layout.miner_object("d"), b"x" * (build.params().max_batch_bytes + 1))
    assert {hotkey: verdict.rule for hotkey, verdict in adm.verdicts(["a", "b", "c", "d"]).items()} == {"a": "", "b": "hotkey", "c": "train", "d": "size"}
