"""Admission at launch: one file never stops the others (grader, tokenizer or checker fault), a hard time limit, no file that trains nothing, and a cap on the training cost."""

from __future__ import annotations

import ast
import inspect
import json
import random
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import CHECK_FAULT, FAULT, NO_SIGNAL, RULES, TIME_LIMIT, TOKENIZER_FAULT, TRAIN_COST, GraderFault, Invalid, StoreAdmission, check_batch, env_grader, final_answer, pool_grader, trained_tokens
from sn120_v2.canonical import sha256_hex
from sn120_v2.errors import InfraError
from sn120_v2.gpu.batch import fit_tokens, training_rows

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
    assert "5B" not in adm._seen and {"5A", "5C"} <= set(adm._seen), "the fault verdict is not kept"
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
    return StoreAdmission(store, inputs.pop("params"), tokens=WordTokens(), clock=lambda: 7, **inputs)


@pytest.mark.parametrize("error", [KeyError, RecursionError, MemoryError, Panic])
@pytest.mark.parametrize("step", ["_groups", "_cap", "_kept", "_trains", "_graded"])
def test_file_that_breaks_the_checker_is_skipped_and_the_other_files_are_checked(build, monkeypatch, step, error):
    bug_in(monkeypatch, step, error)
    adm = three_files([{"prompt": BUG, "rollouts": [row(6, 1), row(-1, 0)]}], params=build.params(), grader=graders()["pool_grader"])
    found = adm.verdicts(["5A", "5B", "5C"])
    assert [(found[key].valid, found[key].rule) for key in ("5A", "5B", "5C")] == [(True, ""), (False, CHECK_FAULT), (True, "")]
    assert re.fullmatch(rf"the checker fails on this file: {error.__name__} in buggy, test_admission_hardening\.py line \d+", found["5B"].reason), "the class and the place in the code, never the message"
    assert "5B" not in adm._seen and {"5A", "5C"} <= set(adm._seen), "the fault verdict is not kept"
    assert adm.verdicts(["5B"])["5B"] is not found["5B"] and adm.verdicts(["5A"])["5A"] is found["5A"]
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
            if isinstance(code, ast.Constant) or code.id.isupper():  # a literal or a constant of the module, not the argument inside _need
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
