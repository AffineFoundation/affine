"""Admission at launch: one file never stops the others, a hard time limit, no file that trains nothing, and a cap on the training cost."""

from __future__ import annotations

import ast
import inspect
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest

from sn120_v2 import admission, layout
from sn120_v2.admission import FAULT, NO_SIGNAL, RULES, TIME_LIMIT, TRAIN_COST, GraderFault, Invalid, StoreAdmission, check_batch, env_grader, final_answer, pool_grader, trained_tokens
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
    assert order.index("answer_repeat") < order.index(NO_SIGNAL) < order.index(TRAIN_COST) < order.index(TIME_LIMIT) < order.index(FAULT) < order.index("grader") < order.index("pool")


def test_verdict_carries_the_rule_code(build):
    adm = StoreAdmission(Store(), build.params(), tokens=WordTokens(), grader=None, clock=lambda: 5)
    adm.store.put(layout.miner_object("a"), file(hotkey="a"))
    adm.store.put(layout.miner_object("b"), file(hotkey="a"))
    adm.store.put(layout.miner_object("c"), file(hotkey="c", train={"lr": 1}))
    adm.store.put(layout.miner_object("d"), b"x" * (build.params().max_batch_bytes + 1))
    assert {hotkey: verdict.rule for hotkey, verdict in adm.verdicts(["a", "b", "c", "d"]).items()} == {"a": "", "b": "hotkey", "c": "train", "d": "size"}
